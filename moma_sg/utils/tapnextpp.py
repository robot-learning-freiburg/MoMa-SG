# Copyright (c) 2024 Adrian Röfer, Robot Learning Lab, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray
import torch

try:
    from tapnet.tapnext.tapnext_torch import TAPNext
    import torch
except ModuleNotFoundError as e:
    TAPNEXTPP_EXCEPTION = e
    import logging

    logging.warning(f'Could not load tapnet/tapnext: {TAPNEXTPP_EXCEPTION}')
    TAPNext = None


def grid_coords(W, H, n=64):
    """Generate a regular grid of interior query points spanning an image.

    Args:
        W: Image width.
        H: Image height.
        n: Number of grid points per axis before the border row/column is trimmed.

    Returns:
        (M, 2) ndarray of (x, y) pixel coordinates, with M = (n - 2) ** 2.
    """
    coords = np.stack(
        np.meshgrid(np.linspace(0, W, n, endpoint=True), np.linspace(0, H, n, endpoint=True)),
        axis=-1,
    )[1:-1, 1:-1]
    return coords.reshape((-1, 2))


class TAPNextPP:
    """Online point tracker wrapping TAPNext++ (PyTorch).

    Points are anchored on the first frame via initialize(), then the tracker
    is advanced one frame at a time via update().

    Usage:
        tracker = TAPNextPP(frame_size=(640, 480))
        tracker.initialize(first_frame, points_xy)
        for frame in stream:
            locs, vis = tracker.update(frame)
    """

    def __init__(
        self,
        frame_size: tuple,
        checkpoint_path: Optional[Path] = None,
        model_resolution: tuple = (256, 256),
        device: str = 'cuda',
        shared_model: Optional[torch.nn.Module] = None,
        use_fp16: bool = False,
    ) -> None:
        """Create a new TAPNext++ tracker.

        Args:
            frame_size:       (W, H) of input images.
            checkpoint_path:  Path to .pt checkpoint. Downloads automatically if None.
            model_resolution: (H, W) internal resolution the model operates at.
            device:           Torch device string.
            shared_model:     Pre-loaded TAPNext instance to reuse. When provided,
                              checkpoint_path and model_resolution are ignored and the
                              device is inferred from the model. Allows multiple trackers
                              to share a single set of weights (~1 GB saved per instance).
            use_fp16:         Run inference in float16 via torch.amp.autocast (CUDA only).
                              Halves weight memory when shared_model is None. Has no effect
                              on CPU (autocast is silently disabled).
        """
        if TAPNext is None:
            raise ImportError(f'Cannot instantiate TAPNextPP, tapnet was not loaded: {TAPNEXTPP_EXCEPTION}')

        if shared_model is not None:
            self._model = shared_model
            self._model_resolution = tuple(shared_model.image_size)  # (H, W)
            self._device = next(shared_model.parameters()).device
        else:
            self._model_resolution = model_resolution  # (H, W)
            self._device = torch.device(device)

            ckpt_file = Path(checkpoint_path)

            if not ckpt_file.exists():
                from subprocess import Popen

                downloader = Popen(
                    ['wget', '--no-check-certificate', 'https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt'],
                    cwd=str(ckpt_file.parent.absolute()),
                )
                downloader.communicate()
                if downloader.returncode != 0:
                    raise RuntimeError('Failed to download TAPNext++ checkpoint.')

            checkpoint_path = ckpt_file

            self._model = TAPNext(image_size=model_resolution)
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self._model.load_state_dict({k.replace('tapnext.', ''): v for k, v in ckpt['state_dict'].items()})
            if use_fp16:
                self._model.half()
            self._model.to(self._device)
            self._model.eval()

        self._use_fp16 = use_fp16
        self._frame_size = frame_size  # (W, H)

        self._tracking_state: Optional[object] = None
        self._locations: Optional[ndarray] = None  # (N, 2) pixel (x, y)
        self._visible: Optional[ndarray] = None  # (N,)  bool

    def initialize(self, frame: ndarray, points_xy: ndarray) -> "TAPNextPP":
        """Anchor query points on the first frame and initialize tracking state.

        If this instance has already been initialized, a new TAPNextPP sharing
        the same model weights is created, initialized, and returned instead.
        Otherwise this instance is initialized and returned.

        Args:
            frame:      (H, W, 3) uint8 RGB.
            points_xy:  (N, 2) pixel coordinates (x, y) in input frame space.

        Returns:
            The initialized tracker (self, or a new shared-weight instance).
        """
        if self._tracking_state is not None:
            tracker = TAPNextPP(frame_size=self._frame_size, shared_model=self._model, use_fp16=self._use_fp16)
            return tracker.initialize(frame, points_xy)

        W, H = self._frame_size
        N = len(points_xy)

        # query_points: (1, N, 3) as (t=0, y_px, x_px) in pixel coords at model
        # resolution. The model normalises internally via division by image_size.
        mH, mW = self._model_resolution
        query_points = np.zeros((1, N, 3), dtype=np.float32)
        query_points[0, :, 1] = points_xy[:, 1] * mH / H  # y in model pixels
        query_points[0, :, 2] = points_xy[:, 0] * mW / W  # x in model pixels

        video_tensor = self._preprocess(frame)
        query_tensor = torch.from_numpy(query_points).to(self._device)

        _autocast = self._use_fp16 and self._device.type == 'cuda'
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_autocast):
            tracks, _, visible_logits, tracking_state = self._model(
                video=video_tensor,
                query_points=query_tensor,
            )

        self._tracking_state = tracking_state
        self._locations = self._to_input_coords(tracks)
        self._visible = self._to_visible(visible_logits)
        return self

    def update(self, frame: ndarray) -> Tuple[ndarray, ndarray]:
        """Advance the tracker by one frame.

        Args:
            frame: (H, W, 3) uint8 RGB.

        Returns:
            locations: (N, 2) pixel coords (x, y) in input frame space.
            visible:   (N,)  bool.
        """
        if self._tracking_state is None:
            raise RuntimeError('Call initialize() before update().')

        video_tensor = self._preprocess(frame)

        _autocast = self._use_fp16 and self._device.type == 'cuda'
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_autocast):
            tracks, _, visible_logits, tracking_state = self._model(
                video=video_tensor,
                state=self._tracking_state,
            )

        self._tracking_state = tracking_state
        self._locations = self._to_input_coords(tracks)
        self._visible = self._to_visible(visible_logits)

        return self._locations, self._visible

    @property
    def locations(self) -> Optional[ndarray]:
        """(N, 2) last tracked pixel locations (x, y) in input frame space."""
        return self._locations

    @property
    def visible(self) -> Optional[ndarray]:
        """(N,) bool visibility flags from the last processed frame."""
        return self._visible

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: ndarray) -> torch.Tensor:
        """Resize to model resolution, normalise to [-1, 1], return (1,1,H,W,3)."""
        mH, mW = self._model_resolution
        resized = cv2.resize(frame, (mW, mH), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0 * 2.0 - 1.0)
        return tensor.to(self._device)[None, None]  # (1, 1, mH, mW, 3)

    def _to_input_coords(self, tracks: torch.Tensor) -> ndarray:
        """Convert model-resolution tracks to input frame pixel coordinates.

        The model outputs tracks in (row, col) = (y, x) order; we return (x, y).

        tracks: (1, 1, N, 2) in (row, col) pixel coords at model resolution
        Returns: (N, 2) in (x, y) at input frame resolution
        """
        W, H = self._frame_size
        mH, mW = self._model_resolution
        raw = tracks[0, 0].cpu().numpy()  # (N, 2) as (row, col)
        x = raw[:, 1] * W / mW  # col → input x
        y = raw[:, 0] * H / mH  # row → input y
        return np.column_stack([x, y])

    def _to_visible(self, visible_logits: torch.Tensor) -> ndarray:
        """Threshold visibility logits to a boolean (N,) array.

        visible_logits: (1, 1, N, 1) or (1, 1, N)
        """
        # Flatten batch and time dimensions; keep only the N point axis.
        return (visible_logits[0, 0].reshape(-1) > 0).cpu().numpy()
