import colorsys
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import loguru
import matplotlib.pyplot as plt
from moma_sg.utils.tapnextpp import TAPNextPP
import numpy as np
import torch
from tqdm import tqdm


def grid_coords(W, H, n=64):
    """Generate a regular n x n grid of (x, y) pixel coordinates spanning an image,
    with the outermost border row/column dropped.

    Args:
        W: Image width in pixels.
        H: Image height in pixels.
        n: Number of grid points per axis before border trimming.

    Returns:
        (M, 2) float array of (x, y) grid coordinates, M = (n - 2) ** 2.
    """
    coords = np.stack(
        np.meshgrid(np.linspace(0, W, n, endpoint=True), np.linspace(0, H, n, endpoint=True)),
        axis=-1,
    )[1:-1, 1:-1]
    return coords.reshape((-1, 2))


class Tracking:
    """2D/3D keypoint tracker built on TAPNext++, with helpers to seed, project,
    filter, and visualize point tracks across video segments."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize the tracker, optionally building the TAPNextPP model.

        Args:
            cfg: Config dict. Recognized keys: "init_models" (bool, default True;
                if False, model construction is skipped), "feat_type" ("shi" or
                "orb", default "shi"), and, when init_models is True, "device",
                "height", "width", "model_path", and "dataset".
        """
        if "init_models" in cfg:
            self.init_models = cfg["init_models"]
        else:
            self.init_models = True

        # Which detector seeds the extra (non-grid) keypoints handed to the
        # tracker at the start of each segment: "shi" (cv2.goodFeaturesToTrack)
        # or "orb" (cv2.ORB_create).
        self.feat_type = cfg.get("feat_type", "shi")
        self.grid_n = cfg.get("grid_n", 64)  # Number of grid points per axis before border trimming
        if self.feat_type not in ("shi", "orb"):
            raise ValueError(f"Unknown feat_type: {self.feat_type!r}. Expected 'shi' or 'orb'.")

        if self.init_models:
            self.device = cfg["device"]
            init_frame = np.random.randint(0, 255, (cfg["height"], cfg["width"], 3), dtype=np.uint8)
            self.tracker = TAPNextPP(frame_size=(cfg["width"], cfg["height"]), checkpoint_path=cfg["model_path"], device='cuda').initialize(
                init_frame, grid_coords(cfg["width"], cfg["height"], n=self.grid_n)
            )

            self.dataset = cfg["dataset"]

    def _extract_seed_keypoints(self, frame: np.ndarray, max_features: int = 4000) -> np.ndarray:
        """
        Detect keypoints to seed tracking on a frame, using whichever detector
        is selected via cfg["feat_type"] ("shi" or "orb").
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.feat_type == "orb":
            orb = cv2.ORB_create(nfeatures=max_features)
            keypoints = orb.detect(gray, None)
            return np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)
        else:  # "shi"
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=7,
            )
            return corners.squeeze(1).astype(np.float32) if corners is not None and len(corners) > 0 else np.empty((0, 2), dtype=np.float32)

    @staticmethod
    def _create_queries_bbox(bbox: np.ndarray, grid: int = 10, frames: List[int] = [0], device: str = "cuda") -> torch.Tensor:
        """
        Create a grid of queries around a bounding box.
        """
        x1, y1, x2, y2 = bbox
        x = np.linspace(x1, x2, grid)
        y = np.linspace(y1, y2, grid)
        x, y = np.meshgrid(x, y)
        frames = np.repeat(frames, grid**2)
        queries = torch.tensor(np.stack([frames, x.ravel(), y.ravel()], axis=1)).to(device)
        return queries.float()

    @staticmethod
    def _create_queries_points(points: List[Tuple[int, int]], frames: List[int] = [0], device: str = "cuda") -> torch.Tensor:
        """
        Create queries from a list of point coordinates.
        """
        points_arr = np.array(points)
        frames_arr = np.repeat(frames, points_arr.shape[0])
        queries = torch.tensor(np.concatenate([frames_arr[:, None], points_arr], axis=1)).to(device)
        return queries.float()

    @staticmethod
    def calculate_variance(points: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Calculate variance for point tracks.
        """
        variances = []
        for track_idx in range(points.shape[1]):
            visible_points = points[visibility[:, track_idx], track_idx]
            if visible_points.shape[0] > 0:
                variances.append(np.var(visible_points, axis=0))
            else:
                variances.append(np.zeros(points.shape[2]))
        return np.array(variances)

    # def _cotracker_process(
    #     self,
    #     window_frames: List[np.ndarray],
    #     queries: torch.Tensor,
    #     backward_tracking: bool = False,
    # ) -> torch.Tensor:
    #     """
    #     Process a window of frames through the CoTracker.
    #     """
    #     video = (
    #         torch.tensor(np.stack(window_frames))
    #         .permute(0, 3, 1, 2)[None]
    #         .float()
    #         .to(self.device)
    #     )
    #     return self.cotracker(
    #         video, queries=queries[None], backward_tracking=backward_tracking
    #     )

    @staticmethod
    def _select_points(img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Let the user select points interactively on an image.
        """
        selected_points: List[Tuple[int, int]] = []

        def onclick(event):
            """Matplotlib button-press handler: record and plot the clicked point."""
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                selected_points.append((x, y))
                plt.scatter(x, y, c="red", s=40)
                plt.draw()

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Click to select points. Press Enter to finish.")
        fig.canvas.mpl_connect("button_press_event", onclick)

        def on_key(event):
            """Matplotlib key-press handler: close the figure when Enter is pressed."""
            if event.key == "enter":
                plt.close()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return selected_points

    def _project_2d_tracks_to_3d(
        self,
        depth_window_frames: List[np.ndarray],
        pred_tracks: np.ndarray,
        min_depth: float = 0.3,
        max_depth: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects 2D tracks to 3D coordinates using depth data.
        """
        pred_tracks_3d = []
        valid_depth_masks = []
        K = np.array(self.dataset.depth_intrinsics)
        for i, depth in enumerate(depth_window_frames):
            coord_x = np.clip(pred_tracks[i, :, 0], 0, depth.shape[1] - 1)
            coord_y = np.clip(pred_tracks[i, :, 1], 0, depth.shape[0] - 1)
            p2 = np.stack([coord_x, coord_y, np.ones_like(coord_x)], axis=1)
            d = depth[np.round(coord_y).astype(int), np.round(coord_x).astype(int)] / 1000.0
            points = np.linalg.inv(K) @ p2.T * d
            valid_mask = np.logical_and(
                np.logical_and(points[2] > min_depth, points[2] < max_depth),
                np.isfinite(points[2]),
            )
            valid_depth_masks.append(valid_mask)
            pred_tracks_3d.append(points.T)
        return np.stack(pred_tracks_3d), np.stack(valid_depth_masks)

    @staticmethod
    def select_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Open an interactive window to select a bounding box.
        """
        cv2.namedWindow("Select BBox", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select BBox", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select BBox")
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        return (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])

    @staticmethod
    def extract_orb_features(image: np.ndarray, max_features: int = 500, grid_size: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """
        Extract ORB features from an image.
        """
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints_coords = []
        if grid_size is None:
            keypoints = orb.detect(image, None)
            keypoints_coords = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        else:
            rows, cols = grid_size
            h, w = image.shape[:2]
            grid_h, grid_w = h // rows, w // cols
            for i in range(rows):
                for j in range(cols):
                    x_start, x_end = j * grid_w, (j + 1) * grid_w
                    y_start, y_end = i * grid_h, (i + 1) * grid_h
                    grid_img = image[y_start:y_end, x_start:x_end]
                    keypoints = sorted(
                        orb.detect(grid_img, None),
                        key=lambda kp: kp.response,
                        reverse=True,
                    )[:max_features]
                    keypoints_coords.extend([(int(kp.pt[0]) + x_start, int(kp.pt[1]) + y_start) for kp in keypoints])
        return keypoints_coords

    @staticmethod
    def extract_good_features_to_track(
        image: np.ndarray,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Extract good features to track using OpenCV.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return [(int(c[0][0]), int(c[0][1])) for c in corners] if corners is not None else []

    @staticmethod
    def calc_tracks_stats(
        pred_visibility: torch.Tensor,
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Calculate track statistics.
        """
        num_frames, num_tracks = pred_visibility.shape
        visible_tracks_last_frame = pred_visibility[-1]
        num_visible_tracks = int(np.sum(visible_tracks_last_frame))
        percentage_visible = (num_visible_tracks / num_tracks) * 100
        reliability = np.sum(pred_visibility, axis=0) / num_frames
        return num_visible_tracks, percentage_visible, reliability

    @staticmethod
    def visualize_2d_tracks(
        frames: List[np.ndarray],
        pred_2d_tracks: np.ndarray,
        pred_visibility: np.ndarray,
        save_path: str,
    ) -> None:
        """Write a video with 2D tracks overlaid on each frame.

        Args:
            frames:           List of (H, W, 3) uint8 RGB frames.
            pred_2d_tracks:   (T, N, 2) float array of (x, y) pixel locations.
            pred_visibility:  (T, N) bool array.
            save_path:        Output .mp4 path.
        """
        H, W = frames[0].shape[:2]
        T, N = pred_2d_tracks.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, 15, (W, H))

        def idx_to_bgr(i):
            """Map a track index to a distinct BGR color via an evenly spaced hue."""
            hue = float(i) / max(1, N)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            return (int(b * 255), int(g * 255), int(r * 255))

        colors = [idx_to_bgr(i) for i in range(N)]

        for t in range(T):
            frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            locs = pred_2d_tracks[t]  # (N, 2)
            vis = pred_visibility[t]  # (N,)
            for i, ((x, y), visible) in enumerate(zip(locs, vis)):
                if visible:
                    cv2.circle(frame_bgr, (int(x), int(y)), 2, color=colors[i], thickness=-1)
                else:
                    cv2.circle(frame_bgr, (int(x), int(y)), 1, color=(200, 200, 200), thickness=-1)
            writer.write(frame_bgr)

        writer.release()

    def track_and_project_queries(
        self,
        segments: List[Tuple[int, int]],
        prior_masks: List[List[np.ndarray]],
        bidir: bool,
        save_dir: str,
        rgb_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        For each segment:
          - Track queries using TapNext++.
          - Project 2D tracks to 3D using depth information.
          - Apply prior mask and valid depth filtering.
          - If bidir=True, also runs additional an additional tracking pass at the end
            of the articulation but in inverse time.
            Additional tracks and visibility are NaN-filled at unused time steps.
        """
        # save 2D tracks the results:
        os.makedirs(save_dir, exist_ok=True)

        pred_3d_tracks_segments = []
        pred_visibility_segments = []
        recon_3d_tracks_segments = []
        recon_visibility_segments = []
        for idx, (seg_start, seg_end) in enumerate(segments):
            st = time.time()
            loguru.logger.info(f"Tracking segment {idx} from frame {seg_start} to {seg_end}...")

            frame0 = rgb_frames[seg_start]
            seed_kp0 = self._extract_seed_keypoints(frame0)
            kp0 = np.concatenate([grid_coords(frame0.shape[1], frame0.shape[0], n=self.grid_n), seed_kp0], axis=0)
            tracker = self.tracker.initialize(frame0, kp0)

            all_locs = [tracker.locations]
            all_vis = [tracker.visible]
            for frame_id in tqdm(range(seg_start + 1, seg_end), desc=f"Tracking segment {idx}", total=seg_end - seg_start - 1):
                locs, vis = tracker.update(rgb_frames[frame_id])
                all_locs.append(locs)
                all_vis.append(vis)

            pred_2d_tracks = np.squeeze(np.stack(all_locs, axis=0))  # (T, N, 2)
            pred_visibility = np.squeeze(np.stack(all_vis, axis=0))  # (T, N)

            # Visualize 2D tracks
            self.visualize_2d_tracks(
                rgb_frames[seg_start : seg_end + 1],
                pred_2d_tracks,
                pred_visibility,
                save_path=os.path.join(save_dir, f"segment_{idx}.mp4"),
            )

            loguru.logger.info(f"Tracking for segment {idx} with {seg_end - seg_start} frames took {time.time() - st} seconds.")
            torch.cuda.empty_cache()
            pred_3d_tracks, valid_depth_masks = self._project_2d_tracks_to_3d(depth_frames[seg_start:seg_end], pred_2d_tracks)
            # Filter out tracks falling into prior regions
            prior_masks_stack = prior_masks[seg_start:seg_end].astype(bool)
            x_loc = np.clip(pred_2d_tracks[:, :, 0], 0, prior_masks_stack.shape[2] - 1).astype(int)
            y_loc = np.clip(pred_2d_tracks[:, :, 1], 0, prior_masks_stack.shape[1] - 1).astype(int)
            pred_visibility &= ~prior_masks_stack[np.arange(prior_masks_stack.shape[0])[:, None], y_loc, x_loc]
            # Combine with valid depth masks
            pred_visibility &= valid_depth_masks
            pred_3d_tracks_segments.append(pred_3d_tracks)
            pred_visibility_segments.append(pred_visibility)

            if bidir:
                # # --- multi-keyframe forward passes (0.25 / 0.5 / 0.75) ---
                # seg_len = seg_end - seg_start
                # n_points = pred_2d_tracks.shape[1]
                # recon_3d_tracks_this_seg = []
                # recon_visibility_this_seg = []
                # for kf_frac in [0.25, 0.5, 0.75]:
                #     kf = int(seg_len * kf_frac)
                #     kf_abs = seg_start + kf
                #     kf_tracker = self.tracker.initialize(
                #         rgb_frames[kf_abs],
                #         grid_coords(rgb_frames[kf_abs].shape[1], rgb_frames[kf_abs].shape[0], n=self.grid_n),
                #     )
                #     kf_all_locs = [kf_tracker.locations]
                #     kf_all_vis  = [kf_tracker.visible]
                #     for frame_id in tqdm(range(kf_abs + 1, seg_end),
                #                          desc=f"Track segment {idx} from keyframe {kf_abs}",
                #                          total=seg_end - kf_abs - 1):
                #         locs, vis = kf_tracker.update(rgb_frames[frame_id])
                #         kf_all_locs.append(locs)
                #         kf_all_vis.append(vis)
                #     torch.cuda.empty_cache()
                #     kf_2d_tracks  = np.stack(kf_all_locs, axis=0)
                #     kf_visibility = np.stack(kf_all_vis,  axis=0)
                #     kf_3d_tracks, kf_valid_depth = self._project_2d_tracks_to_3d(
                #         depth_frames[kf_abs:seg_end], kf_2d_tracks
                #     )
                #     prior_kf  = prior_masks_stack[kf:]
                #     x_kf = np.clip(kf_2d_tracks[:, :, 0], 0, prior_kf.shape[2] - 1).astype(int)
                #     y_kf = np.clip(kf_2d_tracks[:, :, 1], 0, prior_kf.shape[1] - 1).astype(int)
                #     kf_visibility &= ~prior_kf[np.arange(prior_kf.shape[0])[:, None], y_kf, x_kf]
                #     kf_visibility &= kf_valid_depth
                #     full_3d  = np.zeros((seg_len, n_points, 3), dtype=np.float32)
                #     full_vis = np.zeros((seg_len, n_points),    dtype=bool)
                #     full_3d[kf:]  = kf_3d_tracks
                #     full_vis[kf:] = kf_visibility.astype(bool)
                #     recon_3d_tracks_this_seg.append(full_3d)
                #     recon_visibility_this_seg.append(full_vis)
                # recon_3d_tracks_segments.append(recon_3d_tracks_this_seg)
                # recon_visibility_segments.append(recon_visibility_this_seg)

                # --- single backward pass from the terminal frame ---
                # Initialise at the last frame, play frames in reverse order,
                # then flip all outputs so they align with the forward time axis.
                seg_len = seg_end - seg_start
                frame_rev = rgb_frames[seg_end - 1]
                seed_kp_rev = self._extract_seed_keypoints(frame_rev)
                kp_coords = np.concatenate([grid_coords(frame_rev.shape[1], frame_rev.shape[0], n=self.grid_n), seed_kp_rev], axis=0)

                rev_tracker = self.tracker.initialize(frame_rev, kp_coords)
                rev_locs = [rev_tracker.locations]
                rev_vis = [rev_tracker.visible]
                for frame_id in tqdm(
                    range(seg_end - 2, seg_start - 1, -1),
                    desc=f"Backward tracking segment {idx}",
                    total=seg_len - 1,
                ):
                    locs, vis = rev_tracker.update(rgb_frames[frame_id])
                    rev_locs.append(locs)
                    rev_vis.append(vis)
                torch.cuda.empty_cache()

                # rev_locs[0] → seg_end-1, rev_locs[-1] → seg_start; flip to forward order
                rev_2d_tracks = np.stack(rev_locs, axis=0)[::-1]  # (T, N, 2)
                rev_visibility = np.stack(rev_vis, axis=0)[::-1]  # (T, N)

                rev_3d_tracks, rev_valid_depth = self._project_2d_tracks_to_3d(depth_frames[seg_start:seg_end], rev_2d_tracks)
                x_rev = np.clip(rev_2d_tracks[:, :, 0], 0, prior_masks_stack.shape[2] - 1).astype(int)
                y_rev = np.clip(rev_2d_tracks[:, :, 1], 0, prior_masks_stack.shape[1] - 1).astype(int)
                rev_visibility &= ~prior_masks_stack[np.arange(seg_len)[:, None], y_rev, x_rev]
                rev_visibility &= rev_valid_depth

                recon_3d_tracks_segments.append([rev_3d_tracks])
                recon_visibility_segments.append([rev_visibility.astype(bool)])

        loguru.logger.info(f"Processed tracking for {len(pred_3d_tracks_segments)} segments.")
        return pred_3d_tracks_segments, pred_visibility_segments, recon_3d_tracks_segments, recon_visibility_segments
