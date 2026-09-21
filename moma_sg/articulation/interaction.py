import json
import os
from typing import Any, Dict, List, Tuple

from hmmlearn import hmm
import loguru
from moma_sg.utils.geometry import warp_depth_torch
import numpy as np
from omegaconf import DictConfig
from scipy.signal import medfilt
import torch
from tqdm import tqdm


def compute_depth_disparity(
    cfg: DictConfig, dataset: Any, prior_masks: List[np.ndarray], depth_frames: List[np.ndarray], interval: int = 6, disp_thresh: float = 0.05
) -> List[Tuple[int, int]]:
    """
    Identify interaction segments based on depth disparity.
    This method identifies segments where the prior mask is present and the disparity between
    the arm and the environment is small.
    """
    loguru.logger.info("Allocating tensors for depth warping...")
    batched_depth_frames = (torch.from_numpy(np.stack(depth_frames)).float() / 1000.0).to("cpu")  # Convert to meters and batch
    batched_prior_masks = torch.from_numpy(np.stack(prior_masks)).bool().to("cpu")
    batched_camera_poses = torch.from_numpy(np.stack([dataset[i]["pose"] for i in range(len(depth_frames))])).float().to("cpu")
    batched_transforms = (torch.linalg.inv(batched_camera_poses[:-interval]) @ batched_camera_poses[interval:]).float().to("cpu")

    # filter for valid depth values and remove prior-occluded regions
    valid_mask = (
        (batched_prior_masks == False) & (batched_depth_frames > cfg.dataset.depth_min) & (batched_depth_frames < cfg.dataset.depth_max)
    ).to()
    batched_depth_frames[~valid_mask] = float('nan')

    cam_intrinsics = torch.from_numpy(dataset.depth_intrinsics).float().to()
    warped_depth_frames = torch.full_like(batched_depth_frames, float('nan')).to("cpu")  # preallocate warped depth frames

    warp_scores = torch.zeros(len(prior_masks), dtype=torch.int64).to("cpu")
    disparities = []
    for local_idx, curr_idx in tqdm(enumerate(range(interval, len(depth_frames))), desc="Warping depth maps"):
        warped_depth_frames[curr_idx] = warp_depth_torch(batched_depth_frames[curr_idx], cam_intrinsics, batched_transforms[local_idx])
        valid_disp_mask = (
            ~torch.isnan(batched_depth_frames[local_idx])
            * ~torch.isnan(warped_depth_frames[curr_idx])
            * (warped_depth_frames[curr_idx] > cfg.dataset.depth_min)
            * (warped_depth_frames[curr_idx] < cfg.dataset.depth_max)
        )
        warped_depth_frames[curr_idx][~valid_disp_mask] = float('nan')
        disparity = torch.abs(batched_depth_frames[local_idx] - warped_depth_frames[curr_idx])
        disparity[~valid_disp_mask] = float('nan')

        disp_relevant = disparity > disp_thresh
        disparities.append(disp_relevant)

        warp_scores[local_idx:curr_idx] += torch.sum(disp_relevant)

    return warp_scores.cpu().numpy(), torch.stack(disparities, axis=0).cpu().numpy()


class InteractionHMM:
    """
    TODO: remove before release
    Two-state Gaussian HMM that segments a pair of aligned scalar interaction
    signals into 'interaction' (ON) and 'no_interaction' (OFF) states via Viterbi decoding."""

    def __init__(self, cfg: DictConfig):
        """
        Definition of HMM hidden state: ['interaction', 'no_interaction']
        """
        self.cfg = cfg
        self.model = self.build(self.cfg.interaction.hmm)

    def build(self, hmm_params: Dict):
        """Construct and configure the 2-state Gaussian HMM (start probabilities,
        transition matrix, emission means/covariances) from `hmm_params`, falling
        back to hand-tuned defaults if `hmm_params` is None."""
        # 2 states: OFF(0), ON(1)
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="diag",
            init_params="",  # we set everything manually
        )
        # 1. Start probabilities (you can tweak)
        if hmm_params is not None:
            model.startprob_ = np.array(hmm_params.start_prob).reshape(
                2,
            )
        else:
            model.startprob_ = np.array([0.9, 0.1])  # mostly OFF initially
        # ---------
        # 2. State transition probabilities
        #
        # OFF -> OFF   OFF -> ON
        # ON  -> OFF   ON  -> ON
        #
        # Set ON->ON high to force persistence (long segments)
        # Set OFF->ON lower to avoid too many false ONs
        # ---------
        if hmm_params is not None:
            model.transmat_ = np.array(hmm_params.trans_prob).reshape(2, 2)
        else:
            model.transmat_ = np.array(
                [
                    [0.9, 0.1],  # OFF transitions
                    [0.02, 0.98],  # ON transitions
                ]
            )
        # ---------
        # 3. Emission model: mean + variance for each of your two signals
        #
        # A very common assumption:
        #   - OFF has smaller values on average
        #   - ON has larger values
        #
        # If you don’t know: take empirical quantiles
        # ---------

        # Example: approximate means (customize!)
        if hmm_params is not None:
            model.means_ = np.array(hmm_params.means).reshape(2, 2)
        else:
            model.means_ = np.array(
                [
                    [0.2, 0.2],  # OFF-state typical signal values
                    [0.5, 0.5],  # ON-state typical signal values
                ]
            )
        # Variances
        if hmm_params is not None:
            model.covars_ = np.array(hmm_params.covars).reshape(2, 2)
        else:
            model.covars_ = np.array(
                [
                    [0.05, 0.05],  # OFF noise level
                    [0.05, 0.05],  # ON noise level
                ]
            )
        return model

    def forward(self, signal1, signal2):
        """Decode the most likely ON/OFF state sequence for two aligned 1-D signals
        via Viterbi decoding. Returns an array of state labels (0=OFF, 1=ON)."""
        X = np.column_stack([signal1, signal2]).astype(float)

        # Viterbi decoding → clean ON/OFF sequence
        return self.model.predict(X)


def compute_egocentric_scores(cfg: DictConfig, human_masks: List[np.ndarray], depth_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
    """
    Identify interaction segments based on human masks.
    This method identifies segments where the human mask is present and the disparity between
    the arm and the environment is small
    """

    interaction_scores = []
    for frame_idx, mask in tqdm(enumerate(human_masks)):
        depth = depth_frames[frame_idx]
        metric_depth = depth / 1000.0  # Convert depth to meters

        valid_mask = (metric_depth > cfg.dataset.depth_min) & (metric_depth < cfg.dataset.depth_max)
        metric_depth[~valid_mask] = np.nan

        # check if the current depth map contains any set of considered depth values
        if np.all(np.isnan(metric_depth)):
            interaction_scores.append(0.0)
            continue

        human_depth = metric_depth[mask > 0]
        if human_depth.size > 0 and np.nansum(human_depth) > 0:
            # check that an ego-centric human is present in the image
            ys, xs = np.where(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            img_h, img_w = mask.shape
            tolerance = 10

            left = x <= tolerance
            right = (x + w) >= (img_w - tolerance)
            top = y <= tolerance
            bottom = (y + h) >= (img_h - tolerance)

            if not (left or right or top or bottom):
                interaction_scores.append(0.0)
            else:
                interaction_scores.append(np.nansum(human_depth))
        else:
            interaction_scores.append(0.0)

    return interaction_scores


def compute_exocentric_scores(cfg: DictConfig, human_masks: List[np.ndarray], depth_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
    """
    Compute per-frame interaction scores from a third-person (exocentric) view.
    For each frame, sums the valid metric depth values under the human mask;
    frames with no valid depth or no masked human pixels score 0.
    """

    interaction_scores = []
    for frame_idx, mask in tqdm(enumerate(human_masks)):
        depth = depth_frames[frame_idx]
        metric_depth = depth / 1000.0  # Convert depth to meters

        valid_mask = (metric_depth > cfg.dataset.depth_min) & (metric_depth < cfg.dataset.depth_max)
        metric_depth[~valid_mask] = np.nan

        # check if the current depth map contains any set of considered depth values
        if np.all(np.isnan(metric_depth)):
            interaction_scores.append(0.0)
            continue

        human_depth = metric_depth[mask > 0]
        if human_depth.size > 0 and np.nansum(human_depth) > 0:
            interaction_scores.append(np.nansum(human_depth))
        else:
            interaction_scores.append(0.0)

    return interaction_scores


def filter_interaction_signals(
    warp_scores: np.ndarray, prior_scores: np.ndarray, warp_filt_ksize: int = 25, prior_filt_ksize: int = 5, filter_type: str = "average"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Predict interaction segments based on warp and prior scores.
    This function finds peaks in the scores and assigns them to interaction segments.
    Returns normalized filtered warp and prior scores.
    """

    if filter_type == "median":
        filt_warp_scores = medfilt(warp_scores, kernel_size=warp_filt_ksize)
        filt_prior_scores = medfilt(prior_scores, kernel_size=prior_filt_ksize)
    elif filter_type == "average":
        filt_warp_scores = np.convolve(warp_scores, np.ones(warp_filt_ksize) / warp_filt_ksize, mode='valid')
        filt_prior_scores = np.convolve(prior_scores, np.ones(prior_filt_ksize) / prior_filt_ksize, mode='valid')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filt_warp_scores / np.max(filt_warp_scores), filt_prior_scores / np.max(filt_prior_scores)


def cond_probability_model(cfg: DictConfig, filt_warp_scores: np.ndarray, filt_prior_scores: np.ndarray) -> np.ndarray:
    """
    Compute a per-frame interaction probability as a weighted combination of the
    normalized warp and prior scores, using the conditional-probability weights
    p(signal | depth-triggered, human-present) etc. from cfg.interaction.cond_prob.
    """

    p_s_cond_dt_ht = cfg.interaction.cond_prob.p_s_cond_dt_ht
    p_s_cond_dt_hf = cfg.interaction.cond_prob.p_s_cond_dt_hf
    p_s_cond_df_ht = cfg.interaction.cond_prob.p_s_cond_df_ht
    p_s_cond_df_hf = cfg.interaction.cond_prob.p_s_cond_df_hf

    prob_warp_scores = filt_warp_scores / np.max(filt_warp_scores)

    prob_prior_scores = filt_prior_scores / np.max(filt_prior_scores)

    interaction_prob = (
        p_s_cond_dt_ht * prob_warp_scores * prob_prior_scores
        + p_s_cond_dt_hf * prob_warp_scores * (1 - prob_prior_scores)
        + p_s_cond_df_ht * (1 - prob_warp_scores) * prob_prior_scores
        + p_s_cond_df_hf * (1 - prob_warp_scores) * (1 - prob_prior_scores)
    )

    return interaction_prob


def parse_segments(cfg: DictConfig, boolean_scores: np.ndarray):
    """
    Extract interaction segments based on interaction probability.
    """

    # get consecutive non-zero indices
    diff = np.diff(boolean_scores.astype(int))
    start_idxs = np.where(diff == 1)[0] + 1
    end_idxs = np.where(diff == -1)[0] + 1

    # Handle edge cases: if it starts/ends with non-zero
    if boolean_scores[0]:
        start_idxs = np.r_[0, start_idxs]
    if boolean_scores[-1]:
        end_idxs = np.r_[end_idxs, len(boolean_scores)]

    # Extract sequences
    pred_segments = [
        (int(start), int(end) + cfg.interaction.depth_warp_interval)
        for start, end in zip(start_idxs, end_idxs)
        if end - start > cfg.interaction.min_window_size and end - start < cfg.interaction.max_window_size
    ]
    pred_segments_arr = np.zeros(len(boolean_scores), dtype=bool)
    for start, end in pred_segments:
        pred_segments_arr[start:end] = True

    if len(pred_segments) == 0:
        loguru.logger.warning("No interaction segments found. Returning empty list.")

    return pred_segments, pred_segments_arr


def plot_interaction_segmentation(
    warp_scores: np.ndarray = None,
    prior_scores: np.ndarray = None,
    interaction_prob: np.ndarray = None,
    pred_segments: list = None,
    gt_segments: np.ndarray = None,
    save_dir: str = None,
):
    """Plot interaction segmentation results."""
    import matplotlib.pyplot as plt

    # plot raw time series signals
    plt.figure(figsize=(10, 5))
    if warp_scores is not None:
        plt.plot(warp_scores / np.max(warp_scores), label="Warp Interaction scores", color='blue')
    if prior_scores is not None:
        plt.plot(prior_scores / np.max(prior_scores), label="Prior Interaction scores", color='orange')
    if interaction_prob is not None:
        plt.plot(interaction_prob / np.max(interaction_prob), label="Interaction probability", color='red')

    # plot predicted interaction segments
    if pred_segments is not None:
        for start, end in pred_segments:
            plt.axvspan(start, end, color='green', alpha=0.3, label='Extracted Segment' if start == pred_segments[0][0] else "")
    # plot gt interaction segments
    if gt_segments is not None:
        plt.plot(gt_segments, label="GT Interaction segments", color='green', linestyle='--')

    plt.xlabel("Frame index")
    plt.ylabel("Interaction score")
    plt.title(f"Interaction scores: {save_dir.split('/')[-3]}/{save_dir.split('/')[-2]}")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "prob_interaction_scores.png"))
        loguru.logger.info(f"Saved interaction segmentation plot to {save_dir}")
    plt.close()


def save_interaction_segments(segments: List[Tuple[int, int]], save_dir: str) -> List[Tuple[int, int]]:
    """Save interaction segments to a JSON file (pred_segments.json) under save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "pred_segments.json"), "w") as f:
        json.dump(segments, f, indent=4)
    loguru.logger.info(f"Saved interaction segments to {os.path.join(save_dir, 'pred_segments.json')}")


def run_interaction_model(cfg: DictConfig, filt_warp_scores: np.ndarray, filt_prior_scores: np.ndarray) -> List[Tuple[int, int]]:
    """
    Run any of the interaction segmentation models.
    """

    interaction_prob = None
    boolean_scores = None
    if cfg.interaction.model == "cond-prob":
        interaction_prob = cond_probability_model(cfg, filt_warp_scores, filt_prior_scores)
        boolean_scores = interaction_prob > cfg.interaction.cond_prob.min_interaction_p
    elif cfg.interaction.model == "hmm":
        hmm_model = InteractionHMM(cfg)
        boolean_scores = hmm_model.forward(filt_warp_scores, filt_prior_scores).astype(bool)
    else:
        raise ValueError(f"Unknown interaction model: {cfg.interaction.model}")

    return boolean_scores, interaction_prob
