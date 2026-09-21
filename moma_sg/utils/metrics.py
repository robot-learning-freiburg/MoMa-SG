from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    """
    Compute the 3D IoU (or maximum overlap ratio) between two axis-aligned bounding boxes.

    Args:
        bbox1: Open3D-style bounding box exposing get_min_bound()/get_max_bound().
        bbox2: Open3D-style bounding box exposing get_min_bound()/get_max_bound().
        padding: Amount to expand each box's bounds by before computing overlap.
        use_iou: If True, return the intersection-over-union; otherwise return the
            maximum of the two per-box overlap ratios.

    Returns:
        float: IoU (or max overlap ratio) between the two boxes.
    """
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap


def interaction_iou(pred_segments: np.ndarray, gt_segments: np.ndarray) -> float:
    """
    Compute IoU between two boolean interaction masks.

    Args:
        pred_segments (np.ndarray): Boolean array marking predicted interaction frames.
        gt_segments (np.ndarray): Boolean array marking ground-truth interaction frames.

    Returns:
        float: Intersection-over-union of the two boolean arrays, 0 if the union is empty.
    """
    intersection = np.sum(pred_segments & gt_segments)
    union = np.sum(pred_segments | gt_segments)
    return intersection / union if union > 0 else 0


def interval_iou(a, b):
    """Compute IoU between two 1D intervals a=(s1,e1), b=(s2,e2)."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0


def compute_interaction_iou_matrix(pred_segments, gt_segments):
    """Build IoU matrix between predicted and GT segments."""
    M = np.zeros((len(pred_segments), len(gt_segments)))
    for i, p in enumerate(pred_segments):
        for j, g in enumerate(gt_segments):
            M[i, j] = interval_iou(p, g)
    return M


def match_segments(pred_segments, gt_segments, iou_threshold=0.5):
    """
    # TODO: clean this up
    Match predicted and GT intervals using maximum IoU and Hungarian algorithm.
    Returns:
        matches : list of (pred_idx, gt_idx)
        unmatched_gt
        unmatched_pred
    """
    if len(gt_segments) == 0:
        return [], [], list(range(len(pred_segments)))

    if len(pred_segments) == 0:
        return [], list(range(len(gt_segments))), []

    # IoU matrix (pred x gt)
    iou_mat = compute_interaction_iou_matrix(pred_segments, gt_segments)

    # Convert maximization to minimization for Hungarian algorithm
    cost = 1 - iou_mat
    pred_idx, gt_idx = linear_sum_assignment(cost)

    matches = []
    unmatched_gt = list(range(len(gt_segments)))
    unmatched_pred = list(range(len(pred_segments)))

    for p, g in zip(pred_idx, gt_idx):
        if iou_mat[p, g] >= iou_threshold:
            matches.append((p, g))
            unmatched_gt.remove(g)
            unmatched_pred.remove(p)

    return matches, unmatched_gt, unmatched_pred


def segmentation_metrics(pred_segments, gt_segments, pred_segments_arr, gt_segments_arr, iou_threshold=0.5):
    """
    # TODO: clean this up
    Compute segment-level metrics for 1D segmentation.
    pred_segments, gt_segments : list of (start, end)
    """

    matches, unmatched_gt, unmatched_pred = match_segments(pred_segments, gt_segments, iou_threshold)

    # ---- TP, FP, FN ----
    TP = len(matches)
    FP = len(unmatched_pred)
    FN = len(unmatched_gt)

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # ---- IoU of matched pairs ----
    ious = []
    onset_errors = []
    offset_errors = []

    for p, g in matches:
        gt = gt_segments[g]
        pred = pred_segments[p]

        ious.append(interval_iou(pred, gt))
        onset_errors.append(abs(pred[0] - gt[0]))
        offset_errors.append(abs(gt[1] - pred[1]))

    mean_iou = np.mean(ious) if ious else 0
    mean_onset_error = np.mean(onset_errors) if onset_errors else 0
    mean_offset_error = np.mean(offset_errors) if offset_errors else 0

    # ---- Over/under-segmentation ----
    # Over-seg: GT matched by multiple predictions (shouldn’t happen with one-to-one matching)
    # Under-seg: Prediction overlaps multiple GT segments (check raw overlaps)
    over_seg = 0
    under_seg = 0

    # Check under-segmentation: a prediction overlaps >1 GT
    for p, p in enumerate(pred_segments):
        overlaps = sum(interval_iou(p, g) > 0 for g in gt_segments)
        if overlaps > 1:
            under_seg += 1

    # Check over-segmentation: a GT overlaps >1 prediction
    for gi, g in enumerate(gt_segments):
        overlaps = sum(interval_iou(g, p) > 0 for p in pred_segments)
        if overlaps > 1:
            over_seg += 1

    interaction_1d_iou = interaction_iou(pred_segments_arr, gt_segments_arr)

    return {
        "interaction_1d_iou": float(interaction_1d_iou),
        "tp": int(TP),
        "fp": int(FP),
        "fn": int(FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "mean_onset_error": float(mean_onset_error),
        "mean_offset_error": float(mean_offset_error),
        "over_segmentation_count": int(over_seg),
        "under_segmentation_count": int(under_seg),
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }


def pointcloud_iou(pc1, pc2, voxel_size=0.015):
    """
    Compute point-cloud IoU via voxelization.

    Parameters
    ----------
    pc1 : (N,3) ndarray
        First point cloud
    pc2 : (M,3) ndarray
        Second point cloud
    voxel_size : float
        Size of the voxel grid cell (smaller = more accurate, slower)

    Returns
    -------
    iou : float
        Intersection over Union between the voxelized occupancy sets.
    """

    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)

    # Compute a shared bounding box
    all_points = np.vstack([pc1, pc2])
    min_bound = all_points.min(axis=0)
    max_bound = all_points.max(axis=0)

    # Convert points to voxel indices
    def voxelize(pc):
        """Convert point coordinates to integer voxel grid indices."""
        return np.floor((pc - min_bound) / voxel_size).astype(np.int32)

    vox1 = voxelize(pc1)
    vox2 = voxelize(pc2)

    # Convert to sets of tuples for fast intersection/union
    set1 = {tuple(v) for v in vox1}
    set2 = {tuple(v) for v in vox2}

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_object_iou_matrix(pred_objects, gt_objects):
    """Build IoU matrix between predicted and GT objects."""
    M = np.zeros((len(pred_objects), len(gt_objects)))
    for i, pred_obj in enumerate(pred_objects):
        for j, gt_obj in enumerate(gt_objects):
            M[i, j] = compute_3d_iou(pred_obj["bbox"], gt_obj["bbox"])

    return M


def match_3d_masks(pred_objects, gt_objects, iou_threshold=0.5):
    """
    # TODO: clean this up
    Match predicted and GT intervals using maximum IoU and Hungarian algorithm.
    Returns:
        matches : list of (pred_idx, gt_idx)
        unmatched_gt
        unmatched_pred
    """
    if len(gt_objects) == 0:
        return [], [], list(range(len(pred_objects)))

    if len(pred_objects) == 0:
        return [], list(range(len(gt_objects))), []
    # IoU matrix (pred x gt)
    iou_mat = compute_object_iou_matrix(pred_objects, gt_objects)

    # Convert maximization to minimization for Hungarian algorithm
    cost = 1 - iou_mat
    pred_idx, gt_idx = linear_sum_assignment(cost)

    thresholded_matches = [(p, g) for p, g in zip(pred_idx, gt_idx) if iou_mat[p, g] >= iou_threshold]
    thresholded_ious = [iou_mat[p, g] for p, g in thresholded_matches]

    return thresholded_matches, thresholded_ious, iou_mat


from typing import Dict, Union

IOU_THRESHOLDS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def _match_predictions(pred_objects: np.ndarray, gt_objects: np.ndarray, iou_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Matches predictions to ground truths for a single scene and IoU threshold.

    Returns:
        - tp (np.ndarray): Boolean array indicating if each prediction is a True Positive (TP)
                           at the optimal matching (sorted by score).
        - matched_ious (np.ndarray): IoU value for the matched GT (0.0 if FP).
        - num_gt (int): Total number of GT boxes (used for Recall denominator).
    """
    if len(pred_objects) == 0 or len(gt_objects) == 0:
        raise ValueError("No predictions or ground truth objects available for matching.")

    # 3. Match GT masks against predictions using Hungarian algorithm
    matches, match_ious, match_iou_matrix = match_3d_masks(pred_objects, gt_objects, iou_threshold)  # (pred_idx, gt_idx) & iou_mat

    # Return results in score-sorted order
    return matches, match_ious, match_iou_matrix


# --- 2. Core Metrics Calculation (Recall and IoU) ---


def _calculate_recall_iou_metrics(num_tp: int, num_gt: int) -> Dict[str, Union[float, int]]:
    """
    Calculates True Positives, Recall, and Average IoU at the IOU_THRESHOLD.
    Note: Recall depends on the provided 'num_gt' which may be incomplete.
    """

    # Calculate Recall
    # This metric should be interpreted with caution if 'num_gt' is incomplete.
    recall = num_tp / num_gt if num_gt > 0 else 0.0
    return {
        "tp": int(num_tp),
        "recall": recall,
        "iou": 0.0,  # Placeholder for Average IoU if needed
    }


# --- 3. Main Evaluation Function ---


def evaluate_part_segmentation(pred_objects: np.ndarray, gt_objects: np.ndarray) -> Dict[str, Union[float, int]]:
    """
    Evaluates True Positives, Recall, and Average IoU at IoU=0.50 for a single set of 3D detections.
    Metrics sensitive to incomplete Ground Truth (Precision, AP, mAP, FP, FN) are excluded.

    Args:
        pred_boxes: Predicted boxes for the scene (N, 7).
        pred_scores: Confidence scores corresponding to pred_boxes (N,).
        gt_boxes: Ground Truth boxes (M, 7).

    Returns:
        A dictionary containing True Positives, Recall, and Average IoU.
    """
    # 1. Initialization and Edge Case Handling
    total_num_gt = len(gt_objects)
    num_pred = len(pred_objects)

    # Handle perfect/empty edge cases
    if total_num_gt == 0:
        raise ValueError("No ground truth objects available for evaluation.")
        return {
            "tp": 0,
            "recall": 1.0,  # If no GTs, Recall is often considered 100%
            "iou": 0.0,
        }
    if num_pred == 0:
        return {
            "tp": 0,
            "recall": 0.0,
            "iou": 0.0,
        }

    metrics_at_iou = {}
    for iou_thresh in IOU_THRESHOLDS:
        # 2. Match predictions and calculate core metrics
        matches, match_ious, match_iou_matrix = _match_predictions(pred_objects, gt_objects, iou_thresh)
        recall = len(matches) / len(gt_objects) if len(gt_objects) > 0 else 0.0

        # Per-GT-object recall (1 if matched, 0 otherwise) so callers can aggregate
        # mean/std across objects rather than across scenes.
        matched_mask = np.zeros(len(gt_objects), dtype=bool)
        for _, gt_idx in matches:
            matched_mask[gt_idx] = True

        metrics_at_iou[iou_thresh] = {
            "tp": len(matches),
            "recall": recall,
            "iou": np.nanmean(match_ious) if len(match_ious) > 0 else 0.0,
            # Per-object values (one entry per matched/GT object) for cross-scene aggregation.
            "ious_per_object": list(match_ious),
            "recall_per_object": matched_mask.tolist(),
        }

    return metrics_at_iou


# Axis estimator metrics
from moma_sg.data.arti4d import Articulation


def compute_twist_center(w_T_xi: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Compute the world-frame point on the twist's rotation axis closest to its local origin.

    Args:
        w_T_xi (np.ndarray): (4,4) world transform of the twist's local frame.
        xi (np.ndarray): (6,) twist vector [omega (3,), v (3,)].

    Returns:
        np.ndarray: (3,) world-frame point on the rotation axis, or the transform's
            translation if the twist's angular part is near zero (pure translation).
    """
    xi_P_rotation_origin = np.cross(xi[:3], xi[3:]) / omega_sq_norm if (omega_sq_norm := (xi[:3] ** 2).sum()) > 1e-4 else np.zeros(3)
    return w_T_xi[:3, :3] @ xi_P_rotation_origin + w_T_xi[:3, 3]


def compute_articulation_delta_twist(gt: Articulation, w_T_xi: np.ndarray, xi: np.ndarray) -> np.ndarray[float, float]:
    """Computes the delta between a given articulation and a twist model consisting of
       world transform and twist vector. It extracts the rotation or translation axis
       from the twist and the center of the rotation/a point on the translation.

       Returned errors are angular divergence of the main axis and positional divergence for the centers of rotations.

    Args:
        gt (Articulation): Articulation to compare against.
        w_T_xi (np.ndarray): World offset of the predicted articulation.
        xi (np.ndarray): Predicted articulation in the form of a twist.

    Returns:
        np.ndarray[float, float]: Axis divergens in rad, positional divergence in meters.
    """
    if gt.type == 'PRISMATIC':
        normalized_translation_direction = xi[3:] / np.linalg.norm(xi[3:])
        return compute_articulation_delta_point_and_axis(gt, w_T_xi[:3, 3], w_T_xi[:3, :3] @ normalized_translation_direction)
    elif gt.type == 'REVOLUTE':
        normalized_rotation_axis = xi[:3] / np.linalg.norm(xi[:3])
        w_P_rotation_origin = compute_twist_center(w_T_xi, xi)
        w_V_rotation = w_T_xi[:3, :3] @ normalized_rotation_axis
        return compute_articulation_delta_point_and_axis(gt, w_P_rotation_origin, w_V_rotation)
    else:
        raise ValueError(f'Unknown articulation type: "{gt.type}"')


def compute_articulation_delta_point_and_axis(gt: Articulation, w_P_center: np.ndarray, w_V_axis: np.ndarray) -> np.ndarray[float, float]:
    """Computes the delta between a given articulation and a predicted model consisting of a supporting point and
       an axis. The type of error is computed based on the ground-thruth's articulation type.

       Eeturned errors are angular divergence of the main axis and positional divergence for the centers of rotations.

    Args:
        gt (Articulation): Articulation to compare against.
        w_P_center (np.ndarray): Supporting point of articulation in world frame.
        w_V_axis (np.ndarray): Axis of articulation in world frame.

    Returns:
        np.ndarray[float, float]: Axis divergence in rad, positional divergence in meters.
    """
    if gt.type == 'PRISMATIC':
        # Amount of rotation per translation. Should be 0
        return np.array([np.arccos(np.clip(np.abs(w_V_axis @ gt.axis), 0, 1)), 0])
    elif gt.type == 'REVOLUTE':
        axis_cos = np.clip(np.abs(w_V_axis.T @ gt.axis), 0, 1)
        v_cross_v = np.cross(w_V_axis, gt.axis)
        # If lines are not parallel, calculate their distance
        if (vv_norm := np.linalg.norm(v_cross_v)) >= 1e-4:
            articulation_distance = np.abs(((w_P_center - gt.position) * v_cross_v).sum()) / vv_norm
        else:
            articulation_distance = np.linalg.norm(np.cross(w_P_center - gt.position, gt.axis))
        return np.array([np.arccos(axis_cos), articulation_distance])
    else:
        raise ValueError(f'Unknown articulation type: "{gt.type}"')
