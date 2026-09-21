from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
import json
import os
from typing import Any, Optional, Tuple, get_args, get_origin

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.metrics.pairwise import cosine_similarity


class SEM_STATE(Enum):
    """Discrete semantic state of an articulated object (e.g. a door or drawer)."""

    OPEN = "open"
    SLIGHTLY_OPEN = "half open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class ARTICULATION_MODE(Enum):
    """
    Semantic action type of an observed articulation motion.
    """

    OPENING = "opening"
    CLOSING = "closing"
    OPENING_CLOSING = "opening-closing"
    CLOSING_OPENING = "closing-opening"
    UNKNOWN = "unknown"


def _serialize_value(val: Any):
    """Convert numpy arrays and numpy scalars to JSON-serializable types."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, Enum):
        return val.value
    return val


def _infer_numpy_dtype_from_list(lst):
    """Infer numpy dtype based on element types in a list."""
    if not lst:
        return float

    sample = lst
    # Flatten nested lists (for matrices)
    while isinstance(sample, list) and sample and isinstance(sample[0], list):
        sample = sample[0]

    if all(isinstance(x, int) for x in sample):
        return int
    if all(isinstance(x, (int, float)) for x in sample):
        return float
    return object


def _convert_to_annotated_type(value: Any, annotation: Any):
    """Convert value loaded from JSON into type defined by dataclass annotation."""
    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] → unwrap
    if origin is Optional:
        annotation = args[0]
        return _convert_to_annotated_type(value, annotation)

    # numpy array
    if annotation is np.ndarray:
        if isinstance(value, list):
            dtype = _infer_numpy_dtype_from_list(value)
            return np.asarray(value, dtype=dtype)
        return value

    # Tuple[T1, T2]
    if origin is tuple or origin is Tuple:
        inner_types = args
        if isinstance(value, list) and len(inner_types) == len(value):
            return tuple(inner_types[i](value[i]) for i in range(len(value)))
        return tuple(value)

    return value


@dataclass
class InferencePointAxis:
    """Axis estimation result of an interaction segment:
    the estimated revolute/prismatic joint parameters (position, axis, twist,
    pitch, per-frame joint states) etc."""

    position: Optional[np.ndarray] = None
    axis: Optional[np.ndarray] = None
    type: Optional[str] = None
    collinearity: Optional[float] = None
    w_T_a: Optional[np.ndarray] = None
    cos_median: Optional[float] = None
    twist: Optional[np.ndarray] = None
    pitch: Optional[float] = None
    thetas: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float]] = None
    id: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    success: Optional[bool] = None  # whether its successfully estimated
    loss: Optional[float] = None  # optimization loss
    motion_type: Optional[ARTICULATION_MODE] = None
    last_observed_state: Optional[SEM_STATE] = None
    pairs_t: Optional[defaultdict] = None

    # -----------------------------
    # Serialization helpers
    # -----------------------------
    def to_dict(self):
        """Convert dataclass to a JSON-serializable dict."""
        base = {f.name: _serialize_value(getattr(self, f.name)) for f in fields(self)}

        # Add any extra attributes not in the dataclass
        extra_keys = set(self.__dict__.keys()) - set(base.keys())
        for k in extra_keys:
            if k in ["pairs", "pairs_t"]:
                continue
            base[k] = _serialize_value(getattr(self, k))

        return base

    @classmethod
    def from_dict(cls, data: dict):
        """Create dataclass from a dict (JSON-compatible)."""
        field_map = {f.name: f.type for f in fields(cls)}

        kwargs = {}
        extras = {}

        for key, value in data.items():
            if key in field_map:
                annotation = field_map[key]
                kwargs[key] = _convert_to_annotated_type(value, annotation)
            else:
                extras[key] = value

        obj = cls(**kwargs)

        # Restore dynamic fields
        for k, v in extras.items():
            setattr(obj, k, v)

        return obj

    def save(self, path: str):
        """Save to a JSON file."""
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


VALID_OPENING = {
    (SEM_STATE.CLOSED, SEM_STATE.OPEN, SEM_STATE.OPEN),
    (SEM_STATE.CLOSED, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN),
    (SEM_STATE.CLOSED, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.SLIGHTLY_OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN, SEM_STATE.OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN, SEM_STATE.SLIGHTLY_OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN),
}

VALID_CLOSING = {
    (SEM_STATE.OPEN, SEM_STATE.CLOSED, SEM_STATE.CLOSED),
    (SEM_STATE.OPEN, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.SLIGHTLY_OPEN),
    (SEM_STATE.OPEN, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED, SEM_STATE.CLOSED),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED),
}

VALID_OPENING_CLOSING = {
    (SEM_STATE.CLOSED, SEM_STATE.OPEN, SEM_STATE.CLOSED),
    (SEM_STATE.CLOSED, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN, SEM_STATE.SLIGHTLY_OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN, SEM_STATE.CLOSED),
}

VALID_CLOSING_OPENING = {
    (SEM_STATE.OPEN, SEM_STATE.CLOSED, SEM_STATE.OPEN),
    (SEM_STATE.OPEN, SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED, SEM_STATE.SLIGHTLY_OPEN),
    (SEM_STATE.SLIGHTLY_OPEN, SEM_STATE.CLOSED, SEM_STATE.OPEN),
}


LAST_INFERRED_STATE_MAP = {
    ARTICULATION_MODE.OPENING: SEM_STATE.OPEN,
    ARTICULATION_MODE.CLOSING: SEM_STATE.CLOSED,
    ARTICULATION_MODE.OPENING_CLOSING: SEM_STATE.CLOSED,
    ARTICULATION_MODE.CLOSING_OPENING: SEM_STATE.OPEN,
    ARTICULATION_MODE.UNKNOWN: SEM_STATE.UNKNOWN,
}


def evaluate_first(frame_idcs: list, scalers: np.ndarray, start_idx: int) -> int:
    """Return the (start_idx-offset) index of the first frame in `frame_idcs`."""
    return frame_idcs[0] + start_idx


def evaluate_last(frame_idcs: list, scalers: np.ndarray, start_idx: int) -> int:
    """Return the (start_idx-offset) index of the last frame in `frame_idcs`."""
    return frame_idcs[-1] + start_idx


def evaluate_max(frame_idcs: list, scalers: np.ndarray, start_idx: int) -> int:
    """Return the (start_idx-offset) index of the frame with the largest `scalers` value."""
    return frame_idcs[np.argmax(scalers)] + start_idx


def evaluate_min(frame_idcs: list, scalers: np.ndarray, start_idx: int) -> int:
    """Return the (start_idx-offset) index of the frame with the smallest `scalers` value."""
    return frame_idcs[np.argmin(scalers)] + start_idx


MAX_OPENING_MAP = {
    ARTICULATION_MODE.OPENING: evaluate_last,
    ARTICULATION_MODE.CLOSING: evaluate_first,
    ARTICULATION_MODE.OPENING_CLOSING: evaluate_max,
    ARTICULATION_MODE.CLOSING_OPENING: evaluate_min,
    ARTICULATION_MODE.UNKNOWN: evaluate_max,
}


def understand_articulation(motion_trend: str, transition: Tuple[SEM_STATE, SEM_STATE, SEM_STATE]) -> str:
    """
    TODO: remove before release
    DEPRECATED: we now use LAST_INFERRED_STATE_MAP
    Parse the motion trend and transition tuple to determine the motion type.
    Args:
        motion_trend (str): "single" or "cycle"
        transition (tuple): tuple of SEM_STATE representing the transitions
    Returns:
        str: motion type ("opening", "closing", "opening-closing", "closing-opening", or "unknown")
    """
    motion_type = ARTICULATION_MODE.UNKNOWN
    last_observed = SEM_STATE.UNKNOWN
    if motion_trend == "single":
        # capture sole opening actions
        if transition in VALID_OPENING:
            motion_type = ARTICULATION_MODE.OPENING
            last_observed = SEM_STATE.OPEN
        elif transition in VALID_CLOSING:
            motion_type = ARTICULATION_MODE.CLOSING
            last_observed = SEM_STATE.CLOSED
        else:
            motion_type = ARTICULATION_MODE.UNKNOWN
            last_observed = SEM_STATE.UNKNOWN
    elif motion_trend == "cycle":
        if transition in VALID_OPENING_CLOSING:
            motion_type = ARTICULATION_MODE.OPENING_CLOSING
            last_observed = SEM_STATE.CLOSED
        elif transition in VALID_CLOSING_OPENING:
            motion_type = ARTICULATION_MODE.CLOSING_OPENING
            last_observed = SEM_STATE.OPEN
        else:
            motion_type = ARTICULATION_MODE.UNKNOWN

    if motion_type in [ARTICULATION_MODE.CLOSING_OPENING, ARTICULATION_MODE.OPENING]:
        last_observed = SEM_STATE.OPEN
    elif motion_type in [ARTICULATION_MODE.OPENING_CLOSING, ARTICULATION_MODE.CLOSING]:
        last_observed = SEM_STATE.CLOSED
    else:
        last_observed = SEM_STATE.UNKNOWN

    return motion_type, last_observed


def parse_response(response: str, cls):
    """
    Parses the model's response to determine the articulation state.

    Args:
        response (str): The model's response.
        categories (list): List of valid categories to match against.

    Returns:
        SEM_STATE: The determined articulation state.
    """
    for state in cls:
        if state.value == response:
            return state
    return cls.UNKNOWN


def _skew(v):
    """Create a skew-symmetric matrix from a vector."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _unskew(matrix):
    """Extract a vector from a skew-symmetric matrix."""
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def identify_cloud_pairs(tracks: np.ndarray, vis: np.ndarray):
    """
    TODO: remove before release
    Identify pairs of point clouds that have traveled a significant distance apart.
    - tracks: N x T x 3 array of point tracks
    - vis: N x T array of visibility (1 if visible, 0 otherwise)
    Returns a dictionary of pairs indexed by time step.
    """
    pairs, pairs_t = defaultdict(list), defaultdict(list)
    if tracks.shape[0] == 0 or tracks.shape[1] == 0:
        print("No tracks or no time steps in tracks, returning empty pairs.")
        return pairs, pairs_t

    # Calculate cumulative distances traveled along trajectories
    incr_track_dists = np.linalg.norm(tracks[:, 1:, :] - tracks[:, :-1, :], axis=2)
    cum_track_dists = np.zeros((tracks.shape[0], tracks.shape[1] - 1))
    for i in range(1, tracks.shape[1]):
        cum_track_dists[:, i - 1] = np.sum(incr_track_dists[:, :i], axis=1)
    cum_track_dists_aug = np.insert(cum_track_dists, 0, 0, axis=1)

    if np.mean(cum_track_dists_aug, axis=0)[-1] < 0.15:
        print("Tracks have not traveled enough distance, returning empty pairs.")
        return pairs, pairs_t

    for track_idx in range(tracks.shape[0]):
        if cum_track_dists_aug[track_idx, -1] < 0.5 * np.mean(cum_track_dists_aug[track_idx]):
            # remove tracks that have not traveled enough distance
            vis[track_idx, :] = 0

    # Identify pairs of point clouds with significant distance to one another
    prev_t_i = 0
    for t_i in range(tracks.shape[1]):
        if t_i > 0 and t_i - prev_t_i < 0.02 * tracks.shape[1]:
            continue
        for t_j in range(t_i + 1, tracks.shape[1]):
            # check if cumulative and absolute distance traveled is significant
            mean_cum_traveled_dist = np.mean(cum_track_dists_aug[:, t_j]) - np.mean(cum_track_dists_aug[:, t_i])
            mean_abs_dist = np.mean(np.linalg.norm(tracks[:, t_j, :] - tracks[:, t_i, :], axis=1))
            if mean_cum_traveled_dist > 0.2 * np.mean(cum_track_dists_aug[:, -1]) and mean_abs_dist > 0.2 * np.mean(cum_track_dists_aug[:, -1]):
                mutual_vis_points = vis[:, t_i] * vis[:, t_j]
                pairs[t_i].append((tracks[mutual_vis_points, t_i, :], tracks[mutual_vis_points, t_j, :]))
                pairs_t[t_i].append(t_j)
                del mutual_vis_points, mean_cum_traveled_dist
                break
        prev_t_i = t_i
    return pairs, pairs_t


def identify_cloud_pairs_from_start(tracks: np.ndarray, vis: np.ndarray):
    """
    Identify point-cloud pairs anchored at a single "start" frame.
    Unlike `identify_cloud_pairs`, which forms pairs across a sliding set of
    anchor timesteps, this anchors on the first frame with sufficient mutual
    visibility and pairs it against every later frame that shows significant
    cumulative and absolute displacement.
    - tracks: N x T x 3 array of point tracks
    - vis: N x T array of visibility (1 if visible, 0 otherwise)
    Returns a dictionary of pairs indexed by the anchor time step.
    """
    pairs, pairs_t = defaultdict(list), defaultdict(list)
    if tracks.shape[0] == 0 or tracks.shape[1] == 0:
        print("No tracks or no time steps in tracks, returning empty pairs.")
        return pairs, pairs_t

    # Calculate cumulative distances traveled along trajectories
    incr_track_dists = np.linalg.norm(tracks[:, 1:, :] - tracks[:, :-1, :], axis=2)
    cum_track_dists = np.zeros((tracks.shape[0], tracks.shape[1] - 1))
    for i in range(1, tracks.shape[1]):
        cum_track_dists[:, i - 1] = np.sum(incr_track_dists[:, :i], axis=1)
    cum_track_dists_aug = np.insert(cum_track_dists, 0, 0, axis=1)
    if np.mean(cum_track_dists_aug, axis=0)[-1] < 0.10:
        print("Tracks have not traveled enough distance, returning empty pairs.")
        return pairs, pairs_t

    for track_idx in range(tracks.shape[0]):
        if cum_track_dists_aug[track_idx, -1] < 0.5 * np.mean(cum_track_dists_aug[track_idx]):
            # remove tracks that have not traveled enough distance
            vis[track_idx, :] = 0

    # Identify pairs of point clouds with significant distance to one another
    pair_matrix = np.zeros((tracks.shape[1], tracks.shape[1]), dtype=bool)

    # find first viable timestamp for start index
    for t_i in range(0, tracks.shape[1] - 1):
        if np.sum(vis[:, t_i]) > 0.2 * vis.shape[0]:
            start_idx = t_i
            print(t_i, "is the first viable timestamp for start index")
            break

    for t_j in range(start_idx + 1, tracks.shape[1]):
        # check if cumulative and absolute distance traveled is significant
        mean_cum_traveled_dist = np.mean(cum_track_dists_aug[:, t_j]) - np.mean(cum_track_dists_aug[:, start_idx])
        mean_abs_dist = np.mean(np.linalg.norm(tracks[:, t_j, :] - tracks[:, start_idx, :], axis=1))
        if mean_cum_traveled_dist > 0.1 * np.mean(cum_track_dists_aug[:, -1]) and mean_abs_dist > 0.1 * np.mean(cum_track_dists_aug[:, -1]):
            mutual_vis_points = vis[:, start_idx] * vis[:, t_j]
            if np.sum(mutual_vis_points) > 0:
                pairs[start_idx].append((tracks[mutual_vis_points, start_idx, :], tracks[mutual_vis_points, t_j, :]))
                pairs_t[start_idx].append(t_j)
                pair_matrix[start_idx, t_j] = True
                del mutual_vis_points, mean_cum_traveled_dist, mean_abs_dist

    return pairs, pairs_t


def prepare_estimation_data(traj_pairs: dict, limit: int = 1000):
    """
    Prepare data for axis estimation.
    - traj_pairs: pairs of point clouds of a given segment
    - limit: maximum number of pairs to consider
    Returns:
    - earlier_points: concatenated points forming the "starting" point clouds
    - later_points: concatenated points forming the successor point clouds
    - batch_idcs: indices for batching distinct point cloud pairs
    """
    earlier_points, later_points = [], []
    batch_idcs = [0]
    row_counter = 0
    for t_k, pairs in traj_pairs.items():
        for pair in pairs:
            if pair[0].shape[0] == 0 or pair[1].shape[0] == 0:
                print(f"Skipping empty pair at time {t_k} with shapes {pair[0].shape} and {pair[1].shape}")
                continue
            earlier_points.append(pair[0])
            later_points.append(pair[1])
            row_counter += pair[0].shape[0]
            batch_idcs.append(row_counter)
            if len(batch_idcs) > limit:
                break
        if len(batch_idcs) > limit:
            break
    batch_idcs.pop()  # remove last entry of batch_idcs
    earlier_points = np.concatenate(earlier_points, axis=0)
    later_points = np.concatenate(later_points, axis=0)

    return earlier_points, later_points, np.array(batch_idcs)


def angle_error(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle error between two vectors."""
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def pos_error(p: np.ndarray, x: np.ndarray, axis: np.ndarray) -> float:
    """
    REVOLUTES_ONLY
    Computes the distance between a point and a line defined by a supporting point and a direction vector.
    Args:
        p (_type_): Point
        x (_type_): Supporting point
        axis (_type_): Direction of line
    Returns:
        float: float
    """
    return np.linalg.norm(np.cross(p - x, axis), axis=-1)


def twist_transform(twist, theta, points):
    """
    Apply the twist transformation to a point cloud.
    - twist: 6D vector (angular velocity + linear velocity)
    - theta: rotation angle
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = gtsam.Pose3.Expmap(twist * theta).matrix() @ points_homogeneous.T
    return transformed_points.T[:, :3]  # Return only the 3D points, excluding the homogeneous coordinate


def prismatic_transform(twist, theta, points):
    """
    Apply translation transformation to points.
    - v: translation vector
    - theta: translation distance
    """
    v = twist[3:]
    rotation = np.eye(3)  # No rotation for prismatic joints
    translation = theta * v
    return (rotation @ points.T).T + translation  # Simply translate the points by theta


def revolute_transform(twist, theta, points):
    """
    Apply rotation transformation to points.
    - omega: rotation axis (unit vector)
    - theta: rotation angle
    """
    w = twist[:3]
    v = twist[3:]

    w_norm = np.linalg.norm(w)
    if np.isclose(w_norm, 0.0):
        return np.eye(3)

    w = w / w_norm  # unit axis
    # theta = theta * w_norm   # absorb magnitude into angle
    w_skew = _skew(w)
    R_theta = np.eye(3) + np.sin(theta) * w_skew + (1 - np.cos(theta)) * (w_skew @ w_skew)

    # used as a compromise
    return (R_theta @ points.T).T

    # the following two lines should be correct but do not work
    # translation = (np.eye(3) - R_theta) @ np.cross(w, v)
    # return (R_theta @ (points).T).T + translation


def twist_residual_pairs(params, p_start, p_end, batch_idcs):
    """
    TODO: remove before release
    Residual function for optimization.
    - twist: 6D vector (angular velocity + linear velocity)
    - thetas: rotation angles for each point cloud
    - P0: initial points (N x 3)
    - all_P: list of point sets at later timesteps [P1, P2, ..., Pk]
    """
    K = len(batch_idcs)  # number of pairs to consider, each pair consists of two point clouds

    twist = params[:6]
    thetas = params[6 : 6 + K]
    residuals = []

    # print(len(batch_idcs), "batches to process")
    for batch_idx in range(len(batch_idcs)):
        # print(batch_idx, ":", batch_idcs[batch_idx], batch_idcs[batch_idx + 1])
        # print(p_start.shape, p_end.shape)
        p_start_i = p_start[batch_idcs[batch_idx : batch_idx + 1]]
        p_end_i = p_end[batch_idcs[batch_idx : batch_idx + 1]]
        transformed = twist_transform(twist, thetas[batch_idx], p_start_i)
        residuals.append((transformed - p_end_i).ravel())
    return np.concatenate(residuals)


def twist_type_residual_pairs(params, p_start, p_end, batch_idcs, cos_median, cos_thresh=0.994, type_amb_weight=8000, alpha=1.0):
    """
    TODO: remove before release
    Residual function for optimization.
    - twist: 6D vector (angular velocity + linear velocity)
    - thetas: rotation angles for each point cloud
    - P0: initial points (N x 3)
    - all_P: list of point sets at later timesteps [P1, P2, ..., Pk]
    """
    K = len(batch_idcs)  # number of pairs to consider, each pair consists of two point clouds

    twist = params[:6]
    thetas = params[6 : 6 + K]
    twist_residuals = []

    # add twist residuals
    for batch_idx in range(len(batch_idcs)):
        # print(batch_idx, ":", batch_idcs[batch_idx], batch_idcs[batch_idx + 1])
        # print(p_start.shape, p_end.shape)
        p_start_i = p_start[batch_idcs[batch_idx : batch_idx + 1]]
        p_end_i = p_end[batch_idcs[batch_idx : batch_idx + 1]]
        transformed = twist_transform(twist, thetas[batch_idx], p_start_i)
        twist_residuals.append((transformed - p_end_i).ravel())

    type_cost = type_residual(twist, cos_median, sigmoid_weight=type_amb_weight, cos_thresh=cos_thresh)
    # add type residuals

    # print(np.concatenate(twist_residuals).shape, np.sum(np.concatenate(twist_residuals)))
    # print(np.full((len(twist_residuals)*3,), type_cost).shape, np.sum(np.full((len(twist_residuals)*3,), type_cost)))
    # print(np.append(np.concatenate(twist_residuals), np.full((len(twist_residuals)*3,), type_cost)).shape)
    return np.append(np.concatenate(twist_residuals), alpha * np.full((len(twist_residuals) * 3,), type_cost))
    # return np.append(np.concatenate(twist_residuals), np.array(type_cost)) # np.full((len(twist_residuals)*3,), type_cost))
    # return np.concatenate(twist_residuals)


def scaled_sigmoid(x, k, thresh):
    """Numerically-stable logistic sigmoid centered at `thresh` with steepness `k`."""
    z = -k * (x - thresh)
    z = np.clip(z, -500, 500)  # safe range
    return 1 / (1 + np.exp(z))


def type_residual(twist, value, sigmoid_weight, cos_thresh):
    """
    Joint-type regularization residual for a 6D twist [w; v].

    Blends a revolute cost (|dot(w, v)| scaled by their norms, encouraging w ⟂ v)
    and a prismatic cost (||w||, encouraging zero angular velocity), weighted by
    a sigmoid of `value` (typically the median pairwise cosine similarity) around
    `cos_thresh`: low `value` favors the revolute term, high `value` favors the
    prismatic term.
    """
    w_rev = 1 - scaled_sigmoid(value, sigmoid_weight, cos_thresh)
    w_prism = scaled_sigmoid(value, sigmoid_weight, cos_thresh)

    return w_rev * np.abs(np.dot(twist[:3], twist[3:]) / (np.linalg.norm(twist[:3])) * np.linalg.norm(twist[3:])) + w_prism * np.linalg.norm(
        twist[:3]
    )
    # return w_rev * np.abs(np.dot(twist[:3], twist[3:])) +  w_prism * np.linalg.norm(twist[:3])


def _type_residual_jacobian(twist, value, sigmoid_weight, cos_thresh):
    """Analytical 6D gradient of type_residual w.r.t. twist = [w; v]."""
    w_rev = 1 - scaled_sigmoid(value, sigmoid_weight, cos_thresh)
    w_prism = scaled_sigmoid(value, sigmoid_weight, cos_thresh)

    w = twist[:3]
    v = twist[3:]
    eps = 1e-8
    wn = np.linalg.norm(w)
    vn = np.linalg.norm(v)
    d = np.dot(w, v)

    jac = np.zeros(6)

    if wn > eps and vn > eps:
        s = np.sign(d) if abs(d) > eps else 0.0
        # df1/dw = vn/wn * (sign(d)*v - |d|/wn^2 * w)
        df1_dw = vn / wn * (s * v - abs(d) / wn**2 * w)
        # df1/dv = (sign(d)*vn*w + |d|/vn * v) / wn
        df1_dv = (s * vn * w + abs(d) / vn * v) / wn
        jac[:3] += w_rev * df1_dw
        jac[3:] += w_rev * df1_dv

    if wn > eps:
        jac[:3] += w_prism * w / wn

    return jac


def compute_type_prior(earlier_points: np.ndarray, later_points: np.ndarray, batch_idcs: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute a prior for the type of axis based on the pair-wise scaled dot product among
    all sampled vectors pointing from earlier to later points.
    - earlier_points: points forming the "starting" point clouds
    - later_points: points forming the successor point clouds
    - batch_idcs: indices for batching distinct point cloud pairs
    """
    vectors = later_points - earlier_points
    cos_sim = cosine_similarity(vectors)

    # # blank out the blocks defined by batch_idcs to not consider intra-batch dot products
    for i in range(len(batch_idcs) - 1):
        cos_sim[batch_idcs[i] : batch_idcs[i + 1], batch_idcs[i] : batch_idcs[i + 1]] = np.nan
    cos_sim = np.triu(cos_sim, k=1)  # keep only the upper triangle of the gram matrix

    # # upper diagonal part of the gram matrix
    cos_sim = np.abs(cos_sim)
    cos_sim[cos_sim == 0] = np.nan  # replace zeros with nan to circumvent affected statistics
    abs_dot_product = cos_sim.flatten()

    mean = np.nanmean(abs_dot_product, axis=0)
    std = np.nanstd(abs_dot_product, axis=0)
    median = np.nanmedian(abs_dot_product, axis=0)

    return cos_sim, mean, std, median


def prismatic_residual_pairs(params, p_start, p_end, batch_idcs):
    """
    TODO: remove before release
    Residual function for optimization.
    - twist: 6D vector (angular velocity + linear velocity)
    - thetas: rotation angles for each point cloud
    - P0: initial points (N x 3)
    - all_P: list of point sets at later timesteps [P1, P2, ..., Pk]
    """
    K = len(batch_idcs)  # number of pairs to consider, each pair consists of two point clouds

    twist = params[:6]
    thetas = params[6 : 6 + K]
    residuals = []

    for batch_idx in range(len(batch_idcs)):
        p_start_i = p_start[batch_idcs[batch_idx : batch_idx + 1]]
        p_end_i = p_end[batch_idcs[batch_idx : batch_idx + 1]]
        transformed = prismatic_transform(twist, thetas[batch_idx], p_start_i)
        residuals.append((transformed - p_end_i).ravel())
    return np.concatenate(residuals)


def revolute_residual_pairs(params, p_start, p_end, batch_idcs):
    """
    TODO: remove before release
    Residual function for optimization.
    - twist: 6D vector (angular velocity + linear velocity)
    - thetas: rotation angles for each point cloud
    - P0: initial points (N x 3)
    - all_P: list of point sets at later timesteps [P1, P2, ..., Pk]
    """
    K = len(batch_idcs)  # number of pairs to consider, each pair consists of two point clouds

    twist = params[:6]
    thetas = params[6 : 6 + K]
    residuals = []

    for batch_idx in range(len(batch_idcs)):
        p_start_i = p_start[batch_idcs[batch_idx : batch_idx + 1]]
        p_end_i = p_end[batch_idcs[batch_idx : batch_idx + 1]]
        transformed = revolute_transform(twist, thetas[batch_idx], p_start_i)
        residuals.append((transformed - p_end_i).ravel())
    return np.concatenate(residuals)


def estimate_dual_twist_numeric(batch_idcs: np.ndarray, earlier_points: np.ndarray, later_points: np.ndarray):
    """
    TODO remove before release
    Estimate the twist parameters using numerical optimization.
    - batch_idcs: indices for batching distinct point cloud pairs
    - earlier_points: concatenated points forming the "starting" point clouds
    - later_points: concatenated points forming the successor point clouds
    Returns:
    - opt_twist: optimized twist parameters (6D vector)
    - opt_thetas: optimized rotation angles for each point cloud
    - opt_pitch: optimized pitch parameter
    - type: type of joint ("revolute" or "prismatic")
    - opt_axis: optimized axis of rotation or translation
    - opt_center: optimized center of rotation or translation

    """
    # Provide an initial (random) guess of the twist and scaler parameters
    twist_init = np.array([0.0, 0.0, 0.0, 1.5, 1.5, 0.5])
    theta_init = np.array([0.0] * len(batch_idcs))
    x0 = np.concatenate([twist_init, theta_init])

    # Optimize a least-squares residual over point cloud pairs
    result_prismatic = least_squares(prismatic_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs))
    result_revolute = least_squares(twist_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs))

    # Check which result has a smaller residual
    if np.sum(result_prismatic.cost) <= np.sum(result_revolute.cost):
        type = "prismatic"
        result = result_prismatic
        print("Prismatic joint optimization succeeded with cost:", result.cost, "vs. revolute cost:", result_revolute.cost)
    else:
        type = "revolute"
        result = result_revolute
        print("Revolute joint optimization succeeded with cost:", result.cost, "vs. prismatic cost:", result_prismatic.cost)

    # Extract optimized parameters
    opt_twist = result.x[0:6]  # / np.linalg.norm(result.x[0:6])
    w_norm = np.linalg.norm(opt_twist[:3])
    v_norm = np.linalg.norm(opt_twist[3:])
    opt_thetas = result.x[6 : 6 + len(batch_idcs)]
    opt_pitch = np.dot(opt_twist[:3], opt_twist[3:]) / w_norm**2 if w_norm != 0 else np.inf

    print("optimized twist:", opt_twist)
    print("optimized thetas:", opt_thetas)
    print("optimized pitch:", opt_pitch)

    if type == "revolute":
        # small optimized pitch indicates a revolute joint
        print(f"joint type: {type}")
        opt_axis = opt_twist[:3] / w_norm
        opt_center = np.cross(opt_twist[:3], opt_twist[3:]) / w_norm**2
        opt_pitch = opt_twist[:3].dot(opt_twist[3:]) / w_norm**2
        # print("angle error:", np.rad2deg(angle_error(opt_axis, true_axis)))
        # print("translation error:", pos_error(opt_center, true_center, opt_axis))
    else:
        # large optimized pitch indicates a prismatic joint

        print(f"joint type: {type}")
        opt_axis = opt_twist[3:] / v_norm if v_norm != 0 else np.array([0, 0, 0])  # supposedly correct
        # opt_axis = opt_twist[:3] / np.linalg.norm(opt_twist[:3]) if np.linalg.norm(opt_twist[:3]) != 0 else np.array([0, 0, 0])
        opt_center = np.mean(earlier_points[0 : batch_idcs[1]], axis=0)  # Center can be estimated as the mean of initial points
        # opt_pitch = np.inf if np.all(opt_twist[:3] == 0) else 0
        # print("angle error:", np.rad2deg(angle_error(opt_axis, true_axis)))
        print("translation error:", "-- N/A for prismatic joint --")

    return InferencePointAxis(
        position=opt_center,
        axis=opt_axis,
        twist=opt_twist,
        type=type,
        pitch=opt_pitch,
        thetas=opt_thetas,
    )


def estimate_twist_lq(
    batch_idcs: np.ndarray, earlier_points: np.ndarray, later_points: np.ndarray, pitch_thresh: float = 0.5, optim_params: Optional[dict] = None
):
    """
    TODO remove before release
    Estimate the twist parameters using numerical optimization.
    - batch_idcs: indices for batching distinct point cloud pairs
    - earlier_points: concatenated points forming the "starting" point clouds
    - later_points: concatenated points forming the successor point clouds
    Returns:
    - opt_twist: optimized twist parameters (6D vector)
    - opt_thetas: optimized rotation angles for each point cloud
    - opt_pitch: optimized pitch parameter
    - type: type of joint ("revolute" or "prismatic")
    - opt_axis: optimized axis of rotation or translation
    - opt_center: optimized center of rotation or translation

    """
    # Provide an initial (random) guess of the twist and scaler parameters
    twist_init = np.array([0.0, 0.0, 0.0, 1.5, 1.5, 0.5])
    theta_init = np.array([0.0] * len(batch_idcs))
    x0 = np.concatenate([twist_init, theta_init])

    # Optimize a least-squares residual over point cloud pairs
    if optim_params is not None:
        result = least_squares(twist_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs), **optim_params)
    else:
        result = least_squares(twist_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs))

    # Extract optimized parameters
    opt_twist = result.x[0:6]  # / np.linalg.norm(result.x[0:6])
    w_norm = np.linalg.norm(opt_twist[:3])
    v_norm = np.linalg.norm(opt_twist[3:])
    opt_thetas = result.x[6 : 6 + len(batch_idcs)]
    opt_pitch = np.dot(opt_twist[:3], opt_twist[3:]) / w_norm**2 if w_norm != 0 else np.inf

    print("optimized twist:", opt_twist)
    print("optimized thetas:", opt_thetas)
    print("optimized pitch:", opt_pitch)

    collinearity = opt_twist[:3].dot(opt_twist[3:]) / (w_norm * v_norm)  # cosine of the angle between angular and linear velocity

    if np.abs(collinearity) < 0.02:
        # small optimized pitch indicates a revolute joint
        type = "revolute"
        print(f"joint type: {type}")
        opt_axis = opt_twist[:3] / w_norm

        # variant A (under the assumption that v and w are orthogonal given the collinearity)
        opt_center = np.cross(opt_twist[:3], opt_twist[3:]) / w_norm**2
        # variant B
        # orthogonal_part_v = opt_twist[3:] - opt_pitch * opt_twist[:3]
        # opt_center = np.cross(opt_twist[:3], orthogonal_part_v) / w_norm**2

        # print("center", opt_center)
        opt_pitch = opt_twist[:3].dot(opt_twist[3:]) / w_norm**2
        # print(f"Estimated radius of rotation: {radius:.2f}")
        # print("angle error:", np.rad2deg(angle_error(opt_axis, true_axis)))
        # print("translation error:", pos_error(opt_center, true_center, opt_axis))
    else:
        type = "prismatic"
        print(f"joint type: {type}")
        if np.abs(opt_pitch) < pitch_thresh:
            # catch case of high collinearity but low pitch
            opt_axis = opt_twist[:3] / np.linalg.norm(opt_twist[:3]) if np.linalg.norm(opt_twist[:3]) != 0 else np.array([0, 0, 0])
        else:
            opt_axis = opt_twist[3:] / v_norm if v_norm != 0 else np.array([0, 0, 0])  # supposedly correct
        opt_center = np.mean(earlier_points[0 : batch_idcs[1]], axis=0)  # Center can be estimated as the mean of initial points
        # opt_pitch = np.inf if np.all(opt_twist[:3] == 0) else 0
        # print("angle error:", np.rad2deg(angle_error(opt_axis, true_axis)))
        # print("translation error:", "-- N/A for prismatic joint --")

    return InferencePointAxis(
        position=opt_center,
        axis=opt_axis,
        twist=opt_twist,
        type=type,
        pitch=opt_pitch,
        thetas=opt_thetas,
    )


def estimate_articulation_model_rev(cfg, tracks: np.ndarray, vis: np.ndarray):
    """
    TODO: remove before release
    Fit a revolute-only twist model to point tracks via least-squares optimization
    (assumes the joint is known a priori to be revolute; used for privileged/oracle
    baselines rather than automatic type disambiguation).
    - tracks: N x T x 3 array of point tracks
    - vis: N x T array of visibility (1 if visible, 0 otherwise)
    Returns (articulation_model, pairs, pairs_t); on failure to find pairs, returns
    an InferencePointAxis with success=False.
    """
    pairs, pairs_t = identify_cloud_pairs_from_start(tracks, vis)
    if len(pairs) == 0:
        return InferencePointAxis(position=None, axis=None, type=None, success=False), pairs, pairs_t
    earlier_points, later_points, batch_idcs = prepare_estimation_data(pairs, limit=1000)

    # Provide an initial (random) guess of the twist and scaler parameters
    twist_init = np.array(cfg.articulation.twist_init)
    theta_init = np.array([cfg.articulation.theta_init] * len(batch_idcs))
    x0 = np.concatenate([twist_init, theta_init])

    # Optimize a least-squares residual over point cloud pairs
    if cfg.articulation.optim_params is not None:
        result = least_squares(revolute_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs), **cfg.articulation.optim_params)
    else:
        result = least_squares(
            revolute_residual_pairs,
            x0,
            args=(earlier_points, later_points, batch_idcs),
        )
    print("optimized_twist:", result.x[0:6])

    # Extract optimized parameters
    opt_twist = result.x[0:6]  # / np.linalg.norm(result.x[0:6])
    w_norm = np.linalg.norm(opt_twist[:3])
    v_norm = np.linalg.norm(opt_twist[3:])
    opt_thetas = result.x[6 : 6 + len(batch_idcs)]
    opt_pitch = np.dot(opt_twist[:3], opt_twist[3:]) / w_norm**2 if w_norm != 0 else np.inf
    collinearity = opt_twist[:3].dot(opt_twist[3:]) / (w_norm * v_norm)  # cosine of the angle between angular and linear velocity

    type = "revolute"
    print(f"joint type: {type}")
    opt_axis = opt_twist[:3] / w_norm
    # variant A (under the assumption that v and w are orthogonal given the collinearity)
    # opt_center = np.cross(opt_twist[:3], opt_twist[3:]) / w_norm**2
    opt_center = np.zeros(3)
    # variant B
    # orthogonal_part_v = opt_twist[3:] - opt_pitch * opt_twist[:3]
    # opt_center = np.cross(opt_twist[:3], orthogonal_part_v) / w_norm**2
    print("center", opt_center)
    print("collinearity:", collinearity)

    articulation_model = InferencePointAxis(
        position=opt_center,
        axis=opt_axis,
        type=type,
        twist=opt_twist,
        pitch=opt_pitch,
        thetas=opt_thetas,
        bounds=(np.min(opt_thetas), np.max(opt_thetas)),
        success=True,
        loss=result.cost,
    )
    return articulation_model, pairs, pairs_t


def estimate_articulation_model_pris(cfg, tracks: np.ndarray, vis: np.ndarray):
    """
    TODO: remove before release
    Fit a prismatic-only twist model to point tracks via least-squares optimization
    (assumes the joint is known a priori to be prismatic; used for privileged/oracle
    baselines rather than automatic type disambiguation).
    - tracks: N x T x 3 array of point tracks
    - vis: N x T array of visibility (1 if visible, 0 otherwise)
    Returns (articulation_model, pairs, pairs_t); on failure to find pairs, returns
    an InferencePointAxis with success=False.
    """
    pairs, pairs_t = identify_cloud_pairs_from_start(tracks, vis)
    if len(pairs) == 0:
        return InferencePointAxis(position=None, axis=None, type=None, success=False), pairs, pairs_t
    earlier_points, later_points, batch_idcs = prepare_estimation_data(pairs, limit=1000)

    # Provide an initial (random) guess of the twist and scaler parameters
    twist_init = np.array(cfg.articulation.twist_init)
    theta_init = np.array([cfg.articulation.theta_init] * len(batch_idcs))
    x0 = np.concatenate([twist_init, theta_init])

    # Optimize a least-squares residual over point cloud pairs
    if cfg.articulation.optim_params is not None:
        result = least_squares(prismatic_residual_pairs, x0, args=(earlier_points, later_points, batch_idcs), **cfg.articulation.optim_params)
    else:
        result = least_squares(
            prismatic_residual_pairs,
            x0,
            args=(earlier_points, later_points, batch_idcs),
        )
    print("optimized_twist:", result.x[0:6])

    # Extract optimized parameters
    opt_twist = result.x[0:6]  # / np.linalg.norm(result.x[0:6])
    w_norm = np.linalg.norm(opt_twist[:3])
    v_norm = np.linalg.norm(opt_twist[3:])
    opt_thetas = result.x[6 : 6 + len(batch_idcs)]
    opt_pitch = np.dot(opt_twist[:3], opt_twist[3:]) / w_norm**2 if w_norm != 0 else np.inf
    collinearity = opt_twist[:3].dot(opt_twist[3:]) / (w_norm * v_norm)  # cosine of the angle between angular and linear velocity

    type = "prismatic"
    print(f"joint type: {type}")
    opt_axis = opt_twist[3:] / v_norm if v_norm != 0 else np.array([0, 0, 0])  # supposedly correct
    opt_center = np.mean(earlier_points[0 : batch_idcs[1]], axis=0)  # Center can be estimated as the mean of initial points
    print("center", opt_center)
    print("collinearity:", collinearity)

    articulation_model = InferencePointAxis(
        position=opt_center,
        axis=opt_axis,
        type=type,
        twist=opt_twist,
        pitch=opt_pitch,
        thetas=opt_thetas,
        bounds=(np.min(opt_thetas), np.max(opt_thetas)),
        success=True,
        loss=result.cost,
    )
    return articulation_model, pairs, pairs_t


def identify_cloud_pairs_from_multiple_starts(
    tracks: np.ndarray,
    vis: np.ndarray,
):
    """
    Two-stage pair sampling:
      Stage 1 (forward):  anchor = first viable frame → pairs with all later frames.
      Stage 2 (backward): anchor = last viable frame  → pairs with all earlier frames.

    Using both ends gives full coverage for opening-closing motions where
    the first and last frames share the same state — the backward stage
    captures displacement relative to the terminal configuration that the
    forward stage misses.
    """
    pairs, pairs_t = defaultdict(list), defaultdict(list)
    if tracks.shape[0] == 0 or tracks.shape[1] == 0:
        return pairs, pairs_t

    N, T = tracks.shape[0], tracks.shape[1]

    # cumulative arc lengths per track
    incr = np.linalg.norm(tracks[:, 1:, :] - tracks[:, :-1, :], axis=2)  # (N, T-1)
    cum = np.zeros((N, T))
    for t in range(1, T):
        cum[:, t] = np.sum(incr[:, :t], axis=1)

    if np.mean(cum[:, -1]) < 0.10:
        print("Tracks have not traveled enough distance, returning empty pairs.")
        return pairs, pairs_t

    # suppress low-travel tracks
    vis = vis.copy().astype(float)
    for n in range(N):
        if cum[n, -1] < 0.5 * np.mean(cum[n]):
            vis[n, :] = 0

    viable = [t for t in range(T) if vis[:, t].sum() > 0.2 * N]
    if not viable:
        return pairs, pairs_t

    total_arc = np.mean(cum[:, -1])
    start_fwd = viable[0]
    start_bwd = viable[-1]

    def _add_pairs(anchor, candidates):
        """Pair `anchor` with each frame in `candidates` that shows sufficient
        cumulative and absolute displacement, appending mutually-visible point pairs."""
        for t_j in candidates:
            lo, hi = min(anchor, t_j), max(anchor, t_j)
            mean_cum = np.mean(cum[:, hi]) - np.mean(cum[:, lo])
            mean_abs = np.mean(np.linalg.norm(tracks[:, t_j, :] - tracks[:, anchor, :], axis=1))
            if mean_cum > 0.1 * total_arc and mean_abs > 0.1 * total_arc:
                mutual = (vis[:, anchor] * vis[:, t_j]).astype(bool)
                if mutual.sum() > 0:
                    pairs[anchor].append((tracks[mutual, anchor, :], tracks[mutual, t_j, :]))
                    pairs_t[anchor].append(t_j)

    # stage 1: forward from first viable frame
    _add_pairs(start_fwd, range(start_fwd + 1, T))
    # stage 2: backward from last viable frame
    _add_pairs(start_bwd, range(0, start_bwd))

    return pairs, pairs_t


def estimate_articulation_model(cfg, tracks: np.ndarray, vis: np.ndarray, cos_median: float, alpha=1.0):
    """
    Jointly estimate a single 6D twist and per-pair thetas via GTSAM factor-graph
    (Levenberg-Marquardt) optimization, with an optional soft joint-type prior
    (`type_residual`) that biases the twist toward revolute or prismatic behaviour
    based on `cos_median`.

    - tracks: N x T x 3 array of point tracks
    - vis: N x T array of visibility (1 if visible, 0 otherwise)
    - cos_median: median pairwise cosine similarity (from `compute_type_prior`),
      used both to weight the type prior and to pick revolute vs. prismatic
      post-hoc axis/center extraction
    - alpha: weight applied to the type-regularization factor and its jacobian

    Returns (articulation_model, pairs, pairs_t); on failure to find pairs, returns
    an InferencePointAxis with success=False.
    """
    pairs, pairs_t = identify_cloud_pairs_from_multiple_starts(tracks, vis)
    if len(pairs) == 0:
        return InferencePointAxis(position=None, axis=None, type=None, success=False), pairs, pairs_t
    earlier_points, later_points, batch_idcs = prepare_estimation_data(pairs, limit=1000)

    twist_init = np.array(cfg.articulation.twist_init, dtype=float)
    theta_init_val = float(cfg.articulation.theta_init)
    K = len(batch_idcs)
    cos_thresh = cfg.articulation.cos_thresh
    type_amb_weight = cfg.articulation.type_amb_weight

    # GTSAM keys: one 6D twist variable and K scalar theta variables
    twist_key = gtsam.symbol('w', 0)
    theta_keys = [gtsam.symbol('s', k) for k in range(K)]

    graph = gtsam.NonlinearFactorGraph()
    noise_3d = gtsam.noiseModel.Unit.Create(3)

    # One 3D point factor per batch (mirrors the per-batch point used in the scipy residual)
    def make_point_error(ps, pe):
        """Build a GTSAM CustomFactor error function that penalizes the discrepancy
        between the twist-transformed start point `ps` and the observed end point `pe`."""

        def error_func(this, values, jacobians):
            """GTSAM error function: returns Exp(twist*theta) @ ps - pe, filling in
            the analytical jacobians w.r.t. twist and theta when requested."""
            keys = this.keys()
            twist = values.atVector(keys[0])
            theta = values.atDouble(keys[1])
            xi = twist * theta

            if jacobians is not None:
                D_expmap = np.zeros((6, 6), order='F')
                T = gtsam.Pose3.Expmap(xi, D_expmap)
                D_trans = np.zeros((3, 6), order='F')
                D_pt = np.zeros((3, 3), order='F')
                pt = np.asarray(T.transformFrom(ps, D_trans, D_pt))
                J_xi_p = D_trans @ D_expmap  # 3x6: d(T*p)/d(xi)
                jacobians[0] = J_xi_p * theta  # 3x6: d(error)/d(twist)
                jacobians[1] = J_xi_p @ twist  # 3D: d(error)/d(theta)
            else:
                T = gtsam.Pose3.Expmap(xi)
                pt = np.asarray(T.transformFrom(ps))

            return pt - pe

        return error_func

    for k in range(K):
        p_s = earlier_points[batch_idcs[k]].copy()
        p_e = later_points[batch_idcs[k]].copy()
        graph.add(gtsam.CustomFactor(noise_3d, [twist_key, theta_keys[k]], make_point_error(p_s, p_e)))

    # Type regularization factor: matches scipy's alpha * np.full(K*3, type_cost) appended residuals
    noise_type = gtsam.noiseModel.Unit.Create(K * 3)

    def type_error(this, values, jacobians):
        """GTSAM error function for the type-regularization factor: returns
        `alpha * type_residual(twist, ...)` tiled to length K*3, with its jacobian
        w.r.t. twist when requested."""
        twist = values.atVector(this.keys()[0])
        cost = type_residual(twist, cos_median, sigmoid_weight=type_amb_weight, cos_thresh=cos_thresh)
        if jacobians is not None:
            jac = _type_residual_jacobian(twist, cos_median, type_amb_weight, cos_thresh)
            jacobians[0] = np.tile(alpha * jac, (K * 3, 1))  # (K*3, 6)
        return np.full(K * 3, alpha * cost)

    if cfg.articulation.use_twist_regularization:
        graph.add(gtsam.CustomFactor(noise_type, [twist_key], type_error))

    initial = gtsam.Values()
    initial.insert(twist_key, twist_init)
    for k in range(K):
        initial.insert(theta_keys[k], theta_init_val)

    # Optimize with Levenberg-Marquardt
    lm_params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
    result = optimizer.optimize()
    loss = graph.error(result)

    opt_twist = result.atVector(twist_key)
    opt_thetas = np.array([result.atDouble(theta_keys[k]) for k in range(K)])
    print("optimized_twist:", opt_twist)

    # Extract optimized parameters
    w_norm = np.linalg.norm(opt_twist[:3])
    v_norm = np.linalg.norm(opt_twist[3:])
    opt_pitch = np.dot(opt_twist[:3], opt_twist[3:]) / w_norm**2 if w_norm != 0 else np.inf
    collinearity = opt_twist[:3].dot(opt_twist[3:]) / (w_norm * v_norm)

    # ALGORITHM RELYING ONLY ON SECANT COSINE CRITERION
    # if the median cosine similarity is above the threshold, we assume a revolute joint
    if cos_median < cfg.articulation.cos_thresh:
        type = "revolute"
        print(f"joint type: {type}")
        opt_axis = opt_twist[:3] / w_norm
        # variant A (under the assumption that v and w are orthogonal given the collinearity)
        opt_center = np.cross(opt_twist[:3], opt_twist[3:]) / w_norm**2
        # variant B
        # orthogonal_part_v = opt_twist[3:] - opt_pitch * opt_twist[:3]
        # opt_center = np.cross(opt_twist[:3], orthogonal_part_v) / w_norm**2

        # shift center along the axis direction to be closer to the observed points
        closest_point_on_axis = opt_center + np.dot(np.mean(tracks[:, 0], axis=0) - opt_center, opt_axis) * opt_axis
        opt_center = closest_point_on_axis
    else:
        type = "prismatic"
        print(f"joint type: {type}")
        opt_axis = opt_twist[3:] / v_norm if v_norm != 0 else np.array([0, 0, 0])
        opt_center = np.mean(tracks[:, 0], axis=0)
    print("center", opt_center)
    print("axis:", opt_axis)

    articulation_model = InferencePointAxis(
        position=opt_center,
        axis=opt_axis,
        type=type,
        collinearity=collinearity,
        cos_median=cos_median,
        twist=opt_twist,
        pitch=opt_pitch,
        thetas=opt_thetas,
        bounds=(np.min(opt_thetas), np.max(opt_thetas)),
        success=True,
        loss=loss,
    )
    return articulation_model, pairs, pairs_t


def estimate_motion_mode(y):
    """
    Estimate whether the motion is a combined opening/closing motion or a single motion (opening or closing).
    y: list of observed scalers over time
    Returns:
    - "single" or "cycle"
    """
    # -- Fit quadratic model --
    x = np.linspace(0, len(y) - 1, len(y))
    Xq = np.vstack([np.ones_like(x), x, x**2]).T
    beta_q, *_ = np.linalg.lstsq(Xq, y, rcond=None)
    y_q = Xq.dot(beta_q)

    # compute extremal point of quadratic function
    x_root = -beta_q[1] / (2 * beta_q[2])
    if x_root < 0.3 * len(y) or x_root > len(y) - (0.3 * len(y)):
        return "single"
    else:
        return "cycle"


def estimate_dense_thetas(
    twist: np.ndarray,
    tracks: np.ndarray,
    vis: np.ndarray,
    start_frame: int,
    theta_bounds: Tuple[float, float] = (-np.pi, np.pi),
    min_mutual: int = 3,
) -> Tuple[np.ndarray, list]:
    """
    Given a fixed twist, solve for theta at every frame in the segment via a
    bounded 1-D optimisation, yielding a dense per-frame joint-state sequence.

    The twist is held fixed (as returned by estimate_articulation_model); only
    the scalar theta is optimised for each frame t:

        theta_t = argmin_theta  sum_k || Expmap(twist * theta) @ p_start_k - p_t_k ||^2

    where the sum runs over all tracks mutually visible at start_frame and t.

    Args:
        twist:        (6,) screw axis (world frame, same convention as model.twist).
        tracks:       (N, T, 3) 3-D track positions.
        vis:          (N, T) visibility mask (bool or float).
        start_frame:  Reference frame index; theta is 0 by definition here.
        theta_bounds: (lo, hi) search interval for theta.  Use model.bounds when
                      available; defaults to (-pi, pi) suitable for revolute joints.
        min_mutual:   Minimum number of co-visible tracks required to attempt
                      optimisation; frames below this threshold get theta=nan.

    Returns:
        dense_thetas:        (T,) per-frame theta estimates (nan where undetermined).
        dense_frame_indices: list(range(T)) — kept parallel to dense_thetas for
                             convenient zip-iteration with frame numbers.
    """
    N, T, _ = tracks.shape
    vis_bool = vis.astype(bool)
    p_start = tracks[:, start_frame, :]  # (N, 3)
    start_vis = vis_bool[:, start_frame]  # (N,)
    dense_thetas = np.full(T, np.nan, dtype=np.float64)

    # Expand bounds well beyond the sparse theta range so the optimiser is not
    # clipped to a constant at frames before/after the visible motion window.
    # model.bounds = (min(opt_thetas), max(opt_thetas)) may not include the rest
    # state or post-motion state, causing a flat line at the boundary values.
    lo, hi = theta_bounds
    margin = max(abs(hi - lo) * 0.5, 0.1)
    bounded_interval = (lo - margin, hi + margin)

    for t in range(T):
        mutual = start_vis & vis_bool[:, t]
        if mutual.sum() < min_mutual:
            continue

        ps = p_start[mutual]  # (M, 3)
        pt = tracks[:, t, :][mutual]  # (M, 3)
        ps_h = np.hstack([ps, np.ones((len(ps), 1))])  # (M, 4)

        def residual(theta, ps_h=ps_h, pt=pt):
            """Sum of squared point errors between the twist-transformed reference
            points and the observed points `pt` at candidate joint state `theta`."""
            T_mat = gtsam.Pose3.Expmap(twist * theta).matrix()
            pred = (T_mat @ ps_h.T)[:3].T  # (M, 3)
            return np.sum((pred - pt) ** 2)

        result = minimize_scalar(residual, bounds=bounded_interval, method='bounded')
        dense_thetas[t] = result.x

    return dense_thetas, list(range(T))


def plot_estimate(model, pairs, tracks, save_dir):
    """
    Plot point pairs, estimated axis of motion, and twist components.
    In addition, this also plots the thetas over time.
    - model: estimated articulation model
    - pairs: point cloud pairs used for estimation
    - tracks: point tracks for context
    """
    fig = plt.figure(figsize=(18, 6))

    # plot #0: sampled pairs along all trajectories
    ax0 = fig.add_subplot(131, projection='3d')
    for t_i, list_of_pairs in pairs.items():
        for pair in list_of_pairs:
            if pair[0].shape[0] == 0:
                continue
            for row_idx, _ in enumerate(pair[0]):
                start = pair[0][row_idx]
                end = pair[1][row_idx]
                ax0.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], marker='o')

    # plot #1: point tracks and identified axis of motion
    ax1 = fig.add_subplot(132, projection='3d')
    if tracks is not None and tracks.shape[0] > 0:
        # adjust the bounds for equally scaled axes
        bounds = np.array(
            [
                [np.min(tracks[:, :, 0]), np.min(tracks[:, :, 1]), np.min(tracks[:, :, 2])],
                [np.max(tracks[:, :, 0]), np.max(tracks[:, :, 1]), np.max(tracks[:, :, 2])],
            ]
        )
        coord_max = np.argmax([bounds[1, 0], bounds[1, 1], bounds[1, 2]])
        max_offset = bounds[1, coord_max] - bounds[0, coord_max]
        for i in range(3):
            if i == coord_max:
                continue
            bounds[0, i] = np.median(tracks[:, :, i]) - max_offset / 2
            bounds[1, i] = np.median(tracks[:, :, i]) + max_offset / 2
        ax1.set_xlim([bounds[0, 0], bounds[1, 0]])
        ax1.set_ylim([bounds[0, 1], bounds[1, 1]])
        ax1.set_zlim([bounds[0, 2], bounds[1, 2]])

        # plot point trajectories
        for track_idx in range(tracks.shape[0]):
            ax1.plot(tracks[track_idx, :, 0], tracks[track_idx, :, 1], tracks[track_idx, :, 2], marker='o')

    # identified axis over
    ax1.quiver(
        model.position[0],
        model.position[1],
        model.position[2],
        model.axis[0],
        model.axis[1],
        model.axis[2],
        length=0.1,
        color='g',
        label='Estimated axis of motion',
    )
    # ax1.set_title(f"{segment_idx}: {type} joint, collinearity: {collinearity:.2f}, pitch: {opt_pitch:.2f}")
    ax1.set_title(
        f"{model.id}: {model.type} joint, collinearity: {model.collinearity:.4f}, pitch: {model.pitch:.4f}, cosine median: {model.cos_median:.5f}, loss: {model.loss:.4f}"
    )
    plt.legend()

    # plot #2: plot twist components
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.quiver(0, 0, 0, model.twist[:3][0], model.twist[:3][1], model.twist[:3][2], length=0.1, color='g', label='rot axis')
    ax2.quiver(0, 0, 0, model.twist[3:][0], model.twist[3:][1], model.twist[3:][2], length=0.1, color='r', label='trans axis')
    plt.legend()

    if not os.path.exists(os.path.join(save_dir, "articulation")):
        os.makedirs(os.path.join(save_dir, "articulation"), exist_ok=True)
    plt.savefig(os.path.join(save_dir, "articulation", f"segment_{model.id:04d}_estimated_axis.png"))
    plt.close()

    # plot model theta over time
    plt.figure()
    plt.plot(model.thetas)
    plt.title(f"Articulation Model Thetas over Time for Segment {i}")
    plt.xlabel("Time step (relative)")
    plt.ylabel("Theta value")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "articulation", f"segment_{i:04d}_thetas.png"))
    plt.close()


def plot_dense_theta_estimate(model, save_dir):
    """Plot the dense per-frame theta estimates (from `estimate_dense_thetas`) against
    the original sparse thetas used for axis estimation, and save the figure to
    `save_dir/articulation/segment_<id>_dense_thetas.png`."""
    sparse_frames = list(model.pairs_t.values())[0]  # frame indices relative to segment start
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model.dense_frame_indices, model.dense_thetas, color="steelblue", linewidth=1.5, label="dense thetas")
    ax.scatter(sparse_frames, model.thetas[: len(sparse_frames)], color="tomato", zorder=5, s=40, label="original (sparse) thetas")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("frame index (relative to segment start)")
    ax.set_ylabel("theta")
    ax.set_title(f"Segment {model.id} — sparse vs. dense thetas ({model.type})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "articulation", f"segment_{model.id:04d}_dense_thetas.png"), dpi=120)
    plt.close(fig)
    return
