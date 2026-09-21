from typing import Any, Dict, List, Optional, Tuple
import warnings

import gtsam
import loguru
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering


class TrajFiltering:
    """Filtering pipeline that cleans up per-segment 3D point tracks (smoothing,
    removing static/jerky/short/occluded/off-arc tracks, and clustering out
    non-rigid outliers) before articulation estimation."""

    def __init__(self, cfg: Dict[str, Any], bidirectional: bool) -> None:
        """
        :param cfg: full config; uses cfg.filtering for filter thresholds/options.
        :param bidirectional: if True, also filter the backward-tracked segments
            passed to `forward`.
        """
        self.params = cfg.filtering
        self.bidirectional = bidirectional

    def forward(
        self,
        pred_segments: List[Tuple[int, int]],
        cam_poses: Dict[int, np.ndarray],
        forw_3d_tracks_segments: List[np.ndarray],
        forw_vis_segments: List[np.ndarray],
        backw_3d_tracks_segments: Optional[List[np.ndarray]] = None,
        backw_vis_segments: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the full filtering pipeline (smoothing, smoothness/jerk filtering,
        length-span filtering, camera-distance filtering, occlusion filtering,
        arc-fit filtering, and DBSCAN rigidity clustering) on the forward-tracked
        segments, and optionally on the backward-tracked segments as well.

        :param pred_segments: list of (start_idx, end_idx) frame ranges, one per segment.
        :param cam_poses: mapping from frame index to 4x4 camera pose.
        :param forw_3d_tracks_segments: list of (T, N, 3) forward point tracks, one per segment.
        :param forw_vis_segments: list of (T, N) forward visibility masks, one per segment.
        :param backw_3d_tracks_segments: optional list of (T, N, 3) backward point tracks.
        :param backw_vis_segments: optional list of (T, N) backward visibility masks.
        :return: tuple (forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments,
            backw_vis_segments) after filtering (backward outputs are None/unchanged if not
            bidirectional or not provided).
        """
        if self.params.smooth_tracks:
            forw_3d_tracks_segments, forw_vis_segments = self.smooth_tracks(forw_3d_tracks_segments, forw_vis_segments)

            # pred_3d_tracks_smoothed = copy.deepcopy(forw_3d_tracks_segments)
            # pred_vis_smoothed = copy.deepcopy(forw_vis_segments)

        # Filter out static and jerky points
        forw_3d_tracks_segments, forw_vis_segments, _, _ = self.filter_smoothness_noise(forw_3d_tracks_segments, forw_vis_segments)

        forw_3d_tracks_segments, forw_vis_segments = self.filter_length_span(forw_3d_tracks_segments, forw_vis_segments)
        forw_3d_tracks_segments, forw_vis_segments = self.filter_dist_cam(
            cam_poses, forw_3d_tracks_segments, forw_vis_segments, pred_segments, self.params.max_dist_cam
        )

        # Filter out unreliable tracks
        if self.params.filter_occluded_tracks:
            forw_3d_tracks_segments, forw_vis_segments = self.filter_occluded_tracks(forw_3d_tracks_segments, forw_vis_segments)

        # Filter out point tracks that show considerable drift (not following a rigid body constraint) after smoothing, as they are likely unreliable
        forw_3d_tracks_segments, forw_vis_segments = self.filter_arc_fit(
            forw_3d_tracks_segments, forw_vis_segments, res_thresh=self.params.arc_fit_res_thresh, min_visible=self.params.arc_fit_min_visible
        )

        # pred_3d_tracks_filtered = copy.deepcopy(forw_3d_tracks_segments)
        # pred_vis_filtered = copy.deepcopy(forw_vis_segments)

        forw_3d_tracks_segments, forw_vis_segments = self.filter_dbscan(
            forw_3d_tracks_segments,
            forw_vis_segments,
            eps=self.params.dbscan.eps,
            metric=self.params.dbscan.metric,  # rigidity
            criterion=self.params.dbscan.criterion,  # spatial_extent / longest / largest,
        )

        if self.bidirectional and backw_3d_tracks_segments is not None and backw_vis_segments is not None:
            if self.params.smooth_tracks:
                backw_3d_tracks_segments, backw_vis_segments = self.smooth_tracks(backw_3d_tracks_segments, backw_vis_segments)

            backw_3d_tracks_segments, backw_vis_segments, _, _ = self.filter_smoothness_noise(backw_3d_tracks_segments, backw_vis_segments)
            backw_3d_tracks_segments, backw_vis_segments = self.filter_length_span(backw_3d_tracks_segments, backw_vis_segments)
            backw_3d_tracks_segments, backw_vis_segments = self.filter_dist_cam(
                cam_poses, backw_3d_tracks_segments, backw_vis_segments, pred_segments, self.params.max_dist_cam
            )
            backw_3d_tracks_segments, backw_vis_segments = self.filter_arc_fit(
                backw_3d_tracks_segments, backw_vis_segments, res_thresh=0.03, min_visible=6
            )
            if self.params.filter_occluded_tracks:
                backw_3d_tracks_segments, backw_vis_segments = self.filter_occluded_tracks(backw_3d_tracks_segments, backw_vis_segments)
            backw_3d_tracks_segments, backw_vis_segments = self.filter_dbscan(backw_3d_tracks_segments, backw_vis_segments)

        return forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments

    def smooth_tracks(self, tracks_3d_segments: List[np.ndarray], vis_segments: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Smooth each segment's 3D tracks via `smooth_trajectory_optimization`
        (velocity/jerk-regularized least squares), using self.params.lambda_vel
        and self.params.lambda_jerk. Visibility masks are returned unchanged."""
        # Smooth out the tracks
        tracks_3d_segments_smooth = []
        for i in range(len(tracks_3d_segments)):
            track = smooth_trajectory_optimization(
                tracks_3d_segments[i],
                vis_segments[i],
                lambda_vel=self.params.lambda_vel,
                lambda_jerk=self.params.lambda_jerk,
            )
            tracks_3d_segments_smooth.append(track)
        return tracks_3d_segments_smooth, vis_segments

    def filter_smoothness_noise(
        self,
        tracks_3d_segments: List[np.ndarray],
        vis_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        filter out static points and jerky points
        """
        # create new dict of static tracks per segment
        static_3d_tracks_segments = {}
        static_visibility_segments = {}
        for i in range(len(tracks_3d_segments)):
            tracks_3d_segments[i], vis_segments[i] = self.filter_non_smooth_tracks(
                tracks_3d_segments[i],
                vis_segments[i],
                thrsh=self.params.smoothness_threshold,
            )
            tracks_3d_segments[i], vis_segments[i], static_3d_tracks_segments[i], static_visibility_segments[i] = self.median_filter(
                tracks_3d_segments[i],
                vis_segments[i],
                variance_type=self.params.variance_type,
                percentile=self.params.percentile,
            )

        return tracks_3d_segments, vis_segments, static_3d_tracks_segments, static_visibility_segments

    def filter_length_span(
        self,
        tracks_3d_segments: List[np.ndarray],
        vis_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filter out tracks whose cumulative arc length or maximum pairwise
        span between visible positions falls below the configured minimums
        (self.params.min_arc_length, self.params.min_span).
        """
        # create new dict of static tracks per segment
        for i in range(len(tracks_3d_segments)):
            # Remove tracks whose cumulative arc length is < self.params.min_arc_length cm
            # or whose maximum pairwise distance between any two visible points is < self.params.filtering.min_span cm
            tracks = tracks_3d_segments[i]
            vis = vis_segments[i]
            N = tracks.shape[1]
            arc_lengths = np.zeros(N)
            max_spans = np.zeros(N)
            for n in range(N):
                visible_pos = tracks[vis[:, n].astype(bool), n, :]  # (V, 3)
                if len(visible_pos) >= 2:
                    steps = np.linalg.norm(np.diff(visible_pos, axis=0), axis=1)
                    arc_lengths[n] = np.sum(steps)
                    # max pairwise distance via broadcasting (O(V²), V is small)
                    diffs = visible_pos[:, None, :] - visible_pos[None, :, :]  # (V, V, 3)
                    max_spans[n] = np.sqrt(np.max(np.sum(diffs**2, axis=-1)))
            long_enough = (arc_lengths >= self.params.min_arc_length) & (max_spans > self.params.min_span)
            loguru.logger.info(
                f"Segment {i}: {np.sum(long_enough)} tracks pass length filter, "
                f"{np.sum(~long_enough)} removed "
                f"(arc < 30 cm: {np.sum(arc_lengths < self.params.min_arc_length)}, span <= 10 cm: {np.sum(max_spans <= self.params.min_span)})"
            )
            tracks_3d_segments[i] = tracks[:, long_enough]
            vis_segments[i] = vis[:, long_enough]

        return tracks_3d_segments, vis_segments

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

    def median_filter(
        self,
        pred_3d_tracker: np.ndarray,
        pred_visibility: np.ndarray,
        variance_type: str = "2d",
        percentile: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a median filter to remove static 3D points.
        """
        if variance_type == "2d":
            var = self.calculate_variance(pred_3d_tracker[:, :, :2], pred_visibility)
        elif variance_type == "3d":
            var = self.calculate_variance(pred_3d_tracker, pred_visibility)
        else:
            raise ValueError("Invalid variance_type. Choose either '2d' or '3d'.")
        points_motion = np.linalg.norm(var, axis=1)
        static_points = points_motion < np.percentile(points_motion, percentile)
        loguru.logger.info(f"Static points: {np.sum(static_points)}; Dynamic points: {np.sum(~static_points)}")
        pred_3d_tracker_filtered = pred_3d_tracker[:, ~static_points]
        pred_visibility_filtered = pred_visibility[:, ~static_points]

        static_3d_tracker_filtered = pred_3d_tracker[:, static_points]
        static_visibility_filtered = pred_visibility[:, static_points]
        return pred_3d_tracker_filtered, pred_visibility_filtered, static_3d_tracker_filtered, static_visibility_filtered

    @staticmethod
    def filter_dist_cam(
        cam_poses: Dict[int, np.ndarray],
        tracks_3d_segments: List[np.ndarray],
        vis_segments: List[np.ndarray],
        segments: List[Tuple[int, int]],
        max_dist_cam: float = 1.5,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filter out tracks whose mean 3D position lies farther than `max_dist_cam`
        metres from the mean camera position over the segment's frames.
        """
        # create new dict of static tracks per segment
        for i in range(len(tracks_3d_segments)):
            # Remove tracks whose mean 3D position is more than 1.5 m from the
            # mean camera position over the segment frames.
            seg_start, seg_end = segments[i]
            cam_positions = np.stack([np.array(cam_poses[f])[:3, 3] for f in range(seg_start, seg_end + 1)])  # (F, 3)
            mean_cam_pos = cam_positions.mean(axis=0)  # (3,)

            tracks = tracks_3d_segments[i]
            vis = vis_segments[i]
            N = tracks.shape[1]
            mean_track_pos = np.zeros((N, 3))
            for n in range(N):
                visible_pos = tracks[vis[:, n].astype(bool), n, :]
                if len(visible_pos) > 0:
                    mean_track_pos[n] = visible_pos.mean(axis=0)
            dist_to_cam = np.linalg.norm(mean_track_pos - mean_cam_pos, axis=1)
            close_enough = dist_to_cam <= max_dist_cam
            loguru.logger.info(f"Segment {i}: {np.sum(~close_enough)} tracks removed (mean pos > {max_dist_cam} m from mean camera pos)")
            tracks_3d_segments[i] = tracks[:, close_enough]
            vis_segments[i] = vis[:, close_enough]

        return tracks_3d_segments, vis_segments

    @staticmethod
    def filter_arc_fit(tracks_segments: list, vis_segments: list, min_visible: int = 6, res_thresh: float = 0.03) -> tuple:
        """
        Filter tracks by how well their visible positions fit a circular arc.

        For each track the best-fit plane is found via SVD, positions are
        projected into that plane, and an algebraic circle is fitted.  The
        mean absolute radial residual (in metres) quantifies drift; tracks
        exceeding residual_threshold are discarded.  Tracks with fewer than
        min_visible visible frames are also discarded.

        Args:
            tracks_segments:    list of (T, N, 3) arrays, one per segment.
            vis_segments:       list of (T, N) bool/float arrays, one per segment.
            residual_threshold: maximum allowed mean radial deviation in metres.
            min_visible:        minimum visible frames required to score a track.
            res_thresh:         maximum allowed mean radial deviation in metres.

        Returns:
            Filtered (tracks_segments, vis_segments).
        """
        for seg_idx in range(len(tracks_segments)):
            tracks = tracks_segments[seg_idx]  # (T, N, 3)
            vis = vis_segments[seg_idx]  # (T, N)
            N = tracks.shape[1]

            vis_bool = vis.astype(bool)
            arc_residuals = np.full(N, np.nan)

            for n in range(N):
                pts = tracks[vis_bool[:, n], n, :]  # (V, 3)
                if len(pts) < min_visible:
                    continue
                pts_c = pts - pts.mean(axis=0)

                _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
                Q = pts_c @ Vt[:2].T  # (V, 2) in-plane projection

                A = np.column_stack([2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))])
                b = Q[:, 0] ** 2 + Q[:, 1] ** 2
                res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = res[0], res[1]
                r = np.sqrt(max(res[2] + cx**2 + cy**2, 0.0))
                radii = np.linalg.norm(Q - np.array([cx, cy]), axis=1)
                arc_residuals[n] = np.mean(np.abs(radii - r))

            # discard unscored tracks (< min_visible frames) and tracks above threshold
            keep = np.isfinite(arc_residuals) & (arc_residuals <= res_thresh)
            tracks_segments[seg_idx] = tracks[:, keep]
            vis_segments[seg_idx] = vis[:, keep]
            loguru.logger.info(f"Segment {seg_idx}: arc-fit filter keeps {keep.sum()} / {N} tracks (residual ≤ {res_thresh:.4f} m).")

        return tracks_segments, vis_segments

    @staticmethod
    def filter_dbscan(
        tracks_segments: List[np.ndarray],
        vis_segments: List[np.ndarray],
        eps: float = 30,
        metric: str = 'rigidity',
        criterion: str = 'spatial_extent',
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        For each segment, keep only the dominant rigid-motion cluster of tracks
        found via DBSCAN over a rigidity distance matrix (`find_uniform_trajectory_set`),
        dropping tracks that don't move consistently with the majority.
        """

        def _find_uniform(args):
            """Run `find_uniform_trajectory_set` for one segment (seg_idx, tracks, vis),
            skipping segments with fewer than 5 trajectories and falling back to the
            original tracks/vis on error."""
            seg_idx, tracks, vis = args
            if tracks.shape[1] < 5:
                loguru.logger.info(f"Segment {seg_idx}: Only {tracks.shape[1]} trajectories, skipping uniformity filtering.")
                return seg_idx, tracks, vis
            try:
                uniform_tracks, uniform_vis, _, _ = find_uniform_trajectory_set(
                    tracks,
                    vis,
                    method='dbscan',
                    distance_metric=metric,
                    rigidity_normalize=False,
                    eps_percentile=eps,
                    min_samples=max(10, int(np.ceil(0.05 * tracks.shape[1]))),
                    min_cluster_ratio=0.1,
                    selection_criteria=criterion,
                )
                return seg_idx, uniform_tracks, uniform_vis
            except Exception as e:
                loguru.logger.error(f"Error occurred while processing segment {seg_idx}: {e}")
                return seg_idx, tracks, vis

        results = [_find_uniform((seg_idx, tracks_segments[seg_idx], vis_segments[seg_idx])) for seg_idx in range(len(tracks_segments))]

        out_tracks = list(tracks_segments)
        out_vis = list(vis_segments)
        for seg_idx, uniform_tracks, uniform_vis in results:
            loguru.logger.info(
                f"Segment {seg_idx}: Found {uniform_tracks.shape[1]} uniform trajectories "
                f"out of {tracks_segments[seg_idx].shape[1]} total trajectories."
            )
            out_tracks[seg_idx] = uniform_tracks
            out_vis[seg_idx] = uniform_vis
        return out_tracks, out_vis

    def filter_occluded_tracks(
        self,
        tracks_3d_segments: List[np.ndarray],
        vis_segments: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filter out tracks that are occluded for more than a specified percentage of frames.
        """
        occlusion_threshold = self.params.occlusion_threshold
        filtered_tracks_segments = []
        filtered_visibility_segments = []
        for i in range(len(tracks_3d_segments)):
            pred_3d_tracker = tracks_3d_segments[i]
            pred_visibility = vis_segments[i]

            num_frames = pred_visibility.shape[0]
            visibility_percentage = np.sum(pred_visibility, axis=0) / num_frames
            visible_enough = visibility_percentage >= (1.0 - occlusion_threshold)

            loguru.logger.info(
                f"Keeping {np.sum(visible_enough)} tracks with occlusion rate below {occlusion_threshold * 100:.1f}%, "
                f"removing {np.sum(~visible_enough)} tracks"
            )

            filtered_tracks_segments.append(pred_3d_tracker[:, visible_enough])
            filtered_visibility_segments.append(pred_visibility[:, visible_enough])

        return filtered_tracks_segments, filtered_visibility_segments

    @staticmethod
    def filter_non_smooth_tracks(pred_3d_tracker: np.ndarray, pred_visibility: np.ndarray, thrsh: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter tracks with significant motion change.
        """
        change = np.zeros(pred_3d_tracker.shape[1])
        for i in range(pred_3d_tracker.shape[1]):
            visible_points = pred_3d_tracker[pred_visibility[:, i], i]
            change[i] = np.linalg.norm(np.abs(np.diff(visible_points, axis=0)), axis=0).mean() if len(visible_points) > 1 else 0
        significant_change = change > thrsh
        loguru.logger.info(f"Points with significant change: {np.sum(significant_change)}")
        pred_3d_tracker_filtered = pred_3d_tracker[:, ~significant_change]
        pred_visibility_filtered = pred_visibility[:, ~significant_change]
        return (
            pred_3d_tracker_filtered,
            pred_visibility_filtered,
        )


def filter_tracks_by_twist(
    cfg,
    segments,
    segment_articulations,
    pred_3d_tracks_filtered,
    pred_vis_filtered,
    estimation_tracks,
    estimation_vis,
):
    """
    Enlarge the set of points that describe each articulation by checking all
    filtered 3D point trajectories against the estimated twist.

    The small uniform trajectory subset used to estimate the twist may not cover
    the full spatial extent of the articulated part.  This method tests every
    track in `pred_3d_tracks_filtered` and keeps those whose motion is
    consistent with the twist, subject to three constraints:

    1. **Twist consistency** – the back-projected reference position propagated
       through ``Exp(twist * theta)`` must stay within `inlier_threshold` metres
       of the observed position (mean over all visible known-theta frames).
    2. **Proximity** – the track's mean 3D position must lie within
       `proximity_threshold` metres of the nearest estimation-track mean
       position, preventing spatially unrelated movers from being accepted.
    3. **Minimum travel** – the track must exhibit at least `min_travel` metres
       of displacement (max distance from first visible position across all
       visible frames), ensuring it actually participates in the articulation.

    Algorithm per track:
    1. Reject if the track travels less than ``min_travel`` metres.
    2. Reject if the track's mean position is farther than ``proximity_threshold``
       from every estimation track.
    3. Find the first visible frame among frames with a known theta estimate.
    4. Back-transform to the zero-theta reference:
       ``p_0 = Exp(-twist * theta_ref) @ p(t_ref)``
    5. Forward-predict at every other visible known-theta frame; accept when
       the mean residual < ``inlier_threshold``.

    :param cfg: full config (uses cfg.assoc.twist_val.*)
    :param segments: list of (start_idx, end_idx) tuples
    :param segment_articulations: list of ArticulationModel (with twist/thetas/pairs_t)
    :param pred_3d_tracks_filtered: list of (T, N, 3) float arrays, one per segment
    :param pred_vis_filtered: list of (T, N) bool arrays, one per segment
    :param estimation_tracks: list of (T, M, 3) arrays – the uniform subset used
                              to estimate the twist (i.e. forw_3d_tracks_segments
                              after find_uniform_trajectory_set)
    :param estimation_vis: list of (T, M) bool arrays matching estimation_tracks
    :return: tuple (inlier_masks, inlier_tracks, inlier_vis) where each element
             is a list with one entry per segment:
             - inlier_masks:  (N,) bool array
             - inlier_tracks: (T, N_inlier, 3) array
             - inlier_vis:    (T, N_inlier) bool array
    """

    def _check_one_array(tracks, vis, T_len, N, known_rel_frames, twist_mat, est_means, seg_idx):
        """
        Core per-(tracks, vis) inlier check.  Works for both standard bool visibility
        and reconstruct float visibility that may contain NaN (NaN is treated as not
        visible: numpy comparisons with NaN return False, so ``vis[f, n] > 0`` is
        safely False for NaN entries).
        """
        residuals = np.full(N, np.inf)
        travel_dist = np.zeros(N)
        prox_dist = np.full(N, np.inf)

        for n in range(N):
            # Use "> 0" so NaN evaluates to False rather than truthy.
            vis_frames = vis[:, n] > 0
            if not vis_frames.any():
                continue

            visible_positions = tracks[vis_frames, n, :]  # (V, 3)

            # --- Constraint: minimum travel ---
            first_pos = visible_positions[0]
            travel_dist[n] = float(np.linalg.norm(visible_positions - first_pos, axis=1).max())

            # --- Constraint: proximity to estimation tracks ---
            if est_means.shape[0] > 0:
                mean_pos = visible_positions.mean(axis=0)
                prox_dist[n] = float(np.linalg.norm(est_means - mean_pos, axis=1).min())

            # --- Constraint: twist consistency ---
            ref_frame = None
            for f in known_rel_frames:
                if f < T_len and vis[f, n] > 0:
                    ref_frame = f
                    break
            if ref_frame is None:
                continue

            p_ref = tracks[ref_frame, n]
            T_ref_inv = np.linalg.inv(twist_mat[ref_frame])
            p_0 = (T_ref_inv @ np.append(p_ref, 1.0))[:3]

            errors = []
            for f in known_rel_frames:
                if f == ref_frame or f >= T_len or not (vis[f, n] > 0):
                    continue
                p_obs = tracks[f, n]
                p_pred = (twist_mat[f] @ np.append(p_0, 1.0))[:3]
                errors.append(np.linalg.norm(p_pred - p_obs))

            if errors:
                residuals[n] = float(np.mean(errors))

        inlier_mask = (
            (residuals < cfg.assoc.twist_val.inlier_thresh)
            & (prox_dist < cfg.assoc.twist_val.prox_thresh)
            & (travel_dist >= cfg.assoc.twist_val.min_travel)
        )

        finite_res = residuals[residuals < np.inf]
        median_res = float(np.median(finite_res)) if finite_res.size > 0 else float("nan")
        loguru.logger.info(
            f"Segment {seg_idx}: {inlier_mask.sum()} / {N} tracks are twist-consistent "
            f"(twist_thresh={cfg.assoc.twist_val.inlier_thresh:.3f} m, "
            f"proximity_thresh={cfg.assoc.twist_val.prox_thresh:.2f} m, "
            f"min_travel={cfg.assoc.twist_val.min_travel:.2f} m, "
            f"median residual={median_res:.4f} m | "
            f"travel_fail={int((travel_dist < cfg.assoc.twist_val.min_travel).sum())}, "
            f"prox_fail={int((prox_dist >= cfg.assoc.twist_val.prox_thresh).sum())})"
        )
        return inlier_mask, tracks[:, inlier_mask, :], vis[:, inlier_mask]

    inlier_masks = []
    inlier_tracks = []
    inlier_vis = []

    for i, (start_idx, end_idx) in enumerate(segments):
        model = segment_articulations[i]

        # Detect reconstruct format: segment entry is a list of pass arrays.
        segment_entry = pred_3d_tracks_filtered[i]
        is_recon = isinstance(segment_entry, list)

        # Determine T_len and N from the first (or only) array.
        ref_arr = segment_entry[0] if is_recon else segment_entry
        T_len, N = ref_arr.shape[0], ref_arr.shape[1]
        empty_mask = np.zeros(N, dtype=bool)

        if model.position is None or model.axis is None or model.twist is None:
            loguru.logger.warning(f"Segment {i} has no valid articulation model – skipping twist inlier check.")
            if is_recon:
                inlier_masks.append([empty_mask] * len(segment_entry))
                inlier_tracks.append([a[:, empty_mask, :] for a in segment_entry])
                inlier_vis.append([v[:, empty_mask] for v in pred_vis_filtered[i]])
            else:
                inlier_masks.append(empty_mask)
                inlier_tracks.append(ref_arr[:, empty_mask, :])
                inlier_vis.append(pred_vis_filtered[i][:, empty_mask])
            continue

        twist = np.array(model.twist)
        rel_frame_indices = [int(f) for f in list(model.pairs_t.values())[0]]
        thetas = model.thetas

        if len(rel_frame_indices) < 2:
            loguru.logger.warning(f"Segment {i} has fewer than 2 theta estimates – skipping twist inlier check.")
            if is_recon:
                inlier_masks.append([empty_mask] * len(segment_entry))
                inlier_tracks.append([a[:, empty_mask, :] for a in segment_entry])
                inlier_vis.append([v[:, empty_mask] for v in pred_vis_filtered[i]])
            else:
                inlier_masks.append(empty_mask)
                inlier_tracks.append(ref_arr[:, empty_mask, :])
                inlier_vis.append(pred_vis_filtered[i][:, empty_mask])
            continue

        # Pre-compute Exp(twist * theta) for every known frame.
        frame_to_theta = {f: th for f, th in zip(rel_frame_indices, thetas)}
        known_rel_frames = sorted(frame_to_theta.keys())
        twist_mat = {f: gtsam.Pose3.Expmap(twist * frame_to_theta[f]).matrix() for f in known_rel_frames}

        # Build point cloud of estimation-track mean positions for proximity check.
        est_tracks_i = estimation_tracks[i]  # (T, M, 3)
        est_vis_i = estimation_vis[i]  # (T, M)
        M = est_tracks_i.shape[1]
        est_means = np.full((M, 3), np.nan)
        for m in range(M):
            vis_frames_m = est_vis_i[:, m] > 0
            if vis_frames_m.any():
                est_means[m] = est_tracks_i[vis_frames_m, m, :].mean(axis=0)
        est_means = est_means[~np.isnan(est_means).any(axis=1)]

        if is_recon:
            # Process each keyframe pass independently.
            pass_masks, pass_tracks, pass_vis_out = [], [], []
            for pass_arr, pass_v in zip(segment_entry, pred_vis_filtered[i]):
                mask, t_out, v_out = _check_one_array(
                    pass_arr,
                    pass_v,
                    T_len,
                    N,
                    known_rel_frames,
                    twist_mat,
                    est_means,
                    i,
                )
                pass_masks.append(mask)
                pass_tracks.append(t_out)
                pass_vis_out.append(v_out)
            inlier_masks.append(pass_masks)
            inlier_tracks.append(pass_tracks)
            inlier_vis.append(pass_vis_out)
        else:
            mask, t_out, v_out = _check_one_array(
                segment_entry,
                pred_vis_filtered[i],
                T_len,
                N,
                known_rel_frames,
                twist_mat,
                est_means,
                i,
            )
            inlier_masks.append(mask)
            inlier_tracks.append(t_out)
            inlier_vis.append(v_out)

    return inlier_masks, inlier_tracks, inlier_vis


def compute_trajectory_distance_matrix(
    tracks_3d: np.ndarray,
    visibility: np.ndarray,
    method: str = 'dtw',
    normalize: bool = True,
    rigidity_normalize: bool = True,
) -> np.ndarray:
    """
    TODO: remove dtw, hausdorff, frechet before release
    Compute pairwise distance matrix between trajectories.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)
        method: Distance metric ('dtw', 'hausdorff', 'frechet', 'euclidean',
            'motion_direction', 'rigidity')
        normalize: Whether to normalize by trajectory length (DTW only)
        rigidity_normalize: Whether to normalize rigidity distances by mean
            inter-point distance (scale-invariant). Set False for an absolute
            eps threshold in track coordinate units.

    Returns:
        Distance matrix of shape (N, N)
    """
    # Vectorized / dedicated paths — no O(N²) loop needed
    if method == 'motion_direction':
        return motion_direction_distance_matrix(tracks_3d, visibility)
    elif method == 'rigidity':
        return rigidity_distance_matrix(tracks_3d, visibility, normalize=rigidity_normalize)
    else:
        N = tracks_3d.shape[1]
        dist_matrix = np.zeros((N, N))
        vis_bool = visibility.astype(bool)

        for i in range(N):
            for j in range(i + 1, N):
                if method == 'dtw':
                    dist = dtw_distance(tracks_3d[vis_bool[:, i], i, :], tracks_3d[vis_bool[:, j], j, :], normalize=normalize)
                elif method == 'hausdorff':
                    dist = hausdorff_distance(tracks_3d[vis_bool[:, i], i, :], tracks_3d[vis_bool[:, j], j, :])
                elif method == 'frechet':
                    dist = frechet_distance(tracks_3d[vis_bool[:, i], i, :], tracks_3d[vis_bool[:, j], j, :])
                else:  # euclidean (average point-wise distance)
                    dist = euclidean_trajectory_distance(tracks_3d[vis_bool[:, i], i, :], tracks_3d[vis_bool[:, j], j, :])

                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix


def dtw_distance(traj1: np.ndarray, traj2: np.ndarray, normalize: bool = True) -> float:
    """
    Dynamic Time Warping distance between two 3D trajectories.
    Handles trajectories of different lengths.
    """
    n, m = len(traj1), len(traj2)

    if n == 0 or m == 0:
        return np.inf

    # Initialize DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(traj1[i - 1] - traj2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    distance = dtw[n, m]

    if normalize:
        distance /= n + m

    return distance


def hausdorff_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Hausdorff distance between two 3D trajectories.
    """
    if len(traj1) == 0 or len(traj2) == 0:
        return np.inf

    # Compute pairwise distances
    distances = cdist(traj1, traj2)

    # Hausdorff distance
    h1 = np.max(np.min(distances, axis=1))
    h2 = np.max(np.min(distances, axis=0))

    return max(h1, h2)


def frechet_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    TODO: delete before release
    Discrete Fréchet distance between two 3D trajectories.
    """
    n, m = len(traj1), len(traj2)

    if n == 0 or m == 0:
        return np.inf

    # Initialize distance matrix
    ca = np.full((n, m), -1.0)

    def c(i: int, j: int) -> float:
        """Memoized recursive coupling distance for discrete Fréchet distance
        between traj1[:i+1] and traj2[:j+1]."""
        if ca[i, j] > -1:
            return ca[i, j]

        d = np.linalg.norm(traj1[i] - traj2[j])

        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        else:
            ca[i, j] = np.inf

        return ca[i, j]

    return c(n - 1, m - 1)


def euclidean_trajectory_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Average Euclidean distance between synchronized trajectory points.
    """
    if len(traj1) == 0 or len(traj2) == 0:
        return np.inf

    # Resample to same length
    min_len = min(len(traj1), len(traj2))
    idx1 = np.linspace(0, len(traj1) - 1, min_len).astype(int)
    idx2 = np.linspace(0, len(traj2) - 1, min_len).astype(int)

    distances = np.linalg.norm(traj1[idx1] - traj2[idx2], axis=1)
    return np.mean(distances)


def rigidity_distance_matrix(
    tracks_3d: np.ndarray,
    visibility: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Pairwise rigidity distance matrix.

    Two points that belong to the same rigid body maintain a constant
    pairwise distance over time.  The distance between trajectories i and j
    is the standard deviation of ||p_i(t) - p_j(t)|| over all frames where
    both points are visible.

    Distance = 0  → perfectly rigid pair (same rigid body).
    Distance > 0  → pairwise distance varies → points move independently.

    Args:
        tracks_3d:  (T, N, 3) 3-D positions.
        visibility: (T, N) bool/float mask.
        normalize:  If True (default), divide std by mean inter-point distance
                    so the metric is scale-invariant (coefficient of variation).
                    If False, return raw std in track coordinate units, making
                    eps a physically interpretable absolute threshold.

    Returns:
        dist_matrix: (N, N) symmetric rigidity distance matrix.
    """
    _, N, _ = tracks_3d.shape
    vis_bool = visibility.astype(bool)
    dist_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            both_visible = vis_bool[:, i] & vis_bool[:, j]
            if both_visible.sum() < 2:
                dist_matrix[i, j] = dist_matrix[j, i] = np.inf
                continue

            diff = tracks_3d[both_visible, i, :] - tracks_3d[both_visible, j, :]
            pairwise_dist = np.linalg.norm(diff, axis=1)  # (V,)
            rigidity = pairwise_dist.std()
            if normalize:
                rigidity /= pairwise_dist.mean() + 1e-8
            dist_matrix[i, j] = dist_matrix[j, i] = rigidity

    return dist_matrix


def motion_direction_distance_matrix(
    tracks_3d: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """
    Pairwise cosine distance matrix based on mean motion direction.

    For both prismatic and revolute motion the dominant invariant shared by
    co-moving points is their *direction* of motion (for prismatic: identical;
    for revolute: all tangent vectors lie in the same plane perpendicular to
    the rotation axis, so their mean directions correlate strongly).  Raw
    Euclidean distance on positions fails for revolute parts because arc radius
    scales linear speed — this metric is magnitude-invariant.

    Distance = 1 - cosine_similarity ∈ [0, 2].

    Args:
        tracks_3d:  (T, N, 3) 3-D positions.
        visibility: (T, N) bool/float mask.

    Returns:
        dist_matrix: (N, N) symmetric cosine distance matrix.
    """
    _, N, _ = tracks_3d.shape
    vis_bool = visibility.astype(bool)

    # Finite-difference velocities: (T-1, N, 3)
    velocities = np.diff(tracks_3d, axis=0)
    vis_pairs = vis_bool[:-1] & vis_bool[1:]  # (T-1, N)

    # Mean velocity per trajectory, ignoring invisible frames
    mean_vel = np.zeros((N, 3), dtype=np.float64)
    for n in range(N):
        v = velocities[vis_pairs[:, n], n, :]
        if len(v) > 0:
            mean_vel[n] = v.mean(axis=0)

    # Normalize to unit vectors; zero-motion tracks get zero vector
    norms = np.linalg.norm(mean_vel, axis=1, keepdims=True)
    directions = np.where(norms > 1e-8, mean_vel / norms, 0.0)  # (N, 3)

    # Cosine similarity via matrix multiply, then convert to distance
    cosine_sim = directions @ directions.T  # (N, N)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # numerical safety
    dist_matrix = 1.0 - cosine_sim

    # Tracks with no motion are uninformative — set their distance to max
    no_motion = norms[:, 0] <= 1e-8
    dist_matrix[no_motion, :] = 1.0
    dist_matrix[:, no_motion] = 1.0
    np.fill_diagonal(dist_matrix, 0.0)

    return dist_matrix


def cluster_trajectories_dbscan(
    tracks_3d: np.ndarray,
    visibility: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
    distance_method: str = 'dtw',
    rigidity_normalize: bool = True,
    eps_percentile: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Cluster trajectories using DBSCAN with custom distance metric.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)
        eps: Maximum distance between two samples to be considered neighbors.
            Ignored when eps_percentile is set.
        min_samples: Minimum number of samples in a neighborhood to form a cluster
        distance_method: Distance metric to use ('dtw', 'hausdorff', 'frechet',
            'euclidean', 'motion_direction', 'rigidity')
        rigidity_normalize: Passed to rigidity_distance_matrix. Set False to make
            eps an absolute threshold in track coordinate units.
        eps_percentile: If set (0–100), calibrate eps automatically as this
            percentile of the finite off-diagonal distance values. Useful when
            the absolute scale of the metric is unknown. Typical range: 10–30.

    Returns:
        labels: Cluster labels for each trajectory (-1 for outliers)
        info: Dictionary with clustering information
    """
    N = tracks_3d.shape[1]

    print(f"Computing {distance_method.upper()} distance matrix for {N} trajectories...")
    dist_matrix = compute_trajectory_distance_matrix(tracks_3d, visibility, method=distance_method, rigidity_normalize=rigidity_normalize)

    if eps_percentile is not None:
        # Calibrate eps from the distribution of finite off-diagonal distances.
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        finite_vals = dist_matrix[mask & np.isfinite(dist_matrix)]
        if finite_vals.size == 0:
            warnings.warn("All distances are non-finite; falling back to eps={eps}.")
        else:
            eps = float(np.percentile(finite_vals, eps_percentile))
            p10, p25, p50, p75 = np.percentile(finite_vals, [10, 25, 50, 75])
            print(f"  Rigidity dist distribution — p10: {p10:.4f}, p25: {p25:.4f}, p50: {p50:.4f}, p75: {p75:.4f}")
            print(f"  Calibrated eps = p{eps_percentile:.0f} = {eps:.4f}")

    print(f"Clustering with DBSCAN (eps={eps:.4f}, min_samples={min_samples})...")
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clusterer.fit_predict(dist_matrix)

    # Compute statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    cluster_sizes = []
    for cluster_id in range(n_clusters):
        cluster_size = np.sum(labels == cluster_id)
        cluster_sizes.append(cluster_size)

    info = {'n_clusters': n_clusters, 'n_outliers': n_outliers, 'cluster_sizes': cluster_sizes, 'labels': labels, 'distance_matrix': dist_matrix}

    print("\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of outliers: {n_outliers} ({100 * n_outliers / N:.1f}%)")
    if cluster_sizes:
        print(f"  Largest cluster: {max(cluster_sizes)} trajectories ({100 * max(cluster_sizes) / N:.1f}%)")
        print(f"  Cluster sizes: {sorted(cluster_sizes, reverse=True)}")

    if n_clusters == 0 and n_outliers == N:
        warnings.warn(
            "All trajectories are classified as outliers. Consider adjusting DBSCAN parameters (eps, min_samples) or using a different clustering method."
        )

        return np.ones(N, dtype=int), info  # Return all as outliers

    return labels, info


def cluster_trajectories_hierarchical(
    tracks_3d: np.ndarray,
    visibility: np.ndarray,
    n_clusters: int = 5,
    distance_method: str = 'dtw',
    linkage: str = 'average',
    rigidity_normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Cluster trajectories using hierarchical clustering.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)
        n_clusters: Number of clusters to form
        distance_method: Distance metric to use
        linkage: Linkage criterion ('average', 'complete', 'single')
        rigidity_normalize: Passed to rigidity_distance_matrix. Set False to make
            eps an absolute threshold in track coordinate units.

    Returns:
        labels: Cluster labels for each trajectory
        info: Dictionary with clustering information
    """
    N = tracks_3d.shape[1]

    print(f"Computing {distance_method.upper()} distance matrix for {N} trajectories...")
    dist_matrix = compute_trajectory_distance_matrix(tracks_3d, visibility, method=distance_method, rigidity_normalize=rigidity_normalize)

    print(f"Clustering with Hierarchical (n_clusters={n_clusters}, linkage={linkage})...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage)
    labels = clusterer.fit_predict(dist_matrix)

    # Compute statistics
    cluster_sizes = []
    for cluster_id in range(n_clusters):
        cluster_size = np.sum(labels == cluster_id)
        cluster_sizes.append(cluster_size)

    info = {'n_clusters': n_clusters, 'cluster_sizes': cluster_sizes, 'labels': labels, 'distance_matrix': dist_matrix}

    print("\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Largest cluster: {max(cluster_sizes)} trajectories ({100 * max(cluster_sizes) / N:.1f}%)")
    print(f"  Cluster sizes: {sorted(cluster_sizes, reverse=True)}")

    return labels, info


def compute_trajectory_lengths(tracks_3d: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """
    Compute the total 3D path length for each trajectory.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)

    Returns:
        lengths: Array of shape (N,) with total path length for each trajectory
    """
    N = tracks_3d.shape[1]
    vis_bool = visibility.astype(bool)
    lengths = np.zeros(N)

    for i in range(N):
        vis_frames = vis_bool[:, i]
        if vis_frames.sum() < 2:
            continue

        visible_points = tracks_3d[vis_frames, i, :]
        displacements = np.diff(visible_points, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        lengths[i] = distances.sum()

    return lengths


def extract_dominant_cluster(
    tracks_3d: np.ndarray, visibility: np.ndarray, labels: np.ndarray, selection_criteria: str = 'longest'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the dominant cluster(s) based on selection criteria.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)
        labels: Cluster labels from clustering
        selection_criteria: How to select dominant cluster:
            'longest' - cluster with longest average trajectory length
            'largest' - cluster with most trajectories
            'longest_total' - cluster with highest total distance traveled
            'spatial_extent' - cluster covering the largest 3D spatial volume
                (scored by the product of the three std-dev axes of all visible
                positions, i.e. the volume of the bounding ellipsoid)
            'convex_hull_volume' - cluster with the largest convex hull volume
                of all visible positions (m³); directly measures the 3D ROI
                swept by the trajectories

    Returns:
        dominant_tracks: Tracks from dominant cluster(s)
        dominant_visibility: Visibility from dominant cluster(s)
        dominant_indices: Original indices of dominant trajectories
    """
    N = tracks_3d.shape[1]

    # Compute trajectory lengths
    traj_lengths = compute_trajectory_lengths(tracks_3d, visibility)

    # Count cluster sizes (excluding outliers if present)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    cluster_info = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_size = cluster_mask.sum()
        cluster_indices = np.where(cluster_mask)[0]
        cluster_lengths = traj_lengths[cluster_indices]

        avg_length = cluster_lengths.mean()
        total_length = cluster_lengths.sum()
        max_length = cluster_lengths.max()

        vis_bool = visibility.astype(bool)
        cluster_positions = tracks_3d[:, cluster_mask, :][vis_bool[:, cluster_mask]]  # (V_total, 3)

        # Spatial extent: product of per-axis std devs of all visible positions
        # (proportional to the volume of the bounding ellipsoid)
        if len(cluster_positions) >= 2:
            spatial_extent = float(np.prod(np.std(cluster_positions, axis=0)))
        else:
            spatial_extent = 0.0

        # Convex hull volume: volume of the 3D region swept by all visible positions
        if len(cluster_positions) >= 4:
            try:
                convex_hull_volume = ConvexHull(cluster_positions).volume
            except Exception:
                convex_hull_volume = 0.0
        else:
            convex_hull_volume = 0.0

        cluster_info.append(
            {
                'label': label,
                'size': cluster_size,
                'mask': cluster_mask,
                'avg_length': avg_length,
                'total_length': total_length,
                'max_length': max_length,
                'spatial_extent': spatial_extent,
                'convex_hull_volume': convex_hull_volume,
                'ratio': cluster_size / N,
            }
        )

    # Sort by selection criteria
    if selection_criteria == 'longest':
        cluster_info.sort(key=lambda x: x['avg_length'], reverse=True)
    elif selection_criteria == 'longest_total':
        cluster_info.sort(key=lambda x: x['total_length'], reverse=True)
    elif selection_criteria == 'spatial_extent':
        cluster_info.sort(key=lambda x: x['spatial_extent'], reverse=True)
    elif selection_criteria == 'convex_hull_volume':
        cluster_info.sort(key=lambda x: x['convex_hull_volume'], reverse=True)
    else:  # 'largest'
        cluster_info.sort(key=lambda x: x['size'], reverse=True)

    # Select only the top cluster based on criteria
    dominant_mask = np.zeros(N, dtype=bool)
    selected_clusters = []

    if len(cluster_info) > 0:
        # Select only the first (best) cluster
        best_cluster = cluster_info[0]
        dominant_mask |= best_cluster['mask']
        selected_clusters.append(best_cluster)

    # Extract dominant trajectories
    dominant_indices = np.where(dominant_mask)[0]
    dominant_tracks = tracks_3d[:, dominant_mask, :]
    dominant_visibility = visibility[:, dominant_mask]

    print(f"\nAll Clusters (sorted by {selection_criteria}):")
    for i, info in enumerate(cluster_info):
        marker = "***" if i < len(selected_clusters) else "   "
        print(f"  {marker} Cluster {info['label']}: {info['size']} trajectories ({100 * info['ratio']:.1f}%)")
        print(
            f"      Avg length: {info['avg_length']:.3f}m, Total: {info['total_length']:.3f}m, Max: {info['max_length']:.3f}m, "
            f"Spatial extent: {info['spatial_extent']:.6f}m³, Convex hull volume: {info['convex_hull_volume']:.6f}m³"
        )

    print(f"\n  *** = Selected clusters (criteria: {selection_criteria})")
    print(f"  Total selected: {len(selected_clusters)} cluster(s), {dominant_mask.sum()} trajectories ({100 * dominant_mask.sum() / N:.1f}%)")

    # Print stats of selected set
    selected_lengths = traj_lengths[dominant_indices]
    print("\nSelected Set Statistics:")
    print(f"  Mean length: {selected_lengths.mean():.3f}m")
    print(f"  Median length: {np.median(selected_lengths):.3f}m")
    print(f"  Min length: {selected_lengths.min():.3f}m")
    print(f"  Max length: {selected_lengths.max():.3f}m")
    print(f"  Total distance: {selected_lengths.sum():.3f}m")

    return dominant_tracks, dominant_visibility, dominant_indices


def find_uniform_trajectory_set(
    tracks_3d: np.ndarray, visibility: np.ndarray, method: str = 'auto', distance_metric: str = 'dtw', selection_criteria: str = 'longest', **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Main function to find a uniform set of trajectories representing the majority.

    Args:
        tracks_3d: Trajectory data of shape (T, N, 3)
        visibility: Visibility mask of shape (T, N)
        method: Clustering method ('auto', 'dbscan', 'hierarchical')
        distance_metric: Distance metric ('dtw', 'hausdorff', 'frechet', 'euclidean', 'motion_direction', 'rigidity')
        selection_criteria: How to select dominant cluster:
            'longest' - cluster with longest average trajectory length (DEFAULT)
            'largest' - cluster with most trajectories
            'longest_total' - cluster with highest total distance traveled
            'spatial_extent' - cluster covering the largest 3D spatial volume
            'convex_hull_volume' - cluster with the largest convex hull volume of all visible positions
        **kwargs: Additional arguments for clustering methods
            For DBSCAN: eps, min_samples
            For hierarchical: n_clusters, linkage
            rigidity_normalize (bool, default True): whether to normalize the rigidity
                distance by mean inter-point distance. Set False so that eps is an
                absolute threshold in track coordinate units rather than a ratio.
            eps_percentile (float, optional): when set, calibrate eps automatically
                as this percentile of the finite off-diagonal rigidity distances
                (typical range 10–30). Overrides the eps kwarg.

    Returns:
        uniform_tracks: Uniform set of trajectories
        uniform_visibility: Visibility for uniform set
        indices: Original indices
        info: Clustering information
    """
    print("=" * 60)
    print("TRAJECTORY CLUSTERING FOR UNIFORM SET EXTRACTION")
    print(f"Selection criteria: {selection_criteria}")
    print("=" * 60)

    N = tracks_3d.shape[1]

    if method == 'auto':
        # Auto-select method based on number of trajectories
        if N < 50:
            method = 'hierarchical'
            print(f"Auto-selected: Hierarchical clustering (N={N} < 50)")
        else:
            method = 'dbscan'
            print(f"Auto-selected: DBSCAN clustering (N={N} >= 50)")

    # Perform clustering
    rigidity_normalize = kwargs.get('rigidity_normalize', True)
    if method == 'dbscan':
        eps = kwargs.get('eps', 0.05)
        min_samples = kwargs.get('min_samples', max(3, int(N * 0.02)))
        eps_percentile = kwargs.get('eps_percentile', None)
        labels, info = cluster_trajectories_dbscan(
            tracks_3d,
            visibility,
            eps,
            min_samples,
            distance_metric,
            rigidity_normalize=rigidity_normalize,
            eps_percentile=eps_percentile,
        )
    else:  # hierarchical
        n_clusters = kwargs.get('n_clusters', max(3, int(np.sqrt(N))))
        linkage = kwargs.get('linkage', 'average')
        labels, info = cluster_trajectories_hierarchical(
            tracks_3d,
            visibility,
            n_clusters,
            distance_metric,
            linkage,
            rigidity_normalize=rigidity_normalize,
        )

    # Extract dominant cluster based on selection criteria
    uniform_tracks, uniform_visibility, indices = extract_dominant_cluster(tracks_3d, visibility, labels, selection_criteria)

    info['dominant_indices'] = indices
    info['selection_criteria'] = selection_criteria

    print("=" * 60)

    return uniform_tracks, uniform_visibility, indices, info


def smooth_trajectory_optimization(points, visibility, lambda_vel=0.1, lambda_jerk=1.0):
    """
    Smooths 3D point trajectories using optimization, minimizing a cost function
    combining data fidelity, velocity penalty, and jerk penalty.

    Args:
        points (np.ndarray): A NumPy array of shape (num_frames, num_points, 3)
                            containing the observed 3D coordinates.
        visibility (np.ndarray): A NumPy array of shape (num_frames, num_points)
                                containing boolean or binary (1/0) visibility flags.
                                True or 1 means the point is visible.
        lambda_vel (float): Weight for the velocity regularization term (1st order difference).
                            Controls smoothing based on velocity changes.
        lambda_jerk (float): Weight for the jerk regularization term (3rd order difference).
                            Controls smoothing based on acceleration changes.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, num_points, 3)
                    containing the smoothed 3D coordinates.
                    Returns None if inputs are invalid (e.g., insufficient frames for jerk).
    """
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("`points` must be a 3D array with shape (num_frames, num_points, 3)")
    if visibility.ndim != 2 or visibility.shape != points.shape[:2]:
        raise ValueError("`visibility` must be a 2D array with shape (num_frames, num_points)")
    if points.shape[0] < 1 or points.shape[1] < 1:
        print("Warning: Empty points array provided.")
        return points.copy()  # Return empty/original if no data

    num_frames, num_points, _ = points.shape

    # Jerk requires at least 4 frames
    if lambda_jerk > 0 and num_frames < 4:
        print(f"Warning: num_frames ({num_frames}) < 4. Cannot compute jerk penalty. Setting lambda_jerk to 0 for this run.")
        lambda_jerk = 0.0
    # Velocity requires at least 2 frames
    if lambda_vel > 0 and num_frames < 2:
        print(f"Warning: num_frames ({num_frames}) < 2. Cannot compute velocity penalty. Setting lambda_vel to 0 for this run.")
        lambda_vel = 0.0

    # Flatten the data: Reshape points and visibility so that all frames for point 0
    # come first, then all frames for point 1, etc. (Point-major order)
    # New shape: (N * T, 3) for points, (N * T,) for visibility
    points_flat = np.transpose(points, (1, 0, 2)).reshape(-1, 3)
    visibility_flat = np.transpose(visibility).flatten().astype(float)  # Ensure float for V matrix

    N = num_points
    T = num_frames
    NT = N * T  # Total number of variables per coordinate (X, Y, or Z)

    # 1. Build the Data Fidelity Matrix (V)
    # Diagonal matrix with visibility flags (1 where visible, 0 where not)
    V = sp.diags(visibility_flat, format="csc")

    # 2. Build the Smoothness Penalty Matrices (L_vel, L_jerk)
    I_N = sp.identity(N, format="csc")  # Identity matrix for points

    L_vel = sp.csc_matrix((NT, NT))  # Initialize as empty sparse matrix
    if lambda_vel > 0 and T >= 2:
        D1_single = build_difference_matrix(T, 1)  # Shape (T-1, T)
        # L1 = D1^T * D1 gives the squared velocity penalty matrix for a single trajectory
        L1_single = D1_single.T @ D1_single  # Shape (T, T)
        L_vel = sp.kron(I_N, L1_single, format="csc")  # Apply to all points

    L_jerk = sp.csc_matrix((NT, NT))  # Initialize as empty sparse matrix
    if lambda_jerk > 0 and T >= 4:
        D3_single = build_difference_matrix(T, 3)  # Shape (T-3, T)
        # L3 = D3^T * D3 gives the squared jerk penalty matrix for a single trajectory
        L3_single = D3_single.T @ D3_single  # Shape (T, T)
        L_jerk = sp.kron(I_N, L3_single, format="csc")  # Apply to all points

    # 3. Construct the overall system matrix A
    # A = V + lambda_vel * L_vel + lambda_jerk * L_jerk
    A = V + lambda_vel * L_vel + lambda_jerk * L_jerk
    # Ensure A is in CSC format for efficient solving with spsolve
    A = A.tocsc()

    # Add small regularization to diagonal for numerical stability if needed,
    # especially if a point is never visible AND lambdas are zero.
    # A += 1e-9 * sp.identity(NT, format='csc') # Optional small regularization

    # 4. Solve the linear system A * p'_smooth = V * p_orig for each dimension (X, Y, Z)
    smoothed_points_flat = np.zeros_like(points_flat)

    for dim in range(3):
        p_orig_flat = points_flat[:, dim]
        # Construct the right-hand side: b = V * p_orig
        # Only considers visible points for the data term target
        b = V @ p_orig_flat

        # Solve the sparse linear system
        try:
            p_smooth_flat = spsolve(A, b)
            smoothed_points_flat[:, dim] = p_smooth_flat
        except Exception as e:
            print(f"Error solving linear system for dimension {dim}: {e}")
            print("Returning original points for this dimension.")
            smoothed_points_flat[:, dim] = p_orig_flat

    # 5. Reshape the smoothed points back to the original format (T, N, 3)
    smoothed_points = smoothed_points_flat.reshape(N, T, 3)
    smoothed_points = np.transpose(smoothed_points, (1, 0, 2))  # Back to (T, N, 3)

    return smoothed_points


def build_difference_matrix(size, order):
    """
    Builds a sparse difference matrix of a given order for a single sequence.

    Args:
        size (int): The length of the sequence (T).
        order (int): The order of the difference (1 for velocity, 2 for acceleration, 3 for jerk).

    Returns:
        scipy.sparse.csr_matrix: The difference matrix.
                                 Shape depends on order (e.g., (size-order) x size).
    """
    if order == 1:  # Velocity: p[t] - p[t-1]
        diagonals = [-np.ones(size), np.ones(size)]
        offsets = [0, 1]
        # Matrix maps p' to velocities. Shape (T-1) x T
        D = sp.diags(diagonals, offsets, shape=(size - 1, size), format="csr")
        # We remove the last row which computes p[T]-p[T-1] and depends on p[T] which doesn't exist
        # Correct matrix should have -1 at (i,i) and 1 at (i, i+1)
        diagonals = [-np.ones(size - 1), np.ones(size - 1)]
        offsets = [0, 1]
        D = sp.diags(diagonals, offsets, shape=(size - 1, size), format="csr")

    elif order == 2:  # Acceleration: p[t+1] - 2p[t] + p[t-1]
        diagonals = [np.ones(size), -2 * np.ones(size), np.ones(size)]
        offsets = [0, 1, 2]
        # Matrix maps p' to accelerations. Shape (T-2) x T
        D = sp.diags(diagonals, offsets, shape=(size - 2, size), format="csr")

    elif order == 3:  # Jerk: p[t+2] - 3p[t+1] + 3p[t] - p[t-1]
        # Ensure sufficient length
        if size < 4:
            # Cannot compute 3rd order difference for sequence < 4
            return sp.csr_matrix((0, size))
        diagonals = [
            -np.ones(size),
            3 * np.ones(size),
            -3 * np.ones(size),
            np.ones(size),
        ]
        offsets = [0, 1, 2, 3]
        # Matrix maps p' to jerks. Shape (T-3) x T
        D = sp.diags(diagonals, offsets, shape=(size - 3, size), format="csr")

    else:
        raise ValueError("Order must be 1, 2, or 3")

    return D
