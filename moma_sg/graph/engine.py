import copy
import gzip
import json
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import sys
from typing import Any, List, Tuple, Union

import cv2
import gtsam
import hydra
import loguru
from moma_sg.articulation.filtering import TrajFiltering, filter_tracks_by_twist
from moma_sg.articulation.interaction import (
    compute_depth_disparity,
    compute_egocentric_scores,
    compute_exocentric_scores,
    filter_interaction_signals,
    parse_segments,
    plot_interaction_segmentation,
    run_interaction_model,
    save_interaction_segments,
)
from moma_sg.articulation.point_estimator import (
    ARTICULATION_MODE,
    LAST_INFERRED_STATE_MAP,
    MAX_OPENING_MAP,
    InferencePointAxis,
    compute_type_prior,
    estimate_articulation_model,
    estimate_dense_thetas,
    estimate_motion_mode,
    identify_cloud_pairs_from_multiple_starts,
    parse_response,
    plot_dense_theta_estimate,
    plot_estimate,
    prepare_estimation_data,
)
from moma_sg.data.hsr_dataloader import HSRDataset
from moma_sg.data.kinect_dataloader import KinectRGBDDataset
from moma_sg.graph.mapping import extract_static_keyframes, load_conceptgraphs_map, produce_map
from moma_sg.graph.segmentation import Segmentation
from moma_sg.graph.tracking import Tracking
from moma_sg.utils.caption import plot_articulation_mode, query_articulation_mode
from moma_sg.utils.mapping import (
    Hierarchy,
    MapObjectList,
    compute_clip_features_batched,
    create_object_pcd,
    get_bounding_box,
    iou_aabb,
    parse_masks,
    prepare_semsam_image,
    process_pcd,
    solve_assignment,
)
from moma_sg.utils.visualization import plot_3d_tracks, plot_axis_img, to_bgr
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
from scipy import ndimage
from scipy.spatial import ConvexHull
import torch
from tqdm import tqdm


class MoMaSG:
    """Top-level class for the MoMaSG pipeline: loads an RGB-D dataset, then
    drives interaction-prior extraction, keyframe selection, 3D object mapping,
    keypoint tracking, articulation estimation, and track-to-object association
    to build a hierarchical articulated 3D scene graph."""

    def __init__(self, cfg: DictConfig):
        """
        Build the dataset, deep models (if enabled), and load the RGB-D sequence into memory.

        Args:
            cfg: Full Hydra/OmegaConf config (dataset, engine, model, and pipeline settings).
        """
        self.cfg = cfg
        self.dataset = self.init_dataset()

        self.init_models() if self.cfg.engine and self.dataset else None  # initialize all deep models only if necessary
        self.rgb_frames, self.depth_frames, self.cam_poses, self.scene_mesh = (
            self.load_seq() if self.dataset else None
        )  # load data if dataset is available

    def init_models(self):
        """Instantiate all deep-learning components used by the pipeline: MobileSAM/YOLO
        segmentation, CLIP, Semantic-SAM, the TapNext++-based tracker, and trajectory filtering."""
        # Initialize MobileSAM model
        self.seg = Segmentation(self.cfg)

        # Initialize the CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.clip_model = self.clip_model.to("cuda" if self.cfg.device.use_cuda else "cpu")
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Initialize Semantic-SAM
        sys.path.append(self.cfg.mapping.semsam_path)
        sys.path.append(os.path.join(self.cfg.mapping.semsam_path, "utils"))
        from semantic_sam import SemanticSamAutomaticMaskGenerator, build_semantic_sam

        self.semsam_mask_generator = SemanticSamAutomaticMaskGenerator(
            build_semantic_sam(model_type='T', ckpt=os.path.join(self.cfg.mapping.semsam_path, 'ckpt/swint_only_sam_many2many.pth')), level=[6]
        )  # model_type: 'L' / 'T', depends on your checkpint

        # Initialize TapNext++
        self.track = Tracking(
            {
                "model_path": os.path.join(self.cfg.package_path, self.cfg.tracking.tap_path),
                "dataset": self.dataset,
                "device": "cuda" if self.cfg.device.use_cuda else "cpu",
                "width": self.cfg.dataset.width,
                "height": self.cfg.dataset.height,
                "feat_type": self.cfg.tracking.get("feat_type", "shi"),
            }
        )

        # Initialize trajectory filtering module
        self.traj_filter = TrajFiltering(self.cfg, bidirectional=self.cfg.tracking.bidir)

    def init_dataset(self) -> Union[KinectRGBDDataset, HSRDataset]:
        """Initialize and return an Azure RGB-D dataset."""
        if self.cfg.dataset.camera == "kinect":
            cfg = {
                "root_dir": "data",
                "transforms": None,
                "depth_min": self.cfg.dataset.depth_min,
                "depth_max": self.cfg.dataset.depth_max,
                "root_path": Path(self.cfg.dataset.root_path),
                "tf_file_path": os.path.join(self.cfg.package_path, self.cfg.dataset.tf_file_path),
                "flipped": self.cfg.dataset.flipped,
                "gt_poses": self.cfg.dataset.gt_poses,
                "droid_slam": self.cfg.dataset.droid_slam,
            }
            return KinectRGBDDataset(cfg)
        elif self.cfg.dataset.camera == "hsr":
            cfg = {
                "root_dir": "data",
                "transforms": None,
                "depth_min": self.cfg.dataset.depth_min,
                "depth_max": self.cfg.dataset.depth_max,
                "root_path": Path(self.cfg.dataset.root_path),
                "tf_file_path": os.path.join(self.cfg.package_path, self.cfg.dataset.tf_file_path),
                "flipped": self.cfg.dataset.flipped,
                "gt_poses": self.cfg.dataset.gt_poses,
                "droid_slam": self.cfg.dataset.droid_slam,
            }
            return HSRDataset(cfg)

    def load_seq(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Any]:
        """Extract RGB and depth frames from the dataset."""
        scene_mesh = None

        # __getitem__ is dominated by independent per-frame disk reads + image
        # decoding, which release the GIL, so a thread pool parallelizes it well.
        with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as executor:
            samples = list(tqdm(executor.map(self.dataset.__getitem__, range(len(self.dataset))), total=len(self.dataset), desc="Loading frames"))

        # executor.map yields results in submission order regardless of completion
        # order, so samples[i] should already correspond to self.dataset[i]; assert
        # against the dataset's own idx to make that guarantee self-verifying.
        assert all(sample["idx"] == i for i, sample in enumerate(samples)), "load_seq: frame order does not match dataset index order"

        rgb_frames = [sample["rgb"].astype(np.uint8) for sample in samples]
        depth_frames = [sample["depth"] for sample in samples]
        camera_poses = [sample["pose"] for sample in samples]

        reconstr_path = os.path.join(self.cfg.dataset.root_path, "compressed_mesh.ply")
        if os.path.exists(reconstr_path):
            scene_mesh = o3d.io.read_triangle_mesh(reconstr_path).compute_vertex_normals()

        return rgb_frames, depth_frames, camera_poses, scene_mesh

    def get_human_masks(self, use_precomputed, save_masks, path) -> List[np.ndarray]:
        """Load human masks from the dataset."""
        if use_precomputed and os.path.exists(os.path.join(path, "human_masks.npz")):
            human_masks = np.load(os.path.join(path, "human_masks.npz"), allow_pickle=False)['human_masks']
            np.savez_compressed(os.path.join(path, "human_masks.npz"), human_masks=human_masks)
            return human_masks
        else:
            loguru.logger.warning(f"Human masks not found in {path}/human_masks.npz. Computing human masks...")
            human_masks = self.seg.compute_prior_masks(
                rgb_frames=self.rgb_frames,
                dilate_its=self.cfg.interaction.dilate_iterations,
                dilate_kernel_size=self.cfg.interaction.dilate_kernel_size,
            )
            if save_masks:
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, "human_masks.npz"), human_masks=human_masks)
                loguru.logger.info(f"Human masks saved to {path}/human_masks.npz")
            return human_masks

    def compensate_cam_motion(self, segments: List[Tuple[int, int]], tracks_3d_segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compensate for camera motion by transforming all point tracks to the global frame.

        Accepts two formats for tracks_3d_segments:
          - Standard: List of (T, N, 3) arrays (one per segment).
          - Reconstruct: List of lists of (T, N, 3) arrays (one list per segment,
            each containing multiple keyframe-pass arrays with NaN at unused frames).
            NaN rows are skipped — camera-motion compensation is not applied to them.
        """
        for i, (start_idx, _) in enumerate(segments):
            segment_entry = tracks_3d_segments[i]
            # Reconstruct format: inner element is a list of full-segment pass arrays.
            if isinstance(segment_entry, list):
                for pass_arr in segment_entry:  # (T, N, 3) per keyframe pass
                    for j in range(pass_arr.shape[0]):
                        R = self.cam_poses[start_idx + j][:3, :3]
                        t = self.cam_poses[start_idx + j][:3, 3]
                        pass_arr[j] = (R @ pass_arr[j].T + t[:, None]).T
            else:
                # Standard format: segment_entry is (T, N, 3).
                for j in range(len(segment_entry)):
                    tracks_3d_segments[i][j] = (
                        self.cam_poses[start_idx + j][:3, :3] @ tracks_3d_segments[i][j].T + self.cam_poses[start_idx + j][:3, 3][:, None]
                    ).T
        return tracks_3d_segments

    def write_results(self, save_path, segments, result, free_traj_segments=None):
        """
        Write the results to a JSON file.
        """
        # create folder for each segment
        for i in range(len(result)):
            seg_path = os.path.join(save_path, f"segment_{i}")
            os.makedirs(seg_path, exist_ok=True)
            with open(os.path.join(seg_path, "axis_info.json"), "w") as f:
                json.dump(result[i], f, indent=4)
        for i, seg in enumerate(segments):
            # create metadata file to save start and end frame
            with open(os.path.join(seg_path, "metadata.json"), "w") as f:
                metadata = {"start_frame": seg[0], "end_frame": seg[1], "id": i, "scene_name": Path(self.cfg.dataset.root_path).stem}
                json.dump(metadata, f, indent=4)
        # write free trajectory as npy file
        if free_traj_segments is not None:
            for i, free_traj in enumerate(free_traj_segments):
                poses = np.array(free_traj)
                np.save(os.path.join(save_path, f"segment_{i}", "poses.npy"), poses)
        loguru.logger.info(f"Results saved to {save_path}")

    def segment_keyframes(self, load_dir):
        """
        Compute (or load from cache) per-keyframe segmentation masks used for 3D mapping.

        Runs Semantic-SAM or MobileSAM's automatic "segment everything" (chosen via
        ``cfg.mapping.method``) over ``self.keyframes``, caching the result to
        ``load_dir`` as a gzip pickle so subsequent runs can skip resegmentation.

        Args:
            load_dir: Directory to load a cached masks pickle from / save it to.

        Returns:
            list: Per-keyframe list of segmentation mask annotations, one entry per keyframe.
        """
        if self.cfg.mapping.method == "semantic-sam":
            loguru.logger.info("Using Semantic-SAM for segmentation.")
            if not self.cfg.cache.load_masks:
                seg_masks = []
                for idx in tqdm(self.keyframes, total=len(self.keyframes), desc="Creating semantic-sam masks"):
                    _, input_image = prepare_semsam_image(self.rgb_frames[idx].astype(np.uint8))
                    semsam_masks = self.semsam_mask_generator.generate(input_image)
                    seg_masks.append(semsam_masks)
                # save all_semsam_masks to disk
                with gzip.open(os.path.join(load_dir, "semsam_masks.pkl"), 'wb') as f:
                    pickle.dump(seg_masks, f)
            else:
                if os.path.exists(os.path.join(load_dir, "semsam_masks.pkl")):
                    with gzip.open(os.path.join(load_dir, "semsam_masks.pkl"), 'rb') as f:
                        seg_masks = pickle.load(f)
                else:
                    raise FileNotFoundError(
                        f"Semantic-SAM masks not found in {load_dir}/semsam_masks.pkl. Please run the pipeline with cfg.cache.load_masks=False to generate them."
                    )
        elif self.cfg.mapping.method == "mobile-sam":
            loguru.logger.info("Using Mobile-SAM for segmentation.")
            if not self.cfg.cache.load_masks:
                seg_masks = []
                for idx in tqdm(self.keyframes, total=len(self.keyframes), desc="Creating mobile-sam masks"):
                    mobile_sam_masks = self.seg.msam.segment_everything(self.rgb_frames[idx].astype(np.uint8))
                    seg_masks.append(mobile_sam_masks)
                with gzip.open(os.path.join(load_dir, "mobilesam_masks.pkl"), 'wb') as f:
                    pickle.dump(seg_masks, f)
            else:
                if os.path.exists(os.path.join(load_dir, "mobilesam_masks.pkl")):
                    with gzip.open(os.path.join(load_dir, "mobilesam_masks.pkl"), 'rb') as f:
                        seg_masks = pickle.load(f)
                else:
                    raise FileNotFoundError(
                        f"Mobile-SAM masks not found in {load_dir}/mobilesam_masks.pkl. Please run the pipeline with cfg.cache.load_masks=False to generate them."
                    )
        return seg_masks

    def estimate_articulations(
        self, pred_segments, forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments, save_dir
    ):
        """
        Estimate a joint (articulation) model for each interaction segment from its 3D point tracks.

        For each segment, combines forward (and, if enabled, backward) tracks, then fits
        an articulation model via RANSAC (or a single fit if RANSAC is disabled), keeping
        the sample with the highest twist-inlier ratio. Also estimates the dense per-frame
        joint angle (theta) trajectory, classifies the motion trend (single vs. cyclic,
        optionally refined with a VLM query), and renders debug visualizations to ``save_dir``.

        Args:
            pred_segments: List of (start_idx, end_idx) frame-index tuples for interaction segments.
            forw_3d_tracks_segments: Per-segment (T, N, 3) forward 3D point tracks.
            forw_vis_segments: Per-segment (T, N) forward track visibility.
            backw_3d_tracks_segments: Per-segment backward 3D point tracks (or None if disabled).
            backw_vis_segments: Per-segment backward track visibility (or None if disabled).
            save_dir: Directory to write per-segment estimation/visualization outputs to.

        Returns:
            dict: Mapping from segment index to its fitted ``InferencePointAxis`` articulation model.
        """
        # perform axis estimation
        segment_articulations = dict()
        for i, (start_idx, end_idx) in enumerate(pred_segments):
            # pred tracks: (T, N, 3) → (N, T, 3); vis: (T, N) → (N, T)

            tracks = np.array(forw_3d_tracks_segments[i]).transpose(1, 0, 2)
            vis = np.array(forw_vis_segments[i]).transpose(1, 0)

            # append backw tracks: (T, N, 3) and (T, N) with NaN before their keyframe
            if self.cfg.tracking.bidir and backw_3d_tracks_segments is not None and backw_3d_tracks_segments[i] is not None:
                r_t = np.array(backw_3d_tracks_segments[i]).transpose(1, 0, 2)  # (T, N_total, 3)
                r_v = np.array(backw_vis_segments[i]).T  # (T, N_total)
                tracks = np.concatenate([tracks, r_t], axis=0)
                vis = np.concatenate([vis, r_v], axis=0)

            # RANSAC: repeatedly fit on a random track subset, keep the model
            # with the lowest loss evaluated on all tracks.
            N_tracks = tracks.shape[0]
            n_sample = max(min(N_tracks, self.cfg.articulation.ransac.min_samples), int(self.cfg.articulation.ransac.sample_ratio * N_tracks))
            max_iters = self.cfg.articulation.ransac.max_iters
            inlier_thresh = self.cfg.articulation.ransac.inlier_thresh  # mean L2 point error threshold in metres
            vis_bool_all = vis.astype(bool)

            T_seg = tracks.shape[1]

            def _inlier_ratio(twist):
                """
                For each timestamp find one shared theta (the joint state is
                global, not per-track) that minimises the total squared error
                across all mutually-visible tracks, then accumulate per-track
                residuals at those shared thetas to count inliers.
                """
                from scipy.optimize import minimize_scalar as _ms

                ref_t = int(vis_bool_all.sum(axis=0).argmax())
                ref_vis = vis_bool_all[:, ref_t]
                p_ref_h = np.hstack([tracks[:, ref_t, :], np.ones((N_tracks, 1))])  # (N, 4)
                track_errs = {n: [] for n in range(N_tracks)}

                for t in range(T_seg):
                    if t == ref_t:
                        continue
                    mutual = ref_vis & vis_bool_all[:, t]
                    if mutual.sum() < 3:
                        continue
                    mut_idx = np.where(mutual)[0]
                    ps_h = p_ref_h[mut_idx]  # (M, 4)
                    pe = tracks[mut_idx, t, :]  # (M, 3)

                    # one shared theta for this timestamp across all tracks
                    def res(theta, ps_h=ps_h, pe=pe):
                        """Sum of squared point errors when applying the candidate twist scaled by
                        theta to the reference points ``ps_h`` and comparing to targets ``pe``."""
                        T_mat = gtsam.Pose3.Expmap(twist * theta).matrix()
                        return float(np.sum(((T_mat @ ps_h.T)[:3].T - pe) ** 2))

                    theta_t = _ms(res, bounds=(-np.pi, np.pi), method='bounded').x

                    # per-track L2 error (metres) at the shared theta
                    T_mat = gtsam.Pose3.Expmap(twist * theta_t).matrix()
                    pred = (T_mat @ p_ref_h[mut_idx].T)[:3].T  # (M, 3)
                    errs = np.linalg.norm(pred - pe, axis=1)  # (M,) metres
                    for local_i, n in enumerate(mut_idx):
                        track_errs[n].append(float(errs[local_i]))

                inliers = sum(1 for n in range(N_tracks) if track_errs[n] and np.mean(track_errs[n]) < inlier_thresh)
                return inliers / max(1, N_tracks), np.mean([np.mean(track_errs[n]) for n in range(N_tracks) if track_errs[n]])

            best_model, best_pairs, best_pairs_t = None, None, None
            best_inlier_ratio = -1.0
            # best_loss = float('inf')
            min_error = float('inf')

            # compute type prior
            pairs, _ = identify_cloud_pairs_from_multiple_starts(tracks, vis)
            if len(pairs) == 0:
                loguru.logger.warning(
                    f"No point pairs found for segment {i} ({start_idx}, {end_idx}), skipping type prior and using default sampling."
                )
                segment_articulations[i] = InferencePointAxis(
                    position=None,
                    axis=None,
                    type=None,
                    success=False,
                    id=i,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    motion_type=ARTICULATION_MODE.UNKNOWN,
                    last_observed_state=None,
                )
                continue
            earlier_points, later_points, batch_idcs = prepare_estimation_data(pairs, limit=1000)
            cos_sim, _, _, cos_median = compute_type_prior(earlier_points, later_points, batch_idcs)

            if self.cfg.articulation.use_ransac:
                for _ransac_iter in tqdm(range(max_iters), desc=f"Segment {i}: RANSAC iteration ####", total=max_iters):
                    idx = np.random.choice(N_tracks, size=n_sample, replace=False)
                    m_i, p_i, pt_i = estimate_articulation_model(self.cfg, tracks[idx], vis[idx], cos_median)
                    if m_i.twist is None:
                        continue
                    ratio, error = _inlier_ratio(np.array(m_i.twist))
                    if ratio > best_inlier_ratio or (ratio == best_inlier_ratio and error < min_error):
                        min_error = error
                        best_inlier_ratio = ratio
                        best_model, best_pairs, best_pairs_t = m_i, p_i, pt_i
            else:
                best_model, best_pairs, best_pairs_t = estimate_articulation_model(self.cfg, tracks, vis, cos_median)

            loguru.logger.info(f"Segment {i}: RANSAC best inlier ratio = {best_inlier_ratio:.2%}")
            if best_model is None or best_model.position is None:
                best_model, best_pairs, best_pairs_t = estimate_articulation_model(self.cfg, tracks, vis)

            model, pairs, pairs_t = best_model, best_pairs, best_pairs_t

            model.id = i
            model.start_idx, model.end_idx = start_idx, end_idx
            model.pairs, model.pairs_t = pairs, pairs_t
            segment_articulations[i] = model  # add failed / empty estimate

            plot_estimate(model, pairs, tracks, save_dir)

            # Estimate dense thetas throughout interaction segment
            start_frame = list(pairs_t.keys())[0]  # start from first frame utilized for point pair sampling
            theta_bounds = model.bounds if model.bounds is not None else (-np.pi, np.pi)
            dense_thetas, dense_frame_indices = estimate_dense_thetas(
                np.array(model.twist),
                np.array(forw_3d_tracks_segments[i]).transpose(1, 0, 2),  # only use forward tracks
                np.array(forw_vis_segments[i]).transpose(1, 0),
                start_frame=start_frame,
                theta_bounds=theta_bounds,
            )
            segment_articulations[i].dense_thetas = dense_thetas
            segment_articulations[i].dense_frame_indices = dense_frame_indices
            loguru.logger.info(f"Segment {i}: estimated dense thetas over {(~np.isnan(dense_thetas)).sum()} / {len(dense_thetas)} frames.")

            plot_dense_theta_estimate(model, save_dir)

            motion_trend = estimate_motion_mode(model.dense_thetas)  # returns either "single" or "cycle"

            if self.cfg.articulation.use_vlm:
                # label articulation states with VLM
                prompt = f"Is the following series of images a consecutive \
                    '{ARTICULATION_MODE.OPENING_CLOSING.value}', '{ARTICULATION_MODE.CLOSING_OPENING.value}' \
                    or just '{ARTICULATION_MODE.OPENING.value}' or '{ARTICULATION_MODE.CLOSING.value}'?. \
                    Please answer with one of these four options."
                prompt_kf_idcs = [start_idx, start_idx + (end_idx - start_idx) // 3, start_idx + 2 * (end_idx - start_idx) // 3, end_idx - 1]
                articulation_rgb_frames = [
                    self.rgb_frames[prompt_kf_idcs[0]],
                    self.rgb_frames[prompt_kf_idcs[1]],
                    self.rgb_frames[prompt_kf_idcs[2]],
                    self.rgb_frames[prompt_kf_idcs[3]],
                ]

                parsed_trend = parse_response(query_articulation_mode(self.cfg, articulation_rgb_frames, prompt), ARTICULATION_MODE)
                plot_articulation_mode(self.cfg, self.rgb_frames, prompt_kf_idcs, parsed_trend, motion_trend, i, save_dir)

                concluded_mode = None
                # check that the parsed mode is consistent with the motion trend
                if motion_trend == "single" and parsed_trend in [ARTICULATION_MODE.OPENING_CLOSING, ARTICULATION_MODE.CLOSING_OPENING]:
                    loguru.logger.warning(
                        f"Inconsistent articulation mode for segment {i}: motion trend is 'single' but parsed mode is '{parsed_trend.value}'"
                    )
                    concluded_mode = ARTICULATION_MODE.UNKNOWN
                elif motion_trend == "cycle" and parsed_trend in [ARTICULATION_MODE.OPENING, ARTICULATION_MODE.CLOSING]:
                    loguru.logger.warning(
                        f"Inconsistent articulation mode for segment {i}: motion trend is 'cycle' but parsed mode is '{parsed_trend.value}'"
                    )
                    concluded_mode = ARTICULATION_MODE.UNKNOWN
                else:
                    concluded_mode = parsed_trend
            else:
                loguru.logger.info(f"Segment {i}: VLM disabled, skipping articulation mode labeling.")
                concluded_mode = ARTICULATION_MODE.UNKNOWN

            segment_articulations[i].motion_type = concluded_mode
            segment_articulations[i].last_observed_state = LAST_INFERRED_STATE_MAP[concluded_mode]
            segment_articulations[i].max_theta_frame_idx = np.argmax(model.dense_thetas) + start_idx if len(model.dense_thetas) > 0 else None

            # plot the axis on the frame where the articulation is most open (max theta) for visualization
            img_out = plot_axis_img(
                self.rgb_frames[model.max_theta_frame_idx],
                self.depth_frames[model.max_theta_frame_idx],
                np.array(model.position),
                np.array(model.axis),
                self.dataset.depth_intrinsics,
                np.linalg.inv(self.cam_poses[model.max_theta_frame_idx]),
                None,
            )
            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_axis_.png"), img_out)

            # loguru.logger.info(f"Segment {i} ({start_idx}, {end_idx}): Estimated {model.type} and articulation mode is '{concluded_mode.value}'")
        return segment_articulations

    def match_tracks_objects(
        self,
        segments,
        hierarchy,
        segment_articulations,
        point_tracks_world,
        point_vis,
        dyn_tracks_world,
        dyn_vis,
        dyn_mask,
        save_dir,
        visualize=False,
    ):
        """
        Associate each interaction segment's articulation model with a 3D map object.

        Projects both dynamic (inlier) and static world-frame tracks back into each
        frame's camera view, optionally runs SAM2 reconstruction over dynamic ROIs.
        For each segment, applies the estimated articulation twist at multiple dense
        theta samples to the top-k nearest candidate objects' point clouds, reprojects
        them, and scores each candidate object by how well its projected mask agrees
        with dynamic/static track membership (and, if available, IoU with the SAM2
        reconstruction). Segments are then assigned to objects either greedily or via
        an optimal (binary integer programming) assignment over the resulting cost matrix,
        and the winning model is attached to ``hierarchy.objects[...]["model"]``.

        :param segments: List of (start_idx, end_idx) frame-index tuples for interaction segments.
        :param hierarchy: Object ``Hierarchy`` whose objects are candidates for association.
        :param segment_articulations: Dict mapping segment index to its fitted articulation model.
        :param point_tracks_world: Per-segment (T, N, 3) world-frame static point tracks.
        :param point_vis: Per-segment (T, N) visibility for ``point_tracks_world``.
        :param dyn_tracks_world: Per-segment (T, N, 3) world-frame dynamic (twist-inlier) point tracks.
        :param dyn_vis: Per-segment (T, N) visibility for ``dyn_tracks_world``.
        :param dyn_mask: Per-segment boolean mask selecting which of ``point_tracks_world`` are dynamic.
        :param save_dir: Directory to write debug visualizations/videos to.
        :param visualize: If True, render and save per-frame/per-candidate debug visualizations.
        :return: Tuple (segment_articulations, hierarchy) with ``obj_id`` set on matched models
            and ``"model"`` set on the corresponding entries of ``hierarchy.objects``.
        """
        loguru.logger.info("Matching tracks to objects and projecting to image frames...")
        img_dyn_tracks, img_static_tracks = {}, {}
        img_dyn_vis, img_static_vis = {}, {}

        if self.cfg.assoc.reconst.use_sam:
            tqdm.write("Operating in full reconstruction mode using Semantic-SAM and SAM2")

        instances = {}
        self.reconstructions = {}
        skip_frames = defaultdict(list)
        for i, (start_idx, end_idx) in tqdm(enumerate(segments), total=len(segments), desc="Re-project keypoints to image frames"):
            img_dyn_tracks[i] = []
            img_static_tracks[i] = []
            img_dyn_vis[i] = []
            img_static_vis[i] = []
            instances[i] = {}

            if dyn_tracks_world[i].shape[1] == 0 or point_tracks_world[i].shape[1] == 0:
                loguru.logger.warning(f"No tracks for segment {i}, skipping projection and matching. Cannot associate object.")
                continue

            seg_prior_masks = self.prior_masks[start_idx:end_idx].astype(bool)  # (segment_len, H, W)
            # project all tracks back to camera frame
            tqdm.write(f"Processing segment {i} with frames {start_idx} to {end_idx}")
            for j in tqdm(range(dyn_tracks_world[i].shape[0]), desc=f"Evaluating dyn/static points + masks per frame for segment {i}"):
                dyn_points = dyn_tracks_world[i][j]  # (N, 3)
                static_points = point_tracks_world[i][j, ~dyn_mask[i]]

                if dyn_points.shape[0] == 0 or static_points.shape[0] == 0:
                    loguru.logger.warning(f"No projected points for segment {i}, frame {j}.")
                    skip_frames[i].append(start_idx + j)
                    continue

                cam2world = np.linalg.inv(self.cam_poses[start_idx + j])
                R, t = cam2world[:3, :3], cam2world[:3, 3]
                rvec, _ = cv2.Rodrigues(R)

                img_dyn_points, _ = cv2.projectPoints(
                    dyn_points.reshape(-1, 1, 3),  # Nx1x3 array
                    rvec,  # rotation vector (3x1)
                    t,  # translation vector (3x1)
                    self.dataset.depth_intrinsics,  # K matrix (3x3)
                    self.dataset.camera_params["D"],
                )
                img_static_points, _ = cv2.projectPoints(
                    static_points.reshape(-1, 1, 3),  # Nx1x3 array
                    rvec,  # rotation vector (3x1)
                    t,  # translation vector (3x1)
                    self.dataset.depth_intrinsics,  # K matrix (3x3)
                    self.dataset.camera_params["D"],
                )

                # Generate dynamic mask ROIs using semantic-sam (later used for prompting fine-grained SAM-based articulated object reconstruction)
                if j % 2 == 0 and self.cfg.assoc.reconst.use_sam:
                    _, input_image = prepare_semsam_image(self.rgb_frames[start_idx + j].astype(np.uint8))
                    semsam_masks = self.semsam_mask_generator.generate(input_image)
                    pot_part_gobs = parse_masks(semsam_masks, self.rgb_frames, start_idx + j)

                    masks = np.array([mask for mask in pot_part_gobs['mask'] if mask.sum() > 1000])  # filter out small masks

                    visible_dyn_points = img_dyn_points[dyn_vis[i][j, :].astype(bool)]
                    visible_static_points = img_static_points[point_vis[i][j, ~dyn_mask[i]].astype(bool)]

                    prompt_masks = []
                    # check how many static / dynamic points fall into each mask
                    for mask in masks:
                        # Count the number of dynamic points inside the mask
                        dyn_count = 0
                        static_count = 0
                        for pt in visible_dyn_points:
                            if np.isnan(pt[0][0]) or np.isnan(pt[0][1]):
                                continue
                            x, y = int(pt[0][0]), int(pt[0][1])
                            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                                continue
                            if mask[y, x]:
                                dyn_count += 1

                        for pt in visible_static_points:
                            if np.isnan(pt[0][0]) or np.isnan(pt[0][1]):
                                continue
                            x, y = int(pt[0][0]), int(pt[0][1])
                            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                                continue
                            if mask[y, x]:
                                static_count += 1

                        ratio = dyn_count / (dyn_count + static_count + 1e-5)
                        if ratio > self.cfg.assoc.reconst.mask_ratio:
                            prompt_masks.append(mask)
                    # merge all prompt masks into one mask for this frame
                    frame_mask = np.zeros_like(masks[0])
                    if len(prompt_masks) > 0:
                        for mask in prompt_masks:
                            frame_mask[mask] = 1

                    # cut out prior mask area from frame_mask
                    frame_mask[seg_prior_masks[j]] = 0
                    instances[i][j] = frame_mask

                # draw those onto the image
                if visualize:
                    rgb_frame = cv2.cvtColor(self.rgb_frames[start_idx + j], cv2.COLOR_RGB2BGR)
                    for pt in img_dyn_points:
                        if (
                            np.isnan(pt[0][0])
                            or np.isnan(pt[0][1])
                            or pt[0][0] < 0
                            or pt[0][1] < 0
                            or pt[0][0] >= rgb_frame.shape[1]
                            or pt[0][1] >= rgb_frame.shape[0]
                        ):
                            continue
                        x, y = int(pt[0][0]), int(pt[0][1])
                        cv2.circle(rgb_frame, (x, y), 2, (0, 0, 255), -1)
                    for pt in img_static_points:
                        if (
                            np.isnan(pt[0][0])
                            or np.isnan(pt[0][1])
                            or pt[0][0] < 0
                            or pt[0][1] < 0
                            or pt[0][0] >= rgb_frame.shape[1]
                            or pt[0][1] >= rgb_frame.shape[0]
                        ):
                            continue
                        x, y = int(pt[0][0]), int(pt[0][1])
                        cv2.circle(rgb_frame, (x, y), 2, (255, 0, 0), -1)
                    cv2.imwrite(os.path.join(save_dir, f"assoc_proj_points_{i}_debug.png"), rgb_frame)

                if img_dyn_points is None or img_static_points is None:
                    loguru.logger.warning(f"No projected points for segment {i}, frame {j}.")
                    skip_frames[i].append(start_idx + j)
                    continue

                img_dyn_tracks[i].append(img_dyn_points.squeeze(1))
                img_static_tracks[i].append(img_static_points.squeeze(1))
                img_dyn_vis[i].append(dyn_vis[i][j, :])
                img_static_vis[i].append(point_vis[i][j, ~dyn_mask[i]])

            img_dyn_tracks[i] = np.array(img_dyn_tracks[i]).transpose(1, 0, 2) if len(img_dyn_tracks[i]) > 0 else None  # (N, T, 2)
            img_static_tracks[i] = np.array(img_static_tracks[i]).transpose(1, 0, 2) if len(img_static_tracks[i]) > 0 else None  # (N, T, 2)
            img_dyn_vis[i] = np.array(img_dyn_vis[i]).transpose(1, 0) if len(img_dyn_vis[i]) > 0 else None  # (N, T)
            img_static_vis[i] = np.array(img_static_vis[i]).transpose(1, 0) if len(img_static_vis[i]) > 0 else None  # (N, T)

            if instances[i]:
                self.reconstructions[i] = self.seg.reconstruct_with_sam2(self.rgb_frames[start_idx:end_idx], instances[i], save_dir, i)

        segment_topk = {}
        segment_costs = {}
        rgb_videos = dict()
        for i, (start_idx, end_idx) in enumerate(segments):
            model = segment_articulations[i]
            if model.position is None or model.axis is None:
                loguru.logger.warning(f"No valid articulation model for segment {i}, skipping track-object matching.")
                continue

            # only use estimation tracks for object association
            # thus, remove all rows that start with a NaN (these correspond to frames where no points were projected)
            valid_dyn_rows = ~np.isnan(img_dyn_tracks[i][:, 0, 0])
            est_seg_img_dyn_tracks = img_dyn_tracks[i][valid_dyn_rows]
            est_seg_img_dyn_vis = img_dyn_vis[i][valid_dyn_rows]
            valid_static_rows = ~np.isnan(img_static_tracks[i][:, 0, 0])
            est_seg_img_static_tracks = img_static_tracks[i][valid_static_rows]
            est_seg_img_static_vis = img_static_vis[i][valid_static_rows]

            # query objects close to the interaction position
            interaction_pos_world = np.mean(dyn_tracks_world[i], axis=(0, 1))
            dists = list()
            for obj in hierarchy.objects:
                mean_obj_pos = obj["points"].mean(axis=0)
                dists.append(np.linalg.norm(interaction_pos_world - mean_obj_pos))

            rgb_videos[i] = {}
            # get the top-k closest objects
            num_topk = self.cfg.assoc.matching.topk
            k = min(num_topk, len(dists))
            topk_idcs = np.argsort(dists)[:k]
            costs = list()

            obj_theta_samples = defaultdict(list)
            for idx in tqdm(topk_idcs, total=len(topk_idcs), desc=f"Matching tracks to objects for segment {i}"):
                rgb_videos[i][idx] = []
                obj_points = hierarchy.objects[idx]["points"]
                static_ratio, dyn_ratio, ious, depth_error = (list(), list(), list(), list())

                thetas_iter = zip(model.dense_frame_indices, model.dense_thetas)
                for rel_frame_idx, theta in thetas_iter:
                    if np.isnan(theta):
                        continue
                    # take object point cloud and rotate all its points based on the estimated twist
                    obj_points_homog = np.hstack((obj_points, np.ones((obj_points.shape[0], 1))))
                    trans_obj_points_homog = gtsam.Pose3.Expmap(np.array(model.twist) * theta).matrix() @ obj_points_homog.T
                    obj_theta_samples[idx].append(trans_obj_points_homog[:3].T)
                    frame_idx = rel_frame_idx + start_idx

                    cam2world = np.linalg.inv(self.cam_poses[frame_idx])  # self.cam_poses[frame_idx] is world2cam
                    rvec, _ = cv2.Rodrigues(cam2world[:3, :3])
                    obj_points_cam, _ = cv2.projectPoints(
                        trans_obj_points_homog.T[:, :3].reshape(-1, 1, 3),  # Nx1x3 array
                        rvec,  # rotation vector (3x1)
                        cam2world[:3, 3],  # translation vector (3x1)
                        self.dataset.depth_intrinsics,
                        np.array(self.dataset.camera_params["D"]),
                    )

                    hull = cv2.convexHull(obj_points_cam.squeeze(1).astype(np.float32))
                    obj_mask_cam = np.zeros(self.rgb_frames[frame_idx].shape[:2], dtype=np.uint8)
                    # clip hull to image boundaries
                    hull[:, 0, 0] = np.clip(hull[:, 0, 0], 0, obj_mask_cam.shape[1] - 1)
                    hull[:, 0, 1] = np.clip(hull[:, 0, 1], 0, obj_mask_cam.shape[0] - 1)
                    if hull is not None and len(hull) >= 3:
                        cv2.fillConvexPoly(obj_mask_cam, hull.astype(np.int32), 1)

                    # enlarge mask by dilation
                    # kernel = np.ones((10,10), np.uint8)
                    # obj_mask_cam = cv2.dilate(obj_mask_cam, kernel, iterations=1)

                    if obj_mask_cam.sum() < 100 or obj_mask_cam.sum() == np.prod(obj_mask_cam.shape):
                        static_ratio.append(1)
                        dyn_ratio.append(0)
                        ious.append(0)
                        continue
                    mask_rgb = copy.deepcopy(self.rgb_frames[frame_idx])
                    mask_rgb[obj_mask_cam == 1] = [0, 0, 255]  # white-out the object mask area

                    # evaluate the ratio of dynamic tracks falling into the reprojected mask
                    dyn_kp = est_seg_img_dyn_tracks[est_seg_img_dyn_vis[:, rel_frame_idx].astype(bool), rel_frame_idx, :].astype(np.int32)  # (N, 2)
                    H, W = obj_mask_cam.shape[:2]
                    valid = (dyn_kp[:, 0] >= 0) & (dyn_kp[:, 0] < W) & (dyn_kp[:, 1] >= 0) & (dyn_kp[:, 1] < H)
                    dyn_kp = dyn_kp[valid]
                    inside_mask = obj_mask_cam[dyn_kp[:, 1], dyn_kp[:, 0]] > 0
                    kp_inside = dyn_kp[inside_mask]
                    dyn_ratio.append(len(kp_inside) / dyn_kp.shape[0] if dyn_kp.shape[0] > 0 else 0)

                    static_kp = est_seg_img_static_tracks[est_seg_img_static_vis[:, rel_frame_idx].astype(bool), rel_frame_idx, :].astype(
                        np.int32
                    )  # (N, 2)
                    valid = (static_kp[:, 0] >= 0) & (static_kp[:, 0] < W) & (static_kp[:, 1] >= 0) & (static_kp[:, 1] < H)
                    static_kp = static_kp[valid]
                    inside_mask = obj_mask_cam[static_kp[:, 1], static_kp[:, 0]] > 0
                    kp_inside = static_kp[inside_mask]
                    static_ratio.append(len(kp_inside) / static_kp.shape[0] if static_kp.shape[0] > 0 else 0)

                    # if available, use also IoU between articulated object mask and SAM2 reconstruction mask
                    if i in self.reconstructions and rel_frame_idx in self.reconstructions[i] and 1 in self.reconstructions[i][rel_frame_idx]:
                        recon_mask = self.reconstructions[i][rel_frame_idx][1].squeeze(0)  # (H, W) bool
                        intersection = np.logical_and(obj_mask_cam > 0, recon_mask).sum()
                        union = np.logical_or(obj_mask_cam > 0, recon_mask).sum()
                        ious.append(float(intersection) / (float(union) + 1e-6))

                    if visualize:
                        img_out = plot_axis_img(
                            self.rgb_frames[frame_idx],
                            self.depth_frames[frame_idx],
                            np.array(model.position),
                            np.array(model.axis),
                            self.dataset.depth_intrinsics,
                            np.linalg.inv(self.cam_poses[frame_idx]),
                        )
                        img_out[obj_mask_cam == 1] = [0, 0, 255]  # white-out the object mask area
                        for pt in dyn_kp:
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(img_out, (x, y), 2, (0, 255, 0), -1)
                        for pt in static_kp:
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(img_out, (x, y), 2, (255, 0, 0), -1)
                        # downscale for visualization
                        img_out = cv2.resize(img_out, (img_out.shape[1] // 2, img_out.shape[0] // 2))
                        rgb_videos[i][idx].append(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

                avg_obj_iou = np.mean(ious) if ious else 0
                cost = (
                    (1 - np.mean(dyn_ratio)) + np.mean(static_ratio) + (1 - avg_obj_iou)
                )  # we want high dynamic ratio and low static ratio + high IoU with MobileSAM masks + low depth error
                costs.append(cost)
                segment_topk[i] = topk_idcs
                segment_costs[i] = costs

            # SUPER-GREEDY ASSOCIATION
            # closest_topk_idcs = np.argsort(segment_costs[i])  # sort combined scores from low to high
            # # one by one, go through the sorted list and find the first object that has not been matched yet
            # for rank in closest_topk_idcs:
            #     candidate_obj_idx = topk_idcs[rank]
            #     if not any([hasattr(segment_articulations[j], "obj_id") and segment_articulations[j].obj_id == candidate_obj_idx for j in range(i)]):
            #         closest_obj_idx = candidate_obj_idx
            #         break
            # loguru.logger.info(f"Segment {i} matched to object {closest_obj_idx} with cost {segment_costs[i][closest_topk_idcs[0]]:.4f}")
            # cv2.imwrite(os.path.join(save_dir, f"matched_segment_{i}_object_{closest_obj_idx}.png"), to_bgr(rgb_videos[i][closest_obj_idx][0]))
            # if model.type == "revolute":
            #     obj_centroid = np.mean(hierarchy.objects[closest_obj_idx]["points"], axis=0)
            #     # find point on axis defined by (model.position, model.axis) closest to obj_centroid
            #     closest_point_on_axis = model.position + np.dot(obj_centroid - np.array(model.position), np.array(model.axis)) * np.array(model.axis)
            #     loguru.logger.info(f"Updated revolute axis position for segment {i} by {closest_point_on_axis - np.array(model.position)}")
            #     model.position = closest_point_on_axis
            # hierarchy.objects[closest_obj_idx]["model"] = model
            # segment_articulations[i].obj_id = closest_obj_idx

            if visualize:
                for cand_idx, cand_frames in rgb_videos[i].items():
                    if len(cand_frames) == 0:
                        continue
                    video_writer = cv2.VideoWriter(
                        os.path.join(
                            save_dir, "matching", f"segment_{i}_unmatched_object_{costs[list(rgb_videos[i].keys()).index(cand_idx)]:.4f}.mp4"
                        ),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        10,
                        (cand_frames[0].shape[1], cand_frames[0].shape[0]),
                    )
                    for frame in cand_frames:
                        video_writer.write(to_bgr(frame))
                    video_writer.release()

        # initialize candidate assignments and cost matrix
        loguru.logger.info("Initializing cost matrix for articulation-object assignment.")
        cost_dict = {}
        candidates = defaultdict(list)
        art2obj_cost = np.zeros((len(segments), num_topk), dtype=np.float32)
        topk_matrix = np.zeros((len(segments), num_topk), dtype=np.int32)
        for i in range(len(segments)):
            if i not in segment_costs:
                loguru.logger.warning(f"Segment {i} was skipped during matching, excluding from assignment problem.")
                continue
            for j in range(len(segment_costs[i])):
                art2obj_cost[i, j] = segment_costs[i][j] if not np.isnan(segment_costs[i][j]) else 1000.0
                topk_matrix[i, j] = segment_topk[i][j]
            for k in np.argsort(segment_costs[i])[:10]:
                candidates[i].append(segment_topk[i][k])
                cost_dict[(i, segment_topk[i][k])] = segment_costs[i][k] if not np.isnan(segment_costs[i][k]) else 1000.0

        overlap_matrix = np.zeros((len(hierarchy.objects), len(hierarchy.objects)), dtype=np.float32)
        overlap = dict()
        all_obj_idcs = topk_matrix.flatten().tolist()
        for obj_idx in all_obj_idcs:
            for obj_idx2 in all_obj_idcs:
                if obj_idx >= obj_idx2:
                    continue
                bbox1 = hierarchy.objects[obj_idx]["pcd"].get_axis_aligned_bounding_box()
                bbox2 = hierarchy.objects[obj_idx2]["pcd"].get_axis_aligned_bounding_box()
                iou = iou_aabb(bbox1, bbox2)
                overlap_matrix[obj_idx, obj_idx2] = iou
                overlap_matrix[obj_idx2, obj_idx] = iou
                if iou > 0:
                    overlap[(obj_idx, obj_idx2)] = iou
                    overlap[(obj_idx2, obj_idx)] = iou

        if not self.cfg.assoc.optimal:
            # use greedy association
            loguru.logger.info("Using greedy articulation-object association...")
            matched_articulations = list()
            matched_objects = list()
            discarded_objects = list()
            assignment = dict()
            while len(matched_articulations) < len(segments):
                lowest_cost_idx = np.argsort(art2obj_cost, axis=None)[0]
                lowest_segment = lowest_cost_idx // art2obj_cost.shape[1]
                lowest_obj_idx_rel = lowest_cost_idx % art2obj_cost.shape[1]
                lowest_obj_idx = topk_matrix[lowest_segment][lowest_obj_idx_rel]
                if lowest_obj_idx not in matched_objects and lowest_obj_idx not in discarded_objects:
                    matched_articulations.append(lowest_segment)
                    matched_objects.append(lowest_obj_idx)
                    assignment[lowest_segment] = lowest_obj_idx
                    loguru.logger.info(
                        f"Segment {lowest_segment} matched to object {lowest_obj_idx} with cost {art2obj_cost[lowest_segment, lowest_obj_idx_rel]:.4f}"
                    )
                    art2obj_cost[lowest_segment, :] = 1e6  # mark articulation as matched
                    obj_to_be_discarded = (overlap_matrix[lowest_obj_idx, :] > 0.1).nonzero()[0].tolist()
                    # increase costs of discarded objects for all other segments
                    for seg_idx in range(art2obj_cost.shape[0]):
                        for obj_idx in obj_to_be_discarded:
                            if obj_idx in topk_matrix[seg_idx, :]:
                                obj_rel_idx = np.where(topk_matrix[seg_idx, :] == obj_idx)[0][0]
                                art2obj_cost[seg_idx, obj_rel_idx] = float('inf')
                    discarded_objects.extend(obj_to_be_discarded)
                    hierarchy.objects[lowest_obj_idx]["model"] = segment_articulations[lowest_segment]
                    segment_articulations[lowest_segment].obj_id = lowest_obj_idx
                else:
                    art2obj_cost[lowest_segment, lowest_obj_idx_rel] = float('inf')
                    continue
                loguru.logger.info(
                    f"Segment {lowest_segment} matched to object {lowest_obj_idx} with cost {art2obj_cost[lowest_segment, lowest_obj_idx_rel]:.4f}"
                )

        else:
            # use optimal assignment via binary integer programming
            loguru.logger.info("Solving optimal articulation-object assignment using binary integer programming...")

            assignment, total_cost = solve_assignment(candidates, cost_dict, overlap, lambda_overlap=self.cfg.assoc.matching.lambda_overlap)
            loguru.logger.info(f"Obtained BIP assignment at cost: {total_cost:.4f}")

            for i, (start_idx, end_idx) in enumerate(segments):
                model = segment_articulations[i]
                if model.position is None:
                    loguru.logger.warning(f"No valid articulation model for segment {i}, skipping object association.")
                    continue
                closest_obj_idx = assignment[i]
                loguru.logger.info(f"Segment {i} matched to object {closest_obj_idx} with cost {cost_dict[(i, closest_obj_idx)]:.4f}")
                if visualize:
                    if len(rgb_videos[i][closest_obj_idx]) > 0:
                        cv2.imwrite(
                            os.path.join(save_dir, f"matched_segment_{i}_object_{cost_dict[(i, closest_obj_idx)]:.4f}.png"),
                            rgb_videos[i][closest_obj_idx][0],
                        )

                if model.type == "revolute":
                    model.position, model.axis = (np.array(model.position), np.array(model.axis))
                    obj_centroid = np.mean(hierarchy.objects[closest_obj_idx]["points"], axis=0)
                    # find point on axis defined by (model.position, model.axis) closest to obj_centroid
                    closest_point_on_axis = model.position + np.dot(obj_centroid - model.position, model.axis) * model.axis
                    loguru.logger.info(f"Updated revolute axis position for segment {i} by {closest_point_on_axis - model.position}")
                    model.position = closest_point_on_axis
                hierarchy.objects[closest_obj_idx]["model"] = model
                segment_articulations[i].obj_id = closest_obj_idx
            # END OF OPTIMAL ASSIGNMENT
            loguru.logger.info("Obtained final articulation-object assignments.")

        del rgb_videos
        return segment_articulations, hierarchy

    def discover_children(self, segments, hierarchy, seg_articulations, save_dir):
        """
        Identify child parts (e.g. handles, drawers-within-a-cabinet) attached to each
        matched articulated object and add them to the hierarchy.

        For each segment with a matched object, finds the frame of maximum joint opening,
        projects the (articulated) parent object's point cloud at that theta into an ROI
        mask (a "traveled volume" envelope for prismatic joints, or a rotated-minus-static
        envelope for revolute joints), runs a MobileSAM "segment everything" pass at that
        frame, and keeps masks whose IoU/containment with the ROI (or with the moved-parent
        mask minus the static parent) indicate they are a distinct child part rather than
        the parent itself. Each accepted child is projected to 3D and added to ``hierarchy``
        via ``hierarchy.add_child``, tagged as "STATIC" or "ARTICULATED" relative to the parent.

        Args:
            segments: List of (start_idx, end_idx) frame-index tuples for interaction segments.
            hierarchy: Object ``Hierarchy`` whose matched objects are searched for children.
            seg_articulations: Dict mapping segment index to its fitted, object-matched articulation model.
            save_dir: Directory to write per-segment/per-object debug visualizations to.

        Returns:
            tuple: (seg_articulations, hierarchy), with ``roi_mask``/``max_opening_frame`` set on
            models and any discovered child objects added to ``hierarchy``.
        """
        for i, (start_idx, end_idx) in enumerate(segments):
            model = seg_articulations[i]
            loguru.logger.info(f"Identify children to {model.type} object articulated in segment {i}")
            if model.position is None or model.axis is None:
                loguru.logger.warning(f"No valid articulation model or no matched object for segment {i}, skipping children identification.")
                continue
            if not hasattr(model, "obj_id") or model.obj_id is None:
                loguru.logger.warning(f"No matched object for segment {i}, skipping children identification.")
                continue

            from scipy.interpolate import interp1d

            if np.isnan(model.dense_thetas).any():
                loguru.logger.warning(f"Segment {i} has NaN theta estimates, cubic interpolating thetas.")
                # interpolate missing thetas for envelope computation
                valid_indices = np.where(~np.isnan(model.dense_thetas))[0]
                if len(valid_indices) < 10:
                    loguru.logger.warning(f"Segment {i} has fewer than 2 valid theta estimates, skipping envelope-based children identification.")
                else:
                    interp_func = interp1d(
                        np.array(model.dense_frame_indices)[valid_indices],
                        np.array(model.dense_thetas)[valid_indices],
                        kind='cubic',
                        fill_value="extrapolate",
                    )
                    nan_indices = np.where(np.isnan(model.dense_thetas))[0]
                    for nan_idx in nan_indices:
                        model.dense_thetas[nan_idx] = interp_func(np.array(model.dense_frame_indices)[nan_idx]).squeeze().item()

            model.motion_type = ARTICULATION_MODE(model.motion_type) if isinstance(model.motion_type, str) else model.motion_type
            # Identify children nodes: get time of maximum opening angle
            max_opening_frame_idx = MAX_OPENING_MAP[model.motion_type](
                model.dense_frame_indices, model.dense_thetas, start_idx
            )  # defaults to max theta frame if unknown
            seg_articulations[i].max_opening_frame = max_opening_frame_idx
            rel_max_open_theta_idx = max_opening_frame_idx - start_idx
            # check potential children masks for intersection with the articulated parent mask
            parent_obj_points_homog = np.hstack(
                (hierarchy.objects[model.obj_id]["points"], np.ones((hierarchy.objects[model.obj_id]["points"].shape[0], 1)))
            )
            artic_parent_obj_points = (
                gtsam.Pose3.Expmap(np.array(model.twist) * model.dense_thetas[rel_max_open_theta_idx]).matrix() @ parent_obj_points_homog.T
            )

            intrinsics = o3d.core.Tensor(self.dataset.depth_intrinsics, device=o3d.core.Device("CUDA:0"))
            max_open_extrinsics = o3d.core.Tensor(np.linalg.inv(self.cam_poses[max_opening_frame_idx]), device=o3d.core.Device("CUDA:0"))

            if model.type == "prismatic":
                traveled_volume = []
                for rel_idx in range(np.argmax(model.dense_thetas)):
                    travel_parent_obj_points = (
                        gtsam.Pose3.Expmap(np.array(model.twist) * model.dense_thetas[rel_idx]).matrix() @ parent_obj_points_homog.T
                    )
                    traveled_volume.append(travel_parent_obj_points[:3, :].T)

                traveled_volume = np.concatenate(traveled_volume, axis=0)
                envelope_cloud = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
                envelope_cloud.point.positions = o3d.core.Tensor(traveled_volume, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0"))
                envelope_depth = (
                    envelope_cloud.project_to_depth_image(
                        width=self.depth_frames[max_opening_frame_idx].shape[1],
                        height=self.depth_frames[max_opening_frame_idx].shape[0],
                        intrinsics=intrinsics,
                        extrinsics=max_open_extrinsics,
                    )
                    .as_tensor()
                    .cpu()
                    .numpy()
                )
                if envelope_depth is None or np.count_nonzero(envelope_depth) == 0:
                    loguru.logger.warning(f"No envelope points projected for segment {i}, skipping envelope mask generation.")
                    continue

                envelope_points_2d = np.flip(np.array(envelope_depth.nonzero())[:2, :].T, 1)

                hull = ConvexHull(envelope_points_2d)
                roi_mask = np.zeros_like(self.depth_frames[max_opening_frame_idx], dtype=np.uint8)
                if hull is not None and len(hull.vertices) >= 3:
                    cv2.fillConvexPoly(roi_mask, envelope_points_2d[hull.vertices].astype(np.int32), 1)
                else:
                    for pt in envelope_points_2d.astype(np.int32):
                        x, y = pt
                        if 0 <= x < roi_mask.shape[1] and 0 <= y < roi_mask.shape[0]:
                            roi_mask[y, x] = 1
                cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_envelope_mask.png"), roi_mask * 255)

                # reduce by max open parent mask
                max_open_parent_points = travel_parent_obj_points[:3, :].T
                max_open_parent_cloud = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
                max_open_parent_cloud.point.positions = o3d.core.Tensor(
                    max_open_parent_points, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0")
                )
                max_open_parent_depth = (
                    max_open_parent_cloud.project_to_depth_image(
                        width=self.depth_frames[max_opening_frame_idx].shape[1],
                        height=self.depth_frames[max_opening_frame_idx].shape[0],
                        intrinsics=intrinsics,
                        extrinsics=max_open_extrinsics,
                    )
                    .as_tensor()
                    .cpu()
                    .numpy()
                )
                max_open_points_2d = np.flip(np.array(max_open_parent_depth.nonzero())[:2, :].T, 1)
                if len(max_open_points_2d) == 0:
                    loguru.logger.warning(f"No max open parent points projected for segment {i}, skipping max open parent mask generation.")
                    continue
                max_open_hull = ConvexHull(max_open_points_2d)
                max_open_parent_mask = np.zeros_like(self.depth_frames[max_opening_frame_idx], dtype=np.uint8)
                if max_open_hull is not None and len(max_open_hull.vertices) >= 3:
                    cv2.fillConvexPoly(max_open_parent_mask, max_open_points_2d[max_open_hull.vertices].astype(np.int32), 1)
                else:
                    for pt in max_open_points_2d.astype(np.int32):
                        x, y = pt
                        if 0 <= x < max_open_parent_mask.shape[1] and 0 <= y < max_open_parent_mask.shape[0]:
                            max_open_parent_mask[y, x] = 1
                cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_max_open_parent_mask.png"), max_open_parent_mask * 255)

            elif model.type == "revolute":
                static_parent_cloud = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
                static_parent_cloud.point.positions = o3d.core.Tensor(
                    parent_obj_points_homog[:, :3], dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0")
                )
                static_parent_depth = (
                    static_parent_cloud.project_to_depth_image(
                        width=self.depth_frames[max_opening_frame_idx].shape[1],
                        height=self.depth_frames[max_opening_frame_idx].shape[0],
                        intrinsics=intrinsics,
                        extrinsics=max_open_extrinsics,
                    )
                    .as_tensor()
                    .cpu()
                    .numpy()
                )

                max_open_parent_cloud = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
                max_open_parent_cloud.point.positions = o3d.core.Tensor(
                    artic_parent_obj_points[:3, :].T, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0")
                )
                max_open_parent_depth = (
                    max_open_parent_cloud.project_to_depth_image(
                        width=self.depth_frames[max_opening_frame_idx].shape[1],
                        height=self.depth_frames[max_opening_frame_idx].shape[0],
                        intrinsics=intrinsics,
                        extrinsics=max_open_extrinsics,
                    )
                    .as_tensor()
                    .cpu()
                    .numpy()
                )

                envelope_points_2d = np.flip(np.array(static_parent_depth.nonzero())[:2, :].T, 1)

                if len(envelope_points_2d) == 0:
                    loguru.logger.warning(f"No envelope points projected for segment {i}, skipping envelope mask generation.")
                    continue
                # Create hull from envelope points
                hull = ConvexHull(envelope_points_2d)
                roi_mask = np.zeros_like(self.depth_frames[max_opening_frame_idx], dtype=np.uint8)
                if hull is not None and len(hull.vertices) >= 3:
                    cv2.fillConvexPoly(roi_mask, envelope_points_2d[hull.vertices].astype(np.int32), 1)
                else:
                    loguru.logger.info("cannot compute convex hull")
                    for pt in envelope_points_2d.astype(np.int32):
                        x, y = pt
                        if 0 <= x < roi_mask.shape[1] and 0 <= y < roi_mask.shape[0]:
                            roi_mask[y, x] = 1
                cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_envelope_mask.png"), roi_mask * 255)

                # Create hull from max open parent points
                max_open_points_2d = np.flip(np.array(max_open_parent_depth.nonzero())[:2, :].T, 1)
                max_open_hull = ConvexHull(max_open_points_2d)
                max_open_parent_mask = np.zeros_like(self.depth_frames[max_opening_frame_idx], dtype=np.uint8)
                if max_open_hull is not None and len(max_open_hull.vertices) >= 3:
                    cv2.fillConvexPoly(max_open_parent_mask, max_open_points_2d[max_open_hull.vertices].astype(np.int32), 1)
                else:
                    loguru.logger.info("cannot compute convex hull for max open parent")
                    for pt in max_open_points_2d.astype(np.int32):
                        x, y = pt
                        if 0 <= x < max_open_parent_mask.shape[1] and 0 <= y < max_open_parent_mask.shape[0]:
                            max_open_parent_mask[y, x] = 1
                cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_max_open_parent_mask.png"), max_open_parent_mask * 255)

                roi_mask[max_open_parent_mask] = False
            else:
                raise NotImplementedError(f"Unknown model type: {model.type}")

            img_out = plot_axis_img(
                self.rgb_frames[max_opening_frame_idx],
                self.depth_frames[max_opening_frame_idx],
                np.array(model.position),
                np.array(model.axis),
                self.dataset.depth_intrinsics,
                np.linalg.inv(self.cam_poses[max_opening_frame_idx]),
                max_open_parent_mask,
            )
            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_axis_mask_proj.png"), img_out)

            # erode roi mask by 10 pixels to account for potential projection errors and ensure we only get
            # children that are clearly separate from the parent
            roi_mask = cv2.erode(roi_mask.astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1) > 0

            model.roi_mask = roi_mask
            if roi_mask.sum() == 0:
                loguru.logger.warning(f"Empty ROI mask for segment {i}, skipping child detection")
                continue

            # visualize roi mask on rgb frame
            children_roi_rgb = copy.copy(self.rgb_frames[max_opening_frame_idx])
            children_roi_rgb[model.roi_mask > 0, 2] = 255
            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_obj_{model.obj_id:04d}_max_open.png"), to_bgr(children_roi_rgb))

            # retrieve rgbd frame at max opening and do one MobileSAM forward pass
            pot_child_masks = self.seg.msam.segment_everything(self.rgb_frames[max_opening_frame_idx].astype(np.uint8))
            pot_child_gobs = parse_masks(pot_child_masks, self.rgb_frames, max_opening_frame_idx)

            _, pot_child_clip_feats = compute_clip_features_batched(
                self.rgb_frames[max_opening_frame_idx],
                pot_child_gobs,
                self.clip_model,
                self.clip_preprocess,
                self.clip_tokenizer,
                "cuda" if self.cfg.device else "cpu",
            )

            pot_child_viz = self.seg.msam.visualize_segmentation(
                self.rgb_frames[max_opening_frame_idx], pot_child_gobs['mask'], pot_child_gobs["score"]
            )
            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_obj_{model.obj_id:04d}_pot_children.png"), to_bgr(pot_child_viz))

            # extract all masks that have at least containment wrt to the roi_mask above threshold
            roi_containment_scores = np.array(
                [self.seg.msam.compute_containment(roi_mask, pot_child_gobs['mask'][mask_idx]) for mask_idx in range(len(pot_child_gobs['xyxy']))]
            )
            refined_roi_mask = np.zeros_like(roi_mask, dtype=bool)
            for mask_idx in range(len(pot_child_gobs['xyxy'])):
                if roi_containment_scores[mask_idx] > 0.8:
                    refined_roi_mask[pot_child_gobs['mask'][mask_idx]] = True

            roi_plot = copy.deepcopy(self.rgb_frames[max_opening_frame_idx])
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            roi_plot[refined_roi_mask] = roi_plot[refined_roi_mask] * 0.5 + color * 0.5

            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_obj_{model.obj_id:04d}_refined_roi.png"), to_bgr(roi_plot))

            model.roi_mask = refined_roi_mask

            # fill holes in refined roi mask to get uniform coverage
            refined_roi_mask = ndimage.binary_fill_holes(refined_roi_mask).astype(bool)
            refined_roi_rgb = copy.copy(self.rgb_frames[max_opening_frame_idx])
            refined_roi_rgb[refined_roi_mask > 0, 2] = 255
            cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_obj_{model.obj_id:04d}_refined_roi.png"), to_bgr(refined_roi_rgb))

            valid_children = []
            for mask_idx in range(len(pot_child_gobs['xyxy'])):
                if self.seg.msam.compute_iou(refined_roi_mask, pot_child_gobs['mask'][mask_idx]) > 0.9:
                    # this mask is basically the same as the refined roi mask, likely corresponds to the parent object, skip it
                    continue
                # reject mask with too small size
                if pot_child_gobs['mask'][mask_idx].sum() < 200:
                    continue
                roi_iou = self.seg.msam.compute_iou(refined_roi_mask, pot_child_gobs['mask'][mask_idx])
                roi_containment_score = self.seg.msam.compute_containment(refined_roi_mask, pot_child_gobs['mask'][mask_idx])

                max_open_parent_mask_minus_prior = copy.deepcopy(max_open_parent_mask)
                max_open_parent_mask_minus_prior[self.prior_masks[max_opening_frame_idx]] = 0
                articulated_part_iou = self.seg.msam.compute_iou(max_open_parent_mask_minus_prior, pot_child_gobs['mask'][mask_idx])
                articulated_containment_score_pred = self.seg.msam.compute_containment(
                    max_open_parent_mask_minus_prior, pot_child_gobs['mask'][mask_idx]
                )
                articulated_containment_score_gt = self.seg.msam.compute_containment(
                    pot_child_gobs['mask'][mask_idx], max_open_parent_mask_minus_prior
                )

                # we aim to reject a child if it its mask covers the full articulated object (front + inside)
                full_object_mask = copy.deepcopy(refined_roi_mask)
                full_object_mask[max_open_parent_mask > 0] = True
                full_object_iou = self.seg.msam.compute_iou(full_object_mask, pot_child_gobs['mask'][mask_idx])
                full_containment_score_gt = self.seg.msam.compute_containment(pot_child_gobs['mask'][mask_idx], full_object_mask)

                is_child, relation = False, None
                if full_object_iou > 0.9 and full_containment_score_gt > 0.9:
                    is_child = False
                elif roi_iou > 0.0 and roi_iou < 0.8 and roi_containment_score > 0.7:
                    is_child = True
                    relation = "STATIC" if model.type == "revolute" else "ARTICULATED"
                    iou, containment = roi_iou, roi_containment_score
                elif (
                    articulated_part_iou > 0.0
                    and articulated_part_iou < 0.8
                    and articulated_containment_score_pred > 0.7
                    and articulated_containment_score_gt < 0.6
                ):
                    # low gt containment score ensures that children do not cover most of the parent
                    # high pred containment score ensures that children do mostly lie within the articulated parent
                    is_child = True
                    relation = "ARTICULATED"
                    iou, containment = (articulated_part_iou, articulated_containment_score_pred)

                if is_child:
                    valid_children.append(mask_idx)
                    child_camera_cloud, mean_child_depth = create_object_pcd(
                        self.depth_frames[max_opening_frame_idx],
                        pot_child_gobs['mask'][mask_idx],
                        self.dataset.depth_intrinsics,
                        self.rgb_frames[max_opening_frame_idx],
                    )
                    child_global_cloud = process_pcd(child_camera_cloud.transform(self.cam_poses[max_opening_frame_idx]), self.cfg.mapping)

                    child_obj = {
                        "mask": pot_child_gobs['mask'][mask_idx],
                        "xyxy": pot_child_gobs['xyxy'][mask_idx],
                        "conf": pot_child_gobs['score'][mask_idx],
                        "frame_idx": [max_opening_frame_idx],
                        "clip_ft": torch.from_numpy(pot_child_clip_feats[mask_idx]),
                        "pcd": child_global_cloud,
                        "color_path": [self.dataset.data_list[max_opening_frame_idx][0]],
                        "num_detections": 1,
                        "n_points": [len(child_global_cloud.points)],
                        "pixel_area": [pot_child_gobs['mask'][mask_idx].sum()],
                        'contain_number': [None],  # This will be computed later
                        "inst_color": np.random.rand(3),  # A random color used for this segment instance
                        'bbox': get_bounding_box(self.cfg.mapping, child_global_cloud),
                        'mean_depth': mean_child_depth,
                        'relation': relation,
                    }
                    hierarchy.add_child(child_obj, parent_id=model.obj_id)

                    loguru.logger.info(
                        f"Found {relation.lower()} child mask with containtment score {containment:.2f} and  score {1 - iou:.2f} for segment {i}"
                    )

            if len(valid_children) > 0:
                valid_child_viz = self.seg.msam.visualize_segmentation(
                    self.rgb_frames[max_opening_frame_idx],
                    np.stack([pot_child_gobs['mask'][i] for i in valid_children]),
                    np.stack([pot_child_gobs['score'][i] for i in valid_children]),
                )
                cv2.imwrite(
                    os.path.join(save_dir, "articulation", f"segment_{i:04d}_obj_{model.obj_id:04d}_valid_children.png"), to_bgr(valid_child_viz)
                )
            else:
                loguru.logger.info(f"--> No valid children found for segment {i}.")

        return seg_articulations, hierarchy

    def save_articulations(self, seg_articulations, save_dir):
        """Save each segment's articulation model as a JSON file under ``save_dir/articulation/``."""
        os.makedirs(os.path.join(save_dir, "articulation"), exist_ok=True)
        for _, model in seg_articulations.items():
            model.save(os.path.join(save_dir, "articulation", f"segment_{model.id:04d}_model.json"))
        loguru.logger.info(f"Articulation models saved to {save_dir}/articulation/")

    def load_articulations(self, save_dir):
        """
        Load previously-saved per-segment articulation models from ``save_dir/articulation/``.

        Args:
            save_dir: Directory containing ``*_model.json`` files written by ``save_articulations``.

        Returns:
            dict: Mapping from segment index to its loaded ``InferencePointAxis`` articulation model.
        """
        from moma_sg.articulation.point_estimator import InferencePointAxis

        seg_articulations = dict()
        articulation_dir = os.path.join(save_dir, "articulation")
        model_files = [f for f in os.listdir(articulation_dir) if f.endswith("_model.json")]
        for model_file in model_files:
            model = InferencePointAxis.load(os.path.join(articulation_dir, model_file))
            seg_articulations[model.id] = model
        return seg_articulations

    def run(self):
        """
        Execute the full MoMaSG pipeline end-to-end on the loaded sequence.

        Stages, in order: (1) compute human/interaction prior masks and depth-disparity
        scores and parse interaction segments; (2) extract static keyframes and segment
        them, then build the 3D object map/hierarchy; (3) track 2D/3D keypoints through
        each interaction segment (optionally bidirectionally) and transform tracks to
        world coordinates; (4) estimate an articulation model per segment; (5) filter
        tracks for twist consistency and match segments to map objects; (6) discover
        child parts within each matched articulated object. Intermediate results are
        cached/loaded to/from disk according to ``self.cfg.cache``, and several early-return
        checkpoints (``return_after_mapping``, ``return_after_tracking``,
        ``return_after_articulation``, ``return_after_assoc``) allow stopping early.
        """
        loguru.logger.info(f"Configuration: \n{OmegaConf.to_yaml(self.cfg)}")

        scene_name = Path(self.cfg.dataset.root_path).stem
        scene_type = Path(self.cfg.dataset.root_path).parent.name
        loguru.logger.info(f"Processing scene: {scene_name} of type {scene_type}")

        # Create results directory if needed
        if self.cfg.cache.save_results:
            save_dir = os.path.join(self.cfg.cache.output_dir, scene_type, scene_name)
            os.makedirs(save_dir, exist_ok=True)
            loguru.logger.info(f"Results saving directory: {save_dir}")
            # write metadata file of all configs
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                metadata = {"scene_name": scene_name, "scene_type": scene_type, "config": OmegaConf.to_container(self.cfg)}
                json.dump(metadata, f, indent=4)

        # Set the directory for loading intermediate results
        if self.cfg.cache.load_results:
            load_dir = os.path.join(self.cfg.cache.load_dir, scene_type, scene_name)

        # Extract interaction prior
        if self.cfg.interaction.mode == "ego-centric":
            self.prior_masks = self.get_human_masks(use_precomputed=True, save_masks=self.cfg.cache.save_prior_masks, path=save_dir)
            self.prior_scores = compute_egocentric_scores(self.cfg, self.prior_masks, self.depth_frames)
        elif self.cfg.interaction.mode in ["exo-centric", "robot-centric"]:
            self.prior_masks = self.get_human_masks(use_precomputed=True, save_masks=self.cfg.cache.save_prior_masks, path=save_dir)
            self.prior_scores = compute_exocentric_scores(self.cfg, self.prior_masks, self.depth_frames)
        else:
            raise NotImplementedError("Robot-centric interaction mode not implemented yet.")

        # Extract warped depth disparity
        if (
            self.cfg.cache.load_depth_disp
            and os.path.exists(os.path.join(load_dir, "warp_scores.npz"))
            and os.path.exists(os.path.join(load_dir, "disparities.npz"))
        ):
            self.warp_scores = np.load(os.path.join(load_dir, "warp_scores.npz"), allow_pickle=True)['warp_scores']
            loguru.logger.info(f"Warp scores loaded from {load_dir}/warp_scores.npz")
            self.disparities = np.load(os.path.join(load_dir, "disparities.npz"), allow_pickle=True)['disparities']
            loguru.logger.info(f"Disparities loaded from {load_dir}/disparities.npz")
        else:
            self.warp_scores, self.disparities = compute_depth_disparity(
                self.cfg,
                self.dataset,
                self.prior_masks,
                self.depth_frames,
                self.cfg.interaction.depth_warp_interval,
                self.cfg.interaction.depth_disp_thresh,
            )
            loguru.logger.info("Computed warp scores and disparities.")
        if self.cfg.cache.save_results and not self.cfg.cache.load_depth_disp:
            np.savez_compressed(os.path.join(save_dir, "warp_scores.npz"), warp_scores=self.warp_scores)
            loguru.logger.info(f"Warp scores saved to {save_dir}/warp_scores.npz")
            np.savez_compressed(os.path.join(save_dir, "disparities.npz"), disparities=self.disparities)
            loguru.logger.info(f"Disparities saved to {save_dir}/disparities.npz")

        # Filter both interaction signals
        filt_warp_scores, filt_prior_scores = filter_interaction_signals(
            warp_scores=self.warp_scores,
            prior_scores=self.prior_scores,
            warp_filt_ksize=self.cfg.interaction.warp_filt_ksize,
            prior_filt_ksize=self.cfg.interaction.prior_filt_ksize,
            filter_type="median",
        )

        # Parse ground truth interaction segments
        self.gt_interaction_segments = np.zeros(len(self.prior_masks), dtype=bool)
        self.gt_interaction_segments[self.dataset.interaction_timestamps] = True

        if self.cfg.interaction.use_gt_segments:
            pred_segments = [(s[1][0], s[1][1]) for s in self.dataset.interactions]
            pred_segments_dense = self.gt_interaction_segments
            loguru.logger.info("Using ground truth interaction segments.")
        else:
            hidden_states, interaction_prob = run_interaction_model(self.cfg, filt_warp_scores, filt_prior_scores)
            pred_segments, pred_segments_dense = parse_segments(self.cfg, boolean_scores=hidden_states)

            if save_dir is not None:
                save_interaction_segments(pred_segments, save_dir)
                if self.cfg.vis.plot_interaction_seg and self.gt_interaction_segments is not None:
                    plot_interaction_segmentation(
                        warp_scores=filt_warp_scores,
                        prior_scores=filt_prior_scores,
                        interaction_prob=interaction_prob if self.cfg.interaction.model == "cond-prob" else None,
                        pred_segments=pred_segments,
                        gt_segments=self.gt_interaction_segments,
                        save_dir=save_dir,
                    )

        if save_dir is not None:
            save_interaction_segments(pred_segments, save_dir)

        # Keyframe extraction based on interaction segments and camera motion
        self.keyframes = extract_static_keyframes(self.cam_poses, self.rgb_frames, self.prior_scores, pred_segments, pred_segments_dense)

        if self.cfg.mapping.source == "conceptgraphs":
            conceptgraphs_dir = os.path.join(self.cfg.mapping.conceptgraphs_root, scene_type, scene_name)
            loguru.logger.info(f"Loading pre-merged 3D object map from ConceptGraphs: {conceptgraphs_dir}")
            objects = load_conceptgraphs_map(self.cfg, conceptgraphs_dir)
        else:
            self.keyframe_masks = self.segment_keyframes(load_dir=load_dir)

            if self.cfg.cache.load_map and os.path.exists(os.path.join(save_dir, "objects_optim.pkl.gz")):
                with gzip.open(os.path.join(save_dir, "objects_optim.pkl.gz"), 'rb') as f:
                    result = pickle.load(f)
                objects = MapObjectList()
                objects.load_serializable(result['objects'])
            else:
                objects = produce_map(
                    self.cfg,
                    self.keyframes,
                    self.keyframe_masks,
                    self.rgb_frames,
                    self.depth_frames,
                    self.cam_poses,
                    self.dataset,
                    self.clip_model,
                    self.clip_preprocess,
                    self.clip_tokenizer,
                    save_dir,
                )
        self.hierarchy = Hierarchy(self.cfg, objects)

        if self.cfg.cache.return_after_mapping:
            if self.cfg.cache.save_hierarchy:
                self.hierarchy.save(os.path.join(save_dir, "hierarchy.pkl"))
                loguru.logger.info(f"Hierarchy saved to {save_dir}/hierarchy.pkl")
            return

        torch.cuda.empty_cache()

        # Visualize hand action segments
        if self.cfg.vis.display_hand_segments and not self.cfg.cache.load_results:
            self.seg.play_hand_action_segments(self.rgb_frames, pred_segments)

        # Track queries and project to 3D
        if self.cfg.cache.load_raw_tracks:
            # Load tracks and visibility from file
            npz_path = os.path.join(load_dir, "tracks_3d_tap.npz")
            if os.path.exists(npz_path):
                loaded_data = np.load(npz_path, allow_pickle=True)
                forw_3d_tracks_segments = loaded_data["tracks"]
                forw_vis_segments = loaded_data["visibility"]
                if self.cfg.tracking.bidir and loaded_data["recon_tracks"] is not None and loaded_data["recon_visibility"] is not None:
                    backw_3d_tracks_segments, backw_vis_segments = list(), list()
                    for backw_tracks, backw_visibility in zip(loaded_data["recon_tracks"], loaded_data["recon_visibility"]):
                        backw_3d_tracks_segments.append(list(backw_tracks) if backw_tracks is not None else None)
                        backw_vis_segments.append(list(backw_visibility) if backw_visibility is not None else None)
                else:
                    backw_3d_tracks_segments, backw_vis_segments = None, None
                loguru.logger.info("Loaded 3D tracks and visibility after tracking")
            else:
                raise FileNotFoundError(f"Could not find {npz_path}")
        else:
            forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments = self.track.track_and_project_queries(
                pred_segments,
                self.prior_masks,
                self.cfg.tracking.bidir,
                save_dir,
                self.rgb_frames,
                self.depth_frames,
            )

            # Save tracks and visibility after tracking
            if self.cfg.cache.save_results:
                np.savez(
                    os.path.join(save_dir, "tracks_3d_tap.npz"),
                    tracks=np.array(forw_3d_tracks_segments, dtype=object),
                    visibility=np.array(forw_vis_segments, dtype=object),
                    backw_tracks=np.array(backw_3d_tracks_segments, dtype=object) if backw_3d_tracks_segments is not None else None,
                    backw_visibility=np.array(backw_vis_segments, dtype=object) if backw_vis_segments is not None else None,
                )

        # Transform point tracks to world coordinates
        forw_3d_tracks_segments = self.compensate_cam_motion(pred_segments, forw_3d_tracks_segments)
        backw_3d_tracks_segments = (
            self.compensate_cam_motion(pred_segments, backw_3d_tracks_segments)
            if self.cfg.tracking.bidir and backw_3d_tracks_segments is not None
            else None
        )

        # Save tracks after transforming to world coordinates
        if self.cfg.cache.save_results:
            np.savez(
                os.path.join(save_dir, "tracks_world_tap.npz"),
                tracks=np.array(forw_3d_tracks_segments, dtype=object),
                visibility=np.array(forw_vis_segments, dtype=object),
                backw_tracks=np.array(backw_3d_tracks_segments, dtype=object) if backw_3d_tracks_segments is not None else None,
                backw_visibility=np.array(backw_vis_segments, dtype=object) if backw_vis_segments is not None else None,
            )

        # Store original tracks for later object matching/reconstruction
        forw_3d_tracks_before = copy.deepcopy(forw_3d_tracks_segments)
        forw_vis_before = copy.deepcopy(forw_vis_segments)
        if self.cfg.tracking.bidir and backw_3d_tracks_segments:
            backw_3d_tracks_before = copy.deepcopy(backw_3d_tracks_segments)
            backw_vis_before = copy.deepcopy(backw_vis_segments)

        if self.cfg.cache.return_after_tracking:
            return

        if not self.cfg.cache.load_articulations:
            # Apply the same filtering to recon tracks (one pass per keyframe per segment).
            # NaN positions/visibility (frames before the keyframe) are set to 0 so the
            # filter operates only on the frames that were actually tracked.
            if self.cfg.tracking.bidir and backw_3d_tracks_segments is not None:
                for seg_idx, (seg_recon, seg_recon_vis) in enumerate(zip(backw_3d_tracks_segments, backw_vis_segments)):
                    if seg_recon is None:
                        continue
                    # concatenate all keyframe passes along the track axis → (T, N_total, 3) / (T, N_total)
                    combined_t = np.concatenate([np.array(r) for r in seg_recon], axis=1)
                    combined_v = np.concatenate([np.array(v) for v in seg_recon_vis], axis=1)
                    # store back as a single combined array (replaces the per-pass list)
                    backw_3d_tracks_segments[seg_idx] = combined_t
                    backw_vis_segments[seg_idx] = combined_v

            (forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments) = self.traj_filter.forward(
                pred_segments, self.cam_poses, forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments
            )

            # Save tracks after smoothing
            if self.cfg.cache.save_results:
                np.savez(
                    os.path.join(save_dir, "tracks_filtered.npz"),
                    tracks=np.array(forw_3d_tracks_segments, dtype=object),
                    visibility=np.array(forw_vis_segments, dtype=object),
                    backw_tracks=np.array(backw_3d_tracks_segments, dtype=object) if backw_3d_tracks_segments is not None else None,
                    backw_visibility=np.array(backw_vis_segments, dtype=object) if backw_vis_segments is not None else None,
                )

            # Visualize 3D tracks for each segment
            if self.cfg.vis.show_3d_tracks:
                for i, (start_idx, end_idx) in enumerate(pred_segments):
                    plot_3d_tracks(
                        forw_3d_tracks_segments[i],
                        forw_vis_segments[i],
                        self.rgb_frames[start_idx:end_idx],
                        self.depth_frames[start_idx:end_idx],
                        dataset=self.dataset,
                        tracks_leave_trace=self.cfg.vis.tracks_leave_trace,
                        camera_poses=self.cam_poses[start_idx:end_idx],
                        save_frames=True,
                        save_video=True,
                        output_dir=f"3d_video/{i}/",
                    )

            # Perform articulation estimation
            self.segment_articulations = self.estimate_articulations(
                pred_segments, forw_3d_tracks_segments, forw_vis_segments, backw_3d_tracks_segments, backw_vis_segments, save_dir
            )
            if self.cfg.cache.save_articulations:
                self.save_articulations(self.segment_articulations, save_dir)
        else:
            # Load articulation models from file
            self.segment_articulations = self.load_articulations(load_dir)
            loguru.logger.info(f"Loaded articulation models from {load_dir}/articulations/")

        if self.cfg.cache.return_after_articulation:
            if self.cfg.cache.save_hierarchy:
                self.hierarchy.save(os.path.join(save_dir, "hierarchy.pkl"))
                loguru.logger.info(f"Hierarchy saved to {save_dir}/hierarchy.pkl")
            return

        # Merge forward 3D point tracks with backwards tracks if available
        if self.cfg.tracking.bidir and backw_3d_tracks_segments is not None:
            tracks_3d_segments = [None] * len(backw_3d_tracks_segments)
            vis_segments = [None] * len(backw_vis_segments)
            for i in range(len(backw_3d_tracks_segments)):
                tracks_3d_segments[i] = [forw_3d_tracks_before[i]] + backw_3d_tracks_before[i]
                vis_segments[i] = [forw_vis_before[i].astype(np.float32)] + backw_vis_before[i]
            # cat arrays for each segment
            for p in range(len(pred_segments)):
                tracks_3d_segments[p] = np.concatenate(tracks_3d_segments[p], axis=1)
                vis_segments[p] = np.concatenate(vis_segments[p], axis=1)

        # Filter all tracks regarding their consistency under the estimated articulation models
        inlier_masks, inlier_tracks, inlier_vis = filter_tracks_by_twist(
            self.cfg,
            pred_segments,
            self.segment_articulations,
            tracks_3d_segments if backw_3d_tracks_segments is not None else forw_3d_tracks_before,
            vis_segments if backw_vis_segments is not None else forw_vis_before,
            estimation_tracks=forw_3d_tracks_segments,
            estimation_vis=forw_vis_segments,
        )
        loguru.logger.info(f"Number of inlier tracks after twist filtering: {[in_t.shape[1] for in_t in inlier_tracks]}")

        # Match map objects against articulations and optionally execute pixel-wise object reconstruction
        self.segment_articulations, self.hierarchy = self.match_tracks_objects(
            pred_segments,
            self.hierarchy,
            self.segment_articulations,
            tracks_3d_segments if backw_3d_tracks_segments is not None else forw_3d_tracks_before,
            vis_segments if backw_vis_segments is not None else forw_vis_before,
            inlier_tracks,
            inlier_vis,
            inlier_masks,
            save_dir,
            visualize=self.cfg.vis.plot_assoc,
        )

        if self.cfg.cache.return_after_assoc:
            if self.cfg.cache.save_hierarchy:
                self.hierarchy.save(os.path.join(save_dir, "hierarchy_wo_children.pkl"))
                loguru.logger.info(f"Hierarchy saved to {save_dir}/hierarchy_wo_children.pkl")
            return

        # Cast articulation volume and obtain children nodes within that volume
        self.segment_articulations, self.hierarchy = self.discover_children(pred_segments, self.hierarchy, self.segment_articulations, save_dir)

        if self.cfg.cache.save_hierarchy:
            self.hierarchy.save(os.path.join(save_dir, "hierarchy.pkl"))
            loguru.logger.info(f"Hierarchy saved to {save_dir}/hierarchy.pkl")


@hydra.main(version_base=None, config_path="../../configs", config_name="momasg")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: build a MoMaSG instance from the resolved config and run the pipeline."""
    moma_sg = MoMaSG(cfg)
    moma_sg.run()


if __name__ == "__main__":
    main()
