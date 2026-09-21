import glob
import gzip
import os
from pathlib import Path
import pickle

import loguru
from moma_sg.utils.geometry import estimate_motion_blur, get_rotation_magnitude
from moma_sg.utils.mapping import (
    DetectionList,
    MapObjectList,
    aggregate_similarities,
    compute_clip_features_batched,
    compute_semantic_similarities,
    compute_spatial_similarities,
    create_object_pcd,
    denoise_objects,
    filter_objects,
    get_bounding_box,
    merge_detections_to_objects,
    merge_objects,
    parse_masks,
    prepare_objects_save_vis,
    process_pcd,
)
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def extract_static_keyframes(cam_poses, rgb_frames, prior_scores, pred_segments, pred_segments_arr):
    """
    Keyframe extraction based on interaction segments and camera motion.

    Walks the frame sequence and, outside of interaction segments, whenever the camera
    has moved enough since the last keyframe (rotation > 0.1 rad or translation > 0.15 m),
    selects a new keyframe. Prefers the current frame if it is sharp enough (motion-blur
    variance > 40); otherwise looks up to 10 frames back for a sharp, non-prior frame.

    :param cam_poses: List/array of (4,4) camera-to-world pose matrices, one per frame.
    :param rgb_frames: List of RGB frames.
    :param prior_scores: Per-frame prior/interaction score used to reject blurry fallback frames.
    :param pred_segments: List of (start, end) frame-index tuples for interaction segments.
    :param pred_segments_arr: Boolean array, one entry per frame, True inside an interaction segment.
    :return: List of selected keyframe frame indices.
    """

    keyframes = []
    interval_frames = []
    for start, end in pred_segments:
        interval_frames.extend([i for i in range(start - 1, end + 1)])
    prev_pose = None
    dist = 0.0
    rot_mag = 0.0
    for frame_idx, rgb_frame in tqdm(enumerate(rgb_frames), desc="Extracting keyframes", total=len(rgb_frames)):
        if frame_idx == 0:
            prev_keyframe_pose = cam_poses[frame_idx]
        else:
            dist = np.linalg.norm(np.array(cam_poses[frame_idx][:3, 3]) - np.array(prev_keyframe_pose[:3, 3]))
            T = np.linalg.inv(prev_keyframe_pose) @ cam_poses[frame_idx]
            rot_mag = get_rotation_magnitude(T)

        # check if the current frame is an interaction segment
        if rot_mag > 0.1 or dist > 0.15:
            # extract keyframe
            if pred_segments_arr[frame_idx] == False:
                blur_var = estimate_motion_blur(rgb_frames[frame_idx])
                if blur_var > 40:
                    # create keyframe
                    keyframes.append(frame_idx)
                    prev_keyframe_pose = cam_poses[frame_idx]
                else:
                    # go at max 10 frames back and find a non-blurry keyframe
                    for i in range(1, 10):
                        if frame_idx - i < 0:
                            break
                        blur_var = estimate_motion_blur(rgb_frames[frame_idx - i])
                        if blur_var > 40 and frame_idx - i not in keyframes and prior_scores[frame_idx] == 0:
                            # create keyframe
                            keyframes.append(frame_idx - i)
                            prev_keyframe_pose = cam_poses[frame_idx - i]
                            break
    return keyframes


def load_conceptgraphs_map(cfg, conceptgraphs_dir):
    """
    Load an already-merged 3D object map produced by ConceptGraphs instead of building
    one from moma-sg's own keyframe segmentation + fusion pipeline.

    Expects per-object point clouds at ``<conceptgraphs_dir>/objects/object_<idx>.ply``.
    If ``<conceptgraphs_dir>/pcd_saves/*.pkl.gz`` (a ConceptGraphs results pickle, preferring
    one whose name contains "_post") is also present, its per-object CLIP features and
    detection metadata are attached to the matching ``object_<idx>.ply``; otherwise a
    zero CLIP feature is used so the rest of the pipeline (which requires a 'clip_ft' key
    per object) still runs.

    :param cfg: Full config object (uses ``cfg.mapping``).
    :param conceptgraphs_dir: ConceptGraphs scene directory, e.g.
        ``.../output_conceptgraphs/arti4d/din080/scene_2025-04-11-11-44-32``.
    :return: MapObjectList of the loaded objects.
    """
    objects_dir = os.path.join(conceptgraphs_dir, "objects")
    ply_files = sorted(glob.glob(os.path.join(objects_dir, "object_*.ply")), key=lambda p: int(Path(p).stem.split("_")[-1]))
    if len(ply_files) == 0:
        raise FileNotFoundError(f"No object_*.ply files found in {objects_dir}")

    pkl_candidates = sorted(glob.glob(os.path.join(conceptgraphs_dir, "pcd_saves", "*.pkl.gz")))
    pkl_objects = None
    if pkl_candidates:
        post_candidates = [p for p in pkl_candidates if "_post" in Path(p).stem]
        pkl_path = post_candidates[-1] if post_candidates else pkl_candidates[-1]
        with gzip.open(pkl_path, "rb") as f:
            pkl_objects = pickle.load(f)["objects"]
        loguru.logger.info(f"Loaded ConceptGraphs object metadata (CLIP features etc.) from {pkl_path}")
        if len(pkl_objects) != len(ply_files):
            loguru.logger.warning(
                f"ConceptGraphs pickle has {len(pkl_objects)} objects but {len(ply_files)} object_*.ply files were "
                f"found in {objects_dir}; indices beyond the pickle will get a zero-filled CLIP feature."
            )
    else:
        loguru.logger.warning(f"No ConceptGraphs pickle found under {conceptgraphs_dir}/pcd_saves; CLIP features will be zero-filled.")

    objects = MapObjectList()
    for ply_path in tqdm(ply_files, desc="Loading ConceptGraphs objects"):
        obj_idx = int(Path(ply_path).stem.split("_")[-1])
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) < cfg.mapping.obj_min_points:
            continue

        pkl_obj = pkl_objects[obj_idx] if pkl_objects is not None and obj_idx < len(pkl_objects) else None

        pcd_bbox = get_bounding_box(cfg.mapping, pcd)
        pcd_bbox.color = [0, 1, 0]

        objects.append(
            {
                "image_idx": pkl_obj["image_idx"] if pkl_obj is not None else [],
                "mask_idx": pkl_obj["mask_idx"] if pkl_obj is not None else [],
                "color_path": pkl_obj["color_path"] if pkl_obj is not None else [],
                "num_detections": pkl_obj["num_detections"] if pkl_obj is not None else 1,
                "conf": pkl_obj["conf"] if pkl_obj is not None else [1.0],
                "n_points": [len(pcd.points)],
                "pixel_area": pkl_obj["pixel_area"] if pkl_obj is not None else [0],
                "contain_number": [None],
                "inst_color": np.random.rand(3),
                "pcd": pcd,
                "points": np.asarray(pcd.points),
                "color": np.asarray(pcd.colors),
                "bbox": pcd_bbox,
                "clip_ft": torch.from_numpy(np.asarray(pkl_obj["clip_ft"])) if pkl_obj is not None else torch.zeros(1024),
                "mean_depth": 0.0,
                "last_observed": 0,
                "conceptgraphs_idx": obj_idx,
            }
        )

    loguru.logger.info(f"Loaded {len(objects)} ConceptGraphs objects from {objects_dir}")
    return objects


def produce_map(
    cfg,
    keyframes,
    keyframe_masks,
    rgb_frames,
    depth_frames,
    cam_poses,
    dataset,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    save_dir,
):
    """
    Build the 3D object map by incrementally fusing per-keyframe segmentation masks
    into 3D point clouds and merging them into a running set of tracked objects.

    For each keyframe, converts each valid, non-border mask into a colored point cloud
    (using depth + camera pose), wraps it as a detection, and either seeds the object
    list (if empty) or merges it into existing objects using spatial/semantic similarity
    (per ``cfg.mapping.match_strategy``). Periodically runs denoising, filtering, and
    merging post-processing passes on the object list. Optionally saves a combined
    point cloud and the objects (with CLIP features etc.) to disk.

    :param cfg: Full config object (uses ``cfg.mapping``, ``cfg.device``, ``cfg.cache``, ``cfg.dataset``).
    :param keyframes: List of keyframe frame indices to process, in order.
    :param keyframe_masks: Per-keyframe segmentation data consumed by ``parse_masks``.
    :param rgb_frames: List of RGB frames indexed by frame index.
    :param depth_frames: List of depth frames indexed by frame index.
    :param cam_poses: List/array of (4,4) camera-to-world pose matrices, indexed by frame index.
    :param dataset: Dataset object providing ``depth_intrinsics`` and ``data_list`` (RGB paths).
    :param clip_model: CLIP model used to compute per-mask semantic features.
    :param clip_preprocess: CLIP image preprocessing transform.
    :param clip_tokenizer: CLIP tokenizer.
    :param save_dir: Directory to write the combined point cloud and pickled results to
        when ``cfg.cache.save_results`` is set.
    :return: The final ``MapObjectList`` of merged, denoised, filtered 3D objects.
    """
    objects = MapObjectList()
    pbar = tqdm(keyframes, total=len(keyframes), desc="Mapping: Keyframe ####, 0000 objects")
    for mapping_idx, idx in enumerate(pbar):
        # update tqdm description
        pbar.set_description(f"Mapping: Keyframe {idx}, {len(objects):04d} objects")

        detection_list = DetectionList()
        gobs = parse_masks(keyframe_masks[keyframes.index(idx)], rgb_frames, idx)

        image_crops, image_feats = compute_clip_features_batched(
            rgb_frames[idx],
            gobs,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
            "cuda" if cfg.device else "cpu",
        )

        for mask_idx in range(len(gobs["xyxy"])):
            mask = gobs["mask"][mask_idx]

            isolated = True
            if mask[0, :].sum() > 0 or mask[-1, :].sum() > 0 or mask[:, 0].sum() > 0 or mask[:, -1].sum() > 0:
                isolated = False

            # check if the mask touches the image border
            if cfg.mapping.remove_bordering_masks:
                if isolated == False:
                    continue

            # make the pcd and color it
            camera_object_pcd, mean_depth = create_object_pcd(
                depth_frames[idx],
                mask,
                dataset.depth_intrinsics,
                rgb_frames[idx],
                obj_color=None,
            )
            if len(camera_object_pcd.points) < cfg.mapping.obj_min_points:
                continue

            global_object_pcd = camera_object_pcd.transform(cam_poses[idx])

            # get largest cluster, filter out noise
            global_object_pcd = process_pcd(camera_object_pcd, cfg.mapping)

            pcd_bbox = get_bounding_box(cfg.mapping, global_object_pcd)
            pcd_bbox.color = [0, 1, 0]

            if pcd_bbox.volume() < 1e-6 or np.linalg.norm(cam_poses[idx][:3, 3] - global_object_pcd.get_center()) > 7.0:
                continue

            # Treat the detection in the same way as a 3D object
            # Store information that is enough to recover the detection
            detected_object = {
                "image_idx": [idx],  # idx of the image
                "mask_idx": [mask_idx],  # idx of the mask/detection
                "color_path": [dataset.data_list[idx][0]],  # path to the RGB image
                "num_detections": 1,  # number of detections of this object
                "mask": [mask],
                "xyxy": [gobs["xyxy"][mask_idx]],
                "conf": gobs["score"][mask_idx],
                "n_points": [len(global_object_pcd.points)],
                "isolated": isolated,
                "pixel_area": [mask.sum()],
                "contain_number": [None],  # This will be computed later
                "inst_color": np.random.rand(3),  # A random color used for this segment instance
                # These are for the entire 3D object
                "pcd": global_object_pcd,
                "points": np.asarray(global_object_pcd.points),
                "color": np.asarray(global_object_pcd.colors),
                "bbox": pcd_bbox,
                "clip_ft": torch.from_numpy(image_feats[mask_idx]),
                "mean_depth": mean_depth,
                "last_observed": idx,  # Last frame this object was observed
            }
            detection_list.append(detected_object)

        if len(detection_list) == 0:
            continue

        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(detection_list)):
                objects.append(detection_list[i])
            # Skip the similarity computation
            continue

        spatial_sim = compute_spatial_similarities(cfg, detection_list, objects)

        if cfg.mapping.match_strategy == "spatial":
            agg_sim = spatial_sim
            agg_sim[agg_sim < cfg.mapping.spatial_thresh] = float("-inf")
        elif cfg.mapping.match_strategy == "sim_sum":
            semantic_sim = compute_semantic_similarities(cfg, detection_list, objects)
            agg_sim = aggregate_similarities(cfg, spatial_sim, semantic_sim)
            agg_sim[agg_sim < cfg.mapping.sim_sum_thresh] = float("-inf")
        else:
            raise NotImplementedError(f"Unknown match strategy: {cfg.mapping.match_strategy}")

        objects = merge_detections_to_objects(cfg, detection_list, objects, agg_sim)

        # Perform post-processing periodically if told so
        if cfg.mapping.denoise_interval > 0 and (mapping_idx) % cfg.mapping.denoise_interval == 0:
            objects = denoise_objects(cfg, objects)
        if cfg.mapping.filter_interval > 0 and (mapping_idx) % cfg.mapping.filter_interval == 0:
            objects = filter_objects(cfg, objects, keyframe_idcs=keyframes, curr_frame_idx=idx)
        if cfg.mapping.merge_interval > 0 and (mapping_idx) % cfg.mapping.merge_interval == 0:
            objects = merge_objects(cfg, objects)

    objects = denoise_objects(cfg, objects)
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    # create a o3d point cloud from the objects
    if cfg.cache.save_results:
        # create combined point cloud
        all_instances = o3d.geometry.PointCloud()
        for detection in objects:
            bla = o3d.geometry.PointCloud()
            bla.points = o3d.utility.Vector3dVector(
                np.asarray(detection["pcd"].points) + np.random.normal(0, 0.002, size=np.asarray(detection["pcd"].points).shape)
            )
            bla.paint_uniform_color(detection["inst_color"])
            all_instances += bla
        o3d.io.write_point_cloud(os.path.join(save_dir, "cloud_partial_optim.ply"), all_instances)

        objects_to_save = prepare_objects_save_vis(objects)

        result = {
            "cfg": cfg,
            "root_path": cfg.dataset.root_path,
            "objects": objects_to_save,
        }

        with gzip.open(os.path.join(save_dir, "objects_optim.pkl.gz"), "wb") as f:
            pickle.dump(result, f)

    return objects
