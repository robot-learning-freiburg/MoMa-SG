import copy
import os
from pathlib import Path
from typing import List

import cv2
import loguru
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from ultralytics import YOLO


class Segmentation:
    """Segmentation of articulated objects using hand detection and SAM."""

    def __init__(self, params):
        """
        Initialize the ArticulatedObjectSegmentor.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.params = params
        self.device = torch.device("cuda" if self.params.get("use_cuda", True) and torch.cuda.is_available() else "cpu")

        # Initialize MobileSAM segmentation model
        sam_config = {
            "checkpoint": os.path.join(self.params.package_path, self.params.interaction.msam_checkpoint),
            "model_type": self.params.get("sam_model_type", "vit_t"),
            "device": self.device,
        }
        self.msam = MobileSAM(config=sam_config)
        self.yolo = YOLO(os.path.join(self.params.package_path, self.params.interaction.yolo_path))

        # Azure dataset for 3D processing
        self.dataset = self.params.get("dataset", None)

        # SAM2 video predictor, lazily built on first use (see reconstruct_with_sam2)
        self._sam2_predictor = None

    def _get_sam2_predictor(self):
        """Build the SAM2 video predictor on first use and cache it."""
        if self._sam2_predictor is None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.build_sam import _load_checkpoint

            import sam2

            sam2_dir = os.path.dirname(sam2.__file__)
            sam2_cfg = OmegaConf.load(os.path.join(sam2_dir, "configs/sam2.1/sam2.1_hiera_l.yaml"))
            OmegaConf.update(
                sam2_cfg,
                "model._target_",
                "sam2.sam2_video_predictor.SAM2VideoPredictor",
            )
            OmegaConf.update(sam2_cfg, "model.pred_obj_scores", True)
            OmegaConf.update(sam2_cfg, "model.pred_obj_scores_mlp", True)
            OmegaConf.update(sam2_cfg, "model.fixed_no_obj_ptr", True)

            predictor = instantiate(sam2_cfg.model, _recursive_=True).to(self.device)
            _load_checkpoint(
                predictor,
                self.params.articulation.sam2_path,
            )
            self._sam2_predictor = predictor

        return self._sam2_predictor

    def reconstruct_with_sam2(self, rgb_frames, prompt_masks, save_dir, segment_idx):
        """
        Run SAM2 video propagation over a segment's frames, seeded with per-frame
        prompt masks, and return the resulting per-frame per-object masks.

        :param rgb_frames: RGB frames for this segment (index 0 is the first
            frame of the segment); frame indices in prompt_masks and the
            returned dict are relative to this sequence.
        :param prompt_masks: dict {frame_idx: np.ndarray} of prompt masks to
            seed SAM2 with.
        :param save_dir: directory to save overlay visualizations to.
        :param segment_idx: index of the segment, used for output filenames.
        :return: dict {frame_idx: {obj_id: mask}} of the propagated masks.
        """
        if not prompt_masks:
            return {}

        import tempfile

        predictor = self._get_sam2_predictor()
        segment_len = len(rgb_frames)

        with tempfile.TemporaryDirectory() as tmpdir:
            # save segment frames as JPEGs so SAM2 can load them
            for j in range(segment_len):
                Image.fromarray(rgb_frames[j].astype(np.uint8)).save(os.path.join(tmpdir, f"{j:05d}.jpg"))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(video_path=tmpdir)
                predictor.reset_state(inference_state)

                # register each prompt mask with SAM2 via add_new_mask
                for j, frame_mask in prompt_masks.items():
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=j,
                        obj_id=1,  # SAM2 uses 1-based object IDs
                        mask=frame_mask.astype(bool),
                    )

                # propagate through the full segment
                video_segments = {}  # {frame_idx: {obj_id: mask}}
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {obj_id: (out_mask_logits[k] > 0.0).cpu().numpy() for k, obj_id in enumerate(out_obj_ids)}

                # render video with masks
                for j in range(segment_len):
                    frame = rgb_frames[j].copy()
                    if j in video_segments:
                        for obj_id, mask in video_segments[j].items():
                            colored_mask = np.zeros_like(frame)
                            colored_mask[mask.squeeze(0)] = [0, 255, 0]  # Green masks
                            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
                    cv2.imwrite(
                        os.path.join(save_dir, f"segment_{segment_idx}_frame_{j:05d}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )

                # reset inference state to free GPU memory
                predictor.reset_state(inference_state)

        return video_segments

    def compute_prior_masks(
        self,
        rgb_frames: List[np.ndarray],
        dilate_its: int,
        dilate_kernel_size: int,
    ) -> List[List[np.ndarray]]:
        """Compute human masks for each segment."""

        all_human_masks = []
        for i, img in tqdm(enumerate(rgb_frames), total=len(rgb_frames), desc="Computing human masks"):
            human_masks = self._segment_human(img)
            if human_masks:
                # Use first detected human mask and process it
                # translate mask down on the image by 10px
                translated_mask = np.zeros_like(human_masks[0])
                translated_mask[8:, :] = human_masks[0][:-8, :]

                mask = cv2.dilate(
                    translated_mask,
                    np.ones(
                        (
                            self.params.interaction.dilate_kernel_size,
                            self.params.interaction.dilate_kernel_size,
                        ),
                        np.uint8,
                    ),
                    iterations=self.params.interaction.dilate_iterations,
                )
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                mask = np.zeros((img.shape[0], img.shape[1]))
            all_human_masks.append(mask.astype(bool))

        # free GPU memory
        torch.cuda.empty_cache()
        return np.array(all_human_masks)

    def _segment_human(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Segment human regions using a YOLO model.
        """
        results = self.yolo.predict(source=Image.fromarray(img), verbose=False, device=self.device)
        indices = [i for i, cls in enumerate(results[0].boxes.cls) if cls == 0]
        masks = results[0].masks
        human_masks = [masks[i].cpu().numpy().data.transpose(1, 2, 0) for i in indices]

        # for mask in human_masks:
        #     # resize mask to first_image size
        #     mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
        #     # plt.imshow(mask, alpha=0.5)
        return human_masks

    @staticmethod
    def sample_points_from_mask(mask, prior_mask, K, eps_pixel=50):
        """
        Given an array of disparity indices (N,2), sample K random points that are
        within eps_pixel (in Euclidean distance) of the mask boundary.

        Parameters:
        hand_mask_indices (np.ndarray): (N,2) array of (row, col) coordinates of mask pixels.
        K (int): Number of points to sample.
        eps_pixel (float): Radius (in pixels) defining the vicinity of the mask.

        Returns:
        np.ndarray: (K,2) array of (row, col) coordinates of the sampled points.
        """
        assert mask.dtype == bool, "Mask should be a boolean array."

        if np.sum(mask) == 0:
            # loguru.logger.warning("Hand mask is empty. Cannot sample points.")
            return None

        sampling_mask = copy.deepcopy(mask)
        sampling_mask[prior_mask == True] = False  # Exclude the hand mask from sampling

        # First erode the mask to erase noise, then dilate
        sampling_mask = cv2.erode(sampling_mask.astype(np.uint8), np.ones((10, 10), np.uint8))
        sampling_mask = cv2.dilate(sampling_mask.astype(np.uint8), np.ones((eps_pixel, eps_pixel), np.uint8))

        # perform uniform sampling on the mask
        mask_indices = np.argwhere(sampling_mask > 0)
        if mask_indices.shape[0] == 0:
            # loguru.logger.warning("No valid mask indices found for sampling.")
            return None

        # Randomly sample K indices from the mask
        sampled_indices = np.random.choice(mask_indices.shape[0], size=K, replace=False)
        sampled_points = mask_indices[sampled_indices]
        sampled_points = sampled_points[:, ::-1]  # Convert (row, col) to (col, row)

        return sampled_points

    def _process_masks(self, masks, scores, hand_mask):
        """Process and filter segmentation masks."""
        masks = self.msam.filter_masks_by_score(masks, scores)
        masks = self.msam.remove_small_regions(masks, 1000, mode="islands")
        masks = self.msam.filter_masks_by_size(masks, 2000, 100000)
        masks = self.msam.filter_redundant_masks(masks, iou_thresh=0.25)
        masks = self.remove_hand_mask(masks, hand_mask, iou_thresh=0.01)
        return masks

    def _filter_objects_3d(
        self,
        rgb_image,
        depth_image,
        obj_masks,
        hand_mask,
        dist_thresh=0.15,
        camera_pose=None,
    ):
        """
        Process and filter objects based on their 3D distance from the hand.

        Args:
            rgb_image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            obj_masks (list): Object masks.
            hand_mask (np.ndarray): Hand mask.
            dist_thresh (float): Distance threshold.

        Returns:
            tuple: (obj_pcds, filtered_2D_masks)
        """
        # Project masks to 3D
        obj_pcds, resized_obj_masks = self.project_mask_3d(rgb_image, depth_image, obj_masks, camera_pose)
        hand_pcds, resized_hand_mask = self.project_mask_3d(rgb_image, depth_image, [hand_mask], camera_pose)
        hand_pcd = hand_pcds[0]

        # Filter objects by distance from hand
        obj_pcds, idx = self.remove_objects_far_from_hand(hand_pcd, obj_pcds, dist_thrsh=dist_thresh)

        if not obj_pcds:
            return None, None

        # Get corresponding masks
        filtered_2D_masks = [resized_obj_masks[i] for i in idx]

        return obj_pcds, filtered_2D_masks

    def sample_kp(self, rgb_image, masks, num_points=10, feat_type="shi"):
        """
        Sample features on the segmented masks.

        Args:
            rgb_image (np.ndarray): RGB image.
            masks (list): List of binary masks.
            num_points (int): Number of points to sample.
            feat_type (str): Feature type (orb or good_features).

        Returns:
            list: List of sampled feature points.
        """

        track_points = []
        for obj_mask in masks:
            if feat_type == "orb":
                points = self.sample_orb_features(rgb_image, obj_mask.astype(np.uint8) * 255, nfeatures=num_points)
            else:
                points = self.sample_good_features_to_track(rgb_image, obj_mask.astype(np.uint8) * 255, max_corners=num_points)
            track_points.append(points)

        # Flatten point lists
        track_points = [p for sublist in track_points for p in sublist]

        return track_points

    def remove_hand_mask(self, masks, hand_mask, iou_thresh=0.5):
        """
        Removes masks that overlap with the hand mask based on IoU threshold.

        Args:
            masks (list): List of binary masks to filter.
            hand_mask (np.ndarray): Binary hand mask.
            iou_thresh (float): IoU threshold for filtering.

        Returns:
            list: Filtered masks.
        """
        if not masks:
            return []

        # Convert hand mask to proper format
        if hand_mask.dtype == bool:
            hand_mask = hand_mask.astype(np.uint8)

        # Resize hand mask to match other masks
        hand_mask_resized = cv2.resize(
            hand_mask,
            (masks[0].shape[1], masks[0].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        hand_mask_binary = hand_mask_resized > 0.5

        # Filter masks by IoU
        filtered_masks = []
        for mask in masks:
            iou = self.msam.compute_iou(mask, hand_mask_binary)
            if iou <= iou_thresh:
                filtered_masks.append(mask)

        return filtered_masks

    def project_mask_3d(self, rgb, depth, masks, pose=None):
        """
        Projects 2D masks to 3D using depth image and camera parameters.

        Args:
            rgb (np.ndarray): RGB image.
            depth (np.ndarray): Depth image.
            masks (list): List of binary masks.
            poses (np.ndarray, optional): Camera poses.

        Returns:
            list: List of 3D point clouds.
        """
        if not masks or self.dataset is None:
            return []

        projected_masks, resized_masks = [], []
        for i, mask in enumerate(masks):
            # Prepare mask
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask > 0.5

            # Create point cloud
            pcd, _ = self.dataset.create_pcd(rgb, depth, camera_pose=pose, maskout=~mask)
            projected_masks.append(pcd)
            resized_masks.append(mask)

        return projected_masks, resized_masks

    def remove_objects_far_from_hand(self, hand_pcd, objects_pcd, dist_thrsh=0.1):
        """
        Removes 3D masks that are far from the hand

        Args:
            hand_3d_mask (o3d.geometry.PointCloud): 3D mask of the hand.
            objects_3d_masks (list of o3d.geometry.PointCloud): List of 3D masks of objects.
            inter_ratio (float, optional): Chamfer distance threshold for filtering masks.
                          Masks with Chamfer distance greater than this threshold will be removed.
                          Default is 0.1.

        Returns:
            list of np.ndarray: List of filtered 3D masks that are close to the hand
        """
        filtered_masks = []
        idx = []
        for i, mask in enumerate(objects_pcd):
            z_median_hand = np.median(np.asarray(hand_pcd.points)[:, 2])
            z_median_obj = np.median(np.asarray(mask.points)[:, 2])
            diff = np.linalg.norm(z_median_hand - z_median_obj)
            if diff < dist_thrsh:
                filtered_masks.append(mask)
                idx.append(i)

        return filtered_masks, idx

    def sample_orb_features(self, image, mask, nfeatures=500):
        """
        Detect ORB keypoints and compute descriptors within the ROI.

        Args:
            image (np.ndarray): Input RGB image.
            mask (np.ndarray): Binary mask indicating ROI.
            nfeatures (int): Maximum number of features to detect.

        Returns:
            list: List of keypoint coordinates.
        """
        # Create an ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures)

        # Resize mask to match image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Detect keypoints and compute descriptors
        keypoints, _ = orb.detectAndCompute(image, mask)

        # Convert keypoints to (x, y) coordinates
        return [kp.pt for kp in keypoints]

    def sample_good_features_to_track(self, image, mask, max_corners=500):
        """
        Detect good features to track within the ROI.
        """
        # Convert RGB image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize mask to match image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Detect good features to track (using gray_image instead of image)
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            mask=mask,
            qualityLevel=0.01,
            minDistance=25,
        )

        if corners is None:
            return []

        # Convert corners to (x, y) coordinates
        return [tuple(c[0]) for c in corners]

    def sample_random_points_on_mask(self, image, mask, num_points=100):
        """
        Sample random points within the ROI defined by the mask.

        Args:
            image (np.ndarray): Input RGB image.
            mask (np.ndarray): Binary mask indicating ROI.
            num_points (int): Number of points to sample.

        Returns:
            list: List of sampled point coordinates.
        """
        # Resize mask to image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Get non-zero pixels in mask
        mask_indices = np.argwhere(mask > 0.5)

        if len(mask_indices) == 0:
            return []

        # Sample random points
        indices = np.random.choice(
            len(mask_indices),
            size=min(num_points, len(mask_indices)),
            replace=len(mask_indices) < num_points,
        )
        random_points = mask_indices[indices]

        return random_points

    def extract_hand_action_segments(
        self,
        rgb_frames,
        min_window_size=30,
        max_window_size=60,
        smoothing_window_size=10,
        threshold_avg=0.5,
        hand_mask_size=250,
    ):
        """
        Extracts segments corresponding to hand-based actions from an egocentric RGBD video sequence,
        using a moving average to smooth out intermittent false negatives.

        Args:
            rgb_frames (list): List of RGB frames.
            segmentor (ArticulatedObjectSegmentor): Instance with the segment_articulated_object method.
            min_window_size (int): Minimum number of frames for a valid hand action segment.
            smoothing_window_size (int): Number of frames over which to compute the moving average.
            threshold_avg (float): Average detection threshold (between 0 and 1) to consider the hand present.
            hand_mask_size (int): Minimum size for valid hand mask.

        Returns:
            list of tuple: List of (start_index, end_index) tuples for detected hand action segments.
        """
        segments = []
        in_segment = False
        segment_start = None
        detection_history = []
        hand_frames = []

        for idx, frame in tqdm(
            enumerate(rgb_frames),
            total=len(rgb_frames),
            desc="Extracting action segments",
        ):
            hand_mask = self.hand_segmentor.segment(frame)
            hand_mask = self.hand_segmentor.remove_small_regions(hand_mask, 1000.0, mode="islands")

            # draw the hand mask on the image
            hand_mask = hand_mask.astype(np.uint8) * 255
            hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
            # # rescale hand_mask to the same size as the image
            hand_mask = cv2.resize(hand_mask, (frame.shape[1], frame.shape[0]))
            image_rgb = cv2.addWeighted(frame * 255, 0.5, hand_mask, 0.5, 0)
            hand_frames.append(hand_mask)

            # Determine binary detection for current frame.
            detected = 0
            if np.sum(hand_mask) > hand_mask_size:
                detected = 1

            # Append the detection result to the history.
            detection_history.append(detected)
            # Use only the last 'window_size' frames for the moving average.
            window = detection_history[-smoothing_window_size:]
            avg_detection = np.mean(window) if window else 0

            # Use the smoothed result to decide if the hand is present.
            if avg_detection >= threshold_avg:
                if not in_segment:
                    # Start a new segment; adjust the start to account for the smoothing window.
                    segment_start = max(0, idx - smoothing_window_size + 1)
                    in_segment = True
            else:
                if in_segment:
                    # End the segment if the moving average falls below threshold.
                    segments.append((segment_start, idx - 1))
                    in_segment = False

        # Close any open segment at the end of the video.
        if in_segment:
            segments.append((segment_start, len(rgb_frames) - 1))

        # Filter out segments that are too short or too long.
        loguru.logger.info(f"Detected {len(segments)} segments.")
        segments = [seg for seg in segments if max_window_size >= seg[1] - seg[0] >= min_window_size]
        loguru.logger.info(f"Filtered to {len(segments)} segments.")

        return segments, hand_frames

    def extract_hand_action_segments_smoothed(
        self,
        rgb_frames,
        min_window_size=30,
        max_window_size=60,
        smoothing_window_size=10,
        threshold_avg=0.5,
        hand_mask_size=250,
    ):
        """Extracts hand masks from an egocentric RGBD video sequence,
        using a moving average to smooth out intermittent false negatives.
        """

        hand_frames = self.extract_hand_masks(rgb_frames)

        segments = []

    def extract_hand_masks(
        self,
        rgb_frames,
    ):
        """
        Segment the hand in each frame of an egocentric RGBD video sequence.

        Args:
            rgb_frames (list): List of RGB frames.

        Returns:
            list: List of per-frame binary hand masks (small regions removed).
        """
        hand_frames = []

        for idx, frame in tqdm(
            enumerate(rgb_frames),
            total=len(rgb_frames),
            desc="Identifying hand masks",
        ):
            hand_mask = self.hand_segmentor.segment(frame)
            hand_mask = self.hand_segmentor.remove_small_regions(hand_mask, 1000.0, mode="islands")
            hand_frames.append(hand_mask)

            # # draw the hand mask on the image
            # hand_mask = hand_mask.astype(np.uint8) * 255
            # hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
            # # # rescale hand_mask to the same size as the image
            # hand_mask = cv2.resize(hand_mask, (rgb_frames[idx].shape[1], rgb_frames[idx].shape[0]))
            # image_rgb = cv2.addWeighted(rgb_frames[idx] * 255, 0.5, hand_mask, 0.5, 0)
            # hand_frames.append(hand_mask)

        return hand_frames

    @staticmethod
    def play_hand_action_segments(video_frames, segments, window_name="Hand Action Segment", frame_delay=100):
        """
        Plays the segments (windows) corresponding to hand-based actions from the video frames.

        Args:
            video_frames (list): List of dictionaries representing frames (each should contain "rgb" key).
            segments (list of tuple): List of (start_index, end_index) tuples representing segments.
            window_name (str): Name of the OpenCV window.
            frame_delay (int): Delay in milliseconds between frames.
        """
        for seg_idx, (start, end) in enumerate(segments):
            loguru.logger.info(f"Playing segment {seg_idx + 1}/{len(segments)} (frames {start} to {end}) with length {end - start + 1}")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            for idx in range(start, end + 1):
                # Get the RGB frame, convert it to BGR for OpenCV.
                frame_rgb = video_frames[idx]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord("q"):
                    # Skip the current segment if 'q' is pressed.
                    print("Skipping current segment...")
                    break
            cv2.destroyWindow(window_name)

        cv2.destroyAllWindows()

    def draw_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        point_radius: int = 6,
    ):
        """
        Draw points on an image for visualization using OpenCV.

        :param image: PIL Image.
        :param points: List of [x, y] coordinates.
        :param point_labels: List of labels (1 for foreground, 0 for background).
        :param point_radius: Radius of the drawn point.
        :return: The annotated image as a numpy array.
        """
        img = np.array(image)
        # ensure that points are integers
        points = [(int(x), int(y)) for x, y in points]
        for (x, y), label in zip(points, point_labels):
            color = (255, 255, 0) if label == 1 else (255, 0, 255)
            cv2.circle(img, (x, y), point_radius, color, -1)
        return img


class MobileSAM:
    """Wrapper around MobileSAM providing automatic and point-prompted segmentation, plus mask filtering utilities."""

    def __init__(self, config: dict):
        """
        Initialize the MobileSAM model using a configuration dictionary.

        :param config: Configuration dictionary containing 'checkpoint', 'model_type', and 'device'.
        """
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"])
        self.model.to(self.device)
        self.model.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=config.get("points_per_side", 20),
            points_per_batch=config.get("points_per_batch", 128),
            pred_iou_thresh=config.get("pred_iou_thresh", 0.5),
            stability_score_thresh=config.get("stability_score_thresh", 0.92),
            stability_score_offset=config.get("stability_score_offset", 0.7),
        )
        self.predictor = SamPredictor(self.model)

    def _resize_image(self, image: Image.Image | np.ndarray, input_size: int):
        """Resize image to maintain aspect ratio with the largest side equal to input_size."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        w, h = image.size
        scale = input_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size), scale

    def segment_everything(self, image: Image.Image, input_size: int = 1024):
        """
        Automatically segment all objects in the image.

        :param image: PIL Image.
        :param input_size: Target size for the largest image dimension.
        :return: List of segmentation annotations.
        """
        resized, _ = self._resize_image(image, input_size)
        annotations = self.mask_generator.generate(np.array(resized))
        return annotations

    def segment_object_with_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        input_size: int = 1024,
        multimask_output: bool = True,
    ):
        """
        Segment objects in the image guided by user-provided points.

        :param image: PIL Image.
        :param points: List of [x, y] coordinates.
        :param point_labels: List of labels (1 for foreground, 0 for background).
        :param input_size: Target size for the largest image dimension.
        :param multimask_output: Whether to return multiple masks.
        :return: (masks, scores, logits) as returned by the SAM predictor.
        """
        resized, scale = self._resize_image(image, input_size)
        # Scale the points to the resized image coordinates.
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in points])
        labels = np.array(point_labels)
        self.predictor.set_image(np.array(resized))
        masks, scores, logits = self.predictor.predict(point_coords=pts, point_labels=labels, multimask_output=multimask_output)
        return masks, scores, logits

    def segment_multiple_objects_with_points(
        self,
        image: Image.Image,
        points: list,
        point_labels: list,
        input_size: int = 1024,
    ):
        """
        Segments multiple objects in an image based on provided points and their labels.

        Args:
            image (Image.Image): The input image to be segmented.
            points (list): A list of (x, y) tuples representing the coordinates of the points.
            point_labels (list): A list of labels corresponding to each point.
            input_size (int, optional): The size to which the input image should be resized. Default is 1024.

        Returns:
            tuple: A tuple containing:
            - masks (numpy.ndarray): The segmented masks for each object.
            - scores (numpy.ndarray): The confidence scores for each mask.
            - logits (torch.Tensor): The raw logits from the predictor.
        """
        resized, scale = self._resize_image(image, input_size)
        # Convert the points to torch tensors.
        pts_tensor = torch.tensor(points.copy(), dtype=torch.float32, device=self.device).view(-1, 1, 2) * scale
        labels_tensor = torch.tensor(point_labels, dtype=torch.int64, device=self.device).view(-1, 1)
        self.predictor.set_image(np.array(resized))
        masks, scores, logits = self.predictor.predict_torch(pts_tensor, labels_tensor)
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()
        return masks, scores, logits

    @staticmethod
    def remove_small_regions(masks: list, min_area_thresh: float, mode: str = "islands"):
        """
        Removes regions that are either too small (below area_thresh) or too big (above max_area_thresh)
        in a mask. Returns the processed mask based on the provided mode.

        Parameters:
            masks (list): List of binary masks.
            area_thresh (float): Minimum area threshold. Regions smaller than this are removed.
            max_area_thresh (float): Maximum area threshold. Regions larger than this are removed.
            mode (str): Either "holes" or "islands".
                        - "holes": Remove holes (small or too big holes will be filled).
                        - "islands": Remove islands (keep only regions not removed).

        Returns:
            list: List of filtered masks.
        """
        filtered_masks = []
        for mask in masks:
            if mask.dtype != bool:
                mask = mask > 0.5
            correct_holes = mode == "holes"
            working_mask = (correct_holes ^ mask).astype(np.uint8)
            n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
            sizes = stats[:, -1][1:]  # Row 0 is background label
            # Identify regions that are too small or too big.
            remove_regions = [i + 1 for i, s in enumerate(sizes) if s < min_area_thresh]
            if len(remove_regions) == 0:
                filtered_masks.append(mask)
                continue
            if correct_holes:
                fill_labels = [0] + remove_regions
            else:
                fill_labels = [i for i in range(n_labels) if i not in ([0] + remove_regions)]
                # If every region is removed, keep the largest region
                if len(fill_labels) == 0:
                    fill_labels = [int(np.argmax(sizes)) + 1]
            mask = np.isin(regions, fill_labels)
            filtered_masks.append(mask)
        return filtered_masks

    def filter_masks_by_score(self, masks, scores):
        """
        Filter masks by selecting the highest scoring mask for each object.

        :param masks: Array of masks.
        :param scores: Array of scores corresponding to the masks.
        :return: List of best masks based on scores.
        """
        best_masks = []
        for i in range(masks.shape[0]):
            best_mask = masks[i][np.argmax(scores[i])]
            best_masks.append(best_mask)
        return best_masks

    def filter_masks_by_size(self, masks, min_size=500, max_size=5000):
        """
        Filter masks by size, keeping only those within the specified size range.

        :param masks: List of binary masks.
        :param min_size: Minimum size of the mask to keep.
        :param max_size: Maximum size of the mask to keep.
        :return: List of filtered masks.
        """
        filtered_masks = []
        for mask in masks:
            mask_size = np.sum(mask)
            if min_size <= mask_size <= max_size:
                filtered_masks.append(mask)
        return filtered_masks

    def filter_redundant_masks(self, masks, iou_thresh=0.75):
        """
        Filter out redundant masks based on their IoU with the highest scoring mask.

        :param masks: List of binary masks.
        :param iou_thresh: IoU threshold to consider two masks as the same object.
        :return: List of filtered masks.
        """
        filtered_masks = []
        for i, mask in enumerate(masks):
            if i == 0:
                filtered_masks.append(mask)
                continue
            iou = self.compute_iou(mask, filtered_masks[0])
            if iou < iou_thresh:
                filtered_masks.append(mask)
        return filtered_masks

    @staticmethod
    def compute_iou(mask1, mask2):
        """
        Compute the Intersection over Union (IoU) between two binary masks.

        :param mask1: First binary mask.
        :param mask2: Second binary mask.
        :return: The IoU value.
        """
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        return np.sum(intersection) / np.sum(union)

    @staticmethod
    def compute_mask_recall(gt_mask, pred_mask):
        """
        Compute the recall of a predicted mask against a ground-truth mask.

        :param gt_mask: Ground-truth binary mask.
        :param pred_mask: Predicted binary mask.
        :return: The recall value (intersection area / ground-truth area).
        """
        intersection = np.logical_and(gt_mask, pred_mask)
        return np.sum(intersection) / np.sum(gt_mask)

    @staticmethod
    def compute_containment(gt_mask, pred_mask):
        """
        Compute how much of a predicted mask falls within a ground-truth mask.

        :param gt_mask: Ground-truth binary mask.
        :param pred_mask: Predicted binary mask.
        :return: The containment ratio (intersection area / predicted-mask area).
        """

        intersection = np.logical_and(gt_mask, pred_mask)
        return np.sum(intersection) / np.sum(pred_mask)

    def visualize_segmentation(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        input_size: int = 1024,
    ):
        """
        Visualize segmentation annotations on the image using OpenCV.

        :param image: np.ndarray.
        :param masks: Batched output of multiple object masks for the same image.
        :param input_size: Target size for the largest image dimension.
        :return: Image with visualized annotations.
        """
        img = copy.deepcopy(image)
        for mask, score in zip(masks, scores):
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img[mask] = img[mask] * 0.5 + color * 0.5
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color.tolist(), 2)
            cv2.putText(
                img,
                f"{score:.2f}",
                (int(np.where(mask)[1].mean()), int(np.where(mask)[0].mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                1,
            )
        return img

    def segment_everything_batch(self, images: List[np.ndarray], input_size: int = 720, visualize: bool = False):
        """
        Run automatic "segment everything" over a batch of images.

        :param images: List of RGB images.
        :param input_size: Target size for the largest image dimension.
        :param visualize: If True, also render an annotated visualization per image.
        :return: Tuple (frame_masks, frame_scores, frame_plots), one entry per input
            image (frame_plots is empty unless visualize=True).
        """
        frame_masks, frame_scores, frame_plots = [], [], []
        for i, img in tqdm(enumerate(images), total=len(images)):
            masks, scores, _ = self.segment_everything(img.astype(np.uint8))
            frame_masks.append(masks)
            frame_scores.append(scores)
            if visualize:
                plot = self.visualize_segmentation(img, masks, scores)
                frame_plots.append(plot)
        return frame_masks, frame_scores, frame_plots


def main():
    """Main function for smoke-testing the models wrapped by Segmentation/MobileSAM:
    MobileSAM (automatic + point-prompted), the YOLO human segmentor, the
    prior-mask pipeline, and the lazily-built SAM2 video predictor."""
    from moma_sg.data.kinect_dataloader import KinectRGBDDataset
    from omegaconf import OmegaConf

    root_path = Path("/path/to/arti4d/raw/rh201/scene_2025-04-25-15-16-29")
    out_dir = Path(__file__).resolve().parents[2] / "scripts" / "tests" / "quicktest_output" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configuration. tf_file_path/flipped/gt_poses/droid_slam are required by
    # KinectRGBDDataset._load_parameters and were missing from the original stub config.
    dataset_cfg = OmegaConf.create(
        {
            "root_dir": "data",
            "transforms": None,
            "depth_min": 0.5,
            "depth_max": 3.0,
            "root_path": root_path,
            "tf_file_path": "/path/to/MoMa-SG/calibration/azure_tf.json",
            "flipped": True,
            "gt_poses": True,
            "droid_slam": False,
        }
    )
    dataset = KinectRGBDDataset(dataset_cfg)
    print(f"Dataset length: {len(dataset)}")

    # Segmentor configuration, built by hand instead of loading configs/momasg.yaml so this
    # doesn't pull in the "keys" defaults (which hold the OpenAI API key).
    seg_cfg = OmegaConf.create(
        {
            "package_path": "/path/to/MoMa-SG",
            "use_cuda": True,
            "sam_model_type": "vit_t",
            "interaction": {
                "yolo_path": "./checkpoints/yolo11x-seg.pt",
                "msam_checkpoint": "./checkpoints/mobile_sam.pt",
                "dilate_kernel_size": 5,
                "dilate_iterations": 1,
            },
        }
    )
    segmentor = Segmentation(seg_cfg)
    print(f"Segmentation initialized (MobileSAM + YOLO), device={segmentor.device}")

    rgb = dataset[0]["rgb"].astype(np.uint8)
    print(f"Sample frame 0 shape: {rgb.shape}")

    # --- MobileSAM: automatic "segment everything" ---
    annotations = segmentor.msam.segment_everything(Image.fromarray(rgb), input_size=720)
    print(f"[MobileSAM.segment_everything] {len(annotations)} masks")
    if annotations:
        areas = [a["area"] for a in annotations]
        ious = [a["predicted_iou"] for a in annotations]
        print(f"  area min/mean/max = {min(areas)}/{np.mean(areas):.0f}/{max(areas)}, mean IoU = {np.mean(ious):.3f}")
        masks = np.stack([a["segmentation"] for a in annotations])
        h, w = masks.shape[1:]
        rgb_resized = cv2.resize(rgb, (w, h))
        vis = segmentor.msam.visualize_segmentation(rgb_resized, masks, np.array(ious))
        Image.fromarray(vis).save(out_dir / "segment_everything.png")

    # --- MobileSAM: point-prompted segmentation on the image center ---
    h, w = rgb.shape[:2]
    masks_b, scores_b, _ = segmentor.msam.segment_object_with_points(Image.fromarray(rgb), [[w // 2, h // 2]], [1], input_size=720)
    print(f"[MobileSAM.segment_object_with_points] masks shape={masks_b.shape}, scores={scores_b}")

    # --- YOLO human segmentor ---
    human_counts = []
    for i in range(0, min(len(dataset), 60), 12):
        frame = dataset[i]["rgb"].astype(np.uint8)
        human_counts.append((i, len(segmentor._segment_human(frame))))
    print(f"[Segmentation._segment_human] (frame_idx, #human_masks) = {human_counts}")

    # --- prior-mask pipeline: YOLO detection + dilation over a handful of frames ---
    frame_idxs = list(range(0, min(len(dataset), 40), 8))
    rgb_frames = [dataset[i]["rgb"].astype(np.uint8) for i in frame_idxs]
    prior_masks = segmentor.compute_prior_masks(rgb_frames, dilate_its=1, dilate_kernel_size=5)
    print(f"[Segmentation.compute_prior_masks] shape={prior_masks.shape}, foreground px/frame={[int(m.sum()) for m in prior_masks]}")

    # --- SAM2 video predictor: propagate a mask seeded from the "segment everything" result ---
    if annotations:
        window_frames = [dataset[i]["rgb"].astype(np.uint8) for i in range(8)]
        best = max(annotations, key=lambda a: a["area"])
        seed_mask = cv2.resize(best["segmentation"].astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        video_segments = segmentor.reconstruct_with_sam2(window_frames, {0: seed_mask}, str(out_dir), segment_idx=0)
        print(f"[Segmentation.reconstruct_with_sam2] propagated {len(video_segments)} frames, overlays in {out_dir}")

    print("\nAll quick tests completed.")


if __name__ == "__main__":
    main()
