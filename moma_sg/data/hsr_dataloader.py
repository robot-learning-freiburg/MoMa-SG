import bisect
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import loguru
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import tqdm


class HSRDataset(Dataset):
    """HSRRGBDDataset class.
    This class implements a dataset class for the HSR Head RGB-D camera data.
    Returns a dictionary containing the keys "rgb", "depth", "pose", and "idx".

    Args:
        cfg (dict): Configuration dictionary.
    """

    def __init__(self, cfg):
        """Initialize the dataset: index RGB/depth frames, load camera parameters/poses, and
        read interaction-segment annotations.

        Args:
            cfg (dict): Configuration dictionary. Expected keys include "root_path" (Path to
                the scene directory containing "rgb"/"depth"/"odom" subfolders), and optional
                "depth_min", "depth_max", "gt_poses", and "droid_slam".
        """
        self.depth_min = cfg.get("depth_min", 0.0)
        self.depth_max = cfg.get("depth_max", 10.0)
        self.root_path = cfg["root_path"]
        self.rgb_path = cfg["root_path"] / "rgb"
        self.depth_path = cfg["root_path"] / "depth"
        self.cam_poses_path = cfg["root_path"] / "odom"
        self.gt_poses = cfg.get("gt_poses", None)
        self.droid_slam = cfg.get("droid_slam", None)
        self.data_list = self._get_data_list()
        self._load_parameters()
        self.interaction_segments_path = os.path.join(self.root_path, "matched_cues.csv")
        self._read_interaction_segment_data()

    def __len__(self):
        """Return the number of RGB-D frames in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Load the RGB-D frame and camera pose at the given index.

        Args:
            idx (int): Index into the sorted list of RGB/depth frame pairs.

        Returns:
            dict: Sample with keys "rgb" (H,W,3 uint8 array), "depth" (H,W float32 array),
                "pose" (4x4 homogeneous camera-to-world transform, or identity if no camera
                poses are available), and "idx" (the requested index).
        """
        rgb_path, depth_path = self.data_list[idx]
        rgb = self._load_image(rgb_path)
        depth = self._load_depth(depth_path)
        timestamp = float(Path(rgb_path).stem.split("_")[-1])
        pose = self._get_closest_pose_in_se3(timestamp) if self.cam_poses else np.eye(4)
        if self.droid_slam:
            if self.registered:
                pose = self.reg_matrix @ pose  # apply the registration matrix
            else:
                pose
        if self.gt_poses:  # poses generated from the HTC tracking system
            pose = pose  # @ self.base_T_depth @ self.depth_T_rgb
        sample = {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "pose": pose,
            "idx": idx,
        }
        return sample

    def _load_parameters(self):
        """Load camera intrinsics, camera poses (from ground-truth or DroidSLAM), and the
        scene's ground-truth articulation file, populating the corresponding instance
        attributes.
        """
        self.rgb_intrinsics = np.array(self._read_camera_params(self.rgb_path / "camera_info.txt")["K"]).reshape(3, 3)
        self.depth_intrinsics = np.array(self._read_camera_params(self.depth_path / "camera_info.txt")["K"]).reshape(3, 3)
        if self.cam_poses_path and self.gt_poses:
            self.cam_poses = self._read_camera_poses(self.cam_poses_path / Path(self.root_path.stem + ".csv"))
        elif self.droid_slam:
            self.cam_poses = self._read_camera_poses(self.root_path / "cam_trajectory.csv")
            loguru.logger.info(f"Loaded DroidSLAM camera poses from: {self.root_path / 'cam_trajectory.csv'}")
            if Path(self.root_path / "registration_matrix.json").exists():
                self.registered = True
                with open(self.root_path / "registration_matrix.json", "r") as f:
                    reg_matrix = json.load(f)
                    self.reg_matrix = np.array(reg_matrix["transformation_matrix"]).reshape(4, 4)
            else:
                self.registered = False
        if getattr(self, "cam_poses", None):
            # Sort once so the closest-pose lookup can binary-search instead of
            # linear-scanning the full pose list (which can be 10-100x larger
            # than the frame count) on every __getitem__ call.
            self.cam_poses.sort(key=lambda pose: pose["timestamp"])
            self._cam_pose_timestamps = [pose["timestamp"] for pose in self.cam_poses]
        if Path(os.path.join(self.root_path, Path(self.root_path).stem + ".json")).exists():
            self._read_articulations(os.path.join(self.root_path, Path(self.root_path).stem + ".json"))
            loguru.logger.info(f"Loaded GT articulations:{os.path.join(self.root_path, Path(self.root_path).stem + '.json')}")
        else:
            loguru.logger.warning(
                "Failed to load scene articulation file. Please check if the file exists and the path is correct: %s",
                os.path.join(self.root_path, Path(self.root_path).stem + ".json"),
            )

    def _read_camera_params(self, file_path: Path) -> Dict[str, Any]:
        """
        Read camera parameters from a text file.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            Dict[str, Any]: Dictionary containing the camera parameters.
        """
        self.camera_params = {}
        with open(file_path, "r") as file:
            for line in file:
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if key in ["D", "K", "R", "P"]:
                        self.camera_params[key] = tuple(map(float, value.strip("()").split(",")))
                    elif key in [
                        "binning_x",
                        "binning_y",
                        "width",
                        "height",
                        "x_offset",
                        "y_offset",
                        "seq",
                        "secs",
                        "nsecs",
                    ]:
                        # ignore if key already exists
                        if key in self.camera_params:
                            continue
                        self.camera_params[key] = int(value)
                    elif key == "do_rectify":
                        self.camera_params[key] = value.lower() == "true"
                    else:
                        self.camera_params[key] = value
        return self.camera_params

    def _read_articulations(self, path):
        """
        Read articulated objects information from a JSON file.

        Args:
            path (str): Path to the JSON file containing articulation data.

        The JSON file should contain a dictionary where keys are object names and values
        are dictionaries with 'position' and 'axis' keys, each containing a list of 3 coordinates.
        """
        try:
            with open(path, "r") as f:
                self.articulated_objects = json.load(f)

            # Basic structure validation
            if not isinstance(self.articulated_objects, dict):
                loguru.logger.warning("Articulation data is not in the expected dictionary format")
                self.articulated_objects = {}
            else:
                loguru.logger.info(f"Successfully loaded {len(self.articulated_objects)} articulated objects from {path}")
        except Exception as e:
            loguru.logger.error(f"Failed to load articulated objects from {path}: {e}")
            self.articulated_objects = {}

    def _read_camera_poses(self, path):
        """Read camera poses from a CSV file into a list of pose dicts.

        Args:
            path (str or Path): Path to a CSV file with columns "timestamp", "x", "y", "z",
                "qx", "qy", "qz", "qw".

        Returns:
            list[dict] or None: List of pose dicts (timestamp plus position/quaternion), or
                None if the file does not exist.
        """
        if not os.path.exists(path):
            loguru.logger.warning(f"Camera poses file not found at {path}")
            return None
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            camera_poses = []
            for row in reader:
                # Strip whitespace from keys
                row = {k.strip(): v for k, v in row.items()}
                camera_pose = {
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                }
                if self.droid_slam:
                    camera_pose["timestamp"] = float(str(int(camera_pose["timestamp"]))[:10] + "." + str(int(camera_pose["timestamp"]))[10:])
                    # as it saved in the wrong order
                    (
                        camera_pose["qy"],
                        camera_pose["qz"],
                        camera_pose["qw"],
                        camera_pose["qx"],
                    ) = (
                        camera_pose["qx"],
                        camera_pose["qy"],
                        camera_pose["qz"],
                        camera_pose["qw"],
                    )
                camera_poses.append(camera_pose)
        return camera_poses

    def _find_closest_pose(self, data, target_timestamp):
        """Binary-search `data` for the pose whose timestamp is closest to `target_timestamp`.

        Args:
            data (list[dict]): Pose dicts sorted by "timestamp" (typically `self.cam_poses`).
            target_timestamp (float): Timestamp to search for.

        Returns:
            dict: The pose dict from `data` with the closest timestamp.
        """
        # data (self.cam_poses) is sorted by timestamp in _load_parameters, and
        # self._cam_pose_timestamps mirrors it index-for-index, so we can
        # binary-search instead of scanning every pose for every frame.
        idx = bisect.bisect_left(self._cam_pose_timestamps, target_timestamp)
        if idx == 0:
            return data[0]
        if idx == len(data):
            return data[-1]
        before, after = data[idx - 1], data[idx]
        if target_timestamp - before["timestamp"] <= after["timestamp"] - target_timestamp:
            return before
        return after

    def _quat_to_se3(self, quat):
        """Convert a position+quaternion pose dict into a 4x4 homogeneous transform.

        Args:
            quat (dict): Pose dict with keys "x", "y", "z", "qx", "qy", "qz", "qw".

        Returns:
            np.ndarray: 4x4 homogeneous SE(3) transformation matrix.
        """
        x, y, z = quat["x"], quat["y"], quat["z"]
        qx, qy, qz, qw = quat["qx"], quat["qy"], quat["qz"], quat["qw"]
        rotation = R.from_quat([qx, qy, qz, qw])
        R_matrix = rotation.as_matrix()
        # Construct the SE3 matrix
        SE3 = np.eye(4)
        SE3[:3, :3] = R_matrix
        SE3[:3, 3] = [x, y, z]
        return SE3

    def _get_closest_pose_in_se3(self, target_timestamp):
        """Look up the camera pose closest to a frame timestamp as a 4x4 transform.

        Args:
            target_timestamp (float): Frame timestamp, as encoded in the RGB filename.

        Returns:
            np.ndarray: 4x4 homogeneous SE(3) transformation matrix of the closest camera pose.
        """
        target_timestamp = float(str(int(target_timestamp))[:10] + "." + str(int(target_timestamp))[10:])
        closest_pose_quat = self._find_closest_pose(self.cam_poses, target_timestamp)
        pose_se3 = self._quat_to_se3(closest_pose_quat)
        return pose_se3

    def _get_data_list(self):
        """List and pair up the RGB and depth frame files in the scene directory.

        Returns:
            list[tuple[str, str]]: Sorted list of (rgb_path, depth_path) pairs.
        """
        rgb_files = glob.glob(str(Path(self.root_path) / "rgb" / "*.jpg"))
        depth_files = glob.glob(str(Path(self.root_path) / "depth" / "*.png"))
        rgb_files.sort()
        depth_files.sort()
        return list(zip(rgb_files, depth_files))

    def _read_interaction_segment_data(self):
        """Load interaction segments from `self.interaction_segments_path` into
        `self.interactions` (list of [obj_name, (start, end)]) and `self.interaction_timestamps`
        (flattened list of all frame indices covered by any segment). Sets both to empty lists
        on failure.
        """
        # read in interaction segments
        try:
            df = pd.read_csv(self.interaction_segments_path)
            self.interactions = []
            self.interaction_timestamps = []
            for index, row in df.iterrows():
                self.interactions.append([row.iloc[0], (row.iloc[1], row.iloc[2])])
                self.interaction_timestamps.extend([i for i in range(int(row.iloc[1]), int(row.iloc[2]))])
        except Exception as e:
            loguru.logger.error(f"Failed to read interaction segments from {self.interaction_segments_path}: {e}")
            self.interactions = []
            self.interaction_timestamps = []

    def _load_image(self, path):
        """Load an RGB image from disk, converting from BGR to RGB.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: (H,W,3) RGB image array.
        """
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return rgb

    def _load_depth(self, path):
        """Load a depth image from disk unchanged (preserving its native dtype/units).

        Args:
            path (str): Path to the depth image file.

        Returns:
            np.ndarray: (H,W) depth image array.
        """
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return depth

    def get_articulated_objects(self):
        """
        Returns the articulated objects in the scene.
        Returns:
            dict: A dictionary containing the articulated objects.
        """
        return self.articulated_objects

    def create_pcd(self, rgb_image, depth_image, camera_pose=None, maskout=None):
        """
        Creates a point cloud from RGB and depth images.

        This function generates a point cloud using the provided RGB and depth images.
        Optionally, a camera pose can be applied to transform the point cloud, and a mask
        can be used to filter out specific regions.

        Args:
            rgb_image (np.ndarray): The RGB image.
            depth_image (np.ndarray): The depth image.
            camera_pose (np.ndarray, optional): The camera pose. Defaults to None.
            maskout (np.ndarray, optional): The mask to filter out regions. Defaults to None.

        Returns:
            tuple: A tuple containing the point cloud and the valid mask.
        """
        # assert depth_image.shape == rgb_image.shape[:2], "Depth and RGB image dimensions do not match"

        # Convert depth to meters.
        z = depth_image / 1000.0

        # Create mask for valid depth values within the specified range.
        valid_mask = (z > self.depth_min) & (z < self.depth_max)
        if maskout is not None:
            valid_mask &= ~maskout

        # Get valid pixel indices (row, col).
        valid_indices = np.nonzero(valid_mask)  # valid_indices[0] = y, valid_indices[1] = x

        # Compute valid depth values.
        z_valid = z[valid_indices]
        x_valid = (valid_indices[1] - self.depth_intrinsics[0, 2]) * z_valid / self.depth_intrinsics[0, 0]
        y_valid = (valid_indices[0] - self.depth_intrinsics[1, 2]) * z_valid / self.depth_intrinsics[1, 1]

        # Stack to form (N, 3) point coordinates.
        points = np.column_stack((x_valid, y_valid, z_valid))

        # Apply camera pose transformation if provided.
        if camera_pose is not None:
            # print("Camera pose:", camera_pose)
            points = (camera_pose[:3, :3] @ points.T).T + camera_pose[:3, 3]

        # Extract color information.
        if rgb_image.ndim == 3:
            colors = rgb_image[valid_indices] / 255.0
        else:
            colors = np.zeros((points.shape[0], 3))

        # Create Open3D point cloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd, valid_mask

    def get_scene_tsdf(
        self,
        voxel_size: float = 0.02,
        truncation_distance: float = 0.03,
    ) -> o3d.pipelines.integration.TSDFVolume:
        """
        Aggregates the RGB-D images in the dataset into a TSDF volume.

        Args:
            voxel_size (float): The size of each voxel in meters.
            truncation_distance (float): The distance (in meters) within which
                depth values are considered for surface reconstruction.

        Returns:
            o3d.pipelines.integration.TSDFVolume: The aggregated TSDF volume.
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=truncation_distance,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i in tqdm.tqdm(range(len(self)), desc="Integrating frames into TSDF"):
            sample = self[i]
            rgb_array = sample["rgb"].astype(np.uint8)
            rgb = o3d.geometry.Image(rgb_array)
            depth_array = (sample["depth"] / 1000.0).astype(np.float32)
            depth = o3d.geometry.Image(depth_array)

            # Create RGBD image with explicit convert_rgb_to_intensity=False
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0,
                depth_trunc=self.depth_max,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=sample["rgb"].shape[1],
                height=sample["rgb"].shape[0],
                fx=self.depth_intrinsics[0, 0],
                fy=self.depth_intrinsics[1, 1],
                cx=self.depth_intrinsics[0, 2],
                cy=self.depth_intrinsics[1, 2],
            )

            volume.integrate(
                rgbd_image,
                intrinsic,
                np.linalg.inv(sample["pose"]),  # TSDF integration expects world-to-camera transform
            )

        return volume

    def vis_cam_poses(self):
        """
        Visualizes the camera poses in 3D space using Open3D.
        This function creates a 3D visualization of the camera poses in the dataset with
        poses as spheres and lines connecting consecutive poses.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

        prev_pose = None
        for i in range(len(self)):
            sample = self[i]
            pose = sample["pose"]
            sephere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sephere.compute_vertex_normals()
            sephere.paint_uniform_color([1, 0, 0])
            sephere.transform(pose)
            if prev_pose is not None:
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([prev_pose[:3, 3], pose[:3, 3]])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
                vis.add_geometry(line)
            vis.add_geometry(sephere)
            prev_pose = pose
            vis.poll_events()
            vis.update_renderer()
            # vis.clear_geometries()
        vis.run()
        vis.destroy_window()


# Example usage:
if __name__ == "__main__":
    # TODO: add a root placeholder
    root_path = Path("/path/to/arti4d/raw/rh201/scene_2025-04-25-10-36-37")
    cfg = {
        "root_dir": "data",
        "transforms": None,
        "depth_min": 0.5,
        "depth_max": 3.0,
        "root_path": root_path,
        "tf_file_path": "calibration/azure_tf.json",
        "flipped": True,
        "gt_poses": True,
    }

    dataset = HSRDataset(cfg)
    print("Dataset length:", len(dataset))

    # Aggregate TSDF
    tsdf_volume = dataset.get_scene_tsdf(
        voxel_size=0.01,
        truncation_distance=0.015,
    )

    # # Extract and visualize the mesh from the TSDF volume
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer('Open3D - O3DVisualizer', 640, 480)
    vis.show_settings = True

    vis.point_size = 1
    # vis.set_background((0, 0, 0, 1), None)

    # make background completely white
    vis.set_background((1, 1, 1, 1), None)

    vis.show_skybox(False)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry('axis', axis_pcd)
    vis.add_geometry('mesh', mesh)

    app.add_window(vis)
    app.run()
