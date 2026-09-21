import copy
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Union

import cv2
import gtsam
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import torch


def to_bgr(img_rgb):
    """Convert RGB image to BGR format for OpenCV."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def to_rgb(img_bgr):
    """Convert BGR image to RGB format."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def create_coordinate_frame(size=1.0):
    """Create a coordinate frame for visualization"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def tensor_to_o3d_point_cloud(points_tensor):
    """Convert PyTorch tensor to Open3D point cloud"""
    # Move tensor to CPU and convert to numpy
    points_np = points_tensor.detach().cpu().numpy()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    return pcd


def sample_points_from_mesh(mesh, num_points=1000):
    """
    Sample points from a mesh to convert it into a point cloud-like structure.

    Args:
        mesh: The TriangleMesh object (e.g., a coordinate frame).
        num_points: Number of points to sample from the mesh.

    Returns:
        point_cloud: A PointCloud object containing the sampled points.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def calculate_zy_rotation_for_arrow(vec):
    """
    Compute rotation matrices aligning the z-axis to a target direction vector.

    Used to orient arrow/cylinder meshes (which default to pointing along +z)
    towards an arbitrary direction, via a rotation about z followed by a
    rotation about y.

    Args:
        vec: (3,) direction vector to align to.

    Returns:
        tuple: (Rz, Ry), the (3,3) rotation matrices to apply in sequence to
            rotate the z-axis onto vec.
    """
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return Rz, Ry


def plot_point_tracks(pred_3d_tracks_segments: List[np.ndarray], pred_visibility_segments: List[np.ndarray], seg_idx: int, save_path: str):
    """
    Plot the 3D trajectories of tracked points for one segment and optionally save the figure.

    Args:
        pred_3d_tracks_segments: Per-segment 3D point tracks, each array shaped
            (T, N, 3) (transposed internally to (N, T, 3)).
        pred_visibility_segments: Per-segment visibility masks matching the tracks.
        seg_idx: Index of the segment to plot.
        save_path: If given, path to save the figure to (figure is closed after saving).
    """
    import matplotlib.pyplot as plt

    tracks = pred_3d_tracks_segments[seg_idx].transpose(1, 0, 2)
    vis = pred_visibility_segments[seg_idx].transpose(1, 0)
    fig = plt.figure(figsize=(18, 18))
    ax1 = fig.add_subplot(111, projection='3d')
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
        ax1.plot(
            tracks[track_idx, vis[track_idx, :] == True, 0],
            tracks[track_idx, vis[track_idx, :] == True, 1],
            tracks[track_idx, vis[track_idx, :] == True, 2],
            marker='o',
            alpha=1.0,
        )

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


def visualize_trajectory(
    free_trajectories=None,
    pcd=None,
    axes=None,
    motion_centers=None,
    cam_poses=None,
    save_frames=False,
    output_dir=None,
    output_prefix="frame_",
    save_video=False,
    fps=15,
    resolution=(1280, 720),
):
    """
    Visualize original and transformed point cloud trajectories

    Args:
        trajectories: List of lists of SE(3) transformation matrices
        axes: List of motion axes
        pcd: Point cloud to visualize
        gt_trajectory: Ground truth trajectory (list of SE(3) matrices)
        free_trajectories: List of lists of free trajectories
        motion_centers: List of motion centers
        cam_poses: List of camera poses for rendering viewpoints
        save_frames: Whether to save rendered frames as images
        output_dir: Directory to save rendered frames (created if it doesn't exist)
        output_prefix: Prefix for output frame filenames
        save_video: Whether to compile frames into a video (requires ffmpeg)
        fps: Frames per second for the video
        resolution: (width, height) tuple for rendering resolution
    """

    # Create output directory if saving frames
    if save_frames and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to {output_dir}")

    # Create Open3D visualization window with specified resolution
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])

    # Set rendering options for better quality
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # White background

    # add pcd if provided
    if pcd is not None:
        vis.add_geometry(pcd)

    # Handle multiple trajectories with different colors
    colors = [
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]  # Blue, Magenta, Cyan, Yellow

    # Make free_trajectories a list of lists if it's a single trajectory
    if free_trajectories:
        # add free trajectories
        for traj_idx, free_trajectory in enumerate(free_trajectories):
            for i, transform in enumerate(free_trajectory):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                frame.transform(transform)
                vis.add_geometry(frame)

    # Default motion center if not provided
    if motion_centers is None and axes is not None:
        motion_centers = [[0, 0, 0]] * len(axes)

    # add axes of motion if provided
    if axes is not None:
        for i, axis in enumerate(axes):
            # Get the center and normalize the axis
            center = np.array(motion_centers[i])
            axis_norm = np.array(axis) / np.linalg.norm(axis)

            # Define arrow properties
            arrow_length = 0.5  # Length of the arrow
            cylinder_radius = 0.01  # Radius of cylinder
            cone_radius = cylinder_radius * 2  # Radius of cone (arrowhead)

            # Create an arrow
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius,
                cone_radius=cone_radius,
                cylinder_height=arrow_length * 0.8,
                cone_height=arrow_length * 0.2,
            )

            # Calculate rotation to align with axis direction
            # Default arrow is along the positive y-axis
            y_axis = np.array([0, 0, 1])

            # Get rotation from y-axis to the axis direction
            rotation_matrix = np.eye(3)
            if not np.allclose(axis_norm, y_axis):
                rotation_axis = np.cross(y_axis, axis_norm)
                if np.linalg.norm(rotation_axis) > 1e-6:  # Check if not zero
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(y_axis, axis_norm), -1.0, 1.0))
                    rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()

            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = center

            # Apply transformation
            arrow.transform(transform)

            # Set color (yellow)
            arrow.paint_uniform_color([1, 1, 0.5])

            # Add to visualizer
            vis.add_geometry(arrow)

    # add motion centers if provided as spheres
    if motion_centers is not None:
        for i, center in enumerate(motion_centers):
            print(f"Motion center {i}: {center}")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(center)
            sphere.paint_uniform_color([0, 0, 1])
            vis.add_geometry(arrow)

    # If camera poses are provided and we're saving frames, render from each pose
    frame_files = []
    if cam_poses is not None and save_frames:
        # First render the scene once to initialize
        vis.poll_events()
        vis.update_renderer()

        for i, pose in enumerate(cam_poses):
            # Set camera view based on pose
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            # Convert pose to camera parameters
            extrinsic = np.linalg.inv(pose)  # Camera pose is inverse of object pose
            param.extrinsic = extrinsic
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)

            # Update view
            vis.poll_events()
            vis.update_renderer()

            # Save frame
            frame_path = os.path.join(output_dir, f"{output_prefix}{i:04d}.png")
            vis.capture_screen_image(frame_path)
            frame_files.append(frame_path)
            print(f"Saved frame {i + 1}/{len(cam_poses)} to {frame_path}")
    else:
        # Run interactive visualization if not saving frames or no camera poses provided
        vis.run()

    vis.destroy_window()

    # Compile video if requested
    if save_video and save_frames and frame_files:
        video_path = os.path.join(output_dir, f"{output_prefix}video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, f"{output_prefix}%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print("To manually create video, use ffmpeg:")
            print(f"ffmpeg -framerate {fps} -i {output_dir}/{output_prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

    return frame_files if save_frames else None


def project_points(pts, K, T):
    """
    Projects Nx3 3D points into the RGB image.
    pts: np.array (N, 3) 3D points in world coordinates
    K: np.array (3, 3) camera intrinsics
    T: np.array (4, 4) SE(3) camera extrinsics (world → camera)
    """
    R, t = T[:3, :3], T[:3, 3][:, None]
    pts = pts.reshape(-1, 3)
    cam = (R @ pts.T + t).T  # world → camera
    uv = (K @ cam.T).T  # camera → image homogeneous
    uv = uv[:, :2] / uv[:, 2, None]
    return uv.astype(int)


def plot_axis_img(
    rgb_img,
    depth_img,
    pos,  # (3,) axis origin in world/cam coords
    ori,  # (3,) orientation direction vector (defines Z axis)
    K,  # 3x3 intrinsics
    T,  # extrinsics R=3x3, t=3x1 mapping world→camera
    mask=None,
    axis_length=0.2,  # scale of drawn axes
):
    """
    Plots a 3D axis on the RGB image.
    Args:
    rgb_img: np.array (H, W, 3) RGB image
    depth_img: np.array (H, W) depth image
    pos: np.array(3,)  → world position
    ori: np.array(3,)  → orientation direction (Z-axis)
    K: np.array (3, 3) camera intrinsics
    T: np.array (4, 4) SE(3) camera extrinsics (world → camera)
    mask: np.array (H, W) boolean mask to overlay
    axis_length: float, length of the drawn axis
    """

    # --- Normalize orientation vector (Z axis) ---
    axis = ori / np.linalg.norm(ori)

    # --- Create 3D endpoints ---
    origin = pos
    axis_tip = pos + axis * axis_length

    pts_3d = np.vstack([origin, axis_tip])

    # Project
    o, z = project_points(pts_3d, K, T)

    # Draw
    img = to_bgr(rgb_img.copy())

    if mask is not None:
        # plot mask with alpha overlay
        colored_mask = np.zeros_like(img)
        colored_mask[mask == True] = [127, 255, 0]
        img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

    cv2.circle(img, tuple(o), 10, (0, 0, 255), -1)  # red origin
    cv2.line(img, tuple(o), tuple(z), (0, 0, 255), 3)  # red axis

    return img


def create_line_set(points_start, points_end, colors=None):
    """Create LineSet from start and end points."""
    num_points = len(points_start)

    # Create indices for lines
    lines = [[i, i + num_points] for i in range(num_points)]

    # Combine start and end points
    points = np.vstack((points_start, points_end))

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if colors is None:
        colors = np.tile(np.array([1, 0, 0]), (len(lines), 1))  # Default red color
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def remove_live_depth_near_object(live_pcd, obj_pts, radius=0.025):
    """Return a copy of live_pcd with all points within `radius` metres of any obj_pt removed."""
    live_pts = np.asarray(live_pcd.points)
    if len(live_pts) == 0 or len(obj_pts) == 0 or np.isnan(obj_pts).any() or np.isnan(live_pts).any():
        return live_pcd
    try:
        tree = KDTree(obj_pts)
        dists, _ = tree.query(live_pts, k=1)
        keep = dists > radius
        filtered = live_pcd.select_by_index(np.where(keep)[0])
    except Exception as e:
        print(f"Error occurred while filtering points: {e}")
        filtered = live_pcd
    return filtered


def visualize_scene_flow(pc1, pc2, flow=None, voxel_size=None):
    """
    Visualize two point clouds and their scene flow.

    Args:
        pc1 (numpy.ndarray): First point cloud (N x 3)
        pc2 (numpy.ndarray): Second point cloud (N x 3)
        flow (numpy.ndarray, optional): Scene flow vectors (N x 3)
        sample_ratio (float): Ratio of points to visualize (0.0 to 1.0)
    """
    # Create Open3D geometries
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    # Set points
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    # Sample points if ratio < 1.0
    if voxel_size is not None:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Set colors
    pcd1.paint_uniform_color([1, 0, 0])  # Red for first point cloud
    pcd2.paint_uniform_color([0, 0, 1])  # Blue for second point cloud

    # Create visualization list
    vis_list = [pcd1, pcd2]

    # Add flow vectors if provided
    if flow is not None:
        # Create flow line set
        flow_lines = create_line_set(
            pc1,
            pc1 + flow,
            colors=np.tile(np.array([0, 1, 0]), (len(pc1), 1)),  # Green for flow vectors
        )
        vis_list.append(flow_lines)

    # Calculate coordinate frame size based on point cloud bounds
    pc_range = np.ptp(pc1, axis=0)  # Gets range in each dimension
    coord_frame_size = np.mean(pc_range) * 0.2  # Use mean of ranges
    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(coord_frame_size))  # Ensure it's a float
    vis_list.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        vis_list,
        window_name="Scene Flow Visualization",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False,
    )


class TransformVisualizer:
    """Open3D-based helper for interactively visualizing coordinate-frame
    transforms and trajectories (coordinate frames, trajectory lines, grids)."""

    def __init__(self):
        """Create the Open3D visualizer window with a dark background."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Set rendering options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.point_size = 2.0

    @staticmethod
    def create_coordinate_frame(size: float = 1.0, transform: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
        """
        Create a coordinate frame with optional transform

        Args:
            size: Size of coordinate frame
            transform: 4x4 homogeneous transformation matrix
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

        if transform is not None:
            frame.transform(transform)

        return frame

    @staticmethod
    def create_trajectory_line(transforms: List[np.ndarray], color: List[float] = [1, 0, 0]) -> o3d.geometry.LineSet:
        """
        Create a line set connecting the origins of transforms

        Args:
            transforms: List of 4x4 transformation matrices
            color: RGB color for the trajectory line
        """
        points = []
        lines = []

        # Extract translation components
        for transform in transforms:
            points.append(transform[:3, 3])

        # Create lines between consecutive points
        for i in range(len(points) - 1):
            lines.append([i, i + 1])

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

        # Set color
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

        return line_set

    def plot_transform(
        self,
        transform: Union[np.ndarray, torch.Tensor],
        frame_size: float = 1.0,
        show_grid: bool = True,
    ) -> None:
        """
        Plot a single homogeneous transformation

        Args:
            transform: 4x4 homogeneous transformation matrix
            frame_size: Size of coordinate frames
            show_grid: Whether to show reference grid
        """
        # Convert torch tensor to numpy if needed
        if isinstance(transform, torch.Tensor):
            transform = transform.detach().cpu().numpy()

        # Add world coordinate frame
        world_frame = self.create_coordinate_frame(size=frame_size)
        self.vis.add_geometry(world_frame)

        # Add transformed coordinate frame
        transformed_frame = self.create_coordinate_frame(size=frame_size, transform=transform)
        self.vis.add_geometry(transformed_frame)

        # Add reference grid
        if show_grid:
            grid = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.001)
            grid.compute_vertex_normals()
            grid.paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    grid_copy = copy.deepcopy(grid)
                    grid_copy.translate(np.array([i, j, -0.5]))
                    self.vis.add_geometry(grid_copy)

        # Set default viewpoint
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.5])
        ctr.set_up([0.0, 1.0, 0.0])

    def plot_transform_sequence(
        self,
        transforms: List[Union[np.ndarray, torch.Tensor]],
        frame_size: float = 1.0,
        show_trajectory: bool = True,
        show_grid: bool = True,
    ) -> None:
        """
        Plot a sequence of homogeneous transformations

        Args:
            transforms: List of 4x4 homogeneous transformation matrices
            frame_size: Size of coordinate frames
            show_trajectory: Whether to show trajectory line
            show_grid: Whether to show reference grid
        """
        # Convert torch tensors to numpy if needed
        transforms_np = []
        for transform in transforms:
            if isinstance(transform, torch.Tensor):
                transforms_np.append(transform.detach().cpu().numpy())
            else:
                transforms_np.append(transform)

        # Add world coordinate frame
        world_frame = self.create_coordinate_frame(size=frame_size)
        self.vis.add_geometry(world_frame)

        # Add transformed coordinate frames
        for transform in transforms_np:
            frame = self.create_coordinate_frame(size=frame_size, transform=transform)
            self.vis.add_geometry(frame)

        # Add trajectory line
        if show_trajectory:
            trajectory = self.create_trajectory_line(transforms_np)
            self.vis.add_geometry(trajectory)

        # Add reference grid
        if show_grid:
            grid = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.001)
            grid.compute_vertex_normals()
            grid.paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    grid_copy = copy.deepcopy(grid)
                    grid_copy.translate(np.array([i, j, -0.5]))
                    self.vis.add_geometry(grid_copy)

        # Set default viewpoint
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.5])
        ctr.set_up([0.0, 1.0, 0.0])

    def show(self):
        """Run the visualizer"""
        self.vis.run()
        self.vis.destroy_window()


def visualize_pairs_with_motion_axis(pairs, pairs_est, axis, motion_center, point_cloud=None):
    """
    Visualize the pairs of 3D points along with the estimated axis of motion and center of motion.

    Args:
        pairs (list): List of pairs of 3D points for each frame.
        pairs_est (list): List of estimated pairs of 3D points for each frame.
        axis (np.ndarray): Estimated axis of motion.
        motion_center (np.ndarray): Estimated center of motion.
        point_cloud (np.ndarray, optional): Point cloud data to be plotted. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original pairs
    for pair in pairs:
        ax.plot(
            pair[:, 0, 0],
            pair[:, 0, 1],
            pair[:, 0, 2],
            "bo-",
            label="Original Pair" if pair is pairs[0] else "",
        )
        ax.plot(
            pair[:, 1, 0],
            pair[:, 1, 1],
            pair[:, 1, 2],
            "ro-",
            label="Original Pair" if pair is pairs[0] else "",
        )

    # Plot the estimated pairs
    for pair_est in pairs_est:
        ax.plot(
            pair_est[:, 0],
            pair_est[:, 1],
            pair_est[:, 2],
            "go-",
            label="Estimated Pair" if pair_est is pairs_est[0] else "",
        )

    # Plot the axis of motion
    axis_line = np.array([motion_center - axis * 1.5, motion_center + axis * 1.5])
    ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], "k-", label="Axis of Motion")

    # Plot the center of motion
    ax.scatter(
        motion_center[0],
        motion_center[1],
        motion_center[2],
        c="k",
        marker="x",
        s=100,
        label="Center of Motion",
    )

    # Plot the point cloud if provided
    if point_cloud is not None:
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c="gray",
            marker=".",
            s=1,
            label="Point Cloud",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def plot_3d_tracks(
    points,
    visibles,
    rgb_images,
    depth_images,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False,
    fps=15,
    azure_dataset=None,
    clusters_labels_segments=None,
    camera_poses=None,
    save_frames=False,
    output_dir=None,
    output_prefix="frame_",
    save_video=False,
    resolution=(1280, 720),
):
    """
    Visualize 3D point trajectories frame-by-frame using Open3D.
    Args:
        points (np.ndarray): 3D points of shape (num_frames, num_points, 3) it should be in Global coordinate system.
        visibles (np.ndarray): Visibility of points of shape (num_frames, num_points).
        rgb_images (np.ndarray): RGB images of shape (num_frames, H, W, 3).
        depth_images (np.ndarray): Depth images of shape (num_frames, H, W).
        infront_cameras (np.ndarray): Visibility of points in front of cameras of shape (num_frames, num_points).
        tracks_leave_trace (int): Number of frames to leave trace for each point.
        show_occ (bool): Show occlusion if True, else show visibility.
        fps (int): Frames per second for visualization.
        azure_dataset Azure dataset processor.
        clusters_labels_segments (np.ndarray): Cluster labels for each point.
        camera_poses (np.ndarray): Camera poses for each frame.
        save_frames (bool): Whether to save rendered frames as images.
        output_dir (str): Directory to save rendered frames. Created if it doesn't exist.
        output_prefix (str): Prefix for output frame filenames.
        save_video (bool): Whether to compile frames into a video (requires ffmpeg).
        resolution (tuple): Width and height resolution for rendering window.
    """

    num_frames, num_points = points.shape[0:2]

    # Create distinct colors for different clusters
    color_map = cm.get_cmap("tab10")  # Using tab10 colormap for distinct cluster colors

    # Default colors if no clusters provided
    if clusters_labels_segments is None:
        colors = [color_map(i % 10)[:3] for i in range(num_points)]
    else:
        # Get unique cluster IDs
        unique_clusters = np.unique(clusters_labels_segments)
        cluster_colors = {c: color_map(i % 10)[:3] for i, c in enumerate(unique_clusters)}
        # Assign colors based on cluster labels
        colors = [cluster_colors[clusters_labels_segments[i]] for i in range(num_points)]

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    # Create output directory if saving frames
    if save_frames and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to {output_dir}")

    # Setup visualizer with specified resolution
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])

    # Set rendering options for better quality output
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # White background

    # Keep track of geometries and camera parameters
    camera_params = None  # To store the camera viewpoint
    scene_pc = o3d.geometry.PointCloud()
    frame_files = []

    # add scene pcd as initial frame
    if azure_dataset is not None:
        pcd, _ = azure_dataset.create_pcd(rgb_images[0], depth_images[0], camera_poses[0])
        scene_pc += pcd

    for t in range(num_frames):
        if t > 0:
            vis.clear_geometries()  # Clear previous frame's geometries

        vis.add_geometry(scene_pc)  # Add the scene point cloud

        # add the scene point cloud to the visualizer
        if azure_dataset is not None:
            pcd, _ = azure_dataset.create_pcd(rgb_images[t], depth_images[t], camera_poses[t])
            # scene_pc += pcd
            # scene_pc = scene_pc.voxel_down_sample(voxel_size=0.03)
            vis.add_geometry(pcd)

        for i in range(num_points):
            if show_occ:
                visible = infront_cameras[t, i]
            else:
                visible = visibles[t, i]

            if visible:
                # Create sphere for current position
                color = colors[i]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(points[t, i])
                sphere.paint_uniform_color(color)
                vis.add_geometry(sphere)

                # Trace the trajectory
                trace_start = max(0, t - tracks_leave_trace)
                trace = points[trace_start : t + 1, i]
                trace_vis = visibles[trace_start : t + 1, i]
                trace = trace[trace_vis]
                if len(trace) > 1:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(trace)
                    lines = [[k, k + 1] for k in range(len(trace) - 1)]
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                    vis.add_geometry(line_set)

        if camera_params is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        # Save frame if requested
        if save_frames and output_dir is not None:
            frame_path = os.path.join(output_dir, f"{output_prefix}{t:04d}.png")
            vis.capture_screen_image(frame_path)
            frame_files.append(frame_path)
            print(f"Saved frame {t}/{num_frames} to {frame_path}")

        # Save the current camera parameters for the next frame
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    # Compile video if requested
    if save_video and save_frames and frame_files:
        video_path = os.path.join(output_dir, f"{output_prefix}video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, f"{output_prefix}%04d.png"),
                # "-c:v",
                # "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print("To manually create video, use ffmpeg:")
            print(f"ffmpeg -framerate {fps} -i {output_dir}/{output_prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

    return frame_files if save_frames else None


def create_partial_torus_arrow(
    center, axis, major_radius=0.12, minor_radius=0.015, arc_degrees=300, n_major=40, n_minor=12, cone_radius=0.04, cone_length=0.06, color=(1, 0, 0)
):
    """300-deg arc tube with an arrowhead at the open end, ring plane perpendicular to `axis`."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    arc_rad = np.radians(arc_degrees)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    verts = []
    tris = []
    for i in range(n_major + 1):
        a_maj = arc_rad * i / n_major
        radial = np.cos(a_maj) * u + np.sin(a_maj) * v
        tube_c = major_radius * radial
        for j in range(n_minor):
            a_min = 2 * np.pi * j / n_minor
            verts.append(tube_c + minor_radius * (np.cos(a_min) * radial + np.sin(a_min) * axis))

    for i in range(n_major):
        for j in range(n_minor):
            i1 = i * n_minor + j
            i2 = i * n_minor + (j + 1) % n_minor
            i3 = (i + 1) * n_minor + j
            i4 = (i + 1) * n_minor + (j + 1) % n_minor
            tris.append([i1, i2, i4])
            tris.append([i1, i4, i3])

    tube = o3d.geometry.TriangleMesh()
    tube.vertices = o3d.utility.Vector3dVector(np.array(verts))
    tube.triangles = o3d.utility.Vector3iVector(np.array(tris))
    tube.compute_vertex_normals()
    tube.translate(center)
    tube.paint_uniform_color(color)

    radial_end = np.cos(arc_rad) * u + np.sin(arc_rad) * v
    tangent_end = -np.sin(arc_rad) * u + np.cos(arc_rad) * v
    tip_pos = center + major_radius * radial_end

    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_length, resolution=20)
    cone.compute_vertex_normals()
    cone.translate((0, 0, cone_length / 2))
    z_vec = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(z_vec, tangent_end)
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_vec, tangent_end), -1, 1))
        Rmat = cone.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cone.rotate(Rmat, center=(0, 0, 0))
    cone.translate(tip_pos)
    cone.paint_uniform_color(color)

    return tube, cone


def cylinder_cone_arrow(start, end, cyl_radius=0.03, cone_radius=0.06, cone_length=0.2, color=(1, 0, 0)):
    """Cylinder shaft + cone tip arrow from `start` to `end`."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    total_length = np.linalg.norm(direction)
    direction /= total_length

    shaft_length = total_length - cone_length
    if shaft_length <= 0:
        raise ValueError("cone_length must be smaller than arrow length")

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=shaft_length, resolution=30)
    cyl.compute_vertex_normals()
    cyl.translate((0, 0, shaft_length / 2))

    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_length, resolution=30)
    cone.compute_vertex_normals()
    cone.translate((0, 0, cone_length / 2))

    z = np.array([0, 0, 1])
    rot_axis = np.cross(z, direction)
    angle = np.arccos(np.clip(np.dot(z, direction), -1, 1))
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis /= np.linalg.norm(rot_axis)
        Rmat = cyl.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cyl.rotate(Rmat, center=(0, 0, 0))
        cone.rotate(Rmat, center=(0, 0, 0))

    cyl.translate(start)
    cone.translate(start + (shaft_length - cone_length / 2) * direction)
    cyl.paint_uniform_color(color)
    cone.paint_uniform_color(color)

    return cyl, cone


def plot_3d_articulated_object(
    points,
    articulation_model,
    rgb_images,
    depth_images=None,
    camera_poses=None,
    azure_dataset=None,
    axis_length=0.3,
    axis_color=(1.0, 0.0, 0.0),
    object_color=(0.2, 0.6, 1.0),
    fps=15,
    save_frames=False,
    output_dir=None,
    output_prefix="frame_",
    save_video=False,
    resolution=(1280, 720),
    children=None,
    plot_children=True,
):
    """
    Visualize an articulated object with its segmentation mask and articulation axis frame-by-frame.

    The object mask is rendered in a distinct color; the articulation axis is drawn as a line
    through the joint center. If points is 2D (num_points, 3) the articulation model is applied
    to transform the reference frame at each joint state; if 3D (num_frames, num_points, 3)
    the per-frame positions are used directly.

    Args:
        points: Point cloud, shape (num_points, 3) or (num_frames, num_points, 3).
        articulation_model: InferencePointAxis with fields: position, axis, type, thetas.
        rgb_images: RGB images of shape (num_frames, H, W, 3).
        depth_images: Depth images of shape (num_frames, H, W).
        camera_poses: Camera-to-world transforms of shape (num_frames, 4, 4).
        azure_dataset: instance for building scene point clouds per frame.
        axis_length: Length of the drawn articulation axis line.
        axis_color: RGB tuple for the articulation axis line and joint sphere.
        object_color: RGB tuple for the articulated object points.
        fps: Frames per second used for the saved video and interactive playback delay.
        save_frames: Whether to save rendered frames as images.
        output_dir: Directory to save rendered frames. Created if it does not exist.
        output_prefix: Prefix for output frame filenames.
        save_video: Whether to compile frames into a video (requires ffmpeg).
        resolution: (width, height) of the rendering window.
        children: Optional list of child objects from hierarchy.children.  Each entry
            must have a ``"pcd"`` (o3d.geometry.PointCloud in world frame),
            ``"relation"`` ("ARTICULATED" | "STATIC"), and ``"inst_color"``.
            ARTICULATED children are transformed with the same twist as the parent;
            STATIC children are rendered at their original world-frame position.
    """
    if points.ndim == 3:
        num_frames, num_points = points.shape[:2]
        precomputed = True
    else:
        num_points = points.shape[0]
        num_frames = len(rgb_images)
        precomputed = False

    def _apply_transform(pts, theta):
        """Apply the articulation model's 6D twist via GTSAM's exponential map."""
        pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
        transformed = gtsam.Pose3.Expmap(np.array(articulation_model.twist) * theta).matrix() @ pts_homog.T
        return transformed[:3].T

    if save_frames and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to {output_dir}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.array([1, 1, 1])

    obj_color = np.random.rand(3)

    camera_params = None
    frame_files = []

    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar

    if np.isnan(articulation_model.dense_thetas).any():
        valid_indices = np.where(~np.isnan(articulation_model.dense_thetas))[0]
        if len(valid_indices) < 10:
            print("Segment has fewer than 10 valid theta estimates, skipping envelope-based children identification.")
        elif not precomputed and depth_images is not None and camera_poses is not None and azure_dataset is not None:
            # Optimize theta for each missing frame by minimizing depth reprojection error.
            K = np.array(azure_dataset.depth_intrinsics)
            twist = np.array(articulation_model.twist)
            valid_thetas = np.array(articulation_model.dense_thetas)[valid_indices]
            theta_lo = float(valid_thetas.min())
            theta_hi = float(valid_thetas.max())
            margin = max(abs(theta_hi - theta_lo) * 0.2, 1e-3)

            def _reprojection_cost(theta, frame_idx):
                """
                Depth-reprojection error for a candidate joint angle at a given frame:
                projects the articulated point cloud into the camera and compares
                to the observed depth (used by minimize_scalar to fill in missing thetas).

                Args:
                    theta: Candidate articulation joint angle.
                    frame_idx: Index of the frame whose depth image to reproject against.

                Returns:
                    float: Mean squared depth-reprojection error (np.inf if too few
                        points land in front of/inside the camera or have valid depth).
                """
                pts_h = np.hstack((points, np.ones((points.shape[0], 1))))
                world_pts = (gtsam.Pose3.Expmap(twist * theta).matrix() @ pts_h.T).T[:, :3]
                T_wc = np.linalg.inv(camera_poses[frame_idx])
                cam_pts = (T_wc @ np.hstack((world_pts, np.ones((world_pts.shape[0], 1)))).T).T[:, :3]
                in_front = cam_pts[:, 2] > 0
                if in_front.sum() < 5:
                    return np.inf
                cam_pts = cam_pts[in_front]
                u = np.round(K[0, 0] * cam_pts[:, 0] / cam_pts[:, 2] + K[0, 2]).astype(int)
                v = np.round(K[1, 1] * cam_pts[:, 1] / cam_pts[:, 2] + K[1, 2]).astype(int)
                H, W = depth_images[frame_idx].shape
                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if in_bounds.sum() < 5:
                    return np.inf
                depth_obs = depth_images[frame_idx][v[in_bounds], u[in_bounds]] / 1000.0
                z_proj = cam_pts[in_bounds, 2]
                valid_depth = depth_obs > 0
                if valid_depth.sum() < 5:
                    return np.inf
                return float(np.mean((z_proj[valid_depth] - depth_obs[valid_depth]) ** 2))

            nan_indices = np.where(np.isnan(articulation_model.dense_thetas))[0]
            for nan_idx in nan_indices:
                res = minimize_scalar(
                    _reprojection_cost,
                    bounds=(theta_lo - margin, theta_hi + margin),
                    method='bounded',
                    args=(int(nan_idx),),
                )
                articulation_model.dense_thetas[nan_idx] = res.x
        else:
            interp_func = interp1d(
                np.array(articulation_model.dense_frame_indices)[valid_indices],
                np.array(articulation_model.dense_thetas)[valid_indices],
                kind='cubic',
                fill_value="extrapolate",
            )
            nan_indices = np.where(np.isnan(articulation_model.dense_thetas))[0]
            for nan_idx in nan_indices:
                articulation_model.dense_thetas[nan_idx] = interp_func(np.array(articulation_model.dense_frame_indices)[nan_idx]).squeeze().item()

    thetas = articulation_model.dense_thetas

    # ARTICULATED children were captured at the max-opening frame.
    # Pre-compute T(theta_max)^{-1} once so each frame only needs T(theta) @ T_max_inv.
    child_T_max_inv = None
    if children and plot_children:
        theta_max = float(np.nanmax(articulation_model.dense_thetas))
        child_T_max_inv = np.linalg.inv(gtsam.Pose3.Expmap(np.array(articulation_model.twist) * theta_max).matrix())

    for t in range(num_frames):
        if t > 0:
            vis.clear_geometries()

        # Joint state for this frame
        theta = float(thetas[t]) if thetas is not None and len(thetas) > t else 0.0
        if precomputed:
            obj_pts = points[t]
        else:
            obj_pts = _apply_transform(points, theta)

        # Pre-transform children for this theta so we reuse the result for both
        # depth hollowing and rendering.
        # ARTICULATED children: undo max-opening transform, then apply current theta.
        # STATIC children: remain at their captured world-frame position.
        children_pts_this_frame = []
        if children and plot_children:
            T_curr = gtsam.Pose3.Expmap(np.array(articulation_model.twist) * theta).matrix()
            T_child = T_curr @ child_T_max_inv  # T(theta) @ T(theta_max)^{-1}
            for child in children:
                c_pts = np.asarray(child["pcd"].points)
                if child.get("relation") == "ARTICULATED":
                    pts_h = np.hstack((c_pts, np.ones((c_pts.shape[0], 1))))
                    c_pts = (T_child @ pts_h.T)[:3].T
                c_color = np.asarray(child.get("inst_color", np.random.rand(3)))
                children_pts_this_frame.append((c_pts, c_color))

        if azure_dataset is not None and depth_images is not None and camera_poses is not None:
            pcd, _ = azure_dataset.create_pcd(rgb_images[t], depth_images[t], camera_poses[t])
            if obj_pts.shape[0] > 0 and np.isnan(obj_pts).sum() == 0:
                pcd = remove_live_depth_near_object(pcd, obj_pts, radius=0.015)
            for c_pts, _ in children_pts_this_frame:
                if c_pts.shape[0] > 0 and not np.isnan(c_pts).any():
                    pcd = remove_live_depth_near_object(pcd, c_pts, radius=0.015)
            vis.add_geometry(pcd)

        if obj_pts.shape[0] > 0 and np.isnan(obj_pts).sum() == 0:
            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(obj_pts)
            obj_pc.paint_uniform_color(obj_color)
            vis.add_geometry(obj_pc)

        for c_pts, c_color in children_pts_this_frame:
            if c_pts.shape[0] > 0 and not np.isnan(c_pts).any():
                child_pc = o3d.geometry.PointCloud()
                child_pc.points = o3d.utility.Vector3dVector(c_pts)
                child_pc.paint_uniform_color(c_color)
                vis.add_geometry(child_pc)

        # Articulation axis: cylinder+cone arrow, plus arc indicator for revolute joints
        if articulation_model.axis is not None and articulation_model.position is not None:
            axis_dir = np.array(articulation_model.axis, dtype=float)
            axis_dir /= np.linalg.norm(axis_dir)
            axis_offset = 0.05 * axis_dir
            axis_start = np.array(articulation_model.position) - (axis_length / 2) * axis_dir + axis_offset
            axis_end = np.array(articulation_model.position) + (axis_length / 2) * axis_dir + axis_offset
            cyl, cone = cylinder_cone_arrow(
                axis_start,
                axis_end,
                cyl_radius=0.03,
                cone_radius=0.056,
                cone_length=0.084,
                color=tuple(axis_color),
            )
            vis.add_geometry(cyl)
            vis.add_geometry(cone)
            if articulation_model.type == "revolute":
                arc_center = axis_start + (1 / 3) * (axis_end - axis_start)
                arc_tube, arc_cone = create_partial_torus_arrow(
                    arc_center,
                    articulation_model.axis,
                    major_radius=0.12,
                    minor_radius=0.015,
                    color=(0.0, 0.0, 1.0),
                )
                vis.add_geometry(arc_tube)
                vis.add_geometry(arc_cone)

        if camera_params is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        if save_frames and output_dir is not None:
            frame_path = os.path.join(output_dir, f"{output_prefix}{t:04d}.png")
            vis.capture_screen_image(frame_path)
            frame_files.append(frame_path)
            print(f"Saved frame {t}/{num_frames} to {frame_path}")

        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    if save_video and save_frames and frame_files:
        video_path = os.path.join(output_dir, f"{output_prefix}video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, f"{output_prefix}%04d.png"),
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print(f"ffmpeg -framerate {fps} -i {output_dir}/{output_prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

    return frame_files if save_frames else None


def plot_full_demo_video(
    segment_objs,
    pred_segments,
    rgb_images,
    depth_images,
    camera_poses,
    azure_dataset,
    fps=15,
    output_dir="demo_video/",
    resolution=(1280, 720),
    save_video=True,
    egocentric=True,
    point_size=3.0,
):
    """
    Visualize the entire demonstration as a single 3D video.

    For every frame in the recording the live scene point cloud is rendered.
    When a frame falls within an interaction segment the corresponding articulated
    object is overlaid in a unique random color at its current joint state together
    with its axis.  At the end of each interaction the object, its axis, and the
    terminal depth scan are frozen into a growing map that persists in all
    subsequent frames.

    Args:
        segment_objs: dict mapping (start_idx, end_idx) -> hierarchy object.
        pred_segments: ordered list of (start_idx, end_idx) tuples.
        rgb_images: list of all RGB frames (H, W, 3).
        depth_images: list of all depth frames (H, W).
        camera_poses: list of camera-to-world 4×4 poses.
        azure_dataset: for building live scene point clouds.
        fps: frame rate for the saved video.
        output_dir: directory for frames and the video.
        resolution: (width, height) render window size.
        save_video: if True compile all frames into an mp4 via ffmpeg.
        egocentric: if True the viewpoint tracks the recorded camera trajectory
            (elevated 50 cm, tilted 30° down).  If False a static viewpoint is
            used that the user can freely adjust in the Open3D window while the
            video plays; the adjusted view is preserved across frames.
        point_size: Open3D point size for all rendered point clouds.

    Returns:
        list of saved frame paths.
    """
    from scipy.interpolate import interp1d

    os.makedirs(output_dir, exist_ok=True)
    map_dir = os.path.join(output_dir, "map_frames")
    os.makedirs(map_dir, exist_ok=True)

    num_frames = len(rgb_images)
    H, W = rgb_images[0].shape[:2]

    # Build Open3D intrinsics from azure_dataset for camera-following
    K = azure_dataset.depth_intrinsics  # (3, 3)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    # Assign a stable random color per segment so objects are distinguishable.
    # Colors are generated in HSV with high saturation and brightness so they
    # appear vibrant; only the hue varies between segments.
    rng = np.random.default_rng(42)
    from matplotlib.colors import hsv_to_rgb

    seg_colors = {seg: hsv_to_rgb([rng.random(), 0.85, 0.95]) for seg in pred_segments}

    # Build per-frame lookup: frame_idx -> (start, end, obj)
    frame_to_seg = {}
    for start_idx, end_idx in pred_segments:
        obj = segment_objs.get((start_idx, end_idx))
        if obj is None:
            continue
        for t in range(start_idx, end_idx):
            frame_to_seg[t] = (start_idx, end_idx, obj)

    # Pre-interpolate dense_thetas for every segment
    def _interpolate_thetas(obj, seg_len):
        """Fill NaN entries in obj's dense_thetas via cubic interpolation, or zeros if unavailable."""
        model = obj["model"]
        raw = model.dense_thetas
        if raw is None or len(raw) == 0:
            return [0.0] * seg_len
        thetas = list(raw)
        if any(np.isnan(thetas)):
            valid_idx = [i for i, v in enumerate(thetas) if not np.isnan(v)]
            if len(valid_idx) >= 2:
                indices = list(model.dense_frame_indices) if model.dense_frame_indices is not None else list(range(len(thetas)))
                f = interp1d(
                    np.array(indices)[valid_idx],
                    np.array(thetas)[valid_idx],
                    kind="cubic",
                    fill_value="extrapolate",
                )
                thetas = [float(f(idx)) for idx in indices]
        return thetas

    seg_thetas = {}
    for start_idx, end_idx in pred_segments:
        obj = segment_objs.get((start_idx, end_idx))
        if obj is None:
            continue
        seg_thetas[(start_idx, end_idx)] = _interpolate_thetas(obj, end_idx - start_idx)

    # ------------------------------------------------------------------ helpers

    def _apply_transform(pts, model, theta):
        """Apply the articulation model's 6D twist, scaled by theta, to a set of points."""
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        T = gtsam.Pose3.Expmap(np.array(model.twist) * theta).matrix()
        return (T @ pts_h.T)[:3].T

    def _make_frozen_pcd(obj, theta, color):
        """Return a colored PointCloud of the object frozen at the given theta."""
        pts = _apply_transform(np.asarray(obj["pcd"].points), obj["model"], theta)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        return pcd

    def _make_axis_geoms(model, color):
        """Return cylinder+cone (and torus for revolute) for the articulation axis."""
        geoms = []
        if model.axis is None or model.position is None:
            return geoms
        axis_dir = np.array(model.axis, dtype=float)
        axis_dir /= np.linalg.norm(axis_dir)
        offset = 0.15 * axis_dir
        start = np.array(model.position) - 0.15 * axis_dir + offset
        end = np.array(model.position) + 0.15 * axis_dir + offset
        cyl, cone = cylinder_cone_arrow(
            start,
            end,
            cyl_radius=0.03,
            cone_radius=0.056,
            cone_length=0.084,
            color=tuple(color),
        )
        geoms += [cyl, cone]
        if model.type == "revolute":
            arc_center = start + (1 / 3) * (end - start)
            arc_tube, arc_cone = create_partial_torus_arrow(
                arc_center,
                model.axis,
                major_radius=0.12,
                minor_radius=0.015,
                color=tuple(color),
            )
            geoms += [arc_tube, arc_cone]
        return geoms

    def _apply_camera(vis, pose_c2w, saved_params):
        """Set the viewpoint according to the egocentric flag.

        egocentric=True  → follow pose_c2w (elevated 50 cm, tilted 30° down).
        egocentric=False → restore saved_params so the user's interactive
                           adjustments are preserved; falls back to pose_c2w
                           on the very first frame before any params are saved.
        """
        if not egocentric and saved_params is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(saved_params, allow_arbitrary=True)
            except TypeError:
                vis.get_view_control().convert_from_pinhole_camera_parameters(saved_params)
            return

        # Egocentric mode (or first frame of static mode): derive from pose_c2w
        modified = pose_c2w.copy().astype(float)

        # Elevate 50 cm along camera up direction.
        # Camera Y points DOWN in OpenCV convention, so camera up = -col1.
        # cam_up = -modified[:3, 1]
        # cam_up /= np.linalg.norm(cam_up)
        # modified[:3, 3] += 0.5 * cam_up

        # Tilt 30° downward: positive rotation around camera right axis (col0).
        # In OpenCV convention (+Y down, +Z forward) a positive rotation around
        # the right axis pitches the forward vector toward the down direction.
        # cam_right = modified[:3, 0]
        # cam_right /= np.linalg.norm(cam_right)
        # R_tilt = R.from_rotvec(np.radians(30) * cam_right).as_matrix()
        # modified[:3, :3] = R_tilt @ modified[:3, :3]

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d_intrinsic
        params.extrinsic = np.linalg.inv(modified)
        try:
            vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:
            # older Open3D builds don't have allow_arbitrary
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)

    # ----------------------------------------------------------------- main loop

    # Accumulated map from completed interactions
    map_obj_pcds = []  # frozen articulated-object point clouds (used for depth rejection too)
    map_depth_pcds = []  # frozen terminal depth scans (rendered only, not used for rejection)
    map_geoms = []  # frozen axis meshes (TriangleMesh)

    # Preserved camera params for non-egocentric (static) mode
    static_camera_params = None

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([1.0, 1.0, 1.0])

    frame_files = []

    for t in range(num_frames):
        vis.clear_geometries()

        # Live scene point cloud
        scene_pcd, _ = azure_dataset.create_pcd(rgb_images[t], depth_images[t], camera_poses[t])

        # Compute active articulation state so we can hollow out the scene PCD
        active_seg = frame_to_seg.get(t)
        if active_seg is not None:
            start_idx, end_idx, obj = active_seg
            rel_t = t - start_idx
            thetas = seg_thetas[(start_idx, end_idx)]
            theta = float(thetas[rel_t]) if rel_t < len(thetas) else 0.0
            cur_pts = _apply_transform(np.asarray(obj["pcd"].points), obj["model"], theta)
            if cur_pts.shape[0] > 0 and not np.isnan(cur_pts).any():
                scene_pcd = remove_live_depth_near_object(scene_pcd, cur_pts, radius=0.015)

        # Also hollow live depth near every frozen articulated object (2 cm)
        if map_obj_pcds:
            all_obj_pts = np.concatenate([np.asarray(p.points) for p in map_obj_pcds], axis=0)
            scene_pcd = remove_live_depth_near_object(scene_pcd, all_obj_pts, radius=0.02)

        vis.add_geometry(scene_pcd)

        # Accumulated map: frozen objects, terminal depth scans, axis meshes
        for pcd in map_obj_pcds:
            vis.add_geometry(pcd)
        for pcd in map_depth_pcds:
            vis.add_geometry(pcd)
        for geom in map_geoms:
            vis.add_geometry(geom)

        # Active interaction: animated object + axis
        if active_seg is not None:
            color = seg_colors[(start_idx, end_idx)]

            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(cur_pts)
            obj_pc.paint_uniform_color(color)
            vis.add_geometry(obj_pc)

            for geom in _make_axis_geoms(obj["model"], color):
                vis.add_geometry(geom)

            # Interaction just ended: freeze object + depth scan into the growing map
            if t == end_idx - 1:
                frozen_pcd = _make_frozen_pcd(obj, theta, color)
                frozen_geoms = _make_axis_geoms(obj["model"], color)
                map_obj_pcds.append(frozen_pcd)
                map_geoms.extend(frozen_geoms)

                # Freeze the terminal depth scan (scene already hollowed near the object)
                terminal_depth = copy.deepcopy(scene_pcd)
                map_depth_pcds.append(terminal_depth)

                # Static map snapshot at the moment the interaction closes
                seg_idx = pred_segments.index((start_idx, end_idx))
                _apply_camera(vis, camera_poses[t], static_camera_params)
                vis.poll_events()
                vis.update_renderer()
                snap_path = os.path.join(map_dir, f"map_after_interaction_{seg_idx:04d}.png")
                vis.capture_screen_image(snap_path)
                print(f"Map snapshot saved to {snap_path}")

        # Apply viewpoint and render
        _apply_camera(vis, camera_poses[t], static_camera_params)
        vis.poll_events()
        vis.update_renderer()

        # In static mode: read back whatever view the user has set so it is
        # preserved across frames (including interactive adjustments mid-playback)
        if not egocentric:
            static_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        frame_path = os.path.join(output_dir, f"frame_{t:06d}.png")
        vis.capture_screen_image(frame_path)
        frame_files.append(frame_path)

    vis.destroy_window()

    if save_video and frame_files:
        video_path = os.path.join(output_dir, "demo_video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, "frame_%06d.png"),
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Demo video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")

    return frame_files


def plot_full_momasg(
    segment_objs,
    pred_segments,
    rgb_images,
    depth_images,
    camera_poses,
    azure_dataset,
    hierarchy=None,
    human_masks=None,
    fps=15,
    output_dir="momasg_video/",
    resolution=(1920, 1080),
    save_video=True,
    point_size=3.0,
    node_radius=0.05,
    graph_height_obj=0.75,
    graph_height_child_offset=0.3,
    graph_height_root_offset=0.5,
    orbit_revolutions=2,
    orbit_frames_per_revolution=120,
):
    """
    Exo-centric scene-graph visualization that grows an overlay graph as interactions complete.

    The 3D scene and articulated objects are rendered exactly as in plot_full_demo_video
    (exo-centric / static camera).  On top of that, a floating graph is built frame by frame:

      Root node  – grey sphere at mean XY of all articulated objects, Z = highest object Z + 1.5 m.
      Object node – colored sphere at (obj_centroid_XY, graph_height_obj), same color as the
                    animated point cloud.  Added the first frame an interaction becomes active.
                    An edge connects Root → Object node, and a grey pointer line runs from the
                    Object node down to the physical object centroid.
      Child nodes – smaller spheres 0.3 m below the Object node, colored by child inst_color.
                    Added together with edges Object → Child when an interaction finishes.

    Args:
        segment_objs: dict mapping (start_idx, end_idx) -> hierarchy object dict.
        pred_segments: ordered list of (start_idx, end_idx) tuples.
        rgb_images: list of RGB frames (H, W, 3).
        depth_images: list of depth frames (H, W), values in mm.
        camera_poses: list of camera-to-world 4×4 poses.
        azure_dataset: KinectRGBDDataset instance for live scene point clouds and intrinsics.
        hierarchy: Hierarchy instance giving access to hierarchy.children (needed for child nodes).
        fps: frame rate for the saved video.
        output_dir: directory for frames and the video.
        resolution: (width, height) render window size.
        save_video: if True compile frames into an mp4 via ffmpeg.
        point_size: Open3D point size.
        node_radius: radius (m) of object-node spheres; root is 1.5×, children are 0.8×.
        graph_height_obj: world-Z coordinate for object nodes (m).
        graph_height_child_offset: how far below the object node child nodes sit (m).
        graph_height_root_offset: how far above the object nodes the root sits (m).
        orbit_revolutions: number of full camera revolutions appended after all frames.
        orbit_frames_per_revolution: frames rendered per revolution (controls orbit speed).
        human_masks: optional boolean array (num_frames, H, W); when provided the terminal
            depth scan frozen at the end of each interaction is recreated with these pixels
            masked out so the human is excluded from the persistent map.

    Returns:
        list of saved frame paths.
    """
    from scipy.interpolate import interp1d

    os.makedirs(output_dir, exist_ok=True)
    num_frames = len(rgb_images)
    H_img, W_img = rgb_images[0].shape[:2]

    K = azure_dataset.depth_intrinsics
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W_img, H_img, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    rng = np.random.default_rng(42)
    from matplotlib.colors import hsv_to_rgb

    seg_colors = {seg: hsv_to_rgb([rng.random(), 0.85, 0.95]) for seg in pred_segments}

    frame_to_seg = {}
    for start_idx, end_idx in pred_segments:
        obj = segment_objs.get((start_idx, end_idx))
        if obj is None:
            continue
        for t in range(start_idx, end_idx):
            frame_to_seg[t] = (start_idx, end_idx, obj)

    def _interpolate_thetas(obj, seg_len):
        """Fill NaN entries in obj's dense_thetas via cubic interpolation, or zeros if unavailable."""
        model = obj["model"]
        raw = model.dense_thetas
        if raw is None or len(raw) == 0:
            return [0.0] * seg_len
        thetas = list(raw)
        if any(np.isnan(thetas)):
            valid_idx = [i for i, v in enumerate(thetas) if not np.isnan(v)]
            if len(valid_idx) >= 2:
                indices = list(model.dense_frame_indices) if model.dense_frame_indices is not None else list(range(len(thetas)))
                f = interp1d(
                    np.array(indices)[valid_idx],
                    np.array(thetas)[valid_idx],
                    kind="cubic",
                    fill_value="extrapolate",
                )
                thetas = [float(f(idx)) for idx in indices]
        return thetas

    seg_thetas = {}
    for start_idx, end_idx in pred_segments:
        obj = segment_objs.get((start_idx, end_idx))
        if obj is None:
            continue
        seg_thetas[(start_idx, end_idx)] = _interpolate_thetas(obj, end_idx - start_idx)

    # ------------------------------------------------------------------ helpers

    def _apply_transform(pts, model, theta):
        """Apply the articulation model's 6D twist, scaled by theta, to a set of points."""
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        T = gtsam.Pose3.Expmap(np.array(model.twist) * theta).matrix()
        return (T @ pts_h.T)[:3].T

    def _make_axis_geoms(model, color):
        """Return cylinder+cone (and torus for revolute) geometries for the articulation axis."""
        geoms = []
        if model.axis is None or model.position is None:
            return geoms
        axis_dir = np.array(model.axis, dtype=float)
        axis_dir /= np.linalg.norm(axis_dir)
        offset = 0.15 * axis_dir
        start = np.array(model.position) - 0.15 * axis_dir + offset
        end = np.array(model.position) + 0.15 * axis_dir + offset
        cyl, cone = cylinder_cone_arrow(
            start,
            end,
            cyl_radius=0.03,
            cone_radius=0.056,
            cone_length=0.084,
            color=tuple(color),
        )
        geoms += [cyl, cone]
        if model.type == "revolute":
            arc_center = start + (1 / 3) * (end - start)
            arc_tube, arc_cone = create_partial_torus_arrow(
                arc_center,
                model.axis,
                major_radius=0.12,
                minor_radius=0.015,
                color=tuple(color),
            )
            geoms += [arc_tube, arc_cone]
        return geoms

    def _cube(center, half_size, color):
        """Create a solid colored box mesh centered at center with the given half-size."""
        s = half_size * 2
        mesh = o3d.geometry.TriangleMesh.create_box(width=s, height=s, depth=s)
        mesh.translate(np.asarray(center, dtype=float) - half_size)
        mesh.paint_uniform_color(np.clip(np.asarray(color, dtype=float), 0.0, 1.0))
        mesh.compute_vertex_normals()
        return mesh

    def _sphere(center, radius, color):
        """Create a solid colored sphere mesh centered at center."""
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(np.asarray(center, dtype=float))
        mesh.paint_uniform_color(np.clip(np.asarray(color, dtype=float), 0.0, 1.0))
        mesh.compute_vertex_normals()
        return mesh

    def _edge(p0, p1, color):
        """Create a single colored line segment (LineSet) between two 3D points."""
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([p0, p1])
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        ls.colors = o3d.utility.Vector3dVector([color])
        return ls

    # ------------------------------------------------------------------ graph init

    all_centroids = []
    max_pts_z = 0.0
    for seg in pred_segments:
        obj = segment_objs.get(seg)
        if obj is not None:
            pts = np.asarray(obj["pcd"].points)
            if pts.shape[0] > 0:
                all_centroids.append(pts.mean(axis=0))
                max_pts_z = max(max_pts_z, float(pts[:, 2].max()))

    if all_centroids:
        mean_centroid = np.mean(all_centroids, axis=0)
    else:
        mean_centroid = np.zeros(3)

    # Place object nodes 0.5 m above the highest point of any articulated part,
    # overriding the graph_height_obj parameter so nodes never fall inside clouds.
    graph_height_obj = max_pts_z + 0.5

    root_color = np.array([0.35, 0.35, 0.35])

    seg_node_pos = {}  # (start, end) -> object node world position
    seg_node_color = {}  # (start, end) -> darker node color
    seg_node_discovered = set()
    seg_children_discovered = set()

    def _current_root_pos():
        """Mean XY of discovered object nodes; falls back to scene mean before any are found."""
        if seg_node_pos:
            positions = np.array(list(seg_node_pos.values()))
            xy = positions[:, :2].mean(axis=0)
        else:
            xy = mean_centroid[:2]
        return np.array([xy[0], xy[1], graph_height_obj + graph_height_root_offset])

    def _build_root_geoms():
        """Root sphere + edges to every discovered object node; empty before first object."""
        if not seg_node_pos:
            return []
        rp = _current_root_pos()
        geoms = [_sphere(rp, node_radius * 1.5 * 1.2, root_color)]
        for sk, np_ in seg_node_pos.items():
            geoms.append(_edge(rp, np_, seg_node_color.get(sk, root_color)))
        return geoms

    # graph_static_geoms: object nodes, pointer lines, child nodes/edges.
    # These are appended once and never change, so they stay in a flat list.
    graph_static_geoms = []

    seg_node_discovered = set()
    seg_children_discovered = set()

    # ------------------------------------------------------------------ map state (same as plot_full_demo_video)
    map_obj_pcds = []
    map_depth_pcds = []
    map_geoms = []

    # ------------------------------------------------------------------ camera
    # Exo-centric: fixed view.  Initialise from an elevated position behind the scene;
    # after frame 0 preserve whatever the user adjusts interactively.
    mean_cam_pos = np.mean([p[:3, 3] for p in camera_poses], axis=0)
    cam_fwd = camera_poses[0][:3, 2].copy()
    cam_fwd[2] = 0.0
    cam_fwd_norm = np.linalg.norm(cam_fwd)
    if cam_fwd_norm > 1e-6:
        cam_fwd /= cam_fwd_norm
    view_eye = mean_cam_pos - 2.5 * cam_fwd + np.array([0.0, 0.0, 2.0])
    look_at = mean_centroid

    world_up = np.array([0.0, 0.0, 1.0])
    z_ax = look_at - view_eye
    z_ax /= np.linalg.norm(z_ax)
    x_ax = np.cross(z_ax, world_up)
    if np.linalg.norm(x_ax) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        x_ax = np.cross(z_ax, world_up)
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(x_ax, z_ax)  # OpenCV: Y points down → negate below
    initial_c2w = np.eye(4)
    initial_c2w[:3, 0] = x_ax
    initial_c2w[:3, 1] = -y_ax  # OpenCV convention: camera Y down
    initial_c2w[:3, 2] = z_ax
    initial_c2w[:3, 3] = view_eye

    static_camera_params = None

    def _set_camera(vis):
        """
        Apply the static camera viewpoint to the visualizer. Uses the cached
        pinhole camera parameters after the first call; otherwise builds and
        caches the initial fixed view computed above (elevated above the mean
        camera position, looking at the mean object centroid).
        """
        nonlocal static_camera_params
        if static_camera_params is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(static_camera_params, allow_arbitrary=True)
            except TypeError:
                vis.get_view_control().convert_from_pinhole_camera_parameters(static_camera_params)
        else:
            params = o3d.camera.PinholeCameraParameters()
            params.intrinsic = o3d_intrinsic
            params.extrinsic = np.linalg.inv(initial_c2w)
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            except TypeError:
                vis.get_view_control().convert_from_pinhole_camera_parameters(params)

    # ------------------------------------------------------------------ main loop

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1])
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([1.0, 1.0, 1.0])

    frame_files = []

    for t in range(num_frames):
        vis.clear_geometries()

        # Live scene point cloud
        scene_pcd, _ = azure_dataset.create_pcd(rgb_images[t], depth_images[t], camera_poses[t])

        active_seg = frame_to_seg.get(t)
        if active_seg is not None:
            start_idx, end_idx, obj = active_seg
            rel_t = t - start_idx
            thetas = seg_thetas[(start_idx, end_idx)]
            theta = float(thetas[rel_t]) if rel_t < len(thetas) else 0.0
            cur_pts = _apply_transform(np.asarray(obj["pcd"].points), obj["model"], theta)
            if cur_pts.shape[0] > 0 and not np.isnan(cur_pts).any():
                scene_pcd = remove_live_depth_near_object(scene_pcd, cur_pts, radius=0.015)

        if map_obj_pcds:
            all_obj_pts = np.concatenate([np.asarray(p.points) for p in map_obj_pcds], axis=0)
            scene_pcd = remove_live_depth_near_object(scene_pcd, all_obj_pts, radius=0.02)

        vis.add_geometry(scene_pcd)

        for pcd in map_obj_pcds:
            vis.add_geometry(pcd)
        for pcd in map_depth_pcds:
            vis.add_geometry(pcd)
        for geom in map_geoms:
            vis.add_geometry(geom)

        if active_seg is not None:
            color = seg_colors[(start_idx, end_idx)]
            seg_key = (start_idx, end_idx)

            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(cur_pts)
            obj_pc.paint_uniform_color(color)
            vis.add_geometry(obj_pc)

            for geom in _make_axis_geoms(obj["model"], color):
                vis.add_geometry(geom)

            # Add object node the first frame this interaction becomes active.
            if seg_key not in seg_node_discovered:
                seg_node_discovered.add(seg_key)
                obj_centroid = np.asarray(obj["pcd"].points).mean(axis=0)
                node_pos = np.array([obj_centroid[0], obj_centroid[1], graph_height_obj])
                node_color = color * 0.55
                seg_node_pos[seg_key] = node_pos
                seg_node_color[seg_key] = node_color
                graph_static_geoms.append(_cube(node_pos, node_radius, node_color))
                # Pointer from graph node to physical object location
                graph_static_geoms.append(_edge(node_pos, obj_centroid, np.array([0.6, 0.6, 0.6])))

            # Interaction just finished: freeze into map and add child nodes.
            if t == end_idx - 1:
                frozen_pcd = o3d.geometry.PointCloud()
                frozen_pcd.points = o3d.utility.Vector3dVector(cur_pts)
                frozen_pcd.paint_uniform_color(color)
                map_obj_pcds.append(frozen_pcd)
                map_geoms.extend(_make_axis_geoms(obj["model"], color))

                # Recreate terminal depth scan with human pixels masked out.
                if human_masks is not None:
                    terminal_pcd, _ = azure_dataset.create_pcd(
                        rgb_images[t],
                        depth_images[t],
                        camera_poses[t],
                        maskout=human_masks[t],
                    )
                    if cur_pts.shape[0] > 0 and not np.isnan(cur_pts).any():
                        terminal_pcd = remove_live_depth_near_object(terminal_pcd, cur_pts, radius=0.015)
                    if map_obj_pcds:
                        all_obj_pts = np.concatenate([np.asarray(p.points) for p in map_obj_pcds], axis=0)
                        terminal_pcd = remove_live_depth_near_object(terminal_pcd, all_obj_pts, radius=0.02)
                else:
                    terminal_pcd = copy.deepcopy(scene_pcd)
                map_depth_pcds.append(terminal_pcd)

                if hierarchy is not None and seg_key not in seg_children_discovered:
                    seg_children_discovered.add(seg_key)
                    parent_node_pos = seg_node_pos.get(seg_key)
                    child_idcs = obj.get("children_idcs", [])
                    if parent_node_pos is not None:
                        for child_idx in child_idcs:
                            child = hierarchy.children[child_idx]
                            child_color = np.asarray(child.get("inst_color", rng.random(3)), dtype=float)
                            child_pts = np.asarray(child["pcd"].points)
                            if child_pts.shape[0] == 0:
                                continue
                            child_centroid = child_pts.mean(axis=0)
                            child_node_pos = np.array(
                                [
                                    child_centroid[0],
                                    child_centroid[1],
                                    parent_node_pos[2] - graph_height_child_offset,
                                ]
                            )
                            graph_static_geoms.append(_sphere(child_node_pos, node_radius * 0.8, child_color))
                            graph_static_geoms.append(_edge(parent_node_pos, child_node_pos, child_color))

        # Graph overlay: root recomputed each frame, static parts accumulated over time.
        for geom in _build_root_geoms() + graph_static_geoms:
            vis.add_geometry(geom)

        _set_camera(vis)
        vis.poll_events()
        vis.update_renderer()
        # Preserve interactive camera adjustments
        static_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        frame_path = os.path.join(output_dir, f"frame_{t:06d}.png")
        vis.capture_screen_image(frame_path)
        frame_files.append(frame_path)

    # ------------------------------------------------------------------ orbit
    # After all interaction frames, revolve the camera around the scene so the
    # final graph state can be inspected from every angle.
    orbit_total = orbit_revolutions * orbit_frames_per_revolution
    orbit_xy = (view_eye - np.array([mean_centroid[0], mean_centroid[1], view_eye[2]]))[:2]
    orbit_radius = float(np.linalg.norm(orbit_xy))
    if orbit_radius < 0.1:
        orbit_radius = 3.0
    orbit_start_angle = float(np.arctan2(orbit_xy[1], orbit_xy[0]))
    orbit_height = float(view_eye[2])
    orbit_center = mean_centroid.copy()

    for i in range(orbit_total):
        angle = orbit_start_angle + 2.0 * np.pi * i / orbit_frames_per_revolution
        orbit_eye = np.array(
            [
                orbit_center[0] + orbit_radius * np.cos(angle),
                orbit_center[1] + orbit_radius * np.sin(angle),
                orbit_height,
            ]
        )
        z_ax = orbit_center - orbit_eye
        z_ax /= np.linalg.norm(z_ax)
        world_up_o = np.array([0.0, 0.0, 1.0])
        x_ax = np.cross(z_ax, world_up_o)
        if np.linalg.norm(x_ax) < 1e-6:
            world_up_o = np.array([0.0, 1.0, 0.0])
            x_ax = np.cross(z_ax, world_up_o)
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(x_ax, z_ax)
        c2w = np.eye(4)
        c2w[:3, 0] = x_ax
        c2w[:3, 1] = -y_ax  # OpenCV: Y down
        c2w[:3, 2] = z_ax
        c2w[:3, 3] = orbit_eye

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d_intrinsic
        params.extrinsic = np.linalg.inv(c2w)
        try:
            vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        vis.poll_events()
        vis.update_renderer()
        frame_path = os.path.join(output_dir, f"frame_{num_frames + i:06d}.png")
        vis.capture_screen_image(frame_path)
        frame_files.append(frame_path)

    vis.destroy_window()

    if save_video and frame_files:
        video_path = os.path.join(output_dir, "momasg_video.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(output_dir, "frame_%06d.png"),
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"MoMa-SG video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")

    return frame_files


def visualize_camera_movement(
    scene_pcd,
    camera_pose_list,
    keypoints=None,
    frame_size=0.2,
    trajectory_color=[0.0, 1.0, 0.0],
    keypoint_color=[1.0, 0.0, 0.0],
    window_width=1280,
    window_height=720,
):
    """
    Visualize a scene point cloud with camera poses, trajectory, and optional keypoints.

    Parameters:
      scene_pcd (o3d.geometry.PointCloud): The scene point cloud.
      camera_pose_list (list of np.ndarray): List of 4x4 camera poses.
      keypoints (np.ndarray, optional): Array of shape (N, 3) or list of arrays with keypoint locations.
                                       If None, no keypoints are visualized.
      animation_speed (float): Controls the speed of the camera animation (lower is slower).
      keypoint_radius (float): Radius of the spheres representing keypoints.
      frame_size (float): Size of the coordinate frames representing camera poses.
      trajectory_color (list): RGB color for the camera trajectory line.
      keypoint_color (list): RGB color for the keypoint spheres.
      save_images (bool): If True, save images of each frame during animation.
      output_dir (str, optional): Directory to save images. Required if save_images is True.
      window_width (int): Width of the visualization window.
      window_height (int): Height of the visualization window.

    Returns:
      None
    """

    # Validate inputs
    if len(camera_pose_list) == 0:
        raise ValueError("camera_pose_list must contain at least one pose")

    # Create a visualizer window with specified dimensions
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Camera Trajectory Visualization",
        width=window_width,
        height=window_height,
    )

    # Add the scene point cloud
    vis.add_geometry(scene_pcd)

    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0

    # create point cloud for keypoints
    if keypoints is not None:
        for i, points in enumerate(keypoints):
            points = camera_pose_list[i][:3, :3] @ points.T + camera_pose_list[i][:3, 3][:, None]
            keypoint_pcd = o3d.geometry.PointCloud()
            keypoint_pcd.points = o3d.utility.Vector3dVector(points.T)
            keypoint_pcd.paint_uniform_color(keypoint_color)
            vis.add_geometry(keypoint_pcd)

    # List to store camera positions for trajectory
    camera_positions = []

    # For each camera pose, create a coordinate frame and add it
    for i, pose in enumerate(camera_pose_list):
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose)
        vis.add_geometry(frame)

        # Extract camera position from the pose
        camera_positions.append(pose[:3, 3])

    # Create camera trajectory if we have more than one pose
    if len(camera_positions) > 1:
        lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
        colors = [trajectory_color for _ in lines]

        trajectory = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(camera_positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        trajectory.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(trajectory)

    # Get the view control
    view_ctl = vis.get_view_control()

    # Set initial camera view
    view_ctl.set_zoom(0.8)

    # First render the scene with all geometries
    vis.poll_events()
    vis.update_renderer()

    # Keep the window open until closed by user
    print("Visualization complete. Close the window to exit.")
    vis.run()
    vis.destroy_window()


def visualize_articulations(
    scene_path,
    gt_data=None,
    predictions=None,
    objects=None,
    gt_idcs=None,
    pred_idcs=None,
    axis_length=1.0,
    gt_color=(0, 1, 0),
    pred_color=(1, 0, 0),
    background=(1.0, 1.0, 1.0),
):
    """
    Visualize scene mesh/pointcloud and articulation axes (both ground truth and predicted).

    Args:
        scene_path: Path to scene mesh (.obj, .ply) or pointcloud (.pcd, .ply)
        gt_data: Ground truth articulation data from load_gt_data()
        predictions: Predicted articulation data from load_prediction_data()
        gt_indices: Indices of ground truth that match the predicted data
        pred_indices: Indices of predicted data that match the ground truth
        axis_length: Length of drawn axis lines
        gt_color: RGB color tuple for ground truth axes (default: green)
        pred_color: RGB color tuple for predicted axes (default: red)
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Articulation Visualization", width=1280, height=720)

    # Load scene geometry
    scene_geometry = None
    file_ext = Path(scene_path).suffix.lower()

    if file_ext in [".obj", ".ply", ".stl"]:
        try:
            scene_geometry = o3d.io.read_triangle_mesh(scene_path)
            scene_geometry.compute_vertex_normals()
            # Add to visualizer with default color
            vis.add_geometry(scene_geometry)
        except Exception as e:
            print(f"Failed to load mesh {e}")
            scene_geometry = None
            return

    # Helper function to create axis line
    def create_axis_line(position, axis_dir, length, color):
        """Create a line segment representing an axis"""
        position = np.array(position)
        axis_dir = np.array(axis_dir)
        axis_dir = axis_dir / np.linalg.norm(axis_dir)  # normalize

        start_point = position - (axis_dir * length / 2)
        end_point = position + (axis_dir * length / 2)

        points = [start_point, end_point]
        lines = [[0, 1]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])

        return line_set

    # Add GT axes
    if gt_data:
        for obj_name, obj_data in gt_data.items():
            if "position" in obj_data and "axis" in obj_data:
                axis_line = create_axis_line(obj_data["position"], obj_data["axis"], axis_length, gt_color)
                vis.add_geometry(axis_line)

                # Add a small sphere at the position
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(obj_data["position"])
                sphere.paint_uniform_color(gt_color)
                vis.add_geometry(sphere)

    # Add prediction axes
    if predictions:
        for model in predictions:
            if model.axis is not None and model.position is not None:
                axis_line = create_axis_line(model.position, model.axis, axis_length, pred_color)
                vis.add_geometry(axis_line)

                # Add a small sphere at the position
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(model.position)
                print(f"Pred position: {model.position}")
                sphere.paint_uniform_color(pred_color)
                vis.add_geometry(sphere)

    # Add object primitives
    if objects:
        for obj in objects:
            if "points" in obj:
                primitive = o3d.geometry.PointCloud()
                primitive.points = o3d.utility.Vector3dVector(obj["points"])
                primitive.paint_uniform_color((0.5, 0.5, 0.0))  # gray color
                vis.add_geometry(primitive)
                del primitive

    # draw line between matched gt and pred indicies
    if gt_idcs is not None and pred_idcs is not None:
        for gt_idx, pred_idx in zip(gt_idcs, pred_idcs):
            if gt_data and predictions:
                gt_obj = gt_data[list(gt_data.keys())[gt_idx]]

                # get gt_data at index gt_idx
                pred_obj = predictions[pred_idx]

                if "position" in gt_obj and "position" in pred_obj:
                    start_point = np.array(gt_obj["position"])
                    end_point = np.array(pred_obj["position"])

                    points = [start_point, end_point]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([(0, 0, 1)])
                    vis.add_geometry(line_set)

                    # plot coordinate frame of the prediction
                    if "w_T_a" in pred_obj.keys():
                        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                        frame.transform(pred_obj["w_T_a"])
                        vis.add_geometry(frame)

                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(pred_obj["motion_paths"])
                        line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pred_obj["motion_paths"]) - 1)])
                        line_set.colors = o3d.utility.Vector3dVector([(1, 1, 0)])
                        vis.add_geometry(line_set)

                        prev_point = None
                        for idx, point in enumerate(pred_obj["motion_paths"]):
                            if prev_point is not None:
                                line = o3d.geometry.LineSet()
                                line.points = o3d.utility.Vector3dVector([prev_point, point])
                                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                                line.colors = o3d.utility.Vector3dVector([[1, 1, 0]])
                                vis.add_geometry(line)
                            prev_point = point

                        start_point = pred_obj["w_T_a"][:3, 3]
                        end_point = np.array(pred_obj["position"])

                    points = [start_point, end_point]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([(1, 1, 0)])

                    vis.add_geometry(line_set)

    # Set initial camera view
    opt = vis.get_render_option()
    if background is not None:
        opt.background_color = np.array(background)
    else:
        opt.background_color = np.array([0.1, 0.1, 0.1]) if background is None else np.array(background)
    opt.point_size = 3.0

    # Run visualization
    vis.run()
    vis.destroy_window()


def create_cylinder(length, radius=0.01, color=[0.8, 0.3, 0.0]):
    """Helper to create a colored cylinder."""
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl


def transform(mesh, translation=None, rotation=None):
    """Apply translation and rotation to a mesh."""
    if rotation is not None:
        R = mesh.get_rotation_matrix_from_xyz(rotation)  # (rx, ry, rz)
        mesh.rotate(R, center=(0, 0, 0))
    if translation is not None:
        mesh.translate(translation)
    return mesh


def create_grasp_viz(width: float = 0.05, finger_length: float = 0.1, transl: List[int] = [0, 0, 0], rot: List[int] = [0, 0, 0]):
    """
    Build a simple two-finger gripper visualization mesh (stem + thumb/index
    branches + connecting palm) and place it at a given pose.

    Args:
        width: Distance between the thumb and index finger cylinders.
        finger_length: Length of the stem and finger cylinders.
        transl: (3,) global translation applied to the assembled gripper mesh.
        rot: (3,) global rotation (Euler angles, radians) applied to the mesh.

    Returns:
        open3d.geometry.TriangleMesh: The combined gripper mesh.
    """
    tcp_offset = finger_length
    stem = create_cylinder(length=finger_length, radius=0.008, color=[0.3, 0.3, 0.9])
    transform(stem, translation=[0, 0, -finger_length / 2])

    # Thumb branch
    thumb = create_cylinder(length=finger_length, radius=0.006, color=[0.9, 0.3, 0.3])
    transform(thumb, rotation=[0, 0, 0], translation=[width / 2, 0, finger_length / 2])

    # Index branch
    index = create_cylinder(length=finger_length, radius=0.006, color=[0.9, 0.3, 0.3])
    transform(index, rotation=[0, 0, 0], translation=[-width / 2, 0, finger_length / 2])

    # -------------------------------
    # Connection between branches (palm)
    # -------------------------------
    # Midpoint and approximate orientation
    palm = create_cylinder(length=width, radius=0.006, color=[0.9, 0.3, 0.3])
    # Rotate along Y-axis to connect horizontally
    transform(palm, rotation=[0, 1.57, 0], translation=[0, 0, 0])

    # Combine meshes into one
    combined_mesh = stem + thumb + index + palm
    transform(combined_mesh, rotation=[0, 0, 0], translation=[0, 0, -tcp_offset])

    # apply global transform
    transform(combined_mesh, rotation=rot, translation=transl)
    return combined_mesh
