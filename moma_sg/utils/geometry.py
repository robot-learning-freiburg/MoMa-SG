import cv2
import numpy as np
import open3d as o3d
import torch


def get_rotation_magnitude(T: np.ndarray) -> float:
    """
    Compute the rotation angle encoded in the rotation part of a homogeneous transform.

    Parameters:
        T (np.ndarray): (4,4) homogeneous transformation matrix.

    Returns:
        float: Rotation magnitude in radians, derived from the trace of the rotation matrix.
    """
    R = T[:3, :3]
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # Clip to avoid numerical issues
    return theta  # in radians


def estimate_motion_blur(image):
    """
    Estimate the amount of motion blur in an image using Laplacian variance.

    Parameters:
        image (np.ndarray): Input image (grayscale or color)

    Returns:
        float: Blur amount (lower = more blur, higher = less blur)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def dist_point_axis(points, axis_point, axis_direction):
    """
    Compute shortest distances from a 3D point cloud to an axis (infinite line).

    Parameters
    ----------
    points : (N,3) ndarray
        Point cloud coordinates.
    axis_point : (3,) array_like
        A point on the axis.
    axis_direction : (3,) array_like
        Direction vector of the axis (does not need to be normalized).

    Returns
    -------
    distances : (N,) ndarray
        Distance from each point in the cloud to the axis.
    """
    points = np.asarray(points)
    axis_point = np.asarray(axis_point)
    axis_direction = np.asarray(axis_direction)
    axis_direction = axis_direction / np.linalg.norm(axis_direction)  # produce unit axis vector
    distances = np.linalg.norm(np.cross(points - axis_point, axis_direction), axis=1) / np.linalg.norm(axis_direction)
    return distances


def get_dominant_normal(points: np.ndarray) -> np.ndarray:
    """
    Estimate the dominant normal direction of a point set via PCA (SVD of centered points).

    Parameters
    ----------
    points : (N,3) ndarray
        3D point coordinates.

    Returns
    -------
    normal : (3,) ndarray
        Unit normal vector corresponding to the smallest-variance direction.
    """
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]  # last row of vh (smallest singular value) is the normal
    return normal / np.linalg.norm(normal)


def warp_depth_torch(D_B: torch.Tensor, K: torch.Tensor, T_A_B: torch.Tensor) -> torch.Tensor:
    """
    Warp depth map from frame B into frame A using camera intrinsics and pose.

    Args:
        D_B: (H, W) depth map from frame B, float32, >0
        K: (3, 3) camera intrinsic matrix, float32
        T_A_B: (4, 4) transformation from B to A, float32
    Returns:
        D_A: (H, W) depth map expressed in frame A
    """
    device = D_B.device
    H, W = D_B.shape

    # ---------------------------------------------------------
    # 1. Create a pixel grid
    # ---------------------------------------------------------
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    # ---------------------------------------------------------
    # 2. Back-project depth pixels in frame B
    # ---------------------------------------------------------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = D_B.reshape(-1)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    ones = torch.ones_like(z)
    pts_B = torch.stack([x, y, z, ones], dim=0)  # (4, N)

    # ---------------------------------------------------------
    # 3. Transform into frame A
    # ---------------------------------------------------------
    pts_A = T_A_B @ pts_B  # (4, N)
    X, Y, Z = pts_A[0], pts_A[1], pts_A[2]

    # valid depth > 0
    valid = Z > 0

    # ---------------------------------------------------------
    # 4. Project into A camera
    # ---------------------------------------------------------
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    # Round to nearest integer pixel
    u = torch.round(u).long()
    v = torch.round(v).long()

    # ---------------------------------------------------------
    # 5. Initialize output depth map with +inf for z-buffer
    # ---------------------------------------------------------
    D_A = torch.full((H, W), float("inf"), device=device)

    # Filter valid projections inside image bounds
    mask = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_valid = u[mask]
    v_valid = v[mask]
    z_valid = Z[mask]

    # ---------------------------------------------------------
    # 6. Z-buffer: choose the smallest depth for each pixel
    # ---------------------------------------------------------
    # Flatten index
    idx = v_valid * W + u_valid
    D_A_flat = D_A.reshape(-1)

    # For duplicate indices, keep nearest depth
    D_A_flat.index_reduce_(0, idx, z_valid, reduce="amin")

    # Replace inf (unfilled) with zero
    D_A[D_A == float("inf")] = 0.0

    return D_A


def warp_depth_batched(D_B: torch.Tensor, K: torch.Tensor, T_A_B: torch.Tensor) -> torch.Tensor:
    """
    NOT USED CURRENTLY
    Batched warp depth maps from frame B into frame A using camera intrinsics and pose.

    Args:
        D_B: (B, H, W) depth maps from frame B, float32, >0
        K: (3, 3) camera intrinsic matrix
        T_A_B: (B, 4, 4) transforms from B to A (one per batch)

    Returns:
        D_B_warped: (B, H, W) depth maps warped to frame A
    """
    B, H, W = D_B.shape
    device = D_B.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1. Create pixel grid
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = u.flatten()
    v = v.flatten()
    N = H * W

    # Repeat pixel coordinates for batch
    u_b = u.unsqueeze(0).repeat(B, 1)  # (B, N)
    v_b = v.unsqueeze(0).repeat(B, 1)
    z_b = D_B.reshape(B, N)  # (B, N)

    # 2. Valid depth mask
    valid = z_b > 0
    u_b = u_b[valid]
    v_b = v_b[valid]
    z_b = z_b[valid]

    # Need batch indices for scatter
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, N).reshape(-1)[valid]

    # 3. Back-project to 3D homogeneous coordinates
    x = (u_b - cx) * z_b / fx
    y = (v_b - cy) * z_b / fy
    ones = torch.ones_like(z_b)
    pts_B = torch.stack([x, y, z_b, ones], dim=0)  # (4, total_points)

    # 4. Gather corresponding transforms
    T = T_A_B[batch_idx]  # (total_points, 4, 4)
    pts_B = pts_B.T.unsqueeze(-1)  # (total_points, 4, 1)
    pts_A = torch.bmm(T, pts_B).squeeze(-1)  # (total_points, 4)
    x_a, y_a, z_a = pts_A[:, 0], pts_A[:, 1], pts_A[:, 2]

    # 5. Project back to image plane
    u_a = (fx * x_a / z_a + cx).long()
    v_a = (fy * y_a / z_a + cy).long()

    # 6. Keep only pixels inside frame
    inside = (u_a >= 0) & (u_a < W) & (v_a >= 0) & (v_a < H)
    u_a, v_a, z_a, batch_idx = u_a[inside], v_a[inside], z_a[inside], batch_idx[inside]

    # 7. Initialize warped depth map
    D_B_warped = torch.zeros((B, H, W), device=device, dtype=D_B.dtype)

    # 8. Compute flattened indices for scatter
    idx = v_a * W + u_a
    D_flat = D_B_warped.reshape(B, -1)

    # Scatter z-buffer using amin to keep closest depth
    D_flat.scatter_reduce_(0, idx + batch_idx * H * W, z_a, reduce='amin', include_self=True)

    return D_B_warped


def warp_depth(D_B: np.ndarray, K: np.ndarray, T_A_B: np.ndarray) -> np.ndarray:
    """
    Warp depth map from frame B into frame A using camera intrinsics and pose.

    Args:
        D_B: (H, W) depth map from frame B
        K: (3, 3) camera intrinsic matrix
        T_A_B: (4, 4) transformation from B to A (i.e., T_A_B = T_A * inv(T_B))

    Returns:
        D_B_warped: (H, W) depth map from B warped to A's frame
    """
    H, W = D_B.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = D_B[v, u]  # Depth values

    # Valid depth mask
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    # Back-project to 3D in frame B
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_B = np.vstack((x, y, z, np.ones_like(z)))  # Shape: (4, N)

    # Transform to frame A
    pts_A = T_A_B @ pts_B  # Shape: (4, N)
    x_a, y_a, z_a = pts_A[0], pts_A[1], pts_A[2]

    # Project back to image plane of frame A
    u_a = (fx * x_a / z_a + cx).astype(np.int32)
    v_a = (fy * y_a / z_a + cy).astype(np.int32)

    # Create warped depth map
    D_B_warped = np.zeros_like(D_B)
    inside = (u_a >= 0) & (u_a < W) & (v_a >= 0) & (v_a < H)
    u_a, v_a, z_a = u_a[inside], v_a[inside], z_a[inside]

    # Fill depth values with z-buffering (closest wins)
    for ua, va, za in zip(u_a, v_a, z_a):
        if D_B_warped[va, ua] == 0 or za < D_B_warped[va, ua]:
            D_B_warped[va, ua] = za

    return D_B_warped


def warp_depth_o3d(depth_array: np.ndarray, intrinsics: np.ndarray, transform_a_b: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    Deprecated because slow
    Warp depth map from frame B into frame A using camera intrinsics and pose on GPU.
    Args:
        depth_array: (H, W) depth map from frame B
        intrinsics: (3, 3) camera intrinsic matrix
        transform_a_b: (4, 4) transformation from B to A (i.e., T_A_B = T_A * inv(T_B))
        depth_scale: Scale factor to convert depth values to meters
    Returns:
        warped_depth: (H, W) metric depth map from B warped to A's frame in original scale (not meters) as numpy array
    """
    input_cloud = o3d.t.geometry.PointCloud.create_from_depth_image(
        o3d.t.geometry.Image(depth_array * depth_scale), o3d.core.Tensor(intrinsics), depth_scale=depth_scale
    ).cuda()
    output_cloud = input_cloud.transform(o3d.core.Tensor(transform_a_b))
    warped_input = (
        output_cloud.project_to_depth_image(
            width=depth_array.shape[1],
            height=depth_array.shape[0],
            intrinsics=o3d.core.Tensor(intrinsics).cuda(),
        )
        .as_tensor()
        .cpu()
        .numpy()
    )
    return warped_input.squeeze(-1) / depth_scale


def extract_predominant_axis(pcd: o3d.geometry.PointCloud):
    """
    Fit an oriented bounding box to a point cloud and extract its longest axis.

    Input:
        pcd : open3d.geometry.PointCloud

    Returns:
        axis_dir : (3,) numpy array   → unit vector of the dominant axis
        p_min    : (3,) numpy array   → one endpoint of the center axis
        p_max    : (3,) numpy array   → other endpoint of the center axis
    """
    obb = pcd.get_oriented_bounding_box()
    center = obb.center  # (3,)
    R = obb.R  # (3,3) rotation matrix, columns are axes
    extends = obb.extent / 2.0  # Open3D stores full lengths → convert to half-lengths

    # Index of longest dimension
    idx = np.argmax(extends)

    # Longest axis is the corresponding column of R
    axis_dir = R[:, idx]  # unit vector

    # Half-length
    half_len = extends[idx]

    # Endpoints of the central axis
    p_min = center - half_len * axis_dir
    p_max = center + half_len * axis_dir

    return axis_dir, p_min, p_max


def cloud_to_mesh(pcd):
    """
    Reconstruct a triangle mesh from a point cloud via Poisson surface reconstruction.

    Estimates and consistently orients normals, runs Poisson reconstruction, then
    removes low-density vertices that correspond to the spurious outer hull.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to reconstruct (normals are estimated/overwritten in place).

    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        Reconstructed mesh with low-density vertices removed.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(10)

    # --- 2. Poisson reconstruction (returns mesh and densities) ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # --- 3. Remove low-density “outer hull” triangles (recommended) ---
    densities = np.asarray(densities)
    vertices_to_keep = densities > np.quantile(densities, 0.02)
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    return mesh


def rotation_from_axis_to_axis(ref_axis, cand_axis, eps=1e-8):
    """
    Compute SO(3) rotation matrix R such that:
        R @ ref_axis = cand_axis

    Parameters
    ----------
    ref_axis : (3,) array_like
        Reference axis
    cand_axis : (3,) array_like
        Candidate axis

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix in SO(3)
    """
    from moma_sg.articulation.point_estimator import _skew

    r = np.asarray(ref_axis, dtype=float) / np.linalg.norm(ref_axis)
    c = np.asarray(cand_axis, dtype=float) / np.linalg.norm(cand_axis)

    v = np.cross(r, c)
    s = np.linalg.norm(v)
    d = np.dot(r, c)

    # Case 1: vectors are the same
    if s < eps and d > 0:
        return np.eye(3)

    # Case 2: vectors are opposite
    if s < eps and d < 0:
        # Find any orthogonal axis
        if abs(r[0]) < abs(r[1]):
            ortho = np.array([1.0, 0.0, 0.0])
        else:
            ortho = np.array([0.0, 1.0, 0.0])

        v = np.cross(r, ortho)
        v = v / np.linalg.norm(v)
        K = _skew(v)
        return np.eye(3) + 2.0 * (K @ K)  # Rodrigues with theta = pi

    # General case
    K = _skew(v)
    R = np.eye(3) + K + (K @ K) * ((1.0 - d) / (s**2))
    return R


def to_homog(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian points to homogeneous coordinates by appending a column of ones.

    Parameters
    ----------
    points : (N,3) or (3,) ndarray
        Cartesian point(s).

    Returns
    -------
    (N,4) ndarray
        Homogeneous coordinates.
    """
    if len(points.shape) == 1:
        points = points[None, :]
    return np.hstack((points, np.ones((points.shape[0], 1))))


def from_homog(points: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous coordinates back to Cartesian by dropping the last column.

    Parameters
    ----------
    points : (N,4) ndarray
        Homogeneous point coordinates.

    Returns
    -------
    (3,) ndarray if N == 1, otherwise (N,3) ndarray
        Cartesian point coordinates.
    """
    if points.shape[0] == 1:
        return points[:, :3].ravel()
    else:
        return points[:, :3]
