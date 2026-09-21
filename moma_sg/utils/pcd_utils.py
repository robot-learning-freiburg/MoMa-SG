from collections import Counter
from functools import wraps

import numpy as np
import open3d as o3d


def apply_to_list(func):
    """
    A decorator that applies the wrapped function to each element if the first argument is a list.
    Otherwise, it just calls the function with the provided arguments.
    """

    @wraps(func)
    def wrapper(pcd_or_list, *args, **kwargs):
        """Apply func to each element if pcd_or_list is a list, else call func directly."""
        # If the input is a list, process each element individually.
        if isinstance(pcd_or_list, list):
            return [func(pcd, *args, **kwargs) for pcd in pcd_or_list]
        # Otherwise, process the single input.
        return func(pcd_or_list, *args, **kwargs)

    return wrapper


@apply_to_list
def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    """
    Denoise the point cloud using DBSCAN.
    :param pcd: Point cloud to denoise.
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Denoised point cloud.
    """
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd
