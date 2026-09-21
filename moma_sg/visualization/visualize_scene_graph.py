import glob
import json
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path
import pickle

import hydra
import loguru
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d


def create_partial_torus_arrow(
    center, axis, major_radius=0.12, minor_radius=0.015, arc_degrees=300, n_major=40, n_minor=12, cone_radius=0.04, cone_length=0.06, color=(1, 0, 0)
):
    """300-deg arc tube with an arrowhead at the open end, ring plane perpendicular to `axis`."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    arc_rad = np.radians(arc_degrees)

    # Orthonormal basis in the torus plane
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

    # Arrowhead tangent to the arc at the open end
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
        R = cone.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cone.rotate(R, center=(0, 0, 0))
    cone.translate(tip_pos)
    cone.paint_uniform_color(color)

    return tube, cone


def cylinder_cone_arrow(start, end, cyl_radius=0.03, cone_radius=0.06, cone_length=0.2, color=(1, 0, 0)):
    """Build a straight arrow mesh (cylindrical shaft + conical tip) from `start` to `end`.

    Args:
        start: 3D coordinate of the arrow's base.
        end: 3D coordinate of the arrow's tip direction target.
        cyl_radius: Radius of the shaft cylinder.
        cone_radius: Radius of the tip cone.
        cone_length: Length of the tip cone; must be smaller than the total arrow length.
        color: RGB color applied to both the shaft and the tip.

    Returns:
        Tuple of (cylinder mesh, cone mesh) forming the arrow.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    total_length = np.linalg.norm(direction)
    direction /= total_length

    # Lengths
    shaft_length = total_length - cone_length
    if shaft_length <= 0:
        raise ValueError("cone_length must be smaller than arrow length")

    # ------------------
    # Cylinder (shaft)
    # ------------------
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=shaft_length, resolution=30)
    cyl.compute_vertex_normals()

    # Cylinder is centered at origin → move base to z=0
    cyl.translate((0, 0, shaft_length / 2))

    # ------------------
    # Cone (tip)
    # ------------------
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_length, resolution=30)
    cone.compute_vertex_normals()

    # Cone base at z=0, tip at +cone_length
    # Open3D cone is centered → move base to z=0
    cone.translate((0, 0, cone_length / 2))

    # ------------------
    # Rotate both from +Z to direction
    # ------------------
    z = np.array([0, 0, 1])
    axis = np.cross(z, direction)
    angle = np.arccos(np.clip(np.dot(z, direction), -1, 1))

    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        R = cyl.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=(0, 0, 0))
        cone.rotate(R, center=(0, 0, 0))

    # ------------------
    # Translate into place
    # ------------------
    cyl.translate(start)
    cone.translate(start + 0.88 * direction * shaft_length)

    # Colors (optional)
    cyl.paint_uniform_color(color)
    cone.paint_uniform_color(color)

    return cyl, cone


def color_mesh_near_multiple_pointclouds(mesh: o3d.geometry.TriangleMesh, pointclouds, radius: float, colors, default_color=(0.8, 0.8, 0.8)):
    """
    Colors mesh vertices that are within `radius` of any point cloud.

    Parameters
    ----------
    mesh : TriangleMesh
        Target mesh
    pointclouds : list[PointCloud]
        List of point clouds
    radius : float
        Distance threshold
    colors : list[tuple]
        RGB color per point cloud
    default_color : tuple
        Base mesh color
    """

    assert len(pointclouds) == len(colors), "pointclouds and colors must have the same length"

    vertices = np.asarray(mesh.vertices)
    n_vertices = len(vertices)

    # Initialize colors
    mesh_colors = np.tile(default_color, (n_vertices, 1))

    # KD-tree on mesh vertices
    mesh_tree = o3d.geometry.KDTreeFlann(mesh)

    for pcd, color in zip(pointclouds, colors):
        for p in np.asarray(pcd.points):
            [_, idx, _] = mesh_tree.search_radius_vector_3d(p, radius)
            mesh_colors[idx] = color

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)


def load_articulations(save_dir):
    """Load all cached articulation models from `save_dir`/articulation.

    Args:
        save_dir: Directory containing an "articulation" subfolder with "*_model.json" files.

    Returns:
        Dict mapping model id to the loaded InferencePointAxis model.
    """
    from moma_sg.articulation.point_estimator import InferencePointAxis

    # load articulation models from dir
    seg_articulations = dict()
    articulation_dir = os.path.join(save_dir, "articulation")
    model_files = [f for f in os.listdir(articulation_dir) if f.endswith("_model.json")]
    for model_file in model_files:
        model = InferencePointAxis.load(os.path.join(articulation_dir, model_file))
        seg_articulations[model.id] = model
    return seg_articulations


def remove_triangles_near_point_cloud(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, distance_threshold: float
) -> o3d.geometry.TriangleMesh:
    """
    Removes triangles from a mesh whose centroids are within
    distance_threshold of any point in the point cloud.
    """

    # Convert to numpy
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    pcd_points = np.asarray(pcd.points)

    # Build KD-tree for the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    triangles_to_remove = []

    for i, tri in enumerate(triangles):
        # Compute triangle centroid
        centroid = vertices[tri].mean(axis=0)

        # Search nearest point in point cloud
        [k, idx, dist2] = pcd_tree.search_knn_vector_3d(centroid, 1)

        if k > 0 and dist2[0] <= distance_threshold**2:
            triangles_to_remove.append(i)

    # Remove triangles
    mesh.remove_triangles_by_index(triangles_to_remove)

    # Cleanup
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    return mesh


def run(cfg: DictConfig) -> None:
    """Load a scene's mesh, ground-truth and predicted articulation data, and open an
    interactive Open3D viewer showing the segmented objects with their articulation axes.

    Args:
        cfg: Hydra configuration with dataset, cache, and load-directory settings.
    """
    loguru.logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    scene = Path(cfg.dataset.root_path).stem
    room = Path(cfg.dataset.root_path).parent.name
    loguru.logger.info(f"Processing scene: {scene} of type {room}")

    # Create output directory for intermediate results if needed
    if cfg.cache.save_results:
        save_dir = os.path.join(
            cfg.cache.output_dir,
            room,
            scene,
        )
        os.makedirs(save_dir, exist_ok=True)
        loguru.logger.info(f"Saving intermediate results to {save_dir}")
        # write metadata file of all configs
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            metadata = {
                "scene_name": scene,
                "scene_type": room,
                "config": OmegaConf.to_container(cfg),
            }
            json.dump(metadata, f, indent=4)

    # Set the directory for loading intermediate results
    if cfg.cache.load_results:
        load_dir = os.path.join(
            cfg.cache.load_dir,
            room,
            scene,
        )

    # Load the scene mesh
    file_path = os.path.join(
        cfg.dataset.root_path,
        "compressed_mesh.ply",
    )
    mesh = o3d.io.read_triangle_mesh(file_path).compute_vertex_normals()

    pcd_path = os.path.join(
        cfg.dataset.root_path,
        "compressed_point_cloud.ply",
    )
    pcd = o3d.io.read_point_cloud(pcd_path)

    hierarchy_path = Path(save_dir) / "hierarchy.pkl"
    with open(hierarchy_path, 'rb') as f:
        hierarchy = pickle.load(f)

    #### USING GROUND TRUTH
    from moma_sg.data.arti4d import load_arti4d_ground_truth

    # TODO: modify
    GT_DATA_ROOT = Path('/path/to/arti4d/raw')
    GT_ARTICULATION, _, _ = load_arti4d_ground_truth(GT_DATA_ROOT)

    # # plot_axes = []
    # # for _, segment_data in GT_ARTICULATION[room][scene].items():
    # #     if segment_data.type.lower() == 'prismatic':
    # #         axis_start = np.array(segment_data.position) - 0.5 * np.array(segment_data.axis)
    # #         axis_end = np.array(segment_data.position) + 0.5 * np.array(segment_data.axis)
    # #         cyl, cone = cylinder_cone_arrow(axis_start, axis_end, cyl_radius=0.01, cone_radius=0.02, cone_length=0.05)
    # #         plot_axes.append(cyl)
    # #         plot_axes.append(cone)
    # #     elif segment_data.type.lower() == 'revolute':
    # #         axis_start = np.array(segment_data.position) - 0.5 * np.array(segment_data.axis)
    # #         axis_end = np.array(segment_data.position) + 0.5 * np.array(segment_data.axis)
    # #         cyl, cone = cylinder_cone_arrow(axis_start, axis_end, cyl_radius=0.01, cone_radius=0.02, cone_length=0.05)
    # #         plot_axes.append(cyl)
    # #         plot_axes.append(cone)

    # ### LOAD GT PARENTS AND CHILDREN
    # print(f'Processing room {room}, scene {scene}')
    # gt_children_dir = GT_DATA_ROOT / room / scene / "children"
    gt_parents_dir = GT_DATA_ROOT / room / scene / "objects"

    plot_objects = []
    segment_objs = {}
    parent_names = [Path(p).stem.split(".")[0] for p in glob.glob(str(gt_parents_dir / "*.ply"))]
    for parent_name in parent_names:
        # load parent ply
        gt_parent_ply_path = gt_parents_dir / f"{parent_name}.ply"
        parent_pcd = o3d.io.read_point_cloud(str(gt_parent_ply_path))
        parent_pcd.paint_uniform_color(np.random.rand(3))

        # identify interaction segment
        segment = [seg for seg in GT_ARTICULATION[room][scene].values() if seg.axis_name == parent_name][0]
        segment_objs[(segment.start_time, segment.end_time)] = parent_pcd

        # parent_pcd.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #         radius=0.05, max_nn=30
        #     )
        # )

        # parent_pcd.orient_normals_consistent_tangent_plane(50)

        # radii = [0.005, 0.01, 0.02, 0.04, 0.08]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     parent_pcd,
        #     o3d.utility.DoubleVector(radii)
        # )

        plot_objects.append(parent_pcd)

    # USING PREDICTED ARTICULATION MODELS AND OBJECTS
    plot_objects = []
    plot_axes = []
    for idx, obj in enumerate(hierarchy["objects"]):
        if 'model' in obj:
            print(f"Object {idx} has model {obj['model'].type}")
            print(obj['model'])

            obj_color = np.random.rand(3)

            # create object point clouds

            if (obj["model"].start_idx, obj["model"].end_idx) not in segment_objs:
                print(f"Warning: no segment object found for indices {obj['model'].start_idx}, {obj['model'].end_idx}")
                continue

            seg_obj = segment_objs[(obj["model"].start_idx, obj["model"].end_idx)]
            seg_obj.paint_uniform_color(obj_color)
            plot_objects.append(seg_obj)

            axis_color = obj_color.copy()
            axis_color[0] = min(1.0, axis_color[0] + 0.2)
            axis_color[1] = max(0.0, axis_color[1] - 0.2)
            axis_color[2] = max(0.0, axis_color[2] - 0.2)
            # axis_color = (1,0,0)  # red for axes
            # visualize object model axes
            if obj['model'].type == 'prismatic':
                axis_start = np.array(obj['model'].position) - 0.5 * np.array(obj['model'].axis)
                axis_end = np.array(obj['model'].position) + 0.5 * np.array(obj['model'].axis)
                cyl, cone = cylinder_cone_arrow(axis_start, axis_end, cyl_radius=0.03, cone_radius=0.08, cone_length=0.12, color=axis_color)
                plot_axes.append(cyl)
                plot_axes.append(cone)
            elif obj['model'].type == 'revolute':
                axis_start = np.array(obj['model'].position) - 0.5 * np.array(obj['model'].axis)
                axis_end = np.array(obj['model'].position) + 0.5 * np.array(obj['model'].axis)
                cyl, cone = cylinder_cone_arrow(axis_start, axis_end, cyl_radius=0.03, cone_radius=0.08, cone_length=0.12, color=axis_color)
                plot_axes.append(cyl)
                plot_axes.append(cone)
                arc_center = axis_start + (1 / 3) * (axis_end - axis_start)
                arc_tube, arc_cone = create_partial_torus_arrow(
                    arc_center, obj['model'].axis, major_radius=0.12, minor_radius=0.015, color=(0.0, 0.0, 1.0)
                )
                plot_axes.append(arc_tube)
                plot_axes.append(arc_cone)

    # reject all points in pcd that are within 0.02m of any plot_object
    combined_pcd = o3d.geometry.PointCloud()
    for pobj in plot_objects:
        combined_pcd += pobj
    combined_tree = o3d.geometry.KDTreeFlann(combined_pcd)
    pcd_points = np.asarray(pcd.points)
    mask = np.ones(len(pcd_points), dtype=bool)
    for i, p in enumerate(pcd_points):
        [_, idx, _] = combined_tree.search_radius_vector_3d(p, 0.05)
        if len(idx) > 0:
            mask[i] = False
    pcd.points = o3d.utility.Vector3dVector(pcd_points[mask])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

    mesh = remove_triangles_near_point_cloud(mesh, combined_pcd, distance_threshold=0.03)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # white
    opt.light_on = False  # disables main light, but NOT fully unlit

    # convert plot_objectrs to TriangleMesh if they are PointCloud
    # for i in range(len(plot_objects)):
    #     if isinstance(plot_objects[i], o3d.geometry.PointCloud):
    #         pcd = plot_objects[i]
    #         radii = [0.005, 0.01, 0.02]
    #         # compute normals
    #         pcd.estimate_normals(
    #             search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #                 radius=0.05, max_nn=30
    #             )
    #         )
    #         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #             pcd,
    #             o3d.utility.DoubleVector(radii)
    #         )
    #         plot_objects[i] = mesh

    for pobj in plot_objects:
        vis.add_geometry(pobj)
    for paxis in plot_axes:
        vis.add_geometry(paxis)
    # vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

    # Example visualization of an arrow
    start = [0, 0, 0]
    end = [1, 0.5, 0.2]

    cyl, cone = cylinder_cone_arrow(start, end)
    art_viz = [cyl, cone]
    o3d.visualization.draw_geometries([mesh] + plot_objects)

    npz_path = os.path.join(load_dir, "tracks_smoothed.npz")
    if os.path.exists(npz_path):
        loaded_data = np.load(npz_path, allow_pickle=True)
        pred_3d_tracks_segments = loaded_data["tracks"]
        pred_visibility_segments = loaded_data["visibility"]
        loguru.logger.info("Loaded 3D tracks and visibility after tracking")

    # Load articulation models from file
    segment_articulations = load_articulations(load_dir)
    loguru.logger.info(f"Loaded articulation models from {load_dir}/articulations/")


@hydra.main(version_base=None, config_path="../../configs", config_name="visualize")
def main(cfg: DictConfig) -> None:
    """Hydra entry point that runs the articulated scene graph visualization.

    Args:
        cfg: Hydra configuration for the visualization run.
    """
    run(cfg)


if __name__ == "__main__":
    main()
