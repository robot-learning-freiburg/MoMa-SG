"""
Synthetic trajectory generator for revolute and prismatic joints.

    - Revolute: a door/lid that opens by `opening_angle_deg` degrees.
                            N points are sampled on the surface being swept (random radii
                            and random positions along the hinge axis).
    - Prismatic: a drawer that slides out by `travel_cm` centimetres.
                             N points are sampled on the face of the drawer.

Both return:
        trajectories : np.ndarray, shape (N, T, 3), float32  – 3-D positions
        visibility   : np.ndarray, shape (N, T),    float32  – all ones
"""

import random
from typing import Literal

import hydra
import matplotlib
from moma_sg.articulation.point_estimator import estimate_articulation_model, estimate_articulation_model_pris, estimate_articulation_model_rev
import numpy as np
from omegaconf import DictConfig

random.seed(42)

# TODO remove before release

# ---------------------------------------------------------------------------
# Revolute  ("door opens by X degrees")
# ---------------------------------------------------------------------------


def generate_revolute_trajectories(
    N: int,
    T: int,
    opening_angle_deg: float = 90.0,
    radius_range: tuple[float, float] = (0.05, 1.0),
    hinge_length: float = 1.0,
    hinge_origin: np.ndarray = None,
    hinge_axis: np.ndarray = None,
    noise_std: float = 0.0,
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a door/lid opening from 0° to `opening_angle_deg`.

    Points are randomly distributed on the swinging surface:
      - random radius in `radius_range`  (distance from hinge axis)
      - random offset along the hinge axis in [0, hinge_length]

    Args:
        N:                 Number of point trajectories.
        T:                 Number of timesteps.
        opening_angle_deg: Total opening angle in degrees (e.g. 90 for a door).
        radius_range:      (r_min, r_max) – spread of points away from the hinge.
        hinge_length:      Extent of the hinge along its axis (e.g. door height).
        hinge_origin:      3-D origin of the hinge line (default: [0, 0, 0]).
        hinge_axis:        Unit vector along the hinge (default: [0, 0, 1]).
        noise_std:         Std-dev of isotropic Gaussian noise per point.
        random_seed:       Optional RNG seed.

    Returns:
        trajectories: (N, T, 3) float32
        visibility:   (N, T)    float32, all ones
    """
    rng = np.random.default_rng(random_seed)

    if hinge_origin is None:
        hinge_origin = np.zeros(3)
    hinge_origin = np.asarray(hinge_origin, dtype=float)

    if hinge_axis is None:
        hinge_axis = np.array([0.0, 0.0, 1.0])
    hinge_axis = np.asarray(hinge_axis, dtype=float)
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)

    # Radial directions: two basis vectors perpendicular to the hinge axis
    u = _perpendicular_unit_vector(hinge_axis)  # "door closed" direction
    v = np.cross(hinge_axis, u)  # "door swings into" direction

    # Per-point random radius and position along the hinge
    radii = rng.uniform(radius_range[0], radius_range[1], size=N)  # (N,)
    axial = rng.uniform(0.0, hinge_length, size=N)  # (N,)

    # Angle schedule: 0 → opening_angle_deg over T steps
    angles = np.deg2rad(np.linspace(0.0, opening_angle_deg, T))  # (T,)

    # Point positions at each timestep
    # p(n, t) = hinge_origin
    #         + axial[n] * hinge_axis
    #         + radii[n] * (cos(angle[t]) * u + sin(angle[t]) * v)
    cos_a = np.cos(angles)  # (T,)
    sin_a = np.sin(angles)  # (T,)

    # (N, T, 3)
    trajectories = (
        hinge_origin[None, None, :]
        + axial[:, None, None] * hinge_axis[None, None, :]
        + radii[:, None, None] * (cos_a[None, :, None] * u[None, None, :] + sin_a[None, :, None] * v[None, None, :])
    )

    if noise_std > 0.0:
        trajectories += rng.normal(0.0, noise_std, size=trajectories.shape)

    visibility = np.ones((N, T), dtype=np.float32)
    return trajectories.astype(np.float32), visibility


# ---------------------------------------------------------------------------
# Prismatic  ("drawer slides out by Y cm")
# ---------------------------------------------------------------------------


def generate_prismatic_trajectories(
    N: int,
    T: int,
    travel_cm: float = 20.0,
    face_size: tuple[float, float] = (0.3, 0.2),
    slide_origin: np.ndarray = None,
    slide_axis: np.ndarray = None,
    noise_std: float = 0.0,
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a drawer sliding out by `travel_cm` centimetres.

    Points are randomly distributed on the front face of the drawer
    (a rectangle of size `face_size`).

    Args:
        N:           Number of point trajectories.
        T:           Number of timesteps.
        travel_cm:   Total travel distance in centimetres (converted to metres
                     internally: 1 cm = 0.01 m).
        face_size:   (width, height) of the drawer face in metres.
        slide_origin: 3-D starting position of the drawer face centre (default: origin).
        slide_axis:  Unit vector of the sliding direction (default: [1, 0, 0]).
        noise_std:   Std-dev of isotropic Gaussian noise per point.
        random_seed: Optional RNG seed.

    Returns:
        trajectories: (N, T, 3) float32
        visibility:   (N, T)    float32, all ones
    """
    rng = np.random.default_rng(random_seed)

    travel_m = travel_cm * 0.01  # convert cm → m

    if slide_origin is None:
        slide_origin = np.zeros(3)
    slide_origin = np.asarray(slide_origin, dtype=float)

    if slide_axis is None:
        slide_axis = np.array([1.0, 0.0, 0.0])
    slide_axis = np.asarray(slide_axis, dtype=float)
    slide_axis = slide_axis / np.linalg.norm(slide_axis)

    # Two axes spanning the drawer face (perpendicular to slide direction)
    face_u = _perpendicular_unit_vector(slide_axis)
    face_v = np.cross(slide_axis, face_u)

    w, h = face_size
    # Random 2-D position on the face for each point
    offset_u = rng.uniform(-w / 2, w / 2, size=N)  # (N,)
    offset_v = rng.uniform(-h / 2, h / 2, size=N)  # (N,)

    # Base positions on the face at t=0
    base = slide_origin[None, :] + offset_u[:, None] * face_u[None, :] + offset_v[:, None] * face_v[None, :]  # (N, 3)

    # Displacement schedule: 0 → travel_m over T steps
    displacements = np.linspace(0.0, travel_m, T)  # (T,)

    # trajectories[n, t] = base[n] + displacements[t] * slide_axis
    trajectories = base[:, None, :] + displacements[None, :, None] * slide_axis[None, None, :]  # (N, T, 3)

    if noise_std > 0.0:
        trajectories += rng.normal(0.0, noise_std, size=trajectories.shape)

    visibility = np.ones((N, T), dtype=np.float32)
    return trajectories.astype(np.float32), visibility


# ---------------------------------------------------------------------------
# Unified entry-point
# ---------------------------------------------------------------------------


def generate_trajectories(
    joint_type: Literal["revolute", "prismatic"],
    N: int,
    T: int,
    # --- revolute ---
    opening_angle_deg: float = 90.0,
    radius_range: tuple[float, float] = (0.05, 1.0),
    hinge_length: float = 1.0,
    hinge_origin: np.ndarray = None,
    hinge_axis: np.ndarray = None,
    # --- prismatic ---
    travel_cm: float = 20.0,
    face_size: tuple[float, float] = (0.3, 0.2),
    slide_origin: np.ndarray = None,
    slide_axis: np.ndarray = None,
    # --- shared ---
    noise_std: float = 0.0,
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified entry-point – dispatches to revolute or prismatic generator.

    Returns:
        trajectories : (N, T, 3) float32
        visibility   : (N, T)    float32, all ones
    """
    if joint_type == "revolute":
        return generate_revolute_trajectories(
            N=N,
            T=T,
            opening_angle_deg=opening_angle_deg,
            radius_range=radius_range,
            hinge_length=hinge_length,
            hinge_origin=hinge_origin,
            hinge_axis=hinge_axis,
            noise_std=noise_std,
            random_seed=random_seed,
        )
    elif joint_type == "prismatic":
        return generate_prismatic_trajectories(
            N=N,
            T=T,
            travel_cm=travel_cm,
            face_size=face_size,
            slide_origin=slide_origin,
            slide_axis=slide_axis,
            noise_std=noise_std,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown joint_type '{joint_type}'. Choose 'revolute' or 'prismatic'.")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)


def plot_trajectories(
    trajectories: np.ndarray,
    gt_origin: np.ndarray = None,
    gt_axis: np.ndarray = None,
    pred_origin: np.ndarray = None,
    pred_axis: np.ndarray = None,
    axis_length: float = None,
    title: str = "Joint trajectories",
    show: bool = True,
    save_path: str = None,
) -> "plt.Figure":
    """
    3-D plot of point trajectories with optional GT and predicted joint axes.

    Each of the N trajectories is drawn in a distinct colour. The first and
    last positions are marked with a circle (○) and a cross (×) respectively
    so the direction of motion is immediately visible.

    Args:
        trajectories : (N, T, 3) array of point positions.
        gt_origin    : (3,) origin of the ground-truth axis.
        gt_axis      : (3,) direction of the GT axis (need not be unit).
        pred_origin  : (3,) origin of the predicted axis.
        pred_axis    : (3,) direction of the predicted axis (need not be unit).
        axis_length  : Length to draw each axis arrow. Defaults to ~35 % of
                       the scene bounding-box diagonal.
        title        : Figure title.
        show         : Call plt.show() at the end.
        save_path    : If given, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib

    matplotlib.use('TkAgg')
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

    N, T, _ = trajectories.shape

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- colour palette (cycles for N > 20) ----------------------------------
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(N)]

    # --- point trajectories --------------------------------------------------
    for n in range(N):
        pts = trajectories[n]  # (T, 3)
        c = colors[n]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, linewidth=1.4, alpha=0.85)
        ax.scatter(*pts[0], color=c, s=30, marker="o", zorder=5, edgecolors="none")
        ax.scatter(*pts[-1], color=c, s=35, marker="X", zorder=5, edgecolors="k", linewidths=0.4)

    # --- auto axis_length from bounding box ----------------------------------
    if axis_length is None:
        all_pts = trajectories.reshape(-1, 3)
        diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        axis_length = max(diag * 0.35, 1e-3)

    # --- helper: draw one labelled axis arrow + origin marker ----------------
    def _draw_axis(origin, direction, color, label, linestyle="-"):
        """Draw a labelled 3-D quiver arrow (plus origin marker) for a joint axis
        and return an invisible Line2D proxy artist for the legend."""
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        ax.quiver(
            o[0],
            o[1],
            o[2],
            d[0] * axis_length,
            d[1] * axis_length,
            d[2] * axis_length,
            color=color,
            linewidth=2.5,
            arrow_length_ratio=0.12,
            linestyle=linestyle,
        )
        ax.scatter(*o, color=color, s=70, zorder=6, edgecolors="k", linewidths=0.6)
        # invisible proxy for legend
        return Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="k",
        )

    legend_handles = []

    if gt_origin is not None and gt_axis is not None:
        h = _draw_axis(gt_origin, gt_axis, color="#2ecc71", label="GT axis", linestyle="-")
        legend_handles.append(h)

    if pred_origin is not None and pred_axis is not None:
        h = _draw_axis(pred_origin, pred_axis, color="#e74c3c", label="Pred axis", linestyle="--")
        legend_handles.append(h)

    # --- shared legend entries for trajectories + markers --------------------
    legend_handles += [
        Line2D([0], [0], color="grey", linewidth=1.4, label=f"trajectories (N={N})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=6, label="t = 0  (start)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="grey", markeredgecolor="k", markersize=6, label="t = T−1  (end)"),
    ]

    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  figure saved → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _perpendicular_unit_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to `v`."""
    v = v / np.linalg.norm(v)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    perp = candidate - np.dot(candidate, v) * v
    return perp / np.linalg.norm(perp)


import os

import matplotlib.pyplot as plt


def plot_results(
    pris_results_ours: dict,
    pris_results_general: dict,
    pris_results_pris: dict,
    rev_results_ours: dict,
    rev_results_general: dict,
    rev_results_rev: dict,
    output_dir: str = "outputs",
):
    """
    Plot and save comparison figures for the synthetic revolute/prismatic
    articulation experiments: prismatic axis error, revolute axis error, and
    revolute position error, each vs. the independent variable (travel distance
    or opening angle) for the "ours" (regularized twist), "general" (twist,
    no regularization), and privileged (type-specific transform) estimators.

    Each `*_results_*` argument maps the independent variable (cm or degrees)
    to an (axis_error_deg, position_error_m) tuple. Figures are saved as PDFs
    under `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _unzip(d, key_idx):
        """Sort dict `d` by key and extract the `key_idx`-th element of each value
        into a parallel (xs, ys) pair of float arrays."""
        xs = sorted(d.keys())
        ys = [d[x][key_idx] for x in xs]
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    AXIS_IDX = 0
    POS_IDX = 1

    # Plot 1 – Prismatic axis error
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 5)
    x_ours, y_ours = _unzip(pris_results_ours, AXIS_IDX)
    x_priv, y_priv = _unzip(pris_results_pris, AXIS_IDX)
    x_general, y_general = _unzip(pris_results_general, AXIS_IDX)
    ax.plot(x_ours, y_ours, marker="o", label="Regularized Twist (Ours)")
    ax.plot(x_priv, y_priv, marker="s", linestyle="--", label="On-manifold (Prismatic Transform)")
    ax.plot(x_general, y_general, marker="^", linestyle="-.", label="Twist (no regularization)")
    ax.set_xlabel("Distance traveled [cm]")
    ax.set_ylabel("Axis error [deg]")
    ax.set_title("Prismatic Joint – Axis Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pris_axis_error.pdf"), dpi=150)
    plt.close(fig)

    # Plot 2 – Revolute axis error
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 5)
    x_ours, y_ours = _unzip(rev_results_ours, AXIS_IDX)
    x_priv, y_priv = _unzip(rev_results_rev, AXIS_IDX)
    x_general, y_general = _unzip(rev_results_general, AXIS_IDX)
    ax.plot(x_ours, y_ours, marker="o", label="Regularized Twist (Ours)")
    ax.plot(x_priv, y_priv, marker="s", linestyle="--", label="On-manifold (Revolute Transform)")
    ax.plot(x_general, y_general, marker="^", linestyle="-.", label="Twist (no regularization)")
    ax.set_xlabel("Opening angle [deg]")
    ax.set_ylabel("Axis error [deg]")
    ax.set_title("Revolute Joint - Axis Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rev_axis_error.pdf"), dpi=150)
    plt.close(fig)

    # Plot 3 – Revolute position error
    fig, ax = plt.subplots()
    # set plot ratio to 10:5
    fig.set_size_inches(6, 5)
    x_ours, y_ours = _unzip(rev_results_ours, POS_IDX)
    x_priv, y_priv = _unzip(rev_results_rev, POS_IDX)
    x_general, y_general = _unzip(rev_results_general, POS_IDX)
    ax.plot(x_ours, y_ours, marker="o", label="Regularized Twist (Ours)")
    ax.plot(x_priv, y_priv, marker="s", linestyle="--", label="On-manifold (Revolute Transform)")
    ax.plot(x_general, y_general, marker="^", linestyle="-.", label="Twist (no regularization)")
    ax.set_xlabel("Opening angle [°]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Revolute Joint – Position Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rev_pos_error.pdf"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

from moma_sg.data.arti4d import Articulation
from moma_sg.utils.metrics import compute_articulation_delta_point_and_axis


@hydra.main(version_base=None, config_path="../../configs", config_name="momasg")
def main(cfg: DictConfig):
    """
    Demo/evaluation entry point: generates synthetic revolute (varying opening
    angle) and prismatic (varying travel distance) trajectories, fits axis
    models with the regularized-twist ("ours"), unregularized-twist
    ("general"), and type-privileged estimators for each, computes axis/position
    error against the known ground-truth articulation, and plots the comparison
    via `plot_results`.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless – remove when running interactively
    np.set_printoptions(precision=4, suppress=True)

    cfg.articulation.cos_thresh = 0.999
    cfg.articulation.type_amb_weight = 5e8
    cfg.articulation.optim_params.ftol = 1e-5
    cfg.articulation.optim_params.gtol = 1e-6
    cfg.articulation.optim_params.xtol = 1e-6
    cfg.articulation.optim_params.max_nfev = 100
    # cfg.articulation.optim_params.loss = "soft_l1"
    # cfg.articulation.twist_init = np.random.rand(6).tolist()
    # cfg.articulation.theta_init = 0.0

    # ------------------------------------------------------------------
    # Example 1 – Revolute: door opens 90° with a slightly wrong pred axis
    # ------------------------------------------------------------------

    rev_results_ours = dict()
    rev_results_general = dict()
    rev_results_rev = dict()
    pris_results_ours = dict()
    pris_results_general = dict()
    pris_results_pris = dict()

    for angle in range(0, 91, 5):
        print(f"\n--- Generating revolute trajectories with opening angle {angle}° ---")

        gt_rev = Articulation(
            type="REVOLUTE",
            position=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            twist=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            difficulty="easy",
            axis_name="demo_revolute_axis",
            idx=0,
            start_time=0.0,
            end_time=10.0,
        )

        traj_rev, vis_rev = generate_trajectories(
            joint_type=gt_rev.type.lower(),
            N=12,
            T=20,
            opening_angle_deg=angle,
            radius_range=(0.1, 0.8),
            hinge_length=1.5,
            hinge_origin=gt_rev.position,
            hinge_axis=gt_rev.axis,
            noise_std=0.001,
            random_seed=42,
        )
        print(f"Revolute  traj: {traj_rev.shape}  vis all-ones: {vis_rev.all()}")

        # Assume regularized twist
        model, pairs, pairs_t = estimate_articulation_model(cfg, traj_rev, vis_rev.astype(bool), alpha=1.0)
        if model.position is None or model.axis is None:
            print(f"  estimation failed for opening angle {angle}°")
            rev_results_ours[angle] = (float('nan'), float('nan'))
            continue
        rev_error_ours = compute_articulation_delta_point_and_axis(gt_rev, np.array(model.position), np.array(model.axis))
        rev_results_ours[angle] = rev_error_ours
        print(f"Revolute joint with opening angle {angle}°: position error = {rev_error_ours[1]:.4f} m, axis error = {rev_error_ours[0]:.2f}°")

        # Assume general twist
        model, pairs, pairs_t = estimate_articulation_model(cfg, traj_rev, vis_rev.astype(bool), alpha=0.0)
        if model.position is None or model.axis is None:
            print(f"  estimation failed for opening angle {angle}°")
            rev_results_general[angle] = (float('nan'), float('nan'))
            continue
        rev_error_general = compute_articulation_delta_point_and_axis(gt_rev, np.array(model.position), np.array(model.axis))
        rev_results_general[angle] = rev_error_general
        print(f"Revolute joint with opening angle {angle}°: position error = {rev_error_general[1]:.4f} m, axis error = {rev_error_general[0]:.2f}°")

        # Assume revolute twist
        cfg.articulation.optim_params.ftol = 5e-8
        cfg.articulation.optim_params.gtol = 1e-8
        cfg.articulation.optim_params.xtol = 1e-8
        model_rev, pairs_rev, pairs_t_rev = estimate_articulation_model_rev(cfg, traj_rev, vis_rev.astype(bool))
        if model_rev.position is None or model_rev.axis is None:
            print(f"  estimation failed for opening angle {angle}°")
            rev_results_rev[angle] = (float('nan'), float('nan'))
            continue
        rev_error_rev = compute_articulation_delta_point_and_axis(gt_rev, np.array(model_rev.position), np.array(model_rev.axis))
        rev_results_rev[angle] = rev_error_rev

        print(
            f"Privileged/ours: {rev_error_rev[0]:.5f}/{rev_error_ours[0]:.5f}° axis error, {rev_error_rev[1]:.5f}/{rev_error_ours[1]:.5f} m pos error"
        )

        # plot_trajectories(
        #     trajectories=traj_rev,
        #     gt_origin=gt_rev.position,
        #     gt_axis=gt_rev.axis,
        #     pred_origin=pred_hinge_origin,
        #     pred_axis=pred_hinge_axis,
        #     title="Revolute joint",
        #     show=True,
        #     save_path="outputs/revolute_plot.png",
        # )

        plot_results(
            pris_results_ours=pris_results_ours,
            pris_results_general=pris_results_general,
            pris_results_pris=pris_results_pris,
            rev_results_ours=rev_results_ours,
            rev_results_general=rev_results_general,
            rev_results_rev=rev_results_rev,
            output_dir="outputs",
        )

    # ------------------------------------------------------------------
    # Example 2 – Prismatic: drawer slides 25 cm with a slightly wrong pred axis
    # ------------------------------------------------------------------

    for dist in range(10, 100, 5):
        gt_pris = Articulation(
            type="prismatic".upper(),
            position=np.array([0.0, 0.0, 0.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            twist=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            difficulty="easy",
            axis_name="demo_prismatic_axis",
            idx=0,
            start_time=0.0,
            end_time=10.0,
        )

        traj_pris, vis_pris = generate_trajectories(
            joint_type=gt_pris.type.lower(),
            N=12,
            T=20,
            travel_cm=dist,
            face_size=(0.4, 0.25),
            slide_origin=gt_pris.position,
            slide_axis=gt_pris.axis,
            noise_std=0.0006,
            random_seed=7,
        )
        print(f"Prismatic traj: {traj_pris.shape}  vis all-ones: {vis_pris.all()}")

        # Assume regularized twist
        cfg.articulation.optim_params.ftol = 5e-5
        cfg.articulation.optim_params.gtol = 1e-6
        cfg.articulation.optim_params.xtol = 1e-6
        model, pairs, pairs_t = estimate_articulation_model(cfg, traj_pris, vis_pris.astype(bool), alpha=1.0)
        if model.position is None or model.axis is None:
            print(f"  estimation failed for opening dist {dist}°")
            pris_results_ours[dist] = (float('nan'), float('nan'))
            continue
        pris_error_ours = compute_articulation_delta_point_and_axis(gt_pris, np.array(model.position), np.array(model.axis))
        pris_results_ours[dist] = pris_error_ours
        print(f"Prismatic joint with travel distance {dist} cm: position error = {pris_error_ours[1]:.4f} m, axis error = {pris_error_ours[0]:.2f}°")

        # Assume general twist
        cfg.articulation.optim_params.ftol = 5e-5
        cfg.articulation.optim_params.gtol = 1e-6
        cfg.articulation.optim_params.xtol = 1e-6
        model_general, pairs_general, pairs_t_general = estimate_articulation_model(cfg, traj_pris, vis_pris.astype(bool), alpha=0.0)
        if model_general.position is None or model_general.axis is None:
            print(f"  estimation failed for opening dist {dist}°")
            pris_results_general[dist] = (float('nan'), float('nan'))
            continue
        pris_error_general = compute_articulation_delta_point_and_axis(gt_pris, np.array(model_general.position), np.array(model_general.axis))
        pris_results_general[dist] = pris_error_general

        # Assume prismatic twist
        cfg.articulation.optim_params.ftol = 5e-8
        cfg.articulation.optim_params.gtol = 1e-8
        cfg.articulation.optim_params.xtol = 1e-8
        model_pris, pairs_rev, pairs_t_rev = estimate_articulation_model_pris(cfg, traj_pris, vis_pris.astype(bool))
        if model_pris.position is None or model_pris.axis is None:
            print(f"  estimation failed for opening dist {dist}°")
            pris_results_pris[dist] = (float('nan'), float('nan'))
            continue
        pris_error_pris = compute_articulation_delta_point_and_axis(gt_pris, np.array(model_pris.position), np.array(model_pris.axis))
        pris_results_pris[dist] = pris_error_pris

        print(
            f"Privileged/ours: {pris_error_pris[0]:.5f}/{pris_error_ours[0]:.5f}° axis error, {pris_error_pris[1]:.5f}/{pris_error_ours[1]:.5f} m pos error"
        )

        # plot_trajectories(
        #     trajectories=traj_pris,
        #     gt_origin=gt_pris.position,
        #     gt_axis=gt_pris.axis,
        #     pred_origin=pred_pris_origin,
        #     pred_axis=pred_pris_axis,
        #     title="Prismatic joint – drawer slides 25 cm",
        #     show=True,
        #     save_path="outputs/prismatic_plot.png",
        # )
        plot_results(
            pris_results_ours=pris_results_ours,
            pris_results_general=pris_results_general,
            pris_results_pris=pris_results_pris,
            rev_results_ours=rev_results_ours,
            rev_results_general=rev_results_general,
            rev_results_rev=rev_results_rev,
            output_dir="outputs",
        )


if __name__ == "__main__":
    main()
