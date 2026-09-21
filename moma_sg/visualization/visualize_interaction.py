import json
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path

import hydra
import loguru
from moma_sg.graph.engine import MoMaSG
from moma_sg.utils.mapping import Hierarchy
from moma_sg.utils.visualization import plot_full_momasg
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="../../configs", config_name="visualize")
def main(cfg: DictConfig) -> None:
    """Render a video of the hand-object interaction segmentation for a cached scene.

    Loads the cached MoMaSG frames/hierarchy and articulation models for the scene
    specified in `cfg`, optionally uses ground-truth interaction segments, loads the
    precomputed human masks, and writes an overlay video via `plot_full_momasg`.

    Args:
        cfg: Hydra configuration with dataset, cache, debugging, and interaction settings.
    """
    cfg.engine = False  # visualization only needs cached frames/hierarchy, not the heavy models
    moma_sg = MoMaSG(cfg)

    loguru.logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    scene_name = Path(cfg.dataset.root_path).stem
    scene_type = Path(cfg.dataset.root_path).parent.name
    loguru.logger.info(f"Processing scene: {scene_name} of type {scene_type}")

    save_dir = None
    if cfg.cache.save_results:
        save_dir = os.path.join(cfg.cache.output_dir, scene_type, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        loguru.logger.info(f"Saving results to {save_dir}")
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "scene_name": scene_name,
                    "scene_type": scene_type,
                    "config": OmegaConf.to_container(cfg),
                },
                f,
                indent=4,
            )

    if cfg.interaction.use_gt_segments:
        pred_segments = [(s[1][0], s[1][1]) for s in moma_sg.dataset.interactions]

    hierarchy = Hierarchy(cfg, None)
    hierarchy.load(Path(save_dir) / "hierarchy.pkl")

    torch.cuda.empty_cache()

    # Build lookup: (start_idx, end_idx) -> hierarchy object with articulation model
    segment_objs = {(obj["model"].start_idx, obj["model"].end_idx): obj for obj in hierarchy.objects if "model" in obj}

    human_masks_path = os.path.join(
        cfg.debugging.output_dir,
        scene_type,
        scene_name,
        "human_masks.npz",
    )
    human_masks = moma_sg.get_human_masks(
        use_precomputed=True,
        save_masks=cfg.cache.save_prior_masks,
        path=os.path.dirname(human_masks_path),
    )

    plot_full_momasg(
        segment_objs=segment_objs,
        pred_segments=pred_segments,
        rgb_images=moma_sg.rgb_frames,
        depth_images=moma_sg.depth_frames,
        camera_poses=moma_sg.cam_poses,
        dataset=moma_sg.dataset,
        hierarchy=hierarchy,
        human_masks=human_masks,
        fps=15,
        output_dir=f"momasg_video/{scene_type}/{scene_name}/",
        save_video=True,
        point_size=6.0,
    )


if __name__ == "__main__":
    main()
