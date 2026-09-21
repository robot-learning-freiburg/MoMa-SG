import csv
from dataclasses import dataclass
from glob import glob
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass
class InteractionSegment:
    """A single labeled human-object interaction interval within a scene recording.

    Attributes:
        obj_name (str): Name of the interacted-with object.
        start_time (float): Interaction start time (or frame index).
        end_time (float): Interaction end time (or frame index).
        dense (np.ndarray, optional): Optional dense per-frame annotation array for the segment. Defaults to None.
    """

    obj_name: str
    start_time: float
    end_time: float
    dense: np.ndarray = None


@dataclass
class Articulation:
    """Ground-truth articulation (joint) parameters for one labeled axis of an articulated object.

    Attributes:
        position (np.ndarray): (3,) point on the articulation axis, in scene coordinates.
        axis (np.ndarray): (3,) unit direction vector of the articulation axis.
        type (str): Joint type, e.g. "PRISMATIC" or "REVOLUTE".
        twist (np.ndarray): (6,) screw/twist vector [angular; linear] representing the joint motion.
        difficulty (str): Qualitative difficulty label for this articulation.
        axis_name (str): Name of the axis/joint as given in the annotation files.
        idx (int): Row index in the source interaction-cue CSV this articulation was built from.
        start_time (float): Interaction start time associated with this articulation instance.
        end_time (float): Interaction end time associated with this articulation instance.
        category (str, optional): Semantic category of the object. Defaults to None.
    """

    position: np.ndarray
    axis: np.ndarray
    type: str
    twist: np.ndarray
    difficulty: str
    axis_name: str
    idx: int
    start_time: float
    end_time: float
    category: str = None


def load_arti4d_ground_truth(arti4d_data_root):
    """Load ARTI4D ground-truth articulation annotations for all rooms/scenes.

    Reads ``metadata.yaml`` under `arti4d_data_root` for per-axis joint types and difficulty
    labels, then walks every ``matched_cues.csv`` interaction-cue file paired with its scene's
    ``scene_*.json`` axis-definition file to build a screw-twist for each verified interaction
    segment.

    Args:
        arti4d_data_root (str or Path): Root directory of the ARTI4D ground-truth dataset,
            containing `metadata.yaml` and per-room/scene subdirectories with `matched_cues.csv`
            and `scene_*.json` files.

    Returns:
        tuple:
            - GT_ARTICULATION (dict): Nested dict `room -> scene -> (start_time, end_time) ->
              Articulation` of the loaded ground-truth articulations.
            - GT_OBJECT_TYPES (dict): Nested dict `room -> scene -> axis_name -> joint type`
              from the metadata.
            - GT_OBJECT_DIFFICULTY (dict): Nested dict `room -> scene -> axis_name -> difficulty`
              from the metadata.
    """

    GT_DATA_ROOT = Path(arti4d_data_root)
    with open(GT_DATA_ROOT / 'metadata.yaml', 'r') as f:
        # SCENE -> OBJECT -> TYPE
        metadata = yaml.safe_load(f)
        GT_OBJECT_TYPES = metadata['joint_types']
        GT_OBJECT_DIFFICULTY = metadata['difficulty']

    # List all rooms, scenes, and segments based on the metadata
    print('Listing all gt rooms, scenes, and segments...')
    rooms = list(GT_OBJECT_TYPES.keys())
    print(f'Found {len(rooms)} rooms:')
    for room in rooms:
        print(f'  - {room}')

    scenes = []
    print('\nScenes:')
    for room in rooms:
        for scene in GT_OBJECT_TYPES[room]:
            scenes.append(f'{scene}')
            print(f'  - {scene}')
    print(f'Total scenes: {len(scenes)}')

    segments = []
    print('\nSegments:')
    for room in rooms:
        for scene in GT_OBJECT_TYPES[room]:
            for segment in GT_OBJECT_TYPES[room][scene]:
                segments.append(f'{segment}')
                print(f'  - {room}/{scene}/{segment}')
    print(f'Total segments: {len(segments)}')

    # Clean up INTERACTION_SEGMENTS before loading
    GT_ARTICULATION = {}
    GT_ARTICULATION.clear()
    segment_counter = 0
    interaction_files = GT_DATA_ROOT.glob('**/matched_cues.csv')
    for path in interaction_files:
        room = path.parent.parent.stem
        scene = path.parent.stem
        if room not in GT_ARTICULATION:
            GT_ARTICULATION[room] = {}
        if scene not in GT_ARTICULATION[room]:
            GT_ARTICULATION[room][scene] = {}

        # open axis json to get object data
        json_axis_file = glob(str(GT_DATA_ROOT / room / scene / "scene_*.json"))[0]
        print(json_axis_file)
        axis_dict = {}
        with open(json_axis_file) as f:
            for axis_name, axis_data in json.load(f).items():
                axis_dict[axis_name] = axis_data

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row.get("VERIFICATION", "").strip().upper() == "VERIFIED":
                    axis_name = row["AXIS_NAME"].strip()
                    start = int(row["CUE_START"])
                    end = int(row["CUE_END"])
                    # semantics = row["SEMANTICS"].strip()

                    if GT_OBJECT_TYPES[room][scene][axis_name].upper() == "PRISMATIC":
                        print(f"Loading prismatic axis {axis_name} for {room}/{scene} from {start} to {end}")
                        gt_twist = np.array(
                            [0, 0, 0, axis_dict[axis_name]['axis'][0], axis_dict[axis_name]['axis'][1], axis_dict[axis_name]['axis'][2]]
                        )
                    elif GT_OBJECT_TYPES[room][scene][axis_name].upper() == "REVOLUTE":
                        print(f"Loading revolute axis {axis_name} for {room}/{scene} from {start} to {end}")
                        transl_part = np.cross(-np.array(axis_dict[axis_name]["axis"]), np.array(axis_dict[axis_name]['position']))
                        gt_twist = np.array(
                            [
                                axis_dict[axis_name]['axis'][0],
                                axis_dict[axis_name]['axis'][1],
                                axis_dict[axis_name]['axis'][2],
                                transl_part[0],
                                transl_part[1],
                                transl_part[2],
                            ]
                        )
                    GT_ARTICULATION[room][scene][(start, end)] = Articulation(
                        axis_name=axis_name,
                        position=np.asarray(axis_dict[axis_name]['position']),
                        axis=np.asarray(axis_dict[axis_name]['axis']),
                        type=GT_OBJECT_TYPES[room][scene][axis_name],
                        difficulty=GT_OBJECT_DIFFICULTY[room][scene][axis_name],
                        twist=gt_twist,
                        # category=semantics,
                        start_time=start,
                        end_time=end,
                        idx=i,
                    )
                    segment_counter += 1

    print(f"Total number of loaded interaction + axis segments: {segment_counter}")
    rooms = []
    for room in GT_ARTICULATION.keys():
        if room not in rooms:
            rooms.append(room)
    print(f"Total number of rooms with interaction segments: {len(rooms)}")

    scenes = []
    for room in GT_ARTICULATION.keys():
        for scene in GT_ARTICULATION[room].keys():
            if scene not in scenes:
                scenes.append(scene)
    print(f"Total number of scenes with interaction segments: {len(scenes)}")

    return GT_ARTICULATION, GT_OBJECT_TYPES, GT_OBJECT_DIFFICULTY
