from collections import Counter
from collections.abc import Iterable
import copy
from typing import List

import cv2
import faiss
import loguru
import matplotlib
import numpy as np
import open3d as o3d
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def prepare_semsam_image(image):
    """
    Preprocess an RGB image for SEM-SAM: resize so the shorter side is 640 px
    (bicubic) and produce both a numpy array and a CUDA tensor copy.

    Args:
        image (np.ndarray): Input RGB image (H, W, 3).

    Returns:
        tuple: (image_ori, images) where image_ori is the resized (H', W', 3)
            numpy array and images is the corresponding (3, H', W') CUDA tensor.
    """
    image = Image.fromarray(image).convert('RGB')
    t = []
    t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    return image_ori, images


def parse_masks(annotations, frames, idx):
    """
    Convert a list of SAM-style mask annotations into a grounded-observations
    dict and resize the masks/boxes to match the corresponding frame.

    Args:
        annotations (list[dict]): SAM annotations, each with 'segmentation',
            'stability_score', and 'bbox' (x, y, w, h) keys.
        frames: Sequence of frames indexable by idx, used as the resize target.
        idx (int): Index of the frame these annotations belong to.

    Returns:
        dict: Grounded observations with 'xyxy', 'mask', 'score', and 'frame_idx' keys.
    """
    masks = np.array([ann["segmentation"] for ann in annotations])
    scores = np.array([ann["stability_score"] for ann in annotations])
    xyxy = np.array([[ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]] for ann in annotations])
    gobs = resize_gobs({"xyxy": xyxy, "mask": masks, "score": scores, "frame_idx": idx}, frames[idx])
    return gobs


def to_numpy(tensor):
    """
    Convert a numpy array, torch tensor, or list of tensors/arrays to a numpy array.

    Args:
        tensor: np.ndarray, torch.Tensor, or list of either.

    Returns:
        np.ndarray: The converted array (None if given a list of an unsupported type).
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        if isinstance(tensor[0], torch.Tensor):
            return np.array([t.detach().cpu().numpy() for t in tensor])
        elif isinstance(tensor[0], np.ndarray):
            return np.array(tensor)


def to_tensor(input, device=None):
    """
    Convert a numpy array, list of arrays/tensors, or tensor to a torch.Tensor.

    Args:
        input: torch.Tensor, np.ndarray, or list of either.
        device: Optional torch device to move the result to.

    Returns:
        torch.Tensor: The converted tensor.
    """
    if isinstance(input, torch.Tensor):
        return input
    elif isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    elif isinstance(input, list):
        if isinstance(input[0], torch.Tensor):
            input = torch.stack(input)
        elif isinstance(input[0], np.ndarray):
            input = torch.tensor(input)
    if device is not None:
        return input.to(device)
    return input


def to_scalar(d: np.ndarray | torch.Tensor | float) -> int | float:
    '''
    Convert the d to a scalar
    '''
    if isinstance(d, float):
        return d

    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()

    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()

    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")


class DetectionList(list):
    """List of per-object/detection dicts with helpers for stacking, slicing,
    coloring, and (de)serializing their point-cloud/feature fields."""

    def get_values(self, key, idx: int = None):
        """
        Collect a given field from every detection in the list.

        Args:
            key: Dict key to extract from each detection.
            idx: Optional index into each value, extracted instead of the whole value.

        Returns:
            list: The extracted values, one per detection.
        """
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]

    def get_stacked_values_torch(self, key, idx: int = None):
        """
        Stack a given field from every detection into a single torch tensor.

        Open3D oriented/axis-aligned bounding boxes are converted to their
        (8,3) corner points, and numpy arrays are converted to tensors,
        before stacking.

        Args:
            key: Dict key to extract from each detection.
            idx: Optional index into each value, extracted instead of the whole value.

        Returns:
            torch.Tensor: Stacked values, shape (len(self), ...).
        """
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx: int = None):
        """
        Same as get_stacked_values_torch, but returns a numpy array.

        Args:
            key: Dict key to extract from each detection.
            idx: Optional index into each value, extracted instead of the whole value.

        Returns:
            np.ndarray: Stacked values, shape (len(self), ...).
        """
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)

    def __add__(self, other):
        """Return a new list that is a deep copy of self, extended with other."""
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list

    def __iadd__(self, other):
        """Extend this list in place with the items of other."""
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self

    def get_most_common_class(self) -> list[int]:
        """
        Determine the most frequently occurring class_id for each detection.

        Returns:
            list[int]: The most common class id per detection, in list order.
        """
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes

    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool = True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color

    def color_by_instance(self):
        """
        Paint each detection's point cloud and bounding box with a distinct
        instance color, using 'inst_color' if present, otherwise a color sampled
        evenly from the 'turbo' colormap across all detections.
        """
        if len(self) == 0:
            # Do nothing
            return

        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]

    def get_pcd(self):
        '''
        Get the point cloud of each detection
        '''
        pcds = []
        all_instances = o3d.geometry.PointCloud()
        for detection in self:
            if isinstance(detection['pcd'], o3d.geometry.PointCloud):
                all_instances += detection['pcd']
        return pcds

    def to_serializable(self):
        """
        Convert detections to a plain, picklable representation.

        Replaces the open3d point cloud/bbox and CLIP/text feature tensors with
        numpy arrays ('points', 'bbox_np', 'pcd_color_np') and drops the
        original 'pcd'/'bbox' entries.

        Returns:
            list[dict]: Serializable copies of the detections.
        """
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)

            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            if 'text_ft' in s_obj_dict:
                s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])

            s_obj_dict['points'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']

            s_obj_list.append(s_obj_dict)

        return s_obj_list


class MapObjectList(DetectionList):
    """DetectionList specialized for map objects, adding CLIP-feature
    similarity computation and (de)serialization that reconstructs point
    clouds and bounding boxes when loading."""

    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor
        new_clip_ft = to_tensor(new_clip_ft)

        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities

    def to_serializable(self):
        """
        Convert map objects to a plain, picklable representation.

        Replaces the open3d point cloud/bbox and CLIP/text feature tensors with
        numpy arrays ('points', 'bbox_np', 'pcd_color_np') and drops the
        original 'pcd'/'bbox' entries.

        Returns:
            list[dict]: Serializable copies of the objects.
        """
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)

            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            if 'text_ft' in s_obj_dict:
                s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])

            s_obj_dict['points'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']

            s_obj_list.append(s_obj_dict)

        return s_obj_list

    def load_serializable(self, s_obj_list):
        """
        Populate this (empty) MapObjectList from a serialized object list,
        reconstructing point clouds, oriented bounding boxes, and feature
        tensors from their numpy representations.

        Args:
            s_obj_list (list[dict]): Serialized objects as produced by to_serializable().

        Raises:
            AssertionError: If this list is not already empty.
        """
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)

            if 'clip_ft' in new_obj:
                new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            if 'text_ft' in new_obj:
                new_obj['text_ft'] = to_tensor(new_obj['text_ft'])

            new_obj['pcd'] = o3d.geometry.PointCloud()
            if 'points' in new_obj:
                new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['points'])
                new_obj['points'] = np.asarray(new_obj['points'])
            if 'pcd_color_np' in new_obj:
                new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])

            if 'bbox_np' in new_obj:
                try:
                    if new_obj['bbox_np'].sum() != 0:
                        new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(new_obj['bbox_np']))
                        new_obj['bbox'].color = new_obj['pcd_color_np'][0]
                except Exception as e:
                    loguru.logger.error(f"Error occurred while creating oriented bounding box: {e}")

            # del new_obj['points']
            # del new_obj['bbox_np']
            # del new_obj['pcd_color_np']

            self.append(new_obj)


class Hierarchy:
    """Two-level object hierarchy (parent objects and their child parts) with
    edges linking children to parents, plus grasp storage and pickle-based I/O."""

    def __init__(self, cfg, objects: MapObjectList = None):
        """
        Args:
            cfg: Configuration object.
            objects (MapObjectList, optional): Initial parent objects; a new
                empty MapObjectList is created if None.
        """
        self.cfg = cfg

        self.objects = MapObjectList() if objects is None else objects
        self.children = MapObjectList()
        self.grasps = []

        self.edges = []

    def __len__(self):
        """Return the number of parent objects."""
        return len(self.objects)

    def __getitem__(self, idx):
        """Return the parent object at the given index."""
        return self.objects[idx]

    def add_child(self, child_obj, parent_id: int):
        """
        Add a child object under a parent object, recording the parent-child edge.

        Args:
            child_obj: Child object dict to append to self.children.
            parent_id (int): Index of the parent in self.objects.
        """
        self.children.append(child_obj)
        self.edges.append((parent_id, len(self.children) - 1))
        if "children_idcs" not in self.objects[parent_id]:
            self.objects[parent_id]["children_idcs"] = []
        self.objects[parent_id]["children_idcs"].append(len(self.children) - 1)

    def get_children(self, parent_id: int = None) -> MapObjectList:
        """
        Get all children, or only those belonging to a given parent.

        Args:
            parent_id (int, optional): Index of the parent in self.objects.
                If None, all children are returned.

        Returns:
            MapObjectList: The requested children.
        """
        if parent_id is None:
            return self.children
        child_idcs = self.objects[parent_id]["children_idcs"]
        return self.children.slice_by_indices(child_idcs)

    def save(self, filepath):
        """Pickle the hierarchy's objects, children, and edges to filepath."""
        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump({"objects": self.objects.to_serializable(), "children": self.children.to_serializable(), "edges": self.edges}, f)

    def load(self, filepath):
        """Load objects and edges from a pickle file written by save() (children are not restored)."""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.objects.load_serializable(data["objects"])
            # self.children.load_serializable(data["children"])
            self.edges = data["edges"]


def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float = 0.025):
    """
    Prepare a lightweight, serializable copy of the map objects for saving/visualization.

    Deep-copies the objects, voxel-downsamples each point cloud, strips all
    fields except a fixed visualization-relevant subset, and serializes them.

    Args:
        objects (MapObjectList): Map objects to prepare.
        downsample_size (float): Voxel size used to downsample each point cloud.

    Returns:
        list[dict]: Serializable, downsampled copies of the objects.
    """
    objects_to_save = copy.deepcopy(objects)

    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in ['pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color', 'conf']:
                del objects_to_save[i][k]

    return objects_to_save.to_serializable()


def resize_gobs(gobs, image):
    """
    Resize grounded-observation masks and boxes to match a target image's resolution.

    Args:
        gobs (dict): Grounded observations with 'xyxy' and 'mask' entries.
        image (np.ndarray): Target image whose shape the masks/boxes are resized to.

    Returns:
        dict: gobs with 'mask' (and corresponding 'xyxy') resized in place.
    """
    n_masks = len(gobs['xyxy'])
    new_mask = []

    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]

            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)

    return gobs


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.

    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


def iou_aabb(box1: o3d.geometry.AxisAlignedBoundingBox, box2: o3d.geometry.AxisAlignedBoundingBox) -> float:
    """Compute the intersection-over-union of two axis-aligned bounding boxes.

    Returns:
        IoU in [0, 1]; 0.0 if the boxes don't overlap.
    """
    # Get min and max corners
    min1, max1 = box1.min_bound, box1.max_bound
    min2, max2 = box2.min_bound, box2.max_bound

    # Intersection box
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Check for no overlap
    if np.any(inter_max <= inter_min):
        return 0.0

    # Volumes
    inter_vol = np.prod(inter_max - inter_min)
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    return inter_vol / (vol1 + vol2 - inter_vol)


def compute_semantic_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects

    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft')  # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft')  # (N, D)

    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)

    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)

    return visual_sim


def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects

    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    return compute_iou_batch(det_bboxes, obj_bboxes)


def aggregate_similarities(cfg, spatial_sim: torch.Tensor, semantic_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score

    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if cfg.mapping.match_method == "sim_sum":
        sims = (1 + cfg.mapping.phys_bias) * spatial_sim + (1 - cfg.mapping.phys_bias) * semantic_sim  # (M, N)
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    return sims


def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']

    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in [
            'pcd',
            'bbox',
            'clip_ft',
            "text_ft",
            "mean_depth",
            "points",
            "color",
            "last_observed",
            "conf",
            "isolated",
        ]:  # if either is not isolated, the merged one is not isolated
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k in ["inst_color"]:
                obj1[k] = obj1[k]  # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else:  # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd, bbox, mean inverse depth
    obj1["isolated"] = obj1["isolated"]
    obj1['pcd'] += obj2['pcd']
    obj1['points'] = np.asarray(obj1['pcd'].points)
    obj1['color'] = np.asarray(obj1['pcd'].colors)
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg.mapping)
    obj1['bbox'] = get_bounding_box(cfg.mapping, obj1['pcd'])
    obj1['bbox'].color = [0, 1, 0]
    obj1['mean_depth'] = np.mean([obj1['mean_depth'], obj2['mean_depth']])
    obj1['conf'] = np.mean([obj1['conf'], obj2['conf']])
    obj1['last_observed'] = max(obj1['last_observed'], obj2['last_observed'])  # last occurence

    # # merge clip ft
    if "clip_ft" in obj1:
        obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det + obj2['clip_ft'] * n_obj2_det) / (n_obj1_det + n_obj2_det)
        obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    return obj1


def merge_detections_to_objects(cfg, detection_list: DetectionList, objects: MapObjectList, agg_sim: torch.Tensor) -> MapObjectList:
    """
    Merge each detection into the map, either as a new object or by fusing it
    into its best-matching existing object based on aggregated similarity scores.

    Isolated objects/detections are handled specially: a match is only merged
    if the aggregated similarity clears a stricter threshold, or the pair has
    a strong point-overlap.

    Args:
        cfg: Configuration object.
        detection_list (DetectionList): New detections for the current frame.
        objects (MapObjectList): Existing map objects (updated in place).
        agg_sim (torch.Tensor): (M, N) aggregated similarity between detections and objects.

    Returns:
        MapObjectList: The updated objects list.
    """
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            score = agg_sim[i, j]
            matched_det = detection_list[i]
            matched_obj = objects[j]

            # viz_pcd = o3d.geometry.PointCloud()
            # det_copy = copy.deepcopy(matched_det)
            # det_copy['pcd'].paint_uniform_color([1,0,0])
            # det_copy['pcd'] = det_copy['pcd'].voxel_down_sample(cfg.mapping.downsample_size)
            # obj_copy = copy.deepcopy(matched_obj)
            # obj_copy['pcd'].paint_uniform_color([0,1,0])
            # obj_copy['pcd'] = obj_copy['pcd'].voxel_down_sample(cfg.mapping.downsample_size)
            # viz_pcd += det_copy["pcd"]
            # viz_pcd += obj_copy["pcd"]
            # o3d.io.write_point_cloud("debug_merge.ply", viz_pcd)
            # CASE: object isolated
            if matched_obj["isolated"]:
                if score >= cfg.mapping.spatial_thresh_isolated:
                    merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=True)
                    objects[j] = merged_obj
                else:
                    # check if detection is a near strong inlier of the isolated object
                    overlap = compute_overlap(cfg, matched_det["pcd"], matched_obj["pcd"])
                    if overlap > cfg.mapping.merge_overlap_thresh:
                        merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=True)
                        objects[j] = merged_obj
            else:
                # lets investigate matched non-isolated objects
                if matched_det["isolated"]:
                    # CASE: object non-isolated & detection isolated
                    # compute whether the non-isolated object is a near strong inlier of the isolated detection
                    overlap = compute_overlap(cfg, matched_obj["pcd"], matched_det["pcd"])
                    if overlap > cfg.mapping.merge_overlap_thresh:
                        merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=True)
                        objects[j] = merged_obj
                        objects[j]["isolated"] = True  # check if this actually helps
                else:
                    # CASE: object non-isolated & detection non-isolated
                    merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=True)
                    objects[j] = merged_obj

    return objects


def denoise_objects(cfg, objects: MapObjectList):
    """
    Denoise each object's point cloud (voxel downsample + DBSCAN) and recompute
    its bounding box, keeping the original point cloud if denoising leaves too
    few points.

    Args:
        cfg: Configuration object.
        objects (MapObjectList): Objects to denoise (updated in place).

    Returns:
        MapObjectList: The denoised objects.
    """
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg.mapping)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg.mapping, objects[i]['pcd'])
        objects[i]['bbox'].color = [0, 1, 0]

    return objects


def filter_objects(cfg, objects: MapObjectList, keyframe_idcs: List[int] = None, curr_frame_idx: int = None):
    """
    Drop objects with too few points or too few detections, unless they were
    recently observed within cfg.mapping.filter_interval keyframes.

    Args:
        cfg: Configuration object.
        objects (MapObjectList): Objects to filter.
        keyframe_idcs (List[int], optional): Ordered keyframe indices, used to
            check the recency of an object's last observation.
        curr_frame_idx (int, optional): Index of the current frame.

    Returns:
        MapObjectList: The filtered objects.
    """
    # Remove the object that has very few points or viewed too few times
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg.mapping.obj_min_points:
            if obj['num_detections'] >= cfg.mapping.obj_min_detections:
                objects_to_keep.append(obj)
            else:
                if keyframe_idcs is None or curr_frame_idx is None:
                    continue
                # check how many frames have passed since the last observation
                if keyframe_idcs.index(curr_frame_idx) - keyframe_idcs.index(obj["last_observed"]) <= cfg.mapping.filter_interval:
                    objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))

    return objects


def merge_objects(cfg, objects: MapObjectList):
    """
    Merge objects whose pairwise overlap exceeds cfg.mapping.merge_overlap_thresh.

    Args:
        cfg: Configuration object.
        objects (MapObjectList): Objects to merge.

    Returns:
        MapObjectList: The merged objects.
    """
    if cfg.mapping.merge_overlap_thresh > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))

    return objects


def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    """
    Greedily merge pairs of objects whose symmetric overlap ratio exceeds the
    configured threshold, processing pairs in descending order of overlap.

    Args:
        cfg: Configuration object.
        objects (MapObjectList): Objects to merge.
        overlap_matrix (np.ndarray): (N, N) pairwise overlap ratios, as produced
            by compute_overlap_matrix.

    Returns:
        MapObjectList: Objects remaining after merging (merged-away objects removed).
    """
    # symmetricize the overlap matrix
    overlap_matrix_sym = np.minimum(overlap_matrix, overlap_matrix.T)

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        merge = False
        min_ratio = min(overlap_matrix_sym[i, j], overlap_matrix_sym[j, i])
        if min_ratio > cfg.mapping.merge_overlap_thresh and kept_objects[j]:
            objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
            kept_objects[i] = False
            # if ratio > 0.7:
            #     if objects[j]['isolated'] == True:
            #         # Use a higher threshold for isolated objects
            #         min_ratio = min(overlap_matrix_sym[i, j], overlap_matrix_sym[j, i])
            #         if min_ratio > cfg.mapping.merge_overlap_thresh:
            #             merge = True
            #     elif objects[j]['isolated'] == False:
            #         if ratio > 0.7:
            #             merge = True
            #     if merge and kept_objects[j]:
            #         # Then merge object i into object j
            #         objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
            #         kept_objects[i] = False
        else:
            break

    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects


def nms(scores, iou_matrix, iou_thresh=0.3):
    """
    Non-Maximum Suppression for Oriented Bounding Boxes.

    boxes:  Nx5 array  (cx, cy, w, h, angle_deg)
    scores: Nx1 array
    """

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remaining = []
        for j in order[1:]:
            iou = iou_matrix[i, j]
            if iou < iou_thresh:
                remaining.append(j)

        order = np.array(remaining)

    return keep


def compute_overlap_matrix(cfg, objects: Hierarchy):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor.
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object.
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i
    that are within a distance threshold of any point in point cloud j.
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    bboxes = objects.get_stacked_values_torch("bbox")
    ious = compute_iou_batch(bboxes, bboxes)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                # box_i = objects[i]['bbox']
                # box_j = objects[j]['bbox']

                # Skip if the boxes do not overlap at all (saves computation)
                # iou = compute_3d_iou(box_i, box_j)
                if ious[i, j] < 0.1:
                    continue

                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < cfg.mapping.overlap_dist_thresh**2).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix


def compute_overlap(cfg, obj1: o3d.geometry.PointCloud, obj2: o3d.geometry.PointCloud) -> float:
    '''
    compute overlapping between two objects in terms of point nearest neighbor.
    Suppose we have two point clouds, each of which is a o3d.geometry.PointCloud object.
    Now we want to compute the ratio of points in point cloud 1 that are within a distance threshold of any point in point cloud 2.
    '''
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points1 = np.asarray(obj1.points, dtype=np.float32)
    points2 = np.asarray(obj2.points, dtype=np.float32)

    index2 = faiss.IndexFlatL2(points2.shape[1])
    index2.add(points2)

    # Use range_search to find points within the threshold
    D, I = index2.search(points1, 1)

    # Calculate the ratio of points within the threshold
    overlap = (D < cfg.mapping.overlap_dist_thresh**2).sum()  # D is the squared distance
    overlap_ratio = overlap / len(points1)

    return overlap_ratio


def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    """
    Compute the 3D IoU (or maximum overlap ratio) between two axis-aligned bounding boxes.

    Args:
        bbox1: Open3D-style bounding box exposing get_min_bound()/get_max_bound().
        bbox2: Open3D-style bounding box exposing get_min_bound()/get_max_bound().
        padding: Amount to expand each box's bounds by before computing overlap.
        use_iou: If True, return the intersection-over-union; otherwise return the
            maximum of the two per-box overlap ratios.

    Returns:
        float: IoU (or max overlap ratio) between the two boxes.
    """
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap


def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    """
    Back-project a masked region of a depth image into a colored 3D point cloud.

    Args:
        depth_array (np.ndarray): (H, W) depth image in millimeters.
        mask (np.ndarray): (H, W) boolean mask selecting the object's pixels.
        cam_K (np.ndarray): (3, 3) camera intrinsic matrix.
        image (np.ndarray): (H, W, 3) RGB image used for per-point coloring.
        obj_color (optional): If given, used as a uniform color for all points
            instead of sampling from image.

    Returns:
        tuple: (pcd, mean_depth) where pcd is an open3d.geometry.PointCloud
            (voxel-downsampled to 0.01 m) and mean_depth is the mean masked
            depth in meters (0.0 if the mask is empty).
    """
    fx, fy, cx, cy = to_scalar(cam_K[0, 0]), to_scalar(cam_K[1, 1]), to_scalar(cam_K[0, 2]), to_scalar(cam_K[1, 2])

    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)
    depth_array = depth_array / 1000.0  # Convert depth from mm to meters

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd, 0.0

    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)

    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask]  # (N, )
    u = u[mask]  # (N, )
    v = v[mask]  # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    mean_depth = np.mean(masked_depth) if masked_depth.size > 0 else 0.0

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)

    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None:  # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else:  # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)

    if points.shape[0] == 0:
        import pdb

        pdb.set_trace()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # downsample the point cloud to 0.01
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    return pcd, mean_depth


def process_pcd(pcd, cfg):
    """
    Voxel-downsample a point cloud and optionally denoise it via DBSCAN.

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud to process.
        cfg: Configuration with 'downsample_size', 'dbscan', 'eps', and 'min_samples'.

    Returns:
        open3d.geometry.PointCloud: The processed point cloud.
    """

    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_size)

    if cfg.dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(pcd, eps=cfg.eps, min_points=cfg.min_samples)
        # print("After dbscan:", len(pcd.points))

    return pcd


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    """
    Denoise a point cloud by keeping only its largest DBSCAN cluster.

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud to denoise.
        eps (float): DBSCAN neighborhood radius.
        min_points (int): Minimum number of points to form a DBSCAN core point.

    Returns:
        open3d.geometry.PointCloud: The largest cluster (or the original point
            cloud if it has fewer than 5 points or no clusters are found).
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


def get_bounding_box(cfg, pcd):
    """
    Compute a bounding box for a point cloud, preferring a robust oriented
    bounding box when the spatial similarity type calls for it and enough
    points are available, falling back to an axis-aligned box otherwise.

    Args:
        cfg: Configuration with a 'spatial_sim_type' string.
        pcd (open3d.geometry.PointCloud): Point cloud to bound.

    Returns:
        open3d.geometry.OrientedBoundingBox or open3d.geometry.AxisAlignedBoundingBox.
    """
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()


def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    """
    Compute CLIP image and text embeddings for each detection's padded image crop.

    Args:
        image (np.ndarray): Full RGB image.
        detections: Detections with 'xyxy' boxes and 'class_id' per detection.
        clip_model: CLIP model exposing encode_image/encode_text.
        clip_preprocess: CLIP image preprocessing transform.
        clip_tokenizer: CLIP text tokenizer.
        classes (list[str]): Class-id-to-name lookup used to build text prompts.
        device: Unused torch device (inference runs on 'cuda' directly).

    Returns:
        tuple: (image_crops, image_feats, text_feats) - list of PIL crops,
            (N, D) normalized image features, and (N, D) normalized text features.
    """
    backup_image = image.copy()

    image = Image.fromarray(image)

    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    image_feats = []
    text_feats = []

    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        # Get the preprocessed image for clip from the crop
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)

        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)

    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats


def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, device):
    """
    Compute CLIP image embeddings for all detections in a single batched forward pass.

    Args:
        image (np.ndarray): Full RGB image.
        detections (dict): Detections with an 'xyxy' entry in (x, y, w, h) format.
        clip_model: CLIP model exposing encode_image.
        clip_preprocess: CLIP image preprocessing transform.
        clip_tokenizer: Unused (kept for API symmetry with compute_clip_features).
        device: Torch device to run inference on.

    Returns:
        tuple: (image_crops, image_feats) - list of PIL crops and an (N, D)
            normalized image-feature numpy array.
    """

    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    preprocessed_images = []

    # Prepare data for batch processing
    for idx in range(len(detections["xyxy"])):
        x_min, y_min, x_max, y_max = detections["xyxy"][idx]

        # Convert from xyhw format to xyxy format
        x_max += x_min
        y_max += y_min

        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        image_crops.append(cropped_image)

    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)

    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # image_feats = []

    return image_crops, image_feats


from ortools.sat.python import cp_model


def solve_assignment(candidates, cost, overlap, lambda_overlap=1.0):
    """
    Solve a min-cost assignment of articulations to candidate objects with a
    CP-SAT model, penalizing pairs of assigned objects that overlap.

    Each articulation is assigned exactly one candidate object, each object is
    used by at most one articulation, and a penalty is added whenever two
    objects that both appear in `overlap` are simultaneously selected.

    Args:
        candidates (dict): Mapping from articulation index to a list of
            candidate object indices.
        cost (dict): Mapping from (articulation_idx, object_idx) to assignment cost.
        overlap (dict): Mapping from (object_idx_a, object_idx_b) to an overlap penalty.
        lambda_overlap (float): Weight applied to the overlap penalty term.

    Returns:
        tuple: (assignment, objective_value) where assignment maps articulation
            index to its chosen object index.

    Raises:
        RuntimeError: If the solver does not find a feasible solution.
    """
    model = cp_model.CpModel()

    # Binary variables x[i,j]
    x = {}
    for i, objs in candidates.items():
        for j in objs:
            x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Each articulation gets exactly one object
    for i, objs in candidates.items():
        model.Add(sum(x[(i, j)] for j in objs) == 1)

    # Helper: y[j] = object j is used by any articulation
    objects = set(j for objs in candidates.values() for j in objs)
    y = {}
    for j in objects:
        y[j] = model.NewBoolVar(f"y_{j}")
        model.AddMaxEquality(y[j], [x[(i, j)] for i in candidates if j in candidates[i]])

    # Overlap activation variables
    z = {}
    for (j, k), pen in overlap.items():
        if j in y and k in y:
            z[(j, k)] = model.NewBoolVar(f"z_{j}_{k}")
            model.Add(z[(j, k)] <= y[j])
            model.Add(z[(j, k)] <= y[k])
            model.Add(z[(j, k)] >= y[j] + y[k] - 1)

    # Each object can be used by at most one articulation (exclusivity)
    objects = set(j for objs in candidates.values() for j in objs)

    for j in objects:
        model.Add(sum(x[(i, j)] for i in candidates if j in candidates[i]) <= 1)

    # Objective
    assignment_cost = sum(cost[(i, j)] * x[(i, j)] for (i, j) in x)

    overlap_cost = sum(lambda_overlap * overlap[(j, k)] * z[(j, k)] for (j, k) in z)

    model.Minimize(assignment_cost + overlap_cost)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No solution found")

    assignment = {i: j for (i, j), var in x.items() if solver.Value(var) == 1}

    return assignment, solver.ObjectiveValue()
