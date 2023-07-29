import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import hdf5plugin  # type: ignore
import PIL
import numpy as np
import torch  # type: ignore
import torchvision
import torchvision.transforms as tf  # type: ignore
import tqdm

import symbolic

from . import dnf_utils, video_utils


class Dataset:
    """Gridworld dataset for pytorch DataLoader.

    Data is stored as (action_call, img_pre, img_post) tuples.

    Args:
        pddl: Pddl instance.
        path: Path of dataset.
        dataset: Filename of dataset.
        split: Start and end range of dataset to load as a fraction.
        max_size: Maximum size of dataset.
        transform: Transform images to tensors.
        max_num_conjunctions: Max number of conjunctions in DNFs, or if None, compute from dataset.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        dataset: str = "dataset.hdf5",
        split: Tuple[float, float] = (0.0, 1.0),
        max_size: Optional[int] = None,
        transform: Callable[
            [Union[PIL.Image.Image, torch.Tensor]], torch.Tensor
        ] = torchvision.transforms.ToTensor(),
        max_num_conjunctions: Optional[int] = None,
    ):
        self._pddl = pddl
        self._path = pathlib.Path(path)
        self._dataset = dataset
        self._transform = transform

        self._N = len(pddl.state_index)

        self._idx_static = dnf_utils.get_static_props(self._pddl)

        # Compute dataset length
        with h5py.File(self.path / self.dataset, "r") as f:
            if "actions" in f:
                dset_actions = f["actions"]
                D = dset_actions.shape[0]

            else:
                D = 0
                for dset in f.values():
                    D = max(D, dset.shape[0])

        if max_num_conjunctions is None:
            print("Computing max number of conjunctions...")
            actions = set(tqdm.tqdm(dset_actions))
            max_num_conjunctions = dnf_utils.compute_max_num_conjunctions(
                self._pddl, actions
            )
        self._M = max_num_conjunctions

        self._idx_start = int(split[0] * D)
        idx_end = int(split[1] * D + 0.5)
        self._len = idx_end - self._idx_start
        if max_size is not None:
            self._len = min(self._len, max_size)

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev) from ImageNet."""
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    @property
    def dim_image(self) -> Tuple[int, int, int]:
        """Standardized image size."""
        return (3, 224, 224)

    def __len__(self) -> int:
        """Length of dataset."""
        return self._len

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """Get the data point at the specified index.

        * img: [2, *IMG_SIZE]
        * dnf: [2, 2, N, M]
        * mask: [2, 2, N, M]
        * s: [2, N]

        Returns:
            5-tuple (action, img, dnf, mask, s).
        """
        if idx < 0:
            idx += self._len

        assert idx >= 0 and idx <= self._len
        if idx == self._len:
            raise StopIteration()

        return self._get_data_point(idx + self._idx_start)

    @property
    def path(self) -> pathlib.Path:
        """Gridworld dataset path."""
        return self._path

    @property
    def pddl(self) -> symbolic.Pddl:
        """Pddl instance."""
        return self._pddl

    @property
    def dataset(self) -> str:
        """Gridworld dataset filename."""
        return self._dataset

    @property
    def transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
        """Transformation applied to images."""
        return self._transform

    @property
    def N(self) -> int:
        """Number of propositions in state vector."""
        return self._N

    @property
    def M(self) -> int:
        """Max number of conjunctions in pre/post-condition DNFs."""
        return self._M

    @property
    def collate_fn(self) -> Optional[Callable[[List], Any]]:
        """Collate function used to create batches with torch.DataLoader."""
        return None

    # @functools.lru_cache(maxsize=1024 * 1024)
    def _get_data_point(self, idx, raw=False):
        """Get a single data point from cache or disk.

        Args:
            idx (int): Index of data point.
        Returns:
            (str, torch.Tensor, np.ndarray, np.ndarray, np.ndarray):
            5-tuple (action, img, idx, mask, s)
        """
        # Get data point from disk
        with h5py.File(self.path / self.dataset, "r") as f:
            action = f["actions"][idx]
            img_pre = f["img_pre"][idx]
            img_post = f["img_post"][idx]
            s_pre = f["s_pre"][idx]
            s_post = f["s_post"][idx]

            if raw:
                dnf, mask = dnf_utils.get_dnf(self._pddl, action, self.M)
                dnf = dnf_utils.convert_to_partial_state(
                    self._pddl, dnf, mask, self._idx_static
                )
                key = f["keys"][idx]
                # TODO: Don't return mask.
                return (key, action, img_pre, img_post, s_pre, s_post, dnf, mask)

        # [2 x DIM[0] x DIM[1] x 3] array (pre/post, rgb image)
        img_pre = PIL.Image.fromarray(img_pre)
        img_post = PIL.Image.fromarray(img_post)
        img_pre = self.transform(img_pre)
        img_post = self.transform(img_post)
        img = torch.stack((img_pre, img_post), dim=0)

        # [2 x 2 x N x M] array (pre/post, pos/neg, N props, M conjs)
        dnf, mask = dnf_utils.get_dnf(self._pddl, action, self.M)
        s_partial = dnf_utils.convert_to_partial_state(
            self._pddl, dnf, mask, self._idx_static
        )

        # [2 x N] array (pre/post, N props)
        s = np.stack((s_pre, s_post), axis=0)
        s = s.astype(np.float32)

        # TODO: Don't return mask.
        return (action, img, s_partial, mask, s)

    def _len_split(self, keys, train, split=0.8):
        """Length of the dataset split.

        Args:
            keys (list): Keys to split.
            train (bool): Whether to use train split or val split.
            split (float, optional): Proportion of the dataset for the train split.
        Returns:
            (int): Length of split.
        """
        num_keys = len(keys)
        idx_split = int(split * num_keys)
        if train:
            return idx_split
        else:
            return num_keys - idx_split

    def _split_train(self, keys, train, split=0.8):
        """Split the dataset.

        The train split is the first 0.8 of the dataset, while the val split is
        the last 0.2.

        Args:
            keys (list): Keys to split.
            train (bool): Whether to use train split or val split.
            split (float, optional): Proportion of the dataset for the train split.
        Returns:
            (generator): Split keys.
        """
        num_keys = len(keys)
        idx_split = int(split * num_keys)
        if train:
            # Iterate over first split
            for i, key in enumerate(keys[:idx_split]):
                yield key
        else:
            # Iterate over second split
            for i, key in enumerate(keys[idx_split:]):
                yield key


def count_prop_occurrences(pddl: symbolic.Pddl, dataset: Dataset) -> np.ndarray:
    """Counts co-occurrences of propositions among all the DNFs.

    Args:
        pddl: Pddl instance.
        dataset: Dataset indexed with (_, ..., s) tuple.
    Returns:
        num_xy [N, N].
    """
    N = len(pddl.state_index)

    num_xy = np.zeros((N, N, 4), dtype=int)
    for i in tqdm.tqdm(range(len(dataset))):
        # TODO: Assumes state is last index of data tuple.
        s = dataset[i][-1].astype(bool)

        # [N, N]
        num_xy[:, :, 0] += s[:, None] & s[None, :]
        num_xy[:, :, 1] += s[:, None] & ~s[None, :]
        num_xy[:, :, 2] += ~s[:, None] & s[None, :]
        num_xy[:, :, 3] += ~s[:, None] & ~s[None, :]

    return num_xy


class GridworldPredicateDataset(Dataset):
    """Gridworld dataset for pytorch DataLoader.

    Args:
        pddl: Pddl instance.
        path: Path of dataset.
        dataset: Name of dataset file.
        split: Start and end range of dataset to load as a fraction.
        ground_truth: Whether to use full ground truth state label or dnf label.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        dataset: str = "dataset.hdf5",
        split: Tuple[float, float] = (0.0, 0.8),
        ground_truth: bool = False,
    ):
        super().__init__(pddl, path, dataset, split=split, max_num_conjunctions=8)
        self._ground_truth = ground_truth

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev)."""
        return (0.7170, 0.7143, 0.7097), (0.2719, 0.2740, 0.2784)

    @property
    def dim_image(self) -> Tuple[int, int, int]:
        """Standard image size."""
        return (3, 220, 220)

    def _get_data_from_disk(
        self, idx: int
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Get data point from disk.

        Args:
            idx: Index of data point.
        Returns:
            4-tuple (
                action,
                images [2, H, W, 3] (pre/post, H, W, rgb) uint8,
                s [2, N] (pre/post, num_props),
                boxes [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32,
            ).
        """
        # Get data point from disk
        with h5py.File(self.path / self.dataset, "r") as f:
            action = f["actions"][idx].decode("utf-8")

            # [2, H, W, 3] (pre/post, H, W, rgb)
            _, H, W = self.dim_image
            images = np.empty((2, H, W, 3), dtype=np.uint8)

            # [H, W, 3]
            images[0] = f["img_pre"][idx]
            images[1] = f["img_post"][idx]

            # [2, N] (pre/post, num_props)
            N = len(self.pddl.state_index)
            s = np.empty((2, N), dtype=bool)

            # [N]
            s[0] = f["s_pre"][idx]
            s[1] = f["s_post"][idx]

            # [2, 9, 4] (pre/post, num_obj, x1/y1/x2/y2)
            NUM_OBJECTS = len(self.pddl.objects)
            boxes = np.empty((2, NUM_OBJECTS, 4), dtype=np.float32)

            # [9, 4]
            boxes[0] = f["boxes_pre"][idx]
            boxes[1] = f["boxes_post"][idx]

            return action, images, s, boxes

    def _create_sp_partial(
        self,
        boxes: np.ndarray,
        action: Optional[str] = None,
        s: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a predicate partial state for the given action, or if the
        full ground truth label is given, return the full state.

        Args:
            boxes: [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32.
            action: Action call. If not None, this function returns a partial state.
            s: [2, N] Ground truth state. If not None, this function returns a full state.
        Returns:
            2-tuple (
                sp_partial [U, 2, P] (pre_args/post_args, pos/neg, num_pred) bool array,
                idx_used [2, MC] (pre/post, arg_combos) bool array,
            ).
        """
        if action is not None:
            # Get partial state.
            # [2, MC, 2, P] (pre/post, arg_combos, pos/neg, num_preds)
            sp_partial = dnf_utils.get_partial_predicate_state(self.pddl, action)
        elif s is not None:
            # Reshape full state.
            # [2, N] -> [2, 2, N] (pre/post, pos/neg, N)
            s_partial = np.stack((s, np.logical_not(s)), axis=1)

            # Filter out invalid states.
            # [2, N] (pre/post, num_props)
            idx_used_obj = np.stack(
                (
                    dnf_utils.get_used_props(self.pddl, boxes[0]),
                    dnf_utils.get_used_props(self.pddl, boxes[1]),
                ),
                axis=0,
            )

            # [2, 2, N] (pre/post, pos/neg, num_props)
            # s_partial &= (idx_valid & idx_used_obj)[:, None, :]
            s_partial &= idx_used_obj[:, None, :]

            # [2, 2, N] -> [2, 2, MC, P] (pre/post, pos/neg, arg_combos, num_props)
            sp_partial = dnf_utils.proposition_to_predicate_indices(
                self.pddl, s_partial
            )

            # [2, 2, MC, P] -> [2, MC, 2, P] (pre/post, arg_combos, pos/neg, num_props)
            sp_partial = np.swapaxes(sp_partial, 2, 1)
        else:
            raise ValueError("One of action or s must not be None.")

        # [2, MC, 2, P] -> [2 * MC, 2 * P]
        sp_flat = sp_partial.reshape((-1, sp_partial.shape[2] * sp_partial.shape[3]))

        # [2, MC, 2, P] -> [2 * MC, 2, P]
        sp_partial = sp_partial.reshape((-1, *sp_partial.shape[2:]))

        # [2 * MC, 2 * P] -> [2 * MC]
        idx_used = (sp_flat != 0).any(axis=1)

        # [U, 2, P] (pre_args/post_args, pos/neg, num_preds)
        sp_partial = sp_partial[idx_used, :, :]

        # [2 * MC] -> [2, MC] (pre/post, arg_combos)
        idx_used = idx_used.reshape((2, -1))

        return sp_partial, idx_used

    def _create_boxes(self, boxes: np.ndarray, idx_used: np.ndarray) -> np.ndarray:
        """Organizes boxes into indexed boxes.

        Args:
            boxes [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32 array,
            idx_used: [2, MC] (pre/post, arg_combos) bool array.
        Returns:
            [U, M + 1, 5] (pre_args/post_args, roi/max_args, idx/x1/y1/x2/y2) float32 array.
        """
        NUM_PRE, NUM_POST = idx_used.sum(axis=1)

        # Add NaN channel for selection by arg combo map.
        # [2, O + 1, 4] (pre/post, num_objects + 1, x1/y1/x2/y2)
        O = boxes.shape[1]  # noqa: E741
        param_boxes = np.full((2, O + 1, 4), np.nan, dtype=np.float32)
        param_boxes[:, :O, :] = boxes

        # [MC, M] (arg_combos, max_args)
        idx_param_to_arg_combo = dnf_utils.param_to_arg_combo_indices(self.pddl)

        # Get index array directly to used pre/post arg combos.
        # [MC, M] -> [U, M] (used_args, max_args)
        idx_param_to_pre = idx_param_to_arg_combo[idx_used[0], :]
        idx_param_to_post = idx_param_to_arg_combo[idx_used[1], :]

        # [U, M + 1, 5] (pre_args/post_args, roi/max_args, idx/x1/y1/x2/y2)
        M = idx_param_to_arg_combo.shape[1]
        indexed_boxes = np.empty((NUM_PRE + NUM_POST, M + 1, 5), np.float32)

        # [O + 1, 4] -> [U, M, 4] (used_args, max_args, x1/y1/x2/y2)
        indexed_boxes[:NUM_PRE, 1:, 1:] = param_boxes[0, idx_param_to_pre, :]
        indexed_boxes[NUM_PRE:, 1:, 1:] = param_boxes[1, idx_param_to_post, :]

        # [U, M, 2] -> [U, 2] (pre/post, x/y)
        indexed_boxes[:, 0, 1:3] = np.nanmin(indexed_boxes[:, 1:, 1:3], axis=1)  # xy1
        indexed_boxes[:, 0, 3:] = np.nanmax(indexed_boxes[:, 1:, 3:], axis=1)  # xy2

        # [U, M + 1]
        indexed_boxes[:, :, 0] = np.arange(NUM_PRE + NUM_POST)[:, None]

        # Convert nans to -inf.
        indexed_boxes[np.isnan(indexed_boxes)] = -float("inf")

        return indexed_boxes

    def _create_images(self, images: np.ndarray, idx_used: np.ndarray) -> np.ndarray:
        """Creates a 3-channel CNN image.

        Args:
            images: [2, H, W, 3] uint8 array.
            idx_used: [2, MC] (pre/post, arg_combos) bool array.
        Returns:
            [U, 3, H, W] (pre_args/post_args, rgb, H, W) float32 array.
        """
        NUM_PRE, NUM_POST = idx_used.sum(axis=1)

        images_cnn = np.zeros((NUM_PRE + NUM_POST, *self.dim_image), dtype=np.float32)

        # Normalize.
        # [H, W, 3] -> [U, 3, H, W]
        images_cnn[:NUM_PRE] = video_utils.image_rgb_to_cnn(
            images[0], self.image_distribution
        )[None, ...]
        images_cnn[NUM_PRE:] = video_utils.image_rgb_to_cnn(
            images[1], self.image_distribution
        )[None, ...]

        return images_cnn

    def _create_masks(
        self,
        indexed_boxes: np.ndarray,
        idx_used: np.ndarray,
    ) -> np.ndarray:
        """Creates a 3-channel bounding box mask.

        Args:
            indexed_boxes: [U, M + 1, 5] (pre_args/post_args, roi/max_args, idx/x1/y1/x2/y2) float32 array.
            idx_used: [2, MC] (pre/post, arg_combos) bool array.
        Returns:
            [U, M, H, W] (pre_args/post_args, max_args, H, W) float32 array.
        """
        _, H, W = self.dim_image
        U = indexed_boxes.shape[0]
        M = indexed_boxes.shape[1] - 1

        # masks = np.zeros((NUM_PRE + NUM_POST, M, H, W), dtype=np.float32)
        # [U, M] (pre_args/post_args, max_args)
        labels = np.zeros((U, M), dtype=np.float32)

        # [U] (pre_args/post_args)
        idx_arg_combos = np.concatenate(
            (idx_used[0].nonzero()[0], idx_used[1].nonzero()[0]), axis=0
        )
        for i, idx_arg_combo in enumerate(idx_arg_combos):
            args = dnf_utils.arg_combo(self.pddl, idx_arg_combo, indices=True)
            labels[i, : len(args)] = np.array(args) + 1

        labels /= len(self.pddl.objects)

        # [U, M, 4] -> [U, M, H, W] (used_arg_combos, max_args, H, W)
        masks = dnf_utils.bbox_masks(W, H, indexed_boxes[:, 1:, 1:], labels=labels)

        return masks

    def _get_data_point(  # type: ignore
        self, idx: int
    ) -> Tuple[
        str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Get a single data point from cache or disk.

        Args:
            idx: Index of data point.
        Returns:
            5-tuple (
                action,
                images [U, 3, H, W] (pre_args/post_args, rgb, H, W) float32 array,
                masks [U, M, H, W] (pre_args/post_args, max_args, H, W) float32 array,
                boxes [U, 1 + M, 5] (pre_args/post_args, roi/max_args, idx/x1/y1/x2/y2),
                sp_partial [U, 2, P] (pre_args/post_args, pos/neg, num_pred),
                idx_used [2, MC] (pre/post, arg_combos),
                boxes [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32,
            ).
        """
        action, images, s, boxes = self._get_data_from_disk(idx)

        if self._ground_truth:
            sp_partial, idx_used = self._create_sp_partial(boxes, s=s)
        else:
            sp_partial, idx_used = self._create_sp_partial(boxes, action=action)

        indexed_boxes = self._create_boxes(boxes, idx_used)
        images = self._create_images(images, idx_used)
        masks = self._create_masks(indexed_boxes, idx_used)

        return action, images, masks, indexed_boxes, sp_partial, idx_used, boxes

    @property
    def collate_fn(self) -> Optional[Callable[[List], Any]]:
        """Collate function used to create batches with torch.DataLoader."""
        return self._collate_fn

    @staticmethod
    def _collate_fn(
        data: List[
            Tuple[
                str,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batches data by concatenating along the variable-length 0-axis.

        Args:
            data: List of data points.
        Returns:
            Batched data (
                images [-1, 3, H, W] (pre_args/post_args/..., rgb, H, W),
                masks [-1, M, H, W] (pre_args/post_args/..., max_args, H, W),
                indexed_boxes [-1, 1 + M, 5] (pre_args/post_args/..., roi/max_args, idx/x1/y1/x2/y2),
                sp_partial [-1, 2, P] (pre_args/post_args/..., pos/neg, num_preds),
            ).
        """
        actions, images, masks, indexed_boxes, sp_partial, idx_used, boxes = zip(*data)

        # [-1, 5, H, W] (pre_args/post_args/..., rgb, H, W)
        images = np.concatenate(images, axis=0)

        # [-1, 5, H, W] (pre_args/post_args/..., max_args, H, W)
        masks = np.concatenate(masks, axis=0)

        # [-1, 3, 5] (pre_args/post_args/..., roi/a/b, idx/x1/y1/x2/y2),
        indexed_boxes = np.concatenate(indexed_boxes, axis=0)
        indexed_boxes[:, :, 0] = np.arange(images.shape[0], dtype=np.float32)[:, None]

        # [-1, 2, P] (pre_args/post_args/..., pos/neg, num_preds),
        sp_partial = np.concatenate(sp_partial, axis=0)

        images = torch.from_numpy(images)
        masks = torch.from_numpy(masks)
        indexed_boxes = torch.from_numpy(indexed_boxes)
        sp_partial = torch.from_numpy(sp_partial)

        return images, masks, indexed_boxes, sp_partial


class GridworldHalfPredicateDataset(GridworldPredicateDataset):
    """Gridworld dataset for pytorch DataLoader.

    Args:
        pddl: Pddl instance.
        path: Path of dataset.
        dataset: Name of dataset file.
        split: Start and end range of dataset to load as a fraction.
        ground_truth: Whether to use full ground truth state label or dnf label.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        dataset: str = "dataset.hdf5",
        split: Tuple[float, float] = (0.0, 0.8),
        ground_truth: bool = False,
    ):
        super().__init__(pddl, path, dataset, split, ground_truth)
        half_dataset = f"{dataset.split('.')[0]}_half.hdf5"
        with h5py.File(pathlib.Path(path) / half_dataset, "r") as f:
            self._idx_pre_post = np.array(f["idx_pre_post"])

    def _create_sp_partial(  # type: ignore
        self,
        idx_pre_post: int,
        boxes: np.ndarray,
        action: Optional[str] = None,
        s: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a predicate partial state for the given action, or if the
        full ground truth label is given, return the full state.

        Args:
            idx_pre_post: Whether to extract pre or post image from current data point.
            boxes: [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32.
            action: Action call. If not None, this function returns a partial state.
            s: [2, N] Ground truth state. If not None, this function returns a full state.
        Returns:
            2-tuple (
                sp_partial [U, 2, P] (pre_args/post_args, pos/neg, num_pred) bool array,
                idx_used [2, MC] (pre/post, arg_combos) bool array,
            ).
        """
        if action is not None:
            # Get partial state.
            # [2, MC, 2, P] (pre/post, arg_combos, pos/neg, num_preds)
            sp_partial = dnf_utils.get_partial_predicate_state(self.pddl, action).copy()
        elif s is not None:
            # Reshape full state.
            # [2, N] -> [2, 2, N] (pre/post, pos/neg, N)
            s_partial = np.stack((s, np.logical_not(s)), axis=1)

            # Filter out invalid states.
            # [2, N] (pre/post, num_props)
            idx_used_obj = np.stack(
                (
                    dnf_utils.get_used_props(self.pddl, boxes[0]),
                    dnf_utils.get_used_props(self.pddl, boxes[1]),
                ),
                axis=0,
            )

            # [2, 2, N] (pre/post, pos/neg, num_props)
            # s_partial &= (idx_valid & idx_used_obj)[:, None, :]
            s_partial &= idx_used_obj[:, None, :]

            # [2, 2, N] -> [2, 2, MC, P] (pre/post, pos/neg, arg_combos, num_props)
            sp_partial = dnf_utils.proposition_to_predicate_indices(
                self.pddl, s_partial
            )

            # [2, 2, MC, P] -> [2, MC, 2, P] (pre/post, arg_combos, pos/neg, num_props)
            sp_partial = np.swapaxes(sp_partial, 2, 1)
        else:
            raise ValueError("One of action or s must not be None.")

        # Zero out unused pre/post image.
        sp_partial[1 - idx_pre_post] = 0

        # [2, MC, 2, P] -> [2 * MC, 2 * P]
        sp_flat = sp_partial.reshape((-1, sp_partial.shape[2] * sp_partial.shape[3]))

        # [2, MC, 2, P] -> [2 * MC, 2, P]
        sp_partial = sp_partial.reshape((-1, *sp_partial.shape[2:]))

        # [2 * MC, 2 * P] -> [2 * MC]
        idx_used = (sp_flat != 0).any(axis=1)

        # [U, 2, P] (pre_args/post_args, pos/neg, num_preds)
        sp_partial = sp_partial[idx_used, :, :]

        # [2 * MC] -> [2, MC] (pre/post, arg_combos)
        idx_used = idx_used.reshape((2, -1))

        return sp_partial, idx_used

    def _get_data_point(  # type: ignore
        self, idx: int
    ) -> Tuple[
        str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Get a single data point from cache or disk.

        Args:
            idx: Index of data point.
        Returns:
            5-tuple (
                action,
                images [U, 3, H, W] (pre_args/post_args, rgb, H, W) float32 array,
                masks [U, M, H, W] (pre_args/post_args, max_args, H, W) float32 array,
                boxes [U, 1 + M, 5] (pre_args/post_args, roi/max_args, idx/x1/y1/x2/y2),
                sp_partial [U, 2, P] (pre_args/post_args, pos/neg, num_pred),
                idx_used [2, MC] (pre/post, arg_combos),
                boxes [2, O, 4] (pre/post, num_objects, x1/y1/x2/y2) float32,
            ).
        """
        action, images, s, boxes = self._get_data_from_disk(idx)

        idx_pre_post = self._idx_pre_post[idx]
        if self._ground_truth:
            sp_partial, idx_used = self._create_sp_partial(idx_pre_post, boxes, s=s)
        else:
            sp_partial, idx_used = self._create_sp_partial(
                idx_pre_post, boxes, action=action
            )

        indexed_boxes = self._create_boxes(boxes, idx_used)
        images = self._create_images(images, idx_used)
        masks = self._create_masks(indexed_boxes, idx_used)

        return action, images, masks, indexed_boxes, sp_partial, idx_used, boxes


def create_canvas_resize_transform(
    size_canvas: Tuple[int, int]
) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Args:
        size_canvas: Size of cropped image (H, W). Use maximum size computed from
        dataset.
    """

    def resize(image: np.ndarray) -> torch.Tensor:
        """Resizes the image.

        Args:
            image: [..., C, H_canvas, W] uint8 array where the first 3 channels are rgb.
        Returns:
            [..., C, H_canvas, W_canvas]"""
        image_new = np.zeros((*image.shape[:-2], *size_canvas), dtype=np.float32)
        image_new[..., :3, :, : image.shape[-1]] = (
            image[..., :3, :, :].astype(np.float32) / 255
        )
        image_new[..., 3:, :, : image.shape[-1]] = image[..., 3:, :, :].astype(
            np.float32
        )
        return torch.from_numpy(image_new)

    return resize


def create_normalize_transform(
    img_mean: np.ndarray, img_stddev: np.ndarray
) -> Callable[[torch.Tensor], torch.Tensor]:
    tf_normalize = tf.Normalize(img_mean, img_stddev, inplace=True)

    def normalize(image: torch.Tensor) -> torch.Tensor:
        tf_normalize(image[..., :3, :, :])
        return image

    return normalize


class TwentyBnDataset(Dataset):
    """20BN dataset for pytorch DataLoader."""

    def __init__(
        self,
        pddl: symbolic.Pddl,
        path: Union[pathlib.Path, str] = "../data/twentybn",
        dataset="dataset.hdf5",
    ):
        size_img = self.dim_image[1:]
        img_mean, img_stddev = self.image_distribution

        transform = tf.Compose(
            [
                create_canvas_resize_transform(size_img),
                create_normalize_transform(np.array(img_mean), np.array(img_stddev)),
            ]
        )
        super().__init__(
            pddl, path, dataset, transform=transform, max_num_conjunctions=2
        )

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev) from ImageNet."""
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    @property
    def dim_image(self) -> Tuple[int, int, int]:
        """Standardized image size.

        Images in 20BN have a height of 240 with variable widths. The maximum is 494.
        """
        return (7, 240, 494)

    def _get_data_point(  # type: ignore
        self, idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single data point from cache or disk.

        Args:
            idx: Index of data point.
        Returns:
            (id_video, images, boxes, s_partial)
        """
        raise NotImplementedError("Proposition datset.")
        # Get data point from disk.
        with h5py.File(self.path / self.dataset, "r") as f:
            id_video = str(f["videos"][idx])
            id_action = f["actions"][idx]

            # Get data from linked dataset.
            if "link" in f:
                link = f["link"][0].decode("utf8")
                with h5py.File(self.path / link, "r") as ff:
                    # [2, 7, H, W] (pre/post, rgb/hand/a/b/c, H, W)
                    images = np.array(ff[id_video]["images"])

                    # [2, 5, 4] (pre/post, image/hand/a/b/c, x1/y1/x2/y2)
                    boxes = np.array(ff[id_video]["boxes"])
            else:
                # [2, 7, H, W] (pre/post, rgb/hand/a/b/c, H, W)
                images = np.array(f[id_video]["images"])

                # [2, 5, 4] (pre/post, image/hand/a/b/c, x1/y1/x2/y2)
                boxes = np.array(f[id_video]["boxes"])

        action = str(self.pddl.actions[id_action])

        # [2, 2, N] (pre/post, pos/neg, N)
        s_partial = dnf_utils.get_partial_state(self.pddl, action)

        # [7, H, W]
        img_pre = self.transform(images[0])
        img_post = self.transform(images[1])

        # [7, H, W] => [2, 7, H, W]
        images = torch.stack((img_pre, img_post), dim=0)

        boxes = torch.from_numpy(boxes)
        s_partial = torch.from_numpy(s_partial)

        return id_video, images, boxes, s_partial


class TwentyBnPredicateDataset(TwentyBnDataset):
    """20BN dataset for pytorch DataLoader.

    Args:
        pddl: Pddl instance.
        path: Path of dataset.
        dataset: Name of dataset file.
        shuffle: Whether to randomly select a pre/post frame from the available
            frames, or to deterministically select the first one.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        path: Union[pathlib.Path, str] = "../data/twentybn",
        dataset: str = "dataset.hdf5",
        shuffle: bool = True,
    ):
        super().__init__(pddl, path, dataset)
        self._shuffle = shuffle

    @property
    def dim_image(self) -> Tuple[int, int, int]:
        """Standardized image size.

        Images in 20BN have a height of 240 with variable widths. The maximum is 494.
        """
        return (3, 240, 494)

    def _get_data_from_disk(
        self, idx: int
    ) -> Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gets data point from disk.

        Args:
            idx: Index of data point.
        Returns:
            6-tuple (
                id_video,
                id_action,
                pre_image [H, W, 3],
                post_image [H, W, 3],
                pre_box [16, 3, 4] (arg_combos, roi/a/b, x1/y1/x2/y2),
                post_box [16, 3, 4] (arg_combos, roi/a/b, x1/y1/x2/y2),
            ).
        """

        def sample_weights(
            n: int, peak: float = 1 / 3, reverse: bool = False
        ) -> np.ndarray:
            """Generates a skewed distribution where the peak is at `n / peak`
            and p(n) = 0.

            Args:
                n: Number of elements.
                peak: Peak of distribution.
                reverse: Whether to reverse the distribution (across the x-axis).
            Returns:
                [n] array of weights summing to 1.
            """
            x = np.arange(0, n) if not reverse else np.arange(n - 1, -1, -1)

            y = np.cos((x - n * peak) * (np.pi / 2) / (n * (1 - peak)))
            y /= y.sum()

            return y

        # Get data point from disk.
        with h5py.File(self.path / self.dataset, "r") as f:

            id_video = str(f["videos"][idx])
            id_action = f["actions"][idx]

            dset = f[id_video]

            # Randomly select a pre and post frame.
            if self._shuffle:
                N_PRE = dset["pre_frames"].shape[0]
                N_POST = dset["post_frames"].shape[0]
                idx_pre = np.random.choice(N_PRE, p=sample_weights(N_PRE))
                idx_post = np.random.choice(
                    N_POST, p=sample_weights(N_POST, reverse=True)
                )
                # idx_pre = random.randrange(dset["pre_frames"].shape[0])
                # idx_post = random.randrange(dset["post_frames"].shape[0])
            else:
                idx_pre = 0
                idx_post = 0

            # [H, W, 3]
            pre_image = np.array(dset["pre_images"][idx_pre])
            post_image = np.array(dset["post_images"][idx_post])

            # [16, 3, 4] (arg_combos, roi/a/b, x1/y1/x2/y2)
            pre_box = np.array(dset["pre_boxes"][idx_pre])
            post_box = np.array(dset["post_boxes"][idx_post])

            return id_video, id_action, pre_image, post_image, pre_box, post_box

    def _create_sp_partial(self, id_action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a predicate partial state for the given action.

        Args:
            id_action: Action id.
        Returns:
            (sp_partial [U, 2, P], idx_used [2, MC]) boolean arrays.
        """
        # [2, MC, 2, P] (pre/post, arg_combos, pos/neg, num_preds)
        action = str(self.pddl.actions[id_action])
        sp_partial = dnf_utils.get_partial_predicate_state(self.pddl, action)

        # [2, MC, 2, P] -> [2 * MC, 2 * P]
        sp_flat = sp_partial.reshape((-1, sp_partial.shape[2] * sp_partial.shape[3]))

        # [2, MC, 2, P] -> [2 * MC, 2, P]
        sp_partial = sp_partial.reshape((-1, *sp_partial.shape[2:]))

        # [2 * MC, 2 * P] -> [2 * MC]
        idx_used = (sp_flat != 0).any(axis=1)

        # [U, 2, P] (pre_args/post_args, pos/neg, num_preds)
        sp_partial = sp_partial[idx_used, :, :]

        # [2 * MC] -> [2, MC] (pre/post, arg_combos)
        idx_used = idx_used.reshape((2, -1))

        return sp_partial, idx_used

    def _create_boxes(
        self, pre_box: np.ndarray, post_box: np.ndarray, idx_used: np.ndarray
    ) -> np.ndarray:
        """Organizes boxes into indexed boxes.

        Args:
            pre_box: [16, 3, 4] (arg_combos, roi/a/b, x1/y1/x2/y2) float32 array.
            post_box: [16, 3, 4] (arg_combos, roi/a/b, x1/y1/x2/y2) float32 array.
            idx_used: [2, MC] (pre/post, arg_combos) bool array.
        Returns:
            [U, 1 + M, 5] (pre_args/post_args, roi/a/b, idx/x1/y1/x2/y2) float32 array.
        """
        NUM_PRE, NUM_POST = idx_used.sum(axis=1)

        # [U, 1 + M, 5] (pre_args/post_args, roi/a/b, idx/x1/y1/x2/y2)
        indexed_boxes = np.empty(
            (NUM_PRE + NUM_POST, pre_box.shape[1], pre_box.shape[2] + 1),
            dtype=np.float32,
        )

        # [MC, 1 + M, 4] -> [U, 1 + M, 4]
        indexed_boxes[:NUM_PRE, :, 1:] = pre_box[idx_used[0]]
        indexed_boxes[NUM_PRE:, :, 1:] = post_box[idx_used[1]]

        indexed_boxes[:, :, 0] = np.arange(indexed_boxes.shape[0], dtype=np.float32)[
            :, None
        ]

        return indexed_boxes

    def _create_images(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        idx_used: np.ndarray,
    ) -> np.ndarray:
        """Creates a 3-channel CNN image.

        Args:
            pre_image: [H, W, 3] uint8 array.
            post_image: [H, W, 3] uint8 array.
            idx_used: [2, 16] (pre/post, arg_combos) bool array.
        Returns:
            [U, 3, H, W] (pre_args/post_args, rgb, H, W) float32 array.
        """
        NUM_PRE, NUM_POST = idx_used.sum(axis=1)
        H, W = pre_image.shape[:2]

        # [2 * U, 3, 240, 494] (pre_args/post_args, rgb/a/b, H, W)
        images = np.zeros((NUM_PRE + NUM_POST, *self.dim_image), dtype=np.float32)

        # Normalize.
        # [H, W, 3] -> [3, H, W]
        images[:NUM_PRE, :, :, :W] = video_utils.image_rgb_to_cnn(
            pre_image, self.image_distribution
        )[None, ...]
        images[NUM_PRE:, :, :, :W] = video_utils.image_rgb_to_cnn(
            post_image, self.image_distribution
        )[None, ...]

        return images

    def _create_masks(
        self,
        indexed_boxes: np.ndarray,
        idx_used: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Creates a 2-channel bounding box mask.

        Args:
            indexed_boxes: [U, M + 1, 5] (pre_args/post_args, roi/a/b, idx/x1/y1/x2/y2) float32 array.
            idx_used: [2, 16] (pre/post, arg_combos) bool array.
            width: Width of current image.
        Returns:
            [U, M, H, W] (pre_args/post_args, max_args, H, W) float32 array.
        """
        NUM_PRE, NUM_POST = idx_used.sum(axis=1)
        _, H, W_ALL = self.dim_image
        W = width

        # [U, M, 240, 494] (pre_args/post_args, a/b, H, W)
        masks = np.zeros((NUM_PRE + NUM_POST, 2, H, W_ALL), dtype=np.float32)

        # [U, M, 4] -> [U, M, H, W] (arg_combos, a/b, H, W)
        dnf_utils.bbox_masks(W, H, indexed_boxes[:, 1:, 1:], mask=masks[:, :, :, :W])

        return masks

    def _get_data_point(  # type: ignore
        self, idx: int
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a single data point from cache or disk.

        Args:
            idx: Index of data point.
        Returns:
            (id_video, images, boxes, s_partial)
        """
        # [H, W, 3], [H, W, 3], [16, 3, 4], [16, 3, 4]
        (
            id_video,
            id_action,
            pre_image,
            post_image,
            pre_box,
            post_box,
        ) = self._get_data_from_disk(idx)

        sp_partial, idx_used = self._create_sp_partial(id_action)
        images = self._create_images(pre_image, post_image, idx_used)
        indexed_boxes = self._create_boxes(pre_box, post_box, idx_used)
        masks = self._create_masks(indexed_boxes, idx_used, width=pre_image.shape[1])

        return id_video, images, masks, indexed_boxes, sp_partial, idx_used

    @property
    def collate_fn(self) -> Optional[Callable[[List], Any]]:
        """Collate function used to create batches with torch.DataLoader."""
        return self._collate_fn

    @staticmethod
    def _collate_fn(
        data: List[
            Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batches data by concatenating along the variable-length 0-axis.

        Args:
            data: List of data points.
        Returns:
            Batched data (
                images [-1, 3, H, W] (pre_args/post_args/..., rgb, H, W),
                masks [-1, 2, H, W] (pre_args/post_args/..., a/b, H, W),
                indexed_boxes [-1, 3, 5] (pre_args/post_args/..., roi/a/b, idx/x1/y1/x2/y2),
                sp_partial [-1, 2, P] (pre_args/post_args/..., pos/neg, num_preds),
            ).
        """
        id_video, images, masks, indexed_boxes, sp_partial, idx_used = zip(*data)

        # [-1, 3, H, W] (pre_args/post_args/..., rgb, H, W)
        images = np.concatenate(images, axis=0)

        # [-1, 2, H, W] (pre_args/post_args/..., a/b, H, W)
        masks = np.concatenate(masks, axis=0)

        # [-1, 3, 5] (pre_args/post_args/..., roi/a/b, idx/x1/y1/x2/y2),
        indexed_boxes = np.concatenate(indexed_boxes, axis=0)
        indexed_boxes[:, :, 0] = np.arange(images.shape[0], dtype=np.float32)[:, None]

        # [-1, 2, P] (pre_args/post_args/..., pos/neg, num_preds),
        sp_partial = np.concatenate(sp_partial, axis=0)

        images = torch.from_numpy(images)
        masks = torch.from_numpy(masks)
        indexed_boxes = torch.from_numpy(indexed_boxes)
        sp_partial = torch.from_numpy(sp_partial)

        return images, masks, indexed_boxes, sp_partial
