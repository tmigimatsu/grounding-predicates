import pathlib
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import PIL
import symbolic  # type: ignore
import torch  # type: ignore
import torchvision.transforms as tf  # type: ignore
import tqdm

from . import dataset, dnf_utils


class Problem:
    """Class containing problem-specific data and loss.

    Args:
        pddl: Pddl instance.
        device: Cpu or gpu.
        path: Path of dataset.
        dataset: Filename of dataset.
        max_num_conjunctions: Maximum number of conjunctions. If None, will be computed.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        device: torch.device,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        dataset: Optional[str] = "dataset.hdf5",
        max_num_conjunctions: Optional[int] = 8,
    ):
        self._device = device

        size_img = self.dim_x[1:]
        img_mean, img_stddev = self.image_distribution
        self._transform = tf.Compose(
            [
                tf.Resize(size_img, PIL.Image.BICUBIC),
                tf.ToTensor(),
                tf.Normalize(img_mean, img_stddev),
            ]
        )

        # Load datasets
        if dataset is not None:
            import gpred.dataset

            self._train_set = gpred.dataset.Dataset(
                pddl=pddl,
                path=path,
                dataset=dataset,
                split=(0.0, 0.8),
                transform=self._transform,
                max_num_conjunctions=max_num_conjunctions,
            )
            self._val_set = gpred.dataset.Dataset(
                pddl=pddl,
                path=path,
                dataset=dataset,
                split=(0.8, 1.0),
                transform=self._transform,
                max_num_conjunctions=max_num_conjunctions,
            )
            self._test_set: Optional[gpred.dataset.Dataset] = None

    @property
    def train_set(self) -> dataset.Dataset:
        """Train set."""
        return self._train_set

    @property
    def test_set(self) -> Optional[dataset.Dataset]:
        """Validation set."""
        return self._test_set

    @property
    def val_set(self) -> dataset.Dataset:
        """Test set."""
        return self._val_set

    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._device

    @property
    def dim_x(self) -> Tuple[int, int, int]:
        """Size of network input."""
        return (3, 220, 220)

    @property
    def dim_y(self) -> int:
        """Size of network output."""
        return 197

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev)."""
        return (0.7170, 0.7143, 0.7097), (0.2719, 0.2740, 0.2784)

    def compute_image_distribution(
        self, tqdm: tqdm.tqdm = tqdm.tqdm
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes image distribution.

        Args:
            tqdm: Tqdm iterator.
        Returns:
            (mean, stddev) tuple.
        """

        # Compute mean
        img_mean = torch.zeros((3,))
        for i in tqdm(range(len(self.train_set))):
            action, img, dnf, mask, s = self.train_set[i]
            img_mean += img.mean((0, 2, 3))
        img_mean /= len(self.train_set)

        print("img_mean:", img_mean)

        # Compute stddev
        img_var = torch.zeros((3,))
        for i in tqdm(range(len(self.train_set))):
            action, img, dnf, mask, s = self.train_set[i]
            img_var += (img - img_mean[None, :, None, None]).square().mean((0, 2, 3))
        img_var /= len(self.train_set)
        img_stddev = img_var.sqrt()

        print("img_stddev:", img_stddev)

        return img_mean, img_stddev

    def reshape(self, tensors: torch.Tensor) -> torch.Tensor:
        """Reshapes a list of [-1, 2, ...] tensors into a list of [-1 * 2, ...] tensors."""
        if isinstance(tensors, (list, tuple)):
            return tuple(tensor.view(-1, *tensor.shape[2:]) for tensor in tensors)
        return tensors.view(-1, *tensors.shape[2:])

    def set_data_batch(
        self, data: Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Sets data batch.

        Args:
            data: Batch (action, images, mask, dnf, s) from dataset.
        """
        action, images, dnf, mask, s = data
        self._x = self.reshape(images).to(self.device)
        self._y_true = self.reshape(s).to(self.device)

    def get_x(self) -> np.ndarray:
        """Gets neural network input for current batch."""
        return self._x

    def get_y_true(self) -> torch.Tensor:
        """Gets expected neural network output for current batch."""
        return self._y_true

    def get_y_mask(self) -> Optional[torch.Tensor]:
        """Gets an optional mask over valid elements of y."""
        return None

    def compute_loss(self, y_predict: torch.Tensor) -> torch.Tensor:
        """Computes loss for current batch.

        Args:
            y_predict: Predicted neural network output.
        """
        from gpred.loss import binary_cross_entropy_with_logits

        return binary_cross_entropy_with_logits(y_predict, self.get_y_true())


class TwentyBnClassification(Problem):
    """Classification of 20BN predicates."""

    def __init__(
        self,
        pddl: symbolic.Pddl,
        device: torch.device,
        path: Union[pathlib.Path, str] = "../data/twentybn",
        train_dataset: str = "dataset.hdf5",
        val_dataset: str = "dataset_val.hdf5",
        idx_props: Optional[List[int]] = None,
    ):

        self._device = device

        # Load datasets.
        self._train_set = dataset.TwentyBnDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
        )
        self._val_set = dataset.TwentyBnDataset(
            pddl=pddl,
            path=path,
            dataset=val_dataset,
        )

        if idx_props is None:
            self._idx_props = dnf_utils.get_valid_props(pddl).nonzero()[0]
        else:
            self._idx_props = np.array(idx_props)

    @property
    def dim_x(self) -> Tuple[int, int, int]:
        """Size of network input.

        The image can have a variable aspect ratio, but the minimum of (width, height) will be 224."""
        return self.train_set.dim_image

    @property
    def dim_y(self) -> int:
        """Size of network output (all mutable predicates)."""
        return len(self._idx_props)

    @property
    def idx_props(self) -> np.ndarray:
        """Mapping of y-dimensions to proposition indices."""
        return self._idx_props

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev)."""
        return self.train_set.image_distribution

    def set_data_batch(self, data: Tuple) -> None:
        """Sets the data batch.

        Args:
            data: Batch (id_video, images, boxes, s_partial) from dataset."""
        id_video, images, boxes, s_partial = data

        # Get proposition value from partial state.
        # [-1, 2, 2, N] => [-1, 2, 2, P]
        dnf_props = s_partial[..., self._idx_props].float()

        # [-1, 2, 7, H, W] => [-2, 7, H, W]
        images = self.reshape(images)

        # [-1, 2, 5, 4] => [-2, 5, 4]
        boxes = self.reshape(boxes)

        # [-1, 2, 2, P] => [-2, 2, P]
        dnf_props = self.reshape(dnf_props)

        # Filter out data without any learnable props.
        idx_data = dnf_props.sum(dim=2).sum(dim=1) == 0
        images = images[idx_data, :, :, :]
        boxes = boxes[idx_data, :, :]
        dnf_props = dnf_props[idx_data, :, :]

        # Add batch indices to boxes.
        # [-2, 5, 1]
        indices = (
            torch.arange(boxes.shape[0])
            .float()
            .view(-1, 1, 1)
            .repeat(1, boxes.shape[1], 1)
        )

        # [-2, 5, 4] => [-2, 5, 5]
        indexed_boxes = torch.cat((indices, boxes), dim=2)

        # ([-2, 7, H, W], [-2, 5, 5])
        self._x = (images.to(self.device), indexed_boxes.to(self.device))

        # [-2, 2, P]
        self._y_true = dnf_props.to(self.device)

    def compute_loss(self, y_predict):
        """Computes loss for current batch.

        Args:
            y_predict: Predicted neural network output.
        """
        from gpred.loss import dnf_cross_entropy

        return dnf_cross_entropy(y_predict, self.get_y_true())


class TwentyBnPredicateClassification(TwentyBnClassification):
    """Classification of individual 20BN predicates."""

    def __init__(
        self,
        pddl: symbolic.Pddl,
        device: torch.device,
        path: Union[pathlib.Path, str] = "../data/twentybn",
        train_dataset: str = "predicate_train.hdf5",
        val_dataset: str = "predicate_val.hdf5",
        test_dataset: str = "predicate_test.hdf5",
        use_weighted_ce: bool = False,
    ):
        self._device = device

        # Load datasets.
        self._train_set = dataset.TwentyBnPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            shuffle=True,
        )
        self._val_set = dataset.TwentyBnPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=val_dataset,
            shuffle=False,
        )
        self._test_set = dataset.TwentyBnPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=test_dataset,
            shuffle=False,
        )

        self._idx_props = dnf_utils.get_valid_props(pddl).nonzero()[0]

        self._weights: Optional[torch.Tensor] = None
        if use_weighted_ce:
            with h5py.File(pathlib.Path(path) / "predicate_val.hdf5", "r") as f:
                actions = [str(a) for a in pddl.actions]
                action_instances = [actions[idx_a] for idx_a in f["actions"]]
            weights = dnf_utils.compute_predicate_class_weights(
                pddl, action_instances=action_instances
            )
            self._weights = torch.from_numpy(weights).to(self.device)

        self._P = len(pddl.predicates)

    @property
    def dim_y(self) -> int:
        """Size of network output (all predicates)."""
        return self._P

    def set_data_batch(  # type: ignore
        self,
        data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Sets the data batch.

        Args:
            data: Batch (images, indexed_boxes, s_partial) from dataset.
        """
        images, masks, indexed_boxes, sp_partial = data

        # ([-32, 5, H, W], [-32, 3, 5])
        self._x = (
            images.to(self.device),
            masks.to(self.device),
            indexed_boxes.to(self.device),
        )

        # [-32, 2, P]
        self._y_true = sp_partial.float().to(self.device)

    def compute_loss(self, y_predict):
        """Computes loss for current batch.

        Args:
            y_predict: Predicted neural network output.
        """
        from gpred.loss import dnf_cross_entropy

        return dnf_cross_entropy(y_predict, self.get_y_true(), self._weights)


class GridworldPredicateClassification(Problem):
    """Classification of Gridworld predicates.

    Args:
        pddl: Pddl instance.
        device: Torch device.
        path: Path of dataset.
        train_dataset: Filename of dataset.
        ground_truth: Whether to use ground truth or dnf labels to train network.
        use_weighted_ce: Whether to use weighted or unweighted cross entropy.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        device: torch.device,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        train_dataset: str = "dataset.hdf5",
        ground_truth: bool = False,
        use_weighted_ce: bool = False,
    ):
        self._device = device

        # Load datasets.
        self._train_set = dataset.GridworldPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.0, 0.1),
            ground_truth=ground_truth,
        )
        self._val_set = dataset.GridworldPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.9, 1.0),
            ground_truth=True,
        )
        self._test_set = dataset.GridworldPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.8, 0.9),
            ground_truth=True,
        )

        self._idx_props = dnf_utils.get_valid_props(pddl).nonzero()[0].tolist()

        self._weights: Optional[torch.Tensor] = None
        if use_weighted_ce:
            with h5py.File(pathlib.Path(path) / "dataset.hdf5", "r") as f:
                actions = [action.decode("utf-8") for action in set(f["actions"])]
            weights = dnf_utils.compute_predicate_class_weights(pddl, actions=actions)
            self._weights = torch.from_numpy(weights).to(self.device)

        self._P = len(pddl.predicates)

    @property
    def dim_x(self) -> Tuple[int, int, int]:
        """Size of network input.

        The image can have a variable aspect ratio, but the minimum of (width, height) will be 224."""
        return self.train_set.dim_image

    @property
    def dim_y(self) -> int:
        """Size of network output (all mutable predicates)."""
        return self._P

    @property
    def idx_props(self) -> List[int]:
        """Mapping of y-dimensions to proposition indices."""
        return self._idx_props

    @property
    def image_distribution(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Image (mean, stddev)."""
        return self.train_set.image_distribution

    def set_data_batch(self, data: Tuple) -> None:
        """Sets the data batch.

        Args:
            data: Batch (images, masks, indexed_boxes, s_partial) from dataset.
        """
        images, masks, indexed_boxes, sp_partial = data

        # ([-32, 5, H, W], [-32, 3, 5])
        self._x = (
            images.to(self.device),
            masks.to(self.device),
            indexed_boxes.to(self.device),
        )

        # [-32, 2, P]
        self._y_true = sp_partial.float().to(self.device)

    def compute_loss(self, y_predict):
        """Computes loss for current batch.

        Args:
            y_predict: Predicted neural network output.
        """
        from gpred.loss import dnf_cross_entropy

        return dnf_cross_entropy(y_predict, self.get_y_true(), self._weights)


class GridworldHalfPredicateClassification(GridworldPredicateClassification):
    """Classification of Gridworld predicates.

    Args:
        pddl: Pddl instance.
        device: Torch device.
        path: Path of dataset.
        train_dataset: Filename of dataset.
        ground_truth: Whether to use ground truth or dnf labels to train network.
        use_weighted_ce: Whether to use weighted or unweighted cross entropy.
    """

    def __init__(
        self,
        pddl: symbolic.Pddl,
        device: torch.device,
        path: Union[pathlib.Path, str] = "../data/gridworld",
        train_dataset: str = "dataset.hdf5",
        ground_truth: bool = False,
        use_weighted_ce: bool = False,
    ):
        self._device = device

        # Load datasets.
        self._train_set = dataset.GridworldHalfPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.0, 0.2),
            ground_truth=ground_truth,
        )
        self._val_set = dataset.GridworldPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.9, 1.0),
            ground_truth=True,
        )
        self._test_set = dataset.GridworldPredicateDataset(
            pddl=pddl,
            path=path,
            dataset=train_dataset,
            split=(0.8, 0.9),
            ground_truth=True,
        )

        self._idx_props = dnf_utils.get_valid_props(pddl).nonzero()[0].tolist()

        self._weights: Optional[torch.Tensor] = None
        if use_weighted_ce:
            with h5py.File(pathlib.Path(path) / "dataset.hdf5", "r") as f:
                actions = [action.decode("utf-8") for action in set(f["actions"])]
            weights = dnf_utils.compute_predicate_class_weights(pddl, actions=actions)
            self._weights = torch.from_numpy(weights).to(self.device)

        self._P = len(pddl.predicates)
