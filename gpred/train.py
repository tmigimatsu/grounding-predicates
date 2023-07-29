import pathlib
import random
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch  # type: ignore
import numpy as np
import tqdm

from torch.utils.data import DataLoader  # type: ignore
import torch.utils.tensorboard  # type: ignore
from . import dataset, model, problem


class Trainer:
    """Class containing training-specific modules, including the data loaders.

    Args:
        train_set: Training set.
        test_set: Test set.
        optimizer: Factory function that takes in net params and outputs an optimizer.
        scheduler: Factory function that takes in an optimizer and outputs a learning rate scheduler.
        batch_size: Batch size.
        num_epochs: Number of epochs.
        shuffle_data: Whether to shuffle the training data.
        checkpoint: Torch checkpoint to resume.

    Example:
        # >>> from gpred.dataset import Dataset
        # >>> from gpred.train import Trainer
        # >>> import symbolic
        # >>> import torch
        # >>> pddl = symbolic.Pddl("../env/gridworld/domain.pddl", "../env/gridworld/problem.pddl")
        # >>> test_set = Dataset(pddl, split=(0.99, 1.0))
        # >>> trainer = Trainer(None, test_set, optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=0.001))
        # >>> len(next(iter(trainer.val_loader)))
        # 5
    """

    def __init__(
        self,
        train_set: Optional[dataset.Dataset],
        val_set: dataset.Dataset,
        optimizer: Callable[
            [Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer
        ],
        scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        batch_size: int = 64,
        num_epochs: int = 20,
        shuffle_data: bool = True,
        checkpoint: Optional[Dict] = None,
    ):
        self._train_set = train_set
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._checkpoint = checkpoint

        self._val_loader = DataLoader(
            val_set,
            batch_size=min(self._batch_size, 32),  # TODO: Only for gridworld
            shuffle=False,
            pin_memory=True,
            collate_fn=val_set.collate_fn,
        )
        if train_set is not None and shuffle_data is not False:
            self._train_set = None
            self._train_loader = DataLoader(
                train_set,
                batch_size=self._batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=train_set.collate_fn,
            )

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        """Train set loader."""
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        """Validation set loader."""
        return self._val_loader

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Optimizer."""
        return self._optimizer

    @property
    def scheduler(self):
        """Learning rate scheduler."""
        return self._scheduler

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    @property
    def num_epochs(self) -> int:
        """Max number of epochs."""
        return self._num_epochs

    def register_network(self, net: model.Net) -> None:
        """Registers network to create optimizer and scheduler.

        Called at the beginning of Trainer.train().

        Args:
            net: Network model.
        """
        self._optimizer = self._optimizer_factory(net.parameters())
        if self._scheduler_factory is None:
            self._scheduler = None
        else:
            self._scheduler = self._scheduler_factory(self._optimizer)

    def shuffle_data(self) -> None:
        """Shuffles training data.

        Should be called after setting random seed.

        Example:
            # >>> from gpred.dataset import Dataset
            # >>> from gpred.train import Trainer
            # >>> import symbolic
            # >>> import torch
            # >>> pddl = symbolic.Pddl("../env/gridworld/domain.pddl", "../env/gridworld/problem.pddl")
            # >>> train_set = Dataset(pddl, split=(0.0, 0.01))
            # >>> test_set = Dataset(pddl, split=(0.99, 1.0))
            # >>> trainer = Trainer(None, test_set, optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=0.001))
            # >>> _ = torch.manual_seed(0)
            # >>> trainer.shuffle_data()
        """
        if self._train_set is not None:
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=2,
            )

    def train(
        self,
        net: model.Net,
        problem: problem.Problem,
        path_run: pathlib.Path,
        tqdm: tqdm.tqdm = tqdm.tqdm,
        tensorboard_writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
    ) -> int:
        """Trains the network.

        Args:
            net: Neural network.
            problem: Problem specification.
            path_run: Path to save checkpoints.
            tqdm: Tqdm iterator.
            tensorboard_writer: Tensorboard writer.
        Returns:
            Number of training iterations elapsed.
        """
        self.register_network(net)
        if self._checkpoint is not None:
            num_epoch = self._checkpoint["epoch"] + 1
            num_iter = self._checkpoint["num_iter"]

            net.load_state_dict(self._checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self._checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(self._checkpoint["scheduler_state_dict"])
            net.eval()

            torch.random.set_rng_state(self._checkpoint["torch_random_state"])
            random.setstate(self._checkpoint["random_state"])
            np.random.set_state(self._checkpoint["np_random_state"])
        else:
            num_epoch = 0
            num_iter = 0

        # Iterate over epochs
        epoch_loop = tqdm(range(num_epoch, self.num_epochs))
        epoch_loop.set_description("Epochs")
        for epoch in epoch_loop:

            # Iterate over dataset
            train_loop = tqdm(self.train_loader)
            for idx_batch, data in enumerate(train_loop):
                # Forward pass
                problem.set_data_batch(data)
                loss, classification_stats = forward_pass(net, problem)
                if classification_stats is not None:
                    precision, recall, f1 = compute_f1(*classification_stats)

                # Backward pass
                self.optimizer.zero_grad()  # Zero the parameter gradients
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                self.optimizer.step()

                # Logging
                train_loop.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
                train_loop.set_postfix(
                    loss=loss.item(), precision=precision, recall=recall, f1=f1
                )
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Loss/train", loss, num_iter)
                    if classification_stats is not None:
                        tensorboard_writer.add_scalar(
                            "Precision/train", precision, num_iter
                        )
                        tensorboard_writer.add_scalar("Recall/train", recall, num_iter)
                        tensorboard_writer.add_scalar("F1/train", f1, num_iter)

                num_iter += 1

            # Evaluate
            loss, precision, recall, f1 = evaluate(net, problem, self.val_loader)

            # Logging
            epoch_loop.set_postfix(
                loss=loss.item(), precision=precision, recall=recall, f1=f1
            )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/validation", loss, num_iter)
                tensorboard_writer.add_scalar(
                    "Precision/validation", precision, num_iter
                )
                tensorboard_writer.add_scalar("Recall/validation", recall, num_iter)
                tensorboard_writer.add_scalar("F1/validation", f1, num_iter)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "torch_random_state": torch.random.get_rng_state(),
                    "random_state": random.getstate(),
                    "np_random_state": np.random.get_state(),
                    "num_iter": num_iter,
                },
                path_run / f"checkpoint-{epoch}.pth",
            )

            if f1 == 1.0:
                break

        return num_iter


def forward_pass(
    net: model.Net, problem: problem.Problem
) -> Tuple[torch.Tensor, Optional[Tuple[int, int, int]]]:
    """Compute the loss and accuracy of the network.

    Args:
        net: Neural network module.
        problem: Problem specification.
    Returns:
        Tuple (loss [1,], classification_stats [true_positives, selected, relevant]).
    """
    # Forward pass
    y_predict = net(problem.get_x())

    # Compute loss
    loss = problem.compute_loss(y_predict)
    if torch.isnan(loss).any():
        raise ValueError(f"NaN loss: {loss}\ny_predict: {y_predict}")

    # Compute accuracy
    y_true = problem.get_y_true().bool()
    if y_true is None:
        return loss, None

    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        y_true = y_true.bool()

        y_mask = problem.get_y_mask()
        if y_mask is not None:
            y_predict[:, ~y_mask] = -float("inf")
            y_true &= y_mask[None, :]

        true_positives = ((y_predict > 0) & y_true).flatten().sum().item()
        num_selected = (y_predict > 0).flatten().sum().item()
        num_relevant = y_true.flatten().sum().item()
        classification_stats = (true_positives, num_selected, num_relevant)
    else:
        # DNF accuracy.

        # [-1, N]
        y_pos = y_true[:, 0, :].bool()
        y_neg = y_true[:, 1, :].bool()
        y_mask = y_pos | y_neg

        y_predict_masked = (y_predict > 0) & y_mask
        true_positives = (y_predict_masked & y_pos).flatten().sum().item()
        num_selected = y_predict_masked.flatten().sum().item()
        num_relevant = y_pos.flatten().sum().item()
        classification_stats = (true_positives, num_selected, num_relevant)

    return loss, classification_stats


def compute_f1(
    true_positives: int, num_selected: int, num_relevant: int
) -> Tuple[float, float, float]:
    """Compute the precision, recall, and f1 score.

    Args:
        true_positives: Number of true positives.
        num_selected: Number of classified positives.
        num_relevant: Number of actual positives.
    Returns:
        (precision, recall, f1) tuple.
    """
    precision = true_positives / num_selected if num_selected > 0.0 else 0.0
    recall = true_positives / num_relevant if num_relevant > 0.0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return precision, recall, f1


def evaluate(
    net: model.Net, problem: problem.Problem, val_loader: torch.utils.data.DataLoader
) -> Tuple[float, float, float, float]:
    """Compute the loss and accuracy of the network on the validation set.

    Args:
        net: Neural network module.
        problem: Problem specification.
        val_loader: Validation data loader.
    Returns:
        (loss, precision, recall, f1) tuple.
    """
    with torch.no_grad():
        losses = []
        true_positives = 0
        num_selected = 0
        num_relevant = 0
        for data in val_loader:
            # Forward pass
            problem.set_data_batch(data)
            loss, classification_stats = forward_pass(net, problem)

            if classification_stats is not None:
                true_positives_i, num_selected_i, num_relevant_i = classification_stats
                true_positives += true_positives_i
                num_selected += num_selected_i
                num_relevant += num_relevant_i
            losses.append(loss.item())

        loss = np.mean(losses)
        precision, recall, f1 = compute_f1(true_positives, num_selected, num_relevant)
        return loss, precision, recall, f1
