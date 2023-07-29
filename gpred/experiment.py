import datetime
import os
import pathlib
import random
import re
from typing import Any, Dict, Optional, List, Union

import numpy as np
import symbolic  # type: ignore
import torch  # type: ignore

from .problem import (
    GridworldPredicateClassification,
    GridworldHalfPredicateClassification,
    Problem,
    TwentyBnPredicateClassification,
)
from .model import Net, SimplePropositionNet, PredicateNet
from .train import Trainer
from . import dnf_utils


def get_next_experiment(path):
    filenames = os.listdir(path)
    exps = []
    for filename in filenames:
        match = re.match(r"exp-(\d+)", filename)
        if match is not None:
            exps.append(int(match[1]))

    if not exps:
        return "exp-0"

    num_exp = max(exps) + 1
    return f"exp-{num_exp}"


def get_datetime():
    return datetime.datetime.now().strftime("%b%d_%H-%M-%S")


class Experiment:
    """Experiment base class.

    Args:
        device: Torch device (torch.device('cuda:0') or torch.device('cpu')).
        paths: config.EnvironmentPaths.
        num_exp: Name of experiment ('exp-01').
        eval_only: Load model only for evaluation.
    """

    def __init__(
        self, device: torch.device, paths, num_exp: str, eval_only: bool = False
    ):
        self._device = device
        self._paths = paths
        self._pddl = symbolic.Pddl(
            str(self.paths.domain_pddl), str(self.paths.problem_pddl)
        )

        path_models = self.paths.models
        path_models.mkdir(parents=True, exist_ok=True)
        if num_exp is None:
            num_exp = get_next_experiment(path_models)

        self._path_exp = path_models / num_exp
        self._state_dict: Dict[str, Any] = {}
        self._checkpoint: Optional[Dict] = None

        self.eval_only = eval_only

    def __iter__(self):
        """Iterates over experiments."""
        self._problem = Problem(self.pddl, self.device)

        lr = self.get_param("AdamW.lr", 0.0001, verbose=not self.eval_only)
        step_size = self.get_param("StepLR.step_size", 4, verbose=not self.eval_only)
        gamma = self.get_param("StepLR.gamma", 0.5, verbose=not self.eval_only)

        self._trainer = Trainer(
            self.problem.train_set,
            self.problem.test_set,
            optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=lr),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            ),
            shuffle_data=not self.eval_only,
        )
        self._network = SimplePropositionNet(
            dim_input=self.problem.dim_x, dim_output=self.problem.dim_y
        ).to(self.device)

        seed = self.get_param("seed", 0, verbose=not self.eval_only)
        self.seed(seed)

        path_run = self.get_param(
            "path_run", self.path_exp / get_datetime(), verbose=not self.eval_only
        )

        if not self.eval_only:
            path_run.mkdir(parents=True, exist_ok=True)
            self.save(path_run / "experiment.pth")

        yield path_run
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=not self.eval_only)

    #         for ground_truth_loss in (True, False):
    #             for i in range(10):
    #                 self.seed(i)
    #                 dnf_gt = "gt" if ground_truth_loss else "dnf"
    #                 path_run = self.path_exp / dnf_gt / get_datetime()
    #                 yield path_run

    def seed(self, seed: int):
        """Set the random seed.

        Args:
            seed: Seed.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.trainer.shuffle_data()

    def get_param(self, param: str, default: Any, verbose=False):
        """Get parameter from `self.state_dict` if it exists, otherwise set it to default.

        Args:
            param: Name of parameter in `self.state_dict`.
            default: Default value.
            verbose: Print value.
        Returns:
            Param or default.
        """
        if param in self.state_dict:
            if verbose:
                print(f"Loading {param} = {self.state_dict[param]}")
            return self.state_dict[param]

        if verbose:
            print(f"Setting {param} = {default}")
        self.state_dict[param] = default
        return default

    def load(self, path: Union[str, pathlib.Path]):
        """Load experiment state from disk.

        Args:
            path: Path of experiment.pth file.
        """
        self._state_dict = torch.load(path)

    def save(self, path: Union[str, pathlib.Path]):
        """Save experiment state to disk.

        Args:
            path: Path of experiment.pth file.
        """
        torch.save(self.state_dict, path)

    def reset(self):
        """Reset experiment state."""
        self._state_dict = {}

    @staticmethod
    def get_path_exp(paths, num_exp: str) -> pathlib.Path:
        """Returns the experiment path.

        Args:
            paths: Environment paths.
            num_exp: Name of experiment ('exp-01').
        """
        return paths.models / num_exp

    @staticmethod
    def get_path_runs(paths, num_exp: str) -> List[pathlib.Path]:
        """Returns the experiment run paths.

        Args:
            paths: Environment paths.
            num_exp: Name of experiment ('exp-01').
        Returns:
            List of experiment runs sorted by time in ascending order.
        """
        path_exp = Experiment.get_path_exp(paths, num_exp)
        path_runs = []
        for path in path_exp.iterdir():
            try:
                datetime.datetime.strptime(path.name, "%b%d_%H-%M-%S")
            except ValueError:
                continue
            path_runs.append(path)
        path_runs = sorted(
            path_runs, key=lambda p: datetime.datetime.strptime(p.name, "%b%d_%H-%M-%S")
        )
        return path_runs

    def load_model(
        self,
        idx_run: int = -1,
        iteration: int = 0,
        debug: bool = False,
        checkpoint: Optional[int] = None,
    ):
        """Loads a specified model.

        Args:
            idx_run: Index of experiment run, sorted chronologically.
            iteration: Experiment iteration.
            debug: Whether to store intermediate network outputs for debugging.
            checkpoint: Optionally load a checkpoint instead of a finished model.
        """
        for i, path_run in enumerate(self):
            if iteration < i:
                continue

            # Load model
            runs = sorted([p for p in path_run.parent.iterdir() if p.is_dir()])
            MONTHS = {
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            }
            runs = [p for p in runs if p.name[:3] in MONTHS]

            path_run = runs[idx_run]
            if checkpoint is not None:
                print(f"Loading checkpoint {checkpoint} from {path_run}")
                state_dicts = torch.load(path_run / f"checkpoint-{checkpoint}.pth")
                self.network.load_state_dict(state_dicts["model_state_dict"])
            else:
                print(f"Loading model from {path_run}")
                self.network.load_state_dict(torch.load(path_run / "model.pth"))
            self.network.eval()
            self.network._debug = debug

            return path_run

    def resume(self, path_run: Union[str, pathlib.Path], epoch: int):
        """Loads a specified checkpoint to resume training.

        Args:
            path_run: Experiment run path.
            epoch: Epoch checkpoint to load.
        """
        path_run = pathlib.Path(path_run)
        if epoch == -1:
            checkpoints = sorted(
                [p for p in path_run.iterdir() if "checkpoint" in p.name]
            )
            m = re.match(r"checkpoint-(\d)+.pth", checkpoints[-1].name)
            if m is not None:
                epoch = int(m[1])
            else:
                raise RuntimeError(f"Could not find checkpoints at {path_run}.")

        self.load(path_run / "experiment.pth")

        print(f"Loading model from {path_run} at epoch {epoch}.")
        self._checkpoint = torch.load(path_run / f"checkpoint-{epoch}.pth")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state_dict

    @property
    def pddl(self) -> symbolic.Pddl:
        return self._pddl

    @property
    def problem(self) -> Problem:
        return self._problem

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @property
    def network(self) -> Net:
        return self._network

    @property
    def path_exp(self) -> pathlib.Path:
        return self._path_exp

    @property
    def paths(self):
        return self._paths


class TwentyBnPredicateExperiment(Experiment):
    """Test all predicates in the 20BN dataset."""

    def __init__(
        self,
        device: torch.device,
        paths,
        num_exp: str,
        eval_only: bool = False,
        num_epochs: int = 10,
        model: str = "resnet",
        mini: bool = False,
        use_weighted_ce: bool = False,
        batch_size: int = 32,
        freeze_features: bool = True,
    ):
        super().__init__(device, paths, num_exp=num_exp, eval_only=eval_only)
        self._num_epochs = num_epochs
        self._model = model
        self._mini = mini
        self._use_weighted_ce = use_weighted_ce
        self._batch_size = batch_size
        self._freeze_features = freeze_features

    @staticmethod
    def idx_props(pddl: symbolic.Pddl) -> np.ndarray:
        """Returns the proposition indices for this experiment."""
        return dnf_utils.get_valid_props(pddl).nonzero()[0]

    def __iter__(self):
        variant = "_mini" if self._mini else ""
        self._problem = TwentyBnPredicateClassification(
            self.pddl,
            self.device,
            path=self.paths.data,
            train_dataset=f"predicate_train{variant}.hdf5",
            val_dataset=f"predicate_val{variant}.hdf5",
            use_weighted_ce=self._use_weighted_ce,
        )

        lr = self.get_param("AdamW.lr", 0.0001, verbose=not self.eval_only)
        step_size = self.get_param("StepLR.step_size", 4, verbose=not self.eval_only)
        gamma = self.get_param("StepLR.gamma", 0.5, verbose=not self.eval_only)

        self._trainer = Trainer(
            self.problem.train_set,
            self.problem.val_set,
            batch_size=self._batch_size,
            optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=lr),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            ),
            num_epochs=self._num_epochs,
            shuffle_data=not self.eval_only,
            checkpoint=self._checkpoint,
        )

        self._network = PredicateNet(
            dim_input=self.problem.dim_x,
            num_predicates=self.problem.dim_y,
            num_args=2,
            model=self._model,
            freeze_features=self._freeze_features,
        ).to(self.device)

        seed = self.get_param("seed", 0, verbose=not self.eval_only)
        self.seed(seed)

        path_run = self.get_param(
            "path_run", self.path_exp / get_datetime(), verbose=not self.eval_only
        )

        if not self.eval_only:
            path_run.mkdir(parents=True, exist_ok=True)
            if not (path_run / "experiment.pth").exists():
                self.save(path_run / "experiment.pth")

        yield path_run

        self.reset()


class GridworldPredicateExperiment(Experiment):
    """Test all predicates in the 20BN dataset."""

    def __init__(
        self,
        device: torch.device,
        paths,
        num_exp: str,
        eval_only: bool = False,
        num_epochs: int = 5,
        model: str = "resnet",
        mini: bool = False,
        ground_truth: bool = False,
        use_weighted_ce: bool = False,
    ):
        super().__init__(device, paths, num_exp=num_exp, eval_only=eval_only)
        self._num_epochs = num_epochs
        self._model = model
        self._mini = mini
        self._ground_truth = ground_truth
        self._use_weighted_ce = use_weighted_ce

    @staticmethod
    def idx_props(pddl: symbolic.Pddl) -> np.ndarray:
        """Returns the proposition indices for this experiment."""
        return dnf_utils.get_valid_props(pddl).nonzero()[0]

    def __iter__(self):
        variant = "_mini" if self._mini else ""
        self._problem = GridworldPredicateClassification(
            self.pddl,
            self.device,
            path=self.paths.data,
            train_dataset=f"dataset{variant}.hdf5",
            ground_truth=self._ground_truth,
            use_weighted_ce=self._use_weighted_ce,
        )

        lr = self.get_param("AdamW.lr", 0.0001, verbose=not self.eval_only)
        step_size = self.get_param("StepLR.step_size", 4, verbose=not self.eval_only)
        gamma = self.get_param("StepLR.gamma", 0.5, verbose=not self.eval_only)

        self._trainer = Trainer(
            self.problem.train_set,
            self.problem.val_set,
            batch_size=32,
            optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=lr),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            ),
            num_epochs=self._num_epochs,
            shuffle_data=not self.eval_only,
        )

        self._network = PredicateNet(
            dim_input=self.problem.dim_x,
            num_predicates=self.problem.dim_y,
            num_args=3,
            model=self._model,
            freeze_features=True,
        ).to(self.device)

        seed = self.get_param("seed", 0, verbose=not self.eval_only)
        self.seed(seed)

        path_run = self.get_param(
            "path_run", self.path_exp / get_datetime(), verbose=not self.eval_only
        )

        if not self.eval_only:
            path_run.mkdir(parents=True, exist_ok=True)
            self.save(path_run / "experiment.pth")

        yield path_run

        self.reset()


class GridworldHalfPredicateExperiment(Experiment):
    """Test all predicates in the 20BN dataset."""

    def __init__(
        self,
        device: torch.device,
        paths,
        num_exp: str,
        eval_only: bool = False,
        num_epochs: int = 5,
        model: str = "resnet",
        mini: bool = False,
    ):
        super().__init__(device, paths, num_exp=num_exp, eval_only=eval_only)
        self._num_epochs = num_epochs
        self._model = model
        self._mini = mini

    @staticmethod
    def idx_props(pddl: symbolic.Pddl) -> np.ndarray:
        """Returns the proposition indices for this experiment."""
        return dnf_utils.get_valid_props(pddl).nonzero()[0]

    def __iter__(self):
        variant = "_mini" if self._mini else ""
        self._problem = GridworldHalfPredicateClassification(
            self.pddl,
            self.device,
            path=self.paths.data,
            train_dataset=f"dataset{variant}.hdf5",
        )

        lr = self.get_param("AdamW.lr", 0.0001, verbose=not self.eval_only)
        step_size = self.get_param("StepLR.step_size", 4, verbose=not self.eval_only)
        gamma = self.get_param("StepLR.gamma", 0.5, verbose=not self.eval_only)

        self._trainer = Trainer(
            self.problem.train_set,
            self.problem.val_set,
            batch_size=64,
            optimizer=lambda net_params: torch.optim.AdamW(net_params, lr=lr),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            ),
            num_epochs=self._num_epochs,
            shuffle_data=not self.eval_only,
        )

        self._network = PredicateNet(
            dim_input=self.problem.dim_x,
            num_predicates=self.problem.dim_y,
            num_args=3,
            model=self._model,
            freeze_features=True,
        ).to(self.device)

        seed = self.get_param("seed", 0, verbose=not self.eval_only)
        self.seed(seed)

        path_run = self.get_param(
            "path_run", self.path_exp / get_datetime(), verbose=not self.eval_only
        )

        if not self.eval_only:
            path_run.mkdir(parents=True, exist_ok=True)
            self.save(path_run / "experiment.pth")

        yield path_run

        self.reset()
