import pathlib
import pickle
import re

import h5py
import numpy as np

from .constants import COLOR_BG, DIM, DIM_GRID


class nonzeros:
    """Generator for np.nonzero() in a large array."""

    def __init__(self, array):
        self._array = array
        self._N = array.sum()
        self._idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._N

    def __next__(self):
        while self._idx < len(self._array):
            idx = self._idx
            self._idx += 1

            if self._array[idx]:
                return idx

        raise StopIteration()


def filter_data(
    input_file, output_file, data_filter, path="../data/gridworld", tqdm=None
):
    """Applies a filter to generate a subset of the dataset.

    Args:
        input_file (str): Filename of input hdf5.
        output_file (str): Filename of output hdf5.
        data_filter ((action, img_pre, img_post, s_pre, s_post) -> bool): Filter function.
        path (str, optional): Hdf5 directory (default "../data/gridworld").
        tqdm (tqdm, optional): Tqdm iterator.
    """
    print("Calculating dataset size...")
    with h5py.File(pathlib.Path(path) / input_file, "r") as f_in:
        in_actions = f_in["actions"]
        in_img_pre = f_in["img_pre"]
        in_img_post = f_in["img_post"]
        in_s_pre = f_in["s_pre"]
        in_s_post = f_in["s_post"]

        loop = range(len(in_actions))
        if tqdm is not None:
            loop = tqdm(loop)

        # Store which indices pass through the filter
        idx_filtered = np.zeros((len(in_actions),), dtype=bool)
        for i in loop:
            action = in_actions[i]
            img_pre = in_img_pre[i]
            img_post = in_img_post[i]
            s_pre = in_s_pre[i]
            s_post = in_s_post[i]

            idx_filtered[i] = data_filter(action, img_pre, img_post, s_pre, s_post)

        # Compute size of filtered dataset
        N = idx_filtered.sum()

        print("Writing dataset...")
        with h5py.File(pathlib.Path(path) / output_file, "w") as f:
            out_actions = f.create_dataset(
                "actions", (N,) + in_actions.shape[1:], in_actions.dtype
            )
            out_img_pre = f.create_dataset(
                "img_pre", (N,) + in_img_pre.shape[1:], in_img_pre.dtype
            )
            out_img_post = f.create_dataset(
                "img_post", (N,) + in_img_post.shape[1:], in_img_post.dtype
            )
            out_s_pre = f.create_dataset(
                "s_pre", (N,) + in_s_pre.shape[1:], in_s_pre.dtype
            )
            out_s_post = f.create_dataset(
                "s_post", (N,) + in_s_post.shape[1:], in_s_post.dtype
            )

            loop = nonzeros(idx_filtered)
            if tqdm is not None:
                loop = tqdm(loop)

            for idx_out, idx_in in enumerate(loop):
                out_actions[idx_out] = in_actions[idx_in]
                out_img_pre[idx_out] = in_img_pre[idx_in]
                out_img_post[idx_out] = in_img_post[idx_in]
                out_s_pre[idx_out] = in_s_pre[idx_in]
                out_s_post[idx_out] = in_s_post[idx_in]


class LogDatabase:
    """Database of logs and variables indexed by keys.

    Args:
        path (str, optional): Output directory (default "../data/gridworld").
    """

    def __init__(self, path="../data/gridworld"):
        self._path = pathlib.Path(path)
        self._key = None
        self._logs = {}
        self._variables = {}
        self.path_images.mkdir(parents=True, exist_ok=True)
        self.path_logs.mkdir(parents=True, exist_ok=True)
        self.path_variables.mkdir(parents=True, exist_ok=True)

    @property
    def path(self):
        """Output directory.

        :type: pathlib.Path
        """
        return self._path

    @property
    def path_images(self):
        """Images directory.

        :type: pathlib.Path
        """
        return self.path / "images"

    @property
    def path_logs(self):
        """Logs directory.

        :type: pathlib.Path
        """
        return self.path / "logs"

    @property
    def path_variables(self):
        """Variables directory.

        :type: pathlib.Path
        """
        return self.path / "variables"

    @property
    def key(self):
        """Choose the log index (any hashable type)."""
        return self._key

    @key.setter
    def key(self, key):
        self._key = key
        if key not in self._logs:
            self._logs[key] = ""
            self._variables[key] = {}

    def write(self, message, stdout=False):
        """Write a line to the current log.

        Args:
            message (str): Line to write.
            stdout (bool, optional): Print line to stdout (default False).
        """
        self._logs[self.key] += str(message) + "\n"
        if stdout:
            print(message)

    def save(self, variables):
        """Append dictionary of variables for saving.

        Args:
            variables (dict(str, obj)): Dictionary of variable to append.
        """
        import copy

        self._variables[self.key].update(copy.deepcopy(variables))

    def commit(self):
        """Write variables and log to disk."""
        # Write variables
        with open(self.path_variables / f"{self.key}.pkl", "wb") as f:
            pickle.dump(self._variables[self.key], f)

        # Write log
        with open(self.path_logs / f"{self.key}.log", "w") as f:
            f.write(self._logs[self.key])

        # Clear memory
        self._logs.pop(self.key)
        self._variables.pop(self.key)

    def load(self, key, verbose=True):
        """Load variables and print log.

        Args:
            key (hashable): Log to load.
            verbose (bool, optional): Whether to print the log (default True).
        Returns:
            variables
        """
        # Load variables
        with open(self.path / "variables" / f"{key}.pkl", "rb") as f:
            variables = pickle.load(f)

        # Print log
        if verbose:
            with open(self.path / "logs" / f"{key}.log") as f:
                print(f.read())

        return variables

    def publish_dataset(self, filename="dataset.hdf5", tqdm=None):
        """Save the dataset in hdf5 format.

        Args:
            filename (str): Filename of hdf5 file.
            tqdm (tqdm, optional): Tqdm iterator.
        """
        from PIL import Image

        keys = self._get_keys()
        num_keys = len(keys)

        with h5py.File(self.path / filename, "w") as f:
            dset_keys = f.create_dataset("keys", (num_keys,), dtype=int)
            dset_actions = f.create_dataset(
                "actions", (num_keys,), dtype=h5py.string_dtype(encoding="utf-8")
            )
            dset_img_pre = f.create_dataset(
                "img_pre", (num_keys, DIM[0], DIM[1], 3), dtype=np.uint8
            )
            dset_img_post = f.create_dataset(
                "img_post", (num_keys, DIM[0], DIM[1], 3), dtype=np.uint8
            )
            len_state = self.load(self._get_keys()[0], verbose=False)["s_pre"].shape[0]
            dset_s_pre = f.create_dataset("s_pre", (num_keys, len_state), dtype=bool)
            dset_s_post = f.create_dataset("s_post", (num_keys, len_state), dtype=bool)
            NUM_OBJECTS = 9
            dset_boxes_pre = f.create_dataset(
                "boxes_pre", (num_keys, NUM_OBJECTS, 4), dtype=np.float32
            )
            dset_boxes_post = f.create_dataset(
                "boxes_post", (num_keys, NUM_OBJECTS, 4), dtype=np.float32
            )

            if tqdm is None:
                loop = enumerate(self._get_keys())
            else:
                loop = enumerate(tqdm(self._get_keys()))

            for i, key in loop:
                with open(self.path_images / f"{key}_pre.png", "rb") as img:
                    img_pre = np.asarray(Image.open(img))[:, :, :3]
                with open(self.path_images / f"{key}_post.png", "rb") as img:
                    img_post = np.asarray(Image.open(img))[:, :, :3]
                variables = self.load(key, verbose=False)
                action_call = variables["action_call"]
                if not "s_pre" in variables:
                    print(i, key, action_call)
                s_pre = variables["s_pre"]
                s_post = variables["s_post"]
                boxes_pre = variables["boxes_pre"]
                boxes_post = variables["boxes_post"]

                dset_keys[i] = key
                dset_actions[i] = action_call
                dset_img_pre[i] = img_pre
                dset_img_post[i] = img_post
                dset_s_pre[i] = s_pre
                dset_s_post[i] = s_post
                dset_boxes_pre[i] = boxes_pre
                dset_boxes_post[i] = boxes_post

    def _get_keys(self):
        # Load all saved keys from images directory.
        keys = []
        for filepath in (self.path_images).glob("*_pre.png"):
            filename = filepath.name
            matches = re.match(r"(\d+)_pre.png", filename)
            if not matches:
                raise RuntimeError(f"Invalid file in dataset: {filename}")
            key = int(matches[1])
            keys.append(key)
        keys.sort()
        return keys
