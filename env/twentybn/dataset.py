import pathlib
import re
from typing import Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np


class ActionLabel:
    """20BN action labels."""

    def __init__(self, file: str, id_action: int, template: str):
        self._file = file
        self._id_action = id_action
        self._template = template
        self._videos: Optional[np.ndarray] = None

    @property
    def id_action(self) -> int:
        """Action id."""
        return self._id_action

    @property
    def template(self) -> str:
        """Action template."""
        return self._template

    def to_string(self, args: List[str]) -> str:
        """Instantiates the action template with the given arguments.

        Args:
            args: List of argument names.
        Returns:
            String of the format 'Pushing [apple] with [pen]'.
        """
        args = [f"[{arg}]" for arg in args]
        template = re.sub(r"\[[A-Za-z ]+\]", "{}", self.template)
        return template.format(*args)

    @property
    def videos(self) -> np.ndarray:
        """Loads and caches the list of video ids for this action."""
        if self._videos is None:
            with h5py.File(self._file, "r") as f:
                self._videos = np.array(f["actions"][str(self.id_action)]["videos"])

        return self._videos

    def __repr__(self) -> str:
        return f"""ActionLabel {{
    .id_action: {self.id_action}
    .template: "{self.template}"
    .to_string([args, ...]): str
    .videos: [video ids, ...]
}}"""


class Labels:
    pass


class VideoLabel:
    """20BN video labels."""

    def __init__(self, labels: Labels, id_video: int):
        self._labels = labels
        self._id_video = id_video

        # Load data.
        with h5py.File(self._labels._file, "r") as f:
            grp = f["videos"][str(id_video)]
            self._id_action = int(grp.attrs["id_action"])
            self._objects = list(grp.attrs["objects"])
            self._keyframes = np.array(grp["keyframes"])
            self._pre = np.array(grp["pre"])
            self._post = np.array(grp["post"])
            self._boxes = np.array(grp["boxes"])

    @property
    def id_video(self) -> int:
        """Video id."""
        return self._id_video

    @property
    def id_action(self) -> int:
        """Action id."""
        return self._id_action

    @property
    def action(self) -> ActionLabel:
        """Action label."""
        return self._labels.actions[self.id_action]

    @property
    def action_name(self) -> str:
        """Action template instantiated with video objects."""
        return self.action.to_string(self.objects)

    @property
    def objects(self) -> List[str]:
        """List of object names."""
        return self._objects

    @property
    def keyframes(self) -> np.ndarray:
        """List of keyframe numbers."""
        return self._keyframes

    @property
    def pre(self) -> np.ndarray:
        """List of pre-condition timesteps."""
        return self._pre

    @property
    def post(self) -> np.ndarray:
        """List of post-condition timesteps."""
        return self._post

    @property
    def boxes(self) -> np.ndarray:
        """[T, 1+O, 4] (num_keyframes, hand/num_objects, x1/y1/x2/y2) bounding boxes."""
        return self._boxes

    def __repr__(self) -> str:
        return f"""VideoLabel {{
    .id_video: {self.id_video}
    .id_action: {self.id_action}
    .action: <ActionLabel>
    .action_name: <ActionLabel.to_string(objects)>
    .objects: {self.objects}
    .keyframes: {self.keyframes}
    .boxes: {self.boxes}
}}"""


class VideoLabels:
    """Iterable map of 20BN video labels indexed by video id."""

    class VideoLabelsIterator:
        def __init__(self, labels: Labels, video_ids: np.ndarray):
            self._idx = -1
            self._labels = labels
            self._video_ids = video_ids

        def __next__(self):
            self._idx += 1
            id_video = self._video_ids[self._idx]
            return VideoLabel(self._labels, id_video)

        def __iter__(self):
            return self

    def __init__(self, labels: Labels):
        self._labels = labels
        self._video_ids: Optional[List[int]] = None
        self._cache: Tuple[Optional[int], Optional[VideoLabel]] = (
            None,
            None,
        )

    def __getitem__(self, id_video: int) -> VideoLabel:
        """Loads and caches one label for the given video."""
        if self._cache[0] == id_video:
            return self._cache[1]
        self._cache = (id_video, VideoLabel(self._labels, id_video))
        return self._cache[1]

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> np.ndarray:
        """List of video ids."""
        if self._video_ids is None:
            with h5py.File(self._labels._file, "r") as f:
                self._video_ids = np.array(f["video_ids"])
        return self._video_ids

    def values(self) -> Iterable[VideoLabel]:
        """List of video labels."""
        return VideoLabels.VideoLabelsIterator(self._labels, self.keys())

    def items(self) -> Iterable[Tuple[int, VideoLabel]]:
        """List of (id_video, label) pairs."""
        it = VideoLabels.VideoLabelsIterator(self._labels, self.keys())
        for label in it:
            yield label.id_video, label

    def __repr__(self) -> str:
        return """VideoLabels: {id_video: VideoLabel}"""


class Labels:
    """Helper class to load 20BN labels to balance memory usage and IO reads.

    h5py = {
        "actions": {
            "id_action": {
                attrs: {
                    "id_action": int,
                    "template": utf8,
                },
                "videos": [V] (num_videos) uint32,
            }
        },
        "videos": {
            "id_video": {
                attrs: {
                    "id_video": int,
                    "id_action": int,
                    "objects": [O] (num_objects) utf8,
                },
                "keyframes": [T] (num_keyframes) uint32 (guaranteed to exist in video),
                "pre": [T_pre] (num_pre_keyframes) uint32.
                "post": [T_post] (num_post_keyframes) uint32.
                "boxes": [T, 1 + O, 4] (num_keyframes, hand/num_objects, x1/y1/x2/y2) float32,
            }
        },
        "video_ids": [V] (num_videos) uint32,
    }

    Args:
        file: Path of hdf5 file.
    """

    def __init__(
        self,
        file: Union[str, pathlib.Path] = str(
            (
                pathlib.Path(__file__).parent / "../../data/twentybn/labels.hdf5"
            ).absolute()
        ),
    ):
        self._file = str(file)
        self._actions: Optional[List[ActionLabel]] = None
        self._videos = VideoLabels(self)

    @property
    def actions(self) -> List[ActionLabel]:
        """Loads and caches a list of ActionLabels indexed by action id."""
        if self._actions is None:
            with h5py.File(self._file, "r") as f:
                grp = f["actions"]
                self._actions = [None] * len(grp)
                for dset in grp.values():
                    id_action = dset.attrs["id_action"]
                    template = dset.attrs["template"]
                    self._actions[id_action] = ActionLabel(
                        self._file, id_action, template
                    )

        return self._actions

    @property
    def videos(self) -> VideoLabels:
        """Returns the video labels."""
        return self._videos

    def __repr__(self) -> str:
        return """Label {
    .actions: [<ActionLabel>, ...]
    .videos: <VideoLabels>
}"""
