import re
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import PIL
import PIL.ImageFont  # type: ignore
import PIL.ImageDraw  # type: ignore
import seaborn as sns  # type: ignore
import symbolic  # type: ignore

from gpred import video_utils, dnf_utils
from env import twentybn


def bbox_mask(
    width: int,
    height: int,
    bbox: Optional[
        Tuple[Tuple[float, float], Tuple[float, float]]
    ],
) -> np.ndarray:
    """Creates a bounding box image mask.

    Args:
        width: Image width.
        height: Image height.
        bbox: Optional ((x1, y1), (x2, y2)) bounding box.
    Returns:
        [H, W] array with ones inside the box and zeros everywhere else.
    """
    img = PIL.Image.new("L", (width, height))
    draw = PIL.ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        draw.rectangle((x1, y1, x2, y2), fill=(1,))
    return np.array(img)


# TODO: Delete
def bbox_masks(
    width: int, height: int, boxes: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Creates bounding box image masks from a list of bounding boxes.

    Args:
        width: Image width.
        height: Image height.
        boxes: [..., num_boxes, 4] array of boxes (x1/y1/x2/y2).
        mask: Optional [..., num_boxes, H, W] float32 array to populate.
    Returns:
        [..., num_boxes, H, W] float32 array with ones inside the box and zeros everywhere else.
    """
    print("twentybn.utils.bbox_masks(): SWITCH TO dnf_utils!")
    dim = boxes.shape[:-1]

    if mask is None:
        mask = np.zeros((*dim, height, width), dtype=np.float32)

    for sub in np.ndindex(*boxes.shape[:-1]):
        box = boxes[sub]
        if (box == -float("inf")).all():
            continue

        box = (box + 0.5).astype(int)
        sub += (slice(box[1], box[3]), slice(box[0], box[2]))
        mask[sub] = 1

    return mask


def create_bbox_masks(
    id_video: int,
    dimensions: Tuple[int, int],
    video_labels: List,
    keyframes: Optional[List[int]] = None,
    return_masks: bool = True,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Create bounding box mask image.

    Args:
        id_video: Video ID.
        dimensions: (height, width).
        video_labels: Bounding box labels from Something-Else dataset.
        keyframes: List of frames to extract, or None to extract all.
        return_masks: Return image masks.
    Returns:
        (mask [T, 4, H, W], boxes [T, 5, 5]) pair.
    """
    NUM_BOXES = 5

    bboxes = video_labels[id_video]["frames"]
    args = video_labels[id_video]["placeholders"]

    height, width = dimensions

    if keyframes is None:
        keyframes = bboxes.keys()

    boxes = np.zeros((len(keyframes), NUM_BOXES, 5), dtype=np.float32)
    boxes[...] = np.nan
    if return_masks:
        masks: Optional[np.ndarray] = np.zeros(
            (len(keyframes), NUM_BOXES - 1, height, width), dtype=np.float32
        )
    else:
        masks = None

    for t, idx_frame in enumerate(keyframes):
        for id_object, bbox in bboxes[idx_frame].items():
            # Skip objects not part of action specification.
            if id_object != "hand" and int(id_object) >= len(args):
                continue

            idx_object = object_id_to_idx(id_object)

            # Write bounding box.
            boxes[t, 1 + idx_object, 1:] = np.array(bbox).flatten()

            # Write bounding box mask.
            if return_masks:
                masks[t, idx_object, :, :] = bbox_mask(width, height, bbox)  # type: ignore

        boxes[t, 0, 1:3] = np.nanmin(boxes[t, 1:, 1:3], axis=0)
        boxes[t, 0, 3:] = np.nanmax(boxes[t, 1:, 3:], axis=0)
        boxes[t, :, 0] = idx_frame

    # Convert nans to -inf.
    boxes[np.isnan(boxes)] = -float("inf")

    return masks, boxes


def split_bbox_args(
    pddl: symbolic.Pddl,
    video_label: twentybn.dataset.VideoLabel,
    timesteps: Optional[List[int]] = None,
) -> np.ndarray:
    """Create bounding box mask image.

    Vectorized version of `dnf_utils.bbox_arg_combos()`.

    Args:
        pddl: Pddl instance.
        video_label: 20BN video label.
        timesteps: List of timesteps to extract, or None to extract all.
    Returns:
        Indexed boxes as a [T, 16, 3, 4] (num_keyframes, num_arg_combos,
        roi/arg_a/arg_b, x1/y1/x2/y2) float array.
    """

    # [T, 4, 4] (num_keyframes, hand/a/b/c, x1/y1/x2/y2)
    T = video_label.boxes.shape[0] if timesteps is None else len(timesteps)
    boxes = video_label.boxes if timesteps is None else video_label.boxes[timesteps]
    boxes = np.array(boxes)
    boxes[boxes < 0] = np.nan

    # [T, 5, 4] (num_keyframes, hand/a/b/c/nan, x1/y1/x2/y2)
    # NaN channel gets selected by parameter to arg combo map.
    O = len(pddl.objects)  # noqa: E741
    param_boxes = np.concatenate(
        (boxes, np.full((T, O + 1 - boxes.shape[1], 4), np.nan, dtype=np.float32)), axis=1
    )

    # def get_parameter_boxes(
    #     pddl: symbolic.Pddl,
    #     video_label: twentybn.dataset.VideoLabel,
    #     keyframes: List[int],
    # ) -> np.ndarray:
    #     [>Flattens the Something-Else bounding boxes into an array of parameter
    #     boxes.

    #     Args:
    #         pddl: Pddl instance.
    #         video_label: 20BN video label.
    #         keyframes: List of frames to extract.
    #     Returns:
    #         [T, 5, 4] (num_keyframes, hand/a/b/c/nan, x1/y1/x2/y2). NaN channel
    #         gets selected by parameter to arg combo map.
    #     [>
    #     T = len(keyframes)
    #     NUM_PARAMS = len(pddl.objects)

    #     bboxes = video_labels[id_video]["frames"]

    #     # [T, 5, 4] (num_keyframes, hand/a/b/c/nan, x1/y1/x2/y1)
    #     param_boxes = np.full((T, NUM_PARAMS + 1, 4), np.nan, dtype=np.float32)
    #     for t, idx_frame in enumerate(keyframes):
    #         # Iterate over all objects in the frame.
    #         for id_object, bbox in bboxes[idx_frame].items():
    #             try:
    #                 idx_object = object_id_to_idx(id_object)

    #                 # [2, 2] -> [4] (x1/y1/x2/y2)
    #                 param_boxes[t, idx_object, :] = np.array(bbox).flatten()
    #             except:
    #                 continue

    #     return param_boxes

    # if keyframes is None:
    #     keyframes = video_label.keyframes

    # # [T, 5, 4] (num_keyframes, hand/a/b/c, x1/y1/x2/y2)
    # param_boxes = get_parameter_boxes(pddl, video_label, keyframes)

    # [16, 2] (num_arg_combos, num_args)
    idx_param_to_arg_combo = dnf_utils.param_to_arg_combo_indices(pddl)

    # [T, MC, M + 1, 4] (num_keyframes, arg_combos, roi/a/b, x1/y1/x2/y2)
    MC, M = idx_param_to_arg_combo.shape
    boxes = np.empty((T, MC, M + 1, 4), np.float32)

    # [T, 5, 4] -> [T, MC, M, 4] (num_keyframes, arg_combos, num_args, x1/y1/x2/y2)
    boxes[:, :, 1:, :] = param_boxes[:, idx_param_to_arg_combo, :]

    # [T, MC, M, 2] -> [T, MC, 2] (num_keyframes, arg_combos, x/y)
    boxes[:, :, 0, :2] = np.nanmin(boxes[:, :, 1:, :2], axis=2)  # xy1
    boxes[:, :, 0, 2:] = np.nanmax(boxes[:, :, 1:, 2:], axis=2)  # xy2

    # Convert nans to -inf.
    boxes[np.isnan(boxes)] = -float("inf")

    return boxes


def action_label(
    id_video: int, video_labels: Dict, action_labels: Dict
) -> str:
    """Constructs the action label for the given video.

    Args:
        id_video: Video id.
        video_labels: Something-Else labels.
        action_labels: 20bn labels.
    """
    id_action = video_labels[id_video]["id_action"]
    placeholders = video_labels[id_video]["placeholders"]
    placeholders = [f"[{placeholder}]" for placeholder in placeholders]
    template = action_labels[id_action]["template"]
    template = re.sub(r"\[[A-Za-z ]+\]", "{}", template)
    return template.format(*placeholders)


def object_name(id_object: str) -> str:
    """Mapping from object id (hand/0/1/2) to object name (hand/a/b/c)."""
    if id_object in ("hand", "roi"):
        return id_object
    names = "abc"
    return names[int(id_object)]


def object_id(name_object: str) -> str:
    """Mapping from object name (hand/a/b/c) to object id (hand/0/1/2)."""
    if name_object in ("hand", "roi"):
        return name_object
    return str(ord(name_object) - ord("a"))


def object_id_to_idx(id_object: str) -> int:
    """Mapping from object id to idx, where hand = 0."""
    if id_object == "hand":
        return 0
    if id_object == "roi":
        return -1
    return int(id_object) + 1


def object_idx_to_id(idx_object: int) -> str:
    """Mapping from object idx to id, where 0 = hand."""
    if idx_object == 0:
        return "hand"
    if idx_object == -1:
        return "roi"
    return str(idx_object - 1)


def object_id_to_mask_channel(id_object: str) -> int:
    """Mapping from object id to 7-channel mask number, where first 3 channels are rgb."""
    if id_object == "roi":
        raise ValueError("id_object for a mask channel cannot be 'roi'.")
    return object_id_to_idx(id_object) + 3


def mask_channel_to_object_id(idx_channel: int) -> str:
    """Mapping from 7-channel mask number to object id, where first 3 channels arg rgb."""
    return object_idx_to_id(idx_channel - 3)


def object_id_to_bbox_row(id_object: str) -> int:
    """Mapping from object id to 5-channel bbox row, where first row is image roi."""
    return object_id_to_idx(id_object) + 1


def bbox_row_to_id_object(bbox_row: int) -> str:
    """Mapping from 5-channel box row to object id, where first row is image roi."""
    return object_idx_to_id(bbox_row - 1)


def image_overlay_box_mask(
    img: np.ndarray,
    img_boxes: np.ndarray,
) -> np.ndarray:
    """Overlay bounding boxes mask onto the image.

    Args:
        img: [H, W, 3] uint8 array.
        img_boxes = [4, H, W] float32 array.
    Returns:
        [H, W, 3] uint8 array.
    """
    OPACITY = 0.5

    # Convert frame to RGBA.
    img_rgba = video_utils.rgb_to_rgba(img)
    img = PIL.Image.fromarray(img_rgba, "RGBA")

    for idx_object in range(img_boxes.shape[0]):
        # Get channel color
        id_object = object_idx_to_id(idx_object)
        rgb = get_bbox_color(id_object)
        rgba = np.array([*rgb, int(OPACITY * 255)], dtype=np.uint8)

        # Draw the mask for the current object channel.
        # [H, W] => [H, W, 4]
        img_mask = (
            img_boxes[idx_object][:, :, None].astype(np.uint8) * rgba[None, None, :]
        )

        # Create overlay drawing context.
        overlay = PIL.Image.fromarray(img_mask, "RGBA")

        # Composite the overlay.
        img = PIL.Image.alpha_composite(img, overlay)

    img = np.array(img)[:, :, :3]

    return img


def visualize_cnn_input(
    x: np.ndarray,
    image_distribution: Tuple[
        Tuple[int, int, int], Tuple[int, int, int]
    ],
) -> np.ndarray:
    """Convert the cnn input into an RGB image overlaid with the bounding box
    masks.

    Args:
        x: [7, H, W] float32 array.
    Returns:
        [H, W, 3] uint8 array.
    """
    img_rgb = video_utils.image_cnn_to_rgb(x[:3], image_distribution)
    img_rgb = image_overlay_box_mask(img_rgb, x[3:])
    return img_rgb


def get_bbox_color(id_object: str) -> Tuple[int, int, int]:
    """Gets the bounding box color assigned to the object.

    Args:
        id_object: Object id (hand/0/1/2/3).
    Returns:
        RGB color (0-255).
    """
    idx_object = object_id_to_idx(id_object)
    float_color = sns.color_palette()[idx_object]
    int_color = (
        int(255 * float_color[0] + 0.5),
        int(255 * float_color[1] + 0.5),
        int(255 * float_color[2] + 0.5),
    )
    return int_color


def image_overlay_boxes(
    img: np.ndarray,
    frame_boxes: Optional[np.ndarray] = None,
    bbox_labels: Optional[Dict] = None,
) -> np.ndarray:
    """Overlay bounding boxes onto the image.

    Uses either the boxes format in the hdf5 dataset or the dict format from Something-Else.

    Args:
        img: [H, W, 3] uint8 array.
        frame_boxes: [5, 4] array.
        bbox_labels: Something-Else labels for the given frame.
    Returns:
        Modified img as a [H, W, 3] array.
    """

    def draw_labeled_box(id_object: str, box: np.ndarray, draw: PIL.ImageDraw.Draw):
        """Draws a labeled bounding box for the object.

        Args:
            id_object: Object id (roi/hand/0/1/2).
            box: Top-left and bottom-right corners (x1, y1, x2, y2).
            draw: Drawing context.
        """
        OPACITY = 0.75
        FONT = PIL.ImageFont.truetype("arial.ttf", 15)

        # Create color.
        rgb = get_bbox_color(id_object)
        rgba = (*rgb, int(OPACITY * 255))

        # Draw bounding box.
        box = (box + 0.5).astype(int)
        draw.rectangle(box.tolist(), outline=rgba, width=2)

        # Draw label.
        if id_object == "roi":
            # Label above.
            draw.rectangle(
                np.array(
                    [box[0], box[1] - FONT.size - 3, box[2], box[1]], dtype=int
                ).tolist(),
                fill=rgba,
            )
            draw.text(
                (box[0] + 6, box[1] - FONT.size - 3), object_name(id_object), font=FONT
            )
        else:
            # Label below.
            draw.rectangle(
                np.array(
                    [box[0], box[3], box[2], box[3] + FONT.size + 3], dtype=int
                ).tolist(),
                fill=rgba,
            )
            draw.text((box[0] + 6, box[3]), object_name(id_object), font=FONT)

    def enumerate_boxes(
        frame_boxes: Optional[np.ndarray],
        bbox_labels: Optional[Dict],
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Enumerate over all objects in the given boxes list.

        Uses either the boxes format in the hdf5 dataset or the dict format from Something-Else.

        Args:
            frame_boxes: [5, 4] array.
            bbox_labels: Something-Else labels.
        Returns:
            Generator that yields (id_object, box [4,]) tuples.
        """
        # Convert bbox format.
        if frame_boxes is not None:
            for bbox_row in range(frame_boxes.shape[0]):
                # Non-existent boxes are set to -inf.
                if frame_boxes[bbox_row, 0] < -1:
                    continue

                id_object = bbox_row_to_id_object(bbox_row)
                box = frame_boxes[bbox_row]

                yield id_object, box

        # Convert frame label format.
        if bbox_labels is not None:
            for id_object, box in bbox_labels.items():
                box = np.array([*box[0], *box[1]])

                yield id_object, box

    # Convert frame to RGBA.
    frame_rgba = video_utils.rgb_to_rgba(img)
    img = PIL.Image.fromarray(frame_rgba, "RGBA")

    # Create overlay drawing context.
    overlay = PIL.Image.new("RGBA", img.size, 0)
    draw = PIL.ImageDraw.Draw(overlay)

    # Draw labeled boxes for each object.
    for id_object, box in enumerate_boxes(frame_boxes, bbox_labels):
        draw_labeled_box(id_object, box, draw)

    # Composite the overlay.
    img = PIL.Image.alpha_composite(img, overlay)
    img = np.array(img)[:, :, :3]

    return img


def video_overlay_boxes(
    video_frames: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    frame_labels: Optional[Dict] = None,
) -> np.ndarray:
    """Overlay bounding boxes onto the video.

    Uses either the boxes format in the hdf5 dataset or the dict format from Something-Else.

    Args:
        video_frames: [T, H, W, 3] uint8 array.
        boxes: [T, 5, 4] array.
        frame_labels: Something-Else labels.
    Returns:
        Modified video_frames as a [T, H, W, 3] array.
    """
    # Convert boxes format.
    if boxes is not None:
        for t in range(video_frames.shape[0]):
            video_frames[t] = image_overlay_boxes(video_frames[t], boxes[t], None)

    # Convert frame labels format.
    if frame_labels is not None:
        keyframes = list(frame_labels.keys())
        for t in range(video_frames.shape[0]):
            idx_frame = keyframes[t]
            video_frames[t] = image_overlay_boxes(
                video_frames[t], None, frame_labels[idx_frame]
            )

    return video_frames


def image_overlay_propositions(
    img: np.ndarray,
    props: List[str],
    p_props: np.ndarray,
    pddl: Optional[symbolic.Pddl],
    void_args: Optional[Set[str]],
) -> np.ndarray:
    """Overlay bounding boxes onto the image.

    Uses either the boxes format in the hdf5 dataset or the dict format from Something-Else.

    Args:
        img: [H, W, 3] uint8 array.
        props: Proposition names.
        p_props: [P] float array of predicted proposition probabilities.
        pddl: Pddl instance.
        void_args: Which arguments don't exist.
    Returns:
        Modified image as a [H, W, 3] array.
    """
    FONT = PIL.ImageFont.truetype("arial.ttf", 15)
    WIDTH = 150
    HEIGHT = FONT.size + 3
    RGB_TRUE = np.array([0, 255, 0, 100], dtype=np.uint8)
    RGB_FALSE = np.array([255, 0, 0, 100], dtype=np.uint8)

    # Convert frame to RGBA.
    frame_rgba = video_utils.rgb_to_rgba(img)
    img = PIL.Image.fromarray(frame_rgba, "RGBA")

    # Create overlay drawing context.
    overlay = PIL.Image.new("RGBA", img.size, 0)
    draw = PIL.ImageDraw.Draw(overlay)

    # Get valid props
    idx_valid = dnf_utils.get_valid_props(pddl) if pddl is not None else None

    # Draw proposition label.
    row = 0
    for i, prop in enumerate(props):
        p_prop = p_props[i]
        bg_color = tuple(
            ((p_prop * RGB_TRUE + (1 - p_prop) * RGB_FALSE) + 0.5).astype(np.uint8)
        )
        if pddl is not None:
            idx_prop = pddl.state_index.get_proposition_index(prop)
            if not idx_valid[idx_prop]:  # type: ignore
                continue
        if void_args is not None:
            args = dnf_utils.parse_args(prop)
            if void_args.intersection(set(args)):
                continue

        draw.rectangle((0, row * HEIGHT, WIDTH, (row + 1) * HEIGHT), fill=bg_color)
        draw.text((6, row * HEIGHT), f"{prop}: {p_prop:.2f}", font=FONT)

        row += 1

    # Composite the overlay.
    img = PIL.Image.alpha_composite(img, overlay)
    img = np.array(img)[:, :, :3]

    return img


def video_overlay_propositions(
    video_frames: np.ndarray,
    props: List[str],
    p_predict: np.ndarray,
    pddl: Optional[symbolic.Pddl],
    void_args: Optional[Set[str]],
) -> np.ndarray:
    """Overlay bounding boxes onto the video.

    Uses either the boxes format in the hdf5 dataset or the dict format from Something-Else.

    Args:
        video_frames: [T, H, W, 3] uint8 array.
        props: Proposition names.
        p_predict: Predicted proposition probabilities [T,P].
        pddl: Pddl instance.
        void_args: Which arguments don't exist.
    Returns:
        Modified video as a [T, H, W, 3] array.
    """
    for t in range(video_frames.shape[0]):
        video_frames[t] = image_overlay_propositions(
            video_frames[t], props, p_predict[t], pddl, void_args
        )
    return video_frames


def image_append_propositions(
    img: np.ndarray,
    props: List[str],
    p_props: np.ndarray,
    pddl: symbolic.Pddl,
    void_args: Optional[Set[str]],
) -> np.ndarray:
    """Draws proposition predictions to the left of the image.

    Args:
        img: [H, W, 3] uint8 array.
        props: Proposition names.
        p_props: [P] float array of predicted proposition probabilities.
        pddl: Pddl instance.
        void_args: Which arguments don't exist.
    Returns:
        Modified image as a [H, W', 3] array.
    """
    FONT = PIL.ImageFont.truetype("arial.ttf", 11)
    FONT_PROB = PIL.ImageFont.truetype("arial.ttf", 8)
    HEIGHT_LINE = FONT.size + 3
    PADDING_TEXT_X = 5

    NUM_ROWS = 14
    WIDTH_ARGS = img.shape[0] - NUM_ROWS * HEIGHT_LINE

    ARGS = ["a", "b", "c", "hand"]
    NUM_UNARY_ARGS = len(ARGS)
    NUM_BINARY_ARGS = (NUM_UNARY_ARGS - 1) * NUM_UNARY_ARGS

    NUM_UNARY_BINS = 2
    # NUM_BINARY_BINS = 1
    NUM_BINS = 3

    def map_color(x: float) -> Tuple[int, int, int]:
        """Gets the color for the given float (range 0-1)."""
        assert x <= 1.0 and x >= 0.0
        cmap = sns.diverging_palette(10, 130, n=100)
        # cmap = sns.cubehelix_palette(100, dark=0.33, light=1)
        i = int(x * 99 + 0.5)
        rgb = [int(c * 255 + 0.5) for c in cmap[i]]
        return (rgb[0], rgb[1], rgb[2])

    def draw_text(
        img: PIL.Image,
        text: str,
        pos: Tuple[int, int],
        right_align: bool = False,
    ):
        draw = PIL.ImageDraw.Draw(img)
        dim_txt = FONT.getsize(text)
        x, y = pos
        if right_align:
            x -= dim_txt[0]
        draw.text((x, y), text, font=FONT, fill=(0, 0, 0, 255))

    def draw_vertical_text(
        img: PIL.Image,
        text: str,
        pos: Tuple[int, int],
        right_align: bool = False,
    ):
        dim_txt = FONT.getsize(text)
        img_txt = PIL.Image.new("RGBA", dim_txt, (255, 255, 255, 0))
        draw_txt = PIL.ImageDraw.Draw(img_txt)
        draw_txt.text((0, 0), text, font=FONT, fill=(0, 0, 0, 255))
        img_txt = img_txt.rotate(90, expand=True)
        x, y = pos
        if not right_align:
            y -= dim_txt[0]
        img.paste(img_txt, (x, y), mask=img_txt)

    def compute_predicate_positions(
        pddl: symbolic.Pddl,
    ) -> Tuple[Callable[[str], Tuple[int, int]], List[int]]:
        """Computes the positions of all predicates.

        Args:
            pddl: Pddl instance.
        Returns:
            2-tuple (map from predicates to xy positions, x position of each section.
        """
        # Divide predicates into 3 bins - 2 unary, 1 binary.
        pred_bins: List[List[str]] = [[] for _ in range(NUM_BINS)]
        idx_unary = 0
        idx_binary = NUM_UNARY_BINS
        for pred in pddl.predicates:
            assert len(pred.parameters) <= 2
            if len(pred.parameters) == 1:
                if len(pred_bins[idx_unary]) >= NUM_ROWS:
                    idx_unary += 1
                pred_bin = pred_bins[idx_unary]
            else:
                pred_bin = pred_bins[idx_binary]
            pred_bin.append(pred.name)

        # Find max pred label widths.
        bin_label_widths = [0] * NUM_BINS
        for i, pred_bin in enumerate(pred_bins):
            for pred in pred_bin:
                width_pred = FONT.getsize(pred)[0]
                if width_pred > bin_label_widths[i]:
                    bin_label_widths[i] = width_pred

        # Compute x positions of all bins.
        x_bins = [bin_label_widths[0] + 2 * PADDING_TEXT_X]
        x_bins.append(
            x_bins[0]
            + NUM_UNARY_ARGS * HEIGHT_LINE
            + bin_label_widths[1]
            + 2 * PADDING_TEXT_X
        )
        x_bins.append(
            x_bins[1]
            + NUM_UNARY_ARGS * HEIGHT_LINE
            + bin_label_widths[2]
            + 2 * PADDING_TEXT_X
        )
        x_bins.append(x_bins[2] + NUM_BINARY_ARGS * HEIGHT_LINE)

        # Create map from predicates to x positions.
        xy_preds = {}
        for i, pred_bin in enumerate(pred_bins):
            for idx_pred, pred in enumerate(pred_bin):
                xy_preds[pred] = (x_bins[i], WIDTH_ARGS + idx_pred * HEIGHT_LINE)

        def xy_pred(pred: str) -> Tuple[int, int]:
            return xy_preds[pred]

        return xy_pred, x_bins

    def dx_args(args: List[str]) -> int:
        args = [arg if arg != "hand" else "d" for arg in args]

        if len(args) == 1:
            idx_args = ord(args[0]) - ord("a")
        else:
            idx_a = ord(args[0]) - ord("a")
            idx_b = ord(args[1]) - ord("a")
            if idx_b > idx_a:
                idx_b -= 1
            idx_args = 3 * idx_a + idx_b

        return idx_args * HEIGHT_LINE

    def compute_proposition_positions(
        xy_pred,
    ) -> Callable[[str], Tuple[int, int]]:
        def xy_prop(prop: str) -> Tuple[int, int]:
            pred = dnf_utils.parse_head(prop)
            args = dnf_utils.parse_args(prop)
            x_pred, y_pred = xy_pred(pred)
            x_prop = x_pred + dx_args(args)
            return x_prop, y_pred

        return xy_prop

    xy_pred, x_bins = compute_predicate_positions(pddl)
    xy_prop = compute_proposition_positions(xy_pred)

    # Prepare extended canvas.
    img_ext = 255 * np.ones((img.shape[0], x_bins[-1], 4), dtype=np.uint8)
    img_ext = PIL.Image.fromarray(img_ext, "RGBA")

    # Draw predicate labels.
    for pred in pddl.predicates:
        x_pred, y_pred = xy_pred(pred.name)
        x_pred -= PADDING_TEXT_X
        draw_text(img_ext, pred.name, (x_pred, y_pred), right_align=True)

    # Draw argument labels.
    y_args = WIDTH_ARGS - PADDING_TEXT_X
    for i, x_bin in enumerate(x_bins[:-1]):
        if i < NUM_UNARY_BINS:
            for arg in ARGS:
                x_args = x_bin + dx_args([arg])
                draw_vertical_text(img_ext, arg, (x_args, y_args))
            continue

        for arg_a in ARGS:
            for arg_b in ARGS:
                if arg_a == arg_b:
                    continue
                str_args = f"{arg_a}, {arg_b}"
                x_args = x_bin + dx_args([arg_a, arg_b])
                draw_vertical_text(img_ext, str_args, (x_args, y_args))

    # Create drawing context.
    draw = PIL.ImageDraw.Draw(img_ext)

    # Draw grid lines.
    y_max = {x_bins[0]: 0, x_bins[1]: 0, x_bins[2]: 0}
    for pred in pddl.predicates:
        x_pred, y_pred = xy_pred(pred.name)
        if len(pred.parameters) == 1:
            x_pred_end = x_pred + NUM_UNARY_ARGS * HEIGHT_LINE
        else:
            x_pred_end = x_pred + NUM_BINARY_ARGS * HEIGHT_LINE
        draw.line((x_pred, y_pred, x_pred_end, y_pred), fill=(240, 240, 240))
        y_max[x_pred] = max(y_pred, y_max[x_pred])

    for i, x_bin in enumerate(x_bins[:-1]):
        if i < NUM_UNARY_BINS:
            num_lines = NUM_UNARY_ARGS + 1
        else:
            num_lines = NUM_BINARY_ARGS + 1

        y2 = y_max[x_bin] + HEIGHT_LINE
        for j in range(num_lines):
            x = x_bin + j * HEIGHT_LINE
            draw.line((x, WIDTH_ARGS, x, y2), fill=(240, 240, 240))
        draw.line(
            (x_bin, y2, x_bin + (num_lines - 1) * HEIGHT_LINE, y2), fill=(240, 240, 240)
        )

    # Get valid props.
    idx_valid = dnf_utils.get_valid_props(pddl)

    # Draw proposition labels.
    for i, prop in enumerate(props):
        if pddl is not None:
            idx_prop = pddl.state_index.get_proposition_index(prop)
            if not idx_valid[idx_prop]:
                continue
        if void_args is not None:
            args = dnf_utils.parse_args(prop)
            if void_args.intersection(set(args)):
                continue

        x_prop, y_prop = xy_prop(prop)
        p_prop = p_props[i]
        bg_color = map_color(p_prop)
        draw.rectangle(
            (x_prop, y_prop, x_prop + HEIGHT_LINE, y_prop + HEIGHT_LINE),
            fill=bg_color,
        )
        str_p_prop = str(int(10 * p_prop + 0.5) / 10)
        draw.text((x_prop + 2, y_prop + 2), str_p_prop, font=FONT_PROB, fill=0)

    img_ext = np.array(img_ext)[:, :, :3]

    return np.concatenate((img_ext, img), axis=1)


def video_append_propositions(
    video_frames: np.ndarray,
    props: List[str],
    p_predict: np.ndarray,
    pddl: Optional[symbolic.Pddl],
    void_args: Optional[Set[str]] = None,
) -> np.ndarray:
    """Draws proposition predictions to the left of the video.

    Args:
        video_frames: [T, H, W, 3] uint8 array.
        props: Proposition names.
        p_predict: Predicted proposition probabilities [T,P].
        pddl: Pddl instance.
        void_args: Which arguments don't exist.
    Returns:
        Modified video as a [T, H, W', 3] array.
    """
    video_aug = []
    for t in range(video_frames.shape[0]):
        video_aug.append(
            image_append_propositions(
                video_frames[t], props, p_predict[t], pddl, void_args
            )
        )
    return np.stack(video_aug, axis=0)


def extract_boxes(
    video_labels: Dict,
    id_video: int,
    keyframes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extracts boxes from Something-Else labels.

    Args:
        video_labels: Something-Else labels.
        id_video: Video id.
        keyframes: Keyframes to extract.
    Returns:
        2-tuple (
            [T, 4, 4] (num_keyframes, hand/a/b/c, x1/y1/x2/y2) boxes,
            [4] object names
        ).
    """

    bboxes = video_labels[id_video]["frames"]

    if keyframes is None:
        keyframes = list(bboxes.keys())
    T = len(keyframes)

    boxes = np.full((T, 4, 4), -float("inf"), dtype=np.float32)
    for t, keyframe in enumerate(keyframes):
        for id_object, bbox in bboxes[keyframe].items():
            try:
                idx_object = object_id_to_idx(id_object)
            except ValueError:
                continue

            boxes[t, idx_object, :] = np.array(bbox).flatten()

    objects = ["hand"] + video_labels[id_video]["objects"]

    return boxes, objects
