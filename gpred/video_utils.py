import io
import math
import pathlib
import typing

import av
import numpy as np
import PIL
import PIL.Image
import seaborn as sns


def rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
    """Converts an rgb image to an rgba image.

    Args:
        rgb: [H, W, 3] uint8 image.
    Returns:
        [H, W, 4] uint8 image.
    """
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def float_to_rgb(f: np.ndarray) -> np.ndarray:
    """Converts a float image (0-1) to an rgb image.

    Args:
        f: [H, W, C] float (0-1) image.
    Returns:
        [H, W, C] uint8 (0-255) image.
    """
    return (255 * f + 0.5).astype(np.uint8)


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize image to the range 0-1."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return img - img_min
    return (img - img_min) / (img_max - img_min)


def encode_png(img: np.ndarray) -> bytes:
    """Encodes an image to a serialized png."""
    if img.dtype != np.uint8:
        img = float_to_rgb(img)

    pil_img = PIL.Image.fromarray(img)
    with io.BytesIO() as output:
        pil_img.save(output, format="PNG")
        img_png = output.getvalue()

    return img_png


def display_video(id_video: int, path: pathlib.Path) -> None:
    """Display video.

    Args:
        id_video: Video id.
        path: Path of video.
    """
    import IPython

    if (path / f"{id_video}.webm").is_file():
        IPython.display.display(IPython.display.Video(path / f"{id_video}.webm"))
    elif (path / f"{id_video}.mp4").is_file():
        IPython.display.display(IPython.display.Video(path / f"{id_video}.mp4"))
    else:
        raise RuntimeError(
            f"Could not find {id_video}.webm or {id_video}.mp4 in {path}"
        )


def imshow(img: np.ndarray):
    """Shows the image in iPython without scaling."""
    import PIL.Image
    import IPython
    import io

    if img.dtype != np.uint8:
        # Turn img into uint8 array.
        img = (255 * normalize(img) + 0.5).astype(np.uint8)

    img = PIL.Image.fromarray(img)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        img_png = output.getvalue()

    IPython.display.display(IPython.display.Image(img_png))


def display_video_grid(
    id_videos: typing.List[int],
    path: pathlib.Path,
    num_cols: int = 3,
    labels: typing.Optional[typing.List[str]] = None,
) -> None:
    """Display video grid.

    Args:
        id_videos: Video ids.
        path: Path of video.
        num_cols: Number of columns.
        labels: Optional list of labels to print for each video.
    """
    from IPython.display import display
    import ipywidgets

    num_rows = int(math.ceil(len(id_videos) / num_cols) + 0.5)

    outputs = []
    for col in range(num_cols):
        outputs_col = [ipywidgets.Output() for _ in range(num_rows)]
        for row in range(num_rows):
            i = row * num_cols + col
            if i >= len(id_videos):
                continue

            id_video = id_videos[i]
            with outputs_col[row]:
                print(path / f"{id_video}.webm")
                if labels is not None:
                    print(labels[i])
                print("")
                display_video(id_video, path)

        outputs.append(ipywidgets.VBox(outputs_col))

    display(ipywidgets.HBox(outputs))


def get_keyframes(path_video: typing.Union[pathlib.Path, str]) -> typing.List[int]:
    """Get keyframe indices for the given video.

    Args:
        path_video: Path of video.
    Returns:
        [T] (num_keyframes) List of keyframes.
    """
    with av.open(str(path_video)) as container:
        stream = container.streams.video[0]

        get_timestamp = lambda pts: int(
            float(pts * stream.time_base * stream.base_rate) + 0.5
        )

        keyframes = [get_timestamp(frame.pts) for frame in container.decode()]

    return keyframes


def read_video(
    path_video: typing.Union[pathlib.Path, str],
    keyframes: typing.Optional[typing.List[int]] = None,
    return_keyframes: bool = False,
) -> typing.Union[
    typing.List[np.ndarray], typing.Tuple[typing.List[np.ndarray], typing.List[int]]
]:
    """Get selected frames from a video.

    Args:
        path_video: Path of video.
        keyframes: List of keyframes to extract from the video, or None to get all.
        return_keyframes: Whether to return a list of extracted keyframes.
    Returns:
        Selected video frames as a [T] list of [H x W x 3] uint8 arrays.
    """
    with av.open(str(path_video)) as container:
        stream = container.streams.video[0]

        get_timestamp = lambda pts: int(
            float(pts * stream.time_base * stream.base_rate) + 0.5
        )

        video_frames = []

        if keyframes is None:
            keyframes = []
            for frame in container.decode():
                video_frames.append(frame.to_ndarray(format="rgb24"))
                if return_keyframes:
                    t = get_timestamp(frame.pts)
                    keyframes.append(t)
        else:
            idx_keyframe = 0
            for frame in container.decode():
                # Compute timestamp of current frame.
                t = get_timestamp(frame.pts)

                # Repeat current frame until timestamp equals current keyframe.
                while idx_keyframe < len(keyframes) and t >= keyframes[idx_keyframe]:
                    video_frames.append(frame.to_ndarray(format="rgb24"))
                    idx_keyframe += 1

                # Break early if there are no keyframes left.
                if idx_keyframe >= len(keyframes):
                    break

            if len(video_frames) != len(keyframes):
                raise RuntimeError(
                    f"Could not load requested keyframes from {path_video}: {keyframes}"
                )

    # video_frames = np.stack(video_frames, axis=0)

    if return_keyframes:
        return video_frames, keyframes

    return video_frames


def write_video(
    path_video: typing.Union[pathlib.Path, str],
    video_frames: typing.Union[np.ndarray, typing.List[np.ndarray]],
    fps: int = 12,
    codec: str = "vp9",
):
    """Write frames to a video file.

    Args:
        path_video: Path of video.
        video_frames: [T, H, W, 3] uint8 array or list of [H, W, 3] uint8 arrays.
        fps: Framerate.
        codec: Video codec (e.g. "vp9" or "h264").
    """
    path_video = pathlib.Path(path_video)
    path_video.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(path_video), mode="w")
    stream = container.add_stream(codec, rate=fps)
    T = len(video_frames)

    if type(video_frames) is np.ndarray:
        stream.width = video_frames.shape[2]
        stream.height = video_frames.shape[1]
    elif type(video_frames) is list:
        # Compute the maximum size of all the frames.
        is_uniform = True
        h_max, w_max = video_frames[0].shape[:2]
        for t in range(1, T):
            h_t, w_t = video_frames[t].shape[:2]
            if w_t == w_max and h_t == h_max:
                continue
            w_max = max(w_max, w_t)
            h_max = max(h_max, h_t)
            is_uniform = False

        stream.width = w_max
        stream.height = h_max

        if not is_uniform:
            # Resize all frames to the same size.
            for t in range(T):
                h_t, w_t = video_frames[t].shape[:2]
                if w_t == w_max and h_t == h_max:
                    continue
                new_frame = np.fill((h_max, w_max, 3), 255, dtype=np.uint8)
                new_frame[:h_t, :w_t, :] = video_frames[t]
                video_frames[t] = new_frame

    for t in range(T):
        frame = av.VideoFrame.from_ndarray(video_frames[t], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def image_cnn_to_rgb(
    image: np.ndarray,
    image_distribution: typing.Tuple[
        typing.Tuple[float, float, float], typing.Tuple[float, float, float]
    ],
) -> np.ndarray:
    """Converts a [3, H, W] image to a [H, W, 3] image.

    Args:
        image: [3, H, W] float array.
        image_distribution: (img_mean, img_stddev).
    Returns:
        [H, W, 3] uint8 array.
    """
    image = np.moveaxis(image, 0, 2)
    img_mean, img_stddev = (np.array(x)[None, None, :] for x in image_distribution)
    img_rgb = (255 * (img_stddev * image + img_mean) + 0.5).astype(np.uint8)
    return img_rgb


def image_rgb_to_cnn(
    image: np.ndarray,
    image_distribution: typing.Tuple[
        typing.Tuple[float, float, float], typing.Tuple[float, float, float]
    ],
) -> np.ndarray:
    """Converts a [H, W, 3] image to a [3, H, W] image.

    Args:
        image: [H, W, 3] uint8 array.
        image_distribution: (img_mean, img_stddev).
    Returns:
        [3, H, W] float array.
    """
    img_mean, img_stddev = (
        np.array(x, dtype=np.float32)[None, None, :] for x in image_distribution
    )
    img_cnn = (image.astype(np.float32) / 255 - img_mean) / img_stddev
    img_cnn = np.moveaxis(img_cnn, 2, 0)
    return img_cnn


def video_cnn_to_rgb(
    video: np.ndarray,
    image_distribution: typing.Tuple[
        typing.Tuple[int, int, int], typing.Tuple[int, int, int]
    ],
) -> np.ndarray:
    """Converts a [T, 3, H, W] float video to a [T, H, W, 3] uint8 video.

    Args:
        video: [T, 3, H, W] float array.
        image_distribution: (img_mean, img_stddev).
    Returns:
        [T, H, W, 3] uint8 array.
    """
    video = np.moveaxis(video, 1, 3)
    img_mean, img_stddev = (
        np.array(x)[None, None, None, :] for x in image_distribution
    )
    video_rgb = (255 * (img_stddev * video + img_mean) + 0.5).astype(np.uint8)
    return video_rgb


def video_rgb_to_cnn(
    video: np.ndarray,
    image_distribution: typing.Tuple[
        typing.Tuple[int, int, int], typing.Tuple[int, int, int]
    ],
) -> np.ndarray:
    """Converts a [T, H, W, 3] uint8 video to a [T, 3, H, W] float video.

    Args:
        video: [T, H, W, 3] uint8 array.
        image_distribution: (img_mean, img_stddev).
    Returns:
        [T, 3, H, W] float array.
    """
    video = np.moveaxis(video, 3, 1)
    img_mean, img_stddev = (
        np.array(x, dtype=np.float32)[None, :, None, None] for x in image_distribution
    )
    video_cnn = (video.astype(np.float32) / 255 - img_mean) / img_stddev
    return video_cnn


def image_to_rgb(
    img: np.ndarray,
    image_distribution: typing.Tuple[
        typing.Tuple[int, int, int], typing.Tuple[int, int, int]
    ],
) -> np.ndarray:
    """Converts the multiple-channel image/mask to an rgb image.

    Args:
        img: [3+, H, W] array.
        image_distribution: (img_mean, img_stddev).
    Returns:
        [H, W, 3] uint8 array.
    """
    from env import twentybn

    if len(img.shape) != 3:
        raise ValueError("Image must have 3 dimensions.")

    # Assume image is in cnn format.
    if img.shape[0] < 10:
        if img.shape[0] < 3:
            raise ValueError("Image must have at least 3 channels.")

        # Convert cnn image to rgb.
        img_cnn = img[:3]
        img_rgb = image_cnn_to_rgb(img_cnn, image_distribution)

        # Assume remaining channels are box masks.
        if img.shape[0] > 3:
            img_boxes = img[3:]
            img_rgb = twentybn.utils.image_overlay_box_mask(img_rgb, img_boxes)

        return img_rgb

    # Assume image is in rgb format.
    return img


def get_bbox_color(i: int) -> typing.Tuple[int, int, int]:
    """Creates an RGB color from the seaborn color palette."""
    CMAP = sns.color_palette()
    float_color = CMAP[i % len(CMAP)]
    int_color = (
        int(255 * float_color[0] + 0.5),
        int(255 * float_color[1] + 0.5),
        int(255 * float_color[2] + 0.5),
    )
    return int_color


def draw_action_skeleton(
    img: np.ndarray, action_skeleton: typing.List[str], props: typing.Dict[str, float]
) -> np.ndarray:
    OPACITY = 0.6
    FONT = PIL.ImageFont.truetype("arial.ttf", 48)
    FONT_BOLD = PIL.ImageFont.truetype("arialbd.ttf", 48)
    FONT_SYMBOL = PIL.ImageFont.truetype("DejaVuSans.ttf", 48)
    PADDING_TEXT_X = 18
    PADDING_TEXT_Y = 10
    H_ROW = FONT.size + 2 * PADDING_TEXT_Y
    W_ROW = 615
    rgba = (255, 255, 255, int(OPACITY * 255))
    W_IMG = img.shape[1]

    def draw_action(draw: PIL.ImageDraw.Draw, action: str, idx_action: int = -1):
        # Draw label.
        x1 = W_IMG
        y1 = H_ROW * (idx_action + 1) + 2 * PADDING_TEXT_Y
        x2 = x1 + W_ROW
        y2 = y1 + H_ROW
        if idx_action < 0:
            # draw.rectangle([x1, y1 - PADDING_TEXT_Y, x2, y2], fill=rgba)
            draw.text((x1 + PADDING_TEXT_X, y1), action, font=FONT, fill=(0, 0, 0, 255))
        elif action is None:
            draw.text(
                (x1 + PADDING_TEXT_X, y1),
                "Goal satisfied ",
                font=FONT_BOLD,
                fill=(0, 0, 0, 255),
            )
            width_text = draw.textsize("Goal satisfied ", font=FONT_BOLD)[0]
            CHECK = "\u2714"
            draw.text(
                (x1 + PADDING_TEXT_X + width_text, y1),
                CHECK,
                font=FONT_SYMBOL,
                fill=(0, 200, 0, 255),
            )
        elif idx_action == 0:
            # draw.rectangle([x1, y1, x2, y2], fill=rgba)
            draw.text(
                (x1 + PADDING_TEXT_X, y1),
                f"- {action}",
                font=FONT_BOLD,
                fill=(0, 0, 0, 255),
            )
        else:
            # draw.rectangle([x1, y1, x2, y2], fill=rgba)
            draw.text(
                (x1 + PADDING_TEXT_X, y1),
                f"- {action}",
                font=FONT,
                fill=(128, 128, 128, 255),
            )

    def draw_prop(prop: str, props: typing.Dict[str, float], idx_prop: int):
        RGB_TRUE = (0, 200, 0)
        RGB_FALSE = (200, 0, 0)

        # Draw label.
        x1 = W_IMG
        y1 = H_ROW * (idx_prop + 9)
        x2 = x1 + W_ROW
        y2 = y1 + H_ROW

        if prop in props:
            prob = props[prop]
            if prob < 0.5:
                prop = "\u00ac" + prop
            if prob < 0:
                prob = 0
            elif prob > 1:
                prob = 1

            rgb_val = prob * np.array(RGB_TRUE) + (1 - prob) * np.array(RGB_FALSE)
            rgba = (*((rgb_val + 0.5).astype(np.uint8)), 255)
            draw.text(
                (x1 + PADDING_TEXT_X, y1),
                f"{prob:.2f}",
                font=FONT,
                fill=rgba,
            )
            draw.text(
                (x1 + PADDING_TEXT_X + 120, y1),
                prop,
                font=FONT,
                fill=(0, 0, 0, 255),
            )
        else:
            draw.text(
                (x1 + PADDING_TEXT_X + 120, y1),
                prop,
                font=FONT,
                fill=(128, 128, 128, 255),
            )

    # Convert frame to RGBA.
    frame_rgba = np.full((img.shape[0], img.shape[1] + W_ROW, 4), 255, dtype=np.uint8)
    frame_rgba[:, : img.shape[1], :3] = img
    img = PIL.Image.fromarray(frame_rgba, "RGBA")

    # Create overlay drawing context.
    overlay = PIL.Image.new("RGBA", img.size, 0)
    draw = PIL.ImageDraw.Draw(overlay)

    # Draw the boxes.
    draw_action(draw, "Action skeleton:")
    for i, action in enumerate(action_skeleton):
        draw_action(draw, action, i)

    draw_prop("closed(drawer)", props, 0)
    draw_prop("in(banana, hand)", props, 1)
    draw_prop("in(banana, drawer)", props, 2)
    draw_prop("in(orange, hand)", props, 3)
    draw_prop("in(orange, drawer)", props, 4)
    draw_prop("onsurface(banana)", props, 5)
    draw_prop("onsurface(orange)", props, 6)

    # Composite the overlay.
    img = PIL.Image.alpha_composite(img, overlay)
    img = np.array(img)[:, :, :3]

    return img


def draw_bounding_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: typing.List[str],
    detection_ids: typing.Optional[typing.List[int]] = None,
) -> np.ndarray:
    """Overlay bounding boxes onto the image.

    Args:
        img: [H, W, 3] uint8 array.
        boxes: [O, 4] (num_objects, x1/y1/x2/y2) array.
        labels: Object labels for each box.
        detection_ids: Object detection ids to determine box colors.
    Returns:
        Modified img as a [H, W, 3] array.
    """

    def draw_labeled_box(
        draw: PIL.ImageDraw.Draw,
        box: np.ndarray,
        label: str,
        id_detection: int,
        width: int,
        height: int,
        above: bool = False,
    ):
        """Draws a labeled bounding box for the object.

        Args:
            draw: Drawing context.
            box: [4,] (x1/y1/x2/y2) array.
            label: Object label.
            id_detection: Object detection id.
            width: Width of image. Label will be drawn within the width.
            height: Height of image. Label will be drawn within the height.
            above: Whether to draw label above or below the box.
        """
        OPACITY = 0.75
        FONT = PIL.ImageFont.truetype("arial.ttf", 15)  # 48)
        PADDING_TEXT_X = 6  # 12
        PADDING_TEXT_Y = 4  # 8
        WIDTH_LINE = 2  # 6

        # Create color.
        rgb = get_bbox_color(id_detection)
        rgba = (*rgb, int(OPACITY * 255))

        # Draw bounding box.
        box = (box + 0.5).astype(int)
        draw.rectangle(box.tolist(), outline=rgba, width=WIDTH_LINE)

        width_label = draw.textsize(label, font=FONT)[0] + 2 * PADDING_TEXT_X
        width_label = max(width_label, box[2] - box[0])

        # Compute hypothetical label placements above and below.
        y1_above = max(0, int(box[1]) - FONT.size - PADDING_TEXT_Y)
        y2_above = y1_above + FONT.size + PADDING_TEXT_Y
        y2_below = min(height, box[3] + FONT.size + PADDING_TEXT_Y)
        y1_below = y2_below - FONT.size - PADDING_TEXT_Y

        # If one is clipped, use the other.
        is_above_clipped = y2_above > box[1]
        is_below_clipped = y1_below < box[3]
        if is_above_clipped and not is_below_clipped:
            y1, y2 = y1_below, y2_below
        elif (is_below_clipped and not is_above_clipped) or above:
            y1, y2 = y1_above, y2_above
        else:
            y1, y2 = y1_below, y2_below

        # Compute horizontal placement of label.
        x2 = min(width, box[0] + width_label)
        x1 = x2 - width_label

        # Draw label.
        draw.rectangle([x1, y1, x2, y2], fill=rgba)
        draw.text((x1 + PADDING_TEXT_X, y1), label, font=FONT)

    # Use default numbering if detection ids not present.
    O = boxes.shape[0]
    if detection_ids is None:
        detection_ids = list(range(O))
    if len(labels) < O:
        default_labels = ["hand"] + [chr(i + ord("a")) for i in range(O - 1)]
        labels += default_labels[len(labels) :]

    # Convert frame to RGBA.
    frame_rgba = rgb_to_rgba(img)
    img = PIL.Image.fromarray(frame_rgba, "RGBA")

    # Create overlay drawing context.
    overlay = PIL.Image.new("RGBA", img.size, 0)
    draw = PIL.ImageDraw.Draw(overlay)

    # Draw the boxes.
    H = frame_rgba.shape[0]
    W = frame_rgba.shape[1]
    for i in range(O):
        draw_labeled_box(draw, boxes[i], labels[i], detection_ids[i], W, H)

    # Composite the overlay.
    img = PIL.Image.alpha_composite(img, overlay)
    img = np.array(img)[:, :, :3]

    return img


def overlay_predictions(
    img: np.ndarray,
    p_predict: np.ndarray,
    prop_names: typing.List[str],
) -> np.ndarray:
    """Overlays proposition predictions onto the video or image.

    Args:
        img: [T, H, W, 3] or [H, W, 3] uint8 array.
        p_predict: Predicted proposition probabilities [T, P] or [P].
        prop_names: Proposition names [P].
    Returns:
        Modified frames as a [T, H, W, 3] or [H, W, 3] array.
    """
    if len(img.shape) > 3:
        # Overlay predictions for each frame in the video.
        video = []
        for t in range(img.shape[0]):
            video.append(overlay_predictions(img[t], p_predict[t], prop_names))
        return np.stack(video, axis=0)

    FONT = PIL.ImageFont.truetype("arial.ttf", 15)
    PADDING_TEXT_X = 6
    HEIGHT = FONT.size + 3
    RGB_TRUE = np.array([0, 255, 0, 100], dtype=np.uint8)
    RGB_FALSE = np.array([255, 0, 0, 100], dtype=np.uint8)

    # Convert frame to RGBA.
    frame_rgba = rgb_to_rgba(img)
    img = PIL.Image.fromarray(frame_rgba, "RGBA")

    # Create overlay drawing context.
    overlay = PIL.Image.new("RGBA", img.size, 0)
    draw = PIL.ImageDraw.Draw(overlay)

    # Compute width of text.
    width = 0
    for i, prop in enumerate(prop_names):
        p_prop = p_predict[i]
        width_text = draw.textsize(f"{prop}: {p_prop:.2f}", font=FONT)[0]
        width = max(width, width_text + 2 * PADDING_TEXT_X)

    # Draw proposition label.
    row = 0
    for i, prop in enumerate(prop_names):
        p_prop = p_predict[i]

        bg_color = tuple(
            ((p_prop * RGB_TRUE + (1 - p_prop) * RGB_FALSE) + 0.5).astype(np.uint8)
        )

        # Left-align proposition name.
        draw.rectangle((0, row * HEIGHT, width, (row + 1) * HEIGHT), fill=bg_color)
        draw.text((PADDING_TEXT_X, row * HEIGHT), f"{prop}:", font=FONT)

        # Right-align probability.
        width_prob = draw.textsize(f"{p_prop:.2f}", font=FONT)[0]
        x_prob = width - PADDING_TEXT_X - width_prob
        draw.text((x_prob, row * HEIGHT), f"{p_prop:.2f}", font=FONT)

        row += 1

    # Composite the overlay.
    img = PIL.Image.alpha_composite(img, overlay)
    img = np.array(img)[:, :, :3]

    return img


def append_conditions(
    img: np.ndarray,
    action: str,
    action_phase: int,
    p_predict: np.ndarray,
    prop_names: typing.List[str],
) -> np.ndarray:
    """Appends condition satisfaction to the image.

    Args:
        img: [H, W, 3] uint8 array.
        action: Action name.
        action_phase: Action phase (0-4).
        p_predict: Predicted proposition probabilities [P].
        prop_names: Proposition names [P].
    Returns:
        Resized image [H, W + n, 3] uint8 array.
    """
    FONT = PIL.ImageFont.truetype("Arial.ttf", 15)
    FONT_BOLD = PIL.ImageFont.truetype("Arial_Bold.ttf", 15)
    FONT_HEADER = PIL.ImageFont.truetype("Arial_Bold.ttf", 16)
    FONT_SYMBOL = PIL.ImageFont.truetype("DejaVuSans.ttf", 16)
    RGB_FG = (0, 0, 0)

    RGB_TRUE = (0, 200, 0)
    RGB_FALSE = (200, 0, 0)

    PADDING_TEXT_X = 10
    PADDING_HEADER_Y = 4
    H_TEXT = FONT.size + 3
    H_HEADER = FONT_HEADER.size + PADDING_HEADER_Y + 3
    W_SYMBOL = 20
    PADDING_PROP_X = 2 * W_SYMBOL + PADDING_TEXT_X

    H, W = img.shape[:2]

    X_TEXT = W + PADDING_TEXT_X
    X_PROP = W + PADDING_PROP_X

    def get_symbol_color(
        action_phase: int,
    ) -> typing.Tuple[
        typing.Tuple[int, int, int],
        str,
        PIL.ImageFont.ImageFont,
        typing.Tuple[int, int, int],
        str,
        PIL.ImageFont.ImageFont,
    ]:
        """Gets the symbol and color for the current action phase.

        Args:
            action_phase: (0-4) Action phase.
            prop_names: Proposition names [P].
        Returns:
            4-tuple (pre_symbol, pre_color, pre_font, post_symbol, post_color, pre_font).
        """
        PALE_RED = (255, 128, 128)
        PALE_GREEN = (128, 255, 128)
        CHECK = "\u2714"
        CROSS = "\u2718"
        FONT_BG = FONT
        FONT_FG = FONT_BOLD
        if action_phase == 0:
            return RGB_FALSE, CROSS, FONT_FG, PALE_RED, CROSS, FONT_BG
        if action_phase == 1:
            return RGB_TRUE, CHECK, FONT_FG, PALE_RED, CROSS, FONT_BG
        if action_phase == 2:
            return PALE_GREEN, CHECK, FONT_BG, RGB_FALSE, CROSS, FONT_FG
        if action_phase == 3:
            return PALE_GREEN, CHECK, FONT_BG, RGB_TRUE, CHECK, FONT_FG
        return PALE_GREEN, CHECK, FONT_BG, PALE_GREEN, CHECK, FONT_BG

    # Compute width of augmented frame.
    w_text = FONT_BOLD.getsize("Post-conditions")[0] + PADDING_TEXT_X
    for prop in prop_names:
        w_text = max(w_text, FONT.getsize("\u00ac" + prop)[0] + PADDING_PROP_X)
    w_text += PADDING_TEXT_X

    # Create augmented frame.
    frame = np.full((H, W + w_text, 3), 255, dtype=np.uint8)
    frame[:, :W, :] = img
    img = PIL.Image.fromarray(frame, "RGB")

    # Create drawing context.
    draw = PIL.ImageDraw.Draw(img)

    # Action title.
    y_text = PADDING_HEADER_Y
    draw.text((X_TEXT, y_text), action, font=FONT_HEADER, fill=RGB_FG)
    y_text += H_HEADER

    # Pre/post-condition satisfaction.
    (
        pre_color,
        pre_symbol,
        pre_font,
        post_color,
        post_symbol,
        post_font,
    ) = get_symbol_color(action_phase)
    draw.text((X_TEXT, y_text), pre_symbol, font=FONT_SYMBOL, fill=pre_color)
    draw.text((X_TEXT + 20, y_text), "Pre-conditions", font=pre_font, fill=RGB_FG)
    y_text += H_TEXT
    draw.text((X_TEXT, y_text), post_symbol, font=FONT_SYMBOL, fill=post_color)
    draw.text((X_TEXT + 20, y_text), "Post-conditions", font=post_font, fill=RGB_FG)
    y_text += H_TEXT

    # Propositions.
    y_text += H_TEXT
    for i, prop in enumerate(prop_names):
        p_prop = p_predict[i]

        # Proposition value.
        rgb_val = p_prop * np.array(RGB_TRUE) + (1 - p_prop) * np.array(RGB_FALSE)
        rgb_val = tuple((rgb_val + 0.5).astype(np.uint8))
        # draw.rectangle(
        #     (X_TEXT, y_text, W + 2 * W_SYMBOL, y_text + H_TEXT), fill=rgb_val
        # )
        draw.text((X_TEXT, y_text), f"{p_prop:.2f}", font=FONT, fill=rgb_val)

        # Proposition name.
        if p_prop < 0.5:
            prop = "\u00ac" + prop
        draw.text((X_PROP, y_text), prop, font=FONT, fill=RGB_FG)
        y_text += H_TEXT

    # Convert image.
    img = np.array(img)[:, :, :3]

    return img
