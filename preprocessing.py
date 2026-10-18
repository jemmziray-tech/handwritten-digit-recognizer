"""
preprocessing.py - Robust Image Preprocessing Pipeline for MNIST Digit Recognition.

Handles real-world photographs, scans, digital drawings, and phone camera images:
- EXIF orientation correction
- Transparency / alpha channel compositing
- Polarity detection (dark-on-light paper ink vs bright-on-dark MNIST)
- Background illumination flattening and shadow reduction
- Accurate bounding-box cropping with aspect ratio preservation
- Center of Mass (CoM) translation onto 28x28 canvas (canonical Yann LeCun MNIST spec)
- Exact training-time normalisation (mean=0.1307, std=0.3081)
- Multi-digit candidate segmentation for sequential number photographs
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081
CANVAS_SIZE: int = 28
DIGIT_BOX_SIZE: int = 20  # Canonical MNIST digit bounding box target size


class PreprocessingError(Exception):
    """Raised when an uploaded image cannot be processed or contains no digit."""
    pass


@dataclass
class PreprocessedDigit:
    """Holds the result of preprocessing a single digit."""
    tensor: torch.Tensor          # Shape: (1, 1, 28, 28), normalised float32
    display_image: Image.Image    # 28x28 PIL grayscale image (0-255) for UI display
    raw_array: np.ndarray         # 28x28 float array in [0.0, 1.0]
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in original image
    inverted: bool = False
    center_of_mass: Tuple[float, float] = (14.0, 14.0)
    details: Dict[str, Any] = None


def load_input_image(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    max_dimension: int = 1600,
) -> Image.Image:
    """
    Safely loads and validates an image from varied input sources.

    - Resolves EXIF orientation (crucial for phone photos).
    - Composites alpha transparency onto a solid background.
    - Converts to standard RGB mode.
    - Limits oversized uploads to avoid OOM or slow processing.

    Args:
        source: File path, Path object, bytes, BytesIO, PIL Image, or numpy array.
        max_dimension: Downsample images whose longest edge exceeds this.

    Returns:
        Validated PIL Image in RGB mode.

    Raises:
        PreprocessingError: If image data is invalid or corrupt.
    """
    try:
        if isinstance(source, Image.Image):
            pil_img = source
        elif isinstance(source, np.ndarray):
            if source.ndim == 2:
                pil_img = Image.fromarray(source).convert("RGB")
            elif source.ndim == 3 and source.shape[2] in (3, 4):
                pil_img = Image.fromarray(source)
            else:
                raise PreprocessingError(f"Unsupported numpy image shape: {source.shape}")
        elif isinstance(source, (str, Path, io.BytesIO, bytes)):
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            elif isinstance(source, Path):
                source = str(source)
            pil_img = Image.open(source)
        else:
            raise PreprocessingError(f"Unsupported image input type: {type(source)}")

        # Handle EXIF orientation
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception as exif_err:
            logger.debug("EXIF transpose skipped: %s", exif_err)

        # Handle transparency (RGBA / LA / P with alpha)
        if pil_img.mode in ("RGBA", "LA") or (
            pil_img.mode == "P" and "transparency" in pil_img.info
        ):
            rgba = pil_img.convert("RGBA")
            # Create a white background canvas
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(background, rgba)
            pil_img = composite.convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")

        # Downscale if excessively large
        w, h = pil_img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)

        return pil_img

    except Exception as exc:
        if isinstance(exc, PreprocessingError):
            raise
        raise PreprocessingError(f"Could not load or decode image: {exc}") from exc


def detect_polarity_and_invert(
    gray_arr: np.ndarray,
    invert_mode: Literal["auto", "always", "never"] = "auto",
    border_ratio: float = 0.08,
) -> Tuple[np.ndarray, bool]:
    """
    Determines whether the image is dark-on-light (ink on paper) or
    bright-on-dark (standard MNIST/chalkboard), and converts it to MNIST polarity
    where digit strokes are bright (near 255) and background is dark (near 0).

    Args:
        gray_arr: Grayscale image array uint8 [0, 255].
        invert_mode: 'auto', 'always', or 'never'.
        border_ratio: Percentage of image boundary to sample for background brightness.

    Returns:
        Tuple of (polarity_corrected_array, was_inverted_bool)
    """
    if invert_mode == "always":
        return 255 - gray_arr, True
    if invert_mode == "never":
        return gray_arr.copy(), False

    # Auto mode: sample the border pixels
    h, w = gray_arr.shape
    bh = max(2, int(h * border_ratio))
    bw = max(2, int(w * border_ratio))

    border_pixels = np.concatenate([
        gray_arr[:bh, :].flatten(),       # top
        gray_arr[-bh:, :].flatten(),      # bottom
        gray_arr[:, :bw].flatten(),       # left
        gray_arr[:, -bw:].flatten(),      # right
    ])

    border_median = float(np.median(border_pixels))
    # If border is bright (> 128), it's dark ink on light paper -> invert so stroke is bright
    should_invert = border_median > 128.0

    if should_invert:
        return 255 - gray_arr, True
    return gray_arr.copy(), False


def flatten_paper_background(
    gray_arr: np.ndarray,
    kernel_size: int = 41,
) -> np.ndarray:
    """
    Flattens uneven illumination, shadows, and paper gradients using
    morphological background subtraction (top-hat / rolling-ball equivalent).

    Args:
        gray_arr: uint8 array where strokes are expected to be bright.
        kernel_size: Size of structuring element for background estimation.

    Returns:
        Background-flattened uint8 array with enhanced stroke contrast.
    """
    # Morphological opening estimates the smooth background illumination
    k = max(9, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray_arr, cv2.MORPH_OPEN, kernel)

    # Subtract estimated background to isolate bright foreground strokes
    flattened = cv2.subtract(gray_arr, background)

    # Normalize contrast to span [0, 255]
    p_min, p_max = float(np.percentile(flattened, 1)), float(np.percentile(flattened, 99))
    if p_max > p_min + 10:
        stretched = np.clip((flattened - p_min) * (255.0 / (p_max - p_min)), 0, 255).astype(np.uint8)
        return stretched
    return flattened


def extract_digit_bounding_box(
    bright_stroke_arr: np.ndarray,
    threshold_ratio: float = 0.15,
    margin_ratio: float = 0.12,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Intelligently extracts the active bounding box of the handwritten digit.

    Overcomes real-world photograph challenges:
        - Deep shadows and illumination gradients across paper.
        - Faint handwriting show-through / bleed-through from the reverse side.
        - Non-digit margin markings and notebook page borders.
        - Disjoint multi-stroke digits (e.g. crossing bars of '4', top bar of '5' or '7').

    Args:
        bright_stroke_arr: uint8 array where digit strokes are bright.
        threshold_ratio: Intensity cutoff fraction.
        margin_ratio: Padding margin around the bounding box.

    Returns:
        Tuple of (cropped_digit_array, (x, y, w, h))

    Raises:
        PreprocessingError: If no significant digit stroke is found.
    """
    h, w = bright_stroke_arr.shape
    max_val = float(np.max(bright_stroke_arr))
    if max_val < 20.0:
        raise PreprocessingError(
            "Image is nearly blank. No discernible digit stroke was detected."
        )

    # For pre-cropped images (e.g. standard 28x28 MNIST), use direct thresholding
    if h <= 64 and w <= 64:
        thresh_val = max(15.0, max_val * threshold_ratio)
        mask = bright_stroke_arr > thresh_val
        if not np.any(mask):
            raise PreprocessingError("No digit pixels detected above threshold.")
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad_y = max(1, int((rmax - rmin + 1) * margin_ratio))
        pad_x = max(1, int((cmax - cmin + 1) * margin_ratio))
        y1, y2 = max(0, rmin - pad_y), min(h, rmax + pad_y + 1)
        x1, x2 = max(0, cmin - pad_x), min(w, cmax + pad_x + 1)
        return bright_stroke_arr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # For full photographs: use morphological top-hat to eliminate broad background shadows
    k = max(15, int(min(h, w) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(bright_stroke_arr, cv2.MORPH_TOPHAT, kernel)

    # Threshold on isolated stroke features:
    # High-frequency dark handwriting strokes have strong local contrast,
    # whereas faint bleed-through formulas and shadows have low contrast.
    thresh_val = max(25.0, 0.22 * float(tophat.max()))
    bin_mask = (tophat >= thresh_val).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)

    center_x, center_y = w / 2.0, h / 2.0
    diag = np.sqrt(center_x**2 + center_y**2)

    candidates = []
    for i in range(1, num_labels):
        cx_box, cy_box, cw, ch, area = stats[i]
        # Ignore components spanning almost the whole image (page borders or massive shadows)
        if cw >= w * 0.85 or ch >= h * 0.85:
            continue
        # Ignore tiny noise specks
        if area < 15 or cw < 4 or ch < 4:
            continue

        cx, cy = centroids[i]
        dist_norm = np.sqrt((cx - center_x)**2 + (cy - center_y)**2) / diag

        comp_mask = (labels == i)
        mass = float(np.sum(tophat[comp_mask]))
        # Central handwriting with strong ink stroke mass receives highest score
        score = mass / (1.0 + 1.8 * dist_norm)
        candidates.append({"bbox": (cx_box, cy_box, cw, ch), "score": score, "mass": mass, "area": area})

    if not candidates:
        # Fallback to general threshold
        thresh_val = max(20.0, max_val * 0.20)
        mask = bright_stroke_arr > thresh_val
        if not np.any(mask):
            raise PreprocessingError("No digit pixels detected above threshold.")
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        y1, y2 = max(0, rmin - 4), min(h, rmax + 5)
        x1, x2 = max(0, cmin - 4), min(w, cmax + 5)
        return bright_stroke_arr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # Sort candidates by salience score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    init_x, init_y, init_w, init_h = candidates[0]["bbox"]
    init_mass = candidates[0]["mass"]
    best_x, best_y, best_w, best_h = init_x, init_y, init_w, init_h

    # Non-cascading merge strictly anchored to primary component:
    # Only attach nearby fragments (e.g. crossing strokes of '4' or disconnected loops)
    # that do not cause unbounded cascade
    max_gap = int(max(init_w, init_h) * 0.25)
    for cand in candidates[1:]:
        if cand["mass"] < 0.04 * init_mass:
            continue
        cx_box, cy_box, cw, ch = cand["bbox"]
        h_gap = max(0, max(init_x, cx_box) - min(init_x + init_w, cx_box + cw))
        v_gap = max(0, max(init_y, cy_box) - min(init_y + init_h, cy_box + ch))
        if h_gap <= max_gap and v_gap <= max_gap:
            nx1 = min(best_x, cx_box)
            ny1 = min(best_y, cy_box)
            nx2 = max(best_x + best_w, cx_box + cw)
            ny2 = max(best_y + best_h, cy_box + ch)
            if (nx2 - nx1) <= 2.2 * init_w and (ny2 - ny1) <= 2.2 * init_h:
                best_x, best_y, best_w, best_h = nx1, ny1, nx2 - nx1, ny2 - ny1

    # Apply margin padding
    mx = max(4, int(best_w * margin_ratio))
    my = max(4, int(best_h * margin_ratio))
    fx1 = max(0, best_x - mx)
    fy1 = max(0, best_y - my)
    fx2 = min(w, best_x + best_w + mx)
    fy2 = min(h, best_y + best_h + my)

    crop = tophat[fy1:fy2, fx1:fx2]
    p_max = float(np.percentile(crop, 99.5))
    if p_max > 10:
        crop = np.clip(crop * (255.0 / p_max), 0, 255).astype(np.uint8)
    else:
        crop = bright_stroke_arr[fy1:fy2, fx1:fx2]

    return crop, (fx1, fy1, fx2 - fx1, fy2 - fy1)


def center_digit_on_canvas(
    crop_arr: np.ndarray,
    target_digit_box: int = DIGIT_BOX_SIZE,
    canvas_size: int = CANVAS_SIZE,
    use_center_of_mass: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resizes the cropped digit into a target box (default 20x20) preserving aspect ratio,
    places it on a 28x28 zero canvas, and applies Center of Mass alignment.

    This implements Yann LeCun's exact MNIST specification:
    1. Size normalize the digit bounding box to fit in a 20x20 box preserving aspect ratio.
    2. Place inside 28x28 canvas with antialiased gray values.
    3. Shift the center of mass to (14.0, 14.0).

    Args:
        crop_arr: uint8 cropped digit array with bright stroke.
        target_digit_box: Target bounding dimension (20 for standard MNIST).
        canvas_size: Output canvas dimensions (28x28).
        use_center_of_mass: If True, aligns image Center of Mass to canvas center.

    Returns:
        Tuple of (canvas_28x28_float_array_in_0_1, (com_x, com_y))
    """
    ch, cw = crop_arr.shape
    if ch <= 0 or cw <= 0:
        raise PreprocessingError("Invalid cropped digit dimensions.")

    # Scale to fit inside target_digit_box while preserving aspect ratio
    scale = target_digit_box / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    # Resample with INTER_AREA (downscale) or INTER_CUBIC (upscale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop_arr, (new_w, new_h), interpolation=interp)

    # Initial placement: center bounding box inside canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    start_y = (canvas_size - new_h) // 2
    start_x = (canvas_size - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = (
        resized.astype(np.float32) / 255.0
    )

    # Center of Mass alignment (Canonical MNIST method)
    total_mass = float(np.sum(canvas))
    if total_mass > 1e-4 and use_center_of_mass:
        y_indices, x_indices = np.mgrid[0:canvas_size, 0:canvas_size]
        com_x = float(np.sum(x_indices * canvas) / total_mass)
        com_y = float(np.sum(y_indices * canvas) / total_mass)

        # Desired center is (14.0, 14.0) for a 28x28 image (0-indexed midpoint is 13.5 to 14.0)
        target_cx = (canvas_size - 1) / 2.0
        target_cy = (canvas_size - 1) / 2.0

        shift_x = target_cx - com_x
        shift_y = target_cy - com_y

        # Clamp shift to prevent shifting digits completely off canvas
        max_shift = 4.0
        shift_x = np.clip(shift_x, -max_shift, max_shift)
        shift_y = np.clip(shift_y, -max_shift, max_shift)

        # Affine translation matrix
        m_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_canvas = cv2.warpAffine(
            canvas,
            m_trans,
            (canvas_size, canvas_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Compute updated CoM
        new_mass = float(np.sum(shifted_canvas))
        if new_mass > 1e-4:
            new_com_x = float(np.sum(x_indices * shifted_canvas) / new_mass)
            new_com_y = float(np.sum(y_indices * shifted_canvas) / new_mass)
        else:
            new_com_x, new_com_y = com_x, com_y

        canvas = np.clip(shifted_canvas, 0.0, 1.0)
        return canvas, (new_com_x, new_com_y)

    return np.clip(canvas, 0.0, 1.0), (14.0, 14.0)


def preprocess_image(
    source: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    invert_mode: Literal["auto", "always", "never"] = "auto",
    preset: Literal["auto", "clean", "photo", "faint"] = "auto",
    enhance_contrast: bool = True,
    center_of_mass: bool = True,
) -> PreprocessedDigit:
    """
    Full end-to-end preprocessing pipeline for a single handwritten digit image.

    Pipeline stages:
        1. Safe loading, EXIF orientation, transparency alpha compositing.
        2. Grayscale conversion.
        3. Polarity detection & inversion (dark ink -> bright stroke).
        4. Background flattening / shadow reduction (for photos).
        5. Digit stroke bounding box extraction.
        6. Size normalisation into 20x20 box + 28x28 canvas padding.
        7. Center of Mass translation to (14, 14).
        8. Standard MNIST normalisation: (x - 0.1307) / 0.3081.

    Args:
        source: Input image source.
        invert_mode: 'auto', 'always', or 'never'.
        preset: 'auto', 'clean' (digital / synthetic), 'photo' (paper photo), 'faint' (light pencil).
        enhance_contrast: Apply contrast stretching to make strokes well-defined.
        center_of_mass: Apply canonical center of mass centering.

    Returns:
        PreprocessedDigit dataclass containing tensor, display image, and metadata.

    Raises:
        PreprocessingError: On corrupted inputs or blank images.
    """
    pil_img = load_input_image(source)
    gray = np.array(pil_img.convert("L"))

    # Step 1: Polarity detection & inversion
    bright_stroke, was_inverted = detect_polarity_and_invert(gray, invert_mode=invert_mode)

    # Step 2: Extract active digit bounding box directly from bright_stroke
    # (Top-hat inside extract_digit_bounding_box isolates true handwriting strokes from shadows/gradients)
    cropped, bbox = extract_digit_bounding_box(bright_stroke)

    # Step 3: Local contrast and preset enhancement on the cropped digit
    if preset == "faint":
        norm = cropped.astype(np.float32) / 255.0
        cropped = np.clip(np.power(norm, 0.7) * 255.0, 0, 255).astype(np.uint8)

    if enhance_contrast:
        p98 = float(np.percentile(cropped, 99))
        p2 = float(np.percentile(cropped, 2))
        if p98 > p2 + 20:
            cropped = np.clip((cropped - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

    # Step 4: Stroke dilation for high-res downscaled images:
    # When camera photos / high-res scans are scaled down into 20x20 box,
    # fine ballpoint pen strokes become sub-pixel thin (<0.3 px).
    # Adaptively dilate thin strokes to match standard MNIST stroke width (~2.2 - 2.8 px).
    scale = float(DIGIT_BOX_SIZE) / max(cropped.shape)
    if scale < 0.35:
        dil_k = max(1, int(round(0.40 / scale)))
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
        cropped = cv2.dilate(cropped, k_dil)

    # Step 5: Place on 28x28 canvas & center
    canvas_arr, (com_x, com_y) = center_digit_on_canvas(
        cropped,
        target_digit_box=DIGIT_BOX_SIZE,
        canvas_size=CANVAS_SIZE,
        use_center_of_mass=center_of_mass,
    )

    # Step 6: Create display PIL image (0-255 uint8)
    display_arr = (canvas_arr * 255.0).astype(np.uint8)
    display_img = Image.fromarray(display_arr, mode="L")

    # Step 7: Apply exact training-time normalisation
    # (canvas - mean) / std
    norm_tensor = torch.from_numpy(canvas_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    norm_tensor = (norm_tensor - MNIST_MEAN) / MNIST_STD

    details = {
        "original_size": pil_img.size,
        "inverted": was_inverted,
        "bbox": bbox,
        "center_of_mass": (round(com_x, 2), round(com_y, 2)),
        "preset": preset,
    }

    return PreprocessedDigit(
        tensor=norm_tensor.to(dtype=torch.float32),
        display_image=display_img,
        raw_array=canvas_arr,
        bbox=bbox,
        inverted=was_inverted,
        center_of_mass=(com_x, com_y),
        details=details,
    )


def segment_multi_digits(
    source: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    min_area: int = 40,
    min_height: int = 14,
    margin_ratio: float = 0.12,
    invert_mode: Literal["auto", "always", "never"] = "auto",
) -> Tuple[List[PreprocessedDigit], Image.Image]:
    """
    Experimental multi-digit segmentation pipeline.

    Segments an image containing multiple isolated handwritten digits (e.g. '5729'),
    sorts them left-to-right, and preprocesses each digit into a 28x28 model input.

    Also returns an annotated PIL image showing the detected bounding boxes
    and reading order.

    Limitations:
        - Works best on spatially separated digits.
        - May fail on overlapping, touching, or cursive handwriting.
        - Intended as an experimental feature.

    Args:
        source: Input image source.
        min_area: Minimum bounding box pixel area to filter noise.
        min_height: Minimum bounding box height.
        margin_ratio: Padding margin around each detected box.
        invert_mode: Polarity mode.

    Returns:
        Tuple of (list_of_PreprocessedDigit, annotated_rgb_image)

    Raises:
        PreprocessingError: If no digits could be segmented.
    """
    pil_img = load_input_image(source)
    rgb_arr = np.array(pil_img)
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    # Invert so stroke is bright
    bright_stroke, was_inv = detect_polarity_and_invert(gray, invert_mode=invert_mode)
    flattened = flatten_paper_background(bright_stroke)

    # Threshold for connected components
    _, thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to join small gaps in ink strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_h, img_w = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter small noise or entire image borders
        if area < min_area or h < min_height:
            continue
        if w > img_w * 0.95 and h > img_h * 0.95:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        raise PreprocessingError(
            "No distinct digit regions could be segmented. "
            "Ensure digits are written clearly with some space between them."
        )

    # Sort left-to-right by X coordinate
    candidates.sort(key=lambda b: b[0])

    # Merge overlapping or nested bounding boxes if any
    merged = []
    for b in candidates:
        if not merged:
            merged.append(b)
        else:
            prev_x, prev_y, prev_w, prev_h = merged[-1]
            cur_x, cur_y, cur_w, cur_h = b
            # If significant horizontal overlap (> 50%)
            overlap_x = max(0, min(prev_x + prev_w, cur_x + cur_w) - max(prev_x, cur_x))
            if overlap_x > 0.5 * min(prev_w, cur_w):
                # Merge boxes
                nx1 = min(prev_x, cur_x)
                ny1 = min(prev_y, cur_y)
                nx2 = max(prev_x + prev_w, cur_x + cur_w)
                ny2 = max(prev_y + prev_h, cur_y + cur_h)
                merged[-1] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
            else:
                merged.append(b)

    results: List[PreprocessedDigit] = []
    annotated = rgb_arr.copy()

    for idx, (x, y, w, h) in enumerate(merged):
        # Add margin
        mx = max(2, int(w * margin_ratio))
        my = max(2, int(h * margin_ratio))
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_w, x + w + mx)
        y2 = min(img_h, y + h + my)

        digit_crop = flattened[y1:y2, x1:x2]
        canvas_arr, (cx, cy) = center_digit_on_canvas(digit_crop, DIGIT_BOX_SIZE, CANVAS_SIZE)

        norm_tensor = torch.from_numpy(canvas_arr).unsqueeze(0).unsqueeze(0)
        norm_tensor = (norm_tensor - MNIST_MEAN) / MNIST_STD

        disp_img = Image.fromarray((canvas_arr * 255.0).astype(np.uint8), mode="L")

        results.append(
            PreprocessedDigit(
                tensor=norm_tensor.to(dtype=torch.float32),
                display_image=disp_img,
                raw_array=canvas_arr,
                bbox=(x, y, w, h),
                inverted=was_inv,
                center_of_mass=(cx, cy),
                details={"order_index": idx, "bbox": (x, y, w, h)},
            )
        )

        # Draw green bounding box and order badge on annotated image
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (16, 185, 129), 2)
        label_text = f"#{idx + 1}"
        cv2.putText(
            annotated,
            label_text,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (16, 185, 129),
            2,
            cv2.LINE_AA,
        )

    return results, Image.fromarray(annotated)
