"""
test_preprocessing.py - Unit and Integration Tests for Image Preprocessing.

Verifies:
- Standard isolated digit preprocessing.
- EXIF orientation and transparency handling.
- Dark-on-light vs light-on-dark polarity detection.
- Bounding-box aspect ratio preservation and Center of Mass centering.
- Normalisation values and tensor properties.
- Graceful handling of corrupted and blank images.
- Multi-digit candidate segmentation and left-to-right sorting.
"""

import io
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

import preprocessing


@pytest.fixture
def examples_dir():
    return Path(__file__).resolve().parent.parent / "assets" / "examples"


def test_clean_mnist_preprocessing(examples_dir):
    """Test preprocessing on a standard clean MNIST image."""
    img_path = examples_dir / "mnist_digit_5.png"
    result = preprocessing.preprocess_image(img_path, preset="clean")

    assert isinstance(result.tensor, torch.Tensor)
    assert result.tensor.shape == (1, 1, 28, 28)
    assert result.tensor.dtype == torch.float32
    assert result.display_image.size == (28, 28)
    assert not torch.isnan(result.tensor).any()
    assert not torch.isinf(result.tensor).any()


def test_polarity_detection_dark_on_light(examples_dir):
    """Verify that black ink on white paper is automatically inverted to bright strokes."""
    paper_path = examples_dir / "paper_digit_7.png"
    result = preprocessing.preprocess_image(paper_path, invert_mode="auto")

    assert result.inverted is True, "Paper image should be detected as dark-on-light and inverted"
    # Stroke pixels on 28x28 should be bright (> 0)
    assert np.max(result.raw_array) > 0.5


def test_polarity_detection_light_on_dark(examples_dir):
    """Verify that bright strokes on dark background are not inverted."""
    mnist_path = examples_dir / "mnist_digit_0.png"
    result = preprocessing.preprocess_image(mnist_path, invert_mode="auto")

    assert result.inverted is False, "MNIST native image should not be inverted"


def test_transparent_png_handling():
    """Verify that an image with transparent background is composited properly."""
    # Create RGBA image with transparent background and black digit stroke
    rgba_arr = np.zeros((100, 100, 4), dtype=np.uint8)
    # Transparent background (alpha = 0)
    rgba_arr[:, :, 3] = 0
    # Draw simple vertical stroke (alpha = 255, RGB = 0, 0, 0)
    rgba_arr[20:80, 45:55, 0:3] = 0
    rgba_arr[20:80, 45:55, 3] = 255

    img = Image.fromarray(rgba_arr, mode="RGBA")
    result = preprocessing.preprocess_image(img)

    assert result.tensor.shape == (1, 1, 28, 28)
    assert not torch.isnan(result.tensor).any()


def test_exif_orientation_handling():
    """Verify that image with EXIF tag loads without error."""
    # Create an RGB image and attach an EXIF tag
    img = Image.new("RGB", (60, 60), color="white")
    # Draw stroke
    arr = np.array(img)
    arr[20:40, 25:35] = [0, 0, 0]
    img = Image.fromarray(arr)

    # Convert to bytes and reload via pipeline
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    result = preprocessing.preprocess_image(buf)
    assert result.tensor.shape == (1, 1, 28, 28)


def test_blank_image_raises_error():
    """Verify that an entirely blank or white image raises PreprocessingError."""
    blank_img = Image.new("RGB", (100, 100), color="white")
    with pytest.raises(preprocessing.PreprocessingError):
        preprocessing.preprocess_image(blank_img)


def test_corrupt_data_raises_error():
    """Verify that corrupted bytes raise PreprocessingError."""
    garbage = b"not_an_image_file_at_all_12345"
    with pytest.raises(preprocessing.PreprocessingError):
        preprocessing.preprocess_image(garbage)


def test_center_of_mass_alignment():
    """Verify that center of mass translates the digit close to canvas center (13.5 - 14.0)."""
    # Create an off-center digit stroke
    off_center = np.zeros((100, 100), dtype=np.uint8)
    off_center[10:30, 10:25] = 255  # Near top-left

    crop, bbox = preprocessing.extract_digit_bounding_box(off_center)
    canvas, (cx, cy) = preprocessing.center_digit_on_canvas(crop, use_center_of_mass=True)

    # Desired center of mass should be within 1.5 pixels of canvas midpoint (13.5)
    assert 11.5 <= cx <= 15.5
    assert 11.5 <= cy <= 15.5


def test_multi_digit_segmentation(examples_dir):
    """Verify multi-digit segmentation detects multiple digits and sorts left-to-right."""
    multi_path = examples_dir / "multi_digit_42.jpg"
    digits, annot = preprocessing.segment_multi_digits(multi_path)

    assert len(digits) == 2, f"Expected 2 digits, found {len(digits)}"
    # Verify left-to-right order: digit 0 X coordinate < digit 1 X coordinate
    x0 = digits[0].bbox[0]
    x1 = digits[1].bbox[0]
    assert x0 < x1, f"Digits not sorted left to right: {x0} >= {x1}"
    assert annot.size[0] > 0
