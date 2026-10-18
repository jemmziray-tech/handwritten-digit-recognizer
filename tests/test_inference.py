"""
test_inference.py - Unit and Integration Tests for Model Inference and Calibration.

Verifies:
- Softmax output validity (sum to 1.0, finite, non-negative).
- Correct predicted class and top-3 candidate ranking.
- Latency measurement on CPU and CUDA.
- Ambiguity detection and confidence calibration flags.
- Sequence prediction for multi-digit recognition.
- High accuracy across test samples for digits 0 through 9.
"""

from pathlib import Path
import pytest
import torch

import inference
import model
import preprocessing


@pytest.fixture(scope="module")
def loaded_model():
    net, meta, dev = model.load_model()
    return net, meta, dev


@pytest.fixture
def examples_dir():
    return Path(__file__).resolve().parent.parent / "assets" / "examples"


def test_prediction_output_properties(loaded_model, examples_dir):
    """Verify inference output structure, probability sum, and valid types."""
    net, meta, dev = loaded_model
    img_path = examples_dir / "mnist_digit_3.png"
    prep = preprocessing.preprocess_image(img_path, preset="clean")

    res = inference.predict_digit(net, prep.tensor, dev)

    assert isinstance(res.predicted_digit, int)
    assert 0 <= res.predicted_digit <= 9
    assert len(res.all_probabilities) == 10
    assert pytest.approx(sum(res.all_probabilities), abs=1e-3) == 1.0
    for p in res.all_probabilities:
        assert 0.0 <= p <= 1.0
    assert len(res.top_candidates) == 3
    # Top candidates must be strictly sorted in descending probability
    assert res.top_candidates[0].probability >= res.top_candidates[1].probability
    assert res.top_candidates[1].probability >= res.top_candidates[2].probability
    assert res.inference_time_ms > 0.0


def test_cpu_inference(loaded_model, examples_dir):
    """Verify that inference executes successfully when forced onto CPU."""
    net, meta, _ = loaded_model
    img_path = examples_dir / "mnist_digit_1.png"
    prep = preprocessing.preprocess_image(img_path, preset="clean")

    res = inference.predict_digit(net, prep.tensor, device="cpu")
    assert res.device_used == "cpu"
    assert res.predicted_digit == 1
    assert res.confidence > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this system")
def test_cuda_inference(loaded_model, examples_dir):
    """Verify that inference executes successfully on CUDA GPU."""
    net, meta, _ = loaded_model
    img_path = examples_dir / "mnist_digit_8.png"
    prep = preprocessing.preprocess_image(img_path, preset="clean")

    res = inference.predict_digit(net, prep.tensor, device="cuda")
    assert res.device_used == "cuda"
    assert res.predicted_digit == 8
    assert res.confidence > 0.95


def test_all_digits_accuracy(loaded_model, examples_dir):
    """Verify that the loaded model correctly classifies clean test samples for digits 0-9."""
    net, meta, dev = loaded_model
    correct = 0

    for digit in range(10):
        img_path = examples_dir / f"mnist_digit_{digit}.png"
        prep = preprocessing.preprocess_image(img_path, preset="clean")
        res = inference.predict_digit(net, prep.tensor, dev)
        if res.predicted_digit == digit:
            correct += 1

    assert correct == 10, f"Expected 10/10 accuracy on clean test samples, got {correct}/10"


def test_ambiguity_assessment():
    """Verify that close probabilities trigger the ambiguity flag."""
    # Artificially close probabilities between 3 (45%) and 8 (42%)
    probs = [0.01, 0.01, 0.02, 0.45, 0.01, 0.03, 0.01, 0.02, 0.42, 0.02]
    conf_level, is_ambig, warnings, entropy = inference.assess_confidence_and_ambiguity(probs)

    assert is_ambig is True, "Close probabilities should trigger ambiguity"
    assert any("Ambiguous prediction" in w for w in warnings)


def test_sequence_prediction(loaded_model, examples_dir):
    """Verify multi-digit sequence prediction on '42'."""
    net, meta, dev = loaded_model
    multi_path = examples_dir / "multi_digit_42.jpg"
    digits, annot = preprocessing.segment_multi_digits(multi_path)

    seq_res = inference.predict_sequence(net, digits, dev)
    assert seq_res.sequence_string == "42"
    assert len(seq_res.digit_results) == 2
    assert seq_res.average_confidence > 90.0
