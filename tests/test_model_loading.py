"""
test_model_loading.py - Unit and Integration Tests for Model Loading and Architecture.

Verifies:
- Checkpoint discovery and validation.
- Model architecture reconstruction and tensor shapes.
- Strict state_dict loading.
- Safe CPU fallback and CUDA execution.
- Evaluation mode and gradient disabling.
- Error handling on missing or incompatible checkpoints.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

import model


def test_checkpoint_exists_and_valid():
    """Verify the trained checkpoint exists and has valid metadata."""
    ckpt_path = model.resolve_model_path()
    assert ckpt_path.is_file(), f"Checkpoint not found at {ckpt_path}"
    assert ckpt_path.stat().st_size > 1_000_000, "Checkpoint file suspiciously small"


def test_model_architecture_shapes():
    """Instantiate DigitCNN and verify all layer specifications and forward output shape."""
    net = model.DigitCNN()
    assert isinstance(net.conv1, nn.Conv2d)
    assert net.conv1.in_channels == 1
    assert net.conv1.out_channels == 32
    assert net.conv2.out_channels == 64
    assert net.fc1.out_features == 128
    assert net.fc2.out_features == 10

    # Dummy batch of 4 images
    dummy_input = torch.randn(4, 1, 28, 28)
    output = net(dummy_input)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_model_loading_cpu():
    """Test loading the existing checkpoint explicitly onto CPU."""
    net, meta, dev = model.load_model(device="cpu")
    assert dev.type == "cpu"
    assert not net.training, "Model should be in eval() mode"
    for p in net.parameters():
        assert not p.requires_grad, "Parameters should have requires_grad=False"

    dummy_input = torch.randn(1, 1, 28, 28, device="cpu")
    with torch.inference_mode():
        out = net(dummy_input)
    assert out.shape == (1, 10)
    assert "test_accuracy" in meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this system")
def test_model_loading_cuda():
    """Test loading the existing checkpoint onto CUDA."""
    net, meta, dev = model.load_model(device="cuda")
    assert dev.type == "cuda"
    assert "gpu_name" in meta
    dummy_input = torch.randn(2, 1, 28, 28, device="cuda")
    with torch.inference_mode():
        out = net(dummy_input)
    assert out.shape == (2, 10)
    assert out.device.type == "cuda"


def test_missing_checkpoint_raises_error(tmp_path):
    """Test that specifying a non-existent checkpoint path raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent_model.pth"
    with pytest.raises(FileNotFoundError):
        model.load_model(model_path=missing_file)


def test_strict_loading_mismatch_fails(tmp_path):
    """Verify that an incompatible checkpoint raises RuntimeError during strict loading."""
    fake_ckpt = tmp_path / "fake_model.pth"
    # Save checkpoint with missing keys
    torch.save({"model_state_dict": {"conv1.weight": torch.randn(32, 1, 3, 3)}}, fake_ckpt)

    with pytest.raises(RuntimeError):
        model.load_model(model_path=fake_ckpt, device="cpu")
