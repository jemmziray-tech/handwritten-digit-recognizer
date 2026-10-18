"""
model.py - CNN Architecture and Model Loader for MNIST Digit Recognition.

Loads and serves the trained PyTorch DigitCNN model checkpoint with strict
state_dict verification, CUDA acceleration, and safe CPU fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Standard MNIST normalisation constants matching training
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081
INPUT_CHANNELS: int = 1
INPUT_HEIGHT: int = 28
INPUT_WIDTH: int = 28
NUM_CLASSES: int = 10


class DigitCNN(nn.Module):
    """
    Exact PyTorch Convolutional Neural Network architecture used to train
    `mnist_digit_cnn.pth`.

    Architecture:
        - Conv2d(1 -> 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)  [28x28 -> 14x14]
        - Conv2d(32 -> 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2) [14x14 -> 7x7]
        - Flatten(64 * 7 * 7 = 3136)
        - Linear(3136 -> 128) + ReLU + Dropout(0.25)
        - Linear(128 -> 10) [raw logits]
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single-channel 28x28 digit images.

        Args:
            x: Tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Logits tensor of shape (batch_size, 10)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_default_device(preferred: Optional[str] = None) -> torch.device:
    """
    Determine the optimal execution device.

    Args:
        preferred: Optional preference ('cuda', 'cpu', or 'auto').

    Returns:
        torch.device configured for CUDA if available/requested, or CPU.
    """
    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    # Auto mode: use CUDA when available
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_model_path(model_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locates the `mnist_digit_cnn.pth` checkpoint robustly.

    Search behavior:
        1. If explicit `model_path` is passed, it is strictly validated.
        2. If `model_path` is None, searches environment variable `MNIST_MODEL_PATH`,
           `<project_root>/models/mnist_digit_cnn.pth`, the original lab path,
           and current working directory.

    Raises:
        FileNotFoundError: If the checkpoint cannot be found.
    """
    if model_path is not None:
        explicit_path = Path(model_path)
        if explicit_path.is_file():
            return explicit_path.resolve()
        raise FileNotFoundError(
            f"Specified model checkpoint was not found at '{model_path}'."
        )

    candidate_paths = []

    env_path = os.environ.get("MNIST_MODEL_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    # Project-relative path
    base_dir = Path(__file__).resolve().parent
    candidate_paths.extend([
        base_dir / "models" / "mnist_digit_cnn.pth",
        base_dir / "mnist_digit_cnn.pth",
        Path("C:/Users/jemmz/deep-learning-lab/mnist_digit_cnn.pth"),
        Path.cwd() / "models" / "mnist_digit_cnn.pth",
        Path.cwd() / "mnist_digit_cnn.pth",
    ])

    for p in candidate_paths:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue

    searched = "\n  - ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        f"Could not locate the trained model checkpoint `mnist_digit_cnn.pth`.\n"
        f"Searched paths:\n  - {searched}\n"
        f"Please provide the path or set the MNIST_MODEL_PATH environment variable."
    )


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[DigitCNN, Dict[str, Any], torch.device]:
    """
    Instantiates and loads the DigitCNN model from the verified checkpoint.

    Requirements enforced:
        - Exact architecture match with strict state_dict verification.
        - Evaluation mode (model.eval()).
        - Gradient computation disabled.
        - Safe fallback between CUDA and CPU.
        - Never silently uses random uninitialised weights.

    Args:
        model_path: Optional explicit path to `mnist_digit_cnn.pth`.
        device: Target torch device or device string ('cuda', 'cpu', 'auto').

    Returns:
        Tuple of (model, metadata_dict, target_device)

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If checkpoint loading fails or state_dict keys mismatch.
    """
    resolved_path = resolve_model_path(model_path)

    if isinstance(device, str):
        target_device = get_default_device(device)
    elif isinstance(device, torch.device):
        target_device = device
    else:
        target_device = get_default_device()

    logger.info("Loading checkpoint from: %s onto %s", resolved_path, target_device)

    try:
        # Load checkpoint dictionary safely
        checkpoint = torch.load(
            resolved_path,
            map_location=target_device,
            weights_only=True,
        )
    except Exception as exc:
        # Retry with weights_only=False if custom pickling occurs, but strictly handle error
        try:
            checkpoint = torch.load(resolved_path, map_location=target_device, weights_only=False)
        except Exception as retry_exc:
            raise RuntimeError(
                f"Failed to load checkpoint file from '{resolved_path}': {retry_exc}"
            ) from retry_exc

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(resolved_path),
        "file_size_bytes": resolved_path.stat().st_size,
    }

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["test_accuracy"] = checkpoint.get("test_accuracy")
        metadata["epochs"] = checkpoint.get("epochs")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Checkpoint at '{resolved_path}' has unexpected type {type(checkpoint)}. "
            "Expected dict or state_dict."
        )

    # Instantiate model
    model = DigitCNN()

    # Strict loading - will fail loudly if architecture does not match
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        raise RuntimeError(
            f"Architecture mismatch when loading checkpoint '{resolved_path}': {err}"
        ) from err

    model.to(target_device)
    model.eval()

    # Freeze weights for inference
    for param in model.parameters():
        param.requires_grad = False

    metadata["device"] = str(target_device)
    if target_device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(target_device)
        metadata["cuda_capability"] = torch.cuda.get_device_capability(target_device)

    logger.info("Model loaded successfully. Metadata: %s", metadata)
    return model, metadata, target_device
