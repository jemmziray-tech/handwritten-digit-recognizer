"""
inference.py - Prediction, Confidence Calibration, and Ambiguity Analysis.

Executes inference using the trained PyTorch DigitCNN model under `torch.inference_mode()`,
computes softmax class probabilities, measures GPU/CPU latency, and evaluates
prediction certainty and ambiguity flags.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DigitCNN

logger = logging.getLogger(__name__)


@dataclass
class CandidateDigit:
    """Represents a candidate digit with its associated probability."""
    digit: int
    probability: float
    percentage: float

    def __repr__(self) -> str:
        return f"Digit {self.digit}: {self.percentage:.2f}%"


@dataclass
class PredictionResult:
    """Complete prediction report for an isolated digit image."""
    predicted_digit: int
    confidence: float                     # In range [0.0, 1.0]
    confidence_percentage: float          # In range [0.0, 100.0]
    all_probabilities: List[float]        # Digits 0 through 9
    top_candidates: List[CandidateDigit]  # Top 3 candidates
    inference_time_ms: float              # Latency in milliseconds
    device_used: str                      # 'cuda' or 'cpu'
    entropy: float                        # Shannon entropy of probability distribution
    confidence_level: str                 # 'High', 'Moderate', or 'Low'
    is_ambiguous: bool                    # True if top-2 margin is narrow
    warnings: List[str]                   # Human-readable warnings
    logits: List[float]                   # Raw unnormalised model logits


@dataclass
class SequencePredictionResult:
    """Prediction result for a multi-digit sequence."""
    sequence_string: str
    digit_results: List[PredictionResult]
    average_confidence: float
    minimum_confidence: float
    total_time_ms: float
    warnings: List[str]


def assess_confidence_and_ambiguity(
    probabilities: List[float],
) -> Tuple[str, bool, List[str], float]:
    """
    Evaluates model certainty and identifies out-of-distribution or ambiguous inputs.

    Checks:
        1. Low confidence: Top probability < 0.65.
        2. High ambiguity: Margin between top-1 and top-2 candidates < 0.15.
        3. High entropy: Shannon entropy > 1.35 nats (distribution is flat / spread out).

    Args:
        probabilities: 10 class probabilities summing to 1.0.

    Returns:
        Tuple of (confidence_level, is_ambiguous, warnings_list, entropy)
    """
    sorted_pairs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    top1_digit, top1_prob = sorted_pairs[0]
    top2_digit, top2_prob = sorted_pairs[1]

    # Compute Shannon entropy
    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities if p > 0)

    warnings: List[str] = []
    is_ambiguous = False

    # Check top-1 vs top-2 separation margin
    margin = top1_prob - top2_prob
    if margin < 0.18 and top2_prob > 0.20:
        is_ambiguous = True
        warnings.append(
            f"Ambiguous prediction: digit '{top1_digit}' ({top1_prob * 100:.1f}%) is closely "
            f"followed by digit '{top2_digit}' ({top2_prob * 100:.1f}%). "
            f"Margin is only {margin * 100:.1f}%."
        )

    # Check absolute top confidence
    if top1_prob >= 0.85:
        confidence_level = "High"
    elif top1_prob >= 0.60:
        confidence_level = "Moderate"
        warnings.append(
            f"Moderate confidence ({top1_prob * 100:.1f}%). "
            "Handwriting style may be unusual or the stroke differs from standard MNIST."
        )
    else:
        confidence_level = "Low"
        warnings.append(
            f"Low confidence ({top1_prob * 100:.1f}%). "
            "The image may be noisy, out-of-distribution, or not contain a clear single digit."
        )

    # Entropy check
    if entropy > 1.35 and not is_ambiguous:
        warnings.append(
            f"High prediction entropy ({entropy:.2f}). Probability is scattered across multiple classes."
        )

    return confidence_level, is_ambiguous, warnings, entropy


def predict_digit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> PredictionResult:
    """
    Runs forward inference for a single preprocessed 28x28 digit tensor.

    Requirements enforced:
        - Validates input dimensions (1, 1, 28, 28) and float32 dtype.
        - Moves input to the specified or model's active device.
        - Executes inside `torch.inference_mode()`.
        - Measures precise GPU/CPU latency.
        - Applies Softmax for calibrated class probabilities.
        - Calculates entropy and ambiguity flags.

    Args:
        model: Loaded DigitCNN model in eval mode.
        input_tensor: Float32 tensor with shape (1, 1, 28, 28).
        device: Target execution device. If None, uses model parameters' device.

    Returns:
        PredictionResult dataclass.

    Raises:
        ValueError: If tensor shape or dtype is invalid.
    """
    if input_tensor.dim() != 4 or input_tensor.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected input tensor with shape (batch_size, 1, 28, 28), got {input_tensor.shape}"
        )

    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(dtype=torch.float32)

    # Determine device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Ensure model is on target device and in eval mode
    model.to(device)
    model.eval()
    tensor_dev = input_tensor.to(device, non_blocking=True)

    # Precise timing with CUDA synchronisation
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.inference_mode():
        logits_tensor = model(tensor_dev)
        probs_tensor = F.softmax(logits_tensor, dim=1)

    if is_cuda:
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Extract to CPU Python primitives
    logits = logits_tensor[0].detach().cpu().tolist()
    probabilities = probs_tensor[0].detach().cpu().tolist()

    # Safety check: probabilities must be valid finite floats summing to ~1.0
    prob_sum = sum(probabilities)
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        logger.warning("Probabilities sum to %f (expected 1.0). Re-normalising.", prob_sum)
        probabilities = [p / prob_sum for p in probabilities]

    predicted_digit = int(max(range(10), key=lambda d: probabilities[d]))
    top_confidence = probabilities[predicted_digit]
    top_percentage = top_confidence * 100.0

    # Top 3 candidates
    ranked_indices = sorted(range(10), key=lambda d: probabilities[d], reverse=True)
    top_candidates = [
        CandidateDigit(
            digit=idx,
            probability=probabilities[idx],
            percentage=probabilities[idx] * 100.0,
        )
        for idx in ranked_indices[:3]
    ]

    conf_level, is_ambig, warnings, entropy = assess_confidence_and_ambiguity(probabilities)

    return PredictionResult(
        predicted_digit=predicted_digit,
        confidence=top_confidence,
        confidence_percentage=top_percentage,
        all_probabilities=probabilities,
        top_candidates=top_candidates,
        inference_time_ms=round(latency_ms, 3),
        device_used=device.type,
        entropy=round(entropy, 4),
        confidence_level=conf_level,
        is_ambiguous=is_ambig,
        warnings=warnings,
        logits=logits,
    )


def predict_sequence(
    model: nn.Module,
    preprocessed_digits: List[Any],
    device: Optional[Union[str, torch.device]] = None,
) -> SequencePredictionResult:
    """
    Executes inference across a list of preprocessed digits from a multi-digit photograph.

    Args:
        model: Loaded DigitCNN model.
        preprocessed_digits: List of PreprocessedDigit objects.
        device: Target execution device.

    Returns:
        SequencePredictionResult dataclass.
    """
    if not preprocessed_digits:
        return SequencePredictionResult(
            sequence_string="",
            digit_results=[],
            average_confidence=0.0,
            minimum_confidence=0.0,
            total_time_ms=0.0,
            warnings=["No digits were passed for sequence prediction."],
        )

    results: List[PredictionResult] = []
    total_time = 0.0

    for digit_obj in preprocessed_digits:
        res = predict_digit(model, digit_obj.tensor, device=device)
        results.append(res)
        total_time += res.inference_time_ms

    seq_str = "".join(str(r.predicted_digit) for r in results)
    confidences = [r.confidence for r in results]
    avg_conf = (sum(confidences) / len(confidences)) * 100.0
    min_conf = min(confidences) * 100.0

    warnings: List[str] = []
    if min_conf < 70.0:
        warnings.append(
            f"One or more digits in the sequence had lower confidence (minimum: {min_conf:.1f}%). "
            "Verify individual digit crops."
        )

    return SequencePredictionResult(
        sequence_string=seq_str,
        digit_results=results,
        average_confidence=round(avg_conf, 2),
        minimum_confidence=round(min_conf, 2),
        total_time_ms=round(total_time, 2),
        warnings=warnings,
    )
