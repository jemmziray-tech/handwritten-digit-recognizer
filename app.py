"""
app.py - Production-Quality Streamlit Web Application for MNIST Digit Recognition.

Uses a trained PyTorch Convolutional Neural Network (DigitCNN) with CUDA acceleration,
robust image preprocessing for phone photographs, Center-of-Mass alignment,
and experimental multi-digit sequence segmentation.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import inference
import model
import preprocessing

# Hot reload modules on each Streamlit rerun
importlib.reload(preprocessing)
importlib.reload(inference)
importlib.reload(model)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, professional aesthetics
st.markdown(
    """
    <style>
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .hero-container {
        padding: 1.5rem 0rem 1rem 0rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 0.8rem;
    }
    .pred-card {
        background: rgba(16, 185, 129, 0.06);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .pred-digit-large {
        font-size: 5.5rem;
        font-weight: 900;
        line-height: 1.0;
        color: #10b981;
        margin: 0.2rem 0;
    }
    .pred-meta-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #10b981;
        color: white;
    }
    .pred-card-ambig {
        background: rgba(245, 158, 11, 0.08);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .pred-card-ambig .pred-digit-large {
        color: #f59e0b;
    }
    .pred-card-ambig .pred-meta-badge {
        background: #f59e0b;
    }
    .stat-box {
        background: rgba(128, 128, 128, 0.07);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .pixel-zoom {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
        border: 2px solid rgba(128, 128, 128, 0.3);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading PyTorch model checkpoint...")
def get_cached_model(device_preference: str = "auto") -> Tuple[model.DigitCNN, dict, torch.device]:
    """
    Loads and caches the model instance using Streamlit's resource cache
    to ensure zero reload overhead across interactions.
    """
    net, meta, dev = model.load_model(device=device_preference)
    return net, meta, dev


# ==============================================================================
# Sidebar Configuration & Hardware Info
# ==============================================================================
st.sidebar.title("⚙️ Configuration")

device_choice = st.sidebar.selectbox(
    "Inference Device",
    options=["auto", "cuda", "cpu"],
    format_func=lambda x: {
        "auto": "Auto-detect (CUDA if available)",
        "cuda": "Force GPU (CUDA)",
        "cpu": "Force CPU",
    }[x],
    help="Select whether PyTorch executes on your NVIDIA GPU or CPU.",
)

# Load model
try:
    net, model_metadata, active_device = get_cached_model(device_choice)
    model_loaded = True
except Exception as e:
    model_loaded = False
    net, model_metadata, active_device = None, {}, torch.device("cpu")
    st.sidebar.error(f"Failed to load model: {e}")

# Device and Model Badges
st.sidebar.markdown("### 🖥️ Hardware Telemetry")
if active_device.type == "cuda":
    gpu_name = model_metadata.get("gpu_name", "NVIDIA GPU")
    st.sidebar.success(f"🟢 **CUDA Active**: {gpu_name}")
else:
    st.sidebar.info("🔵 **CPU Mode Active**")

st.sidebar.markdown("### 🧠 Model Architecture")
st.sidebar.markdown(
    f"""
    - **Type**: 2-stage CNN + Linear Classifier
    - **Parameters**: 421,642 trainable
    - **Training Set Accuracy**: 99.42% (5 epochs)
    - **Input**: 1×28×28 Grayscale
    - **Checkpoint**: `{Path(model_metadata.get('checkpoint_path', 'mnist_digit_cnn.pth')).name}`
    """
)

# Mode Selector
st.sidebar.markdown("### 🎯 Application Mode")
app_mode = st.sidebar.radio(
    "Select Mode",
    options=["Single Digit Recognition", "Multi-Digit Sequence (Experimental)"],
    help="Choose between single isolated digit classification and multi-digit sequence segmentation.",
)

# Preprocessing Controls in Sidebar
st.sidebar.markdown("### 🎛️ Preprocessing Settings")
preset = st.sidebar.selectbox(
    "Preprocessing Preset",
    options=["auto", "photo", "clean", "faint"],
    format_func=lambda x: {
        "auto": "Auto Adaptive (Balanced)",
        "photo": "Paper Photograph (Shadow / Grain Flattening)",
        "clean": "Clean Scan / Digital Drawing",
        "faint": "Faint Pencil / Thin Stroke (Contrast Boost)",
    }[x],
    help="Tuning for different handwriting mediums and paper illumination.",
)

invert_mode = st.sidebar.selectbox(
    "Polarity Inversion",
    options=["auto", "always", "never"],
    format_func=lambda x: {
        "auto": "Auto-detect (Paper ink -> Invert)",
        "always": "Always Invert (Force Dark to Bright)",
        "never": "Never Invert (Input is already bright on black)",
    }[x],
    help="MNIST requires white digits on black background. Auto-detect samples border pixels.",
)

center_of_mass_toggle = st.sidebar.checkbox(
    "Center of Mass Alignment",
    value=True,
    help="Translates digit center of mass to (14, 14) following Yann LeCun's canonical MNIST specification.",
)

contrast_boost_toggle = st.sidebar.checkbox(
    "Contrast Enhancement",
    value=True,
    help="Applies percentile contrast stretching to isolate strokes cleanly.",
)


# ==============================================================================
# Header Section
# ==============================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🔢 MNIST Digit Recognition</div>
        <div class="hero-subtitle">Production-quality handwritten digit recognition powered by PyTorch CNN & CUDA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(
        "⚠️ The model checkpoint could not be loaded. Please ensure `mnist_digit_cnn.pth` "
        "is present in the `models/` directory."
    )
    st.stop()


# ==============================================================================
# Image Input Section: Upload or Gallery
# ==============================================================================
input_tab_upload, input_tab_gallery = st.tabs(["📤 Upload Image", "🖼️ Sample Gallery"])

selected_image: Optional[Image.Image] = None
source_name: str = ""

with input_tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image or phone camera photograph",
        type=["png", "jpg", "jpeg", "webp"],
        help="Supports phone photos, notebook paper, scans, sketches, and digital drawings.",
        key="uploader",
    )
    if uploaded_file is not None:
        try:
            selected_image = preprocessing.load_input_image(uploaded_file.getvalue())
            source_name = uploaded_file.name
        except Exception as err:
            st.error(f"Could not load image: {err}")

with input_tab_gallery:
    st.markdown("Select a verified test sample to see immediate recognition:")
    examples_dir = Path(__file__).resolve().parent / "assets" / "examples"

    col_cat1, col_cat2, col_cat3 = st.columns(3)

    with col_cat1:
        st.markdown("**Phone Camera Photos on Paper**")
        sample_photos = ["phone_photo_digit_3.jpg", "phone_photo_digit_7.jpg", "phone_photo_digit_5.jpg"]
        for sp in sample_photos:
            p = examples_dir / sp
            if p.is_file():
                if st.button(f"📷 {sp}", key=f"btn_{sp}", use_container_width=True):
                    selected_image = preprocessing.load_input_image(p)
                    source_name = sp

    with col_cat2:
        st.markdown("**Clean MNIST Samples**")
        sample_mnist = [f"mnist_digit_{d}.png" for d in [0, 1, 2, 4, 8]]
        for sm in sample_mnist:
            p = examples_dir / sm
            if p.is_file():
                if st.button(f"🔢 {sm}", key=f"btn_{sm}", use_container_width=True):
                    selected_image = preprocessing.load_input_image(p)
                    source_name = sm

    with col_cat3:
        st.markdown("**Multi-Digit Sequences**")
        sample_multis = ["multi_digit_42.jpg", "multi_digit_789.jpg"]
        for smu in sample_multis:
            p = examples_dir / smu
            if p.is_file():
                if st.button(f"🔤 {smu}", key=f"btn_{smu}", use_container_width=True):
                    selected_image = preprocessing.load_input_image(p)
                    source_name = smu

# Clear action
if selected_image is not None:
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear / Reset Image", use_container_width=True):
        st.rerun()


# ==============================================================================
# Inference Execution & Results
# ==============================================================================
if selected_image is None:
    st.info("👆 Upload an image or select a sample from the gallery above to start digit recognition.")
    st.stop()

st.markdown("---")

# ------------------------------------------------------------------------------
# Mode A: Single Digit Recognition
# ------------------------------------------------------------------------------
if app_mode == "Single Digit Recognition":
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📷 Input & Preprocessing")

        # Run preprocessing
        try:
            with st.spinner("Preprocessing image for CNN..."):
                preprocessed = preprocessing.preprocess_image(
                    selected_image,
                    invert_mode=invert_mode,
                    preset=preset,
                    enhance_contrast=contrast_boost_toggle,
                    center_of_mass=center_of_mass_toggle,
                )
        except preprocessing.PreprocessingError as prep_err:
            st.error(f"⚠️ Preprocessing Error: {prep_err}")
            st.info("💡 Try adjusting the Preprocessing Preset or Polarity Inversion in the sidebar.")
            st.stop()

        # Display input image with detected bounding box overlay
        display_input = selected_image.copy()
        if preprocessed.bbox is not None and max(selected_image.size) > 64:
            import cv2
            bx, by, bw, bh = preprocessed.bbox
            img_np = np.array(display_input.convert("RGB"))
            line_thick = max(2, int(min(img_np.shape[:2]) * 0.005))
            cv2.rectangle(img_np, (bx, by), (bx + bw, by + bh), (16, 185, 129), line_thick)
            display_input = Image.fromarray(img_np)
            st.image(
                display_input,
                caption=f"Detected Digit ROI: ({bx}, {by}, {bw} × {bh} px)",
                use_container_width=True,
            )
        else:
            st.image(
                selected_image,
                caption=f"Original Image: {source_name or 'Input Image'}",
                use_container_width=True,
            )

        # Display preprocessed 28x28 image
        st.markdown("#### 🔬 Model Input (28 × 28 Grayscale)")
        col_img28, col_info = st.columns([1, 1])
        with col_img28:
            # Upscale 28x28 image with nearest-neighbor for sharp pixel display
            zoom_factor = 7
            zoomed_img = preprocessed.display_image.resize(
                (28 * zoom_factor, 28 * zoom_factor),
                Image.Resampling.NEAREST,
            )
            st.image(
                zoomed_img,
                caption="Canonical 28×28 Canvas (Magnified 7×)",
                use_container_width=True,
            )

        with col_info:
            st.markdown(
                f"""
                - **Canvas**: 28 × 28 px
                - **Polarity Inverted**: {'Yes (Dark-on-Light)' if preprocessed.inverted else 'No'}
                - **Center of Mass**: ({preprocessed.center_of_mass[0]:.1f}, {preprocessed.center_of_mass[1]:.1f})
                - **Bounding Box**: {preprocessed.bbox}
                - **Normalised Range**: [{preprocessed.tensor.min().item():.2f}, {preprocessed.tensor.max().item():.2f}]
                """
            )

    with col_results:
        st.markdown("### 🎯 Prediction & Calibration")

        # Run inference
        try:
            pred_result = inference.predict_digit(net, preprocessed.tensor, active_device)
        except Exception as inf_err:
            st.error(f"Inference error: {inf_err}")
            st.stop()

        # Big Hero Card
        card_class = "pred-card-ambig" if pred_result.is_ambiguous else "pred-card"
        badge_text = "⚠️ Ambiguous" if pred_result.is_ambiguous else f"{pred_result.confidence_level} Confidence"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="stat-label">Predicted Digit</div>
                <div class="pred-digit-large">{pred_result.predicted_digit}</div>
                <div class="pred-meta-badge">{badge_text} ({pred_result.confidence_percentage:.2f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Telemetry metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Model Confidence</div>
                    <div class="stat-value">{pred_result.confidence_percentage:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Inference Latency</div>
                    <div class="stat-value">{pred_result.inference_time_ms:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Compute Device</div>
                    <div class="stat-value">{pred_result.device_used.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Warnings / ambiguity banners
        if pred_result.warnings:
            for w in pred_result.warnings:
                st.warning(f"⚠️ {w}")

        st.markdown("#### 🏆 Top 3 Candidates")
        cand_cols = st.columns(3)
        for i, cand in enumerate(pred_result.top_candidates):
            with cand_cols[i]:
                rank_icon = ["🥇", "🥈", "🥉"][i]
                st.metric(
                    label=f"{rank_icon} Digit {cand.digit}",
                    value=f"{cand.percentage:.2f}%",
                )

        # Full 10-Class Probability Distribution Chart
        st.markdown("#### 📊 Probability Distribution (Digits 0 – 9)")
        prob_df = pd.DataFrame({
            "Digit": [str(d) for d in range(10)],
            "Probability (%)": [round(p * 100, 2) for p in pred_result.all_probabilities],
        }).set_index("Digit")

        st.bar_chart(prob_df, y="Probability (%)", color="#10b981", height=220)


# ------------------------------------------------------------------------------
# Mode B: Multi-Digit Sequence (Experimental OCR)
# ------------------------------------------------------------------------------
else:
    st.markdown("### 🔤 Multi-Digit Sequence Recognition (Experimental)")
    st.info(
        "ℹ️ **Experimental Feature Note**: The MNIST CNN is trained specifically on isolated single digits. "
        "This multi-digit mode segments individual digit regions using contour analysis, sorts them left-to-right, "
        "and classifies each candidate through the CNN. It requires spatially separated digits and may fail on "
        "touching, overlapping, or cursive writing."
    )

    try:
        with st.spinner("Segmenting digit candidates from image..."):
            segmented_digits, annotated_image = preprocessing.segment_multi_digits(
                selected_image,
                invert_mode=invert_mode,
            )
    except preprocessing.PreprocessingError as seg_err:
        st.error(f"⚠️ Multi-Digit Segmentation Error: {seg_err}")
        st.stop()

    # Run inference across all segmented candidates
    seq_result = inference.predict_sequence(net, segmented_digits, active_device)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 🖼️ Detected Regions & Reading Order")
        st.image(
            annotated_image,
            caption=f"Segmented {len(segmented_digits)} digit candidate(s)",
            use_container_width=True,
        )

    with col_right:
        st.markdown("#### 🔢 Recognised Sequence")
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="stat-label">Recognised Number</div>
                <div class="pred-digit-large" style="letter-spacing: 0.1em;">{seq_result.sequence_string}</div>
                <div class="pred-meta-badge">Average Confidence: {seq_result.average_confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sm1, sm2 = st.columns(2)
        with sm1:
            st.metric("Total Latency", f"{seq_result.total_time_ms:.2f} ms")
        with sm2:
            st.metric("Minimum Confidence", f"{seq_result.minimum_confidence:.1f}%")

        if seq_result.warnings:
            for w in seq_result.warnings:
                st.warning(f"⚠️ {w}")

    # Display breakdown of individual digits
    st.markdown("---")
    st.markdown("#### 🔍 Individual Digit Candidates Breakdown")
    digit_cols = st.columns(min(len(segmented_digits), 6))

    for idx, (digit_obj, digit_pred) in enumerate(zip(segmented_digits, seq_result.digit_results)):
        col_idx = idx % len(digit_cols)
        with digit_cols[col_idx]:
            st.markdown(f"**Digit #{idx + 1}**")
            zoomed = digit_obj.display_image.resize((140, 140), Image.Resampling.NEAREST)
            st.image(zoomed, caption=f"28×28 Crop", use_container_width=True)
            st.metric(
                label=f"Predicted: {digit_pred.predicted_digit}",
                value=f"{digit_pred.confidence_percentage:.1f}%",
            )
            # Show top 2 alternatives
            alt_text = ", ".join(f"{c.digit} ({c.percentage:.0f}%)" for c in digit_pred.top_candidates[1:3])
            st.caption(f"Alt: {alt_text}")
