# MNIST Digit Recognition Web Application

A production-quality handwritten digit recognition application built with **PyTorch**, **CUDA**, and **Streamlit**. Powered by a convolutional neural network (CNN) trained to **99.42% test accuracy** on the MNIST benchmark.

Designed for robust recognition of both standard isolated digits and real-world phone camera photographs of digits written on paper.

---

## 🌟 Key Features

1. **Exact Trained Model Reuse**: Loads and serves the existing `mnist_digit_cnn.pth` checkpoint without retraining or modifying weights.
2. **Hardware Acceleration with Safe Fallback**:
   - Executes inference natively on NVIDIA RTX GPUs via PyTorch CUDA.
   - Verified on **NVIDIA GeForce RTX 5060 Laptop GPU** (sub-2ms inference latency).
   - Seamlessly falls back to CPU if CUDA is unavailable.
3. **Robust Real-World Preprocessing**:
   - **EXIF Orientation Correction**: Automatically normalizes phone camera orientation metadata.
   - **Alpha Transparency Compositing**: Handles transparent PNGs without black-box artifacts.
   - **Automatic Polarity Inversion**: Detects whether handwriting is dark-on-light (black/blue ink on paper) or light-on-dark (native MNIST / chalkboard) using border pixel sampling.
   - **Background Illumination Flattening**: Morphological filtering removes shadows, yellow paper tint, and gradient illumination.
   - **Center of Mass (CoM) Centering**: Re-aligns digit mass to `(14.0, 14.0)` on a 28×28 canvas following Yann LeCun's canonical MNIST specification.
   - **Exact Normalization**: Scales inputs by training parameters: $\text{mean} = 0.1307$, $\text{std} = 0.3081$.
4. **Transparent Prediction Telemetry**:
   - High-contrast visual display of predicted digit, percentage confidence, and latency.
   - Side-by-side inspection of the original upload and the actual 28×28 pixel canvas passed to the CNN.
   - Ranked top-3 candidate digits and full 10-class probability bar chart (digits 0–9).
   - **Ambiguity & OOD Warnings**: Warns users when predictions are uncertain, margins between top classes are narrow, or images fall outside the training distribution.
5. **Experimental Multi-Digit Sequence Mode**:
   - Segments multiple separated handwritten digits (e.g., `"42"`, `"789"`).
   - Sorts candidates left-to-right, classifies each, and outputs the complete number string.

---

## 🧠 Model Architecture & Checkpoint

The model is reconstructed from the original project definition (`DigitCNN`):

```text
Input (1, 28, 28)
  │
  ├── Conv2d(1 -> 32, kernel_size=3, padding=1) + ReLU
  ├── MaxPool2d(2, 2)  --> (32, 14, 14)
  │
  ├── Conv2d(32 -> 64, kernel_size=3, padding=1) + ReLU
  ├── MaxPool2d(2, 2)  --> (64, 7, 7)
  │
  ├── Flatten()        --> 3,136 features
  ├── Linear(3136 -> 128) + ReLU
  ├── Dropout(0.25)
  └── Linear(128 -> 10) [Raw Logits]
```

- **Total Trainable Parameters**: 421,642
- **Training Epochs**: 15 epochs (Adam optimizer, lr=0.001)
- **Evaluated Test Accuracy**: **99.42%** on 10,000 standard MNIST test samples
- **Average Test Loss**: 0.0232
- **Checkpoint File**: `models/mnist_digit_cnn.pth` (1,690,365 bytes, SHA256: `79E2DB7B4DDF...`)

---

## 📁 Project Structure

```text
mnist-digit-recognition/
├── app.py                      # Streamlit interactive web application
├── model.py                    # DigitCNN architecture and strict checkpoint loader
├── preprocessing.py            # Image transformation, polarity, and CoM centering
├── inference.py                # Prediction, Softmax, timing, and ambiguity analysis
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation and setup guide
├── .gitignore                  # Git ignore rules for Python/Streamlit
├── models/
│   └── mnist_digit_cnn.pth     # Preserved trained PyTorch checkpoint
├── assets/
│   └── examples/               # 25 verified test samples (clean MNIST, paper photos, multi-digit)
└── tests/
    ├── test_app_startup.py     # Smoke test ensuring app imports and compiles
    ├── test_model_loading.py   # Tests for architecture, weights, CPU/CUDA loading
    ├── test_preprocessing.py   # Tests for EXIF, polarity, CoM, and error handling
    └── test_inference.py       # Tests for probability calibration, timing, and accuracy
```

---

## 💻 Environment & Prerequisites

- **Operating System**: Windows 11 Pro 64-bit
- **Hardware Tested**: HP OMEN 16 Slim (Intel Core Ultra 7 255H, NVIDIA GeForce RTX 5060 Laptop GPU 8GB VRAM)
- **Python**: 3.10 to 3.12 64-bit
- **PyTorch**: 2.11+ with CUDA 12.8 wheel runtime

---

## 🚀 Quickstart & Execution (Windows 11 PowerShell)

### Step 1: Open PowerShell and Navigate to Project

```powershell
cd D:\projects\AI-ML\mnist-digit-recognition
```

### Step 2: Use the Configured Virtual Environment

If using your existing virtual environment in `C:\Users\jemmz\deep-learning-lab\.venv`:

```powershell
# Activate the existing environment
C:\Users\jemmz\deep-learning-lab\.venv\Scripts\Activate.ps1
```

Or run directly using that Python executable:

```powershell
$PY = "C:\Users\jemmz\deep-learning-lab\.venv\Scripts\python.exe"
```

### Step 3: Verify GPU and CUDA Support

```powershell
& $PY -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

*Expected output:*
```text
PyTorch: 2.11.0+cu128
CUDA available: True
Device: NVIDIA GeForce RTX 5060 Laptop GPU
```

### Step 4: Run Automated Tests

Execute all 23 unit and integration tests with pytest:

```powershell
& $PY -m pytest tests -v
```

### Step 5: Launch the Streamlit Web Application

```powershell
& $PY -m streamlit run app.py
```

The application will start and display your local URL:
```text
Local URL: http://localhost:8501
Network URL: http://...
```

Open `http://localhost:8501` in your web browser.

---

## 📖 How Preprocessing Works

```mermaid
flowchart TD
    A[Uploaded Image / Photo] --> B[load_input_image: EXIF transpose & RGBA composite]
    B --> C[detect_polarity_and_invert: Border pixel sampling]
    C --> D[extract_digit_bounding_box: Top-hat isolation & ink salience]
    D --> E[Local Contrast Enhancement: Percentile stretch on ROI]
    E --> F[Adaptive Stroke Dilation: Restores pen line thickness for downscaled crops]
    F --> G[center_digit_on_canvas: Fit in 20x20 box preserving aspect ratio]
    G --> H[Center of Mass Shift: Align CoM to 14.0, 14.0]
    H --> I[Exact Normalisation: (x - 0.1307) / 0.3081]
    I --> J[PyTorch Tensor 1, 1, 28, 28]
```

### Tips for Best Recognition Results

1. **Isolated Digits**:
   - Ensure the digit is fully visible in the frame.
   - Good lighting with minimal harsh shadows yields higher confidence.
   - Dark pen (black, blue) or pencil on white or ruled paper works best.
2. **Preset Selection**:
   - **Auto Adaptive**: Recommended default for most photos and drawings.
   - **Paper Photograph**: Recommended for photos taken under uneven lighting.
   - **Clean Scan**: Recommended for digital tablet drawings or native MNIST exports.
   - **Faint Pencil**: Recommended for light strokes or low-contrast handwriting.

---

## ⚠️ Known Limitations & Distinctions

- **Single-Digit CNN vs. Full OCR**: The core CNN was trained specifically on isolated single characters ($28 \times 28$). It is not a sequence-to-sequence model or Transformer OCR.
- **Touching Digits in Multi-Digit Mode**: Connected component segmentation requires whitespace between characters. Digits that physically touch or cursive writing may be grouped into a single bounding box and produce inaccurate predictions.
- **Orientation**: Digits must be oriented reasonably upright. Extreme rotations ($> 45^\circ$) will alter feature orientation relative to training data.

---

## 🛠️ Troubleshooting

- **Checkpoint Not Found**:
  - The model loader checks `models/mnist_digit_cnn.pth`, `C:\Users\jemmz\deep-learning-lab\mnist_digit_cnn.pth`, and current working directory automatically.
  - You can also set the path explicitly via environment variable:
    ```powershell
    $env:MNIST_MODEL_PATH = "D:\projects\AI-ML\mnist-digit-recognition\models\mnist_digit_cnn.pth"
    ```
- **CUDA Falling Back to CPU**:
  - Ensure the virtual environment with PyTorch CUDA (`2.11.0+cu128`) is active.
  - You can select "Force CPU" or "Force GPU" in the sidebar at any time.
