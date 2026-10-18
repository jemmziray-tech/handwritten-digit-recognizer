"""
test_app_startup.py - Smoke test ensuring app.py imports and dependencies resolve cleanly.
"""

import importlib
import py_compile
from pathlib import Path


def test_app_syntax_and_compilation():
    """Verify that app.py compiles without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.is_file()
    py_compile.compile(str(app_path), doraise=True)


def test_app_imports_and_components():
    """Verify that all core modules imported by app.py can be resolved and initialized."""
    import model
    import preprocessing
    import inference

    assert hasattr(model, "DigitCNN")
    assert hasattr(model, "load_model")
    assert hasattr(preprocessing, "preprocess_image")
    assert hasattr(inference, "predict_digit")
