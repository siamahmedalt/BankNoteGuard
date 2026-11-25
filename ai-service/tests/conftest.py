import pytest
import app as ai

@pytest.fixture(autouse=True)
def isolate_ocr(monkeypatch):
    # Pipeline tests exercise model/HTTP behavior without invoking the OCR executable.
    monkeypatch.setattr(ai, 'recognize_denomination', lambda image, full_image=None: None)
