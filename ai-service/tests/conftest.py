import pytest
import app as ai

@pytest.fixture(autouse=True)
def isolate_ocr(monkeypatch):
    # Pipeline tests exercise model/HTTP behavior without invoking the OCR executable.
    monkeypatch.setattr(ai, 'recognize_denomination', lambda image, full_image=None: None)


@pytest.fixture(autouse=True)
def use_legacy_policy_for_regression(monkeypatch):
    monkeypatch.setattr(ai, "INFERENCE_MODE", "consensus")
