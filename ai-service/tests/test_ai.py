"""Small AI service tests; no real model inference is needed."""
import pickle
from PIL import Image
import pytest
import app as ai


def test_real_bundle_metadata_and_threshold():
    data = ai.load_bundle_metadata()
    assert data["class_to_label"] == {"real_notes": 0, "fake_notes": 1}
    assert data["preprocessing"]["image_size"] == [224, 224]
    assert 0 < data["decision_threshold"] < 1


def test_uploaded_single_class_yolo_mode():
    assert ai.detector_mode_from_names({0: "banknote"}) == "single-class"


def test_future_two_class_yolo_mode():
    assert ai.detector_mode_from_names({0: "500_taka", 1: "1000_taka"}) == "denomination"


def test_unknown_yolo_classes_are_rejected():
    with pytest.raises(ValueError, match="Unsupported YOLO classes"):
        ai.detector_mode_from_names({0: "person"})


def test_fail_closed_without_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(ai, "YOLO_PATH", tmp_path / "absent.pt")
    ai.get_detector.cache_clear()
    with pytest.raises(FileNotFoundError, match="YOLO weights missing"):
        ai.get_detector()


def test_rejects_non_note_before_classifier(monkeypatch, tmp_path):
    monkeypatch.setattr(ai, "STORAGE", tmp_path)
    monkeypatch.setattr(ai, "detect_note", lambda image: (None, "not a banknote"))
    monkeypatch.setattr(ai, "classify_note", lambda image: pytest.fail("classifier must not run"))
    result = ai.process_image(Image.new("RGB", (300, 200)), "test-id", "upload")
    assert result["status"] == "REJECTED"
    with (tmp_path / "prediction_results.pkl").open("rb") as stream:
        assert pickle.load(stream)["status"] == "REJECTED"


def test_accepted_single_class_note_has_unknown_denomination(monkeypatch, tmp_path):
    monkeypatch.setattr(ai, "STORAGE", tmp_path)
    monkeypatch.setattr(
        ai, "detect_note",
        lambda image: ({"bbox": (10, 10, 110, 60), "confidence": .9, "denomination": None}, None)
    )

    def classifier(image):
        assert image.size == (100, 50)
        return {"status": "REAL", "confidence": .91, "fakeProbability": .09, "threshold": .8044}

    monkeypatch.setattr(ai, "classify_note", classifier)
    result = ai.process_image(Image.new("RGB", (150, 80)), "test-single", "camera")
    assert result["status"] == "REAL"
    assert result["denomination"] is None
    assert (tmp_path / "cropped" / "test-single.jpg").is_file()
