"""Training-to-serving checks, independent of real/fake model accuracy."""
import hashlib
import io
import json
import shutil
import numpy as np
import pytest
from PIL import Image
import app as ai
import classifier_worker as worker


def test_model_matches_manifest():
    bundle = worker.load_bundle()
    assert bundle['model_sha256'] == hashlib.sha256(ai.MODEL_PATH.read_bytes()).hexdigest()
    assert bundle['class_to_label'] == {'real_notes': 0, 'fake_notes': 1, 'not_banknotes': 2}


def test_wrong_model_checksum_fails_closed(monkeypatch, tmp_path):
    for name in ('class_mapping_3class.json', 'inference_contract.json'):
        shutil.copy(ai.MODEL_PATH.parent / name, tmp_path / name)
    bad = tmp_path / 'other.keras'
    bad.write_bytes(b'wrong checkpoint')
    monkeypatch.setattr(worker, 'MODEL_PATH', bad)
    with pytest.raises(ValueError, match='checksum mismatch'):
        worker.load_bundle()


def test_wrong_resize_contract_fails_closed(monkeypatch, tmp_path):
    shutil.copy(ai.MODEL_PATH.parent / 'class_mapping_3class.json', tmp_path)
    manifest = json.loads((ai.MODEL_PATH.parent / 'inference_contract.json').read_text())
    manifest['resize_method'] = 'NEAREST'
    (tmp_path / 'inference_contract.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(worker, 'MODEL_PATH', tmp_path / 'other.keras')
    with pytest.raises(ValueError, match='Invalid training/inference contract'):
        worker.load_bundle()


def test_native_rgb_transport_no_pre_resize(monkeypatch):
    pixels = np.random.default_rng(42).integers(0, 256, (301, 407, 3), dtype=np.uint8)
    original = Image.fromarray(pixels)
    class Reply:
        def __enter__(self):
            return io.BytesIO(b'{"status":"REAL"}')
        def __exit__(self, *args):
            pass
    def fake_urlopen(request, timeout):
        with Image.open(io.BytesIO(request.data)) as sent:
            assert sent.size == original.size
            assert sent.format == 'PNG'
            np.testing.assert_array_equal(np.asarray(sent), pixels)
        return Reply()
    monkeypatch.setattr(ai, 'urlopen', fake_urlopen)
    assert ai.classify_note(original)['status'] == 'REAL'


def test_tensorflow_resize_matches_training_notebook():
    tf = pytest.importorskip('tensorflow')
    pixels = np.random.default_rng(43).integers(0,256,(307,419,3),dtype=np.uint8)
    buf=io.BytesIO(); Image.fromarray(pixels).save(buf, 'PNG')
    image = tf.io.decode_image(buf.getvalue(), channels=3, expand_animations=False)
    image.set_shape([None,None,3])
    expected = tf.cast(tf.image.resize(image,[224,224]),tf.float32)
    actual = worker.preprocess_png(buf.getvalue())
    assert tuple(actual.shape)==(1,224,224,3)
    assert actual.dtype==tf.float32
    np.testing.assert_array_equal(actual.numpy()[0], expected.numpy())
    assert np.any(actual.numpy() % 1 != 0), 'Should preserve fractional interpolated values'
