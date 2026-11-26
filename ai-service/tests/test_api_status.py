"""FastAPI readiness contracts without executing model loading."""
from unittest.mock import patch
from fastapi.testclient import TestClient
import app as ai


def test_fast_health_and_liveness_while_initializing():
    with patch.object(ai, '_bootstrap_models', lambda: None):
        with TestClient(ai.app) as client:
            response = client.get('/health')
            assert response.status_code == 200
            assert response.json()['ready'] is False
            assert response.json()['classifierBundlePresent'] is True
            assert response.json()['yoloWeightsPresent'] is True
            assert client.get('/health/live').json() == {'service': 'online', 'live': True}


def test_predict_gives_readiness_error_instead_of_hanging():
    with patch.object(ai, '_bootstrap_models', lambda: None):
        with TestClient(ai.app) as client:
            response = client.post('/predict', files={
                'image': ('note.png', b'placeholder', 'image/png')
            })
            assert response.status_code == 503
            assert 'detail' in response.json()


def test_full_note_upload_contract(monkeypatch, tmp_path):
    import io
    from PIL import Image
    monkeypatch.setattr(ai, 'STORAGE', tmp_path)
    monkeypatch.setattr(ai, 'MODEL_STATE', {'ready': True, 'message': 'ready'})
    monkeypatch.setattr(ai, '_worker_health', lambda: {'ready': True})
    monkeypatch.setattr(ai, 'detect_candidates', lambda img: [
        {'bbox': (10,10,150,100), 'confidence': .95, 'denomination': None},
        {'bbox': (200,10,390,100), 'confidence': .94, 'denomination': None},
    ])
    monkeypatch.setattr(ai, 'classify_note', lambda img: {
        'status': 'REAL', 'confidence': .97, 'fakeProbability': .02,
        'threshold': None, 'reason': None, 'probabilities': {'real_notes':.97,'fake_notes':.02,'not_banknotes':.01},
    })
    raw = io.BytesIO()
    Image.new('RGB', (400,200)).save(raw, 'PNG')
    scan_id = 'e719c8d0-e8c7-4c17-a7e3-093407d5fb20'
    with patch.object(ai, '_bootstrap_models', lambda: None):
        with TestClient(ai.app) as client:
            response = client.post('/predict', files={'image': ('note.png',raw.getvalue(),'image/png')},
                                   data={'source':'upload','request_id':scan_id})
            assert response.status_code == 200
            body = response.json()
            assert body['status'] == 'REAL' and body['requestId'] == scan_id
            assert (body['realProbability'],body['fakeProbability'],body['invalidProbability']) == (.97,.02,.01)
            assert body['inferencePath'] == 'rotation_average_argmax' and body['detectorBoxCount'] == 2
            assert client.get('/history-image/'+scan_id).status_code == 200
