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
