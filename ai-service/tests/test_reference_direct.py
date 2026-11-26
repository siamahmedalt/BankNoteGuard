import io
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import app as ai
import classifier_worker as worker

@pytest.fixture(autouse=True)
def direct(monkeypatch, use_legacy_policy_for_regression):
    monkeypatch.setattr(ai, 'INFERENCE_MODE', 'direct')


def make_result(values):
    return worker.decode_prediction(values)

@pytest.mark.parametrize('values,status,path', [
    ([.92,.06,.02],'REAL','reference_direct'),
    ([.08,.91,.01],'FAKE','reference_direct'),
    ([.02,.01,.97],'REJECTED','reference_direct'),
    ([.58,.40,.02],'REAL','reference_direct'),
    ([.34,.32,.34],'REAL','reference_direct'),
    ([.39,.40,.21],'FAKE','reference_direct'),
    ([.33,.32,.35],'REJECTED','reference_direct'),
    ([.3333333,.3333334,.3333333],'FAKE','reference_direct'),
    ([.19,.70,.11],'FAKE','reference_direct'),
])
def test_direct_symmetric_policy(monkeypatch, values, status, path):
    calls=[]
    def classify(image):
        calls.append(image.size)
        return make_result(values)
    monkeypatch.setattr(ai, 'classify_note', classify)
    result, actual, views = ai.classify_for_policy(Image.new('RGB',(300,500)), [], 'test')
    assert calls == [(300,500)] and len(views)==1
    assert (result['status'], actual)==(status,path)
    assert list(result['probabilities'].values())==pytest.approx(values)
    assert result['confidence']==max(values)

@pytest.mark.parametrize('fmt,mime', [('JPEG','image/jpeg'),('PNG','image/png'),('WEBP','image/webp')])
def test_api_original_transport(monkeypatch,tmp_path,fmt,mime):
    monkeypatch.setattr(ai,'STORAGE',tmp_path)
    monkeypatch.setattr(ai,'MODEL_STATE',{'ready':True,'message':'ready'})
    monkeypatch.setattr(ai,'_worker_health',lambda: {'ready':True})
    monkeypatch.setattr(ai,'_bootstrap_models',lambda:None)
    monkeypatch.setattr(ai,'detect_candidates',lambda img:[])
    raw=io.BytesIO(); Image.new('RGB',(321,199)).save(raw,fmt)
    calls=[]
    def classify(data,ct):
        calls.append((data,ct)); return make_result([.88,.10,.02])
    monkeypatch.setattr(ai,'classify_original_bytes',classify)
    with TestClient(ai.app) as client:
        resp=client.post('/predict', files={'image':('sample',raw.getvalue(),mime)})
        assert resp.status_code==200, resp.text
        body=resp.json(); assert body['status']=='REAL' and body['realProbability']==.88
        assert body['inferencePath']=='reference_direct'
        assert len(calls)==1
        if fmt=='WEBP':
            assert calls[0][1]=='image/png'
            assert Image.open(io.BytesIO(calls[0][0])).size==(321,199)
        else:
            assert calls[0]==(raw.getvalue(),mime)
        assert client.get('/history-image/'+body['requestId']).status_code==200


def test_direct_nonfinite_rejected(monkeypatch):
    monkeypatch.setattr(ai,'classify_note',lambda img: {'probabilities':{'real_notes':float('nan'),'fake_notes':.5,'not_banknotes':.5}})
    with pytest.raises(ValueError): ai.direct_prediction(Image.new('RGB',(10,10)),'test')


def test_reference_mapping_matches():
    source=Path(__file__).resolve().parents[2]/'reference-app'/'class_mapping_3class.json'
    assert json.loads(source.read_text())==worker.load_bundle()['class_to_label']

@pytest.mark.parametrize('fmt',['JPEG','PNG'])
def test_reference_decode_parity(fmt):
    tf=pytest.importorskip('tensorflow')
    buf=io.BytesIO(); Image.fromarray(np.random.default_rng(5).integers(0,256,(81,130,3),dtype=np.uint8)).save(buf,fmt)
    data=buf.getvalue()
    # Exact reference function operations.
    im=tf.io.decode_image(data,channels=3,expand_animations=False); im.set_shape([None,None,3])
    expected=tf.expand_dims(tf.cast(tf.image.resize(im,[224,224]),tf.float32),0)
    np.testing.assert_array_equal(worker.preprocess_image(data).numpy(),expected.numpy())

@pytest.mark.parametrize('values,expected', [([.41,.39,.20],'REAL'),([.39,.40,.21],'FAKE'),([.31,.33,.36],'REJECTED')])
def test_scan_api_low_scores_are_not_rejected_by_policy(monkeypatch,tmp_path,values,expected):
    monkeypatch.setattr(ai,'STORAGE',tmp_path)
    monkeypatch.setattr(ai,'MODEL_STATE',{'ready':True,'message':'ready'})
    monkeypatch.setattr(ai,'_worker_health',lambda: {'ready':True})
    monkeypatch.setattr(ai,'_bootstrap_models',lambda:None)
    monkeypatch.setattr(ai,'detect_candidates',lambda img:[])
    monkeypatch.setattr(ai,'classify_original_bytes',lambda data,ct: make_result(values))
    data=io.BytesIO(); Image.new('RGB',(64,64)).save(data,'PNG')
    with TestClient(ai.app) as client:
        response=client.post('/predict',files={'image':('note.png',data.getvalue(),'image/png')})
        assert response.status_code==200
        body=response.json()
        assert body['status']==expected and body['confidence']==max(values)
        assert body['threshold'] is None and body['inferencePath']=='reference_direct'
        assert [body[k] for k in ('realProbability','fakeProbability','invalidProbability')]==values
