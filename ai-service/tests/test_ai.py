"""Pipeline regression tests with mocked model outputs (not accuracy tests)."""
import pickle
from PIL import Image
import pytest
import app as ai


def test_three_class_model_metadata():
    import json, zipfile
    import classifier_worker as worker
    mapping = json.loads((ai.MODEL_PATH.parent / 'class_mapping_3class.json').read_text())
    assert mapping == {'real_notes': 0, 'fake_notes': 1, 'not_banknotes': 2}
    with zipfile.ZipFile(ai.MODEL_PATH) as model:
        config = json.loads(model.read('config.json'))
        assert config['config']['layers'][-1]['config']['units'] == 3
    for values, expected in [([.9,.05,.05], 'REAL'), ([.05,.9,.05], 'FAKE'), ([.05,.05,.9], 'REJECTED')]:
        assert worker.decode_prediction(values)['status'] == expected


@pytest.mark.parametrize('names,mode', [({0:'banknote'},'single-class'),({0:'500_taka',1:'1000_taka'},'denomination')])
def test_detector_modes(names,mode):
    assert ai.detector_mode_from_names(names) == mode


def test_unknown_yolo_classes_are_rejected():
    with pytest.raises(ValueError, match='Unsupported YOLO classes'):
        ai.detector_mode_from_names({0:'person'})


def test_fail_closed_without_detector(monkeypatch,tmp_path):
    monkeypatch.setattr(ai,'YOLO_PATH',tmp_path/'absent.pt')
    ai.get_detector.cache_clear()
    with pytest.raises(FileNotFoundError,match='YOLO weights missing'):
        ai.get_detector()


def box(b,conf=.9,denom=None):
    return {'bbox':b,'confidence':conf,'denomination':denom,'className':'banknote'}


def pred(status='REAL',confidence=.96):
    names = ('real_notes', 'fake_notes', 'not_banknotes')
    label = ('REAL','FAKE','REJECTED').index(status)
    probs = [(1-confidence)/2]*3
    probs[label] = confidence
    return {'status':status,'confidence':confidence,'fakeProbability':probs[1],
            'probabilities':dict(zip(names,probs)),
            'threshold':None,'reason':'Image is not a supported banknote.' if status=='REJECTED' else None}


def run(monkeypatch,tmp_path,boxes,outputs):
    monkeypatch.setattr(ai,'STORAGE',tmp_path)
    monkeypatch.setattr(ai,'detect_candidates',lambda image:boxes)
    calls=[]
    outputs=iter(outputs)
    last=None
    def classify(image):
        nonlocal last
        calls.append(image.size)
        last=next(outputs, last)
        assert last is not None
        return last.copy()
    monkeypatch.setattr(ai,'classify_note',classify)
    result=ai.process_image(Image.new('RGB',(400,200)), 'test-id','upload')
    with (tmp_path/'prediction_results.pkl').open('rb') as stream:
        assert pickle.load(stream)==result
    return result,calls


def test_duplicate_nested_boxes_one_note(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,20,390,180),.72),box((40,40,170,100),.99),box((12,22,385,178),.9)],[pred()])
    assert r['status']=='REAL' and r['detectorBoxCount']==3 and r['candidateCount']==1
    assert r['inferencePath']=='rotation_average_argmax' and len(c)==8
    assert c[0][0]>=380


def test_disjoint_fragments_use_full_image(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,20,140,175)),box((230,20,390,180))],[pred('FAKE')])
    assert r['status']=='FAKE' and r['inferencePath']=='rotation_average_argmax' and len(c)==8


def test_multiple_boxes_do_not_override_argmax(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred('FAKE')])
    assert r['status']=='FAKE' and r['inferencePath']=='rotation_average_argmax' and len(c)==8


def test_second_box_is_false_positive(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred()])
    assert r['status']=='REAL' and r['inferencePath']=='rotation_average_argmax' and (400,200) in c


def test_no_box_non_note(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REJECTED',.99)])
    assert r['status']=='REJECTED' and r['inferencePath']=='rotation_average_argmax' and len(c)==8


def test_no_box_high_confidence_note(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred()])
    assert r['status']=='REAL' and r['denomination'] is None and len(c)==8


def test_no_box_low_confidence_note_argmax(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REAL',.60)])
    assert r['status']=='REAL' and r['threshold'] is None and r['confidence']==pytest.approx(.60)


def test_full_note_accepted_despite_fragment_box(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100),denom=500)],[pred('REAL')])
    assert r['status']=='REAL' and r['inferencePath']=='rotation_average_argmax' and r['denomination']==500
    assert c[0]==(400,200) or (400,200) in c


def test_full_non_note_rejected_despite_detection(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100),denom=500)],[pred('REJECTED')])
    assert r['status']=='REJECTED' and r['denomination'] is None and len(c)==8


def test_denomination_only_when_detector_supplies_it(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,20,390,180),denom=1000)],[pred('FAKE')])
    assert r['status']=='FAKE' and r['denomination']==1000
    assert (tmp_path/'cropped'/'test-id.jpg').is_file()


def test_detector_receives_bgr(monkeypatch):
    from types import SimpleNamespace
    class Detector:
        names={0:'banknote'}
        def predict(self,**kwargs):
            assert tuple(kwargs['source'][0,0])==(3,2,1)
            assert kwargs['iou']==.45
            return [SimpleNamespace(boxes=[])]
    monkeypatch.setattr(ai,'get_detector',lambda:Detector())
    assert ai.detect_candidates(Image.new('RGB',(10,10),(1,2,3)))==[]


def test_legacy_detector_strong_fragments_never_count_as_multiple(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred()])
    assert r['status']=='REAL' and r['inferencePath']=='rotation_average_argmax' and len(c)==8


@pytest.mark.parametrize('boxes', [[], [box((10,20,140,175)),box((230,20,390,180))]])
@pytest.mark.parametrize('confidence', [.70,.71,.8443974])
def test_full_fake_note_not_rejected_by_old_90_percent_gate(monkeypatch,tmp_path,boxes,confidence):
    r,c=run(monkeypatch,tmp_path,boxes,[pred('FAKE',confidence)])
    assert r['status']=='FAKE' and r['confidence']==pytest.approx(confidence,abs=1e-6) and r['threshold'] is None

def test_full_fake_84_percent_accepted_despite_fragment_box(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred('FAKE',.8443974)])
    assert r['status']=='FAKE' and r['inferencePath']=='rotation_average_argmax'

def test_low_confidence_fake_argmax(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('FAKE',.699)])
    assert r['status']=='FAKE' and r['confidence']==pytest.approx(.699)

def test_non_note_still_rejected(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REJECTED',.99)])
    assert r['status']=='REJECTED'


@pytest.mark.parametrize('status', ['REAL', 'FAKE'])
def test_low_confidence_with_yolo_box_argmax(monkeypatch,tmp_path,status):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred(status,.69)])
    assert r['status']==status and r['confidence']==pytest.approx(.69) and len(c)==8 and r['inferencePath']=='rotation_average_argmax'

def test_ocr_failure_preserves_authenticity(monkeypatch,tmp_path):
    def broken(*args): raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(ai,'recognize_denomination',broken)
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred('REAL',.97)])
    assert r['status']=='REAL' and r['denomination'] is None


# V10 consensus regression cases. Model outputs are mocked, not accuracy claims.
@pytest.mark.parametrize('label', ['REAL','FAKE','REJECTED'])
def test_all_eight_views_agree(label):
    r,path=ai.aggregate_stability([pred(label,.9)]*8)
    assert r['status']==label and r['confidence']==.9

@pytest.mark.parametrize('label,opposite', [('REAL','FAKE'),('FAKE','REAL')])
def test_conflicting_families_tie_class_order(label,opposite):
    r,path=ai.aggregate_stability([pred(label,.99)]*4+[pred(opposite,.99)]*4)
    assert r['status']=='REAL' and path=='rotation_average_argmax' and r['confidence']==pytest.approx(.4975)

@pytest.mark.parametrize('label', ['REAL','FAKE'])
def test_low_confidence_average_argmax(label):
    r,path=ai.aggregate_stability([pred(label,.65)]*8)
    assert path=='rotation_average_argmax' and r['status']==label and r['confidence']==pytest.approx(.65)

@pytest.mark.parametrize('turn', [Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270])
@pytest.mark.parametrize('size',[(401,203),(203,401),(201,203)])
def test_quarter_turn_view_sets_identical(turn,size):
    import hashlib
    import numpy as np
    img=Image.fromarray(np.random.default_rng(1).integers(0,256,(size[1],size[0],3),dtype=np.uint8))
    def keys(im):
        return sorted((name.split('_')[0],image.size,hashlib.sha256(image.tobytes()).hexdigest()) for name,image in ai.stability_views(im))
    assert keys(img)==keys(img.transpose(turn))

@pytest.mark.parametrize('label', ['REAL','FAKE'])
def test_vote_order_invariant(label):
    views=[pred(label,.91),pred(label,.89),pred(label,.86),pred('REJECTED',.52),pred(label,.9),pred(label,.95),pred(label,.92),pred('REJECTED',.51)]
    first=ai.aggregate_stability(views)
    for k in range(4):
        rotated=views[k:4]+views[:k]+views[4+k:]+views[4:4+k]
        assert ai.aggregate_stability(rotated)==first

def test_always_eight_views_even_if_first_real(monkeypatch):
    monkeypatch.setattr(ai,'classify_note',lambda img: pred('REAL',.99))
    result,path,views=ai.verify_prediction(Image.new('RGB',(100,220)),[], 'test')
    assert len(views)==8 and result['status']=='REAL'

def test_resize_default_matches_worker():
    import classifier_worker as worker
    assert ai.RESIZE_METHOD == worker.RESIZE_METHOD == 'BILINEAR'

@pytest.mark.parametrize('label,score,path', [
    ('REAL', .92, 'rotation_average_argmax'),
    ('FAKE', .87, 'rotation_average_argmax'),
    ('REJECTED', .98, 'rotation_average_argmax'),
    ('REAL', .55, 'rotation_average_argmax'),
])
def test_v14_scan_exposes_three_scores(monkeypatch,tmp_path,label,score,path):
    expected = pred(label,score)['probabilities']
    r,_ = run(monkeypatch,tmp_path,[],[pred(label,score)])
    assert r['inferencePath'] == path
    for field, name in [('realProbability','real_notes'),('fakeProbability','fake_notes'),('invalidProbability','not_banknotes')]:
        assert r[field] == pytest.approx(expected[name])
    assert sum(r[f] for f in ('realProbability','fakeProbability','invalidProbability')) == pytest.approx(1)
    assert r['confidence'] == pytest.approx(score)


def test_all_scores_for_multiple_boxes(monkeypatch,tmp_path):
    r,_=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred('FAKE',.49)])
    assert r['inferencePath']=='rotation_average_argmax'
    assert r['status']=='FAKE' and r['confidence']==pytest.approx(.49)
    assert sum(r[key] for key in ('realProbability','fakeProbability','invalidProbability')) == pytest.approx(1)
