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
    return {'status':status,'confidence':confidence,'fakeProbability':.02 if status=='REAL' else .96 if status=='FAKE' else .01,
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
    assert r['inferencePath']=='full_image_primary' and c==[(400,200)]
    assert c[0][0]>=380


def test_disjoint_fragments_use_full_image(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((10,20,140,175)),box((230,20,390,180))],[pred('FAKE')])
    assert r['status']=='FAKE' and r['inferencePath']=='multi_view_verified' and len(c)==3


def test_multiple_separate_confirmed(monkeypatch,tmp_path):
    monkeypatch.setattr(ai,'VERIFY_MULTIPLE_NOTES',True)
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred(),pred('FAKE')])
    assert r['status']=='REJECTED' and r['inferencePath']=='multiple_confirmed' and len(c)==2


def test_second_box_is_false_positive(monkeypatch,tmp_path):
    monkeypatch.setattr(ai,'VERIFY_MULTIPLE_NOTES',True)
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred(),pred('REJECTED'),pred()])
    assert r['status']=='REAL' and r['inferencePath']=='full_image_primary' and c[-1]==(400,200)


def test_no_box_non_note(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REJECTED',.99)])
    assert r['status']=='REJECTED' and r['inferencePath']=='non_banknote_confirmed' and len(c)==3


def test_no_box_high_confidence_note(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred()])
    assert r['status']=='REAL' and r['denomination'] is None and c==[(400,200)]


def test_no_box_low_confidence_note_rejected(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REAL',.60)])
    assert r['status']=='REJECTED' and r['threshold']==.70 and r['confidence'] is None


def test_full_note_accepted_despite_fragment_box(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100),denom=500)],[pred('REAL')])
    assert r['status']=='REAL' and r['inferencePath']=='full_image_primary' and r['denomination']==500
    assert c[-1]==(400,200)


def test_full_non_note_rejected_despite_detection(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100),denom=500)],[pred('REJECTED')])
    assert r['status']=='REJECTED' and r['denomination'] is None and len(c)==3


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
    monkeypatch.setattr(ai,'VERIFY_MULTIPLE_NOTES',False)
    r,c=run(monkeypatch,tmp_path,[box((10,10,180,90)),box((210,110,380,190))],[pred()])
    assert r['status']=='REAL' and r['inferencePath']=='full_image_primary' and c==[(400,200)]


@pytest.mark.parametrize('boxes', [[], [box((10,20,140,175)),box((230,20,390,180))]])
@pytest.mark.parametrize('confidence', [.70,.71,.8443974])
def test_full_fake_note_not_rejected_by_old_90_percent_gate(monkeypatch,tmp_path,boxes,confidence):
    r,c=run(monkeypatch,tmp_path,boxes,[pred('FAKE',confidence)])
    assert r['status']=='FAKE' and r['confidence']==pytest.approx(confidence,abs=1e-6) and r['threshold']==.70

def test_full_fake_84_percent_accepted_despite_fragment_box(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred('FAKE',.8443974)])
    assert r['status']=='FAKE' and r['inferencePath']=='multi_view_verified'

def test_low_confidence_fake_still_rejected(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('FAKE',.699)])
    assert r['status']=='REJECTED'

def test_non_note_still_rejected(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REJECTED',.99)])
    assert r['status']=='REJECTED'


@pytest.mark.parametrize('status', ['REAL', 'FAKE'])
def test_low_confidence_with_yolo_box_still_rejected(monkeypatch,tmp_path,status):
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred(status,.69)])
    assert r['status']=='REJECTED' and r['confidence'] is None and len(c)==3 and r['inferencePath']=='review_uncertain'

def test_ocr_failure_preserves_authenticity(monkeypatch,tmp_path):
    def broken(*args): raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(ai,'recognize_denomination',broken)
    r,c=run(monkeypatch,tmp_path,[box((20,20,150,100))],[pred('REAL',.97)])
    assert r['status']=='REAL' and r['denomination'] is None


# Explicit V8 regression scenarios; outputs are simulated, not actual model accuracy.
def test_weak_original_real_not_overridden_by_alternate_views(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REAL',.62),pred('REAL',.86),pred('REAL',.9)])
    assert r['status']=='REJECTED' and r['inferencePath']=='review_uncertain'

def test_strong_fake_conflicting_real_not_called_fake(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('FAKE',.97),pred('REAL',.93),pred('FAKE',.89)])
    assert r['status']=='REJECTED' and r['inferencePath']=='review_uncertain'

def test_single_strong_alternate_is_not_enough(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REAL',.51),pred('REAL',.94),pred('FAKE',.62)])
    assert r['status']=='REJECTED' and r['inferencePath']=='review_uncertain'

def test_confirmed_fake_keeps_fake(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('FAKE',.81),pred('FAKE',.85),pred('FAKE',.8)])
    assert r['status']=='FAKE' and r['confidence']==.82

def test_non_note_cannot_be_rescued_by_one_fake_vote(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REJECTED',.98),pred('REJECTED',.96),pred('FAKE',.99)])
    assert r['status']=='REJECTED'

def test_portrait_conflicting_rotations_trigger_review(monkeypatch):
    outputs=iter([pred('FAKE',.94),pred('FAKE',.95),pred('FAKE',.89),pred('REAL',.92),pred('REAL',.91)])
    monkeypatch.setattr(ai,'classify_note',lambda img:next(outputs))
    result,path,views=ai.verify_prediction(Image.new('RGB',(100,220)),[], 'portrait-test')
    assert result['status']=='REJECTED' and path=='review_uncertain' and len(views)==5

def test_square_crop_not_rotated(monkeypatch):
    outputs=iter([pred('FAKE',.94),pred('FAKE',.95),pred('FAKE',.89)])
    monkeypatch.setattr(ai,'classify_note',lambda img:next(outputs))
    result,path,views=ai.verify_prediction(Image.new('RGB',(200,220)),[], 'crop-test')
    assert result['status']=='FAKE' and len(views)==3


def test_weak_real_cannot_be_overridden_by_smoothed_fake_views(monkeypatch,tmp_path):
    r,c=run(monkeypatch,tmp_path,[],[pred('REAL',.60),pred('FAKE',.94),pred('FAKE',.95)])
    assert r['status']=='REJECTED' and r['inferencePath']=='review_uncertain'

def test_resize_default_matches_worker():
    import classifier_worker as worker
    assert ai.RESIZE_METHOD == worker.RESIZE_METHOD == 'NEAREST'
