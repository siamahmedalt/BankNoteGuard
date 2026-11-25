import pytest
from denomination import parse_evidence
import app as ai
from test_ai import run, box, pred

HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
def tsv(tokens):
    return HEADER + ''.join(f'5\t1\t1\t1\t1\t{i}\t1\t1\t30\t30\t{c}\t{t}\n' for i,(t,c) in enumerate(tokens,1))

@pytest.mark.parametrize('tokens,expected', [
    ([('500',95)],500), ([('৳১০০০',94)],1000), ([('৫০০',89)],500),
    ([('1,000',90)],1000), ([('এক',90),('হাজার',92),('টাকা',89)],1000),
    ([('পাঁচশত',90),('টাকা',95)],500), ([('AB500123',99)],None),
    ([('500',48)],None), ([('500',95),('1000',95)],None), ([('200',99)],None),
])
def test_evidence(tokens,expected):
    assert parse_evidence(tsv(tokens)) == expected

@pytest.mark.parametrize('boxes,status,amount', [([], 'FAKE', 500), ([box((10,20,390,180))], 'REAL',1000)])
def test_ocr_amount_in_scan(monkeypatch,tmp_path,boxes,status,amount):
    monkeypatch.setattr(ai,'recognize_denomination',lambda *args: amount)
    r,_=run(monkeypatch,tmp_path,boxes,[pred(status)])
    assert r['denomination']==amount and r['status']==status


def test_rejected_image_never_ocr(monkeypatch,tmp_path):
    def fail(*args): raise AssertionError('Rejected image must not reach OCR')
    monkeypatch.setattr(ai,'recognize_denomination',fail)
    r,_=run(monkeypatch,tmp_path,[],[pred('REJECTED')])
    assert r['denomination'] is None


def test_known_detector_denomination_has_priority(monkeypatch,tmp_path):
    def fail(*args): raise AssertionError('Known detector amount must not need OCR')
    monkeypatch.setattr(ai,'recognize_denomination',fail)
    r,_=run(monkeypatch,tmp_path,[box((10,20,390,180),denom=500)],[pred()])
    assert r['denomination']==500
