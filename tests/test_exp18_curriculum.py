import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from exp18_curriculum import stage_for_epoch, crop_bounds, sample_center

def test_stages():
    assert stage_for_epoch(0)=='object'
    assert stage_for_epoch(34)=='object'
    assert stage_for_epoch(35)=='hard_negative'
    assert stage_for_epoch(69)=='hard_negative'
    assert stage_for_epoch(70)=='full_image'
    assert stage_for_epoch(99)=='full_image'

def test_bounds():
    assert crop_bounds(600,600,0,0)==(0,0)
    assert crop_bounds(600,600,599,599)==(216,216)
    assert crop_bounds(600,600,300,300)==(108,108)

def test_object_center():
    mask=np.zeros((600,600),dtype=np.int64)
    mask[200:220,300:320]=2
    mask[350:370,100:120]=1
    rng=np.random.default_rng(123)
    for _ in range(100):
        (y,x),kind=sample_center(mask,[], 'object',rng)
        assert 0<=y<600 and 0<=x<600
        if kind=='weed':assert mask[y,x]==2
        if kind=='crop':assert mask[y,x]==1

def test_hard_negatives_are_selected_only_in_hard_stage():
    mask=np.zeros((600,600),dtype=np.int64)
    cache=np.array([[450,460]],dtype=np.int32)
    rng=np.random.default_rng(42)
    assert any(sample_center(mask,cache,'hard_negative',rng)[1]=='hard_bg_to_weed' for _ in range(100))
    rng=np.random.default_rng(42)
    assert all(sample_center(mask,cache,'object',rng)[1]!='hard_bg_to_weed' for _ in range(100))
