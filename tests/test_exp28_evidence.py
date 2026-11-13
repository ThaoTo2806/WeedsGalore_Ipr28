from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'scripts'))
import exp28_paper_report as r

def test_iou_known():
    cm=np.diag([8,6,4])
    assert np.allclose(r.iou(cm),100)

def test_paired_identical():
    a=np.zeros((26,3,3),dtype=np.int64)
    a[:,0,0]=100;a[:,1,1]=30;a[:,2,2]=12
    names=[f'im{i}' for i in range(26)]
    rep=r.paired_bootstrap((names,a),(names,a),draws=200)
    assert rep['delta']==0 and rep['ci95']==[0,0]

def test_unaligned_rejected():
    a=np.ones((26,3,3),dtype=np.int64)
    try:r.paired_bootstrap((['a']*26,a),(['b']*26,a),draws=5)
    except ValueError:return
    assert False


def test_weedmap_frame_parser():
    # torch is provided in Kaggle; optional local test
    try:import exp28_weedmap as w
    except ImportError:return
    assert w.frame_key('000_frame0234_GroundTruth_iMap.png')=='0234'
    assert w.frame_key('frame0234.png')=='0234'
    assert w.frame_key('R.png') is None
