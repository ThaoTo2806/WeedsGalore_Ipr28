import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nets.exp20_icr import ICRSegFormer
from nets.segformer import SegFormer5Band

class FakeEncoder(nn.Module):
    dims=(32,64,160,256)
    def forward(self,x):
        b,_,h,w=x.shape
        # lightweight differentiable fake multi-scale features
        v=x.mean(1,keepdim=True)
        return tuple(F.interpolate(v,size=(max(1,h//s),max(1,w//s)),mode='area').repeat(1,d,1,1)
            for d,s in zip(self.dims,(4,8,16,32)))

def make():
    return ICRSegFormer(encoder=FakeEncoder(),pretrained_backbone=False,backbone='b0',in_channels=5,
                        num_classes=3,use_low_level_refine=True,use_multiscale_fusion=True,llr_source='raw')

def test_shape_and_finite():
    m=make().eval();x=torch.rand(1,5,64,64)
    with torch.no_grad(): y=m(x)
    assert y.shape==(1,3,64,64) and torch.isfinite(y).all()

def test_zero_init_exactly_matches_base():
    m=make().eval();x=torch.rand(1,5,64,64)
    with torch.no_grad():
        base=SegFormer5Band.forward(m,x)
        y=m(x)
    assert torch.allclose(base,y,atol=1e-5)

def test_backward():
    m=make().train();x=torch.rand(1,5,64,64)
    y=m(x);loss=y.square().mean();loss.backward()
    assert m.classifier.weight.grad is not None

def test_no_gt_argument():
    m=make(); assert 'labels' not in m.forward.__code__.co_varnames
