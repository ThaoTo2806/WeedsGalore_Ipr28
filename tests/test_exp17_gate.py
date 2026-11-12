"""CPU tests: fake backbone, no network or pretrained weights."""
import importlib.util,sys,types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

class Fake(nn.Module):
    def __init__(self,*,num_classes=3,in_channels=5,**kwargs):
        super().__init__();self.encoder_conv=nn.Conv2d(in_channels,8,3,padding=1);self.classifier=nn.Conv2d(8,num_classes,1)
    def encoder(self,x):return self.encoder_conv(x)
    def decode_features(self,x,f):return F.avg_pool2d(f,2)

root=Path(__file__).resolve().parents[1]/'src/nets'
ns=types.ModuleType('nets');ns.__path__=[]
seg=types.ModuleType('nets.segformer');seg.SegFormer5Band=Fake
saved={k:sys.modules.get(k) for k in ('nets','nets.segformer','nets.exp16_point')}
sys.modules['nets']=ns;sys.modules['nets.segformer']=seg
try:
    sp=importlib.util.spec_from_file_location('nets.exp16_point',root/'exp16_point.py')
    base=importlib.util.module_from_spec(sp);sys.modules['nets.exp16_point']=base;sp.loader.exec_module(base)
    spec=importlib.util.spec_from_file_location('_exp17_test',root/'exp17_gate.py')
    exp17=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp17)
finally:
    for k,v in saved.items():
        if v is None:sys.modules.pop(k,None)
        else:sys.modules[k]=v

def model(**kw):return exp17.SpectralSafetyGateSegFormer(num_classes=3,in_channels=5,point_budget=24,**kw)

def test_forward_shape_and_eval_repeatability():
    m=model().eval();x=torch.randn(1,5,16,20)
    with torch.no_grad():a=m(x);b=m(x)
    assert a.shape==(1,3,16,20)
    torch.testing.assert_close(a,b)

def test_train_gradients_and_safety_loss():
    torch.manual_seed(4);m=model().train();x=torch.randn(2,5,16,20);y=torch.randint(0,3,(2,16,20))
    out=m(x);loss=F.cross_entropy(out,y)+0.1*exp17.exp17_safety_loss(m,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.point_mlp[-1].weight.grad is not None
    assert m.point_mlp[-1].weight.grad.abs().sum()>0
    assert m.safety_gate[-1].bias.grad is not None
    assert torch.isfinite(m.safety_gate[-1].bias.grad).all()

def test_gate_zero_residual_equivalent_to_coarse():
    m=model().eval();x=torch.randn(1,5,12,14)
    with torch.no_grad():
        expected=F.interpolate(m.classifier(m.decode_features(x,m.encoder(x))),size=x.shape[-2:],mode='bilinear',align_corners=False)
        out=m(x)
    torch.testing.assert_close(out,expected.float())

def test_ignored_label_safe():
    m=model().train();x=torch.randn(1,5,12,14);y=torch.full((1,12,14),-1)
    m(x);v=exp17.exp17_safety_loss(m,y)
    assert torch.isfinite(v)

def test_no_gt_in_inference():
    m=model().eval();x=torch.randn(1,5,12,14)
    with torch.no_grad():z=m(x)
    assert m._safety_cache is None
    assert z.shape==(1,3,12,14)
