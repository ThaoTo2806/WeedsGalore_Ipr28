"""CPU smoke test using a fake SegFormer API (no pretrained download)."""
import importlib.util,sys,types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

class FakeSegFormer(nn.Module):
    def __init__(self, *, num_classes=3, in_channels=5, **kwargs):
        super().__init__()
        self.enc=nn.Conv2d(in_channels,8,3,padding=1)
        self.classifier=nn.Conv2d(8,num_classes,1)
    def encoder(self,x):return self.enc(x)
    def decode_features(self,x,f):return F.avg_pool2d(f,2)

mod=types.ModuleType('nets');mod.__path__=[]
seg=types.ModuleType('nets.segformer');seg.SegFormer5Band=FakeSegFormer
old_nets=sys.modules.get('nets');old_seg=sys.modules.get('nets.segformer')
sys.modules['nets']=mod;sys.modules['nets.segformer']=seg
try:
    path=Path(__file__).resolve().parents[1]/'src/nets/exp16_point.py'
    spec=importlib.util.spec_from_file_location('_exp16_unit',path)
    new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
finally:
    if old_nets is None:sys.modules.pop('nets',None)
    else:sys.modules['nets']=old_nets
    if old_seg is None:sys.modules.pop('nets.segformer',None)
    else:sys.modules['nets.segformer']=old_seg

def test_train_gradient():
    m=new.SpectralPointRefineSegFormer(num_classes=3,in_channels=5,point_budget=16)
    x=torch.randn(2,5,16,16)
    gt=torch.randint(0,3,(2,16,16))
    out=m(x)
    assert out.shape==(2,3,16,16)
    F.cross_entropy(out,gt).backward()
    assert m.point_mlp[-1].weight.grad is not None
    assert m.point_mlp[-1].weight.grad.abs().sum()>0

def test_eval_stable_and_fullres():
    torch.manual_seed(0)
    m=new.SpectralPointRefineSegFormer(num_classes=3,in_channels=5,point_budget=9).eval()
    x=torch.randn(1,5,11,13)
    with torch.no_grad():
        a=m(x);b=m(x)
    assert a.shape==(1,3,11,13)
    assert torch.equal(a,b)

def test_zero_init_matches_baseline_logits():
    m=new.SpectralPointRefineSegFormer(num_classes=3,in_channels=5,point_budget=8).eval()
    x=torch.randn(1,5,12,14)
    with torch.no_grad():
        expected=F.interpolate(m.classifier(m.decode_features(x,m.encoder(x))),size=x.shape[-2:],mode='bilinear',align_corners=False)
        got=m(x)
    torch.testing.assert_close(expected,got)
