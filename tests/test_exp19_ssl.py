import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from exp19_ssl import make_masks,ssl_loss,MaskedSpectralLearner

class FakeEncoder(nn.Module):
    dims=(16,32,64,128)
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(5,16,7,stride=4,padding=3)
    def forward(self,x):
        return (self.proj(x),)

def test_mask_shapes():
    x=torch.rand(2,5,64,64)
    corrupted,mask=make_masks(x,patch=16)
    assert mask.dtype==torch.bool and mask.shape==x.shape
    assert torch.equal(corrupted[mask],torch.zeros_like(corrupted[mask]))
    assert mask.any() and (~mask).any()

def test_masked_loss_gradient():
    x=torch.rand(2,5,64,64)
    corrupted,mask=make_masks(x,patch=16)
    net=MaskedSpectralLearner(FakeEncoder())
    out=net(corrupted)
    target=torch.nn.functional.interpolate(x,size=out.shape[-2:],mode='area')
    m=torch.nn.functional.adaptive_max_pool2d(mask.float(),out.shape[-2:]).bool()
    loss=ssl_loss(out,target,m)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.encoder.proj.weight.grad is not None
    assert net.encoder.proj.weight.grad.abs().sum()>0

def test_ssl_loss_zero_for_identical_inputs():
    x=torch.rand(2,5,16,16)
    assert abs(ssl_loss(x,x,torch.ones_like(x,dtype=torch.bool)).item())<1e-7

def test_train_snapshot_load_hook():
    code=(Path(__file__).resolve().parents[1]/'src/train_exp19.py').read_text()
    assert 'EXP19_SSL_ENCODER_INIT' in code
    assert 'net.encoder.load_state_dict' in code
    assert code.index('net.encoder.load_state_dict')<code.index('ema_model = None')
