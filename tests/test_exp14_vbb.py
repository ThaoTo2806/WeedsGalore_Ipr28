import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from exp14_vbb import vbb_loss

def test_gradients_finite():
    x=torch.randn(2,3,12,12,requires_grad=True)
    y=torch.zeros(2,12,12,dtype=torch.long)
    y[:,:,:6]=1
    y[:,3:8,6:]=2
    loss=vbb_loss(x,y,radius=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_all_one_label_is_zero():
    x=torch.randn(1,3,8,8,requires_grad=True)
    y=torch.zeros(1,8,8,dtype=torch.long)
    # A uniform image should have no vegetation boundary.
    loss=vbb_loss(x,y,radius=1)
    assert loss.item()==0

def test_crop_weed_interface_not_veg_bg_border():
    x=torch.randn(1,3,10,10)
    y=torch.ones(1,10,10,dtype=torch.long)
    y[:,:,5:]=2
    assert vbb_loss(x,y,radius=1).item()==0
