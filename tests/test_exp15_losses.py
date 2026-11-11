import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from exp15_losses import cw_lovasz_loss,spectral_spatial_cw_loss

def data():
    torch.manual_seed(7)
    z=torch.randn(2,3,32,32,requires_grad=True)
    y=torch.randint(0,3,(2,32,32))
    x=torch.randn(2,5,32,32)
    return z,y,x

def test_lovasz_grad():
    z,y,_=data();l=cw_lovasz_loss(z,y);assert l.ndim==0 and torch.isfinite(l);l.backward();assert torch.isfinite(z.grad).all()

def test_spectral_grad():
    z,y,x=data();y[:,10:18,10:18]=1;y[:,18:24,10:18]=2
    l=spectral_spatial_cw_loss(z,y,x);assert l.ndim==0 and torch.isfinite(l);l.backward();assert torch.isfinite(z.grad).all()

def test_empty_valid():
    z,y,x=data();y.fill_(-1);assert cw_lovasz_loss(z,y).item()==0;assert spectral_spatial_cw_loss(z,y,x).item()==0

def test_missing_class():
    z,y,x=data();y.fill_(1);assert torch.isfinite(spectral_spatial_cw_loss(z,y,x));assert torch.isfinite(cw_lovasz_loss(z,y))

def test_shape_guards():
    z,y,x=data()
    for f in (lambda:cw_lovasz_loss(z[:,:2],y),lambda:spectral_spatial_cw_loss(z,y,x[:,:4])):
        try:f()
        except ValueError:pass
        else:raise AssertionError('missing validation')
