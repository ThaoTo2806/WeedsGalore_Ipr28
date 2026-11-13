import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
import torch
from exp27_csbrl import ARMS, perturb_groups, teacher_reliability, csbrl_loss

class TinyTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x[:, :1], x[:, 1:2], x[:, 3:4]], 1)

def setup(seed=10):
    torch.manual_seed(seed)
    x = torch.randn(2,5,24,24)
    y = torch.zeros(2,24,24,dtype=torch.long)
    y[:, 4:18, 6:18] = 1
    y[:, 10:13, 12:17] = 2
    z = torch.randn(2,3,24,24,requires_grad=True)
    return x,y,z

def test_perturbation_preserves_selected_channels():
    x,_,_=setup()
    xr,xm=perturb_groups(x,0.35)
    assert torch.equal(xr[:,:3],x[:,:3])
    assert torch.equal(xm[:,3:],x[:,3:])
    assert not torch.equal(xr[:,3:],x[:,3:])
    assert not torch.equal(xm[:,:3],x[:,:3])

def test_teacher_reliability_bounded_and_detached():
    x,_,_=setup()
    maps=teacher_reliability(TinyTeacher(),x)
    for k in ('joint','rgb','ms'):
        assert maps[k].shape == (2,1,24,24)
        assert torch.isfinite(maps[k]).all()
        assert maps[k].min()>=0 and maps[k].max()<=1
        assert not maps[k].requires_grad

def test_all_arms_backward():
    x,y,z=setup()
    maps=teacher_reliability(TinyTeacher(),x)
    for arm in ARMS:
        zz=z.clone().detach().requires_grad_(True)
        loss,stats=csbrl_loss(zz,y,x,maps,arm,stage_ids=torch.tensor([0,3]))
        assert torch.isfinite(loss)
        loss.backward()
        assert zz.grad is not None
        assert torch.isfinite(zz.grad).all()
        assert 0<=stats['gate_fraction']<=1

def test_no_gt_boundary_zero():
    x,y,z=setup()
    y.zero_()
    maps=teacher_reliability(TinyTeacher(),x)
    loss,_=csbrl_loss(z,y,x,maps,'csbrl')
    assert abs(float(loss.detach())) < 1e-8

def test_reject_missing_maps_and_stage():
    x,y,z=setup()
    with pytest.raises(ValueError): csbrl_loss(z,y,x,None,'csbrl')
    maps=teacher_reliability(TinyTeacher(),x)
    with pytest.raises(ValueError): csbrl_loss(z,y,x,maps,'stage')
