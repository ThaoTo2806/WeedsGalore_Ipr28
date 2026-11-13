import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from nets.exp25_boundary import (ForegroundHead,make_features,edit_logits,boundary_target,
    training_loss,metrics,cm_from_logits,date_value)


def test_zero_change_and_grad():
    x=torch.rand(1,5,16,16);z=torch.randn(1,3,16,16);gt=torch.randint(0,3,(1,16,16))
    f=make_features(x,z,date_value('2023-05-25_0150'))
    assert f.shape==(1,15,16,16)
    net=ForegroundHead();o,e,d=net(f)
    assert torch.equal(edit_logits(z,d,e,False),z)
    loss,_=training_loss(o,e,d,z,gt,True)
    loss.backward()
    assert net.out.weight.grad.abs().sum()>0


def test_date_ablation():
    x=torch.rand(1,5,8,8);z=torch.rand(1,3,8,8)
    a=make_features(x,z,date_value('2023-06-15_0956'),False)
    b=make_features(x,z,date_value('2023-06-15_0956'),True)
    assert a[:,-1].abs().sum()==0 and (b[:,-1]==1).all()


def test_boundary_and_class_odds():
    gt=torch.zeros(1,16,16,dtype=torch.long);gt[:,4:8,4:8]=2
    b=boundary_target(gt)
    assert b[0,0,4,4]>0 and b[0,0,0,0]==0
    z=torch.randn(1,3,16,16);d=torch.ones(1,1,16,16);e=torch.zeros_like(d)
    q=edit_logits(z,d,e,True)
    assert torch.allclose(q[:,1]-q[:,2],z[:,1]-z[:,2])


def test_cm():
    gt=torch.tensor([[[0,1,2]]]);z=torch.tensor([[[[5,0,0]],[[0,0,0]],[[0,4,5]]]],dtype=torch.float)
    cm=cm_from_logits(z,gt)
    assert cm.sum()==3 and metrics(cm)['mIoU']>=0
