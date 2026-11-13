import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from nets.exp23_pixel import (date_value,pixel_features,PixelPolicy,counterfactual_targets,
                              choose_actions,apply_policy,confusion,metrics)

def test_input_date_and_no_date():
    x=torch.rand(1,5,16,16); z=torch.rand(1,3,16,16)
    a=pixel_features(x,z,date_value('2023-05-25_0150'),True)
    b=pixel_features(x,z,date_value('2023-06-15_0150'),True)
    c=pixel_features(x,z,date_value('2023-06-15_0150'),False)
    assert a.shape==(1,16,16,16) and c.shape==a.shape
    assert torch.all(a[:,-1]==0) and torch.all(b[:,-1]==1)
    assert torch.all(c[:,-1]==0)

def test_oracle_fixes_without_damage():
    z=torch.tensor([[[[0.]],[[1.]],[[1.2]]]])
    gt=torch.tensor([[[1]]])
    target,eligible,fixed=counterfactual_targets(z,gt)
    assert fixed.item() and eligible.item() and target.item() in (1,4)
    changed=apply_policy(z,target).argmax(1)
    assert changed.item()==1

def test_keep_no_effect():
    z=torch.randn(1,3,8,8)
    a=torch.zeros(1,8,8,dtype=torch.long)
    assert torch.equal(z,apply_policy(z,a))
    p=PixelPolicy()
    y=p(torch.randn(1,16,8,8))
    y.mean().backward()
    assert any(q.grad is not None for q in p.parameters())

def test_confusion_and_mask():
    gt=torch.tensor([[[0,1,2]]]); pred=torch.tensor([[[0,2,2]]])
    cm=confusion(pred,gt)
    assert cm.sum()==3 and metrics(cm)['IoU_weed']==50
    choices=choose_actions(torch.zeros(1,5,1,3),torch.zeros(1,1,3,dtype=torch.bool))
    assert not choices.any()
