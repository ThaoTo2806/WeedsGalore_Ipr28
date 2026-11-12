import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nets.exp21_hier import fine_to_three_logits

def test_probability_merge():
    x=torch.arange(6,dtype=torch.float32).reshape(1,6,1,1).requires_grad_()
    y=fine_to_three_logits(x)
    assert y.shape==(1,3,1,1)
    p=x.softmax(1);q=y.softmax(1)
    assert torch.allclose(q[:,0:2],p[:,0:2],atol=1e-6)
    assert torch.allclose(q[:,2],p[:,2:].sum(1),atol=1e-6)
    y.sum().backward();assert torch.isfinite(x.grad).all()

def test_no_label_argument_in_forward():
    import inspect
    from nets.exp21_hier import HierarchicalSegFormer
    assert tuple(inspect.signature(HierarchicalSegFormer.forward).parameters)==('self','x')

def test_requires_six_classes():
    import pytest
    with pytest.raises(ValueError):fine_to_three_logits(torch.randn(1,3,4,4))
