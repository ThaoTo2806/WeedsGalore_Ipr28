import pathlib, sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nets.exp13_v2 import VCFContextSegFormer

class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(5, d, 1) for d in (32, 64, 160, 256)])
    def forward(self, x):
        return tuple(F.avg_pool2d(c(x), k) for c, k in zip(self.convs, (4, 8, 16, 32)))

def test_vcfctx_probabilities_and_gradients():
    net = VCFContextSegFormer(pretrained_backbone=False, encoder=FakeEncoder())
    pred = net(torch.randn(2, 5, 64, 64))
    assert pred.shape == (2, 3, 64, 64)
    assert torch.allclose(pred.exp().sum(1), torch.ones_like(pred[:, 0]), atol=1e-4)
    F.cross_entropy(pred, torch.randint(0, 3, (2,64,64))).backward()
    assert net.veg_head.weight.grad is not None
    assert net.cw_head.weight.grad is not None
    assert net.cw_merge[0].weight.grad is not None
    assert all(m.path[0].weight.grad is not None for m in net.cw_branch)

def test_vcfctx_aux_contract():
    net = VCFContextSegFormer(pretrained_backbone=False, encoder=FakeEncoder())
    out, aux = net(torch.randn(1,5,64,64), return_aux=True)
    assert out.shape == (1,3,64,64) and aux == {}
