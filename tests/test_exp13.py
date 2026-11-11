import pathlib
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nets.exp13 import VCFSegFormer  # noqa: E402


class FakeEncoder(nn.Module):
    dims = (32, 64, 160, 256)

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(5, d, 1) for d in self.dims])

    def forward(self, x):
        return tuple(F.avg_pool2d(c(x), k) for c, k in zip(self.convs, (4, 8, 16, 32)))


def test_vcf_output_is_normalised_log_prob_and_trainable():
    net = VCFSegFormer(pretrained_backbone=False, encoder=FakeEncoder())
    x = torch.randn(2, 5, 64, 64)
    out = net(x)
    assert out.shape == (2, 3, 64, 64)
    assert torch.allclose(out.exp().sum(1), torch.ones(2, 64, 64), atol=1e-4)
    loss = F.cross_entropy(out, torch.randint(0, 3, (2, 64, 64)))
    loss.backward()
    assert all(p.grad is not None for p in net.cw_head.parameters())
    assert all(p.grad is not None for p in net.veg_head.parameters())


def test_vcf_return_aux():
    net = VCFSegFormer(pretrained_backbone=False, encoder=FakeEncoder())
    out, extras = net(torch.randn(1, 5, 64, 64), return_aux=True)
    assert extras == {} and out.shape[1] == 3
