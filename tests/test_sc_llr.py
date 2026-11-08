"""Synthetic correctness tests; no WeedsGalore labels/images and no network used."""
import inspect
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer, SpectralContrastOperator  # noqa: E402
from nets.segformer import SegFormer5Band  # noqa: E402


class SyntheticEncoder(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        return tuple(F.adaptive_avg_pool2d(x[:, :1], (h // s, w // s)).repeat(1, c, 1, 1)
                     for s, c in zip((4, 8, 16, 32), (32, 64, 160, 256)))


def model(head, seed=4):
    torch.manual_seed(seed)
    return SCLLRSegFormer(decoder_dim=16, encoder=SyntheticEncoder(),
                          pretrained_backbone=False, head=head)


def baseline_like(m):
    base = SegFormer5Band(num_classes=3, decoder_dim=16, pretrained_backbone=False,
                          in_channels=5, use_low_level_refine=True,
                          use_multiscale_fusion=True, encoder=SyntheticEncoder(),
                          llr_source='raw').eval()
    return base


def test_parameter_counts_and_shared_initialization():
    ms = [model(h) for h in SC_LLR_HEADS]
    assert len({sum(p.numel() for p in m.parameters()) for m in ms}) == 1
    for m in ms[1:]:
        for name in ('spectral_projection', 'semantic_projection'):
            assert torch.equal(getattr(m.sc_llr, name).weight, getattr(ms[0].sc_llr, name).weight)


def test_trainable_parameters_actually_used_differ_by_under_one_percent():
    """All three arms instantiate the same parameter set."""
    counts = {}
    for head in SC_LLR_HEADS:
        counts[head] = sum(parameter.numel() for parameter in model(head).parameters())
    spread = (max(counts.values()) - min(counts.values())) / min(counts.values())
    assert spread < 0.01, counts


def test_identity_and_checkpoint_roundtrip(tmp_path):
    m = model('sc_llr_spec').eval()
    baseline = baseline_like(m)
    shared = {key: value for key, value in m.state_dict().items()
              if key in baseline.state_dict()}
    baseline.load_state_dict(shared, strict=False)
    missing, unexpected = m.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith('sc_llr.') for k in missing), missing   # only SC params are new
    x = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        m.set_alpha_override(0.0)
        assert torch.allclose(m(x), baseline(x), atol=1e-5, rtol=1e-5)
        m.set_alpha_override(None)
        m.set_branch_scale(0.0)                                      # activity-audit hook
        assert torch.allclose(m(x), baseline(x), atol=1e-5, rtol=1e-5)
        m.set_branch_scale(1.0)
        assert (m(x) - baseline(x)).abs().max() > 1e-7               # operator is active
    p = tmp_path / 'scllr.pth'
    torch.save(m.state_dict(), p)
    clone = model('sc_llr_spec')
    clone.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
    m.eval(); clone.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), clone(x), atol=1e-5, rtol=1e-5)


def test_normalization_masking_and_locality():
    op = SpectralContrastOperator(8)
    x, f = torch.randn(1, 5, 10, 10), torch.randn(1, 8, 5, 5)
    for head in SC_LLR_HEADS:
        w = op.weights(f, x, head)
        assert w.shape == (1, 9, 5, 5)
        assert torch.isfinite(w).all() and (w >= 0).all()
        assert torch.allclose(w.sum(1), torch.ones_like(w.sum(1)), atol=1e-5)
        # corner (0,0): neighbours at offset (-1,*) and (*,-1) are outside the image
        assert w[0, 0, 0, 0] < 1e-6 and w[0, 4, 0, 0] > 0
    with torch.no_grad():
        a = op(f, x, 'sc_llr_spec')[..., 1, 1]
        f2 = f.clone(); f2[..., 4, 4] += 20            # 3 cells away (> radius 1)
        b = op(f2, x, 'sc_llr_spec')[..., 1, 1]
        x2 = x.clone(); x2[..., 8:, 8:] += 20          # same far region in the image
        c = op(f, x2, 'sc_llr_spec')[..., 1, 1]
        near = f.clone(); near[..., 2, 2] += 20        # a neighbour: must change the output
        d = op(near, x, 'sc_llr_spec')[..., 1, 1]
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)
    assert not torch.allclose(a, d, atol=1e-6)


def test_arm_definitions_match_readme():
    op = SpectralContrastOperator(8)
    x, f = torch.randn(1, 5, 12, 12), torch.randn(1, 8, 6, 6)
    x2, f2 = x + torch.randn_like(x) * 2, torch.randn_like(f)
    with torch.no_grad():
        # spectral dependence: M1 only
        assert not torch.allclose(op.weights(f, x, 'sc_llr_spec'), op.weights(f, x2, 'sc_llr_spec'))
        for h in ('sc_llr_static', 'sc_llr_sem'):
            assert torch.allclose(op.weights(f, x, h), op.weights(f, x2, h))
        # semantic dependence: M1 and M2, not B1
        for h in ('sc_llr_sem', 'sc_llr_spec'):
            assert not torch.allclose(op.weights(f, x, h), op.weights(f2, x, h))
        assert torch.allclose(op.weights(f, x, 'sc_llr_static'), op.weights(f2, x, 'sc_llr_static'))
        # per-offset scalars belong to the static arm ONLY
        before_static = op.weights(f, x, 'sc_llr_static')
        before = {h: op.weights(f, x, h) for h in ('sc_llr_sem', 'sc_llr_spec')}
        op.offset_logits.copy_(torch.randn(9))
        assert not torch.allclose(before_static, op.weights(f, x, 'sc_llr_static'))
        for h, w_before in before.items():
            assert torch.allclose(w_before, op.weights(f, x, h))


def test_gradients_finite_and_no_label_in_forward():
    assert list(inspect.signature(SCLLRSegFormer.forward).parameters) == ['self', 'x', 'return_aux']
    torch.manual_seed(1)
    m = model('sc_llr_spec')
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    x = torch.randn(2, 5, 64, 64)
    y = torch.randint(0, 3, (2, 64, 64))
    logits, extras = m(x, return_aux=True)
    assert logits.shape == (2, 3, 64, 64) and torch.isfinite(logits).all()
    F.cross_entropy(logits, y).backward()
    op = m.sc_llr
    for name, p in (('phi', op.spectral_projection.weight), ('psi', op.semantic_projection.weight),
                    ('alpha', op.alpha_logit), ('sigma', op.sigma_raw), ('tau', op.tau_raw)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    old = op.alpha_logit.detach().clone()
    opt.step()
    assert not torch.equal(old, op.alpha_logit.detach())
    assert 0 < float(extras['sc_alpha']) < 1
    assert 0 < float(extras['sc_weight_entropy']) <= torch.log(torch.tensor(9.0)) + 1e-4
    assert float(op.sigma) > 0 and float(op.tau) > 0