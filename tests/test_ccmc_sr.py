"""Unit tests for CCMC-SR (plan section 3.1 / 8.4 item 5).

Part 1 (synthetic tensors, no HF download): window mean, identity, edges, shapes, gradient,
           shuffles, corruption, instance matching.
Part 2 (needs transformers + internet for nvidia/mit-b0): encoder equivalence with HF,
           identity at init, Q/stage-1 invariance, K/V change, backward, 512/600, batch>1.

Run:  python -m pytest -q tests/test_ccmc_sr.py
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from nets.ccmc_sr import (corrupt_weed_labels, global_shuffle, image_normalized_factor,  # noqa: E402
                          occupancy, window_normalized_factor, window_shuffle)
from ccmc_metrics import match_instances, weed_instances, image_stats, metrics_from_sums  # noqa: E402

# MiT-B0 on 600x600: (feature size, sr_ratio) of stages 1, 2, 3
STAGE_SHAPES = [(150, 8), (75, 4), (38, 2)]


# ----------------------------------------------------------------------------------------
# Part 1: operator
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize('size,ratio', STAGE_SHAPES + [(128, 8), (64, 4), (32, 2)])
@pytest.mark.parametrize('eps', [0.05, 0.1, 0.3])
def test_window_mean_is_one_and_edges_are_one(size, ratio, eps):
    torch.manual_seed(0)
    prior = torch.rand(2, 1, size, size) ** 3  # skewed, many near-zero values
    lam = torch.tensor(0.7)
    factor = window_normalized_factor(prior, ratio, lam, eps)
    assert factor.shape == prior.shape
    hc = (size // ratio) * ratio
    win_mean = torch.nn.functional.avg_pool2d(factor[..., :hc, :hc], ratio, ratio)
    assert torch.allclose(win_mean, torch.ones_like(win_mean), atol=1e-5)
    assert torch.all(factor[..., hc:, :] == 1) and torch.all(factor[..., :, hc:] == 1)
    # SR output grid equals the avg-pool grid of the factor
    sr = torch.nn.Conv2d(1, 1, ratio, ratio)
    assert sr(prior).shape[-2:] == win_mean.shape[-2:]


def test_identity_uniform_prior_and_zero_lambda():
    prior_uniform = torch.full((2, 1, 75, 75), 0.5)
    f = window_normalized_factor(prior_uniform, 4, torch.tensor(0.9), 0.1)
    assert torch.allclose(f, torch.ones_like(f), atol=1e-6)  # equal up to float rounding
    prior = torch.rand(2, 1, 75, 75)
    f0 = window_normalized_factor(prior, 4, torch.tensor(0.0), 0.1)
    assert torch.equal(f0, torch.ones_like(f0))
    g = image_normalized_factor(prior_uniform, torch.tensor(0.9), 0.1)
    assert torch.allclose(g, torch.ones_like(g))


def test_example_from_plan_single_weed_pixel():
    # one weed pixel in an 8x8 window, eps 0.1 -> w_norm ~ 8.77
    prior = torch.zeros(1, 1, 8, 8)
    prior[0, 0, 3, 3] = 1.0
    f = window_normalized_factor(prior, 8, torch.tensor(1.0), 0.1)
    assert abs(f[0, 0, 3, 3].item() - 1.0 / ((0.1 * 63 + 1.0) / 64)) < 1e-4
    assert abs(f[0, 0, 3, 3].item() - 8.767) < 1e-2


def test_image_normalized_mean_is_one():
    prior = torch.rand(3, 1, 38, 38)
    g = image_normalized_factor(prior, torch.tensor(0.5), 0.1)
    assert torch.allclose(g.mean(dim=(2, 3)), torch.ones(3, 1), atol=1e-5)


def test_gradient_finite_and_nonzero_at_uniform_prior():
    logits = torch.zeros(2, 1, 75, 75, requires_grad=True)  # zero-init head -> p = 0.5
    a = torch.tensor(np.log(0.05 / 0.95), requires_grad=True)
    x = torch.randn(2, 64, 75, 75)
    sr = torch.nn.Conv2d(64, 64, 4, 4)
    factor = window_normalized_factor(torch.sigmoid(logits), 4, torch.sigmoid(a), 0.1)
    out = sr(x * factor)
    (out ** 2).mean().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(a.grad).all()
    assert logits.grad.abs().sum() > 0  # head receives gradient from the main loss


def test_autocast_dtype_safe():
    prior = torch.rand(1, 1, 75, 75).half()
    f = window_normalized_factor(prior, 4, torch.tensor(0.5), 0.1)
    assert f.dtype == torch.float32


def test_window_shuffle_preserves_window_values_and_edges():
    prior = torch.rand(2, 1, 38, 38)
    out = window_shuffle(prior, 2, generator=torch.Generator().manual_seed(1))
    a = torch.nn.functional.unfold(prior, 2, stride=2).sort(dim=1).values
    b = torch.nn.functional.unfold(out, 2, stride=2).sort(dim=1).values
    assert torch.allclose(a, b)
    assert not torch.equal(prior, out)
    f_a = window_normalized_factor(prior, 2, torch.tensor(0.5), 0.1)
    f_b = window_normalized_factor(out, 2, torch.tensor(0.5), 0.1)
    pa = torch.nn.functional.avg_pool2d(f_a, 2, 2)
    pb = torch.nn.functional.avg_pool2d(f_b, 2, 2)
    assert torch.allclose(pa, pb, atol=1e-5)
    p75 = torch.rand(1, 1, 75, 75)
    o75 = window_shuffle(p75, 4)
    assert torch.equal(p75[..., 72:, :], o75[..., 72:, :])
    assert torch.equal(p75[..., :, 72:], o75[..., :, 72:])


def test_global_shuffle_preserves_values():
    prior = torch.rand(2, 1, 20, 20)
    out = global_shuffle(prior)
    assert torch.allclose(prior.flatten(1).sort().values, out.flatten(1).sort().values)


def test_occupancy_and_corruption():
    labels = torch.zeros(1, 600, 600, dtype=torch.long)
    labels[0, 100:110, 100:110] = 2
    labels[0, 300:340, 300:340] = 1
    occ = occupancy(labels, 'weed', (150, 150))
    assert abs(occ.sum().item() * 16 - 100) < 1e-3
    veg = occupancy(labels, 'veg', (150, 150))
    assert veg.sum() > occ.sum()
    dil = corrupt_weed_labels(labels, 'dilate:2')
    assert (dil == 2).sum() == 14 * 14
    ero = corrupt_weed_labels(labels, 'erode:2')
    assert (ero == 2).sum() == 6 * 6
    drop = corrupt_weed_labels(labels, 'drop:1.0')
    assert (drop == 2).sum() == 0 and (drop == 1).sum() == 1600
    assert torch.equal(corrupt_weed_labels(labels, 'none'), labels)


# ----------------------------------------------------------------------------------------
# Part 1b: metrics
# ----------------------------------------------------------------------------------------

def test_sinkhorn_marginals_and_placement():
    from nets.mcsd import sinkhorn_subcell
    torch.manual_seed(0)
    b, c, r2, h, w = 2, 3, 16, 5, 7
    log_q = torch.log_softmax(torch.randn(b, c, h, w), dim=1)
    scores = torch.randn(b, c, r2, h, w) * 2
    log_p = sinkhorn_subcell(log_q, scores, iters=50)
    p = log_p.exp()
    assert torch.allclose(p.sum(1), torch.ones(b, r2, h, w), atol=1e-5)          # columns exact
    assert torch.allclose(p.sum(2), r2 * log_q.exp(), atol=1e-3)                 # class mass kept
    # uniform placement -> every sub-pixel gets the cell fractions
    p0 = sinkhorn_subcell(log_q, torch.zeros(b, c, r2, h, w), iters=5).exp()
    assert torch.allclose(p0, log_q.exp().unsqueeze(2).expand_as(p0), atol=1e-5)
    # a 1/16 weed fraction goes to the sub-pixel with the highest weed score
    log_q1 = torch.log(torch.tensor([15 / 16 - 1e-4, 1e-4, 1 / 16]).view(1, 3, 1, 1))
    sc = torch.zeros(1, 3, 16, 1, 1)
    sc[0, 2, 5] = 8.0
    p1 = sinkhorn_subcell(log_q1, sc, iters=50).exp()
    assert p1[0, :, 5, 0, 0].argmax() == 2 and p1[0, 2].sum() > 0.95


def test_occupancy_conditioned_subcell():
    from nets.mcsd import occupancy_conditioned_subcell
    log_q = torch.log(torch.tensor([0.7, 0.2, 0.1])).view(1, 3, 1, 1).requires_grad_()
    scores = torch.zeros(1, 3, 16, 1, 1, requires_grad=True)
    log_p = occupancy_conditioned_subcell(log_q, scores)
    p = log_p.exp()
    expected = log_q.exp().unsqueeze(2).expand_as(p)
    assert torch.allclose(p, expected, atol=1e-6)
    assert torch.allclose(p.sum(1), torch.ones(1, 16, 1, 1), atol=1e-6)

    placed_scores = scores.detach().clone()
    placed_scores[0, 2, 0] = 4.0
    placed_scores.requires_grad_()
    changed = occupancy_conditioned_subcell(log_q, placed_scores).exp()
    assert changed[0, 2, 0, 0, 0] > p[0, 2, 0, 0, 0]
    assert not torch.allclose(changed.mean(2), log_q.exp(), atol=1e-4)
    weights = torch.linspace(-1.0, 1.0, 16).view(1, 16, 1, 1)
    loss = (changed[:, 2] * weights).sum()
    loss.backward()
    assert torch.isfinite(log_q.grad).all()
    assert log_q.grad.abs().sum() > 1e-8
    assert torch.isfinite(placed_scores.grad).all()
    assert placed_scores.grad.abs().sum() > 1e-8


def test_pixel_shuffle_layout_and_gradient():
    from nets.mcsd import sinkhorn_subcell
    import torch.nn.functional as F
    b, c, r, h, w = 1, 3, 4, 2, 2
    log_q = torch.log_softmax(torch.randn(b, c, h, w), 1).requires_grad_()
    scores = torch.randn(b, c, r * r, h, w, requires_grad=True)
    log_p = sinkhorn_subcell(log_q, scores, iters=5)
    out = F.pixel_shuffle(log_p.reshape(b, c * r * r, h, w), r)
    assert out.shape == (b, c, h * r, w * r)
    # cell (0,0) of the output = sub-pixels of decoder cell (0,0)
    cell = out[0, :, :r, :r].reshape(c, r * r)
    assert torch.allclose(cell, log_p[0, :, :, 0, 0])
    out.exp()[0, 2].sum().backward()
    assert torch.isfinite(log_q.grad).all() and torch.isfinite(scores.grad).all()


def test_occupancy_loss_minimum_at_true_fractions():
    from nets.mcsd import occupancy_loss
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[0, :2, :2] = 2
    target = torch.zeros(1, 3, 2, 2)
    target[0, 0] = 1.0
    target[0, :, 0, 0] = torch.tensor([12 / 16, 0, 4 / 16])
    good = occupancy_loss(torch.log(target.clamp(min=1e-6)), labels, 3)
    bad = occupancy_loss(torch.zeros(1, 3, 2, 2), labels, 3)
    assert good < bad


def test_instance_matching_one_to_one():
    label = np.zeros((60, 60), dtype=np.uint8)
    label[5:10, 5:10] = 2      # small, detected
    label[20:40, 20:40] = 2    # large, detected
    label[50:52, 50:52] = 2    # tiny, missed
    inst, src = weed_instances(label)
    assert src == 'cc' and inst.max() == 3
    pred = np.zeros_like(label, dtype=bool)
    pred[5:10, 5:10] = True
    pred[21:40, 20:40] = True
    m = match_instances(inst, pred)
    assert m['matched_50'].sum() == 2
    # one prediction covering two GT instances can match only one of them
    label2 = np.zeros((30, 30), dtype=np.uint8)
    label2[5:10, 5:10] = 2
    label2[5:10, 11:16] = 2
    inst2, _ = weed_instances(label2)
    pred2 = np.zeros_like(label2, dtype=bool)
    pred2[5:10, 5:16] = True
    m2 = match_instances(inst2, pred2, thresholds=(30,))
    assert m2['matched_30'].sum() == 1
    assert (m2['coverage'] == 1).all()


def test_official_instances_restricted_to_weed():
    label = np.zeros((20, 20), dtype=np.uint8)
    label[0:5, 0:5] = 2
    label[10:15, 10:15] = 1
    label[15:18, 15:18] = 2   # weed pixels without official id
    raw = np.zeros((20, 20), dtype=np.int64)
    raw[0:5, 0:5] = 7
    raw[10:15, 10:15] = 9     # crop instance id, must be ignored
    inst, src = weed_instances(label, raw)
    assert src == 'official' and len(np.unique(inst[inst > 0])) == 2
    assert (inst[10:15, 10:15] == 0).all()


def test_image_stats_perfect_prediction():
    label = np.zeros((100, 100), dtype=np.uint8)
    label[10:20, 10:20] = 2
    label[50:80, 50:80] = 1
    inst, _ = weed_instances(label)
    s = image_stats(label.copy(), label, inst, edges=np.array([50.0, 500.0]))
    m = metrics_from_sums(s)
    assert m['mIoU'] == 100 and m['recall50_all'] == 100 and m['n_medium'] == 1


# ----------------------------------------------------------------------------------------
# Part 2: HF integration (skipped without transformers / network)
# ----------------------------------------------------------------------------------------

def _hf_available():
    try:
        import transformers  # noqa: F401
        from nets.segformer import PretrainedMiTEncoder
        PretrainedMiTEncoder(in_channels=5)
        return True
    except Exception:
        return False


HF = _hf_available()
needs_hf = pytest.mark.skipif(not HF, reason='transformers / nvidia/mit-b0 not available')


@needs_hf
def test_encoder_loop_matches_hf_forward():
    from nets.ccmc_sr import CCMCMiTEncoder
    from nets.segformer import PretrainedMiTEncoder
    torch.manual_seed(0)
    base = PretrainedMiTEncoder(in_channels=5).eval()
    ccmc = CCMCMiTEncoder(in_channels=5, prior_mode='none').eval()
    ccmc.load_state_dict(base.state_dict())
    x = torch.randn(2, 5, 600, 600)
    with torch.no_grad():
        fa = base(x)
        fb, _ = ccmc(x)
    assert [tuple(f.shape) for f in fb] == [(2, 32, 150, 150), (2, 64, 75, 75),
                                           (2, 160, 38, 38), (2, 256, 19, 19)]
    for a, b in zip(fa, fb):
        assert torch.allclose(a, b, atol=1e-5)


def _pair(prior_mode='learned_weed', injection='ccmc_sr', **kw):
    from nets.ccmc_sr import CCMCSegFormer
    torch.manual_seed(0)
    l0 = CCMCSegFormer(prior_mode='none').eval()
    l3 = CCMCSegFormer(prior_mode=prior_mode, injection=injection, **kw).eval()
    missing, unexpected = l3.load_state_dict(l0.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith('prior_head') or k.startswith('encoder.lam_logit') for k in missing)
    return l0, l3


@needs_hf
@pytest.mark.parametrize('size', [512, 600])
def test_identity_at_init_learned_prior(size):
    l0, l3 = _pair('learned_weed')
    x = torch.randn(2, 5, size, size)
    with torch.no_grad():
        assert torch.allclose(l0(x), l3(x), atol=1e-5)


@needs_hf
@pytest.mark.parametrize('injection', ['ccmc_sr', 'pre_sr_gate', 'post_sr_gate'])
def test_identity_with_zero_lambda_and_gt_prior(injection):
    l0, c3 = _pair('gt_weed', injection=injection, lam_init=0.5)
    with torch.no_grad():
        c3.encoder.lam_logit.fill_(-1e4)  # sigmoid -> 0
    labels = torch.zeros(1, 600, 600, dtype=torch.long)
    labels[0, 200:260, 200:260] = 2
    x = torch.randn(1, 5, 600, 600)
    with torch.no_grad():
        assert torch.allclose(l0(x), c3(x, prior_mask=labels), atol=1e-5)


@needs_hf
@pytest.mark.parametrize('injection', ['ccmc_sr', 'pre_sr_gate', 'post_sr_gate'])
def test_only_kv_of_stage_2_3_change(injection):
    l0, c3 = _pair('gt_weed', injection=injection, lam_init=0.9)
    labels = torch.zeros(1, 600, 600, dtype=torch.long)
    labels[0, 100:140, 100:300] = 2
    x = torch.randn(1, 5, 600, 600)
    recorded = {}

    def rec(tag):
        def hook(module, args, output):
            recorded[tag] = output.detach().clone()
        return hook
    handles = []
    for model, tag in ((l0, 'l0'), (c3, 'c3')):
        att = model.encoder.encoder.block[1][0].attention.self
        handles.append(att.query.register_forward_hook(rec(f'{tag}_q2')))
        handles.append(att.key.register_forward_hook(rec(f'{tag}_k2')))
        handles.append(model.encoder.encoder.block[0][0].attention.self.key.register_forward_hook(
            rec(f'{tag}_k1')))
    with torch.no_grad():
        f0, _ = l0.encoder(x)
        f3, _ = c3.encoder(x, prior_fn=lambda f1: c3._build_prior(x, f1, labels, None))
    for h in handles:
        h.remove()
    assert torch.allclose(f0[0], f3[0], atol=1e-6)            # stage 1 output unchanged
    assert torch.allclose(recorded['l0_k1'], recorded['c3_k1'])  # stage 1 K/V unchanged
    assert torch.allclose(recorded['l0_q2'], recorded['c3_q2'])  # Q of stage 2 unchanged
    assert not torch.allclose(recorded['l0_k2'], recorded['c3_k2'], atol=1e-6)  # K/V changed
    assert not torch.allclose(f0[2], f3[2], atol=1e-6)


@needs_hf
@pytest.mark.parametrize('variant', [dict(prior_mode='learned_weed'),
                                     dict(prior_mode='learned_weed', prior_perturb='window_shuffle'),
                                     dict(prior_mode='learned_weed', injection='post_sr_gate'),
                                     dict(prior_mode='ndvi'),
                                     dict(prior_mode='learned_free', injection='none')])
def test_backward_train_mode_batch2(variant):
    from nets.ccmc_sr import CCMCSegFormer
    torch.manual_seed(0)
    net = CCMCSegFormer(**variant).train()
    x = torch.rand(2, 5, 600, 600)
    out, extras = net(x, return_aux=True)
    loss = out.mean() + (extras['prior_logits'].mean() if 'prior_logits' in extras else 0)
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    if net.prior_head is not None and variant.get('injection', 'ccmc_sr') != 'none':
        assert net.prior_head.out.weight.grad.abs().sum() > 0
        assert net.encoder.lam_logit.grad is not None


@needs_hf
def test_stage1_modulation_is_rejected():
    from nets.ccmc_sr import CCMCSegFormer
    with pytest.raises(ValueError):
        CCMCSegFormer(prior_mode='learned_weed', stages=(1, 2))


@needs_hf
@pytest.mark.parametrize('head,guide', [('mcsd', False), ('mcsd', True),
                                       ('pixelshuffle', False), ('occ_conditioned', True)])
def test_mcsd_forward_backward(head, guide):
    from nets.mcsd import MCSDSegFormer
    torch.manual_seed(0)
    net = MCSDSegFormer(head=head, spectral_guide=guide).train()
    x = torch.rand(2, 5, 600, 600)
    labels = torch.randint(0, 3, (2, 600, 600))
    out, extras = net(x, return_aux=True)
    assert out.shape == (2, 3, 600, 600)
    assert torch.allclose(out.exp().sum(1), torch.ones(2, 600, 600), atol=1e-4)  # valid log-probs
    segmentation_loss = torch.nn.functional.nll_loss(out, labels)
    segmentation_loss.backward()
    assert net.placement.weight.grad.abs().sum() > 0
    if head == 'occ_conditioned':
        assert net.occupancy.weight.grad is not None
        assert net.occupancy.weight.grad.abs().sum() > 0
        assert net.guide[-1].weight.grad is not None
        assert net.guide[-1].weight.grad.abs().sum() > 0
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


@needs_hf
def test_mcsd_decoder_init_matches_l0():
    from nets.mcsd import MCSDSegFormer
    from nets.segformer import SegFormer5Band
    torch.manual_seed(0)
    l0 = SegFormer5Band(pretrained_backbone=True, use_low_level_refine=False)
    torch.manual_seed(0)
    m = MCSDSegFormer(head='mcsd')
    sd0, sdm = l0.state_dict(), m.state_dict()
    for k in sd0:
        if k.startswith('classifier'):
            continue
        assert torch.equal(sd0[k], sdm[k]), k


@needs_hf
def test_occ_conditioned_init_matches_mcsd_g_same_seed():
    from nets.mcsd import MCSDSegFormer
    torch.manual_seed(17)
    mcsd_g = MCSDSegFormer(head='mcsd', spectral_guide=True)
    torch.manual_seed(17)
    occ_cond = MCSDSegFormer(head='occ_conditioned', spectral_guide=True)
    state_mcsd, state_occ = mcsd_g.state_dict(), occ_cond.state_dict()
    assert state_mcsd.keys() == state_occ.keys()
    for key in state_mcsd:
        assert torch.equal(state_mcsd[key], state_occ[key]), key
