"""Synthetic tests for 2x2 sub-cell heads on the raw-band LLR feature."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nets.llr_subcell import LLRSubcellSegFormer
from nets.mcsd import occupancy_loss
from nets.segformer import SegFormer5Band


class SyntheticEncoder(nn.Module):
    def forward(self, x):
        _, _, height, width = x.shape
        base = x[:, :1]
        sizes = [(height // 4, width // 4), (height // 8, width // 8),
                 (height // 16, width // 16), (height // 32, width // 32)]
        channels = (32, 64, 160, 256)
        return tuple(
            F.adaptive_avg_pool2d(base, size).repeat(1, count, 1, 1)
            for size, count in zip(sizes, channels)
        )


def build_head(head, seed=0):
    torch.manual_seed(seed)
    return LLRSubcellSegFormer(
        num_classes=3,
        in_channels=5,
        decoder_dim=16,
        pretrained_backbone=False,
        head=head,
        encoder=SyntheticEncoder(),
    )


@pytest.mark.parametrize('head', ['pixelshuffle2', 'occ_conditioned2', 'mcsd2'])
def test_llr_subcell_shapes_and_probabilities(head):
    model = build_head(head).eval()
    x = torch.rand(2, 5, 64, 64)
    with torch.no_grad():
        output, extras = model(x, return_aux=True)
    assert output.shape == (2, 3, 64, 64)
    assert extras['occ_logits'].shape == (2, 3, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(1), torch.ones(2, 64, 64), atol=2e-5)


@pytest.mark.parametrize('head', ['occ_conditioned2', 'mcsd2'])
def test_uniform_placement_preserves_cell_occupancy(head):
    model = build_head(head).eval()
    with torch.no_grad():
        model.placement.weight.zero_()
        model.placement.bias.zero_()
        x = torch.rand(1, 5, 64, 64)
        output, extras = model(x, return_aux=True)
    subcells = F.pixel_unshuffle(output.exp(), 2).view(1, 3, 4, 32, 32)
    expected_q = extras['occ_logits'].softmax(dim=1).unsqueeze(2)
    assert torch.allclose(subcells, expected_q.expand_as(subcells), atol=2e-5)


@pytest.mark.parametrize('head', ['pixelshuffle2', 'occ_conditioned2', 'mcsd2'])
def test_finite_gradients_and_occupancy_auxiliary_loss(head):
    model = build_head(head).train()
    x = torch.rand(2, 5, 64, 64)
    labels = torch.randint(0, 3, (2, 64, 64))
    output, extras = model(x, return_aux=True)
    segmentation_loss = F.nll_loss(output, labels)
    aux_loss = occupancy_loss(extras['occ_logits'], labels, 3)
    (segmentation_loss + 0.3 * aux_loss).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.placement.weight.grad is not None
    assert model.placement.weight.grad.abs().sum() > 0


def test_shared_llr_initialization_matches_bilinear_baseline():
    torch.manual_seed(19)
    baseline = SegFormer5Band(
        num_classes=3, decoder_dim=16, pretrained_backbone=False, in_channels=5,
        use_low_level_refine=True, use_multiscale_fusion=True,
        encoder=SyntheticEncoder(), llr_source='raw',
    )
    head = build_head('occ_conditioned2', seed=19)
    baseline_state = baseline.state_dict()
    head_state = head.state_dict()
    shared_keys = [key for key in baseline_state
                   if key in head_state and not key.startswith('classifier.')]
    assert shared_keys
    for key in shared_keys:
        assert torch.equal(baseline_state[key], head_state[key]), key


def test_three_subcell_variants_share_initial_llr_and_occupancy_weights():
    models = [build_head(head, seed=23) for head in
              ('pixelshuffle2', 'occ_conditioned2', 'mcsd2')]
    reference = models[0].state_dict()
    for model in models[1:]:
        candidate = model.state_dict()
        for key in ('linear1.weight', 'linear2.weight', 'linear3.weight', 'linear4.weight',
                    'fuse.0.weight', 'low_level.proj.0.weight', 'refine.0.weight',
                    'occupancy.weight', 'occupancy.bias', 'placement.weight', 'placement.bias'):
            assert torch.equal(reference[key], candidate[key]), key