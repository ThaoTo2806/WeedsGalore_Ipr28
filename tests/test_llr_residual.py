"""Synthetic correctness gates for the E10 LLR residual heads."""

import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nets.llr_subcell import LLRResidualSubcellSegFormer
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
    return LLRResidualSubcellSegFormer(
        num_classes=3,
        in_channels=5,
        decoder_dim=16,
        pretrained_backbone=False,
        head=head,
        beta=0.5,
        residual_alpha_init=0.02,
        encoder=SyntheticEncoder(),
    )


def test_heads_have_matching_parameter_count_and_initialization():
    b1 = build_head('llr_residual2', seed=11)
    m1 = build_head('llr_residual_occ2', seed=11)
    assert sum(parameter.numel() for parameter in b1.parameters()) == sum(
        parameter.numel() for parameter in m1.parameters())
    assert torch.equal(b1.placement.weight, m1.placement.weight)
    assert torch.equal(b1.placement.bias, m1.placement.bias)
    assert torch.equal(b1.occupancy.weight, m1.occupancy.weight)
    assert torch.equal(b1.occupancy.bias, m1.occupancy.bias)
    assert torch.equal(b1.alpha_logit, m1.alpha_logit)


def test_zero_residual_reconstructs_bilinear_output():
    model = build_head('llr_residual_occ2').eval()
    baseline = SegFormer5Band(
        num_classes=3, decoder_dim=16, pretrained_backbone=False, in_channels=5,
        use_low_level_refine=True, use_multiscale_fusion=True,
        encoder=SyntheticEncoder(), llr_source='raw',
    ).eval()
    baseline.load_state_dict({key: value for key, value in model.state_dict().items()
                              if key in baseline.state_dict()}, strict=False)
    x = torch.rand(1, 5, 64, 64)
    with torch.no_grad():
        expected = baseline(x)
        model.alpha_logit.fill_(-100.0)
        actual, extras = model(x, return_aux=True)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual, extras['base_logits'], atol=1e-6, rtol=1e-6)


def test_cell_residual_is_zero_mean_per_class_and_cell():
    model = build_head('llr_residual_occ2').eval()
    with torch.no_grad():
        _, extras = model(torch.rand(1, 5, 64, 64), return_aux=True)
    residual = extras['cell_residual']
    assert torch.allclose(residual.mean(dim=2), torch.zeros_like(residual.mean(dim=2)), atol=1e-6)


def test_occupancy_changes_m1_but_not_b1_when_placement_is_fixed():
    x = torch.rand(1, 5, 64, 64)
    m1 = build_head('llr_residual_occ2').eval()
    b1 = build_head('llr_residual2').eval()
    with torch.no_grad():
        m1.occupancy.bias.add_(torch.tensor([0.0, 0.7, -0.7]))
        b1.occupancy.bias.add_(torch.tensor([0.0, 0.7, -0.7]))
        m1_output_before = m1(x)
        b1_output_before = b1(x)
        m1.occupancy.bias.add_(torch.tensor([0.5, -0.5, 0.0]))
        b1.occupancy.bias.add_(torch.tensor([0.5, -0.5, 0.0]))
        m1_output_after = m1(x)
        b1_output_after = b1(x)
    assert (m1_output_after - m1_output_before).abs().sum() > 1e-6
    assert torch.allclose(b1_output_after, b1_output_before, atol=1e-6, rtol=1e-6)


def test_separate_segmentation_and_auxiliary_gradients():
    x = torch.rand(2, 5, 64, 64)
    labels = torch.randint(0, 3, (2, 64, 64))
    m1 = build_head('llr_residual_occ2').train()
    output, extras = m1(x, return_aux=True)
    segmentation_loss = F.cross_entropy(output, labels)
    gradients = torch.autograd.grad(
        segmentation_loss, (m1.placement.weight, m1.occupancy.weight, m1.alpha_logit),
        allow_unused=True,
    )
    assert gradients[0] is not None and gradients[0].abs().sum() > 0
    assert gradients[1] is not None and gradients[1].abs().sum() > 0
    assert gradients[2] is not None and gradients[2].abs().sum() > 0

    m1.zero_grad(set_to_none=True)
    aux_loss = occupancy_loss(extras['occ_logits'], labels, 3)
    aux_loss.backward()
    assert m1.occupancy.weight.grad is not None
    assert m1.occupancy.weight.grad.abs().sum() > 0

    b1 = build_head('llr_residual2').train()
    b1_output, b1_extras = b1(x, return_aux=True)
    F.cross_entropy(b1_output, labels).backward()
    assert b1.occupancy.weight.grad is None
    b1.zero_grad(set_to_none=True)
    occupancy_loss(b1_extras['occ_logits'], labels, 3).backward()
    assert b1.occupancy.weight.grad is not None
    assert b1.occupancy.weight.grad.abs().sum() > 0


def test_synthetic_optimizer_step_and_checkpoint_round_trip(tmp_path):
    model = build_head('llr_residual_occ2').train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.rand(2, 5, 64, 64)
    labels = torch.randint(0, 3, (2, 64, 64))
    output, extras = model(x, return_aux=True)
    loss = F.cross_entropy(output, labels) + 0.3 * occupancy_loss(extras['occ_logits'], labels, 3)
    loss.backward()
    optimizer.step()
    path = tmp_path / 'smoke.pth'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)

    restored = build_head('llr_residual_occ2').train()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-4)
    state = torch.load(path, map_location='cpu', weights_only=False)
    restored.load_state_dict(state['model'])
    restored_optimizer.load_state_dict(state['optimizer'])
    with torch.no_grad():
        restored_output = restored(x)
    assert restored_output.shape == output.shape
    assert torch.isfinite(restored_output).all()
    assert torch.allclose(restored_output, model(x), atol=1e-6, rtol=1e-6)
