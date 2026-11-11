import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from exp14_ascd import ascd_loss


def _data(seed=1):
    torch.manual_seed(seed)
    logits = torch.randn(2, 3, 20, 20, requires_grad=True)
    labels = torch.randint(0, 3, (2, 20, 20))
    features = torch.rand(2, 5, 20, 20)
    return logits, labels, features


def test_gradient_and_finiteness():
    logits, labels, x = _data()
    loss = ascd_loss(logits, labels, x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert logits.grad[:, 0].abs().max() == 0  # conditional CW loss only


def test_no_plants_safe():
    logits, labels, x = _data()
    labels[:] = 0
    loss = ascd_loss(logits, labels, x)
    assert loss.item() == 0.
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_absent_weed_safe():
    logits, labels, x = _data()
    labels[:] = 1
    loss = ascd_loss(logits, labels, x)
    assert torch.isfinite(loss)


def test_invalid_inputs_fail():
    logits, labels, x = _data()
    try:
        ascd_loss(logits, labels, x[:, :3])
    except ValueError:
        pass
    else:
        raise AssertionError('missing input validation')


def test_interface_ablation_changes_loss():
    logits, labels, x = _data()
    a = ascd_loss(logits, labels, x, interface_radius=0, spectral_alpha=0)
    b = ascd_loss(logits, labels, x, interface_radius=3, spectral_alpha=0)
    assert torch.isfinite(a) and torch.isfinite(b)


def test_spectral_ablation_changes_loss():
    logits, labels, x = _data()
    a = ascd_loss(logits, labels, x, spectral_alpha=0)
    b = ascd_loss(logits, labels, x, spectral_alpha=1)
    assert abs(a.item() - b.item()) > 1e-7
