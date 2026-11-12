"""EXP22: one-step contextual bandit region policy for crop/weed logit edits.

Action 0 keep; 1/2 add +/-bias to crop; 3/4 add +/-bias to weed.
Only model predictions and raw 5-band image are seen by policy at inference.
Targets are used exclusively to build train reward tables.
"""
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ((-1, 0.), (1, .6), (1, -.6), (2, .6), (2, -.6))
FEATURE_DIM = 14


def region_features(image, logits, tile=24):
    """Input image [1,5,H,W], logits [1,3,H,W]; output [N,14] with tile order row-major.
    Images must be divisible by tile; 600 is divisible by 24.
    """
    assert image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 5
    assert logits.shape[:2] == (1, 3) and image.shape[-2:] == logits.shape[-2:]
    h, w = image.shape[-2:]
    assert h % tile == w % tile == 0
    prob = logits.float().softmax(1)
    top = prob.topk(2, dim=1).values
    uncertainty = 1 - (top[:, 0:1] - top[:, 1:2])
    v = torch.cat([image.float(), prob, uncertainty], 1)
    mu = F.avg_pool2d(v, tile, tile)
    std = (F.avg_pool2d(v.square(), tile, tile) - mu.square()).clamp_min(0).sqrt()
    # 5 band means + 3 class probabilities + 1 uncertainty + 5 band stds = 14
    f = torch.cat([mu[:, :9], std[:, :5]], 1)
    return f.permute(0, 2, 3, 1).reshape(-1, FEATURE_DIM)


def action_logits(logits, action, tile=24):
    """Apply per-tile edits to full-resolution logits. One image only."""
    assert logits.shape[0] == 1 and logits.shape[1] == 3
    h, w = logits.shape[-2:]
    assert h % tile == w % tile == 0
    assert action.numel() == (h // tile) * (w // tile)
    bias = torch.tensor([a[1] for a in ACTIONS], dtype=logits.dtype, device=logits.device)
    target = torch.tensor([a[0] for a in ACTIONS], dtype=torch.long, device=logits.device)
    a = action.to(logits.device).long()
    spatial = bias[a].reshape(1, 1, h // tile, w // tile).repeat_interleave(tile, -2).repeat_interleave(tile, -1)
    channel = target[a].reshape(1, 1, h // tile, w // tile).repeat_interleave(tile, -2).repeat_interleave(tile, -1)
    delta = torch.zeros_like(logits)
    for c in (1, 2):
        delta[:, c:c+1] = spatial * (channel == c)
    return logits + delta


def reward_table(logits, target, tile=24, damage=1.5, action_cost=0.015):
    """Per-tile counterfactual reward for 5 actions on TRAIN ground truth only.
    Reward = corrected pixels - damage * newly incorrect pixels, per tile pixel;
    then penalize non-keep actions. No validation/test GT allowed here.
    """
    assert target.shape == (1, *logits.shape[-2:])
    baseline = logits.argmax(1)
    ok_base = baseline == target
    h, w = target.shape[-2:]
    scores = []
    for idx in range(len(ACTIONS)):
        act = torch.full(((h//tile)*(w//tile),), idx, dtype=torch.long, device=logits.device)
        pred = action_logits(logits, act, tile).argmax(1)
        corrected = (~ok_base) & (pred == target)
        damaged = ok_base & (pred != target)
        gain = corrected.float() - damage * damaged.float()
        r = F.avg_pool2d(gain.unsqueeze(1), tile, tile).flatten()
        if idx:
            r = r - action_cost
        scores.append(r)
    return torch.stack(scores, 1)


class BanditPolicy(nn.Module):
    def __init__(self, features=FEATURE_DIM, width=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, len(ACTIONS)))
        # Prefer no-op from start; other actions must earn reward.
        with torch.no_grad():
            self.net[-1].bias.fill_(-1.0)
            self.net[-1].bias[0] = 1.0

    def forward(self, features):
        return self.net(features)


def reinforce_loss(logits, rewards, entropy_weight=0.002):
    """Sample one action per region, REINFORCE with action-independent baseline."""
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    observed = rewards.gather(1, actions[:, None]).squeeze(1)
    baseline = rewards.mean(1)  # action-independent counterfactual baseline, TRAIN ONLY
    loss = -(dist.log_prob(actions) * (observed - baseline).detach()).mean()
    loss -= entropy_weight * dist.entropy().mean()
    return loss, observed.mean().detach()


def tta_pairs(x, mode):
    """Generate orientation and inverse transform pairs, independently testable."""
    if mode == 'none':
        return [(x, lambda z: z)]
    if mode != 'd4':
        raise ValueError('mode must be none or d4')
    pairs = []
    for k in range(4):
        for flip in (False, True):
            v = torch.rot90(x, k, (-2, -1))
            if flip:
                v = torch.flip(v, (-1,))
            def inverse(z, k=k, flip=flip):
                if flip:
                    z = torch.flip(z, (-1,))
                return torch.rot90(z, -k, (-2, -1))
            pairs.append((v, inverse))
    return pairs
