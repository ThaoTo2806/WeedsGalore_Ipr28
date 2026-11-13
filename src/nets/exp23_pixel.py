"""Exp23 shared pixel correction network: date-agnostic vs growth-stage conditioned.

One frozen SegFormer plus a small pixel correction head. This is supervised
counterfactual action learning, NOT multi-step reinforcement learning.
"""
import torch
from torch import nn
from torch.nn import functional as F

DATE_VALUES = {'2023-05-25': 0.0, '2023-05-30': 5/21,
               '2023-06-06': 12/21, '2023-06-15': 1.0}
ACTIONS = [(0, 0.), (1, .6), (1, -.6), (2, .6), (2, -.6)]


def date_value(name):
    date = str(name)[:10]
    if date not in DATE_VALUES:
        raise ValueError(f'Unexpected date {date!r}; refusing to guess date')
    return DATE_VALUES[date]


def pixel_features(image, logits, date, use_date):
    """Return [1,16,H,W] with one optional date feature channel.
    The no-date arm receives an all-zero date channel, same architecture/cost.
    """
    assert image.ndim == logits.ndim == 4
    assert image.shape[1] == 5 and logits.shape[1] == 3
    assert image.shape[0] == logits.shape[0] == 1
    assert image.shape[-2:] == logits.shape[-2:]
    x = image.float()
    z = logits.float()
    p = z.softmax(1)
    centered = (z-z.mean(1, keepdim=True)).clamp(-8, 8)/4
    probs_sorted = p.topk(2, dim=1).values
    entropy = -(p * p.clamp_min(1.e-7).log()).sum(1,keepdim=True)/1.0986123
    margin = probs_sorted[:,0:1]-probs_sorted[:,1:2]
    ndvi=(x[:,3:4]-x[:,0:1])/(x[:,3:4]+x[:,0:1]+1e-4)
    ndre=(x[:,3:4]-x[:,4:5])/(x[:,3:4]+x[:,4:5]+1e-4)
    t = torch.full_like(x[:,0:1], float(date) if use_date else 0.0)
    return torch.cat([x,p,centered,entropy,margin,ndvi,ndre,t],1)


class PixelPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(32,5,1))
        with torch.no_grad():
            self.network[-1].bias.zero_()
            self.network[-1].bias[0]=1.5
    def forward(self, features):
        return self.network(features)


def counterfactual_targets(logits, truth, bias=.6):
    """GT TRAIN only. Return oracle action IDs, candidate mask, and changeability.

    An action is beneficial only if it fixes a previously wrong pixel. A KEEP
    label otherwise. Ties broken by ACTIONS order; no test/val labels allowed.
    """
    z=logits.float()
    gt=truth.long()
    assert gt.shape == (1,*z.shape[-2:])
    base=z.argmax(1)
    valid=(gt>=0)&(gt<3)
    err=(base!=gt)&valid
    best=torch.zeros_like(gt)
    eligible=torch.zeros_like(valid)
    fixed=torch.zeros_like(valid)
    for a,(channel,delta) in enumerate(ACTIONS[1:],1):
        changed=z.clone()
        changed[:,channel]+=delta*(bias/.6)
        pred=changed.argmax(1)
        flips=(pred!=base)&valid
        eligible |= flips
        choose=(pred==gt)&err&(~fixed)
        best[choose]=a
        fixed |= choose
    return best, eligible, fixed


def apply_policy(logits, action):
    """Exact per-pixel action, uses no labels."""
    assert logits.shape[0]==1 and logits.shape[1]==3
    assert action.shape==logits.shape[:1]+logits.shape[-2:]
    z=logits.clone()
    z[:,1]+=((action==1).float()-(action==2).float())*.6
    z[:,2]+=((action==3).float()-(action==4).float())*.6
    return z


def choose_actions(scores, eligible, minimum_confidence=.8):
    probabilities=scores.softmax(1)
    confidence, actions=probabilities.max(1)
    return torch.where(eligible & (confidence>=minimum_confidence),actions,torch.zeros_like(actions))


def confusion(pred, gt):
    mask=(gt>=0)&(gt<3)
    code=gt[mask]*3+pred[mask]
    return torch.bincount(code,minlength=9).reshape(3,3).double().cpu()


def metrics(cm):
    denom=cm.sum(0)+cm.sum(1)-cm.diag()
    per=100*cm.diag()/denom.clamp_min(1)
    return dict(mIoU=float(per.mean()),IoU_bg=float(per[0]),
                IoU_crop=float(per[1]),IoU_weed=float(per[2]))
