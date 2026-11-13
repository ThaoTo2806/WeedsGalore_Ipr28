"""Exp25: lightweight foreground residual head with occupancy/boundary supervision.

GT is used only for training targets / audit metrics. Inference requires image,
SegFormer logits, and optionally an acquisition date. No GT-based gate.
"""
import torch
from torch import nn
from torch.nn import functional as F

DATES = {'2023-05-25': 0., '2023-05-30': 5/21,
         '2023-06-06': 12/21, '2023-06-15': 1.}
ARMS = {'occ': (False,False), 'boundary': (True,False), 'stage_boundary': (True,True)}


def date_value(name):
    date=str(name)[:10]
    if date not in DATES:
        raise ValueError(f'Unrecognized acquisition date: {date}')
    return DATES[date]


def make_features(image, logits, date=0., with_date=False):
    assert image.ndim==4 and image.shape[1]==5 and logits.shape[1]==3
    assert image.shape[0]==logits.shape[0] and image.shape[-2:]==logits.shape[-2:]
    x=image.float(); z=logits.float(); p=z.softmax(1)
    zs=((z-z.mean(1,keepdim=True))/4).clamp(-8,8)
    ndvi=(x[:,3:4]-x[:,0:1])/(x[:,3:4]+x[:,0:1]+1e-4)
    ndre=(x[:,3:4]-x[:,4:5])/(x[:,3:4]+x[:,4:5]+1e-4)
    uncertainty=1-p.max(1,keepdim=True).values
    d=torch.full_like(x[:,:1],float(date) if with_date else 0.)
    return torch.cat([x,p,zs,ndvi,ndre,uncertainty,d],1) # 5+3+3+1+1+1+1=15


class ForegroundHead(nn.Module):
    """Same architecture in all arms; 15-channel feature input, 3 outputs."""
    def __init__(self, hidden=32):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(15,hidden,3,padding=1), nn.GroupNorm(8,hidden),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU())
        self.out=nn.Conv2d(hidden,3,1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()
            self.out.bias[1]=-2.0 # boundary probability low initially

    def forward(self, features):
        h=self.stem(features)
        q=self.out(h)
        return q[:,0:1],q[:,1:2],q[:,2:3]


def boundary_target(gt):
    """Binary vegetation edge, 5x5 neighborhood; shape [B,1,H,W]."""
    fg=(gt>0).float().unsqueeze(1)
    dilation=F.max_pool2d(fg,5,stride=1,padding=2)
    erosion=-F.max_pool2d(-fg,5,stride=1,padding=2)
    return (dilation-erosion).clamp(0,1)


def edit_logits(logits, delta, boundary_logit, use_boundary=False, cap=1.2):
    """Shift both vegetation logits equally, preserving Crop-vs-Weed odds.

    boundary probability modulates the magnitude, but does not access GT.
    Avoid silent background prior: zero residual always exactly reproduces base.
    """
    shift=cap*torch.tanh(delta)
    if use_boundary:
        # A soft factor in [0.5, 1] to emphasize ambiguous vegetation boundaries.
        factor=0.5+0.5*torch.sigmoid(boundary_logit)
        shift=shift*factor
    result=logits.clone()
    result[:,1:3]=result[:,1:3]+shift
    return result


def cm_from_logits(logits,gt):
    pred=logits.argmax(1)
    valid=(gt>=0)&(gt<3)
    x=gt[valid].long()*3+pred[valid].long()
    return torch.bincount(x,minlength=9).reshape(3,3).double().cpu()


def metrics(cm):
    den=cm.sum(0)+cm.sum(1)-cm.diag()
    v=100*cm.diag()/den.clamp_min(1)
    return {'mIoU':float(v.mean()),'IoU_bg':float(v[0]),
            'IoU_crop':float(v[1]),'IoU_weed':float(v[2])}


def training_loss(occ, edge, residual, base_logits, gt, use_boundary,
                  occ_weight=.25, edge_weight=.15, residual_penalty=.025):
    """Weighted classification + vegetation binary segmentation + optional edge target.

    Residual learning is supervised by multiclass GT and has explicit size penalty.
    The BCE and boundary terms do not see validation/test GT.
    """
    valid=(gt>=0)&(gt<3)
    target=(gt>0).float().unsqueeze(1)
    edge_target=boundary_target(gt)
    final=edit_logits(base_logits,residual,edge,use_boundary)
    ce=F.cross_entropy(final,gt.clamp(0,2),reduction='none')
    # FG is rare early in the season, so increase FG weight without exploding.
    weights=torch.where(gt>0,torch.full_like(ce,2.0),torch.ones_like(ce))
    weights=weights*valid.float()
    cls=(ce*weights).sum()/weights.sum().clamp_min(1)
    bce=F.binary_cross_entropy_with_logits(occ,target,reduction='none')
    fgweight=torch.where(target>0,torch.full_like(target,3.),torch.ones_like(target))
    fg_loss=(bce*fgweight*valid.unsqueeze(1)).sum()/(fgweight*valid.unsqueeze(1)).sum().clamp_min(1)
    edge_loss=(F.binary_cross_entropy_with_logits(edge,edge_target,reduction='none') *
               (1+4*edge_target)*valid.unsqueeze(1)).sum()/((1+4*edge_target)*valid.unsqueeze(1)).sum().clamp_min(1)
    penalty=residual.square().mean()
    loss=cls+occ_weight*fg_loss+residual_penalty*penalty
    if use_boundary: loss=loss+edge_weight*edge_loss
    return loss, {'ce':float(cls.detach()),'occ':float(fg_loss.detach()),
                  'edge':float(edge_loss.detach()),'residual_reg':float(penalty.detach())}
