"""Exp23: two controlled pixel-correction arms on frozen Exp15-A B2.

Usage: --mode cache | fit | val, with --arm nodate | stage for fit.
Only cache reads TRAIN GT for training; val is evaluation only. No TEST mode.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from datasets import WeedsGaloreDataset
from nets.exp23_pixel import (PixelPolicy, date_value, pixel_features,
    counterfactual_targets, choose_actions, apply_policy, confusion, metrics)
from scripts.exp22_run import load_base, data, tta_pairs

ARMS={'nodate':False,'stage':True}


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_path',required=True)
    p.add_argument('--base_ckpt',required=True)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--mode',choices=('cache','fit','val'),required=True)
    p.add_argument('--arm',choices=tuple(ARMS),default='nodate')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--patch',type=int,default=96)
    p.add_argument('--patches_per_image',type=int,default=3)
    p.add_argument('--confidence',type=float,default=.80)
    p.add_argument('--tta',choices=('none','d4'),default='d4')
    p.add_argument('--val_once',action='store_true')
    p.add_argument('--holdout_every',type=int,default=5)
    return p.parse_args()


def file_sha(path):
    sha=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):
            sha.update(buf)
    return sha.hexdigest()


def train_dataset(path):
    ds=WeedsGaloreDataset(path,None,5,3,False,'train',False)
    assert len(ds)==104
    return ds


def split_indices(ds):
    ids=sorted(range(len(ds)),key=lambda i:hashlib.sha256(str(ds.img_list[i]).encode()).hexdigest())
    return ids[20:],ids[:20]


def cache_path(out,i):
    return out/'cache_train'/f'{i:03d}.pt'


def build_cache(args):
    out=Path(args.out_dir)
    cache=out/'cache_train'
    cache.mkdir(parents=True,exist_ok=True)
    meta_path=out/'cache_meta.json'
    sha=file_sha(args.base_ckpt)
    if meta_path.is_file():
        old=json.loads(meta_path.read_text())
        assert old['base_sha256']==sha, 'Cache belongs to another checkpoint'
        assert old['dataset_path']==str(Path(args.dataset_path).resolve())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=train_dataset(args.dataset_path)
    todo=[i for i in range(len(ds)) if not cache_path(out,i).is_file()]
    net=load_base(args.base_ckpt,device) if todo else None
    for j,i in enumerate(todo,1):
        x,y=data(ds,i,device)
        with torch.inference_mode():
            logits=net(x).float()
            labels,eligible,fixed=counterfactual_targets(logits,y)
        item={'name':str(ds.img_list[i]),'image':x[0].half().cpu(),
              'logits':logits[0].half().cpu(),'labels':labels[0].byte().cpu(),
              'eligible':eligible[0].byte().cpu(), 'gt':y[0].byte().cpu()}
        torch.save(item,cache_path(out,i))
        if j%10==0 or j==len(todo):
            print('cached',j,'/',len(todo), 'positive pixels:',int(fixed.sum()),flush=True)
    meta={'base_sha256':sha,'dataset_path':str(Path(args.dataset_path).resolve()),
          'n':len(ds),'notes':'TRAIN ONLY; images, frozen logits, oracle edit labels. No validation/test.'}
    meta_path.write_text(json.dumps(meta,indent=2))
    for i in range(len(ds)):
        assert cache_path(out,i).is_file()
    print('EXP23 CACHE PASS:',len(ds),'TRAIN images. NO VAL/TEST.')


def load_item(out,i,ds):
    d=torch.load(cache_path(out,i),map_location='cpu',weights_only=False)
    assert str(ds.img_list[i])==d['name'],f'cache ordering mismatch {i}'
    assert d['image'].shape==(5,600,600)
    return d


def prepare(d,device,use_date):
    x=d['image'].to(device).float()[None]
    z=d['logits'].to(device).float()[None]
    features=pixel_features(x,z,date_value(d['name']),use_date)
    labels=d['labels'].to(device).long()[None]
    eligible=d['eligible'].to(device).bool()[None]
    return features,labels,eligible,z


def crop_sample(features,labels,eligible,patch, rng):
    _,_,h,w=features.shape
    assert h>=patch and w>=patch
    pos=(labels[0]>0).nonzero(as_tuple=False)
    cand=eligible[0].nonzero(as_tuple=False)
    if len(pos)>0 and rng.random()<.65:
        at=pos[int(rng.randrange(len(pos)))]; yc,xc=map(int,at)
    elif len(cand)>0 and rng.random()<.8:
        at=cand[int(rng.randrange(len(cand)))]; yc,xc=map(int,at)
    else:
        yc,xc=rng.randrange(h),rng.randrange(w)
    top=max(0,min(h-patch,yc-rng.randrange(patch)))
    left=max(0,min(w-patch,xc-rng.randrange(patch)))
    return (features[:,:,top:top+patch,left:left+patch],
            labels[:,top:top+patch,left:left+patch],
            eligible[:,top:top+patch,left:left+patch])


def supervised_loss(scores,labels,eligible):
    # Train on pixels where at least one +/-0.6 action can alter base prediction.
    # KEEP remains the majority; positive correction pixels receive extra weight.
    ce=F.cross_entropy(scores,labels,reduction='none')
    weights=torch.where(labels>0,torch.full_like(ce,8.),torch.ones_like(ce))
    mask=eligible.float()
    if not mask.any():
        return scores.sum()*0
    return (ce*mask*weights).sum()/(mask*weights).sum().clamp_min(1)


@torch.no_grad()
def score_holdout(policy,out,ds,ids,use_date,device,confidence):
    policy.eval()
    corrected=damaged=changed=0
    cm0=torch.zeros(3,3,dtype=torch.float64)
    cm1=torch.zeros_like(cm0)
    for i in ids:
        d=load_item(out,i,ds)
        feat,_,eligible,z=prepare(d,device,use_date)
        choice=choose_actions(policy(feat),eligible,confidence)
        gt=d['gt'].to(device).long()[None]
        base=z.argmax(1)
        edited=apply_policy(z,choice).argmax(1)
        cm0+=confusion(base,gt);cm1+=confusion(edited,gt)
        corrected+=int(((base!=gt)&(edited==gt)).sum())
        damaged+=int(((base==gt)&(edited!=gt)).sum())
        changed+=int((base!=edited).sum())
    original=metrics(cm0); updated=metrics(cm1)
    return {'base':original,'new':updated,'delta_miou_pp':updated['mIoU']-original['mIoU'],
            'corrected':corrected,'damaged':damaged,'changed':changed,
            'net_corrected':corrected-damaged}


def fit(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.out_dir)
    meta=json.loads((out/'cache_meta.json').read_text())
    sha=file_sha(args.base_ckpt)
    assert sha==meta['base_sha256'],'Base checkpoint changed after caching'
    ds=train_dataset(args.dataset_path)
    fit_ids,hold_ids=split_indices(ds)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy=PixelPolicy().to(device)
    optim=torch.optim.AdamW(policy.parameters(),lr=0.0005,weight_decay=1e-4)
    rng=random.Random(args.seed)
    arm_dir=out/args.arm
    arm_dir.mkdir(parents=True,exist_ok=True)
    history=[];best=float('-inf')
    for epoch in range(args.epochs):
        policy.train();ids=fit_ids[:];rng.shuffle(ids);losses=[]
        for i in ids:
            d=load_item(out,i,ds)
            feats,labels,eligible,_=prepare(d,device,ARMS[args.arm])
            for _ in range(args.patches_per_image):
                a,b,c=crop_sample(feats,labels,eligible,args.patch,rng)
                if not bool(c.any()):
                    continue
                output=policy(a)
                loss=supervised_loss(output,b,c)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),2)
                optim.step()
                losses.append(float(loss.detach().cpu()))
        if (epoch+1)%args.holdout_every==0 or epoch==args.epochs-1:
            hold=score_holdout(policy,out,ds,hold_ids,ARMS[args.arm],device,args.confidence)
            item={'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,**hold}
            history.append(item)
            key=hold['delta_miou_pp']
            if key>best:
                best=key
                torch.save({'state_dict':policy.state_dict(),'base_sha256':sha,
                            'arm':args.arm,'seed':args.seed,'confidence':args.confidence,
                            'best_epoch':epoch,'holdout':hold},arm_dir/'policy_best.pth')
            print(f'{args.arm} epoch {epoch+1}/{args.epochs}: holdout delta={key:+.4f} pp; '
                  f'corrected={hold["corrected"]}, damaged={hold["damaged"]}; best={best:+.4f}',flush=True)
    (arm_dir/'policy_history.json').write_text(json.dumps(history,indent=2))
    print(f'EXP23 FIT PASS {args.arm} — holdout is policy-only; base trained on 104.')


def evaluate_val(args):
    if not args.val_once:
        raise ValueError('Validation requires --val_once')
    out=Path(args.out_dir)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sha=file_sha(args.base_ckpt)
    ds=WeedsGaloreDataset(args.dataset_path,None,5,3,False,'val',False)
    assert len(ds)==26
    policies={}
    for arm in ARMS:
        ckpt=out/arm/'policy_best.pth'
        assert ckpt.is_file(),f'Missing {ckpt}; train BOTH arms first'
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert saved['base_sha256']==sha
        assert saved['arm']==arm
        assert abs(saved['confidence']-args.confidence)<1e-8
        p=PixelPolicy().to(device)
        p.load_state_dict(saved['state_dict'],strict=True)
        policies[arm]=p.eval()
    lock=out/'val_once_lock.json'
    if lock.exists():
        raise RuntimeError('Validation already attempted. Refusing to inspect validation repeatedly.')
    lock.write_text(json.dumps({'status':'started','tta':args.tta,'seed':args.seed,'base_sha256':sha}))
    net=load_base(args.base_ckpt,device)
    cms={'base':torch.zeros(3,3,dtype=torch.float64),
         **{a:torch.zeros(3,3,dtype=torch.float64) for a in ARMS}}
    stats={a:{'corrected':0,'damaged':0,'changed':0} for a in ARMS}
    rows=[]
    with torch.inference_mode():
        for i in range(len(ds)):
            x,y=data(ds,i,device)
            date=date_value(ds.img_list[i])
            logits_per={'base':[],'nodate':[],'stage':[]}
            for xx,undo in tta_pairs(x,args.tta):
                z=net(xx).float()
                logits_per['base'].append(undo(z))
                _,eligible,_=counterfactual_targets_without_gt(z)
                for arm in ARMS:
                    feat=pixel_features(xx,z,date,ARMS[arm])
                    act=choose_actions(policies[arm](feat),eligible,args.confidence)
                    logits_per[arm].append(undo(apply_policy(z,act)))
            preds={k:torch.stack(v).mean(0).argmax(1) for k,v in logits_per.items()}
            for arm in cms:
                cms[arm]+=confusion(preds[arm],y)
            item={'image':str(ds.img_list[i])}
            for arm in ARMS:
                corrected=int(((preds['base']!=y)&(preds[arm]==y)).sum())
                damaged=int(((preds['base']==y)&(preds[arm]!=y)).sum())
                changed=int((preds[arm]!=preds['base']).sum())
                for k,n in [('corrected',corrected),('damaged',damaged),('changed',changed)]:
                    stats[arm][k]+=n
                    item[f'{arm}_{k}']=n
            rows.append(item)
            print('evaluated val',i+1,'/',len(ds),flush=True)
    result={'split':'val','tta':args.tta,'n_images':26,'seed':args.seed,
            'base_sha256':sha,'base':metrics(cms['base']),
            'policy_arm_comparison':{},'GT_used_only_for_metrics':True,'TEST_SEALED':True}
    for arm in ARMS:
        result['policy_arm_comparison'][arm]={
            **metrics(cms[arm]),'delta_miou_pp':metrics(cms[arm])['mIoU']-result['base']['mIoU'],
            **stats[arm],'net_corrected':stats[arm]['corrected']-stats[arm]['damaged']}
    (out/'val_exp23_compare.json').write_text(json.dumps(result,indent=2))
    with (out/'val_exp23_per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lock.write_text(json.dumps({'status':'done','summary':'val_exp23_compare.json'}))
    print(json.dumps(result,indent=2))
    print('EXP23 TWO-ARM VALIDATION PASS — TEST SEALED')


def counterfactual_targets_without_gt(z):
    # At inference find only changeable pixels, no label access.
    base=z.argmax(1)
    eligible=torch.zeros_like(base,dtype=torch.bool)
    for channel,delta in ((1,.6),(1,-.6),(2,.6),(2,-.6)):
        changed=z.clone()
        changed[:,channel]+=delta
        eligible |= changed.argmax(1)!=base
    return None,eligible,None


if __name__=='__main__':
    args=arguments()
    assert args.epochs>0 and args.patch>0 and args.patches_per_image>0
    assert 0.2<=args.confidence<=1.0
    if args.mode=='cache': build_cache(args)
    elif args.mode=='fit': fit(args)
    else: evaluate_val(args)
