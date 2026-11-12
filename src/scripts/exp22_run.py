"""EXP22 standalone contextual-bandit experiment. Requires a frozen 3-class B2 LLR checkpoint.

Run train -> val. NO TEST MODE. No changes to src/train.py or original checkpoint.
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

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from datasets import WeedsGaloreDataset
from nets.segformer import SegFormer5Band
from nets.exp22_bandit import BanditPolicy, region_features, action_logits, reward_table, reinforce_loss, tta_pairs


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--base_ckpt', required=True, help='existing Exp12/Exp15-A B2 3-class EMA state_dict')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mode', required=True, choices=['cache_train', 'fit', 'val'])
    p.add_argument('--tile', type=int, default=24)
    p.add_argument('--epochs', type=int, default=80, help='lightweight policy training epochs, not backbone epochs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tta', choices=['none', 'd4'], default='d4')
    p.add_argument('--val_once', action='store_true', help='deliberate one-time val run after fitting policy')
    return p.parse_args()


def load_base(ckpt, device):
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
    # SegFormer5Band constructor uses pretrained weights; Kaggle must have transformers and model cache/network.
    model = SegFormer5Band(num_classes=3, in_channels=5, pretrained_backbone=True,
                           use_low_level_refine=True, use_multiscale_fusion=True,
                           llr_source='raw', backbone='b2')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    state = {k: v for k, v in state.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def data(ds, i, device):
    x, _, y = ds[i]
    x = torch.from_numpy(np.ascontiguousarray(x)).float()[None].to(device)
    y = torch.from_numpy(np.ascontiguousarray(y)).long()[None].to(device)
    return x, y


def split_indices(ds):
    order = sorted(range(len(ds)), key=lambda i: hashlib.sha256(str(ds.img_list[i]).encode()).hexdigest())
    return order[20:], order[:20]


def cache_train(args):
    out = Path(args.out_dir); (out/'train_cache').mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_base(args.base_ckpt, device)
    ds = WeedsGaloreDataset(args.dataset_path, None, 5, 3, False, 'train', False)
    assert len(ds) == 104, f'Expected 104 train images, got {len(ds)}'
    for i in range(len(ds)):
        path = out/'train_cache'/f'{i:03d}.pt'
        if path.exists():
            continue
        x, y = data(ds, i, device)
        with torch.no_grad():
            logits = model(x)
            features = region_features(x, logits, args.tile).cpu()
            rewards = reward_table(logits, y, args.tile).cpu()
        torch.save({'features': features, 'rewards': rewards, 'name':str(ds.img_list[i])}, path)
        if i % 10 == 0: print('cached', i+1, '/',len(ds), flush=True)
    # Checkpoint identity binds cache to correct model. No train gt exposed to inference.
    sha = hashlib.sha256(Path(args.base_ckpt).read_bytes()).hexdigest()
    (out/'cache_meta.json').write_text(json.dumps({'sha256':sha,'tile':args.tile,'n':len(ds)}, indent=2))
    print('EXP22 CACHE PASS', len(ds))


def fit(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out_dir)
    if not (out/'cache_meta.json').exists():
        raise RuntimeError('Run --mode=cache_train first')
    meta=json.loads((out/'cache_meta.json').read_text())
    assert meta['tile']==args.tile
    sha=hashlib.sha256(Path(args.base_ckpt).read_bytes()).hexdigest()
    assert sha==meta['sha256'], 'Cached features from another base checkpoint'
    ds=WeedsGaloreDataset(args.dataset_path,None,5,3,False,'train',False)
    fit_ids, hold_ids = split_indices(ds)
    samples=[]
    for i in range(len(ds)):
        p=out/'train_cache'/f'{i:03d}.pt'
        assert p.exists(), f'Missing train cache {p}'
        row=torch.load(p,map_location='cpu',weights_only=False)
        assert row['name']==str(ds.img_list[i]),'Dataset ordering changed'
        samples.append(row)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy=BanditPolicy().to(device); optim=torch.optim.Adam(policy.parameters(),lr=0.001)
    best=-1e9; history=[]
    for epoch in range(args.epochs):
        policy.train(); random.shuffle(fit_ids); losses=[]
        for i in fit_ids:
            feat=samples[i]['features'].to(device)
            rewards=samples[i]['rewards'].to(device)
            loss, _=reinforce_loss(policy(feat), rewards)
            optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 2.)
            optim.step(); losses.append(float(loss.detach().cpu()))
        policy.eval(); results=[]; kept=[]
        with torch.no_grad():
            for i in hold_ids:
                feat=samples[i]['features'].to(device)
                rewards=samples[i]['rewards'].to(device)
                acts=policy(feat).argmax(1)
                results.append(float(rewards.gather(1,acts[:,None]).mean().cpu()))
                kept.append(float((acts==0).float().mean().cpu()))
        score=float(np.mean(results))
        history.append({'epoch':epoch,'policy_holdout_mean_reward':score,
                        'policy_holdout_keep_fraction':float(np.mean(kept)),
                        'loss':float(np.mean(losses))})
        if score>best:
            best=score
            torch.save({'policy':policy.state_dict(), 'base_sha256':sha,
                        'tile':args.tile,'epoch':epoch,'holdout_reward':score,
                        'seed':args.seed},out/'policy_best.pth')
        if epoch%10==0 or epoch==args.epochs-1:
            print(f'policy epoch {epoch+1}/{args.epochs}: holdout reward={score:.5f} best={best:.5f}',flush=True)
    (out/'policy_history.json').write_text(json.dumps(history,indent=2))
    print('EXP22 FIT PASS; holdout is policy-only, base model already trained on all 104 train images')


def miou(cm):
    d=cm.sum(0)+cm.sum(1)-cm.diag()
    iou=100*cm.diag()/d.clamp_min(1)
    return {'mIoU':float(iou.mean()),'IoU_bg':float(iou[0]),
            'IoU_crop':float(iou[1]),'IoU_weed':float(iou[2])}


def confusion(pred, gt):
    valid=(gt>=0)&(gt<3)
    idx=(gt[valid]*3+pred[valid]).reshape(-1)
    return torch.bincount(idx,minlength=9).reshape(3,3).double().cpu()


def validate(args):
    if not args.val_once:
        raise SystemExit('Validation requires explicit --val_once. No test mode exists.')
    out=Path(args.out_dir)
    lock=out/'val_already_run.json'
    if lock.exists():
        raise RuntimeError('Validation already run; refusing repeat to limit selection bias')
    ck=torch.load(out/'policy_best.pth',map_location='cpu',weights_only=False)
    assert ck['tile']==args.tile
    sha=hashlib.sha256(Path(args.base_ckpt).read_bytes()).hexdigest()
    assert ck['base_sha256']==sha,'Policy incompatible with base checkpoint'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=load_base(args.base_ckpt,device)
    policy=BanditPolicy().to(device)
    policy.load_state_dict(ck['policy'],strict=True)
    policy.eval()
    ds=WeedsGaloreDataset(args.dataset_path,None,5,3,False,'val',False)
    assert len(ds)==26
    # Create reservation marker before labels are examined; do not overwrite on failures.
    lock.write_text(json.dumps({'status':'started','split':'val','seed':args.seed,'tta':args.tta}))
    base_cm=torch.zeros(3,3,dtype=torch.float64); new_cm=base_cm.clone()
    corrected=damaged=changed=0; rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            x, y=data(ds,i,device)
            raw_base=[]; raw_edit=[]; keep=[]
            for xx, undo in tta_pairs(x,args.tta):
                logits=net(xx)
                act=policy(region_features(xx,logits,args.tile)).argmax(1)
                edit=action_logits(logits,act,args.tile)
                raw_base.append(undo(logits)); raw_edit.append(undo(edit))
                keep.append(float((act==0).float().mean().cpu()))
            base_logits=torch.stack(raw_base).mean(0)
            edit_logits=torch.stack(raw_edit).mean(0)
            base=base_logits.argmax(1)
            pred=edit_logits.argmax(1)
            base_cm+=confusion(base,y); new_cm+=confusion(pred,y)
            corr=int(((base!=y)&(pred==y)).sum())
            harm=int(((base==y)&(pred!=y)).sum())
            flip=int((base!=pred).sum())
            corrected+=corr;damaged+=harm;changed+=flip
            rows.append({'name':str(ds.img_list[i]),'corrected':corr,'damaged':harm,
                         'changed':flip,'keep_fraction':float(np.mean(keep))})
    summary={'split':'val','n':len(ds),'seed':args.seed,'tta':args.tta,
             'base':miou(base_cm),'exp22':miou(new_cm),'corrected_pixels':corrected,
             'damaged_pixels':damaged,'changed_pixels':changed,
             'net_correct_pixels':corrected-damaged,
             'mean_keep_fraction':float(np.mean([r['keep_fraction'] for r in rows])),
             'base_sha256':sha,'GT_LEAKAGE':False,
             'note':'Validation was used only for a single final diagnostic, never for rewards.'}
    (out/'val_exp22_summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'val_exp22_per_image.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader();wr.writerows(rows)
    lock.write_text(json.dumps({'status':'done','summary':'val_exp22_summary.json'}))
    print(json.dumps(summary,indent=2))
    print('EXP22 VALIDATION PASS — TEST SEALED')


if __name__=='__main__':
    a=get_args()
    assert a.tile > 0 and 600%a.tile==0
    if a.mode=='cache_train':cache_train(a)
    elif a.mode=='fit':fit(a)
    else:validate(a)
