"""Exp25 frozen backbone occupancy/boundary residual correction.

Modes audit | cache | fit | val. No test access. Uses Exp23 cache if supplied,
otherwise generates own cache with Exp15-A. No GT in inference or validation
feature construction. Validation labels exclusively compute the metrics.
"""
import argparse,csv,hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt

SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from datasets import WeedsGaloreDataset
from nets.exp25_boundary import (ARMS,ForegroundHead,make_features,date_value,
    edit_logits,cm_from_logits,metrics,training_loss)
from scripts.exp22_run import load_base,data,tta_pairs


def arguments():
    p=argparse.ArgumentParser()
    for k in ('dataset_path','base_ckpt','out_dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=('cache','audit','fit','val'),required=True)
    p.add_argument('--arm',choices=tuple(ARMS),default='occ')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--patch',type=int,default=128)
    p.add_argument('--patches_per_image',type=int,default=3)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--holdout_every',type=int,default=3)
    p.add_argument('--tta',choices=('none','d4'),default='d4')
    p.add_argument('--val_once',action='store_true')
    p.add_argument('--reuse_exp23_cache',default='')
    return p.parse_args()


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dataset(path,split):
    ds=WeedsGaloreDataset(path,None,5,3,False,split,False)
    assert len(ds)==(104 if split=='train' else 26)
    return ds


def ids_split(ds):
    indices=sorted(range(len(ds)),key=lambda i: hashlib.sha256(str(ds.img_list[i]).encode()).hexdigest())
    return indices[20:],indices[:20]


def cp(out,i):return out/'cache_train'/f'{i:03d}.pt'


def cached(out,i,ds):
    d=torch.load(cp(out,i),map_location='cpu',weights_only=False)
    assert str(ds.img_list[i])==d['name'],('cache ordering mismatch',i)
    assert d['image'].shape==(5,600,600) and d['logits'].shape==(3,600,600)
    assert d['gt'].shape==(600,600)
    return d


def cache(args):
    out=Path(args.out_dir);cache_dir=out/'cache_train';cache_dir.mkdir(parents=True,exist_ok=True)
    ds=dataset(args.dataset_path,'train');base_sha=sha256(args.base_ckpt)
    meta_path=out/'cache_meta.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        assert meta['base_sha256']==base_sha and meta['n']==104
    existing=Path(args.reuse_exp23_cache) if args.reuse_exp23_cache else None
    if existing is not None:
        meta23=json.loads((existing/'cache_meta.json').read_text())
        assert meta23['base_sha256']==base_sha and meta23['n']==104
        assert str(Path(meta23['dataset_path']).resolve())==str(Path(args.dataset_path).resolve())
    todo=[i for i in range(104) if not cp(out,i).is_file()]
    net=None;dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n,i in enumerate(todo,1):
        if existing is not None:
            src=existing/'cache_train'/f'{i:03d}.pt'
            d=torch.load(src,map_location='cpu',weights_only=False)
            assert d['name']==str(ds.img_list[i]),'cache image mismatch'
            item={k:d[k] for k in ('name','image','logits','gt')}
        else:
            if net is None:net=load_base(args.base_ckpt,dev);net.eval()
            x,y=data(ds,i,dev)
            with torch.inference_mode():z=net(x).float()
            item={'name':str(ds.img_list[i]),'image':x[0].half().cpu(),
                  'logits':z[0].half().cpu(),'gt':y[0].byte().cpu()}
        torch.save(item,cp(out,i))
        if n%10==0 or n==len(todo):print('EXP25 CACHE',n,'/',len(todo),flush=True)
    for i in range(104):assert cp(out,i).is_file()
    meta={'base_sha256':base_sha,'n':104,'split':'train',
          'dataset_path':str(Path(args.dataset_path).resolve()),
          'from_exp23_cache':str(existing) if existing is not None else None}
    meta_path.write_text(json.dumps(meta,indent=2))
    print('EXP25 CACHE PASS — TRAIN ONLY / TEST SEALED')


def audit(args):
    """GT-only characterization of error distance to foreground boundary."""
    out=Path(args.out_dir);ds=dataset(args.dataset_path,'train')
    acc=defaultdict(int);rows=[]
    bins=[('0-3',0,4),('4-10',4,11),('11-25',11,26),('26+',26,float('inf'))]
    for i in range(len(ds)):
        d=cached(out,i,ds)
        gt=d['gt'].numpy().astype(np.int16)
        pr=d['logits'].float().argmax(0).numpy().astype(np.int16)
        valid=(gt>=0)&(gt<3)
        fg=gt>0
        # Distances to nearest opposite ground-truth vegetation occupancy.
        # 1 pixel at a class transition; no GT ever provided to inference head.
        to_other=np.where(fg,distance_transform_edt(fg),distance_transform_edt(~fg))
        errors=(pr!=gt)&valid
        bgveg=errors&((gt==0)^(pr==0))
        cw=errors&(gt>0)&(pr>0)
        rec={'image':d['name'],'date':d['name'][:10],
             'valid':int(valid.sum()),'errors':int(errors.sum()),
             'bgveg':int(bgveg.sum()),'crop_weed':int(cw.sum())}
        for label,low,high in bins:
            near=(to_other>=low)&(to_other<high)&valid
            n=int((bgveg&near).sum());rec['bgveg_'+label]=n
            rec['bgveg_valid_'+label]=int(near.sum())
            acc['bgveg_'+label]+=n;acc['bgveg_valid_'+label]+=int(near.sum())
        acc['errors']+=int(errors.sum());acc['bgveg']+=int(bgveg.sum());acc['cw']+=int(cw.sum())
        rows.append(rec)
        if (i+1)%20==0:print('AUDIT',i+1,'/',len(ds),flush=True)
    report={'split':'train','images':104,'error_total':acc['errors'],
            'bgveg_errors':acc['bgveg'],'crop_weed_errors':acc['cw'],
            'bins':{label:{'bgveg_errors':acc['bgveg_'+label],
                           'share_of_bgveg_pct':100*acc['bgveg_'+label]/max(acc['bgveg'],1),
                           'pixels_in_band':acc['bgveg_valid_'+label]}
                    for label,_,_ in bins},
            'note':'GT used only for error-distance audit; no validation or test accessed'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'boundary_error_audit.json').write_text(json.dumps(report,indent=2))
    with (out/'boundary_error_per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2));print('EXP25 BOUNDARY AUDIT PASS')


def prepare(d,device,with_date):
    x=d['image'].to(device).float()[None]
    z=d['logits'].to(device).float()[None]
    y=d['gt'].to(device).long()[None]
    f=make_features(x,z,date_value(d['name']),with_date)
    return f,z,y


def sample_patch(feats,z,gt,size,rng):
    H,W=gt.shape[-2:]
    assert size<=H and size<=W
    # 70% patches centered on GT vegetation boundaries; only TRAIN uses GT.
    if rng.random()<0.70:
        fg=(gt[0]>0)
        from torch.nn import functional as F
        e=F.max_pool2d(fg.float()[None,None],7,1,3)[0,0]-(-F.max_pool2d(-fg.float()[None,None],7,1,3)[0,0])
        candidates=(e>0).nonzero(as_tuple=False)
        if len(candidates):
            c=candidates[rng.randrange(len(candidates))]
            cy,cx=int(c[0]),int(c[1])
        else:cy,cx=rng.randrange(H),rng.randrange(W)
    else:cy,cx=rng.randrange(H),rng.randrange(W)
    top=max(0,min(H-size,cy-rng.randrange(size)))
    left=max(0,min(W-size,cx-rng.randrange(size)))
    sl=(slice(None),slice(None),slice(top,top+size),slice(left,left+size))
    return feats[sl],z[sl],gt[:,top:top+size,left:left+size]


@torch.no_grad()
def score_holdout(net,out,ds,ids,arm,device):
    net.eval();bcm=torch.zeros(3,3,dtype=torch.double);ncm=bcm.clone()
    correct=damage=changed=0
    boundary,stage=ARMS[arm]
    for i in ids:
        d=cached(out,i,ds)
        feat,z,y=prepare(d,device,stage)
        o,e,delta=net(feat)
        edited=edit_logits(z,delta,e,boundary)
        bcm+=cm_from_logits(z,y);ncm+=cm_from_logits(edited,y)
        p=z.argmax(1);q=edited.argmax(1)
        correct+=int(((p!=y)&(q==y)).sum())
        damage+=int(((p==y)&(q!=y)).sum())
        changed+=int((p!=q).sum())
    base=metrics(bcm);new=metrics(ncm)
    return {'base':base,'new':new,'delta_miou_pp':new['mIoU']-base['mIoU'],
            'corrected':correct,'damaged':damage,'changed':changed}


def fit(args):
    out=Path(args.out_dir);meta=json.loads((out/'cache_meta.json').read_text())
    assert meta['base_sha256']==sha256(args.base_ckpt)
    ds=dataset(args.dataset_path,'train');fit_ids,hold=ids_split(ds)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    net=ForegroundHead().to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    rng=random.Random(args.seed)
    use_boundary,use_date=ARMS[args.arm]
    arm_out=out/args.arm;arm_out.mkdir(parents=True,exist_ok=True)
    best=-float('inf');history=[]
    for epoch in range(args.epochs):
        net.train();ids=fit_ids[:];rng.shuffle(ids);losses=[]
        for i in ids:
            d=cached(out,i,ds);feat,z,gt=prepare(d,device,use_date)
            for _ in range(args.patches_per_image):
                ff,zz,yy=sample_patch(feat,z,gt,args.patch,rng)
                o,e,v=net(ff)
                loss,_=training_loss(o,e,v,zz,yy,use_boundary)
                optim.zero_grad(set_to_none=True)
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.0)
                optim.step();losses.append(float(loss.detach()))
        if (epoch+1)%args.holdout_every==0 or epoch==args.epochs-1:
            r=score_holdout(net,out,ds,hold,args.arm,device)
            info={'epoch':epoch,'train_loss':float(np.mean(losses)),**r}
            history.append(info)
            gain=r['delta_miou_pp']
            if gain>best:
                best=gain
                torch.save({'state_dict':net.state_dict(),'base_sha256':meta['base_sha256'],
                            'arm':args.arm,'seed':args.seed,'best_epoch':epoch,
                            'holdout':r},arm_out/'best.pth')
            print(f'{args.arm} epoch {epoch+1}/{args.epochs}: '
                  f'holdout delta={gain:+.5f} pp; correct={r["corrected"]}, '
                  f'damaged={r["damaged"]}; best={best:+.5f}',flush=True)
    (arm_out/'history.json').write_text(json.dumps(history,indent=2))
    print('EXP25 FIT PASS',args.arm,'TRAIN HOLDOUT ONLY, BACKBONE IN-SAMPLE')


def val(args):
    if not args.val_once:raise ValueError('Requires --val_once for sealed validation')
    out=Path(args.out_dir);lock=out/'val_once_lock.json'
    if lock.exists():raise RuntimeError('Validation already attempted; refuse repeated tuning')
    sha=sha256(args.base_ckpt)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policies={}
    for arm in ARMS:
        path=out/arm/'best.pth'
        assert path.is_file(),f'First fit all three arms: {path}'
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['base_sha256']==sha and state['arm']==arm
        p=ForegroundHead().to(device);p.load_state_dict(state['state_dict']);p.eval()
        policies[arm]=p
    # Lock before reading validation ground truth.
    out.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps({'status':'started','base_sha256':sha,'tta':args.tta}))
    ds=dataset(args.dataset_path,'val')
    base_cm=torch.zeros(3,3,dtype=torch.double)
    arm_cm={arm:base_cm.clone() for arm in ARMS}
    stats={arm:{'corrected':0,'damaged':0,'changed':0} for arm in ARMS}
    per=[];base=load_base(args.base_ckpt,device);base.eval()
    with torch.inference_mode():
        for i in range(len(ds)):
            x,y=data(ds,i,device);date=date_value(ds.img_list[i])
            sums={'base':[],**{arm:[] for arm in ARMS}}
            for xx,undo in tta_pairs(x,args.tta):
                z=base(xx).float()
                sums['base'].append(undo(z))
                for arm,policy in policies.items():
                    boundary,stage=ARMS[arm]
                    feat=make_features(xx,z,date,stage)
                    occ,edge,res=policy(feat)
                    updated=edit_logits(z,res,edge,boundary)
                    sums[arm].append(undo(updated))
            preds={arm:torch.stack(s).mean(0).argmax(1) for arm,s in sums.items()}
            base_cm+=torch.bincount((y.reshape(-1)*3+preds['base'].reshape(-1)).long(),minlength=9).reshape(3,3).double().cpu()
            record={'image':str(ds.img_list[i])}
            for arm in ARMS:
                q=preds[arm];p=preds['base']
                arm_cm[arm]+=torch.bincount((y.reshape(-1)*3+q.reshape(-1)).long(),minlength=9).reshape(3,3).double().cpu()
                c=int(((p!=y)&(q==y)).sum());d=int(((p==y)&(q!=y)).sum());n=int((p!=q).sum())
                for k,v in [('corrected',c),('damaged',d),('changed',n)]:
                    stats[arm][k]+=v;record[f'{arm}_{k}']=v
            per.append(record)
            print('EXP25 VAL',i+1,'/',len(ds),flush=True)
    baseline=metrics(base_cm)
    report={'split':'val','tta':args.tta,'n_images':26,'seed':args.seed,
            'base_sha256':sha,'base':baseline,'arms':{},'TEST_SEALED':True,
            'GT_used_only_for_validation_metrics':True}
    for arm in ARMS:
        m=metrics(arm_cm[arm]);report['arms'][arm]={**m,
            'delta_miou_pp':m['mIoU']-baseline['mIoU'],**stats[arm],
            'net_corrected':stats[arm]['corrected']-stats[arm]['damaged']}
    (out/'val_exp25_compare.json').write_text(json.dumps(report,indent=2))
    with (out/'val_exp25_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    lock.write_text(json.dumps({'status':'completed','result':'val_exp25_compare.json'}))
    print(json.dumps(report,indent=2));print('EXP25 VAL PASS — TEST SEALED')


if __name__=='__main__':
    a=arguments()
    assert a.patch>0 and a.epochs>0 and a.holdout_every>0
    {'cache':cache,'audit':audit,'fit':fit,'val':val}[a.mode](a)
