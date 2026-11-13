"""Exp27 7-arm matched ablation. Train seed0 first; test is explicitly exploratory."""
import argparse, copy, csv, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from datasets import WeedsGaloreDataset
from scripts.exp22_run import load_base, tta_pairs
from exp15_losses import cw_lovasz_loss
from exp27_losses import dice_loss
from exp27_csbrl import ARMS, csbrl_loss, teacher_reliability
from exp27_sbcl import spectral_boundary_consistency

EXPECTED_SPLIT_HASH='675667edf3f35f16'

def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset_path','base_ckpt','out_dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['preflight','fit','val','test'],required=True)
    p.add_argument('--arm',choices=ARMS,default='control')
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--grad_accum',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-6)
    p.add_argument('--sbcl_weight',type=float,default=0.02)
    p.add_argument('--attenuation',type=float,default=0.35)
    p.add_argument('--alpha',type=float,default=5.0)
    p.add_argument('--eval_arms',default=','.join(ARMS))
    p.add_argument('--test_once',action='store_true')
    p.add_argument('--allow_exploratory_test',action='store_true')
    p.add_argument('--radius',type=int,default=3)
    p.add_argument('--lovasz_weight',type=float,default=0.10)
    p.add_argument('--ema_decay',type=float,default=0.99)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--tta',choices=['none','d4'],default='d4')
    p.add_argument('--val_once',action='store_true')
    p.add_argument('--num_workers',type=int,default=2)
    return p.parse_args()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def split_fingerprint():
    p=SRC/'datasets'/'weedsgalore'/'splits'
    h=hashlib.sha256()
    for n in ('train.txt','val.txt','test.txt'):h.update((p/n).read_bytes())
    return h.hexdigest()[:16]

def check_protocol(a):
    if not Path(a.base_ckpt).is_file():raise FileNotFoundError(a.base_ckpt)
    if 'P_B2_LLR' not in Path(a.base_ckpt).parts:
        raise RuntimeError('This package is restricted to Exp12 P_B2_LLR checkpoints')
    if Path(a.base_ckpt).parent.name != f'seed{a.seed}':
        raise RuntimeError('Checkpoint seed directory does not match --seed')
    if a.epochs<1 or a.grad_accum<1 or a.batch_size<1 or a.radius<1:raise ValueError('Invalid training settings')
    if not 0<=a.sbcl_weight<=1 or not 0<=a.ema_decay<1:raise ValueError('Invalid weights/EMA')
    if not 0<a.attenuation<=1 or a.alpha<=0:raise ValueError('Invalid perturbation settings')
    fingerprint=split_fingerprint()
    if fingerprint != EXPECTED_SPLIT_HASH:
        raise RuntimeError(f'SPLIT HASH MISMATCH: {fingerprint} != {EXPECTED_SPLIT_HASH}')
    return sha(a.base_ckpt),fingerprint

def get_data(a, split):
    train=split=='train'
    ds=WeedsGaloreDataset(a.dataset_path,104 if train else None,5,3,train,split,train)
    assert len(ds)==(104 if train else 26)
    return ds

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def make_loader(a):
    ds=get_data(a,'train')
    # Date comes from original train filename, never from validation/test GT.
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, wrapped): self.wrapped=wrapped
        def __len__(self): return len(self.wrapped)
        def __getitem__(self, i):
            item=self.wrapped[i]
            name=str(self.wrapped.img_list[i])
            date=name[:10]
            stage=['2023-05-25','2023-05-30','2023-06-06','2023-06-15'].index(date)
            return item[0],item[2],stage
    def collate(items):
        x=np.stack([np.asarray(it[0],dtype=np.float32) for it in items]);y=np.stack([np.asarray(it[1],dtype=np.int64) for it in items]);
        stage=np.array([it[2] for it in items],dtype=np.int64)
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(stage)
    g=torch.Generator().manual_seed(a.seed)
    return DataLoader(IndexedDataset(ds),batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,
                      drop_last=False,pin_memory=torch.cuda.is_available(),collate_fn=collate,generator=g)

def metrics(cm):
    d=cm.sum(0)+cm.sum(1)-cm.diag()
    iou=cm.diag()/d.clamp_min(1.)*100
    return {'mIoU':float(iou.mean()),'IoU_bg':float(iou[0]),'IoU_crop':float(iou[1]),'IoU_weed':float(iou[2])}

def cm_update(cm,p,y):
    valid=(y>=0)&(y<3)
    return cm+torch.bincount((y[valid].long()*3+p[valid].long()),minlength=9).reshape(3,3).double().cpu()

def loss_parts(z,y,x,w,a,arm,teacher=None,stage=None):
    valid=(y>=0)&(y<3)
    safe=y.masked_fill(~valid, -1)
    ce=F.cross_entropy(z.float(),safe,weight=w,ignore_index=-1)
    dice=dice_loss(z,safe,w)
    lov=cw_lovasz_loss(z,safe)
    base=0.5*ce+0.5*dice+a.lovasz_weight*lov
    maps=None
    if arm in ('csbrl','stage','rgb','ms','random'):
        if teacher is None: raise RuntimeError('Frozen teacher is required')
        maps=teacher_reliability(teacher,x,alpha=a.alpha,attenuation=a.attenuation)
    spec,info=csbrl_loss(z,y,x,maps,arm,stage_ids=stage,radius=a.radius)
    return base+a.sbcl_weight*spec, {'ce':float(ce.detach()),'dice':float(dice.detach()),
        'lovasz':float(lov.detach()),'boundary':float(spec.detach()),**info}


def fit(a):
    base_sha,split_hash=check_protocol(a)
    arm_dir=Path(a.out_dir)/a.arm
    arm_dir.mkdir(parents=True,exist_ok=True)
    final=arm_dir/'final_ema.pth'
    if final.exists():raise RuntimeError('Checkpoint exists; refusing to overwrite: '+str(final))
    set_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=load_base(a.base_ckpt,device).train()
    # Freeze encoder to reduce destructive drift seen in Exp26; decoder+LLR remain trainable.
    for name,param in net.named_parameters():
        if name.startswith('encoder.') or name.startswith('mit.'):
            param.requires_grad_(False)
    teacher=None
    if a.arm in ('csbrl','stage','rgb','ms','random'):
        teacher=load_base(a.base_ckpt,device).eval()
        for tp in teacher.parameters(): tp.requires_grad_(False)
    ema=copy.deepcopy(net).eval()
    for p in ema.parameters():p.requires_grad_(False)
    loader=make_loader(a)
    w=torch.tensor([0.35,1.,1.5],device=device)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    steps_total=a.epochs*math.ceil(len(loader)/a.grad_accum)
    step=0
    history=[]
    signature={'arm':a.arm,'seed':a.seed,'base_sha256':base_sha,'split_hash':split_hash,
               'train_images':104,'epochs':a.epochs,'lr':a.lr,'batch_size':a.batch_size,
               'grad_accum':a.grad_accum,'sbcl_weight':a.sbcl_weight,
               'radius':a.radius,'lovasz_weight':a.lovasz_weight,'ema_decay':a.ema_decay,
               'attenuation':a.attenuation,'alpha':a.alpha,
               'validation_used_for_selection':False,'test_accessed':False}
    (arm_dir/'config.json').write_text(json.dumps(signature,indent=2))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    for epoch in range(a.epochs):
        net.train()
        if hasattr(net, "encoder"): net.encoder.eval()  # frozen encoder buffers must not drift
        opt.zero_grad(set_to_none=True)
        total=0.;seen=0
        tic=time.time()
        for bi,(x,y,stage) in enumerate(loader):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);stage=stage.to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                z=net(x)
                if z.shape[-2:]!=y.shape[-2:]:z=F.interpolate(z,size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss,parts=loss_parts(z,y,x,w,a,a.arm,teacher=teacher,stage=stage)
            scaler.scale(loss/a.grad_accum).backward()
            total+=float(loss.detach());seen+=1
            if (bi+1)%a.grad_accum==0 or bi+1==len(loader):
                # Keep update count identical across arms.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
                oldscale=scaler.get_scale()
                scaler.step(opt);scaler.update()
                updated=(not scaler.is_enabled()) or scaler.get_scale()>=oldscale
                opt.zero_grad(set_to_none=True)
                if updated:
                    step+=1
                    q=max(0.,step/max(steps_total,1))
                    lr=a.lr*(0.1+0.9*0.5*(1+math.cos(math.pi*q)))
                    for g in opt.param_groups:g['lr']=lr
                    with torch.no_grad():
                        for p,pe in zip(net.parameters(),ema.parameters()):
                            pe.mul_(a.ema_decay).add_(p,alpha=1-a.ema_decay)
                        # Running buffers (incl. norm state) must match.
                        for b,be in zip(net.buffers(),ema.buffers()):be.copy_(b)
        entry={'epoch':epoch+1,'train_loss':total/max(seen,1),'time_s':time.time()-tic,'optimizer_steps':step}
        history.append(entry)
        (arm_dir/'history.json').write_text(json.dumps(history,indent=2))
        print(f'EXP27 {a.arm} epoch {epoch+1}/{a.epochs}: train_loss={entry["train_loss"]:.5f}',flush=True)
    # Save EMA weights only, matching load_base's strict state_dict reader.
    torch.save({k:v.detach().cpu() for k,v in ema.state_dict().items()},final)
    (arm_dir/'finished.json').write_text(json.dumps({'checkpoint':str(final),**signature},indent=2))
    print('EXP27 FIT PASS',a.arm,'FINAL EMA:',final,'VAL/TEST NOT ACCESSED')

def evaluate(a, split):
    base_sha,split_hash=check_protocol(a)
    out=Path(a.out_dir)
    target=out/('val_exp27_csbrl_compare.json' if split=='val' else 'test_exp27_csbrl_exploratory.json')
    if target.exists():raise RuntimeError(f'Existing evaluation, refusing to rerun: {target}')
    if split=='val' and not a.val_once:raise RuntimeError('Validation requires --val_once')
    if split=='test':
        if not (a.test_once and a.allow_exploratory_test):
            raise RuntimeError('TEST exploratory only; requires --test_once --allow_exploratory_test')
        if not (out/'val_exp27_csbrl_compare.json').exists():
            raise RuntimeError('Run validation first; do not use TEST for checkpoint selection')
    arms=[v for v in a.eval_arms.split(',') if v]
    if not arms or len(arms)!=len(set(arms)) or any(v not in ARMS for v in arms):
        raise ValueError('Invalid --eval_arms')
    for arm in arms:
        cfg=json.loads((out/arm/'config.json').read_text())
        for k in ('base_sha256','split_hash','seed','epochs','lr','batch_size','grad_accum','lovasz_weight','ema_decay','sbcl_weight','attenuation','alpha'):
            expected={'base_sha256':base_sha,'split_hash':split_hash,'seed':a.seed,'epochs':a.epochs,
                      'lr':a.lr,'batch_size':a.batch_size,'grad_accum':a.grad_accum,
                      'lovasz_weight':a.lovasz_weight,'ema_decay':a.ema_decay,'sbcl_weight':a.sbcl_weight,
                      'attenuation':a.attenuation,'alpha':a.alpha}[k]
            if cfg[k]!=expected:raise RuntimeError(f'Arm {arm}: mismatched {k}')
        if not (out/arm/'final_ema.pth').is_file():raise FileNotFoundError(out/arm/'final_ema.pth')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_data(a,split)
    nets={'base':load_base(a.base_ckpt,dev)}
    for arm in arms:nets[arm]=load_base(out/arm/'final_ema.pth',dev)
    cms={k:torch.zeros((3,3),dtype=torch.float64) for k in nets}
    perimage={k:[] for k in nets}
    # Record started marker *before* reading labels; if interrupted do not reuse test as tuning set.
    started=out/(split+'_started.json')
    if started.exists():raise RuntimeError(f'Evaluation already started: {started}')
    started.write_text(json.dumps({'split':split,'seed':a.seed,'tta':a.tta,'exploratory':split=='test'}))
    for i in range(len(ds)):
        img,_,label=ds[i]
        x=torch.as_tensor(np.ascontiguousarray(img),dtype=torch.float32,device=dev)[None]
        y=torch.as_tensor(np.ascontiguousarray(label),dtype=torch.long,device=dev)[None]
        with torch.inference_mode():
            for k,net in nets.items():
                zs=[]
                for xx,undo in tta_pairs(x,a.tta):
                    zz=net(xx).float()
                    if zz.shape[-2:]!=y.shape[-2:]:zz=F.interpolate(zz,size=y.shape[-2:],mode='bilinear',align_corners=False)
                    zs.append(undo(zz))
                p=torch.stack(zs).mean(0).argmax(1)
                im_cm=cm_update(torch.zeros((3,3),dtype=torch.float64),p,y)
                cms[k]+=im_cm
                perimage[k].append((str(ds.img_list[i]),im_cm.long().numpy().tolist()))
        print('EXP27',split.upper(),i+1,'/',len(ds),flush=True)
    base=metrics(cms['base'])
    report={'split':split,'tta':a.tta,'n_images':26,'seed':a.seed,'base_sha256':base_sha,
            'base':base,'arms':{},'protocol':('exploratory_test_after_base_test_exposure' if split=='test' else 'development_validation'),
            'validation_used_for_training':False}
    for arm in arms:
        m=metrics(cms[arm]);report['arms'][arm]={**m,'delta_miou_pp':m['mIoU']-base['mIoU']}
    # Paired per-image CMs enable image-bootstrap validation confidence intervals.
    for arm, entries in perimage.items():
        csv_path=out/f'perimg_exp27_{arm}_{split}_{a.tta}_seed{a.seed}.csv'
        with csv_path.open('w',newline='',encoding='utf-8') as fh:
            wr=csv.writer(fh)
            wr.writerow(['arm','seed','split','mode','image_idx','name']+[f'cm{i}{j}' for i in range(3) for j in range(3)])
            for idx,(name,mat) in enumerate(entries):
                wr.writerow([arm,a.seed,split,a.tta,idx,name]+[int(v) for row in mat for v in row])
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('EXP27',split.upper(),'PASS')


def main():
    a=args_parser()
    if a.mode=='preflight':
        h,s=check_protocol(a)
        print('EXP27 PREFLIGHT PASS','base_sha256=',h,'split_hash=',s)
    elif a.mode=='fit':fit(a)
    elif a.mode=='val':evaluate(a,'val')
    else:evaluate(a,'test')

if __name__=='__main__':main()
