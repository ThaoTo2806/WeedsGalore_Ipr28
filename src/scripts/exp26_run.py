"""Exp26 two-arm matched fine tuning. Kaggle: train-only until explicit val call."""
import argparse, copy, hashlib, json, math, random, sys, time
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
from exp26_losses import boundary_ce, dice_loss

ARMS=('control','boundary')
EXPECTED_SPLIT_HASH='675667edf3f35f16'

def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset_path','base_ckpt','out_dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['preflight','fit','val'],required=True)
    p.add_argument('--arm',choices=ARMS,default='control')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--grad_accum',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--boundary_weight',type=float,default=0.10)
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
    if a.epochs<1 or a.grad_accum<1 or a.batch_size<1 or a.radius<1:raise ValueError('Invalid training settings')
    if not 0<=a.boundary_weight<=1 or not 0<=a.ema_decay<1:raise ValueError('Invalid weights/EMA')
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
    def collate(items):
        x=np.stack([np.asarray(it[0],dtype=np.float32) for it in items]);y=np.stack([np.asarray(it[2],dtype=np.int64) for it in items]);
        return torch.from_numpy(x),torch.from_numpy(y)
    g=torch.Generator().manual_seed(a.seed)
    return DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,
                      drop_last=False,pin_memory=torch.cuda.is_available(),collate_fn=collate,generator=g)

def metrics(cm):
    d=cm.sum(0)+cm.sum(1)-cm.diag()
    iou=cm.diag()/d.clamp_min(1.)*100
    return {'mIoU':float(iou.mean()),'IoU_bg':float(iou[0]),'IoU_crop':float(iou[1]),'IoU_weed':float(iou[2])}

def cm_update(cm,p,y):
    valid=(y>=0)&(y<3)
    return cm+torch.bincount((y[valid].long()*3+p[valid].long()),minlength=9).reshape(3,3).double().cpu()

def loss_parts(z,y,w,a,arm):
    valid=(y>=0)&(y<3)
    safe=y.masked_fill(~valid, -1)
    ce=F.cross_entropy(z.float(),safe,weight=w,ignore_index=-1)
    dice=dice_loss(z,safe,w)
    lov=cw_lovasz_loss(z,safe)
    base=0.5*ce+0.5*dice +a.lovasz_weight*lov
    edge=boundary_ce(z,safe,w,a.radius) if arm=='boundary' else z.sum()*0
    return base+a.boundary_weight*edge if arm=='boundary' else base, \
        {'ce':float(ce.detach()),'dice':float(dice.detach()),'lovasz':float(lov.detach()),'edge':float(edge.detach())}

def fit(a):
    base_sha,split_hash=check_protocol(a)
    arm_dir=Path(a.out_dir)/a.arm
    arm_dir.mkdir(parents=True,exist_ok=True)
    final=arm_dir/'final_ema.pth'
    if final.exists():raise RuntimeError('Checkpoint exists; refusing to overwrite: '+str(final))
    set_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=load_base(a.base_ckpt,device).train()
    ema=copy.deepcopy(net).eval()
    for p in ema.parameters():p.requires_grad_(False)
    loader=make_loader(a)
    w=torch.tensor([0.35,1.,1.5],device=device)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    steps_total=a.epochs*math.ceil(len(loader)/a.grad_accum)
    step=0
    history=[]
    signature={'arm':a.arm,'seed':a.seed,'base_sha256':base_sha,'split_hash':split_hash,
               'train_images':104,'epochs':a.epochs,'lr':a.lr,'batch_size':a.batch_size,
               'grad_accum':a.grad_accum,'boundary_weight':a.boundary_weight,
               'radius':a.radius,'lovasz_weight':a.lovasz_weight,'ema_decay':a.ema_decay,
               'validation_used_for_selection':False,'test_accessed':False}
    (arm_dir/'config.json').write_text(json.dumps(signature,indent=2))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    for epoch in range(a.epochs):
        net.train();opt.zero_grad(set_to_none=True)
        total=0.;seen=0
        tic=time.time()
        for bi,(x,y) in enumerate(loader):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                z=net(x)
                if z.shape[-2:]!=y.shape[-2:]:z=F.interpolate(z,size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss,parts=loss_parts(z,y,w,a,a.arm)
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
        print(f'EXP26 {a.arm} epoch {epoch+1}/{a.epochs}: train_loss={entry["train_loss"]:.5f}',flush=True)
    # Save EMA weights only, matching load_base's strict state_dict reader.
    torch.save({k:v.detach().cpu() for k,v in ema.state_dict().items()},final)
    (arm_dir/'finished.json').write_text(json.dumps({'checkpoint':str(final),**signature},indent=2))
    print('EXP26 FIT PASS',a.arm,'FINAL EMA:',final,'VAL/TEST NOT ACCESSED')

def validate(a):
    base_sha,split_hash=check_protocol(a)
    out=Path(a.out_dir);target=out/'val_exp26_compare.json'
    if target.exists():raise RuntimeError('Validation file exists; refusing to rerun: '+str(target))
    if not a.val_once:raise RuntimeError('Validation requires --val_once')
    for arm in ARMS:
        cfg=json.loads((out/arm/'config.json').read_text())
        for k in ('base_sha256','split_hash','seed','epochs','lr','batch_size','grad_accum','lovasz_weight','ema_decay'):
            expected={'base_sha256':base_sha,'split_hash':split_hash,'seed':a.seed,'epochs':a.epochs,'lr':a.lr,'batch_size':a.batch_size,'grad_accum':a.grad_accum,'lovasz_weight':a.lovasz_weight,'ema_decay':a.ema_decay}[k]
            if cfg[k]!=expected:raise RuntimeError(f'Arm {arm}: mismatched {k}: {cfg[k]} != {expected}')
        if not (out/arm/'final_ema.pth').is_file():raise FileNotFoundError(out/arm/'final_ema.pth')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_data(a,'val')
    nets={'base':load_base(a.base_ckpt,dev)}
    for arm in ARMS:nets[arm]=load_base(out/arm/'final_ema.pth',dev)
    cms={k:torch.zeros((3,3),dtype=torch.float64) for k in nets}
    per=[]
    for i in range(len(ds)):
        img,_,label=ds[i]
        x=torch.as_tensor(np.ascontiguousarray(img),dtype=torch.float32,device=dev)[None]
        y=torch.as_tensor(np.ascontiguousarray(label),dtype=torch.long,device=dev)[None]
        row={'image':str(ds.img_list[i])}
        with torch.inference_mode():
            for k,net in nets.items():
                zs=[]
                for xx,undo in tta_pairs(x,a.tta):
                    zz=net(xx).float()
                    if zz.shape[-2:]!=y.shape[-2:]:zz=F.interpolate(zz,size=y.shape[-2:],mode='bilinear',align_corners=False)
                    zs.append(undo(zz))
                p=torch.stack(zs).mean(0).argmax(1)
                cms[k]=cm_update(cms[k],p,y)
                row[k+'_correct']=int((p==y).sum())
        per.append(row)
        print('EXP26 VAL',i+1,'/',len(ds),flush=True)
    base=metrics(cms['base'])
    report={'split':'val','tta':a.tta,'n_images':26,'seed':a.seed,'base_sha256':base_sha,
            'base':base,'arms':{},'test_sealed':True}
    for arm in ARMS:
        m=metrics(cms[arm]);report['arms'][arm]={**m,'delta_miou_pp':m['mIoU']-base['mIoU']}
    out.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('EXP26 VAL PASS — TEST SEALED')

def main():
    a=args_parser()
    if a.mode=='preflight':
        h,s=check_protocol(a)
        print('EXP26 PREFLIGHT PASS','base_sha256=',h,'split_hash=',s)
    elif a.mode=='fit':fit(a)
    else:validate(a)

if __name__=='__main__':main()
