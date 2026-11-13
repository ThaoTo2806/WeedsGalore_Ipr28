"""WeedMap RedEdge (5 bands) independent-scene external protocol.

Uses same frozen pretrained WeedsGalore MiT-B2 checkpoint as initialization and
compares matched control/SBCL/CSBRL/random. This is EXTERNAL DOMAIN
ADAPTATION (retrained on WeedMap 000/001); it is not zero-shot transfer.

Custom scene split: train 000/001, val 002, external test 003. Do NOT
claim it reproduces WeedMap's official published split or published SOTA.
"""
import argparse, copy, hashlib, json, random, re, sys, math, csv, gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from scripts.exp22_run import load_base, tta_pairs
from exp27_csbrl import csbrl_loss, teacher_reliability
from exp27_losses import dice_loss
from exp15_losses import cw_lovasz_loss

BANDS=('R','G','B','NIR','RE')
VALID_ARMS=('control','sbcl','csbrl','random','rgb','ms')
SCENE_SPLITS={'train':['000','001'],'val':['002'],'test':['003']}


def hashfile(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for ch in iter(lambda:f.read(1<<20),b''):h.update(ch)
    return h.hexdigest()


def frame_key(filename):
    m=re.search(r'frame(\d{4})',filename,flags=re.IGNORECASE)
    return m.group(1) if m else None


def candidate_map(folder):
    result={}
    for f in sorted(folder.rglob('*')):
        if f.suffix.lower() not in ('.png','.tif','.tiff'):continue
        key=frame_key(f.name)
        if key is None:continue
        if key in result:raise RuntimeError(f'Duplicate frame {key} in {folder}: {result[key]} and {f}')
        result[key]=f
    return result


class WeedMapRedEdge(Dataset):
    def __init__(self,root,scenes,resize=(256,256),augment=False,max_images=0):
        self.root=Path(root)
        self.resize=tuple(resize)
        self.augment=augment
        self.samples=[]
        for scene in scenes:
            base=self.root/'Tiles'/'RedEdge'/scene
            if not base.is_dir():raise FileNotFoundError(f'WeedMap missing RedEdge scene: {base}')
            gt_dir=base/'groundtruth'
            if not gt_dir.is_dir():raise FileNotFoundError(gt_dir)
            gts={}
            for f in sorted(gt_dir.rglob('*GroundTruth_iMap.png')):
                key=frame_key(f.name)
                if key in gts:raise RuntimeError(f'Duplicate GT in scene {scene}, frame{key}')
                gts[key]=f
            if not gts:raise RuntimeError(f'No indexed GroundTruth_iMap.png in {gt_dir}')
            bandmaps={band:candidate_map(base/'tile'/band) for band in BANDS}
            maskmap=candidate_map(base/'mask') if (base/'mask').is_dir() else {}
            for key,gt_path in gts.items():
                if not all(key in bandmaps[b] for b in BANDS):continue
                paths=[bandmaps[b][key] for b in BANDS]
                self.samples.append((f'{scene}_frame{key}',paths,gt_path,maskmap.get(key)))
        self.samples.sort(key=lambda a:a[0])
        if max_images:self.samples=self.samples[:max_images]
        if not self.samples:raise RuntimeError('No complete aligned 5-band samples found')
        names=[x[0] for x in self.samples]
        if len(set(names))!=len(names):raise RuntimeError('Duplicated sample keys')

    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        name,paths,gt_path,validmask=self.samples[i]
        raw=[]
        for p in paths:
            im=np.asarray(Image.open(p))
            if im.ndim==3:
                if im.shape[2]!=1:raise RuntimeError(f'Expected single band tile, got {im.shape} for {p}')
                im=im[:,:,0]
            if im.dtype==np.uint8:div=255.0
            elif im.dtype==np.uint16:div=65535.0
            else:
                raise RuntimeError(f'Unexpected channel dtype {im.dtype} in {p}; define radiometric calibration explicitly')
            raw.append(im.astype('float32')/div)
        shapes=[im.shape for im in raw]
        gt=np.array(Image.open(gt_path)).astype(np.int64)
        if len(set(shapes+[gt.shape]))!=1:raise RuntimeError(f'Misaligned image/mask for {name}: {shapes}, {gt.shape}')
        # Indexed GT exactly 0=soil, 1=crop, 2=weed, 10000=ignore.
        valid=np.isin(gt,[0,1,2])
        if validmask is not None:
            vm=np.asarray(Image.open(validmask))
            if vm.ndim==3:vm=vm[:,:,0]
            if vm.shape!=gt.shape:raise RuntimeError(f'Mask shape mismatch for {name}')
            valid &= vm>0
        gt=np.where(valid,gt,-1)
        x=torch.from_numpy(np.stack(raw))
        y=torch.from_numpy(gt).long()
        if self.resize:
            x=F.interpolate(x[None],size=self.resize,mode='bilinear',align_corners=False)[0]
            y=F.interpolate(y[None,None].float(),size=self.resize,mode='nearest')[0,0].long()
        if self.augment:
            if random.random()<.5:x=x.flip(-1);y=y.flip(-1)
            if random.random()<.5:x=x.flip(-2);y=y.flip(-2)
        return x.contiguous(),y.contiguous(),name


def metric(cm):
    tp=cm.diag();den=cm.sum(0)+cm.sum(1)-tp
    iou=tp/den.clamp_min(1)*100
    return {'mIoU':float(iou.mean()),'IoU_bg':float(iou[0]),'IoU_crop':float(iou[1]),'IoU_weed':float(iou[2])}


def add_cm(cm,p,y):
    v=(y>=0)&(y<3)
    return cm+torch.bincount(3*y[v].long()+p[v].long(),minlength=9).reshape(3,3).cpu().double()


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weedmap_root',required=True,help='Directory containing Tiles/RedEdge/...')
    p.add_argument('--base_ckpt',required=True)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--mode',choices=['preflight','fit','val','test'],required=True)
    p.add_argument('--arm',choices=VALID_ARMS,default='csbrl')
    p.add_argument('--eval_arms',default='control,sbcl,csbrl,random')
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--lr',type=float,default=2e-6)
    p.add_argument('--sbcl_weight',type=float,default=.02)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resize',type=int,default=256)
    p.add_argument('--max_train',type=int,default=0,help='Smoke only; cannot use for paper claims')
    p.add_argument('--allow_external_test',action='store_true')
    p.add_argument('--tta',choices=['none','d4'],default='d4')
    return p.parse_args()


def get_ds(a,split,augment=False):
    return WeedMapRedEdge(a.weedmap_root,SCENE_SPLITS[split],resize=(a.resize,a.resize),augment=augment,
                          max_images=a.max_train if split=='train' else 0)


def run_fit(a):
    out=Path(a.out_dir);adir=out/a.arm;adir.mkdir(parents=True,exist_ok=True)
    saved=adir/'final_ema.pth'
    if saved.exists():raise RuntimeError(f'Refusing checkpoint overwrite: {saved}')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    ds=get_ds(a,'train',augment=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=load_base(a.base_ckpt,dev).train()
    for name,p in net.named_parameters():
        if name.startswith(('encoder.','mit.')):p.requires_grad_(False)
    if hasattr(net,'encoder'):net.encoder.eval()
    teacher=None
    if a.arm in ('csbrl','random','rgb','ms'):
        teacher=load_base(a.base_ckpt,dev).eval()
        for p in teacher.parameters():p.requires_grad_(False)
    ema=copy.deepcopy(net).eval()
    for p in ema.parameters():p.requires_grad_(False)
    loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=0)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    w=torch.tensor([.35,1.,1.5],device=dev)
    history=[]
    for epoch in range(a.epochs):
        net.train()
        if hasattr(net,'encoder'):net.encoder.eval()
        tot=0.
        for x,y,_ in loader:
            x=x.to(dev);y=y.to(dev)
            opt.zero_grad(set_to_none=True)
            z=net(x).float()
            if z.shape[-2:]!=y.shape[-2:]:z=F.interpolate(z,size=y.shape[-2:],mode='bilinear',align_corners=False)
            ce=F.cross_entropy(z,y,weight=w,ignore_index=-1)
            dice=dice_loss(z,y,w)
            lov=cw_lovasz_loss(z,y)
            maps=teacher_reliability(teacher,x) if teacher is not None else None
            b_loss,_=csbrl_loss(z,y,x,maps,a.arm,radius=3)
            loss=.5*ce+.5*dice+.1*lov+a.sbcl_weight*b_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1)
            opt.step()
            with torch.no_grad():
                for p,pe in zip(net.parameters(),ema.parameters()):pe.mul_(.99).add_(p,alpha=.01)
                for b,be in zip(net.buffers(),ema.buffers()):be.copy_(b)
            tot+=float(loss.detach())
        history.append({'epoch':epoch+1,'train_loss':tot/len(loader)})
        print(f'WEEDMAP {a.arm} epoch {epoch+1}/{a.epochs}: loss={history[-1]["train_loss"]:.5f}',flush=True)
    torch.save({k:v.detach().cpu() for k,v in ema.state_dict().items()},saved)
    (adir/'config.json').write_text(json.dumps({'split':SCENE_SPLITS,'arm':a.arm,'seed':a.seed,
        'source_checkpoint_sha256':hashfile(a.base_ckpt),'epochs':a.epochs,'resize':a.resize,
        'max_train':a.max_train,'domain':'WeedMap RedEdge, custom scene split, domain adaptation'},indent=2))
    (adir/'history.json').write_text(json.dumps(history,indent=2))
    print('WEEDMAP FIT PASS',saved)


@torch.inference_mode()
def predict(model,x,tta):
    logits=None;n=0
    for xt,undo in tta_pairs(x,tta):
        z=undo(model(xt).float())
        logits=z if logits is None else logits+z;n+=1
    return (logits/n).argmax(1)


def run_eval(a,split):
    out=Path(a.out_dir);target=out/f'weedmap_{split}_report.json'
    if target.exists():raise RuntimeError(f'Refusing re-evaluation: {target}')
    if split=='test' and not a.allow_external_test:raise RuntimeError('Need --allow_external_test')
    if split=='test' and not (out/'weedmap_val_report.json').exists():
        raise RuntimeError('Do external validation first and freeze configuration')
    arms=[s for s in a.eval_arms.split(',') if s]
    if any(s not in VALID_ARMS for s in arms):raise ValueError('Invalid arm')
    ds=get_ds(a,split)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets={'base':load_base(a.base_ckpt,dev).eval()}
    for arm in arms:
        cfg=json.loads((out/arm/'config.json').read_text())
        if cfg['source_checkpoint_sha256']!=hashfile(a.base_ckpt) or cfg['max_train']:
            raise RuntimeError(f'Invalid full-data/exact-base configuration for {arm}')
        nets[arm]=load_base(out/arm/'final_ema.pth',dev).eval()
    all_cms={k:torch.zeros(3,3,dtype=torch.float64) for k in nets}
    entries={k:[] for k in nets}
    # Marker BEFORE opening TEST labels.
    marker=out/f'weedmap_{split}_started.json'
    if marker.exists():raise RuntimeError(f'Evaluation already attempted: {marker}')
    marker.write_text(json.dumps({'split':split,'arms':list(nets),'source_sha256':hashfile(a.base_ckpt)}))
    for i,(x,y,name) in enumerate(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)):
        x=x.to(dev);y=y.to(dev)
        for k,net in nets.items():
            pr=predict(net,x,a.tta)
            cm=add_cm(torch.zeros(3,3,dtype=torch.float64),pr,y)
            all_cms[k]+=cm
            entries[k].append((name[0],cm.numpy().astype(int).ravel().tolist()))
        if (i+1)%20==0 or i+1==len(ds):print(f'WEEDMAP {split.upper()} {i+1}/{len(ds)}',flush=True)
    result={'dataset':'WeedMap RedEdge','split':split,'custom_scenes':SCENE_SPLITS[split],
            'paper_status':'external_domain_adaptation_custom_split_not_official_SOTA',
            'source_checkpoint':str(a.base_ckpt),'n_images':len(ds),'tta':a.tta,'scores':{}}
    for k,cm in all_cms.items():
        result['scores'][k]=metric(cm)
        with (out/f'perimg_weedmap_{split}_{k}.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['name']+[f'cm{r}{c}' for r in range(3) for c in range(3)])
            for row in entries[k]:wr.writerow([row[0]]+row[1])
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));print('WEEDMAP EVAL PASS')


def main():
    a=parse()
    if not Path(a.base_ckpt).is_file():raise FileNotFoundError(a.base_ckpt)
    if a.mode=='preflight':
        for s in ('train','val','test'):
            ds=get_ds(a,s)
            x,y,name=ds[0]
            print(s,'count',len(ds),'first',name,'x',x.shape,'range',float(x.min()),float(x.max()),'GT',torch.unique(y))
        print('WEEDMAP PREFLIGHT PASS: validate channel order, scene separation and radiometry before fitting')
    elif a.mode=='fit':run_fit(a)
    else:run_eval(a,a.mode)

if __name__=='__main__':main()
