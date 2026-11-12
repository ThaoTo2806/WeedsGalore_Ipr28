"""Exp18 train-only object-centric -> online hard-negative -> full image curriculum.

A single SegFormer-B2. All mining is performed on train annotations and current
model predictions. Validation and test datasets are never wrapped or mined.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def stage_for_epoch(epoch):
    if epoch < 35: return 'object'
    if epoch < 70: return 'hard_negative'
    return 'full_image'


def crop_bounds(h, w, cy, cx, size=384):
    assert size <= min(h,w)
    y = min(max(int(cy) - size // 2, 0), h-size)
    x = min(max(int(cx) - size // 2, 0), w-size)
    return y, x


def sample_center(mask, hard_centers, stage, rng):
    if stage == 'hard_negative':
        branch = int(rng.integers(0, 100))
        if branch < 35 and len(hard_centers):
            j = int(rng.integers(len(hard_centers)))
            return tuple(map(int, hard_centers[j])), 'hard_bg_to_weed'
        target = 2 if branch < 70 else (1 if branch < 93 else 0)
    else:
        branch = int(rng.integers(0, 100))
        target = 2 if branch < 55 else (1 if branch < 90 else 0)
    ys, xs = np.nonzero(mask == target)
    if len(ys):
        j = int(rng.integers(len(ys)))
        return (int(ys[j]), int(xs[j])), ('weed' if target == 2 else 'crop' if target == 1 else 'background')
    h,w = mask.shape
    return (int(rng.integers(h)),int(rng.integers(w))), 'fallback'


class CurriculumTrainDataset(Dataset):
    def __init__(self, dataset_path, dataset_size=104, in_bands=5, num_classes=3, seed=0,
                 patch_size=384):
        from datasets import WeedsGaloreDataset
        self.ds = WeedsGaloreDataset(
            dataset_path=dataset_path, dataset_size=dataset_size,
            in_bands=in_bands, num_classes=num_classes,
            is_training=False, split='train', augmentation=False)
        assert len(self.ds) == 104, f'Expected 104 train images, got {len(self.ds)}'
        self.seed=int(seed); self.patch_size=int(patch_size)
        self.epoch=0; self.hard_cache={}
        a,u,b = self.ds[0]
        assert np.asarray(a).shape == (5,600,600)
        assert np.asarray(b).shape == (600,600)

    def __len__(self): return len(self.ds)

    def set_epoch(self, epoch):
        self.epoch=int(epoch)
        print(f'EXP18 CURRICULUM epoch={epoch} stage={stage_for_epoch(epoch)} '
              f'hard_images={len(self.hard_cache)}',flush=True)

    def __getitem__(self, i):
        im, unique, binary = self.ds[i]
        im = np.asarray(im); unique=np.asarray(unique); binary=np.asarray(binary)
        stage=stage_for_epoch(self.epoch)
        if stage=='full_image':
            return np.ascontiguousarray(im), np.ascontiguousarray(unique), np.ascontiguousarray(binary)
        # Per-worker generator: torch seeds workers independently and reproducibly.
        rng=np.random.default_rng(int(torch.randint(0,2**31,(1,)).item()))
        (cy,cx),_ = sample_center(binary,self.hard_cache.get(int(i),()),stage,rng)
        y,x=crop_bounds(binary.shape[0],binary.shape[1],cy,cx,self.patch_size)
        size=self.patch_size
        im=im[:,y:y+size,x:x+size]
        unique=unique[y:y+size,x:x+size]
        binary=binary[y:y+size,x:x+size]
        # D4 spatial augmentation applied identically to all bands and masks.
        k=int(rng.integers(4)); flip=bool(rng.integers(2))
        if k:
            im=np.rot90(im,k,axes=(-2,-1))
            unique=np.rot90(unique,k)
            binary=np.rot90(binary,k)
        if flip:
            im=np.flip(im,axis=-1); unique=np.flip(unique,axis=-1); binary=np.flip(binary,axis=-1)
        return (np.ascontiguousarray(im),np.ascontiguousarray(unique),
                np.ascontiguousarray(binary))

    @torch.no_grad()
    def refresh_hard_negatives(self,net,device,epoch,max_per_image=256):
        """Mine **train only** background pixels predicted as weed by current model.
        Restores original net mode. Inference uses one image at a time, no TTA.
        """
        assert stage_for_epoch(epoch)=='hard_negative'
        was_training=net.training
        net.eval()
        cache={}; total=0
        try:
            for i in range(len(self.ds)):
                im,_,binary=self.ds[i]
                a=np.asarray(im); gt=np.asarray(binary)
                x=torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float32,device=device)[None]
                out=net(x)
                if isinstance(out,(tuple,list)): out=out[0]
                if out.shape[-2:]!=gt.shape:
                    out=torch.nn.functional.interpolate(out.float(),size=gt.shape,mode='bilinear',align_corners=False)
                pred=out.argmax(dim=1)[0].cpu().numpy()
                ys,xs=np.where((gt==0)&(pred==2))
                if len(ys):
                    rng=np.random.default_rng(self.seed*1000003+epoch*997+i)
                    take=rng.choice(len(ys),size=min(max_per_image,len(ys)),replace=False)
                    cache[i]=np.stack([ys[take],xs[take]],axis=1).astype(np.int32)
                    total+=len(take)
            self.hard_cache=cache
        finally:
            net.train(was_training)
        print(f'EXP18 TRAIN-ONLY HARD MINING epoch={epoch} '
              f'images={len(cache)}/{len(self.ds)} centers={total}',flush=True)
