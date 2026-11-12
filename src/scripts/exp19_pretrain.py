"""Pre-adapt ONE MiT-B2 with train-only masked multispectral reconstruction."""
from pathlib import Path
import argparse, hashlib, json, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / 'src'))
from datasets import WeedsGaloreDataset
from nets.segformer import PretrainedMiTEncoder
from exp19_ssl import MaskedSpectralLearner, make_masks, ssl_loss, assert_train_only_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-5)
    a = p.parse_args()
    assert a.epochs >= 1 and a.crop in (256, 384) and a.crop % 16 == 0
    assert a.batch_size >= 1 and a.lr > 0
    dsroot = Path(a.dataset_path)
    assert dsroot.is_dir(), dsroot
    out = R / 'results/exp19/ssl_seed0'
    ckpt = out / 'encoder_ssl.pth'
    if out.exists():
        raise SystemExit(f'SSL dir exists ({out}); inspect or archive it, do not overwrite')
    out.mkdir(parents=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'A GPU is required for this experiment'
    ds = WeedsGaloreDataset(str(dsroot), 104, 5, 3, False, 'train', False)
    assert_train_only_dataset(ds)
    g = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                        drop_last=True, generator=g)
    encoder = PretrainedMiTEncoder('nvidia/mit-b2', in_channels=5)
    model = MaskedSpectralLearner(encoder).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    history = []
    for epoch in range(a.epochs):
        losses = []
        for x, _labels, _binary in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            # Random spatial crop; no labels ever passed to a loss.
            _, _, h, w = x.shape
            oy = int(torch.randint(0, h - a.crop + 1, (1,), generator=g))
            ox = int(torch.randint(0, w - a.crop + 1, (1,), generator=g))
            x = x[:, :, oy:oy+a.crop, ox:ox+a.crop]
            # Spatial masks and entire-band dropouts prevent trivial identity reconstruction.
            corrupted, m = make_masks(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                predicted = model(corrupted)
            target_small = F.interpolate(x, size=predicted.shape[-2:], mode='area')
            mask_small = F.adaptive_max_pool2d(m.float(), predicted.shape[-2:]).bool()
            loss = ssl_loss(predicted, target_small, mask_small)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite SSL loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        mean = sum(losses)/len(losses)
        print(f'EXP19 SSL epoch={epoch+1}/{a.epochs} train_mask_loss={mean:.5f}', flush=True)
        history.append(mean)
    # Save only the adapted encoder, no auxiliary reconstruction head.
    payload = {'encoder_state': {k:v.detach().cpu() for k,v in model.encoder.state_dict().items()},
               'meta': {'train_only': True, 'n_train':104, 'seed':a.seed,
                        'epochs':a.epochs, 'crop':a.crop, 'loss':'masked-L1 + NDVI/NDRE ratios',
                        'pretrained':'nvidia/mit-b2', 'no_semantic_labels_for_ssl':True}}
    torch.save(payload, ckpt)
    (out/'ssl_history.json').write_text(json.dumps({'loss':history,**payload['meta']},indent=2))
    print('EXP19 SSL PASS:',ckpt,flush=True)

if __name__ == '__main__':
    main()
