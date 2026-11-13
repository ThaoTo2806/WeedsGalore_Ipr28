# Exp26: Paired boundary-aware fine-tuning, seed 0

**No performance guarantee.** The scientific goal is to test whether train-only supervised foreground-edge CE improves on an equally fine-tuned control starting from the same Exp15-A EMA weights.

## Project dependencies

Existing Kaggle project `WeedsGalore_Ipr28` with `src/scripts/exp22_run.py`, `src/exp15_losses.py`, and `src/datasets/` from prior experiments. The package installs `src/exp26_losses.py`, `src/scripts/exp26_run.py`, and `tests/test_exp26_losses.py`; it does not replace or patch `src/train.py`.

Both arms start with the same Exp15-A pretrained B2/Raw-LLR. Control objective: 0.5 weighted CE + 0.5 weighted Dice + 0.10 CW Lovasz. Boundary objective: same plus 0.10 weighted CE, restricted to a 3px dilated vegetation/background boundary from **train GT only**. No GT or auxiliary head at inference. AdamW lr=1e-5, 12 epochs, batch 1, accum 2, EMA 0.99. Both arms use same random seeds, augmentation, and shuffled order. **This is a new fine-tuning experiment, not an exact replay of the original Exp15 loss recipe unless flags match the original recipe.** Inspect Exp15-A run_config.json to adjust --lovasz_weight and other ingredients if available.

## Kaggle commands

```python
from pathlib import Path
import zipfile,subprocess,sys,json
P=Path('/kaggle/working/WeedsGalore_Ipr28')
Z=Path('/kaggle/working/exp26_package.zip')
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith('/') or not name.startswith('exp26_package/'):continue
        rel=Path(name).relative_to('exp26_package')
        target=P/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(z.read(name))
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exp26_losses.py'],cwd=P,check=True)
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
BASE='/kaggle/input/datasets/thothanh28/exp15-ab/results/exp15/E15_A_CW_LOVASZ/seed0/final_ema.pth'
OUT=P/'results/exp26/E26_PAIR/seed0'
common=[sys.executable,str(P/'src/scripts/exp26_run.py'),f'--dataset_path={DS}',f'--base_ckpt={BASE}',f'--out_dir={OUT}','--seed=0','--epochs=12','--lr=1e-5','--batch_size=1','--grad_accum=2','--boundary_weight=0.10','--lovasz_weight=0.10']
subprocess.run(common+['--mode=preflight'],cwd=P,check=True)
for arm in ('control','boundary'):
    subprocess.run(common+['--mode=fit',f'--arm={arm}'],cwd=P,check=True)
# Once BOTH fine-tuning arms finish, one explicit, final 26-image val D4 check:
subprocess.run(common+['--mode=val','--val_once','--tta=d4'],cwd=P,check=True)
print(json.dumps(json.loads((OUT/'val_exp26_compare.json').read_text()),indent=2))
```

## Interpretation

- A (`control`) controls for continuing Exp15-A training.
- B (`boundary`) isolates additional foreground-edge CE.
- No selecting epoch by validation: both use final EMA only.
- TEST never loaded. Validation run refuses a second invocation to protect against tuning.
- A result above 86% is a **target**, never guaranteed. Seed0 validation cannot establish test improvement; independent confirmation is needed.
- The foreground-edge target uses GT only during training; at inference network takes the original 5-channel image only.
