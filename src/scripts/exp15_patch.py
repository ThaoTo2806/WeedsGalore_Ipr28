"""Idempotent Exp15 integration into existing Kaggle Exp14 VBB training code."""
from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[2]
p=R/'src/train.py'
s=p.read_text(encoding='utf-8')
marker='# EXP15_IOU_SPECTRAL_TRAIN_ONLY'
if marker in s:
    assert 'cw_lovasz_loss(out, labels' in s and 'spectral_spatial_cw_loss(' in s
    assert s.count(marker)==1
    print('EXP15 PATCH ALREADY INSTALLED')
else:
    assert s.count('import exp12_config\n')==1
    assert s.count('            scaler.scale(loss).backward()')==1
    assert s.count('            epoch_steps += 1\n')==1
    assert "flags.DEFINE_float('vbb_weight'" in s, 'Exp14 VBB source required'
    assert 'from exp14_vbb import vbb_loss' in s, 'Exp14 VBB integration required'
    assert 'from exp15_losses import' not in s
    backup=R/'src/train_pre_exp15.py'
    if not backup.exists(): shutil.copy2(p,backup)
    s=s.replace('import exp12_config\n','import exp12_config\nfrom exp15_losses import cw_lovasz_loss, spectral_spatial_cw_loss\n',1)
    anchor="flags.DEFINE_float('vbb_weight'"
    pos=s.index('\n',s.index(anchor))+1
    new_flags=(
        "flags.DEFINE_float('exp15_lovasz_weight', 0.0, 'Train-only crop weed Lovasz weight')\n"
        "flags.DEFINE_float('exp15_spectral_weight', 0.0, 'Train-only spectral-spatial CW interface loss weight')\n"
        "flags.DEFINE_integer('exp15_spectral_scale', 4, 'Spatial downsample for spectral loss')\n"
        "flags.DEFINE_float('exp15_spectral_temperature', 0.8, 'Spectral ambiguity temperature')\n"
        "flags.DEFINE_integer('exp15_spectral_radius', 2, 'Crop-weed interface radius at downsampled grid')\n"
    )
    s=s[:pos]+new_flags+s[pos:]
    anchor='            epoch_steps += 1\n'
    injection=(
        '            '+marker+'\n'
        '            if FLAGS.exp15_lovasz_weight > 0 or FLAGS.exp15_spectral_weight > 0:\n'
        '                if FLAGS.num_classes != 3 or FLAGS.head != "bilinear" or not FLAGS.segformer:\n'
        '                    raise ValueError("Exp15 expects 3-class bilinear SegFormer")\n'
        '            if FLAGS.exp15_lovasz_weight > 0:\n'
        '                loss = loss + FLAGS.exp15_lovasz_weight * cw_lovasz_loss(out, labels, ignore_index=FLAGS.ignore_index)\n'
        '            if FLAGS.exp15_spectral_weight > 0:\n'
        '                loss = loss + FLAGS.exp15_spectral_weight * spectral_spatial_cw_loss(\n'
        '                    out, labels, features, ignore_index=FLAGS.ignore_index,\n'
        '                    scale=FLAGS.exp15_spectral_scale,\n'
        '                    temperature=FLAGS.exp15_spectral_temperature, radius=FLAGS.exp15_spectral_radius)\n'
    )
    s=s.replace(anchor, anchor+injection,1)
    compile(s,str(p),'exec')
    p.write_text(s,encoding='utf-8')
    print('EXP15 PATCH PASS; backup:',backup)
