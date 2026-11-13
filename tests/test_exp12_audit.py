"""Torch-free tests for exp12_eval_per_image.py (helpers) and exp12_bootstrap.py.
Run: python -m pytest -q test_exp12_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp12_bootstrap as bs
import exp12_eval_per_image as ev


def synth_cms(rng, n_img=26, n_seed=5, quality=0.93, seed_noise=0.003, image_noise=0.03):
    """Per-image 3x3 confusion matrices. Image difficulty is SHARED by all seeds (large image_noise),
    seeds only add a small perturbation (seed_noise), as in real training runs."""
    out = np.zeros((n_seed, n_img, 3, 3), dtype=np.int64)
    for i in range(n_img):
        prior = rng.dirichlet([60, 4, 3])
        gt_counts = (prior * 360000).astype(int)
        img_q = quality + rng.normal(0, image_noise)
        for s in range(n_seed):
            q = float(np.clip(img_q + rng.normal(0, seed_noise), 0.5, 0.99))
            for g in range(3):
                n = gt_counts[g]
                stay = rng.binomial(n, q if g == 0 else q - 0.12)
                rest = n - stay
                other = [c for c in range(3) if c != g]
                a = rng.binomial(rest, 0.5)
                out[s, i, g, g] = stay
                out[s, i, g, other[0]] = a
                out[s, i, g, other[1]] = rest - a
    return out


def test_cm_and_miou_match_direct_pixel_computation():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, size=(40, 50))
    pred = np.where(rng.random(gt.shape) < 0.8, gt, rng.integers(0, 3, size=gt.shape))
    gt[0, :5] = -1                                        # ignored pixels
    cm = ev.cm_from_arrays(gt, pred)
    assert cm.sum() == (gt >= 0).sum()
    ious = []
    valid = gt >= 0                                       # ignore_index pixels are excluded everywhere
    for c in range(3):
        g, p = (gt == c) & valid, (pred == c) & valid
        ious.append(100.0 * (g & p).sum() / ((g | p).sum()))
    m, iou = ev.miou_from_cm(cm)
    assert np.allclose(iou, ious) and np.isclose(m, np.mean(ious))


def test_two_iou_implementations_agree():
    cm = synth_cms(np.random.default_rng(1), 5, 1)[0].sum(0)
    assert np.isclose(ev.miou_from_cm(cm)[0], bs.iou_from_cm(cm).mean())


def test_vectorised_bootstrap_equals_naive_loop():
    rng = np.random.default_rng(2)
    cms = synth_cms(rng, n_img=26, n_seed=3)
    pts, draws = bs.bootstrap([cms], n_boot=50, rng_seed=7)
    W = np.random.default_rng(7).multinomial(26, np.full(26, 1 / 26), size=50)
    for b in (0, 17, 49):
        vals = []
        for s in range(3):
            pooled = sum(W[b, n] * cms[s, n] for n in range(26))
            vals.append(bs.iou_from_cm(pooled))
        assert np.allclose(draws[0][b], np.mean(vals, axis=0))
    assert np.allclose(pts[0], bs.iou_from_cm(cms.sum(1)).mean(0))


def test_paired_delta_and_widths(tmp_path):
    rng = np.random.default_rng(3)
    a = synth_cms(np.random.default_rng(3), quality=0.945)
    b = synth_cms(np.random.default_rng(3), quality=0.93)         # same rng -> same image difficulty, lower quality
    for arm, cms in (('P_B2_LLR', a), ('P_B0_LLR', b)):
        for s in range(5):
            rows = [(i, f'2023-05-25_{i:04d}', cms[s, i]) for i in range(26)]
            ev.write_rows(tmp_path / f'perimg_{arm}_test_logit_seed{s}.csv', arm, s, 'test', 'logit', rows)
    rep = bs.main([f'--per_image_dir={tmp_path}', '--split=test', '--mode=logit',
                   '--arms=P_B2_LLR,P_B0_LLR', '--thresholds=80', '--n_boot=2000'])
    m = rep['arms']['P_B2_LLR']['mIoU']
    assert m['ci95_low'] <= m['point'] <= m['ci95_high']
    t = rep['arms']['P_B2_LLR']['seed_t_interval']
    assert (m['ci95_high'] - m['ci95_low']) > (t['ci95_high'] - t['ci95_low'])     # image noise > seed noise
    assert rep['deltas']['P_B2_LLR-P_B0_LLR']['point'] > 0
    assert (tmp_path / 'bootstrap_test_logit.json').is_file()
    # identical inputs -> delta exactly 0
    rep0 = bs.main([f'--per_image_dir={tmp_path}', '--split=test', '--mode=logit', '--arms=P_B2_LLR,P_B2_LLR',
                    '--n_boot=200', f'--out_json={tmp_path / "zero.json"}'])
    d0 = rep0['deltas']['P_B2_LLR-P_B2_LLR']
    assert d0['point'] == 0 and d0['ci95_low'] == 0 and d0['ci95_high'] == 0


def test_mismatched_images_rejected(tmp_path):
    cms = synth_cms(np.random.default_rng(5), n_seed=1)
    ev.write_rows(tmp_path / 'perimg_A_val_prob_seed0.csv', 'A', 0, 'val', 'prob',
                  [(i, f'img{i:02d}', cms[0, i]) for i in range(26)])
    ev.write_rows(tmp_path / 'perimg_B_val_prob_seed0.csv', 'B', 0, 'val', 'prob',
                  [(i, f'oth{i:02d}', cms[0, i]) for i in range(26)])
    with pytest.raises(SystemExit):
        bs.main([f'--per_image_dir={tmp_path}', '--split=val', '--mode=prob', '--arms=A,B'])


def test_roundtrip_csv(tmp_path):
    cm = np.arange(9).reshape(3, 3) * 10
    ev.write_rows(tmp_path / 'x.csv', 'A', 0, 'val', 'prob', [(0, 'n0', cm), (1, 'n1', cm + 1)])
    back = ev.read_rows_cm(tmp_path / 'x.csv')
    assert np.array_equal(back[0], cm) and np.array_equal(back[1], cm + 1)


def test_find_ckpts_exact_arm_and_seed(tmp_path):
    for arm in ('P_B2_LLR', 'P_B0_LLR'):
        for s in (0, 1):
            d = tmp_path / 'backup' / 'exp12' / arm / f'seed{s}'
            d.mkdir(parents=True)
            (d / 'final_ema.pth').write_bytes(b'x')
    got = ev.find_ckpts([tmp_path], tmp_path / 'proj', 'P_B2_LLR', [0, 1])
    assert all('P_B2_LLR' in str(p) and p.parent.name == f'seed{s}' for s, p in got.items())
    with pytest.raises(FileNotFoundError):
        ev.find_ckpts([tmp_path], tmp_path / 'proj', 'P_B2_LLR', [0, 3])


def test_find_reference(tmp_path):
    (tmp_path / 'val_none_summary.json').write_text(json.dumps({'split': 'val', 'tta': 'none', 'metrics': {'mIoU': 84.0}}))
    (tmp_path / 'val_d4_summary.json').write_text(json.dumps({'split': 'val', 'tta': 'd4', 'metrics': {'mIoU': 84.7}}))
    p, v = ev.find_reference(tmp_path, 'val', 'd4')
    assert v == 84.7 and p.name == 'val_d4_summary.json'
    assert ev.find_reference(tmp_path, 'test', 'd4') is None


def test_test_split_refused_without_flag():
    with pytest.raises(SystemExit, match='Refusing TEST'):
        ev.main(['--dataset_path=x', '--splits=val,test', '--project=/nonexistent'])


def test_unknown_arm_refused():
    with pytest.raises(SystemExit, match='not supported'):
        ev.main(['--dataset_path=x', '--arms=P_B3_LLR', '--project=/nonexistent'])
