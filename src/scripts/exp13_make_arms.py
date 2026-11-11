"""Create configs/exp13/exp13_arms.json (same schema as exp12_arms.json; the COMMON config is
the unchanged exp12_common_config.json). Run once, then FREEZE (commit) before any training.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = json.loads((ROOT / 'configs/exp12/exp12_arms.json').read_text(encoding='utf-8'))

FSP = {'lr_schedule': 'cosine', 'warmup_epochs': 5, 'min_lr_ratio': 0.01, 'ema_decay': 0.99,
       'use_low_level_refine': True, 'llr_source': 'raw', 'input_scale': 1.0}
TTA = ['none', 'hflip', 'flips', 'd4']
SEEDS = list(range(10))


def arm(branch, backbone, head, note):
    return {'branch': branch, 'stage': 'E13', 'flags': {'backbone': backbone, 'head': head, **FSP},
            'seeds': SEEDS, 'allowed_tta': TTA, 'note': note}


arms = {
    'P_B2_LLR': src['arms']['P_B2_LLR'],                                   # reference (Exp12)
    'P13_B3_LLR': arm('PERF', 'b3', 'bilinear', 'Backbone-only control for B3.'),
    'P13_B2_VCF': arm('SCI', 'b2', 'vcf', 'VCF head, same backbone as reference P_B2_LLR.'),
    'P13_B3_VCF': arm('PERF+SCI', 'b3', 'vcf', 'VCF head on B3.'),
}
out = {'version': 'exp13-v1',
       'description': 'Exp13 arms. Common flags = exp12_common_config.json. Frozen before any run.',
       'rp_bias_grid': src.get('rp_bias_grid', []), 'arms': arms}
dst = ROOT / 'configs/exp13/exp13_arms.json'
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
print('written', dst)
