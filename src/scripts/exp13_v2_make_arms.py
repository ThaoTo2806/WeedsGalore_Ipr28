"""Create a NEW versioned registry, preserving original Exp13 registry and results."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
src = ROOT / 'configs/exp13/exp13_arms.json'
dst = ROOT / 'configs/exp13/exp13_v2_arms.json'
assert src.is_file(), f'Missing {src}'
assert not dst.exists(), f'Already exists: {dst} (do not overwrite frozen registry)'
data = json.loads(src.read_text(encoding='utf8'))
assert 'P13_B2_VCF' in data['arms']
for bb in ('b2', 'b3'):
    ref = dict(data['arms'][f'P13_{bb.upper()}_VCF'])
    ref['flags'] = dict(ref['flags'])
    ref['flags']['head'] = 'vcfctx'
    ref['branch'] = 'SCI-v2'
    ref['stage'] = 'E13v2'
    ref['note'] = 'Single-model vegetation-gated multi-dilation crop-weed context head; exploratory.'
    ref['seeds'] = list(range(5))  # development ONLY; no confirmation bypass
    data['arms'][f'P13_{bb.upper()}_VCFCTX'] = ref
data['version'] = 'exp13-v2-context-dev'
data['description'] = 'New single-network context-head development registry. Do not use for confirmation or test.'
dst.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
print('CREATED', dst)
