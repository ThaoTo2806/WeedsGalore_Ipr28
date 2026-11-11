"""Register vcfctx in an already-patched Exp13 repo. Atomic checks; rerunnable.
Run from repo root: python src/scripts/exp13_v2_apply_patch.py
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def change(rel, changes, indicators):
    p = ROOT / rel
    original = p.read_text(encoding='utf8')
    if all(s in original for s in indicators):
        print('ALREADY PATCHED', rel)
        return
    s = original
    for old, new in changes:
        n = s.count(old)
        if n != 1:
            raise RuntimeError(f'{rel}: match count {n}, expected 1, for {old[:90]!r}; no file changed')
        s = s.replace(old, new)
    p.write_text(s, encoding='utf8')
    print('PATCHED', rel)

change('src/nets/exp13.py', [
    ("EXP13_HEADS = ('vcf',)", "EXP13_HEADS = ('vcf', 'vcfctx')")
], ["EXP13_HEADS = ('vcf', 'vcfctx')"])

for filename, is_train in [('src/train.py', True), ('src/ccmc_common.py', False)]:
    old_import = 'from nets.exp13 import EXP13_HEADS, VCFSegFormer\n'
    new_import = old_import + 'from nets.exp13_v2 import VCFContextSegFormer\n'
    if is_train:
        old = '        net = VCFSegFormer(num_classes=FLAGS.num_classes'
        new = "        cls = VCFContextSegFormer if FLAGS.head == 'vcfctx' else VCFSegFormer\n" \
              '        net = cls(num_classes=FLAGS.num_classes'
    else:
        old = '        return VCFSegFormer(num_classes=F.num_classes'
        new = "        cls = VCFContextSegFormer if F.head == 'vcfctx' else VCFSegFormer\n" \
              '        return cls(num_classes=F.num_classes'
    change(filename, [(old_import, new_import), (old, new)],
           ['from nets.exp13_v2 import VCFContextSegFormer',
            "VCFContextSegFormer if "])
print('VCF-Context registration complete. Run pytest and review git diff.')
