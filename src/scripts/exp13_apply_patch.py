"""Apply the minimal Exp13 edits to the existing repo (run from repo root, on a NEW git branch).

    python src/scripts/exp13_apply_patch.py

Every replacement is asserted (exact text must match exactly once); a file that already contains
the patch is skipped. Review with `git diff` before training.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def patch(rel, guard, edits):
    p = ROOT / rel
    t = p.read_text(encoding='utf-8')
    if guard in t:
        print('SKIP (already patched):', rel)
        return
    for old, new in edits:
        assert t.count(old) == 1, f'{rel}: expected exactly 1 match for: {old[:70]!r}, got {t.count(old)}'
        t = t.replace(old, new)
    p.write_text(t, encoding='utf-8')
    print('PATCHED:', rel)


HEADS_OLD = '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS],'
HEADS_NEW = '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS, *EXP13_HEADS],'
BB_OLD = "['b0', 'b1', 'b2']"
BB_NEW = "['b0', 'b1', 'b2', 'b3']"
IMP_OLD = 'from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer\n'
IMP_NEW = IMP_OLD + 'from nets.exp13 import EXP13_HEADS, VCFSegFormer\n'

patch('src/nets/segformer.py', "'nvidia/mit-b3'", [
    ("'b2': 'nvidia/mit-b2'}", "'b2': 'nvidia/mit-b2', 'b3': 'nvidia/mit-b3'}"),
])

patch('src/ccmc_common.py', 'EXP13_HEADS', [
    (IMP_OLD, IMP_NEW), (HEADS_OLD, HEADS_NEW), (BB_OLD, BB_NEW),
    ('    if F.head in LLR_RESIDUAL_HEADS:\n',
     "    if F.head in EXP13_HEADS:\n"
     "        if F.ccmc or not F.use_low_level_refine or F.llr_source != 'raw' or F.spectral_guide:\n"
     "            raise ValueError('VCF requires --ccmc=False --use_low_level_refine=True "
     "--llr_source=raw --spectral_guide=False')\n"
     "        return VCFSegFormer(num_classes=F.num_classes, in_channels=F.in_channels,\n"
     "                            use_multiscale_fusion=F.use_multiscale_fusion,\n"
     "                            pretrained_backbone=True, backbone=F.backbone)\n"
     '    if F.head in LLR_RESIDUAL_HEADS:\n'),
])

patch('src/train.py', 'EXP13_HEADS', [
    (IMP_OLD, IMP_NEW), (HEADS_OLD, HEADS_NEW), (BB_OLD, BB_NEW),
    ("('bilinear', *SC_LLR_HEADS)", "('bilinear', *SC_LLR_HEADS, *EXP13_HEADS)"),
    ('    elif FLAGS.head in LLR_RESIDUAL_HEADS:\n',
     "    elif FLAGS.head in EXP13_HEADS:\n"
     "        if (FLAGS.ccmc or not FLAGS.segformer or not FLAGS.use_low_level_refine\n"
     "            or FLAGS.llr_source != 'raw' or FLAGS.spectral_guide):\n"
     "            raise SystemExit('VCF requires --ccmc=False --segformer=True "
     "--use_low_level_refine=True --llr_source=raw --spectral_guide=False')\n"
     "        net = VCFSegFormer(num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels,\n"
     "                           use_multiscale_fusion=FLAGS.use_multiscale_fusion,\n"
     "                           pretrained_backbone=FLAGS.pretrained_backbone,\n"
     "                           backbone=FLAGS.backbone)\n"
     '    elif FLAGS.head in LLR_RESIDUAL_HEADS:\n'),
])
print('Done. Next: git diff, python src/scripts/exp13_make_arms.py, pytest tests/test_exp13.py')
