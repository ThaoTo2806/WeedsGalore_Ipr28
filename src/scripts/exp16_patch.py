"""Idempotent, fail-closed integration. Does not overwrite archived experiments."""
from pathlib import Path
import shutil, tempfile, os
R = Path(__file__).resolve().parents[2]


def patch(rel, old, new, marker, required=()):
    p = R / rel
    assert p.is_file(), f'Missing {p}'
    s = p.read_text(encoding='utf-8')
    if marker in s:
        assert s.count(marker) == 1 and all(token in s for token in required), f'Incomplete {rel} patch'
        print('ALREADY PATCHED:', rel)
        return
    assert s.count(old) == 1, f'Anchor mismatch in {rel}: {old!r}'
    original = s
    s = s.replace(old, new, 1)
    compile(s, str(p), 'exec')
    backup = p.with_name(p.name + '.pre_exp16')
    if not backup.exists():
        shutil.copy2(p, backup)
    with tempfile.NamedTemporaryFile(mode='w', dir=p.parent, suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(s)
        temp = f.name
    os.replace(temp, p)
    print('PATCH PASS:', rel, 'backup:', backup)


p = R / 'src/train.py'
assert p.is_file()
s = p.read_text()
marker = '# EXP16_POINT_MODEL_TRAIN'
if marker not in s:
    assert "flags.DEFINE_float('exp15_lovasz_weight'" in s, 'Exp15 source missing: restore working Exp15 train.py first'
    assert s.count('import exp12_config\n') == 1
    assert s.count('        net = SegFormer5Band(num_classes=') == 1
    assert s.count("flags.DEFINE_integer('batch_size',") == 1
    backup = p.with_name('train_pre_exp16.py')
    if not backup.exists(): shutil.copy2(p, backup)
    s = s.replace('import exp12_config\n', 'import exp12_config\nfrom nets.exp16_point import SpectralPointRefineSegFormer\n', 1)
    anchor = "flags.DEFINE_integer('batch_size',"
    pos = s.index('\n', s.index(anchor)) + 1
    s = s[:pos] + (
        "flags.DEFINE_bool('exp16_point_refine', False, 'Experimental point refinement architecture')\n"
        "flags.DEFINE_integer('exp16_point_budget', 4096, 'Number of refined pixels per image')\n"
    ) + s[pos:]
    s = s.replace('        net = SegFormer5Band(num_classes=',
                  '        '+marker+'\n'
                  '        net = (SpectralPointRefineSegFormer if FLAGS.exp16_point_refine else SegFormer5Band)(num_classes=',1)
    # Set point budget only on the custom model without changing the constructor of the base model.
    s = s.replace('    # Model to device\n',
                  '    if FLAGS.exp16_point_refine:\n'
                  '        if not FLAGS.segformer or FLAGS.head != "bilinear" or FLAGS.ccmc:\n'
                  '            raise ValueError("Exp16 point refinement needs standard B2 SegFormer")\n'
                  '        net.point_budget = FLAGS.exp16_point_budget\n\n'
                  '    # Model to device\n',1)
    compile(s, str(p), 'exec')
    p.write_text(s)
    print('EXP16 TRAIN PATCH PASS:', backup)
else:
    assert s.count(marker)==1 and 'SpectralPointRefineSegFormer' in s
    print('EXP16 TRAIN ALREADY PATCHED')

# Use the same model class during training and evaluate_ccmc.py. No backbone changes.
patch('src/ccmc_common.py',
      '    return SegFormer5Band(num_classes=',
      '    # EXP16_POINT_MODEL_EVAL\n'
      '    model_cls = SpectralPointRefineSegFormer if getattr(F, "exp16_point_refine", False) else SegFormer5Band\n'
      '    return model_cls(num_classes=',
      'EXP16_POINT_MODEL_EVAL', required=('model_cls',))
p = R / 'src/ccmc_common.py'
s = p.read_text()
if 'from nets.exp16_point import SpectralPointRefineSegFormer' not in s:
    assert s.count('from nets import CCMCSegFormer, SegFormer5Band\n') == 1
    s = s.replace('from nets import CCMCSegFormer, SegFormer5Band\n',
                  'from nets import CCMCSegFormer, SegFormer5Band\nfrom nets.exp16_point import SpectralPointRefineSegFormer\n',1)
    compile(s, str(p),'exec'); p.write_text(s)

patch('src/evaluate_ccmc.py',
      'define_model_flags()\n',
      "# EXP16_POINT_EVAL_FLAG\nflags.DEFINE_bool('exp16_point_refine', False, 'Point refine architecture for Exp16 only')\ndefine_model_flags()\n",
      'EXP16_POINT_EVAL_FLAG',required=('exp16_point_refine',))
print('EXP16 PATCH PASS')
