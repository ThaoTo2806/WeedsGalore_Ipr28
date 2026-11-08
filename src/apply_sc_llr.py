"""Apply SC-LLR integration to a verified Exp10 project checkout.

Layout expected next to this script's parent directory:
    <package>/integration/apply_sc_llr.py   (this file)
    <package>/src/nets/sc_llr.py
    <package>/tests/test_sc_llr.py

Usage: python integration/apply_sc_llr.py /path/to/WeedsGalore_Ipr28
Run it on a CLEAN Exp10 checkout. Review `git diff` before committing. It refuses unknown
source patterns and never writes anything if one pattern fails. Original line endings
(CRLF/LF) of each edited file are preserved so the diff only shows real changes.
"""
from pathlib import Path
import shutil
import sys

if len(sys.argv) != 2:
    raise SystemExit('Usage: python integration/apply_sc_llr.py /path/to/project')
project = Path(sys.argv[1]).resolve()
package = Path(__file__).resolve().parents[1]
if not (project / 'src/nets/segformer.py').is_file():
    raise SystemExit('Missing src/nets/segformer.py: check project path')


def read(path):
    raw = path.read_bytes().decode('utf-8')
    return raw.replace('\r\n', '\n'), '\r\n' in raw


def write(path, text, crlf):
    path.write_bytes((text.replace('\n', '\r\n') if crlf else text).encode('utf-8'))


def replace_once(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: expected one occurrence, got {count}: {old[:90]!r}')
    return text.replace(old, new, 1)


train_path = project / 'src/train.py'
common_path = project / 'src/ccmc_common.py'
eval_path = project / 'src/evaluate_ccmc.py'
(train, train_crlf), (common, common_crlf), (evaluate, eval_crlf) = (
    read(p) for p in (train_path, common_path, eval_path))
if 'SC_LLR_HEADS' in train + common + evaluate:
    raise SystemExit('SC_LLR_HEADS already integrated: do not apply twice')

# ---- imports and head enum ------------------------------------------------------------
train = replace_once(train, 'from nets.llr_subcell import ',
    'from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer\nfrom nets.llr_subcell import ', 'train imports')
common = replace_once(common, 'from nets.llr_subcell import ',
    'from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer\nfrom nets.llr_subcell import ', 'common imports')
train = replace_once(train, '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS]',
    '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS]', 'train head enum')
common = replace_once(common, '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS]',
    '*LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS]', 'common head enum')

# ---- flags -------------------------------------------------------------------------------
train = replace_once(train,
    "flags.DEFINE_string('run_name',", "flags.DEFINE_integer('sc_radius', 1, 'SC-LLR local radius')\n"
    "flags.DEFINE_integer('sc_guide_dim', 8, 'SC-LLR spectral embedding size')\n"
    "flags.DEFINE_integer('sc_embed_dim', 16, 'SC-LLR semantic embedding size')\n"
    "flags.DEFINE_float('sc_alpha_init', 0.1, 'SC-LLR initial gate')\n"
    "flags.DEFINE_float('sc_alpha_lr', 0.01, 'Separate learning rate for SC gate (0 = same as --lr)')\n"
    "flags.DEFINE_string('run_name',", 'train new flags')
common = replace_once(common,
    "    flags.DEFINE_float('residual_alpha_init', 0.02, 'initial sigmoid gate for LLR residual heads')",
    "    flags.DEFINE_float('residual_alpha_init', 0.02, 'initial sigmoid gate for LLR residual heads')\n"
    "    flags.DEFINE_integer('sc_radius', 1, 'SC-LLR local radius')\n"
    "    flags.DEFINE_integer('sc_guide_dim', 8, 'SC-LLR spectral embedding size')\n"
    "    flags.DEFINE_integer('sc_embed_dim', 16, 'SC-LLR semantic embedding size')\n"
    "    flags.DEFINE_float('sc_alpha_init', 0.1, 'SC-LLR initial gate (architecture only)')\n"
    "    flags.DEFINE_float('sc_alpha_lr', 0.01, 'Separate learning rate for SC gate (training only)')\n"
    "    flags.DEFINE_float('sc_alpha_override', -1.0, 'EVAL ONLY: fix SC gate in [0,1]; -1 keeps the checkpoint gate')",
    'common new flags')

# ---- model builders -----------------------------------------------------------------------
sc_build = '''    if F.head in SC_LLR_HEADS:
        if F.ccmc or not F.use_low_level_refine or F.llr_source != 'raw' or F.spectral_guide:
            raise ValueError('SC-LLR requires non-CCMC raw LLR, spectral_guide=False')
        return SCLLRSegFormer(
            num_classes=F.num_classes, in_channels=F.in_channels,
            use_multiscale_fusion=F.use_multiscale_fusion, head=F.head,
            sc_radius=F.sc_radius, sc_guide_dim=F.sc_guide_dim,
            sc_embed_dim=F.sc_embed_dim, sc_alpha_init=F.sc_alpha_init,
            pretrained_backbone=True,
        )
'''
common = replace_once(common, 'def build_model(F):\n', 'def build_model(F):\n' + sc_build, 'common builder')

sc_train = '''    if FLAGS.head in SC_LLR_HEADS:
        if (FLAGS.ccmc or not FLAGS.segformer or not FLAGS.use_low_level_refine
                or FLAGS.llr_source != 'raw' or FLAGS.spectral_guide):
            raise SystemExit('SC-LLR requires non-CCMC SegFormer + raw LLR; spectral_guide=False')
        if FLAGS.occ_weight > 0:
            raise SystemExit('SC-LLR has no occupancy head: pass --occ_weight=0.0')
        net = SCLLRSegFormer(
            num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels,
            pretrained_backbone=FLAGS.pretrained_backbone,
            use_multiscale_fusion=FLAGS.use_multiscale_fusion,
            head=FLAGS.head, sc_radius=FLAGS.sc_radius,
            sc_guide_dim=FLAGS.sc_guide_dim, sc_embed_dim=FLAGS.sc_embed_dim,
            sc_alpha_init=FLAGS.sc_alpha_init,
        )
    elif FLAGS.head in LLR_RESIDUAL_HEADS:
'''
train = replace_once(train, '    if FLAGS.head in LLR_RESIDUAL_HEADS:\n', sc_train, 'train model branch')

# ---- optimizer: separate lr for the gate logit -----------------------------------------------
old_optimizer = '''    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    # Metric'''
new_optimizer = '''    elif FLAGS.head in SC_LLR_HEADS and FLAGS.sc_alpha_lr > 0:
        alpha_params = [net.sc_llr.alpha_logit]
        alpha_ids = {id(p) for p in alpha_params}
        other_params = [p for p in net.parameters() if id(p) not in alpha_ids]
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': FLAGS.lr},
            {'params': alpha_params, 'lr': FLAGS.sc_alpha_lr},
        ])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    # Metric'''
train = replace_once(train, old_optimizer, new_optimizer, 'train optimizer')

# ---- history: alpha trajectory and EPOCH-MEAN weight entropy (no per-step GPU sync) -------------
train = replace_once(train,
    "        epoch_aux, epoch_seg, epoch_steps = 0.0, 0.0, 0\n",
    "        epoch_aux, epoch_seg, epoch_steps = 0.0, 0.0, 0\n        epoch_sc_entropy = 0.0\n",
    'train entropy init')
train = replace_once(train,
    "            epoch_steps += 1\n            loss.backward()\n",
    "            if FLAGS.head in SC_LLR_HEADS and extras and 'sc_weight_entropy' in extras:\n"
    "                epoch_sc_entropy = epoch_sc_entropy + extras['sc_weight_entropy']\n"
    "            epoch_steps += 1\n            loss.backward()\n",
    'train entropy accumulate')
train = replace_once(train,
    "        with open(history_path, 'a') as f:\n",
    "        if FLAGS.head in SC_LLR_HEADS:\n"
    "            with torch.no_grad():\n"
    "                record['sc_alpha'] = float(torch.sigmoid(net.sc_llr.alpha_logit).item())\n"
    "            record['sc_weight_entropy'] = float(epoch_sc_entropy) / max(epoch_steps, 1)\n"
    "        with open(history_path, 'a') as f:\n", 'train history')

# ---- evaluation ----------------------------------------------------------------------------------
evaluate = replace_once(evaluate,
    'from ccmc_common import build_model, define_model_flags, load_checkpoint, name_parts',
    'from ccmc_common import build_model, define_model_flags, load_checkpoint, name_parts\n'
    'from nets.sc_llr import SC_LLR_HEADS', 'eval head import')
evaluate = replace_once(evaluate,
    '    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()',
    '    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()\n'
    "    if FLAGS.sc_alpha_override >= 0:\n"
    "        if FLAGS.head not in SC_LLR_HEADS:\n"
    "            raise SystemExit('--sc_alpha_override only applies to SC-LLR heads')\n"
    "        net.set_alpha_override(FLAGS.sc_alpha_override)", 'eval override')
evaluate = replace_once(evaluate,
    "               'lambdas': net.encoder.lambdas() if FLAGS.ccmc else [],\n",
    "               'lambdas': net.encoder.lambdas() if FLAGS.ccmc else [],\n"
    "               'sc_alpha_override': FLAGS.sc_alpha_override if FLAGS.sc_alpha_override >= 0 else None,\n",
    'eval summary')

# Validate everything compiles before touching the project.
for name, source in [('train.py', train), ('ccmc_common.py', common), ('evaluate_ccmc.py', evaluate)]:
    compile(source, name, 'exec')

destinations = [
    (package / 'src/nets/sc_llr.py', project / 'src/nets/sc_llr.py'),
    (package / 'tests/test_sc_llr.py', project / 'tests/test_sc_llr.py'),
]
for src, dest in destinations:
    if not src.is_file():
        raise RuntimeError(f'Missing package file: {src}')
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite existing file: {dest}')
for src, dest in destinations:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
write(train_path, train, train_crlf)
write(common_path, common, common_crlf)
write(eval_path, evaluate, eval_crlf)
print('SC-LLR integration applied. Review git diff and run pytest before commit.')