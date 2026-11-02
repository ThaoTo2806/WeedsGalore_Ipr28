"""Params / FLOPs / latency / peak memory with ONE convention for every model (plan section 5).

FLOPs: torch.utils.flop_counter.FlopCounterMode (aten-level; counts conv AND the attention
matmuls that THOP misses). Reported as GFLOPs and GMACs = GFLOPs / 2. Element-wise ops
(CCMC modulation, GELU, LayerNorm) are not counted by FlopCounterMode; their analytic cost
is reported separately for CCMC models.
Latency: batch 1, fp32, eval, CUDA synchronised, warm-up then N timed runs; median/p10/p90.

python src/scripts/profile_model.py --ccmc=True --prior_mode=learned_weed --out=results/profile_L3.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_common import build_model, define_model_flags  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 600, 'input height = width')
flags.DEFINE_integer('warmup', 50, 'warm-up iterations')
flags.DEFINE_integer('runs', 300, 'timed iterations')
flags.DEFINE_string('label', '', 'model label')
flags.DEFINE_string('out', None, 'output json')
define_model_flags()


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    net = build_model(FLAGS).to(device).eval()
    x = torch.randn(1, FLAGS.in_channels, FLAGS.size, FLAGS.size, device=device)
    kwargs = {}
    if FLAGS.ccmc and FLAGS.prior_mode in ('gt_veg', 'gt_weed'):
        kwargs['prior_mask'] = torch.zeros(1, FLAGS.size, FLAGS.size, dtype=torch.long, device=device)

    params = sum(p.numel() for p in net.parameters())
    head_params = sum(p.numel() for p in net.prior_head.parameters()) \
        if getattr(net, 'prior_head', None) is not None else 0

    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            net(x, **kwargs)
    flops = counter.get_total_flops()

    modulation_ops = 0
    if FLAGS.ccmc and getattr(net.encoder, 'injects', False):
        # one multiply per SR-input element per modulated block (+ negligible factor cost)
        with torch.no_grad():
            feats, _ = net.encoder(x, prior_fn=None)
        for sid in net.encoder.stage_ids:
            n_blocks = len(net.encoder.encoder.block[sid])
            modulation_ops += n_blocks * feats[sid].numel()

    timings = []
    with torch.no_grad():
        for _ in range(FLAGS.warmup):
            net(x, **kwargs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(FLAGS.runs):
            if device.type == 'cuda':
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                net(x, **kwargs)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                net(x, **kwargs)
                timings.append((time.perf_counter() - t0) * 1000)
    timings = np.asarray(timings)
    result = {'label': FLAGS.label, 'input': [1, FLAGS.in_channels, FLAGS.size, FLAGS.size],
              'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
              'torch': torch.__version__, 'precision': 'fp32', 'batch': 1,
              'params_M': params / 1e6, 'prior_head_params': head_params,
              'GFLOPs_flopcounter': flops / 1e9, 'GMACs': flops / 2e9,
              'modulation_elementwise_Mops': modulation_ops / 1e6,
              'latency_ms_median': float(np.median(timings)),
              'latency_ms_p10': float(np.percentile(timings, 10)),
              'latency_ms_p90': float(np.percentile(timings, 90)),
              'latency_ms_mean': float(timings.mean()), 'runs': FLAGS.runs, 'warmup': FLAGS.warmup,
              'peak_mem_MB': torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else None}
    print(json.dumps(result, indent=2))
    if FLAGS.out:
        os.makedirs(os.path.dirname(FLAGS.out) or '.', exist_ok=True)
        with open(FLAGS.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    app.run(main)
