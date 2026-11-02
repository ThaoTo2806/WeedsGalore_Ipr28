"""Diag prerequisite: Weed instance size bins (train tertiles) + per-split counts.

Bins are fixed from the TRAIN split before looking at any model output (plan 8.2).
Also inspects the instance-label format so the reviewer can check the instance source.

python src/scripts/make_instance_bins.py --dataset_path=... --out=results/instance_bins.json
"""

import json
import os
import sys

import numpy as np
from absl import app, flags
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_metrics import instance_areas, load_instance_map, weed_instances  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('out', 'results/instance_bins.json', 'output json')
flags.DEFINE_boolean('force_cc', False, 'ignore official instance files (connected components)')

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'datasets', 'weedsgalore', 'splits')


def split_names(split):
    with open(os.path.join(SPLIT_DIR, f'{split}.txt')) as f:
        return [line.strip() for line in f if line.strip()]


def binary_label(name):
    path = os.path.join(FLAGS.dataset_path, name[:10], 'semantics', name + '.png')
    label = np.array(Image.open(path))
    out = label.copy()
    out[out > 1] = 2
    return out


def main(_):
    first = split_names('train')[0]
    inst_path = os.path.join(FLAGS.dataset_path, first[:10], 'instances', first + '.png')
    fmt = {'example': inst_path, 'exists': os.path.exists(inst_path)}
    if fmt['exists']:
        im = Image.open(inst_path)
        arr = np.array(im)
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1)
        fmt.update({'mode': im.mode, 'dtype': str(arr.dtype), 'shape': list(arr.shape),
                    'n_unique': int(len(np.unique(flat, axis=0)))})
    print('instance format:', json.dumps(fmt))

    areas, sources, counts_per_image = {}, {}, {}
    for split in ('train', 'val', 'test'):
        split_areas = []
        for name in split_names(split):
            label = binary_label(name)
            raw = None if FLAGS.force_cc else load_instance_map(FLAGS.dataset_path, name)
            inst, src = weed_instances(label, raw)
            sources[src] = sources.get(src, 0) + 1
            _, a = instance_areas(inst)
            split_areas.extend(a.tolist())
        areas[split] = np.asarray(split_areas, dtype=float)
        counts_per_image[split] = len(split_names(split))

    edges = np.quantile(areas['train'], [1 / 3, 2 / 3]).tolist()
    result = {'edges': edges, 'definition': 'pixel area on original 600x600 image; '
              'small <= q33(train) < medium <= q67(train) < large',
              'source': 'cc' if 'official' not in sources else
              ('official' if 'cc' not in sources else 'mixed'),
              'sources_per_image': sources, 'instance_format': fmt, 'splits': {}}
    for split, a in areas.items():
        b = np.digitize(a, edges, right=True)
        result['splits'][split] = {
            'n_images': counts_per_image[split], 'n_instances': int(len(a)),
            'n_small': int((b == 0).sum()), 'n_medium': int((b == 1).sum()),
            'n_large': int((b == 2).sum()),
            'area_quantiles_px': np.quantile(a, [0, .1, .25, .5, .75, .9, 1]).round(1).tolist()
            if len(a) else []}
    os.makedirs(os.path.dirname(FLAGS.out) or '.', exist_ok=True)
    with open(FLAGS.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    app.run(main)
