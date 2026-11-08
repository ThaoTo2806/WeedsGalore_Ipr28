"""Apply the preregistered Exp11 route decision to error-budget reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def _miou(confusion):
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    return float(np.nanmean(np.divide(100.0 * intersection, union,
                                      out=np.full(3, np.nan), where=union > 0)))


def _bootstrap_ceiling(reports, category, n_boot, seed, n_seeds):
    arrays = []
    for report in reports:
        path = report.get('per_image_npz')
        if not path or not Path(path).is_file():
            arrays = []
            break
        data = np.load(path)
        if category not in data or 'base_confusion' not in data:
            arrays = []
            break
        arrays.append((data['base_confusion'], data[category]))
    if arrays and len(arrays) % n_seeds == 0:
        groups = [arrays[index:index + n_seeds] for index in range(0, len(arrays), n_seeds)]
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            selected = rng.integers(0, len(groups), size=len(groups))
            deltas = []
            for group_index in selected:
                seed_index = rng.integers(0, len(groups[group_index]))
                base, repaired = groups[group_index][seed_index]
                image_indices = rng.integers(0, base.shape[0], size=base.shape[0])
                deltas.append(_miou(repaired[image_indices].sum(0)) -
                              _miou(base[image_indices].sum(0)))
            draws.append(np.mean(deltas))
        draws = np.asarray(draws)
        return {'mean': float(np.mean(draws)),
                'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}

    values = np.asarray([report.get('categories', {}).get(category, {}).get('ceiling_delta_mIoU', 0.0)
                         for report in reports], dtype=float)
    if values.size == 0:
        return {'mean': 0.0, 'ci95': [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[draws].mean(axis=1)
    return {'mean': float(values.mean()),
            'ci95': [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]}


def decide(budget_dir, prereg_path, output_path):
    prereg = json.loads(Path(prereg_path).read_text(encoding='utf-8'))
    paths = sorted(Path(budget_dir).glob('*_budget.json'))
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if not reports:
        raise ValueError(f'No *_budget.json files under {budget_dir}')
    if any(report.get('split') == 'test' or report.get('test_accessed') for report in reports):
        raise ValueError('Exp11 route refuses test-accessed budget reports')
    expected_seeds = len(prereg.get('seeds', [0, 1, 2, 3, 4]))
    expected_reports = expected_seeds * 3
    if len(reports) != expected_reports:
        raise ValueError(f'Expected {expected_reports} E10 reports (B0/B1/M1); found {len(reports)}')

    baseline = float(np.mean([report['baseline']['mIoU'] for report in reports]))
    gap = float(prereg['target_miou'] - baseline)
    rho_max = float(prereg['rho_max'])
    required = gap / rho_max
    category_names = ['boundary_r1', 'boundary_r2', 'boundary_r3', 'small_weed',
                      'component_flip', 'background_fp', 'other']
    ceilings = {name: _bootstrap_ceiling(reports, name, prereg['bootstrap']['n_boot'],
                                          prereg['bootstrap']['seed'], expected_seeds)
                for name in category_names}
    local = ceilings['boundary_r2']['mean']
    small = ceilings['small_weed']['mean']
    component = ceilings['component_flip']['mean']
    activity_values = [report.get('activity', {}).get('delta_mIoU_when_branch_zero')
                       for report in reports]
    activity_values = [value for value in activity_values if value is not None]
    activity = {'available': bool(activity_values),
                'delta_mIoU_when_branch_zero_mean': float(np.mean(activity_values))
                if activity_values else None,
                'branch_supported_all': all(report.get('activity', {}).get('branch_scale_supported', False)
                                            for report in reports)}
    component_vote = float(np.mean([
        report.get('component_vote', {}).get('delta_mIoU', 0.0) or 0.0 for report in reports]))
    routes = []
    if local >= required:
        routes.append('A_LOCAL')
    if component >= required or component_vote >= prereg['component_vote_min_delta_pp']:
        routes.append('B_COMPONENT')
    if small >= required:
        routes.append('C_SMALL_WEED')
    if not routes:
        reason = 'No preregistered error category has sufficient ceiling for the target gap.'
    elif 'A_LOCAL' in routes:
        reason = 'Local error ceiling supports testing SC-LLR under the preregistered route.'
    else:
        reason = 'Dominant ceiling is outside SC-LLR local refinement scope.'
    result = {
        'status': 'PASS_ROUTE_ANALYSIS', 'n_reports': len(reports),
        'baseline_mIoU_mean': baseline, 'gap_pp': gap,
        'min_ceiling_required_pp': required, 'ceilings': ceilings,
        'component_vote': {'delta_mIoU': component_vote},
        'activity_audit': activity,
        'E10_INACTIVE_SUSPECTED': bool(activity_values and
            abs(activity['delta_mIoU_when_branch_zero_mean']) < prereg['activity_min_abs_delta_pp']),
        'eligible_routes': routes,
        'proceed_to_training': 'A_LOCAL' in routes,
        'reason': reason, 'test_accessed': False,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget_dir', required=True)
    parser.add_argument('--prereg', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(decide(args.budget_dir, args.prereg, args.out), indent=2))


if __name__ == '__main__':
    main()
