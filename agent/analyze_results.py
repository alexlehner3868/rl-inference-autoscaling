"""Read-only validation analysis; never trains or loads a policy for inference.

Run from the repository root with python -m agent.analyze_results --help.
Output directories must be new. Missing baseline workload rows can be evaluated
with the existing simulator/controllers, then reused with --baseline-cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RL = ('vanilla', 'nstep', 'dueling', 'dueling_nstep', 'two_dqn', 'ppo')
BASE = ('fixed', 'predictive', 'queue')
METHODS = RL + BASE
NAMES = dict(zip(METHODS, ('DQN', 'N-step DQN', 'Dueling DQN', 'Dueling N-step DQN',
                         'Two DQN', 'PPO', 'Fixed', 'Predictive', 'Queue threshold')))
PARAMS = {'fixed': ('workers',), 'predictive': ('window', 'scale_down_steps'),
          'queue': ('add_threshold', 'remove_threshold', 'scale_down_steps')}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative_change(value, reference):
    """Positive reduction; undefined when the reference is zero or missing."""
    if not all(pd.notna(x) and math.isfinite(x) for x in (value, reference)) or reference == 0:
        return math.nan
    return 100 * (reference - value) / reference


def aggregate(rows, expected, sla):
    """Fail closed for incomplete/undefined SLA data; no aggregate-SLA substitution."""
    if rows.workload.duplicated().any():
        raise ValueError('Duplicate workload in candidate')
    complete = set(rows.workload) == set(expected)
    rates = 100 * rows.on_time / rows.arrived.replace(0, np.nan)
    valid = bool(complete and rates.notna().all() and np.isfinite(rates).all())
    arrived, duration = rows.arrived.sum(), rows.duration_seconds.sum()
    worker_seconds = rows.worker_seconds.sum(min_count=len(rows))
    result = dict(workloads=len(rows), complete_validation=valid,
                  aggregate_on_time_pct=100 * rows.on_time.sum() / arrived if arrived else math.nan,
                  mean_workload_on_time_pct=rates.mean(), min_workload_on_time_pct=rates.min(),
                  sla_met=int((rates >= sla).sum()), sla_failed=int((rates < sla).sum()),
                  sla_unknown=int(rates.isna().sum()) + len(set(expected) - set(rows.workload)),
                  qualifies=bool(valid and (rates >= sla).all()),
                  worst_workload=str(rows.loc[rates.idxmin(), 'workload']) if rates.notna().any() else '',
                  worker_seconds=worker_seconds, average_workers=worker_seconds / duration if duration else math.nan,
                  duration_seconds=duration)
    for key in ('arrived', 'on_time', 'late', 'dropped_deadline', 'dropped_full',
                'unfinished_frames', 'scaling_changes', 'blocked_actions', 'total_reward'):
        result[key] = rows[key].sum(min_count=len(rows)) if key in rows else math.nan
    result['dropped_frames'] = result['dropped_deadline'] + result['dropped_full']
    result['non_on_time_frames'] = result['arrived'] - result['on_time']
    result['peak_workers'] = rows.peak_workers.max() if 'peak_workers' in rows and rows.peak_workers.notna().all() else math.nan
    result['mean_workload_p95_seconds'] = rows.p95_latency_seconds.mean() if rows.p95_latency_seconds.notna().all() else math.nan
    # Pooled P95 is supplied separately by overall.json, never averaged from P95s.
    return result


def select_candidate(candidates, criterion, sla=97):
    valid = candidates[candidates.complete_validation.astype(bool)].copy()
    if criterion == 'sla':
        valid = valid[valid.min_workload_on_time_pct >= sla]
        keys, ascending = ['worker_seconds', 'total_reward', 'candidate'], [True, False, True]
    elif criterion == 'reward':
        keys, ascending = ['total_reward', 'worker_seconds', 'candidate'], [False, True, True]
    else:
        raise ValueError(criterion)
    valid = valid.dropna(subset=keys[:-1])
    if valid.empty:
        return None
    return valid.sort_values(keys, ascending=ascending, kind='stable').iloc[0]


def pareto_flags(candidates, compliance):
    """All candidates: exact weak dominance with at least one strict improvement.

    Resource differences under 1e-6 worker-seconds are treated as numerical ties.
    Identical points do not dominate each other. Missing values remain unknown.
    """
    values = candidates[['worker_seconds', compliance]].to_numpy(float)
    result = []
    for cost, rate in values:
        if not np.isfinite([cost, rate]).all():
            result.append(None)
            continue
        dominates = (values[:, 0] <= cost + 1e-6) & (values[:, 1] >= rate)
        strict = (values[:, 0] < cost - 1e-6) | (values[:, 1] > rate)
        result.append(not bool(np.any(dominates & strict)))
    return result


def baseline_id(method, settings):
    return method + '__' + '_'.join(f'{key}={int(settings[key])}' for key in PARAMS[method])


def simulate_baseline(task):
    """At most 20 x 300 s per configuration; same tick engine and policy interfaces."""
    method, settings, workloads, calibration = task
    from Simulator.Simulator_queue import Simulator, SimulatorConfig, calculate_reward
    from Simulator.baseline_controllers import QueueThresholdController
    from Simulator.predictive_baseline import PredictiveController
    rows, latencies = [], []
    for workload in workloads:
        config = SimulatorConfig(camera_fps=workload['camera_fps'],
            decision_seconds=workload['decision_seconds'],
            service_seconds=calibration['service_time_mean_seconds'],
            startup_seconds=calibration['startup_mean_seconds'],
            initial_workers=settings['workers'] if method == 'fixed' else 1)
        sim = Simulator(config)
        policy = (PredictiveController(settings['window'], settings['scale_down_steps']) if method == 'predictive'
                  else QueueThresholdController(settings['add_threshold'], settings['remove_threshold'], settings['scale_down_steps']) if method == 'queue'
                  else None)
        for interval in workload['intervals']:
            steps = round((interval['end_seconds'] - sim.tick * config.tick_seconds) / config.decision_seconds)
            for _ in range(steps):
                cameras = interval['camera_ids']
                action = ('hold' if method == 'fixed' else policy.choose_action(sim) if method == 'predictive'
                          else policy.choose_action(sim, cameras))
                sim.step(action, cameras)
        last = sim.history[-1]
        scaling = sim.action_counts['add'] + sim.action_counts['remove']
        rows.append(dict(workload=workload['name'], pattern=workload['pattern'], load_level=workload['load_level'],
            arrived=sim.arrived, on_time=sim.on_time, late=sim.late, dropped_deadline=sim.dropped_deadline,
            dropped_full=sim.dropped_full, unfinished_frames=last['waiting'] + last['processing'],
            worker_seconds=sim.worker_seconds, duration_seconds=last['time_seconds'],
            peak_workers=max(h['workers'] for h in sim.history), scaling_changes=scaling,
            blocked_actions=sim.action_counts['blocked'],
            p95_latency_seconds=float(np.percentile(sim.latencies_seconds, 95)) if sim.latencies_seconds else None,
            total_reward=calculate_reward(sim.on_time, sim.late + sim.dropped_deadline + sim.dropped_full,
                                          sim.worker_seconds, scaling)))
        latencies.extend(sim.latencies_seconds)
    return rows, float(np.percentile(latencies, 95)) if latencies else None


def verify_rows(rows, workloads):
    by_name = {w['name']: w for w in workloads}
    for row in rows.to_dict('records'):
        w = by_name[row['workload']]
        expected_arrivals = sum(len(i['camera_ids']) * (i['end_seconds'] - i['start_seconds']) * w['camera_fps'] for i in w['intervals'])
        accounted = sum(row[k] for k in ('on_time', 'late', 'dropped_deadline', 'dropped_full', 'unfinished_frames'))
        if row['arrived'] != accounted or row['arrived'] != expected_arrivals:
            raise ValueError(f"Frame accounting: {row['workload']}")
        reward = row['on_time'] - 2 * (row['late'] + row['dropped_deadline'] + row['dropped_full']) - row['worker_seconds'] - .25 * row['scaling_changes']
        if not math.isclose(reward, row['total_reward'], abs_tol=1e-6):
            raise ValueError(f"Reward accounting: {row['workload']}")
        if row['peak_workers'] > 10 or row['duration_seconds'] != w['duration_seconds']:
            raise ValueError('Worker cap or duration mismatch')
        if row['pattern'] != w['pattern'] or row['load_level'] != w['load_level']:
            raise ValueError('Workload metadata mismatch')


def collect(experiment, output, sla, evaluate_baselines=False, baseline_cache=None, processes=4):
    workloads = [read_json(p) for p in sorted((ROOT / 'workloads/validation').glob('*.json'))]
    expected = [w['name'] for w in workloads]
    if len(expected) != 20 or len(set(expected)) != 20:
        raise ValueError('Expected 20 distinct validation workloads')
    originals = pd.read_csv(experiment / 'all_runs.csv')
    selected = pd.read_csv(experiment / 'selected_models.csv')
    if len(originals) != 33 or set(originals.method) != set(METHODS) or len(selected) != 9 or set(selected.method) != set(METHODS):
        raise ValueError('Shared run/selection table completeness mismatch')
    candidates, detailed, pending = [], [], []
    checks = []
    cache = pd.read_csv(baseline_cache) if baseline_cache else None

    def add(candidate, method, seed, model_path, rows, p95, source):
        rows = pd.DataFrame(rows).reset_index(drop=True)
        verify_rows(rows, workloads)
        metrics = aggregate(rows, expected, sla)
        if not metrics['complete_validation']:
            raise ValueError(f'Incomplete validation: {candidate}')
        candidates.append(dict(candidate=candidate, method=method, seed=seed, model_path=model_path,
                               source=source, pooled_p95_latency_seconds=p95, **metrics))
        detailed.append(rows.assign(candidate=candidate, method=method, source=source,
                                    on_time_pct=100 * rows.on_time / rows.arrived.replace(0, np.nan)))

    for method in RL:
        runs = originals[originals.method == method]
        if len(runs) != 5 or set(runs.seed.astype(int)) != set(range(5)):
            raise ValueError(f'Seed completeness: {method}')
        for record in runs.to_dict('records'):
            seed = int(record['seed'])
            candidate = f'{method}_seed{seed}'
            model_path = ROOT / record['model_path']
            metadata = read_json(model_path.with_suffix('.json'))
            with zipfile.ZipFile(model_path) as archive:
                saved = json.loads(archive.read('data'))
            steps = saved['num_timesteps'] * (2 if method == 'two_dqn' else 1)
            if steps != 61440 or metadata['total_timesteps'] != steps or metadata['seed'] != seed:
                raise ValueError(f'Checkpoint budget: {candidate}')
            if metadata['reward'] != read_json(experiment / 'experiment.json')['reward'] or metadata['max_workers'] != 10:
                raise ValueError(f'Checkpoint config: {candidate}')
            checks.append(dict(candidate=candidate, checkpoint_num_timesteps=saved['num_timesteps'],
                accounted_total_steps=steps, n_steps=saved.get('n_steps'), batch_size=saved.get('batch_size'),
                checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                metadata_simulator_configuration=metadata['simulator_configuration']))
            folder = experiment / 'validation' / candidate
            overall = read_json(folder / 'overall.json')
            rows = pd.read_csv(folder / 'comparison.csv')
            add(candidate, method, seed, record['model_path'], rows, overall['p95_latency_seconds'], 'existing_validation')
            for metric in ('total_reward', 'worker_seconds', 'on_time', 'arrived', 'late', 'unfinished_frames', 'scaling_changes', 'average_workers', 'peak_workers', 'dropped_frames'):
                if not math.isclose(candidates[-1][metric], record[metric], abs_tol=1e-6) or not math.isclose(candidates[-1][metric], overall[metric], abs_tol=1e-6):
                    raise ValueError(f'Source totals disagree: {candidate}/{metric}')
    for method in BASE:
        folder = experiment / 'baselines' / method
        chosen = read_json(folder / 'settings.json')
        tuning = pd.read_csv(folder / 'tuning.csv')
        for setting in tuning.to_dict('records'):
            settings = {key: int(setting[key]) for key in PARAMS[method]}
            candidate = baseline_id(method, settings)
            if all(settings[k] == chosen[k] for k in settings):
                rows = pd.read_csv(folder / 'validation/comparison.csv')
                p95 = read_json(folder / 'validation/overall.json')['p95_latency_seconds']
                add(candidate, method, 'baseline', str((folder / 'settings.json').relative_to(ROOT)), rows, p95, 'existing_validation')
            elif cache is not None and candidate in set(cache.candidate):
                rows = cache[cache.candidate == candidate].copy()
                p95 = rows.pooled_p95_latency_seconds.iloc[0]
                add(candidate, method, 'baseline', '', rows.drop(columns=['candidate', 'pooled_p95_latency_seconds']), p95, 'simulator_baseline_cache')
            elif evaluate_baselines:
                pending.append((candidate, method, settings, setting))
                continue
            else:
                # Aggregate-only candidates participate in aggregate Pareto, never SLA qualification.
                candidates.append(dict(candidate=candidate, method=method, seed='baseline', model_path='',
                    source='aggregate_only_unverified', complete_validation=False, qualifies=False,
                    aggregate_on_time_pct=setting['on_time_percentage'], pooled_p95_latency_seconds=setting['p95_latency_seconds'],
                    min_workload_on_time_pct=math.nan, mean_workload_on_time_pct=math.nan,
                    non_on_time_frames=setting['arrived']-setting['on_time'],
                    **{k: setting[k] for k in ('worker_seconds', 'total_reward', 'arrived', 'on_time', 'average_workers', 'peak_workers', 'late', 'dropped_frames', 'unfinished_frames', 'scaling_changes')}))
                continue
            for metric in ('worker_seconds', 'total_reward', 'on_time', 'arrived', 'late', 'dropped_frames', 'unfinished_frames', 'scaling_changes'):
                if not math.isclose(candidates[-1][metric], setting[metric], abs_tol=1e-6):
                    raise ValueError(f'Baseline source mismatch: {candidate}/{metric}')
            if not math.isclose(candidates[-1]['pooled_p95_latency_seconds'], setting['p95_latency_seconds'], abs_tol=1e-12):
                raise ValueError(f'Baseline pooled latency mismatch: {candidate}')
    if pending:
        started = time.monotonic()
        cached_rows = []
        calibration = read_json(ROOT / 'results/calibration/simulator_parameters.json')
        print(f'Evaluating {len(pending)} missing baseline configurations, {len(pending)*20} bounded simulator workloads; no models run.', flush=True)
        # Incremental compact cache allows reuse even if an analysis is interrupted.
        cache_path = output / 'baseline_workloads.csv'
        with ProcessPoolExecutor(max_workers=processes) as pool:
            futures = {pool.submit(simulate_baseline, (m, s, workloads, calibration)): (c, m, s, tuning)
                       for c, m, s, tuning in pending}
            for future in as_completed(futures):
                candidate, method, settings, tuning = futures[future]
                rows, p95 = future.result()
                add(candidate, method, 'baseline', '', rows, p95, 'bounded_simulator_evaluation')
                for metric in ('worker_seconds', 'total_reward', 'on_time', 'arrived', 'late', 'dropped_frames', 'unfinished_frames', 'scaling_changes'):
                    if not math.isclose(candidates[-1][metric], tuning[metric], abs_tol=1e-6):
                        raise ValueError(f'New baseline evaluation disagrees: {candidate}/{metric}')
                if not math.isclose(p95, tuning['p95_latency_seconds'], abs_tol=1e-12):
                    raise ValueError(f'New baseline pooled latency mismatch: {candidate}')
                batch = pd.DataFrame(rows).assign(candidate=candidate, pooled_p95_latency_seconds=p95)
                batch.to_csv(cache_path, mode='a', header=not cached_rows, index=False)
                cached_rows.append(candidate)
                print(f'{len(cached_rows)}/{len(pending)} {candidate}: verified against tuning totals ({time.monotonic()-started:.1f}s)', flush=True)
    candidates = pd.DataFrame(candidates).sort_values('candidate').reset_index(drop=True)
    if len(candidates) != 85 or candidates.candidate.duplicated().any():
        raise ValueError('Expected 30 RL checkpoints + 55 distinct baseline configurations')
    # Mark Alex's selections and independently reproduce them.
    candidates['reward_selected'] = False
    for method in METHODS:
        best = select_candidate(candidates[candidates.method == method], 'reward')
        reference = selected[selected.method == method].iloc[0]
        if best is None or not math.isclose(best.total_reward, reference.total_reward, abs_tol=1e-6):
            raise ValueError(f'Alex selection mismatch: {method}')
        candidates.loc[candidates.candidate == best.candidate, 'reward_selected'] = True
    pd.DataFrame(checks).to_json(output / 'checkpoint_audit.json', orient='records', indent=2)
    return candidates, pd.concat(detailed, ignore_index=True), workloads


def tables(candidates, rows, output, sla):
    candidates['pareto_aggregate_all_85'] = pareto_flags(candidates, 'aggregate_on_time_pct')
    candidates['pareto_min_workload_all_85'] = pareto_flags(candidates, 'min_workload_on_time_pct')
    candidates.to_csv(output / 'candidate_metrics.csv', index=False)
    rows.sort_values(['candidate', 'workload']).to_csv(output / 'workload_metrics.csv', index=False)
    selections, sensitivity = [], []
    groups = [('ALL', candidates), ('RL_ONLY', candidates[candidates.method.isin(RL)])]
    groups += [(m, candidates[candidates.method == m]) for m in METHODS]
    for scope, group in groups:
        for criterion in ('reward', 'sla'):
            best = select_candidate(group, criterion, sla)
            selections.append(dict(scope=scope, criterion=criterion, sla_pct=sla,
                status='SELECTED' if best is not None else 'NO QUALIFYING MODEL',
                candidate=best.candidate if best is not None else '',
                worker_seconds=best.worker_seconds if best is not None else math.nan,
                min_workload_on_time_pct=best.min_workload_on_time_pct if best is not None else math.nan,
                total_reward=best.total_reward if best is not None else math.nan))
        for threshold in sorted(set((90, 95, 97, 99, sla))):
            best = select_candidate(group, 'sla', threshold)
            sensitivity.append(dict(scope=scope, sla_pct=threshold,
                eligible=int((group.complete_validation & (group.min_workload_on_time_pct >= threshold)).sum()),
                candidate=best.candidate if best is not None else 'NO QUALIFYING MODEL',
                worker_seconds=best.worker_seconds if best is not None else math.nan,
                highest_achievable_min_pct=group.loc[group.complete_validation, 'min_workload_on_time_pct'].max()))
    pd.DataFrame(selections).to_csv(output / 'model_selection.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(output / 'sla_sensitivity.csv', index=False)
    metrics = ('aggregate_on_time_pct', 'min_workload_on_time_pct', 'worker_seconds', 'total_reward',
               'non_on_time_frames', 'pooled_p95_latency_seconds', 'scaling_changes')
    stats = []
    for method in RL:
        group = candidates[candidates.method == method]
        for metric in metrics:
            stats.append(dict(method=method, metric=metric, independent_seeds=len(group),
                              mean=group[metric].mean(), sample_std=group[metric].std(ddof=1),
                              minimum=group[metric].min(), maximum=group[metric].max()))
    pd.DataFrame(stats).to_csv(output / 'seed_statistics.csv', index=False)
    chosen = candidates[candidates.reward_selected]
    comparisons = []
    for r in chosen[chosen.method.isin(RL)].to_dict('records'):
        for b in chosen[chosen.method.isin(BASE)].to_dict('records'):
            comparisons.append(dict(candidate=r['candidate'], baseline=b['candidate'],
                worker_second_savings_pct=relative_change(r['worker_seconds'], b['worker_seconds']),
                on_time_difference_pp=r['aggregate_on_time_pct']-b['aggregate_on_time_pct'],
                non_on_time_reduction_pct=relative_change(r['non_on_time_frames'], b['non_on_time_frames']),
                recorded_failure_reduction_pct=relative_change(r['late']+r['dropped_frames'], b['late']+b['dropped_frames']),
                reward_difference=r['total_reward']-b['total_reward'],
                pooled_p95_difference_seconds=r['pooled_p95_latency_seconds']-b['pooled_p95_latency_seconds'],
                scaling_difference=r['scaling_changes']-b['scaling_changes']))
    pd.DataFrame(comparisons).to_csv(output / 'baseline_comparisons.csv', index=False)
    candidates[['candidate', 'method', 'worker_seconds', 'aggregate_on_time_pct', 'min_workload_on_time_pct',
                'reward_selected', 'pareto_aggregate_all_85', 'pareto_min_workload_all_85']].to_csv(output / 'pareto.csv', index=False)
    selected_rows = rows[rows.candidate.isin(chosen.candidate)].copy()
    summaries = []
    for dimension in ('pattern', 'load_level'):
        for (method, value), group in selected_rows.groupby(['method', dimension], sort=True):
            summaries.append(dict(method=method, dimension=dimension, group=value,
                **aggregate(group.reset_index(drop=True), group.workload, sla)))
    pd.DataFrame(summaries).to_csv(output / 'workload_summary.csv', index=False)
    # Same seed labels give descriptive paired contrasts, not a significance test.
    contrasts = []
    vanilla = candidates[candidates.method == 'vanilla'].set_index('seed')
    for method in RL[1:4]:
        for r in candidates[candidates.method == method].to_dict('records'):
            ref = vanilla.loc[r['seed']]
            contrasts.append(dict(method=method, seed=r['seed'],
                on_time_difference_pp=r['aggregate_on_time_pct']-ref.aggregate_on_time_pct,
                reward_difference=r['total_reward']-ref.total_reward,
                worker_seconds_difference=r['worker_seconds']-ref.worker_seconds))
    pd.DataFrame(contrasts).to_csv(output / 'dqn_seed_contrasts.csv', index=False)
    return chosen, selected_rows


def audit_sources(experiment, candidates, workloads):
    """Independent source checks, including history/summary/CSV agreement."""
    from workloads.generate import demand_signature
    splits = {s: [read_json(p) for p in sorted((ROOT / 'workloads' / s).glob('*.json'))]
              for s in ('train', 'validation', 'test')}
    signatures = {s: {demand_signature(w) for w in ws} for s, ws in splits.items()}
    intersections = {f'{a}/{b}': len(signatures[a] & signatures[b])
                     for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}
    if any(intersections.values()):
        raise ValueError('Demand schedule overlap between splits')
    configuration_values = {}
    history_count = 0
    selected = candidates[candidates.reward_selected]
    original_selected = pd.read_csv(experiment / 'selected_models.csv').set_index('method')
    for r in selected.to_dict('records'):
        method = r['method']
        for key in ('arrived', 'on_time', 'total_reward', 'worker_seconds', 'average_workers', 'late', 'dropped_frames', 'unfinished_frames', 'scaling_changes', 'peak_workers'):
            if not math.isclose(r[key], original_selected.loc[method, key], abs_tol=1e-6):
                raise ValueError(f'Selected aggregate mismatch: {method}/{key}')
        if not math.isclose(r['pooled_p95_latency_seconds'], original_selected.loc[method, 'p95_latency_seconds'], abs_tol=1e-12):
            raise ValueError('Selected pooled latency mismatch')
        folder = experiment / 'selected' / method if method in RL else experiment / 'baselines' / method / 'validation'
        compared = pd.read_csv(folder / 'comparison.csv').set_index('workload')
        if method in RL:
            original = pd.read_csv(experiment / 'validation' / r['candidate'] / 'comparison.csv').set_index('workload')
            pd.testing.assert_frame_equal(compared.sort_index(), original.sort_index(), check_exact=False, atol=1e-6, rtol=0)
        for w in workloads:
            name = w['name']
            summary = read_json(folder / name / 'summary.json')
            history = pd.read_csv(folder / name / 'history.csv')
            config = summary['configuration']
            for key, value in config.items():
                configuration_values.setdefault(key, set()).add(value)
            if config['initial_workers'] != (7 if method == 'fixed' else 1):
                raise ValueError('Unexpected initial workers')
            if len(history) != 300 or history.time_seconds.duplicated().any():
                raise ValueError('History coverage mismatch')
            final = history.iloc[-1]
            for key in ('arrived', 'on_time', 'late', 'dropped_deadline', 'dropped_full', 'worker_seconds'):
                for value in (summary[key], compared.loc[name, key]):
                    if not math.isclose(final[key], value, abs_tol=1e-6):
                        raise ValueError(f'Selected history mismatch: {method}/{name}/{key}')
            if not math.isclose(history.reward.sum(), summary['total_reward'], abs_tol=1e-6):
                raise ValueError('History rewards mismatch')
            if history.workers.max() != compared.loc[name, 'peak_workers']:
                raise ValueError('History peaks mismatch')
            if not math.isclose(summary['completed_latency_seconds']['p95'], compared.loc[name, 'p95_latency_seconds'], abs_tol=1e-12):
                raise ValueError('Workload latency mismatch')
            per_camera = summary['per_camera']
            if sum(v['arrived'] for v in per_camera.values()) != summary['arrived']:
                raise ValueError('Per-camera totals mismatch')
            for v in per_camera.values():
                if v['arrived'] != sum(v[k] for k in ('on_time', 'late', 'dropped_deadline', 'dropped_full', 'waiting', 'processing')):
                    raise ValueError('Per-camera accounting mismatch')
            history_count += 1
    summary = pd.read_csv(experiment / 'method_summary.csv').set_index('method')
    for method in RL:
        values = candidates[candidates.method == method]
        for metric, source in (('total_reward', 'total_reward'), ('worker_seconds', 'worker_seconds'), ('aggregate_on_time_pct', 'on_time_percentage')):
            for suffix, result in (('mean', values[metric].mean()), ('std', values[metric].std(ddof=1))):
                if not math.isclose(result, summary.loc[method, source + '_' + suffix], abs_tol=1e-6):
                    raise ValueError('Existing seed summary mismatch')
    return dict(selected_histories_checked=history_count, selected_history_rows=history_count*300,
        selected_summaries_and_per_camera_accounting='passed', seed_summary_crosscheck='passed',
        split_counts={s: len(ws) for s, ws in splits.items()}, demand_signature_intersections=intersections,
        validation_patterns=pd.Series([w['pattern'] for w in workloads]).value_counts().to_dict(),
        validation_load_levels=pd.Series([w['load_level'] for w in workloads]).value_counts().to_dict(),
        actual_selected_configuration_values={k: sorted(v) for k, v in configuration_values.items()})


def markdown_table(frame, columns, labels=None, digits=2):
    header = labels or columns
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in frame[columns].itertuples(index=False, name=None):
        values = [f'{v:,.{digits}f}' if isinstance(v, (float, np.floating)) and pd.notna(v)
                  else 'unavailable' if pd.isna(v) else str(v) for v in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def write_summary(candidates, output, experiment, sla):
    chosen = candidates[candidates.reward_selected].set_index('method').loc[list(METHODS)].reset_index()
    best_rl = select_candidate(candidates[candidates.method.isin(RL)], 'reward')
    best_sla = select_candidate(candidates, 'sla', sla)
    rl_sla = select_candidate(candidates[candidates.method.isin(RL)], 'sla', sla)
    stats = pd.read_csv(output / 'seed_statistics.csv')
    contrast = pd.read_csv(output / 'dqn_seed_contrasts.csv')
    sensitivity = pd.read_csv(output / 'sla_sensitivity.csv')
    comparisons = pd.read_csv(output / 'baseline_comparisons.csv')
    baselines = comparisons[comparisons.candidate == best_rl.candidate].copy()
    baselines['baseline'] = [s.split('__')[0] for s in baselines.baseline]
    rates = stats[stats.metric == 'aggregate_on_time_pct'].set_index('method')
    rewards = stats[stats.metric == 'total_reward'].set_index('method')
    seed_report = pd.DataFrame([dict(method=NAMES[m], on_time_mean=rates.loc[m, 'mean'], on_time_sd=rates.loc[m, 'sample_std'],
        reward_mean=rewards.loc[m, 'mean'], reward_sd=rewards.loc[m, 'sample_std']) for m in RL])
    threshold_report = sensitivity[sensitivity.scope.isin(('ALL', 'RL_ONLY'))]
    highest = sensitivity[sensitivity.sla_pct == 97][['scope', 'highest_achievable_min_pct']]
    highest = highest[highest.scope.isin(METHODS)]
    closest = candidates.sort_values(['min_workload_on_time_pct', 'worker_seconds', 'candidate'], ascending=[False, True, True]).groupby('method', sort=True).head(1)
    infeasible = [m for m in METHODS if select_candidate(candidates[candidates.method == m], 'sla', sla) is None]
    closest = closest[closest.method.isin(infeasible)]
    reward_fixed = chosen[chosen.method == 'fixed'].iloc[0]
    selected_ppo = chosen[chosen.method == 'ppo'].iloc[0]
    dqn_best = select_candidate(candidates[candidates.method.isin(RL[:4])], 'reward')
    ppo_runs = candidates[candidates.method == 'ppo'].sort_values('aggregate_on_time_pct')
    patterns = pd.read_csv(output / 'workload_summary.csv')
    def value(method, dimension, group, key):
        return patterns[(patterns.method == method) & (patterns.dimension == dimension) & (patterns.group == group)].iloc[0][key]
    def pattern_comparison(pattern, methods):
        return '; '.join(f"{NAMES[m]} {value(m, 'pattern', pattern, 'aggregate_on_time_pct'):.2f}% at {value(m, 'pattern', pattern, 'average_workers'):.2f} average workers" for m in methods)
    sla_resource_comparison = (f"{best_sla.candidate} meets the proposed SLA with {relative_change(best_sla.worker_seconds, rl_sla.worker_seconds):.2f}% fewer worker-seconds than {rl_sla.candidate}."
        if best_sla is not None and rl_sla is not None else 'Resource comparison under this SLA is unavailable when the relevant pool has no qualifying candidate.')
    seed_deltas = '; '.join(f"{NAMES[m]} {rates.loc[m, 'mean']-rates.loc['vanilla', 'mean']:+.2f} pp" for m in RL[1:4])
    infeasible_text = ', '.join(NAMES[m] for m in infeasible) or 'none'
    frontier_rl = candidates[candidates.method.isin(RL) & (candidates.pareto_aggregate_all_85 == True)]
    minimum_frontier_rl = candidates[candidates.method.isin(RL) & (candidates.pareto_min_workload_all_85 == True)]
    dominators = candidates[(candidates.worker_seconds <= best_rl.worker_seconds + 1e-6)
        & (candidates.aggregate_on_time_pct >= best_rl.aggregate_on_time_pct)
        & ((candidates.worker_seconds < best_rl.worker_seconds - 1e-6) | (candidates.aggregate_on_time_pct > best_rl.aggregate_on_time_pct))]
    dominator_note = ''
    if not dominators.empty:
        d = dominators.sort_values(['worker_seconds','candidate']).iloc[0]
        dominator_note = f' For example, {d.candidate} dominates the RL reward winner in aggregate compliance/resource space: {d.aggregate_on_time_pct:.3f}% at {d.worker_seconds:,.1f} worker-seconds.'
    detail_count = sum(candidates.loc[candidates.complete_validation, 'workloads'])
    verified_count = int(candidates.complete_validation.sum())
    baseline_verified = int((candidates.complete_validation & candidates.method.isin(BASE)).sum())
    baseline_note = ('All 52 formerly aggregate-only baseline configurations have workload evidence from bounded simulator evaluation/cache; totals match saved tuning results within 1e-6.'
                     if baseline_verified == 55 else
                     f'{55-baseline_verified} baseline configurations remain unverified at workload level and are excluded from SLA selection. Use --evaluate-baselines or --baseline-cache; any SLA winner is only among verified candidates.')
    contrasts_text = []
    for m in RL[1:4]:
        g = contrast[contrast.method == m]
        contrasts_text.append(f"{NAMES[m]} improves weighted on-time over the same-label vanilla seed in {int((g.on_time_difference_pp > 0).sum())}/5 cases and reward in {int((g.reward_difference > 0).sum())}/5")
    def selection_text(row):
        return (f"**{row.candidate}**, {row.worker_seconds:,.0f} worker-seconds, minimum workload {row.min_workload_on_time_pct:.3f}%, aggregate {row.aggregate_on_time_pct:.3f}%"
                if row is not None else '**NO QUALIFYING MODEL**')
    exp = experiment.relative_to(ROOT).as_posix()
    out = output.relative_to(ROOT).as_posix()
    text = f"""# Validation analysis for Friday, November 27, 2026, 10 AM (Toronto)

## Decision and evidence

Shared reward chooses **{select_candidate(candidates, 'reward').candidate}** overall (reward {select_candidate(candidates, 'reward').total_reward:,.2f}). Among RL, **{best_rl.candidate}** wins (reward {best_rl.total_reward:,.2f}). Best of your four variants is **{dqn_best.candidate}**; exact checkpoint: `{dqn_best.model_path}`.

At the proposed **{sla:g}% on EVERY validation workload**, minimum-resource selection is {selection_text(best_sla)}. Within RL it is {selection_text(rl_sla)}. Methods with **NO QUALIFYING MODEL**: {infeasible_text}. PPO's reward-selected checkpoint has minimum {selected_ppo.min_workload_on_time_pct:.3f}%, so its qualification must be checked separately. Qualification uses the minimum of 20 workload rates, never the aggregate rate. At 99%, {int(((candidates.method.isin(RL)) & (candidates.min_workload_on_time_pct >= 99)).sum())} RL checkpoints qualify.

All rows below are validation results. Best-of-five selections and five-seed averages answer different questions; neither establishes held-out or live performance.

## Completed contributions and repository audit

- Local HEAD is `db04a58`, “Retuned and retrained all models using max workers = 10 and updated reward function”; initial working tree was clean. Local history and the code show Alex implemented Two DQN, worker-cap integration, a common reward evaluation, baseline retuning, and the 30-checkpoint shared retraining/selection pipeline (`agent/retrain_all.py`). Existing comparison figures already show basic reward/resource tradeoffs.
- Jas implemented PPO and queue tuning historically and the Ray Serve worker, external scaling client, live evaluation and comparison scripts. The checkout contains only three live summaries: two fixed runs (30 s and 300 s) and one 30 s queue run, all on `flash_crowds_01`, max four workers. There are no saved shared-checkpoint live results or held-out test results here. Jas's ongoing validation/test work described by the group is not claimed completed from these files.
- Your prior contribution is the four DQN variants and their evaluation. This analysis adds selection under two explicit objectives, 30-checkpoint/five-seed comparisons, 55-baseline-candidate verification, Pareto/SLA/pattern analysis, presentation figures, and analytical tests. No retraining or live experiment was performed.
- Candidate coverage: **30 RL checkpoints**, six methods × seeds 0–4; **55 baseline settings** (10 fixed, 9 predictive, 36 queue). **{verified_count} of {len(candidates)} candidates** have complete workload evidence (**{int(detail_count):,} rows**). {baseline_note} No other historical experiment is pooled with this shared run.
- Training uses 30 schedules, validation 20 (15 standard, 5 stress; four per pattern), test 10. Each is 300 s, 10 FPS/camera, 1 s decisions. Demand signatures have no overlap across splits; all candidate workload identities and arrival totals agree with definitions. Test definitions were inspected for separation only; no test policy scoring was run. Total validation exposure is {best_rl.duration_seconds:,.0f} s and {int(best_rl.arrived):,} arrived frames per candidate.

## Verified configuration and scoring

Max allocations 10; minimum serving workers 1; deadline 1 s; waiting queue capacity 50 frames; tick 0.001 s. Calibration mean service is 0.07197132544888889 s, rounded to **0.072 s**; mean startup 2.3057613082 s, rounded to **2.306 s**. Each ready worker therefore has a reference capacity of 13.8889 FPS. The simulator uses constant calibrated means, not samples from measured service/startup distributions. Camera demand is exogenous; policies control workers.

The reward actually used by all six current training paths and shared evaluation is `on_time - 2*(late + dropped_deadline + dropped_full) - worker_seconds - 0.25*applied_scaling_changes`. Four variants receive these weights explicitly; PPO uses `make_env` defaults; Two DQN overrides only worker cost, which is 1 here. `experiment.json` and all 30 metadata files agree. Saved ZIP counters confirm **61,440 environment steps per run** for ordinary DQN/PPO. Two DQN saves A with 30,720 steps; the loop gives B another 30,720, totaling 61,440. This equalizes environment interactions, not gradient-update counts or compute: Two DQN's default batch is 32, variants 64, and PPO has a different optimizer/update schedule.

All controllers use the same shared evaluator. Adaptive policies start with **one ready worker**; tuned fixed starts with **{int(read_json(experiment / 'baselines/fixed/settings.json')['workers'])} ready workers**, with no initial startup charge. Treat this as a warm-start convention to revisit, not equal startup provisioning. PPO evaluation correctly uses PPO in `retrain_all.py`; standalone `evaluate_dqn.py` is a DQN loader. Current observation demand is instantaneous scheduled demand, not the “recent arrival rate” described in older `docs/rl_design.md`.

## Reward-selected comparison

{markdown_table(chosen.assign(method=chosen.method.map(NAMES)), ['method','aggregate_on_time_pct','mean_workload_on_time_pct','min_workload_on_time_pct','worker_seconds','total_reward','scaling_changes'], ['Method','Weighted on-time %','Workload mean %','Worst workload %','Worker-s','Reward','Scales'])}

Dueling N-step seed 0's worst workload is `{best_rl.worst_workload}`; it fails {int(best_rl.sla_failed)} of 20 workloads at {sla:g}%. Best RL versus each reward-tuned baseline (positive savings/reduction is favorable; negative means worse):

{markdown_table(baselines, ['baseline','worker_second_savings_pct','on_time_difference_pp','non_on_time_reduction_pct','reward_difference','pooled_p95_difference_seconds'], ['Baseline','Worker-s savings %','On-time difference pp','Non-on-time reduction %','Reward difference','P95 difference s'], digits=3)}

The best RL reward exceeds {int((baselines.reward_difference > 0).sum())} of the three tuned baseline rewards. Check the signed table before claiming an efficiency win: better compliance achieved by allocating more workers is a tradeoff. PPO's reward-selected policy uses {-relative_change(selected_ppo.worker_seconds, reward_fixed.worker_seconds):.2f}% more worker-seconds than fixed, with compliance difference {selected_ppo.aggregate_on_time_pct-reward_fixed.aggregate_on_time_pct:+.3f} pp. {sla_resource_comparison} No universal RL superiority is supported; cheaper DQN policies occupy lower-resource/lower-compliance tradeoffs.

## Seed variability and DQN interpretation

{markdown_table(seed_report, ['method','on_time_mean','on_time_sd','reward_mean','reward_sd'], ['Method','Weighted on-time mean %','Sample SD pp','Reward mean','Reward sample SD'])}

{'; '.join(contrasts_text)}. These are descriptive paired seed labels, not a statistical significance test. Mean compliance differences from vanilla: {seed_deltas}. The combination's seed spread means its best checkpoint is not representative of every run. PPO ranges from {ppo_runs.iloc[0].aggregate_on_time_pct:.2f}% ({ppo_runs.iloc[0].candidate}) to {ppo_runs.iloc[-1].aggregate_on_time_pct:.2f}% ({ppo_runs.iloc[-1].candidate}). Baselines are deterministic single evaluations per configuration; no five-trial baseline uncertainty is implied.

## SLA sensitivity and Pareto scope

{markdown_table(threshold_report, ['scope','sla_pct','eligible','candidate','worker_seconds'], ['Scope','Threshold %','Eligible','Minimum-resource candidate','Worker-s'])}

Highest achievable **minimum-workload** compliance per method (not a guarantee beyond these validation schedules):

{markdown_table(highest, ['scope','highest_achievable_min_pct'], ['Method','Highest minimum %'], digits=3)}

Closest candidates for infeasible methods, **not SLA-compliant**, use greatest minimum-workload compliance, then lowest worker-seconds, then lexical ID:

{markdown_table(closest, ['method','candidate','min_workload_on_time_pct'], ['Method','Closest candidate','Minimum %'], digits=3)}

Reward selection breaks ties by fewer worker-seconds then lexical ID; SLA selection breaks ties by higher reward then lexical ID. Comparisons are inclusive, without lowering thresholds. Pareto tables/figures cover **all {len(candidates)}** candidates for aggregate and minimum-workload compliance. Equal points do not dominate one another; worker-second differences below 1e-6 are numerical ties. Reward selections are starred; nondominated candidates are circled. Aggregate/minimum frontiers contain {int(candidates.pareto_aggregate_all_85.sum())}/{int(candidates.pareto_min_workload_all_85.sum())} candidates respectively; flags and identities are in `pareto.csv`.

RL aggregate-frontier candidates: {', '.join(frontier_rl.candidate) or 'none'}. RL minimum-workload-frontier candidates: {', '.join(minimum_frontier_rl.candidate) or 'none'}.{dominator_note}

## Workload behavior and scaling

`workload_summary.csv` reports pattern and standard/stress groups for reward-selected policies. Multi-burst: {pattern_comparison('multi_burst', ('dueling_nstep','queue','predictive'))}. Mixed churn: {pattern_comparison('mixed_churn', ('dueling_nstep','queue'))}. Mass dropoff: {pattern_comparison('mass_dropoff', ('dueling_nstep','queue'))}. Dueling N-step's standard/stress compliance is {value('dueling_nstep','load_level','standard','aggregate_on_time_pct'):.2f}%/{value('dueling_nstep','load_level','stress','aggregate_on_time_pct'):.2f}%, versus queue's {value('queue','load_level','standard','aggregate_on_time_pct'):.2f}%/{value('queue','load_level','stress','aggregate_on_time_pct'):.2f}%. Read compliance alongside resources before attributing gains to better decisions.

Applied scaling actions: {', '.join(f'{NAMES[r.method]} {int(r.scaling_changes):,}' for r in chosen.itertuples())}. Starting/retiring allocations consume worker-seconds. Existing histories supply objectively selected high-demand flash-crowd and mass-dropoff examples (`figure_sources.json`). Ready/allocated counts are end-of-second samples. Demand divided by tick-calibrated capacity is a reference, not optimal provisioning; startup, backlog, deadlines and in-flight work matter. No artificial optimality/excess-capacity metric was constructed.

## Definitions, limitations and unresolved concerns

- Frame identity verified on every available workload: arrivals = on-time + late + deadline drops + full-queue drops + unfinished. Unfinished frames are included in non-on-time counts but **excluded from the failure penalty**; episodes have no drain period. Changing this would change the objective, so it was disclosed and preserved. Failure reductions include unfinished; `recorded_failure_reduction_pct` separately excludes them.
- On-time is frame-weighted globally; workload mean is arithmetic; average workers is total worker-seconds / total duration. Latency describes **completed frames only** and excludes dropped/unfinished work. A low P95 can coexist with many drops. Pooled P95 comes from saved overall results or pooled bounded-evaluation latencies; workload P95 means are separate. Saved RL histories lack per-frame latencies, so their pooled P95 cannot be independently reconstructed. Peak workers is the maximum recorded end-of-decision allocation, which can miss within-interval transients.
- Two DQN only saves A, not B/replay buffers. Loading A with ordinary `DQN.load` preserves greedy policy inference (also checked against `SelectingDQN.load` on three observations), but resuming as ordinary DQN loses the custom target-training behavior. Metadata's `simulator_configuration` is written from dataclass defaults; actual selected summaries and code were checked independently rather than trusting that historical snapshot alone.
- All 30 saved policies were loaded locally without running workloads. SB3 emits an n-step replay-buffer warning on reload; the saved replay-buffer kwargs explicitly retain `n_steps=3` and `gamma=0.99`, so this warning does not indicate missing n-step configuration or changed inference here. Blank CSV metrics mean unavailable/undefined, including relative reductions with zero reference denominators.
- PPO and Two DQN currently match shared weights by default, not because `retrain_all.py` passes the full reward dictionary into those trainers. A future weight edit could silently desynchronize them. Standalone PPO validation hardcodes 300 s and writes a fixed comparison path; historical DQN summarization recomputes reward with current defaults. Neither was executed, and shared artifacts were used instead.
- Historical `Simulator/Tuning/tune_queue.py` sets `scale_down_patiences = 1` then iterates it, a TypeError if run. The shared retraining tuner uses a valid 36-setting grid and is unaffected. No existing code was changed. Older experiment metrics can use different caps/rewards; they cannot be merged uncritically.
- Ray Serve uses wall-clock HTTP/JPEG/model costs, variable startup and shared CPU contention (0.5 Ray CPU allocation per worker, one PyTorch thread), with a 0.5 s allocation-status poll and 0.01 s loop. Its local queue coexists with Serve queuing. It drains for one deadline after arrivals while charging worker-seconds only to nominal duration, unlike the simulator's no-drain cutoff. Shutdown waits for requests without consuming their outcomes into metrics afterward; residual pending requests remain unfinished. These boundary/accounting choices and differing removal guards limit direct comparability.
- `Rayserve/run_all_live.py` still references old `nstep_seed2_30k.zip`, old PPO, and fixed four workers. Existing live results use cap four and do not validate the new shared checkpoints. `results/rayserve_live/comparison.csv` omits the saved 300 s fixed run. The two 30 s live results (fixed 100%, queue 75.42%) are too limited to support deployment claims. No Ray Serve process was started here.
- Validation has been used for checkpoint and controller selection, so estimates are optimistic for deployment. No significance or test-generalization claim is made. `py` is absent from this session's PATH; `python` has the existing needed packages. No dependencies were installed. Broad repository pytest was intentionally not run because it collects the unrelated OpenCV-dependent Rayserve test.

## Tomorrow's decisions and speaking points

Decide whether the primary objective is shared reward or a per-workload SLA, whether 97% is appropriate for stress workloads, whether fixed warm starts are fair, and whether unfinished work needs an explicit terminal penalty/drain convention. Freeze those choices before Jas evaluates the held-out test split. Use the selected shared model paths rather than old live-run defaults.

Suggested speaking points: “I checked every seed and all 55 baseline configurations. Our current reward winner is {select_candidate(candidates, 'reward').candidate}. My best variant is {dqn_best.candidate}, but it fails {int(dqn_best.sla_failed)} workloads at {sla:g}%. The every-workload rule gives a different answer, so we should report both. The DQN extensions improve mean compliance, but burst workloads and scaling churn remain weaknesses. We need to freeze the objective and let Jas test the selected policies on held-out workloads.”

Recommended next experiments (not performed): a single held-out evaluation of frozen selections with Jas; matched warm/cold-start and cutoff conventions; full-duration live comparisons using shared checkpoints, cap ten and repeated measurements; only afterward consider reward/terminal-penalty or burst-response retraining. Avoid further validation tuning that is then described as test performance.

## Reproduce and review

From the repository root, using existing dependencies (choose a **new** output directory):

```powershell
python -m pytest tests -q
python -m agent.analyze_results --experiment {exp} --output results/analysis/reproduction --sla {sla:g} --evaluate-baselines --baseline-workers 4
```

Reuse the bounded baseline results without simulator reruns:

```powershell
python -m agent.analyze_results --experiment {exp} --output results/analysis/reproduction_cached --sla {sla:g} --baseline-cache {out}/baseline_workloads.csv
```

Existing output directories are refused. `--sla` changes the selection/figure line and is included alongside 90/95/97/99 sensitivity. `--no-figures` supports tabular checks. Review this summary, `model_selection.csv`, `candidate_metrics.csv`, `baseline_comparisons.csv`, `seed_statistics.csv`, `workload_summary.csv` and the nine PNG figures. `verification.json` records source hashes and accounting checks; `checkpoint_audit.json` records saved budgets and model hashes. `workload_metrics.csv` is the consolidated analysis table; `baseline_workloads.csv` stores only newly evaluated workload metrics, not raw histories/models.

Transfer `agent/analyze_results.py`, `agent/analysis_figures.py` and `tests/test_analysis_results.py`, plus this dedicated output directory as needed. Existing teammate files are preserved. No stage, commit, push, remote access or GitHub access was performed.
"""
    (output / 'analysis_summary.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'results/shared_objective/run_20261126_112432_666216')
    parser.add_argument('--output', type=Path, required=True, help='New directory inside this workspace; existing directories are refused')
    parser.add_argument('--sla', type=float, default=97)
    parser.add_argument('--evaluate-baselines', action='store_true', help='Evaluate the 52 missing baseline configurations, simulator only')
    parser.add_argument('--baseline-cache', type=Path, help='Reuse baseline_workloads.csv from an earlier analysis')
    parser.add_argument('--baseline-workers', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.sla <= 100:
        parser.error('SLA must be between 0 and 100 percent')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error('Output must be a new directory inside the current workspace')
    output.mkdir(parents=True, exist_ok=False)
    candidates, rows, workloads = collect(args.experiment.resolve(), output, args.sla,
        args.evaluate_baselines, args.baseline_cache, args.baseline_workers)
    chosen, selected_rows = tables(candidates, rows, output, args.sla)
    if not args.no_figures:
        from agent.analysis_figures import figures
        figures(candidates, chosen, selected_rows, workloads, args.experiment.resolve(), output, args.sla)
    source_audit = audit_sources(args.experiment.resolve(), candidates, workloads)
    write_summary(candidates, output, args.experiment.resolve(), args.sla)
    manifest = dict(experiment=str(args.experiment.resolve().relative_to(ROOT)), sla_pct=args.sla,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                         stdin=subprocess.DEVNULL, stderr=subprocess.PIPE).strip(),
        candidates=len(candidates), workload_rows=len(rows), expected_workloads=20,
        verified_candidates=int(candidates.complete_validation.sum()),
        frame_accounting_and_reward_checks='passed for every available workload row',
        source_audit=source_audit,
        package_versions={name: version(name) for name in ('numpy','pandas','matplotlib','pytest','gymnasium','stable-baselines3','torch')},
        baseline_cache_sha256=hashlib.sha256(args.baseline_cache.read_bytes()).hexdigest() if args.baseline_cache else None,
        source_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for directory in ('agent','Simulator','Rayserve','workloads') for p in sorted((ROOT / directory).rglob('*.py'))},
        definitions=dict(reward='on_time - 2*(late+dropped_deadline+dropped_full) - worker_seconds - 0.25*scaling_changes',
            unfinished='Included in non-on-time frames, excluded from failure reward; no drain period',
            pooled_p95='Completed frames only; from overall.json or bounded simulator evaluation',
            workload_p95_mean='Arithmetic mean of workload P95s, not a pooled P95',
            tie_breaking_reward='Highest reward; then lowest worker-seconds; then lexical candidate ID',
            tie_breaking_sla='Lowest worker-seconds; then highest reward; then lexical candidate ID',
            closest='Highest minimum-workload compliance; then lowest worker-seconds; then lexical ID; never labelled compliant',
            pareto='All 30 RL checkpoints and all 55 tuned baseline configurations; cost tolerance 1e-6 worker-seconds',
            baseline_variability='One deterministic result per configuration; no independent-trial standard deviation'),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(args.experiment.resolve().rglob('*')) if p.suffix in ('.csv', '.json')},
        workload_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / 'workloads').rglob('*.json'))},
        generated_files=sorted(p.name for p in output.iterdir()))
    (output / 'verification.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(chosen[['candidate','aggregate_on_time_pct','min_workload_on_time_pct','worker_seconds','total_reward']].to_string(index=False))
    print(f'Analysis complete: {output}', flush=True)


if __name__ == '__main__':
    main()
