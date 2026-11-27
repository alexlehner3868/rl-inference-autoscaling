"""Analytical invariants, separate from simulator/training behavior."""
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from agent.analyze_results import aggregate, pareto_flags, relative_change, select_candidate, verify_rows


def workloads():
    return pd.DataFrame(dict(workload=['small', 'large'], arrived=[10, 90], on_time=[9, 90],
        duration_seconds=[10, 30], worker_seconds=[10., 90.], total_reward=[-3., -4.],
        late=[1, 0], dropped_deadline=[0, 0], dropped_full=[0, 0], unfinished_frames=[0, 0],
        scaling_changes=[0, 0], peak_workers=[1, 3], p95_latency_seconds=[.2, .5]))


def candidates():
    return pd.DataFrame(dict(candidate=['b', 'a', 'c'], worker_seconds=[10., 10., 12.],
        total_reward=[20., 20., 30.], min_workload_on_time_pct=[97., 97., 96.],
        aggregate_on_time_pct=[99., 99., 99.5], complete_validation=[True]*3))


def test_weighted_aggregation_and_sample_definitions():
    result = aggregate(workloads(), ['small', 'large'], 97)
    assert result['aggregate_on_time_pct'] == 99
    assert result['mean_workload_on_time_pct'] == 95
    assert result['min_workload_on_time_pct'] == 90
    assert result['average_workers'] == 2.5
    assert result['worst_workload'] == 'small'
    assert result['non_on_time_frames'] == 1
    assert not result['qualifies']
    assert (result['sla_met'], result['sla_failed']) == (1, 1)
    assert result['mean_workload_p95_seconds'] == pytest.approx(.35)
    assert 'pooled_p95_latency_seconds' not in result


def test_exact_threshold_inclusive_and_no_aggregate_substitution():
    frame = candidates()
    assert select_candidate(frame, 'sla', 97).candidate == 'a'
    assert select_candidate(frame, 'sla', 99) is None
    assert select_candidate(frame, 'reward').candidate == 'c'


def test_tie_breaking_resource_then_lexical_and_order_independence():
    frame = candidates()
    frame['total_reward'] = 20.
    for shuffled in (frame, frame.iloc[::-1]):
        assert select_candidate(shuffled, 'reward').candidate == 'a'
        assert select_candidate(shuffled, 'sla').candidate == 'a'


def test_sla_tie_prefers_reward_then_lexical():
    frame = candidates()
    frame.loc[0, 'total_reward'] = 21.
    assert select_candidate(frame, 'sla').candidate == 'b'


def test_missing_workload_is_unverified():
    result = aggregate(workloads().iloc[:1], ['small', 'large'], 90)
    assert result['sla_unknown'] == 1
    assert not result['qualifies']
    assert not result['complete_validation']


def test_undefined_zero_arrivals_never_qualifies():
    frame = workloads()
    frame.loc[0, ['arrived', 'on_time']] = 0
    result = aggregate(frame, ['small', 'large'], 0)
    assert result['sla_unknown'] == 1
    assert not result['complete_validation']
    assert not result['qualifies']


def test_missing_resources_and_latency_are_not_zero():
    frame = workloads()
    frame.loc[0, ['worker_seconds', 'p95_latency_seconds']] = np.nan
    result = aggregate(frame, ['small', 'large'], 90)
    assert math.isnan(result['worker_seconds'])
    assert math.isnan(result['mean_workload_p95_seconds'])


def test_duplicate_workloads_raise():
    frame = pd.concat([workloads(), workloads().iloc[:1]])
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(frame, ['small', 'large'], 97)


def test_unverified_and_missing_cost_excluded_from_selection():
    frame = candidates()
    frame['complete_validation'] = False
    assert select_candidate(frame, 'sla') is None
    frame['complete_validation'] = True
    frame['worker_seconds'] = np.nan
    assert select_candidate(frame, 'sla') is None


def test_pareto_dominance_ties_missing_and_floating_point_cost():
    frame = pd.DataFrame(dict(worker_seconds=[10., 10., 12., 9., 15., np.nan, 10.+1e-9],
                              rate=[99., 99., 98., 95., 100., 99., 99.]))
    assert pareto_flags(frame, 'rate') == [True, True, False, True, True, None, True]


@pytest.mark.parametrize('value,reference,expected', [(8, 10, 20), (12, 10, -20), (0, 10, 100)])
def test_relative_reductions(value, reference, expected):
    assert relative_change(value, reference) == expected


@pytest.mark.parametrize('value,reference', [(0, 0), (1, 0), (np.nan, 1), (1, np.nan)])
def test_undefined_ratios(value, reference):
    assert math.isnan(relative_change(value, reference))


def test_bad_frame_identity_rejected():
    frame = workloads().iloc[:1].copy()
    frame.loc[0, 'arrived'] = 11
    definitions = [dict(name='small', camera_fps=1, duration_seconds=10,
        intervals=[dict(camera_ids=[1], start_seconds=0, end_seconds=10)])]
    with pytest.raises(ValueError, match='Frame accounting'):
        verify_rows(frame, definitions)


def test_actual_artifact_arithmetic_independent_of_analysis():
    """Use csv/statistics, not pandas aggregation or the analysis functions."""
    import csv
    import json
    from statistics import mean, stdev
    root = Path(__file__).resolve().parents[1]
    experiment = root / 'results/shared_objective/run_20261126_112432_666216'
    with (experiment / 'all_runs.csv').open() as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 33
    with (experiment / 'method_summary.csv').open() as handle:
        summaries = {row['method']: row for row in csv.DictReader(handle)}
    for method in ('vanilla','nstep','dueling','dueling_nstep','two_dqn','ppo'):
        values = []
        for seed in range(5):
            folder = experiment / 'validation' / f'{method}_seed{seed}'
            with (folder / 'comparison.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            assert len({r['workload'] for r in rows}) == len(rows) == 20
            on_time = sum(int(r['on_time']) for r in rows)
            arrived = sum(int(r['arrived']) for r in rows)
            workers = sum(float(r['worker_seconds']) for r in rows)
            failed = sum(int(r[k]) for r in rows for k in ('late','dropped_deadline','dropped_full'))
            scaling = sum(int(r['scaling_changes']) for r in rows)
            overall = json.loads((folder / 'overall.json').read_text())
            assert arrived == 175650
            assert on_time == overall['on_time']
            assert workers == pytest.approx(overall['worker_seconds'], abs=1e-6)
            assert on_time - 2*failed - workers - .25*scaling == pytest.approx(overall['total_reward'], abs=1e-6)
            values.append(100 * on_time / arrived)
        assert mean(values) == pytest.approx(float(summaries[method]['on_time_percentage_mean']))
        assert stdev(values) == pytest.approx(float(summaries[method]['on_time_percentage_std']))


def test_cached_pipeline_and_infeasible_threshold(monkeypatch):
    """Rebuild all outputs in an isolated, new workspace temporary directory."""
    from agent.analyze_results import main, ROOT
    reference = ROOT / 'results/analysis/shared_objective_audit'
    cache = reference / 'baseline_workloads.csv'
    if not cache.exists():
        pytest.skip('Run bounded baseline analysis first to enable cached integration verification')
    with tempfile.TemporaryDirectory(prefix='.analysis_verify_', dir=ROOT / 'results/analysis') as temporary:
        temporary = Path(temporary).resolve()
        assert temporary.is_relative_to(ROOT)  # only our newly created scratch directory is cleaned
        for threshold in (97, 99):
            output = temporary / f'sla{threshold}'
            arguments = ['analyze_results', '--output', str(output), '--sla', str(threshold), '--baseline-cache', str(cache)]
            if threshold == 99:
                arguments.append('--no-figures')
            monkeypatch.setattr(sys, 'argv', arguments)
            main()
            metrics = pd.read_csv(output / 'candidate_metrics.csv')
            rows = pd.read_csv(output / 'workload_metrics.csv')
            assert len(metrics) == metrics.candidate.nunique() == 85
            assert len(rows) == 1700
            assert not rows.duplicated(['candidate','workload']).any()
            assert metrics.complete_validation.all()
            for metric in ('worker_seconds','total_reward','aggregate_on_time_pct','min_workload_on_time_pct','pooled_p95_latency_seconds'):
                original = pd.read_csv(reference / 'candidate_metrics.csv').set_index('candidate')[metric].sort_index()
                actual = metrics.set_index('candidate')[metric].sort_index()
                np.testing.assert_allclose(actual, original, rtol=0, atol=1e-6)
            selections = pd.read_csv(output / 'model_selection.csv')
            selection = selections[(selections.scope == 'RL_ONLY') & (selections.criterion == 'sla')].iloc[0]
            if threshold == 99:
                assert selection.status == 'NO QUALIFYING MODEL'
                assert pd.isna(selection.candidate)
                assert 'NO QUALIFYING MODEL' in (output / 'analysis_summary.md').read_text(encoding='utf-8')
            else:
                assert len(list(output.glob('*.png'))) == 9
                assert selection.candidate == 'ppo_seed2'
                assert selections[(selections.scope == 'ALL') & (selections.criterion == 'sla')].iloc[0].candidate == 'fixed__workers=8'
            # The default entry point refuses to overwrite any existing output directory.
            with pytest.raises(FileExistsError):
                main()
