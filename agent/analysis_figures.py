"""Figures for analyze_results; every plot uses validation data and explicit units."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from agent.analyze_results import METHODS, RL, BASE, NAMES, read_json, select_candidate

COLORS = dict(zip(METHODS, plt.get_cmap('tab10').colors))


def figures(candidates, chosen, rows, workloads, experiment, output, sla):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220, 'axes.titlesize': 11})

    def save(fig, name):
        fig.savefig(output / f'{name}.png', facecolor='white')
        plt.close(fig)

    order = [m for m in METHODS if m in set(chosen.method)]
    chosen = chosen.set_index('method').loc[order].reset_index()
    workload_order = sorted(w['name'] for w in workloads)
    short = {w: ''.join(p[0].upper() for p in w.rsplit('_', 1)[0].split('_')) + w.rsplit('_', 1)[1]
             for w in workload_order}
    for group_name, methods in [('dqn_variants', RL[:4]), ('other_rl', RL[4:]), ('baselines', BASE)]:
        fig, axes = plt.subplots(len(methods), 4, figsize=(21, 3.3 * len(methods)), layout='constrained', squeeze=False)
        metrics = [('on_time_pct', 'On-time frames (%)'), ('worker_seconds', 'Worker-seconds'),
                   ('total_reward', 'Total reward'), ('p95_latency_seconds', 'Completed-frame P95 (s)')]
        for i, method in enumerate(methods):
            group = rows[rows.method == method].set_index('workload').reindex(workload_order)
            summary = chosen[chosen.method == method].iloc[0]
            for j, (metric, label) in enumerate(metrics):
                ax = axes[i, j]
                ax.bar(np.arange(20), group[metric], color=COLORS[method], width=.8)
                ax.axhline(group[metric].mean(), color='black', ls='--', lw=1.2, label='Arithmetic workload mean')
                if metric == 'on_time_pct':
                    ax.axhline(sla, color='#b42222', ls=':', lw=1.7, label=f'Proposed {sla:g}% SLA')
                    ax.set_ylim(0, 105)
                    ax.set_title(f'{NAMES[method]} | weighted {summary.aggregate_on_time_pct:.2f}%')
                else:
                    ax.set_title(f'{NAMES[method]} | {label}')
                ax.set_ylabel(label)
                ax.set_xticks(np.arange(20), [short[w] for w in workload_order], rotation=90, fontsize=8)
                ax.grid(axis='y', alpha=.2)
        axes[0, 0].legend(fontsize=8, loc='lower left')
        fig.suptitle('Validation workload profiles | maximum-reward selections\n'
                     'FC: flash crowds; MD: mass dropoff; MC: mixed churn; MB: multi burst; RWS: ramp with spikes; 04: stress load', fontsize=13)
        save(fig, f'workloads_{group_name}')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
    panels = [('aggregate_on_time_pct', 'Frame-weighted on-time (%)'),
              ('min_workload_on_time_pct', 'Worst-workload on-time (%)'),
              ('worker_seconds', 'Allocated worker-seconds'),
              ('non_on_time_frames', 'Non-on-time frames (includes unfinished)'),
              ('pooled_p95_latency_seconds', 'Pooled completed-frame P95 (s)'),
              ('scaling_changes', 'Applied add/remove actions')]
    for ax, (metric, label) in zip(axes.flat, panels):
        ax.barh([NAMES[m] for m in order], chosen[metric], color=[COLORS[m] for m in order])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
        if metric == 'min_workload_on_time_pct':
            ax.axvline(sla, color='#b42222', ls=':', label=f'Proposed SLA {sla:g}%')
            ax.legend(fontsize=8)
    fig.suptitle('Validation | reward-selected policies and tuned baselines | separate metric units', fontsize=14)
    save(fig, 'overall_selected')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    seed_panels = [('total_reward', 'Total reward'), ('aggregate_on_time_pct', 'Frame-weighted on-time (%)'),
                   ('min_workload_on_time_pct', 'Worst-workload on-time (%)'), ('worker_seconds', 'Worker-seconds'),
                   ('pooled_p95_latency_seconds', 'Pooled completed-frame P95 (s)'), ('scaling_changes', 'Scaling changes')]
    for ax, (metric, label) in zip(axes.flat, seed_panels):
        for x, method in enumerate(RL):
            group = candidates[candidates.method == method].sort_values('seed')
            y = group[metric].to_numpy(float)
            ax.scatter(x + np.linspace(-.18, .18, len(y)), y, s=35, color=COLORS[method], alpha=.8)
            ax.errorbar(x, np.mean(y), yerr=np.std(y, ddof=1), fmt='D', color='black', ms=5, capsize=5, lw=1.5)
        ax.set_xticks(range(6), [NAMES[m].replace(' DQN', '\nDQN') for m in RL], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Validation variability | five independent training seeds per RL method\n'
                 'Dots: seeds 0–4 left to right; black diamonds/bars: mean ± sample SD; no significance claim', fontsize=13)
    save(fig, 'seed_variability')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    for ax, (metric, flag, label) in zip(axes, [
        ('aggregate_on_time_pct', 'pareto_aggregate_all_85', 'Frame-weighted on-time (%)'),
        ('min_workload_on_time_pct', 'pareto_min_workload_all_85', 'Worst-workload on-time (%)')]):
        for method in METHODS:
            group = candidates[candidates.method == method]
            ax.scatter(group.worker_seconds, group[metric], color=COLORS[method],
                       marker='s' if method in BASE else 'o', s=40, alpha=.65, label=NAMES[method])
        frontier = candidates[candidates[flag] == True].sort_values('worker_seconds')
        ax.plot(frontier.worker_seconds, frontier[metric], color='black', lw=1, alpha=.5)
        ax.scatter(frontier.worker_seconds, frontier[metric], facecolors='none', edgecolors='black', s=115, lw=1.2)
        ax.scatter(chosen.worker_seconds, chosen[metric], marker='*', s=170,
                   c=[COLORS[m] for m in order], edgecolor='black', linewidth=.7)
        ax.set(xlabel='Allocated worker-seconds (lower is better)', ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower right', ncol=2)
    axes[1].axhline(sla, color='#b42222', ls=':', label=f'Proposed SLA {sla:g}%')
    axes[1].legend(handles=[Line2D([], [], marker='*', color='none', markeredgecolor='black', markerfacecolor='gray', ms=12, label='Reward-selected'),
                           Line2D([], [], marker='o', color='none', markeredgecolor='black', ms=10, label='Nondominated'),
                           Line2D([], [], marker='s', color='none', markerfacecolor='gray', ms=7, label='Baseline configuration'),
                           Line2D([], [], color='#b42222', ls=':', label=f'Proposed {sla:g}% SLA')], fontsize=9)
    for group, offset in ((candidates, (-55, -40)), (candidates[candidates.method.isin(RL)], (-110, -70))):
        selected_sla = select_candidate(group, 'sla', sla)
        if selected_sla is not None:
            axes[1].annotate(selected_sla.candidate, (selected_sla.worker_seconds, selected_sla.min_workload_on_time_pct),
                             xytext=offset, textcoords='offset points', fontsize=8,
                             arrowprops=dict(arrowstyle='-', color='black', lw=.7))
    fig.suptitle('Validation Pareto analysis | all 30 RL checkpoints + 55 baseline configurations', fontsize=14)
    save(fig, 'pareto_frontiers')

    sens = pd.read_csv(output / 'sla_sensitivity.csv')
    thresholds = sorted(sens.sla_pct.unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    grid = sens[sens.scope.isin(METHODS)].pivot(index='scope', columns='sla_pct', values='eligible').reindex(METHODS)
    mesh = axes[0].imshow(grid, cmap='Blues', aspect='auto')
    for i in range(len(METHODS)):
        for j in range(len(thresholds)):
            axes[0].text(j, i, str(int(grid.iloc[i, j])), ha='center', va='center', color='white' if grid.iloc[i, j] > grid.to_numpy().max()/2 else 'black')
    axes[0].set_xticks(range(len(thresholds)), [f'{t:g}%' for t in thresholds])
    axes[0].set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
    axes[0].set_title('Eligible configurations (every workload passes)')
    fig.colorbar(mesh, ax=axes[0], label='Eligible candidate count', shrink=.8)
    for method in METHODS:
        g = sens[sens.scope == method]
        if g.worker_seconds.notna().any():
            axes[1].plot(g.sla_pct, g.worker_seconds, marker='o', color=COLORS[method], label=NAMES[method])
    axes[1].set(xlabel='Required minimum-workload on-time (%)', ylabel='Minimum qualifying worker-seconds',
                title='Missing point = NO QUALIFYING MODEL')
    axes[1].set_xticks(thresholds)
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=9)
    fig.suptitle('Validation SLA sensitivity | proposed thresholds, not finalized requirements', fontsize=14)
    save(fig, 'sla_sensitivity')

    summary = pd.read_csv(output / 'workload_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    for i, dimension in enumerate(('pattern', 'load_level')):
        for j, (metric, label, cmap) in enumerate((('aggregate_on_time_pct', 'Frame-weighted on-time (%)', 'YlGnBu'),
                                                   ('average_workers', 'Time-weighted average workers', 'Oranges'))):
            ax = axes[i, j]
            data = summary[summary.dimension == dimension].pivot(index='method', columns='group', values=metric).reindex(order)
            im = ax.imshow(data, aspect='auto', cmap=cmap)
            for y in range(len(order)):
                for x in range(len(data.columns)):
                    ax.text(x, y, f'{data.iloc[y,x]:.1f}', ha='center', va='center', fontsize=9,
                            bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=.5))
            ax.set_xticks(range(len(data.columns)), [s.replace('_', ' ') for s in data.columns], rotation=15, ha='right')
            ax.set_yticks(range(len(order)), [NAMES[m] for m in order])
            ax.set_title(f'{dimension.replace("_", " ").title()} | {label}')
            fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Validation workload patterns and load | reward-selected candidates\n'
                 'Resource panels distinguish better compliance from greater allocation', fontsize=14)
    save(fig, 'workload_patterns')

    # Objective choice: highest peak demand, then greatest integrated demand, then name.
    representatives = []
    for pattern in ('flash_crowds', 'mass_dropoff'):
        pool = [w for w in workloads if w['pattern'] == pattern]
        if pool:
            representatives.append(sorted(pool, key=lambda w: (
                -max(len(i['camera_ids']) for i in w['intervals']),
                -sum(len(i['camera_ids']) * (i['end_seconds']-i['start_seconds']) for i in w['intervals']), w['name']))[0])
    best_rl = chosen[chosen.method.isin(RL)].sort_values(['total_reward', 'worker_seconds', 'candidate'], ascending=[False, True, True]).iloc[0]['method']
    best_compliance = chosen[chosen.method.isin(RL)].sort_values(['aggregate_on_time_pct', 'worker_seconds', 'candidate'], ascending=[False, True, True]).iloc[0]['method']
    timeline_methods = list(dict.fromkeys((best_rl, best_compliance, *BASE)))
    if representatives:
        fig, axes = plt.subplots(len(representatives), len(timeline_methods), figsize=(21, 7.5), layout='constrained', squeeze=False)
        for i, w in enumerate(representatives):
            for j, method in enumerate(timeline_methods):
                ax = axes[i, j]
                folder = experiment / 'selected' / method if method in RL else experiment / 'baselines' / method / 'validation'
                path = folder / w['name'] / 'history.csv'
                if not path.exists():
                    ax.text(.5, .5, 'History unavailable', ha='center', transform=ax.transAxes)
                    continue
                history = pd.read_csv(path)
                source = read_json(folder / w['name'] / 'summary.json')
                service = source['effective_service_seconds']
                t = history.time_seconds
                demand_t = np.r_[history.interval_start_seconds, t.iloc[-1]]
                demand = np.r_[history.demand_fps, history.demand_fps.iloc[-1]]
                ax.step(demand_t, demand * service, where='post', color='gray', ls=':', lw=1.5, label='Demand / capacity reference')
                ax.step(t, history.workers, where='post', color=COLORS[method], lw=1.6, label='Allocated (end samples)')
                ax.step(t, history.serving_workers, where='post', color='#9a5d13', ls='--', lw=1, label='Ready, non-retiring')
                ax.set(title=f'{NAMES[method]}\n{w["name"]}', xlabel='Time (s)', ylim=(0, 12), xlim=(0, 300))
                if j == 0:
                    ax.set_ylabel('Workers / equivalent demand workers')
                if j == len(timeline_methods)-1:
                    camera_axis = ax.secondary_yaxis('right', functions=(lambda x: x/(service*w['camera_fps']), lambda x: x*service*w['camera_fps']))
                    camera_axis.set_ylabel('Active cameras (gray line only)')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8, loc='upper right')
        fig.suptitle('Validation worker allocation | existing selected histories only\n'
            'Representative rule: highest peak demand, then integrated demand, then name. '
            'Demand/capacity is a reference, not optimal provisioning.', fontsize=12)
        save(fig, 'worker_timelines')
    (output / 'figure_sources.json').write_text(json.dumps(dict(
        representatives=[w['name'] for w in representatives], timeline_methods=timeline_methods,
        timing='Demand at interval start; worker counts are end-of-interval samples. Tick-integrated worker-seconds come from source metrics.',
        plots='Validation only. Workload bar mean lines are arithmetic; overall rate is frame-weighted. P95 excludes drops and unfinished frames.'), indent=2) + '\n')
