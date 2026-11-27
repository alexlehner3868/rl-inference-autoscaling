# Validation analysis for Friday, November 27, 2026, 10 AM (Toronto)

## Decision and evidence

Shared reward chooses **fixed__workers=7** overall (reward 130,767.00). Among RL, **dueling_nstep_seed0** wins (reward 120,437.87). Best of your four variants is **dueling_nstep_seed0**; exact checkpoint: `results/shared_objective/run_20261126_112432_666216/models/dueling_nstep_seed0.zip`.

At the proposed **97% on EVERY validation workload**, minimum-resource selection is **fixed__workers=8**, 48,000 worker-seconds, minimum workload 97.636%, aggregate 99.886%. Within RL it is **ppo_seed2**, 59,280 worker-seconds, minimum workload 97.674%, aggregate 99.443%. Methods with **NO QUALIFYING MODEL**: DQN, N-step DQN, Dueling DQN, Dueling N-step DQN, Two DQN, Predictive, Queue threshold. PPO's reward-selected checkpoint has minimum 93.414%, so its qualification must be checked separately. Qualification uses the minimum of 20 workload rates, never the aggregate rate. At 99%, 0 RL checkpoints qualify.

All rows below are validation results. Best-of-five selections and five-seed averages answer different questions; neither establishes held-out or live performance.

## Completed contributions and repository audit

- Local HEAD is `db04a58`, “Retuned and retrained all models using max workers = 10 and updated reward function”; initial working tree was clean. Local history and the code show Alex implemented Two DQN, worker-cap integration, a common reward evaluation, baseline retuning, and the 30-checkpoint shared retraining/selection pipeline (`agent/retrain_all.py`). Existing comparison figures already show basic reward/resource tradeoffs.
- Jas implemented PPO and queue tuning historically and the Ray Serve worker, external scaling client, live evaluation and comparison scripts. The checkout contains only three live summaries: two fixed runs (30 s and 300 s) and one 30 s queue run, all on `flash_crowds_01`, max four workers. There are no saved shared-checkpoint live results or held-out test results here. Jas's ongoing validation/test work described by the group is not claimed completed from these files.
- Your prior contribution is the four DQN variants and their evaluation. This analysis adds selection under two explicit objectives, 30-checkpoint/five-seed comparisons, 55-baseline-candidate verification, Pareto/SLA/pattern analysis, presentation figures, and analytical tests. No retraining or live experiment was performed.
- Candidate coverage: **30 RL checkpoints**, six methods × seeds 0–4; **55 baseline settings** (10 fixed, 9 predictive, 36 queue). **85 of 85 candidates** have complete workload evidence (**1,700 rows**). All 52 formerly aggregate-only baseline configurations have workload evidence from bounded simulator evaluation/cache; totals match saved tuning results within 1e-6. No other historical experiment is pooled with this shared run.
- Training uses 30 schedules, validation 20 (15 standard, 5 stress; four per pattern), test 10. Each is 300 s, 10 FPS/camera, 1 s decisions. Demand signatures have no overlap across splits; all candidate workload identities and arrival totals agree with definitions. Test definitions were inspected for separation only; no test policy scoring was run. Total validation exposure is 6,000 s and 175,650 arrived frames per candidate.

## Verified configuration and scoring

Max allocations 10; minimum serving workers 1; deadline 1 s; waiting queue capacity 50 frames; tick 0.001 s. Calibration mean service is 0.07197132544888889 s, rounded to **0.072 s**; mean startup 2.3057613082 s, rounded to **2.306 s**. Each ready worker therefore has a reference capacity of 13.8889 FPS. The simulator uses constant calibrated means, not samples from measured service/startup distributions. Camera demand is exogenous; policies control workers.

The reward actually used by all six current training paths and shared evaluation is `on_time - 2*(late + dropped_deadline + dropped_full) - worker_seconds - 0.25*applied_scaling_changes`. Four variants receive these weights explicitly; PPO uses `make_env` defaults; Two DQN overrides only worker cost, which is 1 here. `experiment.json` and all 30 metadata files agree. Saved ZIP counters confirm **61,440 environment steps per run** for ordinary DQN/PPO. Two DQN saves A with 30,720 steps; the loop gives B another 30,720, totaling 61,440. This equalizes environment interactions, not gradient-update counts or compute: Two DQN's default batch is 32, variants 64, and PPO has a different optimizer/update schedule.

All controllers use the same shared evaluator. Adaptive policies start with **one ready worker**; tuned fixed starts with **7 ready workers**, with no initial startup charge. Treat this as a warm-start convention to revisit, not equal startup provisioning. PPO evaluation correctly uses PPO in `retrain_all.py`; standalone `evaluate_dqn.py` is a DQN loader. Current observation demand is instantaneous scheduled demand, not the “recent arrival rate” described in older `docs/rl_design.md`.

## Reward-selected comparison

| Method | Weighted on-time % | Workload mean % | Worst workload % | Worker-s | Reward | Scales |
| --- | --- | --- | --- | --- | --- | --- |
| DQN | 88.48 | 86.25 | 50.64 | 20,754.16 | 93,567.34 | 2942 |
| N-step DQN | 90.35 | 88.44 | 60.98 | 21,534.19 | 102,810.31 | 2030 |
| Dueling DQN | 91.24 | 89.64 | 64.31 | 21,730.56 | 107,218.94 | 2254 |
| Dueling N-step DQN | 95.18 | 94.62 | 74.73 | 28,958.13 | 120,437.87 | 3596 |
| Two DQN | 91.26 | 89.60 | 63.07 | 23,753.41 | 104,859.59 | 4108 |
| PPO | 98.78 | 98.87 | 93.41 | 52,275.00 | 116,888.00 | 160 |
| Fixed | 99.43 | 99.44 | 89.47 | 42,000.00 | 130,767.00 | 0 |
| Predictive | 94.69 | 94.18 | 74.63 | 24,822.00 | 122,657.75 | 1149 |
| Queue threshold | 96.63 | 96.34 | 80.74 | 29,786.00 | 127,890.25 | 903 |

Dueling N-step seed 0's worst workload is `multi_burst_04`; it fails 10 of 20 workloads at 97%. Best RL versus each reward-tuned baseline (positive savings/reduction is favorable; negative means worse):

| Baseline | Worker-s savings % | On-time difference pp | Non-on-time reduction % | Reward difference | P95 difference s |
| --- | --- | --- | --- | --- | --- |
| fixed | 31.052 | -4.254 | -747.948 | -10,329.130 | 0.330 |
| predictive | -16.663 | 0.486 | 9.158 | -2,219.880 | 0.008 |
| queue | 2.779 | -1.452 | -43.091 | -7,452.380 | 0.222 |

The best RL reward exceeds 0 of the three tuned baseline rewards. Check the signed table before claiming an efficiency win: better compliance achieved by allocating more workers is a tradeoff. PPO's reward-selected policy uses 24.46% more worker-seconds than fixed, with compliance difference -0.655 pp. fixed__workers=8 meets the proposed SLA with 19.03% fewer worker-seconds than ppo_seed2. No universal RL superiority is supported; cheaper DQN policies occupy lower-resource/lower-compliance tradeoffs.

## Seed variability and DQN interpretation

| Method | Weighted on-time mean % | Sample SD pp | Reward mean | Reward sample SD |
| --- | --- | --- | --- | --- |
| DQN | 87.64 | 1.09 | 88,850.65 | 5,918.07 |
| N-step DQN | 89.10 | 0.72 | 95,432.97 | 4,304.35 |
| Dueling DQN | 89.14 | 1.17 | 96,624.69 | 5,939.00 |
| Dueling N-step DQN | 91.29 | 3.50 | 104,774.90 | 14,070.22 |
| Two DQN | 89.94 | 1.37 | 98,575.26 | 4,825.98 |
| PPO | 93.77 | 9.52 | 94,940.05 | 34,940.30 |

N-step DQN improves weighted on-time over the same-label vanilla seed in 5/5 cases and reward in 4/5; Dueling DQN improves weighted on-time over the same-label vanilla seed in 5/5 cases and reward in 4/5; Dueling N-step DQN improves weighted on-time over the same-label vanilla seed in 5/5 cases and reward in 5/5. These are descriptive paired seed labels, not a statistical significance test. Mean compliance differences from vanilla: N-step DQN +1.46 pp; Dueling DQN +1.50 pp; Dueling N-step DQN +3.64 pp. The combination's seed spread means its best checkpoint is not representative of every run. PPO ranges from 77.06% (ppo_seed3) to 99.44% (ppo_seed2). Baselines are deterministic single evaluations per configuration; no five-trial baseline uncertainty is implied.

## SLA sensitivity and Pareto scope

| Scope | Threshold % | Eligible | Minimum-resource candidate | Worker-s |
| --- | --- | --- | --- | --- |
| ALL | 90 | 6 | fixed__workers=8 | 48,000.00 |
| ALL | 95 | 4 | fixed__workers=8 | 48,000.00 |
| ALL | 97 | 4 | fixed__workers=8 | 48,000.00 |
| ALL | 99 | 2 | fixed__workers=9 | 54,000.00 |
| RL_ONLY | 90 | 3 | ppo_seed4 | 52,275.00 |
| RL_ONLY | 95 | 1 | ppo_seed2 | 59,280.00 |
| RL_ONLY | 97 | 1 | ppo_seed2 | 59,280.00 |
| RL_ONLY | 99 | 0 | NO QUALIFYING MODEL | unavailable |

Highest achievable **minimum-workload** compliance per method (not a guarantee beyond these validation schedules):

| Method | Highest minimum % |
| --- | --- |
| vanilla | 53.433 |
| nstep | 60.981 |
| dueling | 64.307 |
| dueling_nstep | 74.727 |
| two_dqn | 64.307 |
| ppo | 97.674 |
| fixed | 100.000 |
| predictive | 75.899 |
| queue | 80.736 |

Closest candidates for infeasible methods, **not SLA-compliant**, use greatest minimum-workload compliance, then lowest worker-seconds, then lexical ID:

| Method | Closest candidate | Minimum % |
| --- | --- | --- |
| queue | queue__add_threshold=5_remove_threshold=1_scale_down_steps=10 | 80.736 |
| predictive | predictive__window=10_scale_down_steps=5 | 75.899 |
| dueling_nstep | dueling_nstep_seed4 | 74.727 |
| dueling | dueling_seed3 | 64.307 |
| two_dqn | two_dqn_seed0 | 64.307 |
| nstep | nstep_seed0 | 60.981 |
| vanilla | vanilla_seed0 | 53.433 |

Reward selection breaks ties by fewer worker-seconds then lexical ID; SLA selection breaks ties by higher reward then lexical ID. Comparisons are inclusive, without lowering thresholds. Pareto tables/figures cover **all 85** candidates for aggregate and minimum-workload compliance. Equal points do not dominate one another; worker-second differences below 1e-6 are numerical ties. Reward selections are starred; nondominated candidates are circled. Aggregate/minimum frontiers contain 28/21 candidates respectively; flags and identities are in `pareto.csv`.

RL aggregate-frontier candidates: dueling_seed3. RL minimum-workload-frontier candidates: none. For example, queue__add_threshold=10_remove_threshold=2_scale_down_steps=10 dominates the RL reward winner in aggregate compliance/resource space: 95.994% at 28,775.0 worker-seconds.

## Workload behavior and scaling

`workload_summary.csv` reports pattern and standard/stress groups for reward-selected policies. Multi-burst: Dueling N-step DQN 86.31% at 4.22 average workers; Queue threshold 96.37% at 4.98 average workers; Predictive 94.38% at 4.82 average workers. Mixed churn: Dueling N-step DQN 93.36% at 5.31 average workers; Queue threshold 90.87% at 4.78 average workers. Mass dropoff: Dueling N-step DQN 98.08% at 5.85 average workers; Queue threshold 97.71% at 5.53 average workers. Dueling N-step's standard/stress compliance is 96.19%/93.39%, versus queue's 97.34%/95.39%. Read compliance alongside resources before attributing gains to better decisions.

Applied scaling actions: DQN 2,942, N-step DQN 2,030, Dueling DQN 2,254, Dueling N-step DQN 3,596, Two DQN 4,108, PPO 160, Fixed 0, Predictive 1,149, Queue threshold 903. Starting/retiring allocations consume worker-seconds. Existing histories supply objectively selected high-demand flash-crowd and mass-dropoff examples (`figure_sources.json`). Ready/allocated counts are end-of-second samples. Demand divided by tick-calibrated capacity is a reference, not optimal provisioning; startup, backlog, deadlines and in-flight work matter. No artificial optimality/excess-capacity metric was constructed.

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

Suggested speaking points: “I checked every seed and all 55 baseline configurations. Our current reward winner is fixed__workers=7. My best variant is dueling_nstep_seed0, but it fails 10 workloads at 97%. The every-workload rule gives a different answer, so we should report both. The DQN extensions improve mean compliance, but burst workloads and scaling churn remain weaknesses. We need to freeze the objective and let Jas test the selected policies on held-out workloads.”

Recommended next experiments (not performed): a single held-out evaluation of frozen selections with Jas; matched warm/cold-start and cutoff conventions; full-duration live comparisons using shared checkpoints, cap ten and repeated measurements; only afterward consider reward/terminal-penalty or burst-response retraining. Avoid further validation tuning that is then described as test performance.

## Reproduce and review

From the repository root, using existing dependencies (choose a **new** output directory):

```powershell
python -m pytest tests -q
python -m agent.analyze_results --experiment results/shared_objective/run_20261126_112432_666216 --output results/analysis/reproduction --sla 97 --evaluate-baselines --baseline-workers 4
```

Reuse the bounded baseline results without simulator reruns:

```powershell
python -m agent.analyze_results --experiment results/shared_objective/run_20261126_112432_666216 --output results/analysis/reproduction_cached --sla 97 --baseline-cache results/analysis/shared_objective_audit/baseline_workloads.csv
```

Existing output directories are refused. `--sla` changes the selection/figure line and is included alongside 90/95/97/99 sensitivity. `--no-figures` supports tabular checks. Review this summary, `model_selection.csv`, `candidate_metrics.csv`, `baseline_comparisons.csv`, `seed_statistics.csv`, `workload_summary.csv` and the nine PNG figures. `verification.json` records source hashes and accounting checks; `checkpoint_audit.json` records saved budgets and model hashes. `workload_metrics.csv` is the consolidated analysis table; `baseline_workloads.csv` stores only newly evaluated workload metrics, not raw histories/models.

Transfer `agent/analyze_results.py`, `agent/analysis_figures.py` and `tests/test_analysis_results.py`, plus this dedicated output directory as needed. Existing teammate files are preserved. No stage, commit, push, remote access or GitHub access was performed.
