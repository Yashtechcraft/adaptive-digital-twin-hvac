# Simulation-Based Adaptive Digital Twin for HVAC Energy Optimization (single zone)

Reproducible code, data and manuscript for the study *"Simulation-Based Adaptive Digital Twin for AI-Assisted HVAC Energy Optimization Under Changing Building Conditions: A Controlled Single-Zone Study"*.

**Scope warning.** Everything here is simulation. The "building" is a synthetic one-capacitance zone with *assumed* parameters. No real building, IoT sensor, measured data or HVAC hardware is involved, and no claim of real-building performance is made.

## Layout
| Path | Content |
|---|---|
| `config.json` | All model, controller and experiment parameters (assumed values) |
| `src/adt_sim.py` | Plant, thermostat, ARX predictor, bounded RLS, controllers, metrics |
| `src/run_experiments.py` | Main experiment (evaluation seeds 0-19) |
| `src/tune_hyperparameters.py` | Hyper-parameter selection on *development* seeds 1000-1009 |
| `src/analyze_results.py` | Summary statistics and paired tests |
| `src/make_figures.py`, `src/make_tables.py` | Figures; LaTeX tables and number macros |
| `tests/test_sim.py` | 17 validation tests (analytic response, units, metrics, RLS, leakage, determinism) |
| `results/` | CSV outputs (per-run metrics, time series, shadow-predictor results, paired tests, dev grid) |
| `figures/` | All figures (PDF + PNG) |
| `paper/` | IEEE two-column manuscript (`main.tex`, `main.pdf`) |
| `docs/` | Literature matrix, presentation explainer, limitations / unverified claims |

## Install and run
```bash
pip install -r requirements.txt
./run_all.sh            # tests -> experiment -> analysis -> figures -> tables -> PDF
```
Runtime is a few seconds for the experiment itself. `ctrl_time_ms` (wall-clock controller time) varies between machines and runs; every other output is deterministic for a given seed and `config.json`.

## Design and information rules
* 15-min steps; 3 initialisation days, then one 24-h evaluation day. Plant integrated at 60-s sub-steps.
* Controllers: two-stage thermostat; fixed ARX predictive controller; gated adaptive digital twin (bounded RLS, rolling-MAE trigger); always-on RLS (ablation).
* All AI controllers share the same initial parameters, cost, power levels and measurements; the thermostat sees only the indoor temperature.
* No future leakage: each decision uses only measurements and inputs at the current step (test `test_no_future_leakage`).
* Common random numbers: weather, occupancy, process and sensor noise are shared across controllers and scenarios within a seed.
* Scenarios (evaluation day only): baseline; occupancy x2 (12:00-16:00); outdoor +5 degC (ramp 11-13 h, hold to 17 h, ramp down to 19 h); combined.

## Change log (what was decided when)
1. Before any simulation: cooling limit set to 4 kW electrical (the earlier planning placeholder of about 6 kW thermal would have made the x2 occupancy case infeasible); disturbance shapes fixed; 22-26 degC evaluation band, 23-25 degC control band and the secondary "deviation from control band" metric fixed.
2. A fourth arm (always-on RLS) was added after the literature check showed that continuous/periodic updating already exists.
3. Plausibility check on development seeds 1000-1002, then hyper-parameter grid on development seeds 1000-1009. Starting values were kappa=2.0, lambda_f=0.98, mu=0.5; only kappa changed (to 1.5, the smallest grid value). The pre-tuning config is kept in `results/config_pretuning.json`; the grid in `results/dev_grid.csv`.
4. Evaluation seeds 0-19 were run once with the final configuration; no parameters were changed afterwards. Later edits were to figures and manuscript text only.

## Known caveats
See `docs/limitations_and_unverified_claims.md`. The key ones: assumed parameters, synthetic plant, coarse thermostat baseline with less information than the AI controllers, threshold factor at the edge of its grid, small absolute effect sizes, exploratory statistics (152 tests).

## AI-assistance disclosure
The code, analysis scripts and manuscript were prepared with the assistance of an AI model (Claude, Anthropic). Authors must verify everything and follow their institution's and venue's disclosure policy.
