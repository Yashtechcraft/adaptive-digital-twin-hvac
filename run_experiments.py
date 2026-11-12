"""Main experiment: 20 evaluation seeds x 4 scenarios x 4 controllers (+ shadow predictors on the thermostat trajectory).
Outputs (results/): metrics_per_run.csv, timeseries_all.csv, shadow_metrics.csv, shadow_timeseries.csv, init_fit.csv"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adt_sim as S

cfg = S.load_config(os.path.join(ROOT, "config.json"))
seeds = cfg["seeds"]["evaluation"]
scns = list(cfg["scenarios"].keys())
kinds = ["thermostat", "fixed", "gated", "always"]
shadow_modes = ["fixed", "gated", "always"]
log_cols = ["hour", "T_true", "T_meas", "Tout", "N", "G", "P", "T_pred_next", "T_true_next", "T_meas_next", "occ",
            "adapt_flag"] + ["th_" + k for k in S.THETA_NAMES]

runs, ts, shm, shts, fits = [], [], [], [], []
t_start = time.perf_counter()
for seed in seeds:
    exo = S.make_exogenous(cfg, seed)
    init = S.run_init(cfg, exo)
    fits.append(dict(seed=seed, mae_ref_C=init["fit"]["mae_ref"],
                     **{f"theta0_{k}": v for k, v in zip(S.THETA_NAMES, init["fit"]["theta0"])}))
    for scn in scns:
        e = S.apply_scenario(cfg, exo, scn)
        for kind in kinds:
            log = S.run_eval(cfg, e, init, kind, shadow_modes=shadow_modes if kind == "thermostat" else None)
            m = S.compute_metrics(cfg, log, scn)
            m.update(seed=seed, scenario=scn, controller=kind, ctrl_time_ms=1000.0 * log["_ctrl_time_s"],
                     n_updates=log["_n_updates"])
            runs.append(m)
            df = pd.DataFrame({c: log[c] for c in log_cols})
            df.insert(0, "k", np.arange(len(df)))
            df.insert(0, "controller", kind); df.insert(0, "scenario", scn); df.insert(0, "seed", seed)
            ts.append(df)
            if kind == "thermostat":
                for sm in shadow_modes:
                    sl = dict(log)
                    ms = S.compute_metrics(cfg, sl, scn, pred_key=f"shadow_{sm}_T_pred_next", flag_key=f"shadow_{sm}_adapt_flag")
                    ms.update(seed=seed, scenario=scn, predictor=sm, ctrl_time_ms=1000.0 * log["_shadow_time_s"][sm],
                              n_updates=log["_shadow_n_updates"][sm])
                    shm.append(ms)
                    sd = pd.DataFrame({"hour": log["hour"], "T_true_next": log["T_true_next"], "T_meas_next": log["T_meas_next"],
                                       "T_pred_next": log[f"shadow_{sm}_T_pred_next"], "adapt_flag": log[f"shadow_{sm}_adapt_flag"],
                                       **{"th_" + k: log[f"shadow_{sm}_th_{k}"] for k in S.THETA_NAMES}})
                    sd.insert(0, "k", np.arange(len(sd)))
                    sd.insert(0, "predictor", sm); sd.insert(0, "scenario", scn); sd.insert(0, "seed", seed)
                    shts.append(sd)
res = os.path.join(ROOT, "results")
pd.DataFrame(runs).to_csv(f"{res}/metrics_per_run.csv", index=False)
pd.concat(ts).to_csv(f"{res}/timeseries_all.csv", index=False, float_format="%.5f")
pd.DataFrame(shm).to_csv(f"{res}/shadow_metrics.csv", index=False)
pd.concat(shts).to_csv(f"{res}/shadow_timeseries.csv", index=False, float_format="%.5f")
pd.DataFrame(fits).to_csv(f"{res}/init_fit.csv", index=False)
print(f"done: {len(runs)} closed-loop runs in {time.perf_counter()-t_start:.1f}s")
