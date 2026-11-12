"""Summaries and paired statistics from results/*.csv -> results/metrics_summary.csv, paired_tests.csv, shadow_summary.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = os.path.join(ROOT, "results")
runs = pd.read_csv(f"{res}/metrics_per_run.csv")
shadow = pd.read_csv(f"{res}/shadow_metrics.csv")
scn_order = ["baseline", "occupancy", "outdoor", "combined"]
ctl_order = ["thermostat", "fixed", "gated", "always"]
metrics = ["energy_kWh", "comfort_viol_mean_C", "comfort_viol_max_C", "comfort_viol_hours", "tight_viol_mean_C",
           "mae_true_C", "mae_meas_C", "bias_true_C", "mae_pre_C", "mae_during_C", "mae_post_C", "peak_kW", "ctrl_time_ms",
           "update_steps", "update_frac", "trigger_delay_min", "false_trigger_frac_pre"]

def summarise(df, group):
    g = df.groupby(group)[metrics]
    out = g.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()

summ = summarise(runs, ["scenario", "controller"])
summ["scenario"] = pd.Categorical(summ["scenario"], scn_order, ordered=True)
summ["controller"] = pd.Categorical(summ["controller"], ctl_order, ordered=True)
summ = summ.sort_values(["scenario", "controller"])
summ.to_csv(f"{res}/metrics_summary.csv", index=False)

ssum = summarise(shadow.rename(columns={"predictor": "controller"}), ["scenario", "controller"])
ssum["scenario"] = pd.Categorical(ssum["scenario"], scn_order, ordered=True)
ssum = ssum.sort_values(["scenario", "controller"])
ssum.to_csv(f"{res}/shadow_summary.csv", index=False)

def paired(df, key, a, b, metric, scn):
    d = df[df.scenario == scn]
    x = d[d[key] == a].sort_values("seed")[metric].to_numpy()
    y = d[d[key] == b].sort_values("seed")[metric].to_numpy()
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    diff = x - y
    n = len(diff)
    if n < 2:
        return None
    se = diff.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=se) if se > 0 else (diff.mean(), diff.mean())
    try:
        p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(scenario=scn, comparison=f"{a} - {b}", metric=metric, n=n, mean_a=x.mean(), mean_b=y.mean(),
                mean_diff=diff.mean(), ci95_lo=ci[0], ci95_hi=ci[1], wilcoxon_p=p,
                a_lower_in=int((diff < 0).sum()), a_higher_in=int((diff > 0).sum()), ties=int((diff == 0).sum()))

rows = []
pairs = [("gated", "fixed"), ("always", "fixed"), ("gated", "always"), ("gated", "thermostat"), ("fixed", "thermostat")]
for scn in scn_order:
    for a, b in pairs:
        for m in ["energy_kWh", "comfort_viol_mean_C", "tight_viol_mean_C", "mae_true_C", "mae_during_C", "mae_post_C", "peak_kW"]:
            if "thermostat" in (a, b) and m.startswith("mae"):
                continue
            r = paired(runs, "controller", a, b, m, scn)
            if r:
                rows.append(r)
    for a, b in [("gated", "fixed"), ("always", "fixed"), ("gated", "always")]:
        for m in ["mae_true_C", "mae_during_C", "mae_post_C"]:
            r = paired(shadow, "predictor", a, b, m, scn)
            if r:
                r["comparison"] = "shadow: " + r["comparison"]
                rows.append(r)
pt = pd.DataFrame(rows)
# Holm-style caution: report raw p; flag family size in the manuscript (exploratory)
pt.to_csv(f"{res}/paired_tests.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = summ[["scenario", "controller", "energy_kWh_mean", "energy_kWh_std", "comfort_viol_mean_C_mean", "comfort_viol_max_C_mean",
             "tight_viol_mean_C_mean", "mae_true_C_mean", "mae_true_C_std", "peak_kW_mean", "update_frac_mean", "ctrl_time_ms_mean"]]
print(show.round(4).to_string(index=False))
print("\nSHADOW predictors (same trajectory = thermostat):")
print(ssum[["scenario", "controller", "mae_true_C_mean", "mae_true_C_std", "mae_pre_C_mean", "mae_during_C_mean", "mae_post_C_mean", "update_frac_mean", "trigger_delay_min_mean", "false_trigger_frac_pre_mean"]].round(4).to_string(index=False))
