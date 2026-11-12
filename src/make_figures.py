"""Generate every figure from results/*.csv (and exogenous signals from the simulation code). Writes figures/*.pdf and *.png"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adt_sim as S
cfg = S.load_config(os.path.join(ROOT, "config.json"))
res, figd = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})
STYLE = {"thermostat": dict(color="#555555", ls="-.", label="Thermostat"),
         "fixed": dict(color="#1f77b4", ls="--", label="Fixed AI"),
         "gated": dict(color="#d62728", ls="-", label="Gated ADT"),
         "always": dict(color="#2ca02c", ls=":", label="Always-on RLS")}
SCN = {"baseline": "Baseline", "occupancy": "Occupancy", "outdoor": "Outdoor temp.", "combined": "Combined"}
WIN = cfg["metric_windows_h"]
ts = pd.read_csv(f"{res}/timeseries_all.csv")
sh = pd.read_csv(f"{res}/shadow_timeseries.csv")
runs = pd.read_csv(f"{res}/metrics_per_run.csv")
REP = 0  # representative seed

def save(fig, name):
    fig.savefig(f"{figd}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{figd}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def hours_axis(ax):
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Time of evaluation day (h)")

# ---- Fig 1: system architecture diagram ----
def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#33507a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3, linespacing=1.25)
def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#222", shrinkA=0, shrinkB=0))
fig, ax = plt.subplots(figsize=(7.1, 2.9)); ax.set_xlim(0, 10); ax.set_ylim(-0.25, 4.3); ax.axis("off"); ax.grid(False)
box(ax, 0.1, 2.3, 2.0, 1.2, "Simulated zone\n(RC plant, COP($T_{out}$),\nventilation, solar lag)", "#fdf2e3", "#8a5a14")
box(ax, 2.6, 2.3, 1.8, 1.2, "Noisy measurements\n$T_{meas}$, $T_{out}$, $N$, $G$")
box(ax, 4.9, 2.3, 2.0, 1.2, "Digital-twin predictor\n$\\hat{T}(t{+}1)=\\theta^{\\top}\\phi(t)$")
box(ax, 7.4, 2.3, 2.4, 1.2, "One-step predictive\ncontroller (17 power\nlevels, cost $J$)")
box(ax, 3.0, 0.35, 2.7, 1.2, "Error monitor\nrolling MAE of $e=T_{meas}-\\hat{T}$\ntrigger $\\tau_{on}$ / $\\tau_{off}$", "#eaf6ea", "#2e6b2e")
box(ax, 6.3, 0.35, 2.1, 1.2, "Bounded RLS update\n$\\theta\\leftarrow\\theta+\\mu K e$\n(only while triggered)", "#eaf6ea", "#2e6b2e")
box(ax, 0.1, 0.35, 2.3, 1.2, "Pre-set disturbances\n(occupancy $\\times$2, $T_{out}$ +5 °C)", "#f6eaf6", "#6b2e6b")
arr(ax, (2.1, 2.9), (2.6, 2.9)); arr(ax, (4.4, 2.9), (4.9, 2.9)); arr(ax, (6.9, 2.9), (7.4, 2.9))
ax.text(7.15, 3.05, "$\\hat{T}$", fontsize=6.3, ha="center")
# return path: controller -> zone (HVAC power)
ax.plot([8.6, 8.6, 1.1], [3.5, 3.95, 3.95], color="#222", lw=0.8); arr(ax, (1.1, 3.95), (1.1, 3.5))
ax.text(4.85, 4.03, "HVAC electrical power $P(t)\\leq 4$ kW", ha="center", fontsize=6.3)
arr(ax, (1.25, 1.55), (1.25, 2.3))                       # disturbances -> zone
arr(ax, (3.5, 2.3), (3.5, 1.55))                         # measurements -> monitor
arr(ax, (5.2, 2.3), (5.2, 1.55)); ax.text(5.32, 1.9, "$\\hat{T}$", fontsize=6.3)  # predictor -> monitor
arr(ax, (5.7, 0.95), (6.3, 0.95))                        # monitor -> RLS
arr(ax, (6.6, 1.55), (6.6, 2.3)); ax.text(6.72, 1.9, "$\\theta$ update", fontsize=6.3)  # RLS -> predictor
ax.text(5.0, -0.18, "Simulation only: no real building, IoT sensors or HVAC hardware is involved.", ha="center", fontsize=6.5, style="italic")
save(fig, "fig01_architecture")

# ---- Fig 2: exogenous inputs (seed REP, evaluation day) ----
exo = S.make_exogenous(cfg, REP); n0 = cfg["timeline"]["init_days"] * S.steps_per_day(cfg)
e0, e3 = S.apply_scenario(cfg, exo, "baseline"), S.apply_scenario(cfg, exo, "combined")
sl = slice(n0, n0 + 96); hh = exo["h"][sl]
fig, axs = plt.subplots(3, 1, figsize=(3.45, 4.1), sharex=True)
axs[0].plot(hh, e0["Tout"][sl], color="#1f77b4", label="Nominal"); axs[0].plot(hh, e3["Tout"][sl], color="#d62728", ls="--", label="+5 °C disturbance")
axs[0].set_ylabel("$T_{out}$ (°C)"); axs[0].legend(loc="upper left")
axs[1].step(hh, e0["N"][sl], where="post", color="#1f77b4", label="Nominal"); axs[1].step(hh, e3["N"][sl], where="post", color="#d62728", ls="--", label="$\\times$2, 12–16 h")
axs[1].set_ylabel("Occupants $N$"); axs[1].legend(loc="upper left")
axs[2].plot(hh, e0["G"][sl], color="#8a5a14"); axs[2].set_ylabel("Irradiance $G$ (W/m²)")
hours_axis(axs[2]); fig.tight_layout(); save(fig, "fig02_inputs")

# ---- Fig 3: indoor temperature, all controllers, 4 scenarios (seed REP) ----
fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.0), sharex=True, sharey=True)
for ax, sc in zip(axs.ravel(), SCN):
    d = ts[(ts.seed == REP) & (ts.scenario == sc)]
    ax.axvspan(*WIN[sc], color="#ffe9a8", alpha=0.5, lw=0)
    ax.axhspan(22, 26, color="#cfe8cf", alpha=0.35, lw=0)
    for c, st in STYLE.items():
        x = d[d.controller == c]
        ax.plot(x.hour, x.T_true, color=st["color"], ls=st["ls"], label=st["label"])
    ax.set_title(SCN[sc], fontsize=7.5); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
for ax in axs[1]: ax.set_xlabel("Time of evaluation day (h)")
for ax in axs[:, 0]: ax.set_ylabel("Indoor temperature (°C)")
axs[0, 0].set_ylim(20.8, 29.6); axs[0, 0].legend(loc="upper left", ncol=2)
fig.tight_layout(); save(fig, "fig03_indoor_temperature")

# ---- Fig 4: predicted vs simulated (combined, seed REP, fixed vs gated, closed loop) ----
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
d = ts[(ts.seed == REP) & (ts.scenario == "combined")]
dg, df_ = d[d.controller == "gated"], d[d.controller == "fixed"]
m = (dg.hour >= 10) & (dg.hour < 19)
axs[0].plot(dg.hour[m], dg.T_true_next[m], color="k", lw=1.3, label="Simulated (gated run)")
axs[0].plot(dg.hour[m], dg.T_pred_next[m], color=STYLE["gated"]["color"], ls="-", lw=0.8, label="Predicted, gated ADT")
mf = (df_.hour >= 10) & (df_.hour < 19)
axs[1].plot(df_.hour[mf], (df_.T_true_next - df_.T_pred_next)[mf], color=STYLE["fixed"]["color"], ls="--", label="Fixed AI")
axs[1].plot(dg.hour[m], (dg.T_true_next - dg.T_pred_next)[m], color=STYLE["gated"]["color"], label="Gated ADT")
axs[1].axhline(0, color="k", lw=0.5)
axs[0].set_ylabel("$T_{in}(t{+}1)$ (°C)"); axs[0].legend(loc="upper left")
axs[1].set_ylabel("Error $T-\\hat T$ (°C)"); axs[1].legend(loc="upper left", ncol=2)
axs[1].set_xlim(10, 19); axs[1].set_xlabel("Time of evaluation day (h)")
fig.tight_layout(); save(fig, "fig04_pred_vs_sim")

# ---- Fig 5: rolling MAE over time (shadow predictors, identical trajectory) + fraction updating ----
def rolling_abs_err(df, w=8):
    out = []
    for _, g in df.groupby("seed"):
        out.append((g.T_true_next - g.T_pred_next).abs().rolling(w, min_periods=w).mean().to_numpy())
    return np.array(out)
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for p in ["fixed", "gated", "always"]:
    g = sh[(sh.scenario == "combined") & (sh.predictor == p)]
    r = rolling_abs_err(g); mu, sd = np.nanmean(r, 0), np.nanstd(r, 0)
    hr = g[g.seed == 0].hour.to_numpy()
    axs[0].plot(hr, mu, color=STYLE[p]["color"], ls=STYLE[p]["ls"], label=STYLE[p]["label"])
    axs[0].fill_between(hr, mu - sd, mu + sd, color=STYLE[p]["color"], alpha=0.08, lw=0)
g = sh[(sh.scenario == "combined") & (sh.predictor == "gated")]
frac = g.groupby("k").adapt_flag.mean(); axs[1].step(g[g.seed == 0].hour, frac.to_numpy(), where="post", color=STYLE["gated"]["color"])
for ax in axs: ax.axvspan(*WIN["combined"], color="#ffe9a8", alpha=0.5, lw=0)
axs[0].set_ylabel("Rolling MAE, 2 h (°C)"); axs[0].legend(loc="upper left", ncol=3)
axs[1].set_ylabel("Seeds updating"); axs[1].set_ylim(0, 1.02); hours_axis(axs[1])
fig.tight_layout(); save(fig, "fig05_prediction_error")

# ---- Fig 6: HVAC power (combined, seed REP) ----
fig, ax = plt.subplots(figsize=(3.45, 2.4))
d = ts[(ts.seed == REP) & (ts.scenario == "combined")]
for c, st in STYLE.items():
    x = d[d.controller == c]
    ax.step(x.hour, x.P, where="post", color=st["color"], ls=st["ls"], label=st["label"], lw=0.9)
ax.axvspan(*WIN["combined"], color="#ffe9a8", alpha=0.5, lw=0); ax.axhline(4.0, color="k", lw=0.5)
ax.text(0.3, 4.08, "$P_{max}$", fontsize=6.5); ax.set_ylabel("HVAC electrical power (kW)"); ax.set_ylim(0, 5.5)
ax.legend(loc="upper left", ncol=2); hours_axis(ax); fig.tight_layout(); save(fig, "fig06_hvac_power")

# ---- Fig 7: summary bars ----
sm = runs.groupby(["scenario", "controller"]).agg(["mean", "std"])
def bars(ax, metric, ylabel, ctrls, title=None):
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        mu = [sm.loc[(s, c), (metric, "mean")] for s in SCN]; sd = [sm.loc[(s, c), (metric, "std")] for s in SCN]
        ax.bar(np.arange(4) + (i - (len(ctrls) - 1) / 2) * w, mu, w * 0.92, yerr=sd, capsize=1.5, color=STYLE[c]["color"], alpha=0.85,
               label=STYLE[c]["label"], error_kw=dict(lw=0.6))
    ax.set_xticks(range(4)); ax.set_xticklabels(list(SCN.values()), fontsize=6.5); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=7.5)
fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.2))
allc = ["thermostat", "fixed", "gated", "always"]; ai = ["fixed", "gated", "always"]
bars(axs[0, 0], "energy_kWh", "Energy per 24 h (kWh)", allc, "(a) HVAC electrical energy")
axs[0, 0].legend(loc="upper left", ncol=2)
bars(axs[0, 1], "comfort_viol_mean_C", "Mean violation (°C)", allc, "(b) Comfort violation, band 22–26 °C")
bars(axs[1, 0], "tight_viol_mean_C", "Mean deviation (°C)", allc, "(c) Deviation from 23–25 °C control band")
bars(axs[1, 1], "mae_true_C", "Prediction MAE (°C)", ai, "(d) One-step prediction MAE (vs. simulated $T$)")
fig.tight_layout(); save(fig, "fig07_summary_bars")

# ---- Fig 8: parameter evolution (closed-loop, combined, mean over seeds) ----
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True)
d = ts[ts.scenario == "combined"]
for p in ["fixed", "gated", "always"]:
    g = d[d.controller == p].groupby("k").agg(hour=("hour", "first"), e=("th_e", "mean"), c=("th_c", "mean"))
    axs[0].plot(g.hour, g.e, color=STYLE[p]["color"], ls=STYLE[p]["ls"], label=STYLE[p]["label"])
    axs[1].plot(g.hour, g.c, color=STYLE[p]["color"], ls=STYLE[p]["ls"])
for ax in axs: ax.axvspan(*WIN["combined"], color="#ffe9a8", alpha=0.5, lw=0)
axs[0].set_ylabel("$\\theta_e$ (°C/kW)"); axs[1].set_ylabel("$\\theta_c$ (°C per 10 occ.)"); axs[0].legend(loc="best"); hours_axis(axs[1])
fig.tight_layout(); save(fig, "fig08_parameter_evolution")
print("figures written:", sorted(f for f in os.listdir(figd) if f.endswith(".pdf")))
