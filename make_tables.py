"""Create LaTeX tables and a macro file of every number quoted in the manuscript from results/*.csv."""
import os, json
import numpy as np, pandas as pd
def sg_(x):
    return x.replace("-", "\\ensuremath{-}")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res, pap = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper")
runs = pd.read_csv(f"{res}/metrics_per_run.csv"); sh = pd.read_csv(f"{res}/shadow_metrics.csv")
pt = pd.read_csv(f"{res}/paired_tests.csv")
cfg = json.load(open(os.path.join(ROOT, "config.json")))
SC = ["baseline", "occupancy", "outdoor", "combined"]; SCN = {"baseline": "Baseline", "occupancy": "Occupancy", "outdoor": "Outdoor", "combined": "Combined"}
CT = ["thermostat", "fixed", "gated", "always"]; CTN = {"thermostat": "Thermostat", "fixed": "Fixed AI", "gated": "Gated ADT", "always": "Always-on"}
SC_C = {"baseline": "Base", "occupancy": "Occ", "outdoor": "Out", "combined": "Comb"}; CT_C = {"thermostat": "Th", "fixed": "Fx", "gated": "Gt", "always": "Al"}
g = runs.groupby(["scenario", "controller"]); mean, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
def pm(sc, ct, m, d=2):
    mu, s = mean.loc[(sc, ct), m], sd.loc[(sc, ct), m]
    return "--" if np.isnan(mu) else f"{mu:.{d}f}$\\pm${s:.{d}f}"
# ---- summary table ----
L = ["\\begin{table*}[!t]", "\\renewcommand{\\arraystretch}{1.08}", "\\caption{Closed-loop results over 20 seeds (mean $\\pm$ SD). Energy is HVAC electrical energy over the 24-h evaluation day. "
     "Violation$_{22\\text{--}26}$ is the mean distance outside 22--26\\,$^\\circ$C over occupied steps; Dev.$_{23\\text{--}25}$ is the same quantity for the 23--25\\,$^\\circ$C control band. "
     "MAE is the one-step prediction error against the simulated (noise-free) temperature; ``--'' = no predictor. Controller time is wall-clock time of controller computations per simulated day on the authors' machine.}",
     "\\label{tab:summary}", "\\centering\\footnotesize", "\\begin{tabular}{llcccccc}", "\\hline",
     "Scenario & Controller & Energy (kWh) & Viol.$_{22\\text{--}26}$ ($^\\circ$C) & Dev.$_{23\\text{--}25}$ ($^\\circ$C) & MAE ($^\\circ$C) & Peak (kW) & Ctrl.\\ time (ms) \\\\", "\\hline"]
for sc in SC:
    for i, ct in enumerate(CT):
        first = f"\\multirow{{4}}{{*}}{{{SCN[sc]}}}" if i == 0 else ""
        L.append(f"{first} & {CTN[ct]} & {pm(sc,ct,'energy_kWh',2)} & {pm(sc,ct,'comfort_viol_mean_C',3)} & {pm(sc,ct,'tight_viol_mean_C',3)} & {pm(sc,ct,'mae_true_C',3)} & {pm(sc,ct,'peak_kW',2)} & {pm(sc,ct,'ctrl_time_ms',1) if ct!='thermostat' else '--'} \\\\")
    L.append("\\hline")
L += ["\\end{tabular}", "\\end{table*}"]
open(f"{pap}/tables/tab_summary.tex", "w").write("\n".join(L))
# ---- paired table ----
def cell(sc, comp, m):
    r = pt[(pt.scenario == sc) & (pt.comparison == comp) & (pt.metric == m)].iloc[0]
    mark = "$^{\\dagger}$" if r.wilcoxon_p < 0.05 / len(pt) else ("$^{*}$" if r.wilcoxon_p < 0.05 else "")
    d = 3
    return sg_(f"{r.mean_diff:+.{d}f} [{r.ci95_lo:+.{d}f}, {r.ci95_hi:+.{d}f}]") + mark
L = ["\\begin{table*}[!t]", "\\renewcommand{\\arraystretch}{1.08}", f"\\caption{{Paired differences across the 20 seeds (first minus second controller): mean difference with 95\\% $t$-interval. "
     f"$^{{*}}$: unadjusted Wilcoxon signed-rank $p<0.05$; $^{{\\dagger}}$: $p<0.05/{len(pt)}$ (Bonferroni over all {len(pt)} tests computed). Tests are exploratory. "
     "Negative MAE or deviation differences favour the first controller; negative energy differences mean less energy.}", "\\label{tab:paired}", "\\centering\\footnotesize",
     "\\begin{tabular}{llcccc}", "\\hline", "Scenario & Comparison & $\\Delta$MAE ($^\\circ$C) & $\\Delta$MAE in window ($^\\circ$C) & $\\Delta$Energy (kWh) & $\\Delta$Dev.$_{23\\text{--}25}$ ($^\\circ$C) \\\\", "\\hline"]
for sc in SC:
    for i, (comp, nm) in enumerate([("gated - fixed", "Gated $-$ Fixed"), ("always - fixed", "Always $-$ Fixed"), ("gated - always", "Gated $-$ Always")]):
        first = f"\\multirow{{3}}{{*}}{{{SCN[sc]}}}" if i == 0 else ""
        L.append(f"{first} & {nm} & {cell(sc,comp,'mae_true_C')} & {cell(sc,comp,'mae_during_C')} & {cell(sc,comp,'energy_kWh')} & {cell(sc,comp,'tight_viol_mean_C')} \\\\")
    L.append("\\hline")
L += ["\\end{tabular}", "\\end{table*}"]
open(f"{pap}/tables/tab_paired.tex", "w").write("\n".join(L))
# ---- shadow table ----
sg = sh.groupby(["scenario", "predictor"]).mean(numeric_only=True)
L = ["\\begin{table}[!t]", "\\renewcommand{\\arraystretch}{1.05}", "\\caption{Predictor-only comparison on the \\emph{same} thermostat-driven trajectory (mean over 20 seeds). MAE in $^\\circ$C by period relative to the disturbance window; "
     "Upd.\\ = share of steps with a parameter update; Delay = mean time from window start to the first update (min), averaged over seeds with an update.}",
     "\\label{tab:shadow}", "\\centering\\scriptsize", "\\begin{tabular}{llccccc}", "\\hline", "Scenario & Predictor & MAE & Pre & Window & Upd.\\ (\\%) & Delay \\\\", "\\hline"]
for sc in SC:
    for i, p in enumerate(["fixed", "gated", "always"]):
        r = sg.loc[(sc, p)]
        first = f"\\multirow{{3}}{{*}}{{{SCN[sc]}}}" if i == 0 else ""
        dl = "--" if (np.isnan(r.trigger_delay_min) or p != "gated") else f"{r.trigger_delay_min:.0f}"
        L.append(f"{first} & {CTN[p]} & {r.mae_true_C:.3f} & {r.mae_pre_C:.3f} & {r.mae_during_C:.3f} & {100*r.update_frac:.0f} & {dl} \\\\")
    L.append("\\hline")
L += ["\\end{tabular}", "\\end{table}"]
open(f"{pap}/tables/tab_shadow.tex", "w").write("\n".join(L))
# ---- macros ----
M = []
def mac(name, val): M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
for sc in SC:
    for ct in CT:
        for key, m, d in [("Energy", "energy_kWh", 2), ("Viol", "comfort_viol_mean_C", 3), ("Tight", "tight_viol_mean_C", 3), ("Mae", "mae_true_C", 3),
                          ("MaeDur", "mae_during_C", 3), ("Peak", "peak_kW", 2), ("Time", "ctrl_time_ms", 1), ("Upd", "update_frac", 2), ("VMax", "comfort_viol_max_C", 2)]:
            v = mean.loc[(sc, ct), m]
            if not np.isnan(v):
                mac(f"{key}{SC_C[sc]}{CT_C[ct]}", f"{(100*v if key=='Upd' else v):.{0 if key=='Upd' else d}f}")
for sc in SC:
    for p in ["fixed", "gated", "always"]:
        r = sg.loc[(sc, p)]
        mac(f"ShMae{SC_C[sc]}{CT_C[p]}", f"{r.mae_true_C:.3f}"); mac(f"ShMaeDur{SC_C[sc]}{CT_C[p]}", f"{r.mae_during_C:.3f}")
        if p == "gated":
            mac(f"ShDelay{SC_C[sc]}", f"{r.trigger_delay_min:.0f}"); mac(f"ShFalse{SC_C[sc]}", f"{100*r.false_trigger_frac_pre:.0f}")
            mac(f"ShUpd{SC_C[sc]}", f"{100*r.update_frac:.0f}")
# percentage changes (closed loop means)
def pct(sc, a, b, m): return 100 * (mean.loc[(sc, a), m] - mean.loc[(sc, b), m]) / mean.loc[(sc, b), m]
def mac_s(name, val): mac(name, sg_(val))
for sc in SC:
    mac_s(f"PctEnergyGtFx{SC_C[sc]}", f"{pct(sc,'gated','fixed','energy_kWh'):+.1f}")
    mac_s(f"PctEnergyFxTh{SC_C[sc]}", f"{pct(sc,'fixed','thermostat','energy_kWh'):+.1f}")
    mac_s(f"PctMaeGtFx{SC_C[sc]}", f"{pct(sc,'gated','fixed','mae_true_C'):+.1f}")
    mac_s(f"PctMaeDurGtFx{SC_C[sc]}", f"{pct(sc,'gated','fixed','mae_during_C'):+.1f}")
    mac_s(f"PctTightGtFx{SC_C[sc]}", f"{pct(sc,'gated','fixed','tight_viol_mean_C'):+.1f}")
    mac_s(f"PctTightAlFx{SC_C[sc]}", f"{pct(sc,'always','fixed','tight_viol_mean_C'):+.1f}")
for sc in SC:
    for comp, nm in [("gated - fixed", "GtFx"), ("always - fixed", "AlFx"), ("gated - always", "GtAl")]:
        for m, k in [("mae_true_C", "Mae"), ("energy_kWh", "Energy"), ("tight_viol_mean_C", "Tight"), ("mae_during_C", "MaeDur")]:
            r = pt[(pt.scenario == sc) & (pt.comparison == comp) & (pt.metric == m)].iloc[0]
            mac_s(f"D{k}{SC_C[sc]}{nm}", f"{r.mean_diff:+.3f}")
            p = r.wilcoxon_p
            mac(f"P{k}{SC_C[sc]}{nm}", "<0.001" if p < 0.001 else f"{p:.3f}")
            mac(f"W{k}{SC_C[sc]}{nm}", f"{int(r.a_lower_in)}")
init = pd.read_csv(f"{res}/init_fit.csv")
for k in ["a", "b", "c", "d", "e", "f"]:
    mac(f"ThetaInit{k.upper()}", f"{init['theta0_'+k].mean():.3f}"); mac(f"ThetaSd{k.upper()}", f"{init['theta0_'+k].std():.3f}")
mac("MaeRef", f"{init.mae_ref_C.mean():.3f}")
mac("NTests", str(len(pt)))
a = cfg["adapt"]
mac("TauOn", f"{a['gated']['tau_on_factor']}"); mac("TauOff", f"{a['gated']['tau_off_ratio']}"); mac("LamF", f"{a['gated']['lambda_f']}"); mac("MuLR", f"{a['gated']['mu']}")
open(f"{pap}/results_macros.tex", "w").write("\n".join(M) + "\n")
print(len(M), "macros written")
