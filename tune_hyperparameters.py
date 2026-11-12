"""Select adaptation hyper-parameters on DEVELOPMENT seeds only (disjoint from evaluation seeds).
Criterion: lowest mean closed-loop prediction MAE (vs true simulated temperature) over all four
scenarios and all development seeds; ties broken by lower energy. Both adaptive arms get a grid of
comparable size. Writes results/dev_grid.csv and updates the 'adapt' block of config.json."""
import itertools, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adt_sim as S

cfg_path = os.path.join(ROOT, "config.json")
cfg = S.load_config(cfg_path)
seeds = cfg["seeds"]["development"]
scns = list(cfg["scenarios"].keys())
cache = {}
for s in seeds:
    exo = S.make_exogenous(cfg, s)
    cache[s] = (exo, S.run_init(cfg, exo), {sc: S.apply_scenario(cfg, exo, sc) for sc in scns})

def evaluate(kind, hp):
    rows = []
    for s in seeds:
        exo, init, scn_exo = cache[s]
        for sc in scns:
            log = S.run_eval(cfg, scn_exo[sc], init, kind, hp)
            m = S.compute_metrics(cfg, log, sc)
            rows.append((m["mae_true_C"], m["energy_kWh"], m["comfort_viol_mean_C"]))
    a = np.array(rows)
    return a.mean(axis=0)

records = []
ref = evaluate("fixed", None)
records.append(dict(arm="fixed", tau_on_factor=np.nan, lambda_f=np.nan, mu=np.nan, mae=ref[0], energy=ref[1], viol=ref[2]))
for tau, lam, mu in itertools.product([1.5, 2.0, 3.0], [0.95, 0.98, 0.995], [0.3, 0.5, 1.0]):
    hp = dict(cfg["adapt"]["gated"], tau_on_factor=tau, lambda_f=lam, mu=mu)
    r = evaluate("gated", hp)
    records.append(dict(arm="gated", tau_on_factor=tau, lambda_f=lam, mu=mu, mae=r[0], energy=r[1], viol=r[2]))
for lam, mu in itertools.product([0.95, 0.98, 0.995], [0.3, 0.5, 1.0]):
    hp = dict(cfg["adapt"]["always"], lambda_f=lam, mu=mu)
    r = evaluate("always", hp)
    records.append(dict(arm="always", tau_on_factor=np.nan, lambda_f=lam, mu=mu, mae=r[0], energy=r[1], viol=r[2]))
df = pd.DataFrame(records)
df.to_csv(os.path.join(ROOT, "results", "dev_grid.csv"), index=False)
for arm in ("gated", "always"):
    d = df[df.arm == arm].sort_values(["mae", "energy"]).iloc[0]
    if arm == "gated":
        cfg["adapt"]["gated"].update(tau_on_factor=float(d.tau_on_factor), lambda_f=float(d.lambda_f), mu=float(d.mu))
    else:
        cfg["adapt"]["always"].update(lambda_f=float(d.lambda_f), mu=float(d.mu))
    print("selected", arm, dict(d))
print("fixed reference:", dict(df[df.arm == "fixed"].iloc[0]))
with open(cfg_path, "w") as fh:
    json.dump(cfg, fh, indent=2)
