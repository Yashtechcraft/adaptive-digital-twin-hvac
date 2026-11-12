"""
adt_sim.py - Simulation-based Adaptive Digital Twin (ADT) for single-zone HVAC cooling.

All physical/experimental parameters are read from config.json and are ASSUMED values
(no real building was measured). Units: temperatures in degC, power in kW (electrical)
or W (thermal heat flows), energy in kWh, time step 15 min.

Controllers
  thermostat : two-stage hysteresis thermostat (conventional baseline)
  fixed      : one-step predictive controller, ARX predictor frozen after initialisation
  gated      : same controller + prediction-error-gated bounded RLS update (the ADT)
  always     : same controller + RLS update at every step (ablation, no gating)

Information rule (no future leakage): at step t a controller sees only the noisy indoor
temperature T_meas(t), and the current T_out(t), occupancy N(t) and irradiance G(t). The
predictor update at step t+1 uses (regressors at t, measurement at t+1).
"""
import json
import time
from collections import deque

import numpy as np

STEP_S = 900.0  # fixed 15-min step in seconds (dt_min is validated against this)
THETA_NAMES = ["a", "b", "c", "d", "e", "f"]


# ----------------------------------------------------------------------------------
# configuration / exogenous signals
# ----------------------------------------------------------------------------------
def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    assert cfg["timeline"]["dt_min"] * 60.0 == STEP_S, "only 15-min steps are supported"
    return cfg


def steps_per_day(cfg):
    return int(24 * 60 / cfg["timeline"]["dt_min"])


def occupancy_fraction(h):
    """Schedule as fraction of peak occupancy (piecewise linear, weekday classroom)."""
    pts_h = [0.0, 7.999, 8.0, 9.0, 13.0, 13.01, 14.0, 14.01, 17.0, 18.0, 18.01, 24.0]
    pts_f = [0.0, 0.0, 1 / 3, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1 / 3, 0.0, 0.0]
    return np.interp(h, pts_h, pts_f)


def make_exogenous(cfg, seed):
    """Generate every random/exogenous array for one seed (shared by all controllers)."""
    rng = np.random.default_rng(seed)
    spd = steps_per_day(cfg)
    tl = cfg["timeline"]
    n_days = tl["init_days"] + tl["eval_days"]
    n = n_days * spd  # number of control steps; state/grid arrays have n+1 points
    idx = np.arange(n + 1)
    h = (idx * tl["dt_min"] / 60.0) % 24.0
    day = np.minimum(idx // spd, n_days - 1)
    w = cfg["weather"]

    day_off = rng.normal(0.0, w["Tmean_day_sd_C"], n_days)
    eps = rng.normal(0.0, w["ar_sigma_C"], n + 1)
    ar = np.zeros(n + 1)
    for i in range(1, n + 1):
        ar[i] = w["ar_phi"] * ar[i - 1] + eps[i]
    Tout = (w["Tmean_C"] + day_off[day]
            + w["amp_C"] * np.sin(2 * np.pi * (h - w["peak_hour"] + 6.0) / 24.0) + ar)

    clear = np.where((h >= 6.0) & (h <= 18.0),
                     w["G_peak_W_m2"] * np.sin(np.pi * (h - 6.0) / 12.0), 0.0)
    ceps = rng.normal(0.0, w["cloud_sigma"], n + 1)
    cx = np.zeros(n + 1)
    for i in range(1, n + 1):
        cx[i] = w["cloud_phi"] * cx[i - 1] + ceps[i]
    cloud = np.clip(w["cloud_mean"] + cx, w["cloud_min"], 1.0)
    G = clear * cloud

    o = cfg["occupancy"]
    day_factor = 1.0 + rng.normal(0.0, o["day_factor_sd"], n_days)
    N_base = o["peak_people"] * occupancy_fraction(h) * day_factor[day]

    proc = rng.normal(0.0, cfg["plant"]["process_noise_W"], n + 1)
    sens = rng.normal(0.0, cfg["sensor"]["T_noise_sd_C"], n + 1)
    dither_u = rng.random(n + 1)
    dither_i = rng.integers(0, 10 ** 6, n + 1)
    return dict(h=h, Tout=Tout, G=G, N=N_base.copy(), occ=(N_base > 0.0), w=proc, v=sens,
                dither_u=dither_u, dither_i=dither_i, seed=seed)


def apply_scenario(cfg, exo, scenario):
    """Return a copy of exo with the pre-specified disturbance applied to the evaluation day."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in exo.items()}
    sc = cfg["scenarios"][scenario]
    d = cfg["disturbance"]
    n_init = cfg["timeline"]["init_days"] * steps_per_day(cfg)
    idx = np.arange(len(exo["h"]))
    ev = idx >= n_init
    h = exo["h"]
    if sc["occ_mult"] != 1.0:
        win = ev & (h >= d["occ_window_h"][0]) & (h < d["occ_window_h"][1])
        out["N"][win] = out["N"][win] * sc["occ_mult"]
    if sc["dT_out_C"] != 0.0:
        k = d["dT_out_knots_h"]
        ramp = np.interp(h, k, [0.0, sc["dT_out_C"], sc["dT_out_C"], 0.0])
        out["Tout"][ev] = out["Tout"][ev] + ramp[ev]
    return out


def hvac_enabled(cfg, hour):
    return cfg["hvac"]["enable_start_h"] <= hour < cfg["hvac"]["enable_end_h"]


# ----------------------------------------------------------------------------------
# plant ("true" building) - simulated at 60 s sub-steps with zero-order-hold inputs
# ----------------------------------------------------------------------------------
class Plant:
    def __init__(self, cfg):
        self.p = cfg["plant"]
        self.T = self.p["T_init_C"]
        self.Qs = 0.0  # lagged solar gain, W

    def get_state(self):
        return (self.T, self.Qs)

    def set_state(self, s):
        self.T, self.Qs = s

    def cop(self, Tout):
        p = self.p
        return float(np.clip(p["cop_ref"] - p["cop_slope_per_K"] * (Tout - p["cop_ref_Tout_C"]),
                             p["cop_min"], p["cop_max"]))

    def step(self, Tout, G, N, occ, P_kW, w_W):
        p = self.p
        n_sub = int(round(STEP_S / p["substep_s"]))
        dts = p["substep_s"]
        cop = self.cop(Tout)
        qs_inst = p["solar_aperture_m2"] * G
        a_lag = dts / p["solar_lag_s"]
        T, Qs = self.T, self.Qs
        q_in = 0.0
        for _ in range(n_sub):
            Qs += (qs_inst - Qs) * a_lag
            q_env = (p["UA_env_W_per_K"] + p["k_vent_W_per_K_per_person"] * N) * (Tout - T)
            q_int = p["q_person_W"] * N + (p["equip_W_occupied"] if occ else 0.0)
            q_net = q_env + q_int + Qs + w_W - cop * P_kW * 1000.0
            T += q_net * dts / p["C_J_per_K"]
            q_in += (q_env + q_int + Qs + w_W) * dts
        self.T, self.Qs = T, Qs
        self.last_heat_in_J = q_in
        self.last_cooling_J = cop * P_kW * 1000.0 * STEP_S
        return T


# ----------------------------------------------------------------------------------
# conventional controller
# ----------------------------------------------------------------------------------
class Thermostat:
    def __init__(self, cfg):
        self.c = cfg["thermostat"]
        self.on = False

    def decide(self, T_meas, enabled):
        c = self.c
        if not enabled:
            self.on = False
            return 0.0
        if self.on and T_meas <= c["setpoint_C"] - c["deadband_C"]:
            self.on = False
        elif (not self.on) and T_meas >= c["setpoint_C"] + c["deadband_C"]:
            self.on = True
        if not self.on:
            return 0.0
        return c["stage2_kW"] if T_meas >= c["setpoint_C"] + c["stage2_offset_C"] else c["stage1_kW"]


# ----------------------------------------------------------------------------------
# predictor (ARX) and bounded RLS
# ----------------------------------------------------------------------------------
def features(T, Tout, N, G, P):
    """phi = [T, Tout, N/10, G/1000, P_kW, 1]  ->  T_hat(t+1) = theta . phi"""
    return np.array([T, Tout, N / 10.0, G / 1000.0, P, 1.0])


def fit_arx(X, y):
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ theta
    return theta, float(np.mean(np.abs(res))), X.T @ X


class BoundedRLS:
    """Recursive least squares with forgetting, step-size mu, error clipping,
    parameter bounds and a covariance-trace cap (protection against unstable updates)."""

    def __init__(self, theta0, P0, lam, mu, lo, hi, err_clip, trace_cap):
        self.theta = np.array(theta0, float)
        self.P = np.array(P0, float)
        self.lam, self.mu = lam, mu
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.err_clip = err_clip
        self.trace_cap = trace_cap

    def update(self, phi, y):
        e = float(np.clip(y - self.theta @ phi, -self.err_clip, self.err_clip))
        Pphi = self.P @ phi
        K = Pphi / (self.lam + phi @ Pphi)
        self.theta = self.theta + self.mu * K * e
        self.P = (self.P - np.outer(K, Pphi)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        tr = np.trace(self.P)
        if tr > self.trace_cap:
            self.P *= self.trace_cap / tr
        self.theta = np.clip(self.theta, self.lo, self.hi)


def initialise_predictor(cfg, X, y):
    """Batch least-squares fit on the initialisation window (shared by all AI controllers)."""
    theta, mae_ref, XtX = fit_arx(np.asarray(X), np.asarray(y))
    return dict(theta0=theta, mae_ref=mae_ref, XtX=XtX, n_train=len(y))


class PredictiveController:
    """One-step-ahead predictive controller. mode in {'fixed','gated','always'}."""

    def __init__(self, cfg, init_fit, mode, hp=None):
        self.cfg, self.mode = cfg, mode
        a = cfg["adapt"]
        hp = hp or (a.get(mode) if mode in ("gated", "always") else {})
        self.hp = hp
        lam = hp.get("lambda_f", 1.0)
        # covariance initialised so that the prior information equals the forgetting window
        n_eff = 1.0 / (1.0 - lam) if lam < 1.0 else init_fit["n_train"]
        P0 = np.linalg.inv(init_fit["XtX"]) * (init_fit["n_train"] / n_eff)
        lo = [a["bounds"][k][0] for k in THETA_NAMES]
        hi = [a["bounds"][k][1] for k in THETA_NAMES]
        self.rls = BoundedRLS(init_fit["theta0"], P0, lam, hp.get("mu", 0.0), lo, hi,
                              a["err_clip_C"], a["trace_cap_factor"] * np.trace(P0))
        h = cfg["hvac"]
        self.levels = np.arange(0.0, h["P_max_kW"] + 1e-9, h["P_step_kW"])
        pc = cfg["predictive"]
        self.band, self.w, self.lam_e = pc["control_band_C"], pc["w_comfort"], pc["lambda_energy_per_kW"]
        self.window = deque(maxlen=int(hp.get("window", 8)))
        self.tau_on = hp.get("tau_on_factor", np.inf) * init_fit["mae_ref"]
        self.tau_off = hp.get("tau_off_ratio", 0.75) * self.tau_on
        self.adapting = False
        self.n_updates = 0
        self.compute_time_s = 0.0
        self._pending = None

    def _predict_levels(self, T, Tout, N, G):
        th = self.rls.theta
        base = th[0] * T + th[1] * Tout + th[2] * N / 10.0 + th[3] * G / 1000.0 + th[5]
        return base + th[4] * self.levels

    def decide(self, T_meas, Tout, N, G, enabled):
        t0 = time.perf_counter()
        if enabled:
            cand = self._predict_levels(T_meas, Tout, N, G)
            lo, hi = self.band
            viol = np.maximum(0.0, cand - hi) ** 2 + np.maximum(0.0, lo - cand) ** 2
            J = self.w * viol + self.lam_e * self.levels
            P = float(self.levels[int(np.argmin(J))])
        else:
            P = 0.0
        self._register(T_meas, Tout, N, G, P)
        self.compute_time_s += time.perf_counter() - t0
        return P

    def shadow(self, T_meas, Tout, N, G, P_applied):
        """Predict only (used when the plant is driven by another controller)."""
        t0 = time.perf_counter()
        self._register(T_meas, Tout, N, G, P_applied)
        self.compute_time_s += time.perf_counter() - t0

    def _register(self, T_meas, Tout, N, G, P):
        phi = features(T_meas, Tout, N, G, P)
        self._pending = (phi, float(self.rls.theta @ phi))

    @property
    def pred_next(self):
        return self._pending[1]

    def observe(self, T_meas_next):
        """Called once the next measurement is available; may update the predictor."""
        t0 = time.perf_counter()
        phi, pred = self._pending
        e = T_meas_next - pred
        self.window.append(abs(e))
        do_update = False
        if self.mode == "always":
            do_update = True
        elif self.mode == "gated":
            if len(self.window) == self.window.maxlen:
                roll = float(np.mean(self.window))
                if (not self.adapting) and roll > self.tau_on:
                    self.adapting = True
                elif self.adapting and roll < self.tau_off:
                    self.adapting = False
            do_update = self.adapting
        if do_update:
            self.rls.update(phi, T_meas_next)
            self.n_updates += 1
        self.compute_time_s += time.perf_counter() - t0
        return do_update


# ----------------------------------------------------------------------------------
# experiment runners
# ----------------------------------------------------------------------------------
def run_init(cfg, exo):
    """Initialisation period (3 days): thermostat + random power dither for excitation."""
    plant, thermo = Plant(cfg), Thermostat(cfg)
    spd = steps_per_day(cfg)
    n_init = cfg["timeline"]["init_days"] * spd
    levels = np.arange(0.0, cfg["hvac"]["P_max_kW"] + 1e-9, cfg["hvac"]["P_step_kW"])
    X, y = [], []
    for t in range(n_init):
        enabled = hvac_enabled(cfg, exo["h"][t])
        T_meas = plant.T + exo["v"][t]
        P = thermo.decide(T_meas, enabled)
        if enabled and exo["dither_u"][t] < cfg["init"]["dither_prob"]:
            P = float(levels[exo["dither_i"][t] % len(levels)])
        Tn = plant.step(exo["Tout"][t], exo["G"][t], exo["N"][t], exo["occ"][t], P, exo["w"][t])
        X.append(features(T_meas, exo["Tout"][t], exo["N"][t], exo["G"][t], P))
        y.append(Tn + exo["v"][t + 1])
    fit = initialise_predictor(cfg, X, y)
    return dict(state=plant.get_state(), fit=fit, n_init=n_init)


def run_eval(cfg, exo, init, kind, hp=None, shadow_modes=None):
    """Run the 24-h evaluation day for one controller. Returns a dict of logged arrays."""
    spd = steps_per_day(cfg)
    n0 = init["n_init"]
    plant = Plant(cfg)
    plant.set_state(init["state"])
    thermo = Thermostat(cfg) if kind == "thermostat" else None
    ctrl = PredictiveController(cfg, init["fit"], kind, hp) if kind != "thermostat" else None
    shadows = {}
    if kind == "thermostat" and shadow_modes:
        shadows = {m: PredictiveController(cfg, init["fit"], m) for m in shadow_modes}

    keys = ["hour", "T_true", "T_meas", "Tout", "N", "G", "P", "T_pred_next", "T_true_next",
            "T_meas_next", "occ", "adapt_flag"] + ["th_" + k for k in THETA_NAMES]
    log = {k: np.full(spd, np.nan) for k in keys}
    for m in shadows:
        for k in ["T_pred_next", "adapt_flag"] + ["th_" + k for k in THETA_NAMES]:
            log[f"shadow_{m}_{k}"] = np.full(spd, np.nan)

    plant_time = 0.0
    for k in range(spd):
        t = n0 + k
        hour = exo["h"][t]
        enabled = hvac_enabled(cfg, hour)
        T_meas = plant.T + exo["v"][t]
        Tout, N, G, occ = exo["Tout"][t], exo["N"][t], exo["G"][t], exo["occ"][t]
        if kind == "thermostat":
            P = thermo.decide(T_meas, enabled)
            for s in shadows.values():
                s.shadow(T_meas, Tout, N, G, P)
        else:
            P = ctrl.decide(T_meas, Tout, N, G, enabled)
        log["hour"][k], log["T_true"][k], log["T_meas"][k] = hour, plant.T, T_meas
        log["Tout"][k], log["N"][k], log["G"][k], log["P"][k], log["occ"][k] = Tout, N, G, P, float(occ)
        if ctrl is not None:
            log["T_pred_next"][k] = ctrl.pred_next
        Tn = plant.step(Tout, G, N, occ, P, exo["w"][t])
        Tm_next = Tn + exo["v"][t + 1]
        log["T_true_next"][k] = Tn
        log["T_meas_next"][k] = Tm_next
        if ctrl is not None:
            upd = ctrl.observe(Tm_next)
            log["adapt_flag"][k] = float(upd)
            for i, nm in enumerate(THETA_NAMES):
                log["th_" + nm][k] = ctrl.rls.theta[i]
        for m, s in shadows.items():
            log[f"shadow_{m}_T_pred_next"][k] = s.pred_next
            upd = s.observe(Tm_next)
            log[f"shadow_{m}_adapt_flag"][k] = float(upd)
            for i, nm in enumerate(THETA_NAMES):
                log[f"shadow_{m}_th_{nm}"][k] = s.rls.theta[i]
    log["_ctrl_time_s"] = ctrl.compute_time_s if ctrl is not None else (
        0.0 if thermo is None else 0.0)
    log["_n_updates"] = ctrl.n_updates if ctrl is not None else 0
    log["_shadow_n_updates"] = {m: s.n_updates for m, s in shadows.items()}
    log["_shadow_time_s"] = {m: s.compute_time_s for m, s in shadows.items()}
    return log


# ----------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------
def _dist_outside(T, lo, hi):
    return np.maximum(0.0, T - hi) + np.maximum(0.0, lo - T)


def compute_metrics(cfg, log, scenario, pred_key="T_pred_next", flag_key="adapt_flag"):
    dt_h = cfg["timeline"]["dt_min"] / 60.0
    out = {}
    P = log["P"]
    out["energy_kWh"] = float(np.sum(P) * dt_h)
    out["peak_kW"] = float(np.max(P))
    occ = log["occ"].astype(bool)
    T = log["T_true"]
    lo, hi = cfg["comfort"]["band_C"]
    v = _dist_outside(T, lo, hi)[occ]
    out["comfort_viol_mean_C"] = float(np.mean(v)) if v.size else 0.0
    out["comfort_viol_max_C"] = float(np.max(v)) if v.size else 0.0
    out["comfort_viol_hours"] = float(np.sum(v > 0) * dt_h)
    tlo, thi = cfg["comfort"]["tight_band_C"]
    vt = _dist_outside(T, tlo, thi)[occ]
    out["tight_viol_mean_C"] = float(np.mean(vt)) if vt.size else 0.0
    out["occupied_hours"] = float(occ.sum() * dt_h)
    pred = log[pred_key]
    if not np.all(np.isnan(pred)):
        err = log["T_true_next"] - pred
        out["mae_true_C"] = float(np.mean(np.abs(err)))
        out["bias_true_C"] = float(np.mean(err))
        out["mae_meas_C"] = float(np.mean(np.abs(log["T_meas_next"] - pred)))
        t_on, t_off = cfg["metric_windows_h"][scenario]
        hr = log["hour"]
        for nm, m in (("pre", hr < t_on), ("during", (hr >= t_on) & (hr < t_off)), ("post", hr >= t_off)):
            out[f"mae_{nm}_C"] = float(np.mean(np.abs(err[m]))) if m.any() else float("nan")
        flag = log[flag_key]
        if not np.all(np.isnan(flag)):
            out["update_steps"] = int(np.nansum(flag))
            out["update_frac"] = float(np.nanmean(flag))
            during = (hr >= t_on) & (hr < t_off)
            idx = np.where((flag > 0) & (hr >= t_on))[0]
            out["trigger_delay_min"] = float((hr[idx[0]] - t_on) * 60.0) if idx.size else float("nan")
            out["false_trigger_frac_pre"] = float(np.mean(flag[hr < t_on] > 0)) if (hr < t_on).any() else float("nan")
    else:
        for k in ["mae_true_C", "bias_true_C", "mae_meas_C", "mae_pre_C", "mae_during_C", "mae_post_C",
                  "update_steps", "update_frac", "trigger_delay_min", "false_trigger_frac_pre"]:
            out[k] = float("nan")
    return out
