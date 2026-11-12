"""Validation tests. Run:  python -m unittest discover -s tests -v   (from the project root)"""
import copy
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adt_sim as S  # noqa: E402

CFG = S.load_config(os.path.join(ROOT, "config.json"))


class TestPlant(unittest.TestCase):
    def test_free_decay_matches_analytic(self):
        """No gains, no cooling: first-order response T = Tout + (T0-Tout) exp(-t UA/C)."""
        p = S.Plant(CFG)
        p.T, p.Qs = 30.0, 0.0
        Tout = 25.0
        n = 40  # 10 h
        for _ in range(n):
            p.step(Tout, 0.0, 0.0, False, 0.0, 0.0)
        tau = CFG["plant"]["C_J_per_K"] / CFG["plant"]["UA_env_W_per_K"]
        analytic = Tout + (30.0 - Tout) * np.exp(-n * S.STEP_S / tau)
        self.assertAlmostEqual(p.T, analytic, delta=2e-3)

    def test_steady_state_with_cooling(self):
        p = S.Plant(CFG)
        p.T = 28.0
        Tout, P = 33.0, 1.5
        for _ in range(96 * 6):  # six days
            p.step(Tout, 0.0, 0.0, False, P, 0.0)
        ua = CFG["plant"]["UA_env_W_per_K"]
        t_ss = Tout - p.cop(Tout) * P * 1000.0 / ua
        self.assertAlmostEqual(p.T, t_ss, delta=5e-3)

    def test_energy_balance_one_step(self):
        p = S.Plant(CFG)
        p.T = 26.0
        T0 = p.T
        p.step(31.0, 500.0, 25.0, True, 2.0, 100.0)
        lhs = CFG["plant"]["C_J_per_K"] * (p.T - T0)
        rhs = p.last_heat_in_J - p.last_cooling_J
        self.assertAlmostEqual(lhs, rhs, delta=1e-3 * max(1.0, abs(rhs)))

    def test_cooling_lowers_temperature_and_cop_decreases_with_Tout(self):
        a, b = S.Plant(CFG), S.Plant(CFG)
        a.step(30.0, 0, 0, False, 0.0, 0.0)
        b.step(30.0, 0, 0, False, 2.0, 0.0)
        self.assertLess(b.T, a.T)
        self.assertGreater(a.cop(27.0), a.cop(40.0))


class TestMetrics(unittest.TestCase):
    def _log(self, T, P, occ, pred=None, tnext=None):
        n = len(T)
        hour = np.arange(n) * 0.25
        return dict(hour=hour, T_true=np.array(T, float), P=np.array(P, float), occ=np.array(occ, float),
                    T_pred_next=np.full(n, np.nan) if pred is None else np.array(pred, float),
                    T_true_next=np.full(n, np.nan) if tnext is None else np.array(tnext, float),
                    T_meas_next=np.full(n, np.nan) if tnext is None else np.array(tnext, float),
                    adapt_flag=np.full(n, np.nan))

    def test_energy_units(self):
        log = self._log([24] * 96, [2.0] * 96, [1] * 96)
        m = S.compute_metrics(CFG, log, "baseline")
        self.assertAlmostEqual(m["energy_kWh"], 2.0 * 24.0, places=9)  # 2 kW for 24 h
        self.assertAlmostEqual(m["peak_kW"], 2.0)

    def test_comfort_violation_definition(self):
        # band 22-26: temps 21 (0.0->1.0 below), 24 (0), 27.5 (1.5 above); only occupied counted
        log = self._log([21, 24, 27.5, 30], [0, 0, 0, 0], [1, 1, 1, 0])
        m = S.compute_metrics(CFG, log, "baseline")
        self.assertAlmostEqual(m["comfort_viol_mean_C"], (1.0 + 0.0 + 1.5) / 3.0)
        self.assertAlmostEqual(m["comfort_viol_max_C"], 1.5)
        self.assertAlmostEqual(m["comfort_viol_hours"], 0.5)

    def test_mae(self):
        log = self._log([24] * 4, [0] * 4, [1] * 4, pred=[24.0, 24.5, 23.0, 25.0], tnext=[24.1, 24.0, 24.0, 24.0])
        m = S.compute_metrics(CFG, log, "baseline")
        self.assertAlmostEqual(m["mae_true_C"], np.mean([0.1, 0.5, 1.0, 1.0]))


class TestThermostat(unittest.TestCase):
    def test_hysteresis_and_stages(self):
        th = S.Thermostat(CFG)
        self.assertEqual(th.decide(24.9, True), 0.0)       # below on-threshold (25.0)
        self.assertEqual(th.decide(25.1, True), 2.0)       # turns on, stage 1
        self.assertEqual(th.decide(24.0, True), 2.0)       # stays on inside deadband
        self.assertEqual(th.decide(25.6, True), 4.0)       # stage 2 above setpoint+1.5
        self.assertEqual(th.decide(22.9, True), 0.0)       # turns off at setpoint-deadband
        self.assertEqual(th.decide(26.0, False), 0.0)      # disabled outside schedule


class TestRLS(unittest.TestCase):
    def test_recovers_parameters(self):
        rng = np.random.default_rng(1)
        true = np.array([0.95, 0.04, 0.012, 0.3, -0.5, 0.1])
        X = np.column_stack([rng.normal(24, 1, 300), rng.normal(30, 3, 300), rng.uniform(0, 6, 300),
                             rng.uniform(0, 0.8, 300), rng.uniform(0, 4, 300), np.ones(300)])
        y = X @ true + rng.normal(0, 0.01, 300)
        th0, _, XtX = S.fit_arx(X, y)
        lo = [CFG["adapt"]["bounds"][k][0] for k in S.THETA_NAMES]
        hi = [CFG["adapt"]["bounds"][k][1] for k in S.THETA_NAMES]
        rls = S.BoundedRLS(th0 * 0 + np.array([0.9, 0.05, 0.02, 0.1, -0.2, 0.0]), np.eye(6) * 10, 0.99, 1.0, lo, hi, 5.0, 1e9)
        for _ in range(3):
            for i in range(300):
                rls.update(X[i], y[i])
        # compare predictions (parameters are collinear with the bias term, so compare outputs)
        self.assertLess(np.mean(np.abs(X @ rls.theta - y)), 0.03)

    def test_bounds_enforced(self):
        lo = [CFG["adapt"]["bounds"][k][0] for k in S.THETA_NAMES]
        hi = [CFG["adapt"]["bounds"][k][1] for k in S.THETA_NAMES]
        rls = S.BoundedRLS(np.array([0.95, 0.04, 0.01, 0.3, -0.5, 0.0]), np.eye(6), 0.98, 1.0, lo, hi, 1.0, 1e6)
        for _ in range(200):
            rls.update(np.array([24.0, 30.0, 3.0, 0.5, 2.0, 1.0]), 100.0)  # absurd target
        self.assertTrue(np.all(rls.theta <= np.array(hi) + 1e-12) and np.all(rls.theta >= np.array(lo) - 1e-12))


class TestExperimentIntegrity(unittest.TestCase):
    def setUp(self):
        self.exo = S.make_exogenous(CFG, 0)
        self.init = S.run_init(CFG, self.exo)

    def test_reproducible(self):
        e2 = S.make_exogenous(CFG, 0)
        i2 = S.run_init(CFG, e2)
        np.testing.assert_allclose(self.init["fit"]["theta0"], i2["fit"]["theta0"])
        a = S.run_eval(CFG, S.apply_scenario(CFG, self.exo, "combined"), self.init, "gated")
        b = S.run_eval(CFG, S.apply_scenario(CFG, e2, "combined"), i2, "gated")
        np.testing.assert_allclose(a["P"], b["P"])
        np.testing.assert_allclose(a["T_true_next"], b["T_true_next"])

    def test_no_future_leakage(self):
        """Changing exogenous data strictly AFTER step k* must not change decisions up to and including k*."""
        base = S.apply_scenario(CFG, self.exo, "combined")
        n0 = self.init["n_init"]
        kstar = 50
        mod = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        rng = np.random.default_rng(99)
        cut = n0 + kstar + 1
        mod["Tout"][cut:] += rng.normal(5, 3, len(mod["Tout"][cut:]))
        mod["G"][cut:] *= 0.3
        mod["N"][cut:] *= 2.0
        mod["w"][cut:] += rng.normal(0, 400, len(mod["w"][cut:]))
        mod["v"][cut:] += rng.normal(0, 0.5, len(mod["v"][cut:]))
        for kind in ["thermostat", "fixed", "gated", "always"]:
            a = S.run_eval(CFG, base, self.init, kind)
            b = S.run_eval(CFG, mod, self.init, kind)
            np.testing.assert_array_equal(a["P"][: kstar + 1], b["P"][: kstar + 1], err_msg=kind)
            if kind != "thermostat":
                np.testing.assert_array_equal(a["T_pred_next"][: kstar + 1], b["T_pred_next"][: kstar + 1], err_msg=kind)

    def test_same_initialisation_for_all_ai_controllers(self):
        c = {m: S.PredictiveController(CFG, self.init["fit"], m) for m in ("fixed", "gated", "always")}
        for m in ("gated", "always"):
            np.testing.assert_array_equal(c[m].rls.theta, c["fixed"].rls.theta)

    def test_fixed_parameters_never_change(self):
        log = S.run_eval(CFG, S.apply_scenario(CFG, self.exo, "combined"), self.init, "fixed")
        for k in S.THETA_NAMES:
            self.assertEqual(np.nanmax(log["th_" + k]), np.nanmin(log["th_" + k]))
        self.assertEqual(log["_n_updates"], 0)

    def test_power_constraints(self):
        for kind in ["thermostat", "fixed", "gated", "always"]:
            log = S.run_eval(CFG, S.apply_scenario(CFG, self.exo, "combined"), self.init, kind)
            self.assertTrue(np.all(log["P"] >= 0) and np.all(log["P"] <= CFG["hvac"]["P_max_kW"] + 1e-9))
            off = (log["hour"] < CFG["hvac"]["enable_start_h"]) | (log["hour"] >= CFG["hvac"]["enable_end_h"])
            self.assertTrue(np.all(log["P"][off] == 0.0))

    def test_identified_parameters_physically_plausible(self):
        th = self.init["fit"]["theta0"]
        a, b, c, d, e, f = th
        self.assertTrue(0.9 < a < 1.0 and 0 < b < 0.12 and c > 0 and d > 0 and e < 0, msg=str(th))

    def test_identical_exogenous_inputs_across_controllers(self):
        base = S.apply_scenario(CFG, self.exo, "outdoor")
        a = S.run_eval(CFG, base, self.init, "thermostat")
        b = S.run_eval(CFG, base, self.init, "fixed")
        for k in ["Tout", "N", "G", "hour"]:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertEqual(a["T_true"][0], b["T_true"][0])  # same initial plant state


if __name__ == "__main__":
    unittest.main()
