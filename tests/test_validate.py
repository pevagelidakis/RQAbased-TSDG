import numpy as np
import pytest

from koopgen import validate as V


def test_mbb_indices_blocks_and_length():
    rng = np.random.default_rng(0)
    idx = V.mbb_indices(21, 10, rng)
    assert idx.size == 21
    assert np.all(np.diff(idx[:10]) == 1) and np.all(np.diff(idx[10:20]) == 1)
    assert idx.min() >= 0 and idx.max() <= 20


def test_holm():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    assert V.holm(p, 0.05).tolist() == [True, False, False, False]
    assert V.holm(np.array([0.01, 0.012]), 0.05).tolist() == [True, True]


def _ar1_window_stat(rng, n_windows, phi=0.6, shift=0.0, L=200, stride=20, total=None):
    """A scalar statistic (window mean) on overlapping windows of one AR(1) realisation."""
    total = total or (n_windows - 1) * stride + L
    e = rng.standard_normal(total)
    y = np.zeros(total)
    for t in range(1, total):
        y[t] = phi * y[t - 1] + e[t]
    y += shift
    return np.array([y[s:s + L].mean() for s in range(0, total - L + 1, stride)])[:, None]


def _iid_gen_stat(rng, M, phi=0.6, shift=0.0, L=200):
    out = []
    for _ in range(M):
        e = rng.standard_normal(L + 200)
        y = np.zeros(L + 200)
        for t in range(1, L + 200):
            y[t] = phi * y[t - 1] + e[t]
        out.append(y[200:].mean() + shift)
    return np.array(out)[:, None]


def test_decision_rule_size_and_power():
    """Under H0 (generator = true process) the rule should reject rarely (it is conservative by
    construction); a generator shifted by ~2 sampling SDs of the statistic should be rejected."""
    rng = np.random.default_rng(1)
    rej_null, rej_alt = 0, 0
    n_rep = 80
    for _ in range(n_rep):
        R = _ar1_window_stat(rng, 61)   # 1400 points = 7 window lengths (>= 6, as pre-registered)
        G0 = _iid_gen_stat(rng, 300)
        G1 = G0 + 0.35          # SD of a 200-point AR(1, 0.6) mean is ~0.177
        rows = V.compare({"mean": R}, {"null": {"mean": G0}, "alt": {"mean": G1}}, ["mean"],
                         block=10, n_null=300, q=0.95, nan_excess=0.05, rng=rng)
        rej_null += not rows[0]["pass"]
        rej_alt += not rows[1]["pass"]
    assert rej_null / n_rep <= 0.10
    assert rej_alt / n_rep >= 0.5


def test_vector_discrepancy_scaled_mean():
    R = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    G = R + np.array([1.0, 10.0])
    sc = V.scales_from_real(R)
    # W1 of a pure shift equals the shift; scaled by the component SD both are 1/sd = same
    assert V._component_w1(R, G, sc) == pytest.approx(1.0 / np.std([0, 1, 2]))


def _rows(passes: dict):
    return [{"statistic": s, "generator": g, "pass": p} for (s, g), p in passes.items()]


def _cfg():
    return {"statistics": {"linear": ["acf", "mean"], "nonlinear": ["d2", "lle"]},
            "controls": {"shuffle_must_fail": ["acf"], "iaaft_must_fail_min_nonlinear": 1},
            "decision": {"max_divergence_rate": 0.05}}


def _passes(koop, shuffle_acf=False, iaaft_nl=(False, True)):
    p = {}
    for s in ["acf", "mean", "d2", "lle"]:
        p[(s, "koopman")] = koop.get(s, True)
        p[(s, "shuffle")] = (shuffle_acf if s == "acf" else True)
        p[(s, "iaaft")] = True
    p[("d2", "iaaft")], p[("lle", "iaaft")] = iaaft_nl
    return p


def test_verdict_logic():
    pre = {"nonlinear_detected": True}
    v = V.controls_and_verdict(_rows(_passes({})), pre, _cfg(), True, 0.0)
    assert v["verdict"] == "PRESERVED"
    v = V.controls_and_verdict(_rows(_passes({"d2": False})), pre, _cfg(), True, 0.0)
    assert v["verdict"] == "PARTIALLY PRESERVED" and v["failing_statistics"] == ["d2"]
    v = V.controls_and_verdict(_rows(_passes({"d2": False, "acf": False})), pre, _cfg(), True, 0.0)
    assert v["verdict"] == "NOT PRESERVED"
    v = V.controls_and_verdict(_rows(_passes({}, shuffle_acf=True)), pre, _cfg(), True, 0.0)
    assert v["verdict"] == "INCONCLUSIVE"
    v = V.controls_and_verdict(_rows(_passes({}, iaaft_nl=(True, True))), pre, _cfg(), True, 0.0)
    assert v["verdict"] == "INCONCLUSIVE"
    v = V.controls_and_verdict(_rows(_passes({})), pre, _cfg(), True, 0.06)
    assert v["verdict"] == "NOT PRESERVED"
    v = V.controls_and_verdict(_rows(_passes({})), pre, _cfg(), False, 0.0)
    assert v["verdict"] == "NOT PRESERVED"
    # no nonlinearity: nonlinear stats are not gated and IAAFT is not required to fail
    v = V.controls_and_verdict(_rows(_passes({"d2": False}, iaaft_nl=(True, True))),
                               {"nonlinear_detected": False}, _cfg(), True, 0.0)
    assert v["verdict"] == "PRESERVED" and v["no_nonlinear_note"]
