"""Validation protocol: statistic battery on windows, block-bootstrap null, W1 decision rule,
posterior predictive p-values, IAAFT nonlinearity pre-test, negative controls and the verdict.

All thresholds come from config['preregistered'].
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from . import metrics as Mx
from .embedding import delay_embed

VECTOR_STATS = ("acf", "pacf", "psd", "quantiles", "time_reversal", "c3")


# ----------------------------------------------------------------------------- parameters
@dataclass
class MetricParams:
    m: int
    tau: int
    theiler_d2: int
    theiler_lle: int
    k_max_lle: int
    train_sorted: np.ndarray
    cfg: dict

    @classmethod
    def from_train(cls, y_train: np.ndarray, m: int, tau: int, cfg_metrics: dict):
        mp = Mx.mean_period(y_train)
        th = int(math.ceil(mp))
        k_max = max(int(cfg_metrics["lle"]["k_max_min"]),
                    int(math.ceil(cfg_metrics["lle"]["k_max_periods"] * mp)))
        return cls(m=m, tau=tau, theiler_d2=th, theiler_lle=th, k_max_lle=k_max,
                   train_sorted=np.sort(y_train), cfg=cfg_metrics)

    def as_dict(self):
        return {"m": self.m, "tau": self.tau, "theiler_d2": self.theiler_d2,
                "theiler_lle": self.theiler_lle, "k_max_lle": self.k_max_lle}


# ----------------------------------------------------------------------------- statistics
def nonlinear_stats(y: np.ndarray, P: MetricParams, curves: bool = False) -> dict:
    c = P.cfg
    X = np.ascontiguousarray(delay_embed(y, P.m, P.tau))
    D = Mx.pairwise_distances(X)
    R, eps = Mx.recurrence_plot(X, c["rqa"]["rr"], c["rqa"]["theiler"], D=D)
    q = Mx.rqa_from_rp(R, c["rqa"]["theiler"], c["rqa"]["lmin"], c["rqa"]["vmin"])
    r, C = Mx.correlation_sum(X, P.theiler_d2, c["d2"]["n_r"], D=D)
    sel = (C >= c["d2"]["c_lo"]) & (C <= c["d2"]["c_hi"]) & (C > 0)
    d2 = float(np.polyfit(np.log(r[sel]), np.log(C[sel]), 1)[0]) if sel.sum() >= 4 else np.nan
    lle, lle_info = Mx.lyapunov_rosenstein(y, P.m, P.tau, P.theiler_lle, P.k_max_lle,
                                           c["lle"]["rise_lo"], c["lle"]["rise_hi"], return_curve=True)
    out = {
        "rqa_det": q["det"], "rqa_lam": q["lam"], "rqa_lmean": q["lmean"], "rqa_lmax": q["lmax"],
        "rqa_entr": q["entr"], "rqa_tt": q["tt"], "rqa_rte": q["rte"],
        "d2": d2, "lle": lle,
        "sampen": Mx.sample_entropy(y, c["sampen"]["m"], c["sampen"]["r_frac"]),
        "permen": Mx.permutation_entropy(y, c["permen"]["order"], c["permen"]["delay"]),
        "time_reversal": Mx.time_reversal_asymmetry(y, c["tra_lags"]),
        "c3": Mx.third_order_autocov(y, c["c3_lags"]),
    }
    if curves:
        fit = (float(r[sel][0]), float(r[sel][-1])) if sel.sum() >= 4 else (np.nan, np.nan)
        out["_curves"] = {"r": r, "C": C, "d2_fit": fit, "lle_k": lle_info["k"],
                          "lle_curve": lle_info["curve"], "lle_fit": lle_info["fit_range"],
                          "rp": R, "eps": eps}
    return out


def linear_stats(y: np.ndarray, P: MetricParams) -> dict:
    c = P.cfg
    _, lpsd = Mx.welch_log_psd(y, c["welch_nperseg"])
    kf = c["koopman_freq"]
    return {
        "acf": Mx.acf(y, c["acf_lags"]),
        "pacf": Mx.pacf(y, c["pacf_lags"]),
        "psd": lpsd,
        "mean": float(np.mean(y)),
        "variance": float(np.var(y)),
        "quantiles": np.quantile(y, c["quantiles"]),
        "ks_marginal": Mx.ks_distance(y, P.train_sorted),
        "koopman_freq": Mx.hankel_dmd_frequency(y, kf["rows"], kf["rank"], kf["f_min"]),
    }


def all_stats(y: np.ndarray, P: MetricParams) -> dict:
    out = linear_stats(y, P)
    out.update(nonlinear_stats(y, P))
    return {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in out.items()}


def _init_worker():
    import numba
    numba.set_num_threads(1)


def _stats_job(args):
    y, P, which = args
    if which == "nonlinear":
        d = nonlinear_stats(y, P)
        return {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in d.items()}
    return all_stats(y, P)


def batch_stats(windows: list[np.ndarray], P: MetricParams, which: str = "all",
                n_jobs: int | None = None) -> dict:
    """Statistics for many windows in parallel processes. Returns name -> (n_windows, d) array."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(np.ascontiguousarray(w, dtype=float), P, which) for w in windows]
    if n_jobs == 1 or len(jobs) < 4:
        res = [_stats_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
            res = list(ex.map(_stats_job, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    keys = res[0].keys()
    return {k: np.stack([r[k] for r in res]) for k in keys}


# ----------------------------------------------------------------------------- windows
def real_windows(y_test: np.ndarray, length: int, stride: int) -> list[np.ndarray]:
    starts = range(0, y_test.size - length + 1, stride)
    return [y_test[s: s + length] for s in starts]


def one_window_per_series(series: np.ndarray, length: int, rng: np.random.Generator):
    n, L = series.shape
    starts = rng.integers(0, L - length + 1, size=n)
    return [series[i, s: s + length] for i, s in enumerate(starts)]


# ----------------------------------------------------------------------------- decision rule
def _component_w1(R: np.ndarray, G: np.ndarray, scales: np.ndarray) -> float:
    vals = []
    for k in range(R.shape[1]):
        r, g = R[:, k], G[:, k]
        r, g = r[np.isfinite(r)], g[np.isfinite(g)]
        if r.size == 0 or g.size == 0:
            return np.nan
        vals.append(wasserstein_distance(r, g) / scales[k])
    return float(np.mean(vals))


def mbb_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def scales_from_real(R: np.ndarray) -> np.ndarray:
    sc = np.nanstd(R, axis=0)
    return np.where(np.isfinite(sc) & (sc > 0), sc, 1.0)


def null_distribution(R: np.ndarray, block: int, n_null: int, rng: np.random.Generator,
                      scales: np.ndarray) -> np.ndarray:
    """W1 between two independent moving-block-bootstrap resamples of the real windows, drawn
    from the two disjoint halves of the test set (windows are in time order). Drawing both
    resamples from the same windows makes them share blocks and shrinks the null: in simulation
    that variant rejected a perfect generator 13-53% of the time (see PLAN.md, section 3)."""
    n = R.shape[0]
    h = n // 2
    A, B = R[:h], R[n - h:]
    return np.array([_component_w1(A[mbb_indices(h, block, rng)], B[mbb_indices(h, block, rng)], scales)
                     for _ in range(n_null)])


def ppc_null(real_like: list[np.ndarray], G: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Secondary (not gated): W1 between window sets of generator stretches of test length and
    the generator's own windows - the exactly calibrated posterior-predictive null."""
    return np.array([_component_w1(Rs, G, scales) for Rs in real_like])


def ppp_value(R: np.ndarray, G: np.ndarray, scales: np.ndarray) -> float:
    """Posterior predictive p-value P_G(T >= T_real), averaged over real windows. Vector
    statistics are reduced to their scaled squared distance from the real-window mean."""
    if R.shape[1] == 1:
        g = G[:, 0][np.isfinite(G[:, 0])]
        r = R[:, 0][np.isfinite(R[:, 0])]
        if g.size == 0 or r.size == 0:
            return np.nan
        return float(np.mean([np.mean(g >= v) for v in r]))
    mu = np.nanmean(R, axis=0)
    dr = np.nanmean(((R - mu) / scales) ** 2, axis=1)
    dg = np.nanmean(((G - mu) / scales) ** 2, axis=1)
    return float(np.mean([np.mean(dg >= v) for v in dr]))


def compare(real: dict, gens: dict[str, dict], stats: list[str], block: int, n_null: int,
            q: float, nan_excess: float, rng: np.random.Generator) -> list[dict]:
    """Scorecard rows: for each statistic and generator, W1 discrepancy, null q-quantile, pass."""
    rows = []
    for st in stats:
        R = real[st]
        scales = scales_from_real(R)
        null = null_distribution(R, block, n_null, rng, scales)
        thr = float(np.nanquantile(null, q))
        nan_real = float(np.mean(~np.isfinite(R).all(axis=1)))
        for gname, G in gens.items():
            Gs = G[st]
            disc = _component_w1(R, Gs, scales)
            nan_gen = float(np.mean(~np.isfinite(Gs).all(axis=1)))
            nan_fail = (nan_gen - nan_real) > nan_excess
            passed = bool(np.isfinite(disc) and disc <= thr and not nan_fail)
            rows.append({"statistic": st, "generator": gname, "w1": disc, "null_q": thr,
                         "null_median": float(np.nanmedian(null)),
                         "ratio": disc / thr if thr > 0 else np.inf, "pass": passed,
                         "nan_rate_real": nan_real, "nan_rate_gen": nan_gen, "nan_fail": nan_fail,
                         "ppp": ppp_value(R, Gs, scales),
                         "real_mean": float(np.nanmean(R)), "gen_mean": float(np.nanmean(Gs))})
    return rows


# ----------------------------------------------------------------------------- pre-test
def holm(pvals: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    reject = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def nonlinearity_pretest(y_train: np.ndarray, P: MetricParams, cfg_pre: dict,
                         rng: np.random.Generator, log=print) -> dict:
    """Two-sided rank test of each nonlinear statistic (each component of vector statistics)
    against IAAFT surrogates of an end-point-matched train segment; Holm across all tests."""
    from .surrogates import endpoint_matched_segment, iaaft

    i0, i1 = endpoint_matched_segment(y_train, int(cfg_pre["max_length"]))
    seg = y_train[i0:i1]
    ns = int(cfg_pre["n_surrogates"])
    surr = iaaft(seg, ns, rng)
    log(f"  pre-test: segment [{i0}, {i1}) of train, {ns} IAAFT surrogates")
    obs = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in nonlinear_stats(seg, P).items()}
    S = batch_stats(list(surr), P, which="nonlinear")
    names, pv, tobs, smed = [], [], [], []
    for st in cfg_pre["statistics"]:
        for k in range(obs[st].size):
            t = obs[st][k]
            s = S[st][:, k]
            s = s[np.isfinite(s)]
            if not np.isfinite(t) or s.size < 10:
                p = 1.0
            else:
                p = min(1.0, 2 * min(np.sum(s >= t) + 1, np.sum(s <= t) + 1) / (s.size + 1))
            names.append(st if obs[st].size == 1 else f"{st}[{k}]")
            pv.append(p)
            tobs.append(float(t))
            smed.append(float(np.median(s)) if s.size else np.nan)
    pv = np.array(pv)
    rej = holm(pv, float(cfg_pre["alpha"]))
    return {"segment": [int(i0), int(i1)], "n_surrogates": ns,
            "tests": [{"test": n, "observed": o, "surrogate_median": m, "p": float(p), "reject": bool(r)}
                      for n, o, m, p, r in zip(names, tobs, smed, pv, rej)],
            "nonlinear_detected": bool(rej.any()), "n_rejections": int(rej.sum()),
            "surrogate_stats": S, "observed_stats": obs}


# ----------------------------------------------------------------------------- verdict
def controls_and_verdict(rows: list[dict], pre: dict, cfg_pr: dict, gates_passed: bool,
                         divergence_rate: float, method: str = "koopman") -> dict:
    lin = list(cfg_pr["statistics"]["linear"])
    nonlin = list(cfg_pr["statistics"]["nonlinear"])
    res = {(r["statistic"], r["generator"]): r["pass"] for r in rows}
    nl_detected = pre["nonlinear_detected"]
    shuffle_fail = {s: not res[(s, "shuffle")] for s in cfg_pr["controls"]["shuffle_must_fail"]}
    iaaft_fail = [s for s in nonlin if not res[(s, "iaaft")]]
    ctrl = {"shuffle_fails_required": shuffle_fail,
            "shuffle_ok": all(shuffle_fail.values()),
            "iaaft_failed_nonlinear": iaaft_fail,
            "iaaft_ok": (len(iaaft_fail) >= cfg_pr["controls"]["iaaft_must_fail_min_nonlinear"])
            if nl_detected else True,
            "iaaft_required": nl_detected}
    ctrl["controls_ok"] = ctrl["shuffle_ok"] and ctrl["iaaft_ok"]
    gated = lin + (nonlin if nl_detected else [])
    failing = [s for s in gated if not res[(s, method)]]
    lin_ok = all(res[(s, method)] for s in lin)
    nl_ok = all(res[(s, method)] for s in nonlin) if nl_detected else None
    div_ok = divergence_rate <= cfg_pr["decision"]["max_divergence_rate"]
    reasons = []
    if not gates_passed:
        reasons.append("MCMC convergence gates failed after remediation")
    if not div_ok:
        reasons.append(f"divergence rate {divergence_rate:.1%} > "
                       f"{cfg_pr['decision']['max_divergence_rate']:.0%}")
    if reasons:
        verdict = "NOT PRESERVED"
    elif not ctrl["controls_ok"]:
        verdict = "INCONCLUSIVE"
        if not ctrl["shuffle_ok"]:
            reasons.append("shuffle control passed a required linear statistic: "
                           + ", ".join(s for s, f in shuffle_fail.items() if not f))
        if not ctrl["iaaft_ok"]:
            reasons.append("IAAFT control passed every nonlinear statistic although the pre-test "
                           "detected nonlinearity")
    elif not failing:
        verdict = "PRESERVED"
    elif lin_ok or nl_ok:
        verdict = "PARTIALLY PRESERVED"
    else:
        verdict = "NOT PRESERVED"
    return {"verdict": verdict, "reasons": reasons, "failing_statistics": failing,
            "gated_statistics": gated, "linear_group_pass": lin_ok, "nonlinear_group_pass": nl_ok,
            "nonlinear_detected": nl_detected, "controls": ctrl, "divergence_ok": div_ok,
            "gates_passed": gates_passed,
            "no_nonlinear_note": None if nl_detected else
            "No detectable nonlinear dynamics to preserve: only the linear claim is evaluated."}


# ----------------------------------------------------------------------------- invariants
def invariants(series: list[np.ndarray], P: MetricParams, dt: float) -> dict:
    vals = batch_stats(series, P, which="nonlinear")
    lle = vals["lle"][:, 0] / dt
    d2 = vals["d2"][:, 0]
    return {"lle": lle, "d2": d2}


def benchmark_check(truth: dict, inv_real: dict, inv_syn: dict, verdict: dict, tol: dict) -> dict:
    out = {"criteria": {}}
    if truth.get("nonlinear"):
        out["criteria"]["nonlinearity_detected"] = verdict["nonlinear_detected"]
        for key, t in (("lle", truth["lle"]), ("d2", truth["d2"])):
            ok = (lambda v: abs(v - t) / t <= tol["lle_rel_tol"]) if key == "lle" else \
                 (lambda v: abs(v - t) <= tol["d2_abs_tol"])
            r = float(np.nanmedian(inv_real[key]))
            s = float(np.nanmedian(inv_syn[key]))
            out["criteria"][f"{key}_real_within_tol"] = bool(ok(r))
            out["criteria"][f"{key}_synthetic_within_tol"] = bool(ok(s))
            out[key] = {"theory": t, "real": r, "synthetic_median": s,
                        "synthetic_iqr": [float(np.nanquantile(inv_syn[key], 0.25)),
                                          float(np.nanquantile(inv_syn[key], 0.75))]}
        out["criteria"]["verdict_preserved"] = verdict["verdict"] == "PRESERVED"
    else:
        out["criteria"]["no_nonlinearity_reported"] = not verdict["nonlinear_detected"]
        out["criteria"]["linear_preserved"] = verdict["verdict"] == "PRESERVED"
        for key in ("lle", "d2"):
            out[key] = {"theory": None, "real": float(np.nanmedian(inv_real[key])),
                        "synthetic_median": float(np.nanmedian(inv_syn[key])),
                        "synthetic_iqr": [float(np.nanquantile(inv_syn[key], 0.25)),
                                          float(np.nanquantile(inv_syn[key], 0.75))]}
    out["passed"] = all(out["criteria"].values())
    return out
