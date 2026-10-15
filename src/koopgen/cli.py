"""Command-line interface and end-to-end pipeline.

    koopgen run --benchmark lorenz [--smoke]
    koopgen run --data file.csv --column y [--time-col t] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from . import data as Dm
from . import embedding as Em
from . import features as Fe
from . import generate as Ge
from . import mcmc as MC
from . import metrics as Mx
from . import model as Mo
from . import spectrum as Sp
from . import surrogates as Su
from . import validate as Va

PKGS = ["numpy", "scipy", "pandas", "jax", "jaxlib", "numpyro", "arviz", "numba", "statsmodels",
        "matplotlib", "pyyaml", "pyarrow", "h5netcdf"]


# ----------------------------------------------------------------------------- utilities
class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


class Timer:
    def __init__(self):
        self.stages = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *a):
                timer.stages[name] = timer.stages.get(name, 0.0) + time.time() - self.t
        return _C()


def seeds(master: int):
    ss = np.random.SeedSequence(master)
    names = ["data", "embedding", "features", "prior", "mcmc", "weights", "generate", "surrogates",
             "windows", "null", "pretest", "forecast", "sensitivity", "ppc"]
    return {n: int(s.generate_state(1)[0]) for n, s in zip(names, ss.spawn(len(names)))}


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prereg_hash(cfg: dict) -> str:
    """SHA-256 of the canonical YAML dump of the `preregistered` section only."""
    canon = yaml.safe_dump(cfg["preregistered"], sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def thin_idx(n_total: int, n: int, rng) -> np.ndarray:
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


# ----------------------------------------------------------------------------- core fit+generate
def fit_and_generate(sp, m, tau, F, cfg, mcfg: MC.MCMCSettings, sd: dict, workdir: Path, log,
                     timer, innovations="gaussian", n_series=None, tag="", remediate=True):
    """Fit the Bayesian kernel-Koopman model on train data and draw posterior-predictive series."""
    X, y, Xn = Em.regression_pairs(sp.train, m, tau)
    floor = float(cfg["model"]["sigma_floor"])
    rff = Fe.draw_rff(m, F, sd["features"])
    prior = Mo.make_prior(X, cfg["model"]["prior"], cfg["model"]["max_pairs_median"], sd["prior"])
    with timer(f"mcmc{tag}"):
        idata, info, gates, attempts = MC.run_with_gates(
            X, y, rff, prior, mcfg, cfg["preregistered"]["mcmc_gates"],
            cfg["mcmc"]["remediation_target_accept"] if remediate else [], sd["mcmc"], workdir, log,
            floor)
    post = np.stack([idata.posterior[k].values.reshape(-1) for k in Mo.THETA_NAMES], axis=1)
    rng = np.random.default_rng(sd["weights"])
    M = int(n_series or cfg["generate"]["n_series"])
    idx = thin_idx(post.shape[0], M, rng)
    thetas = post[idx]
    with timer(f"weights{tag}"):
        weights = Mo.sample_weights(thetas, X, y, rff, sd["weights"], floor)
    span = (m - 1) * tau
    grng = np.random.default_rng(sd["generate"])
    buffers, _ = Ge.bootstrap_buffers(sp.train, span, M, grng)
    T = cfg["generate"]["burn_in"] + sp.test.size
    with timer(f"generate{tag}"):
        innov = Ge.innovations(innovations, thetas, weights, X, y, rff, T, grng, floor)
        ys, div, div_step = Ge.simulate(buffers, thetas, weights, innov, rff, m, tau,
                                        cfg["generate"]["divergence_factor"])
    series = ys[:, cfg["generate"]["burn_in"]:]
    log(f"generated {M} series{tag}: diverged {int(div.sum())} ({div.mean():.1%})")
    return {"X": X, "y": y, "Xn": Xn, "rff": rff, "prior": prior, "idata": idata, "mcmc_info": info,
            "gates": gates, "attempts": attempts, "posterior": post, "thetas": thetas,
            "weights": weights, "series": series, "diverged": div, "div_step": div_step, "floor": floor}


def window_length(cfg, n_test) -> int:
    W = min(int(cfg["metrics"]["window_length"]), n_test // 6)
    if W < int(cfg["metrics"]["min_window_length"]):
        raise Dm.DataValidationError(
            f"test set of {n_test} points gives windows of {W} < {cfg['metrics']['min_window_length']} "
            "points (need N_test >= 6 x min window); a longer series is required.")
    return W


def score_generator(series, diverged, W, P, real_stats, cfg, rng, name):
    ok = np.nonzero(~diverged)[0]
    wins = Va.one_window_per_series(series[ok], W, rng)
    return Va.batch_stats(wins, P)


# ----------------------------------------------------------------------------- pipeline
def run_pipeline(series: Dm.Series, cfg: dict, out_root: Path, smoke: bool, config_path: Path,
                 run_sensitivity: bool = True, run_id: str | None = None) -> dict:
    t_start = time.time()
    sd = seeds(int(cfg["seed"]))
    cfg_hash = file_hash(config_path)
    pr_hash = prereg_hash(cfg)
    run_id = run_id or f"{series.name.replace(':', '_')}_{time.strftime('%Y%m%d-%H%M%S')}" + ("_smoke" if smoke else "")
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run.log")
    timer = Timer()
    log(f"run {run_id}; config sha256 {cfg_hash[:16]}")
    pre_reg = Path(config_path).parent / "PREREGISTRATION.md"
    registered = None
    if pre_reg.exists():
        txt = pre_reg.read_text(encoding="utf-8")
        registered = pr_hash in txt
        log(f"pre-registration check: preregistered-section hash {pr_hash[:16]} "
            f"{'MATCHES' if registered else 'DOES NOT MATCH'} PREREGISTRATION.md")

    # ---- data
    with timer("data"):
        sp = Dm.split_standardise(series.y_raw, cfg["data"]["train_frac"])
        stat = Dm.stationarity(sp.train, cfg["data"]["stationarity_alpha"])
    log(f"data: n={sp.y.size}, train={sp.n_train}, test={sp.test.size}; ADF p={stat['adf_p']:.3g}, "
        f"KPSS p={stat['kpss_p']:.3g}, nonstationary={stat['nonstationary']}")

    # ---- embedding
    with timer("embedding"):
        emb = Em.choose_embedding(sp.train, cfg["embedding"], seed=sd["embedding"])
    m, tau = emb.m, emb.tau
    log(f"embedding: tau={tau} ({emb.tau_method}); m={m} ({emb.m_method}); notes={emb.notes}")

    # ---- MCMC settings
    mc = cfg["mcmc"]
    profile = "smoke" if smoke else "full"
    mcfg = MC.MCMCSettings.from_cfg(mc["smoke"] if smoke else mc, mc)
    log(f"MCMC settings ({profile}): {mcfg}")
    F = int(cfg["features"]["n_features"])
    M = int(cfg["generate"]["n_series"]) if not smoke else min(100, int(cfg["generate"]["n_series"]))
    fit = fit_and_generate(sp, m, tau, F, cfg, mcfg, sd, out / "mcmc", log, timer, n_series=M,
                           remediate=not smoke)
    div_rate = float(fit["diverged"].mean())

    # ---- spectrum
    with timer("spectrum"):
        n_draws = int(cfg["spectrum"]["n_draws"]) if not smoke else 50
        srng = np.random.default_rng(sd["weights"] + 1)
        th_spec = fit["posterior"][thin_idx(fit["posterior"].shape[0], n_draws, srng)]
        mu, rres, spec_summary = Sp.posterior_spectra(th_spec, fit["rff"], fit["X"], fit["Xn"],
                                                      cfg["spectrum"]["resdmd_threshold"],
                                                      cfg["spectrum"]["instability_delta"], log,
                                                      floor=fit["floor"])
    log(f"spectrum: {spec_summary}")

    # ---- controls
    rng_s = np.random.default_rng(sd["surrogates"])
    with timer("controls"):
        iaaft_s = Su.iaaft(sp.test, M, rng_s)
        shuf_s = Su.shuffle(sp.test, M, rng_s)
        p_max = int(min(50, max(5, sp.n_train // 50)))
        ar_model = Su.fit_ar_aic(sp.train, p_max)
        ar_s = Su.simulate_ar(ar_model, sp.train, M, sp.test.size, cfg["generate"]["burn_in"], rng_s)
    log(f"controls: AR order {ar_model.p} (AIC, p_max {p_max})")

    # ---- statistics
    W = window_length(cfg, sp.test.size)
    P = Va.MetricParams.from_train(sp.train, m, tau, cfg["metrics"])
    rng_w = np.random.default_rng(sd["windows"])
    with timer("statistics"):
        real = Va.batch_stats(Va.real_windows(sp.test, W, int(cfg["metrics"]["window_stride"])), P)
        stats = {"real": real,
                 "koopman": score_generator(fit["series"], fit["diverged"], W, P, real, cfg, rng_w, "koopman"),
                 "iaaft": Va.batch_stats(Va.one_window_per_series(iaaft_s, W, rng_w), P),
                 "ar": Va.batch_stats(Va.one_window_per_series(ar_s, W, rng_w), P),
                 "shuffle": Va.batch_stats(Va.one_window_per_series(shuf_s, W, rng_w), P)}
    log(f"statistics: W={W}, real windows={real['mean'].shape[0]}, metric params {P.as_dict()}")

    pr = cfg["preregistered"]
    stat_names = list(pr["statistics"]["linear"]) + list(pr["statistics"]["nonlinear"])
    block = (W // int(cfg["metrics"]["window_stride"])) if pr["decision"]["block_windows"] == "auto" \
        else int(pr["decision"]["block_windows"])
    with timer("decision"):
        rows = Va.compare(real, {k: stats[k] for k in ("koopman", "iaaft", "ar", "shuffle")}, stat_names,
                          block, int(pr["decision"]["n_null"]), float(pr["decision"]["null_quantile"]),
                          float(pr["decision"]["max_nan_rate_excess"]), np.random.default_rng(sd["null"]))
    with timer("pretest"):
        pre_cfg = dict(pr["pretest"])   # full surrogate count even in smoke mode (fewer cannot reject under Holm)
        pre = Va.nonlinearity_pretest(sp.train, P, pre_cfg, np.random.default_rng(sd["pretest"]), log)
    log(f"pre-test: nonlinear detected = {pre['nonlinear_detected']} ({pre['n_rejections']} rejections)")
    verdict = Va.controls_and_verdict(rows, pre, pr, fit["gates"].passed, div_rate)
    log(f"VERDICT: {verdict['verdict']}; failing: {verdict['failing_statistics']}; reasons: {verdict['reasons']}")

    # ---- secondary PPC null (Koopman only)
    ppc = None
    if pr["decision"]["secondary_ppc_null"]["enabled"]:
        with timer("ppc_null"):
            n_str = int(pr["decision"]["secondary_ppc_null"]["n_stretches"]) if not smoke else 10
            ok = np.nonzero(~fit["diverged"])[0]
            rng_p = np.random.default_rng(sd["ppc"])
            pick = rng_p.choice(ok, size=min(n_str, ok.size), replace=False)
            stride = int(cfg["metrics"]["window_stride"])
            per = [Va.real_windows(fit["series"][i], W, stride) for i in pick]
            flat = Va.batch_stats([w for p in per for w in p], P)
            k = len(per[0])
            real_like = [{s: flat[s][j * k:(j + 1) * k] for s in stat_names} for j in range(len(per))]
            ppc = []
            for s in stat_names:
                sc = Va.scales_from_real(real[s])
                nul = np.array([Va._component_w1(rl[s], stats["koopman"][s], sc) for rl in real_like])
                obs = Va._component_w1(real[s], stats["koopman"][s], sc)
                ppc.append({"statistic": s, "w1": obs, "ppc_q95": float(np.nanquantile(nul, 0.95)),
                            "ppc_p": float(np.mean(nul >= obs)), "ppc_pass": bool(obs <= np.nanquantile(nul, 0.95))})

    # ---- forecasts
    with timer("forecast"):
        forecast = forecast_skill(sp, fit, ar_model, m, tau, cfg, sd, smoke)

    # ---- invariants
    with timer("invariants"):
        n_inv = int(pr["invariants"]["n_series"]) if not smoke else 10
        ok = np.nonzero(~fit["diverged"])[0][:n_inv]
        inv_real = Va.invariants([sp.test], P, series.dt)
        inv_syn = Va.invariants(list(fit["series"][ok]), P, series.dt)
    bench = None
    if series.truth:
        bench = Va.benchmark_check(series.truth, inv_real, inv_syn, verdict, pr["benchmarks"])
        log(f"benchmark check: {bench}")

    res = {"name": series.name, "dt": series.dt, "time_unit": series.time_unit, "cfg": cfg,
           "run_id": run_id, "split": sp, "embedding": emb, "stationarity": stat, "findings": series.findings,
           "idata": fit["idata"], "prior": fit["prior"], "gates": fit["gates"], "attempts": fit["attempts"],
           "mcmc_info": fit["mcmc_info"], "spectrum": {"mu": mu, "res": rres, "summary": spec_summary},
           "series": {"koopman": fit["series"], "iaaft": iaaft_s, "ar": ar_s, "shuffle": shuf_s},
           "diverged": fit["diverged"], "div_step": fit["div_step"], "divergence_rate": div_rate,
           "ar_model": ar_model, "metric_params": P, "window_length": W, "stats": stats,
           "scorecard": rows, "pretest": pre, "verdict": verdict, "ppc": ppc, "forecast": forecast,
           "invariants": {"real": inv_real, "synthetic": inv_syn}, "benchmark": bench,
           "smoke": smoke, "config_hash": cfg_hash, "prereg_hash": pr_hash, "preregistered_match": registered,
           "mcmc_settings": mcfg, "mcmc_profile": profile, "F": F, "M": M, "truth": series.truth}

    import pickle
    with open(out / "results_bundle.pkl", "wb") as fh:   # checkpoint before the long sensitivity stage
        pickle.dump(res, fh)

    # ---- sensitivity
    if run_sensitivity:
        with timer("sensitivity"):
            res["sensitivity"] = sensitivity(res, sp, fit, cfg, sd, out, log, timer, smoke, real, P, W,
                                             stat_names, block, pre, stats)

    # ---- save artefacts and report
    with timer("save"):
        save_artifacts(res, out)
    with open(out / "results_bundle.pkl", "wb") as fh:
        pickle.dump(res, fh)
    from . import plots, report
    with timer("figures"):
        captions = plots.all_figures(res, out / "figures")
    res["timings"] = timer.stages
    res["wall_seconds"] = time.time() - t_start
    meta = {"run_id": run_id, "config_sha256": cfg_hash, "preregistered_sha256": pr_hash, "preregistered_match": registered,
            "master_seed": int(cfg["seed"]), "derived_seeds": sd, "versions": versions(),
            "stage_seconds": timer.stages, "wall_seconds": res["wall_seconds"], "smoke": smoke,
            "argv": sys.argv, "series": series.name}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    report.write_report(res, out, captions)
    log(f"done in {res['wall_seconds']:.0f}s -> {out}")
    return res


def forecast_skill(sp, fit, ar_model, m, tau, cfg, sd, smoke):
    """Posterior-predictive ensembles from test origins; CRPS vs horizon vs AR(p), persistence."""
    fc = cfg["forecast"]
    H = int(fc["horizon"])
    n_orig = int(fc["n_origins"]) if not smoke else 10
    n_mem = int(fc["n_members"]) if not smoke else 50
    span = (m - 1) * tau
    y = sp.y
    lo, hi = sp.n_train + max(span, ar_model.p), y.size - H - 1
    origins = np.linspace(lo, hi, n_orig).astype(int)
    rng = np.random.default_rng(sd["forecast"])
    crps_k, crps_a, crps_p = [], [], []
    fan = None
    for o in origins:
        sel = rng.choice(fit["thetas"].shape[0], size=n_mem, replace=True)
        buf = np.repeat(y[o - span: o + 1][None], n_mem, axis=0)
        innov = Mo.sigma_eff(fit["thetas"][sel, 3], fit["floor"])[:, None] * rng.standard_normal((n_mem, H))
        ens, div, _ = Ge.simulate(buf, fit["thetas"][sel], fit["weights"][sel], innov, fit["rff"], m, tau,
                                  cfg["generate"]["divergence_factor"])
        truth = y[o + 1: o + 1 + H]
        crps_k.append(Mx.crps_ensemble(ens.T, truth))
        ar_ens = Su.ar_forecast_ensemble(ar_model, y[: o + 1], H, n_mem, rng)
        crps_a.append(Mx.crps_ensemble(ar_ens.T, truth))
        crps_p.append(np.abs(truth - y[o]))
        if fan is None:
            fan = {"fan_ensemble": ens, "fan_truth": truth, "fan_history": y[max(0, o - 3 * H): o + 1]}
    return {"crps_koopman": np.mean(crps_k, 0), "crps_ar": np.mean(crps_a, 0),
            "crps_persistence": np.mean(crps_p, 0), "n_origins": n_orig, "origins": origins, **fan}


def sensitivity(res, sp, fit, cfg, sd, out, log, timer, smoke, real, P, W, stat_names, block, pre, stats):
    """Re-fit / re-generate under (m-1, tau), (m+1, tau), F in {256, 1024} and residual-bootstrap
    innovations; the scorecard of each variant uses the same real windows and metric embedding."""
    m, tau = res["embedding"].m, res["embedding"].tau
    F = res["F"]
    mc = cfg["mcmc"]
    mcfg = MC.MCMCSettings.from_cfg(mc["smoke"] if smoke else mc["sensitivity"], mc)
    variants = [("base", None)]
    if m - 1 >= 1:
        variants.append((f"m={m - 1}", dict(m=m - 1, tau=tau, F=F)))
    variants.append((f"m={m + 1}", dict(m=m + 1, tau=tau, F=F)))
    for f2 in cfg["features"]["sensitivity_n_features"]:
        variants.append((f"F={f2}", dict(m=m, tau=tau, F=int(f2))))
    variants.append(("residual bootstrap", "resid"))
    pr = cfg["preregistered"]
    out_rows = []
    rng = np.random.default_rng(sd["sensitivity"])
    for label, v in variants:
        if v is None:
            rows, gates_ok, div = res["scorecard"], res["gates"].passed, res["divergence_rate"]
            mcmc_note = "full settings"
        elif v == "resid":
            span = (m - 1) * tau
            grng = np.random.default_rng(sd["generate"] + 5)
            buffers, _ = Ge.bootstrap_buffers(sp.train, span, fit["thetas"].shape[0], grng)
            T = cfg["generate"]["burn_in"] + sp.test.size
            innov = Ge.innovations("residual_bootstrap", fit["thetas"], fit["weights"], fit["X"], fit["y"],
                                   fit["rff"], T, grng, fit["floor"])
            ys, dv, _ = Ge.simulate(buffers, fit["thetas"], fit["weights"], innov, fit["rff"], m, tau,
                                    cfg["generate"]["divergence_factor"])
            s = ys[:, cfg["generate"]["burn_in"]:]
            g = score_generator(s, dv, W, P, real, cfg, rng, "resid")
            rows = Va.compare(real, {"koopman": g}, stat_names, block, int(pr["decision"]["n_null"]),
                              float(pr["decision"]["null_quantile"]), float(pr["decision"]["max_nan_rate_excess"]),
                              np.random.default_rng(sd["null"]))
            gates_ok, div = res["gates"].passed, float(dv.mean())
            mcmc_note = "same posterior as base"
        else:
            log(f"sensitivity variant {label}: refitting")
            sub = fit_and_generate(sp, v["m"], v["tau"], v["F"], cfg, mcfg, sd, out / "mcmc_sens" / label.replace("=", ""),
                                   log, timer, n_series=res["M"], tag=f"[{label}]", remediate=not smoke)
            g = score_generator(sub["series"], sub["diverged"], W, P, real, cfg, rng, label)
            rows = Va.compare(real, {"koopman": g}, stat_names, block, int(pr["decision"]["n_null"]),
                              float(pr["decision"]["null_quantile"]), float(pr["decision"]["max_nan_rate_excess"]),
                              np.random.default_rng(sd["null"]))
            gates_ok, div = sub["gates"].passed, float(sub["diverged"].mean())
            mcmc_note = f"{mcfg.num_chains} chains, warmup {mcfg.num_warmup}, target ESS {mcfg.target_ess}; gates {'PASS' if gates_ok else 'FAIL: ' + '; '.join(sub['gates'].failures)}"
        # controls are generator-independent: reuse the base control rows for the verdict logic
        ctrl_rows = [r for r in res["scorecard"] if r["generator"] != "koopman"]
        vv = Va.controls_and_verdict([r for r in rows if r["generator"] == "koopman"] + ctrl_rows, pre, pr,
                                     gates_ok, div)
        out_rows.append({"label": label, "rows": [r for r in rows if r["generator"] == "koopman"],
                         "verdict": vv["verdict"], "failing": vv["failing_statistics"],
                         "divergence_rate": div, "mcmc": mcmc_note})
        log(f"sensitivity {label}: {vv['verdict']} (failing {vv['failing_statistics']}, divergence {div:.1%})")
    return out_rows


def save_artifacts(res, out: Path):
    import pandas as pd

    res["idata"].to_netcdf(str(out / "posterior.nc"))
    n_test = res["split"].test.size
    frames = []
    for k, s in res["series"].items():
        df = pd.DataFrame(s.astype(np.float32), columns=[f"t{i}" for i in range(n_test)])
        df.insert(0, "series", np.arange(s.shape[0]))
        df.insert(0, "generator", k)
        if k == "koopman":
            df.insert(2, "diverged", res["diverged"])
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(out / "synthetic_series.parquet")
    recs = []
    for src, st in res["stats"].items():
        for name, arr in st.items():
            for w in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    recs.append((src, name, w, c, float(arr[w, c])))
    pd.DataFrame(recs, columns=["source", "statistic", "window", "component", "value"]).to_parquet(
        out / "statistics.parquet")
    pd.DataFrame(res["scorecard"]).to_csv(out / "scorecard.csv", index=False)
    if res.get("ppc"):
        pd.DataFrame(res["ppc"]).to_csv(out / "ppc_secondary.csv", index=False)
    pre = {k: v for k, v in res["pretest"].items() if k not in ("surrogate_stats", "observed_stats")}
    summary = {
        "verdict": res["verdict"], "gates": {k: v for k, v in vars(res["gates"]).items()},
        "mcmc_attempts": res["attempts"], "mcmc_info": {k: v for k, v in res["mcmc_info"].items()},
        "spectrum": res["spectrum"]["summary"], "divergence_rate": res["divergence_rate"],
        "n_diverged": int(res["diverged"].sum()), "pretest": pre, "benchmark": res["benchmark"],
        "embedding": {"m": res["embedding"].m, "tau": res["embedding"].tau,
                      "tau_method": res["embedding"].tau_method, "m_method": res["embedding"].m_method,
                      "notes": res["embedding"].notes},
        "stationarity": res["stationarity"], "metric_params": res["metric_params"].as_dict(),
        "window_length": res["window_length"], "ar_order": res["ar_model"].p,
        "forecast": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res["forecast"].items()
                     if k.startswith("crps") or k == "n_origins"},
        "invariants": {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in res["invariants"].items()},
        "sensitivity": res.get("sensitivity"), "prior": res["prior"].as_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    np.savez_compressed(out / "spectrum.npz", mu=res["spectrum"]["mu"], res=res["spectrum"]["res"])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="koopgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="csv or parquet file")
    src.add_argument("--benchmark", choices=["lorenz", "henon", "ar2"])
    r.add_argument("--column")
    r.add_argument("--time-col")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", default="results")
    r.add_argument("--smoke", action="store_true", help="tiny MCMC / generator settings for fast iteration")
    r.add_argument("--no-sensitivity", action="store_true")
    r.add_argument("--run-id")
    rp = sub.add_parser("replot", help="regenerate figures and REPORT.md from a finished run directory")
    rp.add_argument("run_dir")
    a = ap.parse_args(argv)
    if a.cmd == "replot":
        import pickle
        from . import plots, report
        run_dir = Path(a.run_dir)
        res = pickle.load(open(run_dir / "results_bundle.pkl", "rb"))
        captions = plots.all_figures(res, run_dir / "figures")
        report.write_report(res, run_dir, captions)
        print(f"replotted {run_dir}")
        return 0
    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if a.benchmark:
        sd = seeds(int(cfg["seed"]))
        series = Dm.benchmark_series(a.benchmark, cfg, np.random.default_rng(sd["data"]))
    else:
        if not a.column:
            ap.error("--column is required with --data")
        try:
            series = Dm.load_series(a.data, a.column, a.time_col, int(cfg["data"]["min_length"]),
                                    float(cfg["data"]["dt_rel_tol"]))
        except Dm.DataValidationError as exc:
            print(f"DATA VALIDATION FAILED: {exc}", file=sys.stderr)
            return 2
    run_pipeline(series, cfg, Path(a.out), a.smoke, cfg_path, run_sensitivity=not a.no_sensitivity,
                 run_id=a.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
