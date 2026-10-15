"""NUTS over theta with the collapsed likelihood; one OS process per chain; convergence gates.

Each chain runs in its own process with 1/num_chains of the CPU threads given to BLAS and numba
(the likelihood is BLAS-bound; separate processes avoid thread oversubscription). NUTS runs in
whitened coordinates around the Laplace mode (see model.numpyro_model_whitened).

Run length is set by precision, not by a fixed count: after warmup, each chain draws in chunks
until its own bulk and tail ESS for every component of theta reach target_ess / num_chains (and at
least min_samples), up to max_samples. Chains are then truncated to a common length and the
pre-registered gates (R-hat, total bulk/tail ESS, divergences, E-BFMI) are checked on the union.

Worker usage (internal):  python -m koopgen.mcmc <job.npz> <chain> <out.npz> <log.txt>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

THETA_NAMES = ("log_s", "log_alpha_lin", "log_alpha_rff", "log_sigma")
EXTRA = ("diverging", "energy", "num_steps", "accept_prob", "potential_energy")


@dataclass
class MCMCSettings:
    num_chains: int
    num_warmup: int
    target_ess: int          # total bulk and tail ESS (over chains) for every theta component
    min_samples: int         # per chain
    max_samples: int         # per chain
    target_accept: float
    max_tree_depth: int
    chunk: int = 500
    dense_mass: bool = True

    @classmethod
    def from_cfg(cls, d: dict, base: dict):
        return cls(num_chains=int(d["num_chains"]), num_warmup=int(d["num_warmup"]),
                   target_ess=int(d["target_ess"]), min_samples=int(d["min_samples"]),
                   max_samples=int(d["max_samples"]), target_accept=float(base["target_accept"]),
                   max_tree_depth=int(base["max_tree_depth"]), chunk=int(base.get("chunk", 500)))


@dataclass
class GateResult:
    passed: bool
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int
    ebfmi: list
    failures: list = field(default_factory=list)
    table: dict = field(default_factory=dict)


def _min_ess(samples: dict) -> float:
    import arviz as az

    vals = []
    for k in THETA_NAMES:
        x = np.concatenate(samples[k])[None, :]
        vals += [float(az.ess(x, method="bulk")), float(az.ess(x, method="tail"))]
    return min(vals)


# ----------------------------------------------------------------------------- worker
def _worker(job_path: str, chain: int, out_path: str, log_path: str):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_value

    from .features import RFF
    from .model import NumpyLikelihood, Prior, make_callback_loglik, numpyro_model_whitened

    job = np.load(job_path, allow_pickle=True)
    s = json.loads(str(job["settings"]))
    prior = Prior(**{k: tuple(v) for k, v in json.loads(str(job["prior"])).items()})
    rff = RFF(W=job["W"], b=job["b"])
    lik = NumpyLikelihood(job["X"], job["y"], rff, float(job["floor"]))
    loglik = make_callback_loglik(lik)
    seed = int(job["seed"]) + 1000 * chain

    # Whitened coordinates z (theta = mode + Lc z). Over-dispersed start z0 = 2 * N(0, I),
    # independent per chain, so that R-hat compares chains started in different places.
    rng = np.random.default_rng(seed)
    mode, cov = job["mode"], job["cov"]
    Lc = np.linalg.cholesky(cov)
    for k in range(12):   # shrink the dispersion until the start point has a finite likelihood
        z0 = 2.0 * 0.5 ** k * rng.standard_normal(4)
        if np.isfinite(lik(mode + Lc @ z0)[0]):
            break
    kernel = NUTS(numpyro_model_whitened, target_accept_prob=s["target_accept"],
                  max_tree_depth=s["max_tree_depth"], dense_mass=s["dense_mass"],
                  init_strategy=init_to_value(values={"z": jnp.asarray(z0)}))
    margs = (prior, loglik, jnp.asarray(mode), jnp.asarray(Lc))
    log = open(log_path, "a", buffering=1)
    t0 = time.time()
    mcmc = MCMC(kernel, num_warmup=s["num_warmup"], num_samples=s["chunk"], num_chains=1,
                progress_bar=False)
    mcmc.warmup(jax.random.PRNGKey(seed), *margs, extra_fields=EXTRA, collect_warmup=False)
    log.write(f"chain {chain}: warmup {s['num_warmup']} done in {time.time() - t0:.0f}s\n")
    samples = {k: [] for k in THETA_NAMES}
    extras = {k: [] for k in EXTRA}
    per_chain_target = s["target_ess"] / s["num_chains"]
    done, ess = 0, 0.0
    while done < s["max_samples"]:
        mcmc.num_samples = min(s["chunk"], s["max_samples"] - done)
        mcmc.run(mcmc.post_warmup_state.rng_key, *margs, extra_fields=EXTRA)
        mcmc.post_warmup_state = mcmc.last_state
        sm, ex = mcmc.get_samples(), mcmc.get_extra_fields()
        for k in THETA_NAMES:
            samples[k].append(np.asarray(sm[k]))
        for k in EXTRA:
            extras[k].append(np.asarray(ex[k]))
        done += mcmc.num_samples
        ess = _min_ess(samples)
        log.write(f"chain {chain}: {done} draws, min ESS {ess:.0f}/{per_chain_target:.0f}, "
                  f"{time.time() - t0:.0f}s, mean steps {np.mean(extras['num_steps'][-1]):.1f}, "
                  f"divergences {int(sum(e.sum() for e in extras['diverging']))}\n")
        if done >= s["min_samples"] and ess >= per_chain_target:
            break
    step_size = float(np.asarray(mcmc.last_state.adapt_state.step_size))
    np.savez(out_path, **{k: np.concatenate(v) for k, v in samples.items()},
             **{"x_" + k: np.concatenate(v) for k, v in extras.items()},
             step_size=step_size, wall=time.time() - t0, chain_ess=ess)
    log.write(f"chain {chain}: finished {done} draws in {time.time() - t0:.0f}s (step size {step_size:.3g})\n")
    log.close()


# ----------------------------------------------------------------------------- parent
def run_nuts(X, y, rff, prior, settings: MCMCSettings, seed: int, workdir: Path,
             log=print, floor: float = 0.0):
    """Run settings.num_chains NUTS chains in parallel processes; return (InferenceData, info)."""
    import arviz as az

    from .model import NumpyLikelihood, laplace_approximation

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t_lap = time.time()
    mode, cov, lap_info = laplace_approximation(NumpyLikelihood(X, y, rff, floor), prior)
    log(f"Laplace init: mode {np.round(mode, 3).tolist()}, sd {np.round(np.sqrt(np.diag(cov)), 4).tolist()} "
        f"({time.time() - t_lap:.0f}s, {lap_info})")
    job = workdir / "mcmc_job.npz"
    np.savez(job, X=X, y=y, W=rff.W, b=rff.b, seed=seed, mode=mode, cov=cov, floor=floor,
             settings=json.dumps(asdict(settings)), prior=json.dumps(prior.as_dict()))
    n_cpu = os.cpu_count() or 4
    threads = max(1, n_cpu // settings.num_chains)
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    logf = workdir / "mcmc_progress.log"
    t0 = time.time()
    procs = []
    for c in range(settings.num_chains):
        out = workdir / f"chain{c}.npz"
        if out.exists():
            out.unlink()
        procs.append((c, out, subprocess.Popen(
            [sys.executable, "-m", "koopgen.mcmc", str(job), str(c), str(out), str(logf)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    failed = []
    for c, out, p in procs:
        output, _ = p.communicate()
        if p.returncode != 0 or not out.exists():
            failed.append((c, output[-3000:]))
    if failed:
        msg = "\n".join(f"chain {c} failed:\n{o}" for c, o in failed)
        raise RuntimeError(f"MCMC worker failure (no chain is dropped silently):\n{msg}")
    res = [np.load(workdir / f"chain{c}.npz") for c in range(settings.num_chains)]
    lengths = [int(r[THETA_NAMES[0]].size) for r in res]
    n = min(lengths)   # common length (chains stop independently); later draws are discarded, never earlier
    post = {k: np.stack([r[k][:n] for r in res]) for k in THETA_NAMES}
    stats = {"diverging": np.stack([r["x_diverging"][:n] for r in res]).astype(bool),
             "energy": np.stack([r["x_energy"][:n] for r in res]),
             "n_steps": np.stack([r["x_num_steps"][:n] for r in res]),
             "acceptance_rate": np.stack([r["x_accept_prob"][:n] for r in res]),
             "lp": -np.stack([r["x_potential_energy"][:n] for r in res])}
    n_div_dropped = int(sum(r["x_diverging"][n:].sum() for r in res))
    prior_rng = np.random.default_rng(seed + 7)
    prior_draws = {k: prior_rng.normal(*getattr(prior, k), size=(1, 4000)) for k in THETA_NAMES}
    idata = az.from_dict(posterior=post, sample_stats=stats, prior=prior_draws)
    info = {"wall_seconds": time.time() - t0, "threads_per_chain": threads,
            "step_sizes": [float(r["step_size"]) for r in res],
            "chain_wall_seconds": [float(r["wall"]) for r in res],
            "chain_lengths": lengths, "retained_per_chain": n,
            "divergences_in_truncated_tail": n_div_dropped,
            "mean_leapfrog_steps": float(stats["n_steps"].mean()),
            "n_gradient_evals": int(sum(r["x_num_steps"].sum() for r in res)),
            "laplace_mode": mode.tolist(), "laplace_cov": cov.tolist(), "laplace_info": lap_info,
            "settings": asdict(settings)}
    log(f"MCMC: {settings.num_chains} chains, warmup {settings.num_warmup}, draws per chain {lengths} "
        f"(retained {n}) in {info['wall_seconds']:.0f}s, mean leapfrog steps {info['mean_leapfrog_steps']:.1f}")
    return idata, info


def check_gates(idata, gates: dict, extra_divergences: int = 0) -> GateResult:
    import arviz as az

    summ = az.summary(idata, var_names=list(THETA_NAMES), kind="all", round_to="none")
    rhat = float(summ["r_hat"].max())
    eb = float(summ["ess_bulk"].min())
    et = float(summ["ess_tail"].min())
    div = int(idata.sample_stats["diverging"].values.sum()) + int(extra_divergences)
    ebfmi = [float(v) for v in np.atleast_1d(az.bfmi(idata))]
    fails = []
    if not rhat <= gates["rhat_max"]:
        fails.append(f"R-hat max {rhat:.4f} > {gates['rhat_max']}")
    if not eb >= gates["ess_bulk_min"]:
        fails.append(f"bulk ESS min {eb:.0f} < {gates['ess_bulk_min']}")
    if not et >= gates["ess_tail_min"]:
        fails.append(f"tail ESS min {et:.0f} < {gates['ess_tail_min']}")
    if div > gates["divergences_max"]:
        fails.append(f"{div} divergent transitions")
    if min(ebfmi) <= gates["ebfmi_min"]:
        fails.append(f"E-BFMI min {min(ebfmi):.3f} <= {gates['ebfmi_min']}")
    table = {k: {c: float(summ.loc[k, c]) for c in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean",
                                                    "ess_bulk", "ess_tail", "r_hat")}
             for k in THETA_NAMES}
    return GateResult(passed=not fails, rhat_max=rhat, ess_bulk_min=eb, ess_tail_min=et,
                      divergences=div, ebfmi=ebfmi, failures=fails, table=table)


def run_with_gates(X, y, rff, prior, settings: MCMCSettings, gates: dict, remediation: list,
                   seed: int, workdir: Path, log=print, floor: float = 0.0):
    """Run NUTS; if a gate fails, diagnose and rerun along a pre-declared remediation ladder:
    divergences / low E-BFMI -> raise target_accept; low ESS or high R-hat -> double warmup,
    minimum and maximum draws. Every attempt is recorded. Divergences in draws discarded by the
    common-length truncation still count. Returns (idata, info, gate_result, attempts)."""
    attempts = []
    s = settings
    ladder = list(remediation)
    for attempt in range(1 + len(ladder)):
        idata, info = run_nuts(X, y, rff, prior, s, seed + 101 * attempt, workdir, log, floor)
        g = check_gates(idata, gates, info["divergences_in_truncated_tail"])
        attempts.append({"attempt": attempt, "settings": asdict(s), "gates": asdict(g),
                         "wall_seconds": info["wall_seconds"]})
        log(f"MCMC gates attempt {attempt}: {'PASS' if g.passed else 'FAIL: ' + '; '.join(g.failures)}")
        if g.passed or attempt == len(ladder):
            return idata, info, g, attempts
        new = dict(asdict(s))
        if g.divergences > gates["divergences_max"] or min(g.ebfmi) <= gates["ebfmi_min"]:
            new["target_accept"] = float(ladder[attempt])
        if (g.rhat_max > gates["rhat_max"] or g.ess_bulk_min < gates["ess_bulk_min"]
                or g.ess_tail_min < gates["ess_tail_min"]):
            for k in ("num_warmup", "min_samples", "max_samples", "target_ess"):
                new[k] = 2 * new[k]
        s = MCMCSettings(**new)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
