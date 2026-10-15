"""Posterior-predictive generator: iterate the readout with re-lifting at every step.

For series i with posterior draw (s_i, w_i, sigma_i) and delay buffer B (most recent last):
    x_t     = (B[-1], B[-1-tau], ..., B[-1-(m-1)tau])
    y_{t+1} = psi(x_t; s_i)^T w_i + e_t          (psi recomputed from the new buffer each step)
No linear propagation in feature space ever happens. A series is flagged diverged as soon as
|y| > divergence_threshold; it is then frozen (value held) and reported, never dropped silently.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .features import RFF, features_np

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("m", "tau"))
def _simulate(buffers, s, w, innov, W, b, threshold, m, tau):
    lags = jnp.array([-1 - j * tau for j in range(m)])
    F = W.shape[0]
    c = jnp.sqrt(2.0 / F)
    n_lin = 1 + m

    def step(carry, e):
        buf, div, div_step, t = carry
        x = buf[:, lags]                                         # (M, m) delay vector
        u = x @ W.T / s[:, None] + b                             # (M, F)
        y_next = (w[:, 0] + jnp.sum(x * w[:, 1:n_lin], axis=1)
                  + c * jnp.sum(jnp.cos(u) * w[:, n_lin:], axis=1) + e)
        bad = (~jnp.isfinite(y_next)) | (jnp.abs(y_next) > threshold)
        new_div = div | bad
        div_step = jnp.where(bad & ~div, t, div_step)
        y_next = jnp.where(new_div, buf[:, -1], y_next)          # freeze after divergence
        buf = jnp.concatenate([buf[:, 1:], y_next[:, None]], axis=1)
        return (buf, new_div, div_step, t + 1), y_next

    M = buffers.shape[0]
    init = (buffers, jnp.zeros(M, bool), jnp.full(M, -1, jnp.int64), jnp.int64(0))
    (_, div, div_step, _), ys = jax.lax.scan(step, init, innov.T)
    return ys.T, div, div_step


def simulate(buffers: np.ndarray, thetas: np.ndarray, weights: np.ndarray, innov: np.ndarray,
             rff: RFF, m: int, tau: int, threshold: float, batch: int = 250):
    """buffers (M, (m-1)tau+1), thetas (M, 4), weights (M, P), innov (M, T) additive innovations.
    Returns series (M, T), diverged (M,), divergence step (M,; -1 if none)."""
    outs, divs, steps = [], [], []
    for i in range(0, buffers.shape[0], batch):
        sl = slice(i, i + batch)
        ys, dv, ds = _simulate(jnp.asarray(buffers[sl]), jnp.exp(jnp.asarray(thetas[sl, 0])),
                               jnp.asarray(weights[sl]), jnp.asarray(innov[sl]),
                               jnp.asarray(rff.W), jnp.asarray(rff.b), float(threshold), m, tau)
        outs.append(np.asarray(ys))
        divs.append(np.asarray(dv))
        steps.append(np.asarray(ds))
    return np.concatenate(outs), np.concatenate(divs), np.concatenate(steps)


def bootstrap_buffers(y_train: np.ndarray, span: int, n: int, rng: np.random.Generator):
    idx = rng.integers(span, y_train.size, size=n)
    return np.stack([y_train[i - span: i + 1] for i in idx]), idx


def innovations(kind: str, thetas: np.ndarray, weights: np.ndarray, X: np.ndarray, y: np.ndarray,
                rff: RFF, T: int, rng: np.random.Generator, floor: float = 0.0) -> np.ndarray:
    """Gaussian: sigma_eff_i * N(0,1). Residual bootstrap: resample the train residuals of draw i."""
    M = thetas.shape[0]
    if kind == "gaussian":
        return np.sqrt(np.exp(2 * thetas[:, 3]) + floor ** 2)[:, None] * rng.standard_normal((M, T))
    if kind == "residual_bootstrap":
        out = np.empty((M, T))
        cache = {}
        for i in range(M):
            s = float(np.exp(thetas[i, 0]))
            if s not in cache:
                cache = {s: features_np(X, rff, s)}
            resid = y - cache[s] @ weights[i]
            out[i] = rng.choice(resid - resid.mean(), size=T, replace=True)
        return out
    raise ValueError(kind)


def readout_np(x: np.ndarray, theta: np.ndarray, w: np.ndarray, rff: RFF) -> np.ndarray:
    """One-step mean prediction psi(x)^T w (reference implementation used in tests)."""
    return features_np(x, rff, np.exp(theta[0])) @ w
