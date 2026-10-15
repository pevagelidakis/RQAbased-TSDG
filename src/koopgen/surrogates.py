"""Control generators: IAAFT surrogates, random shuffles and a fitted AR(p) by AIC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


def iaaft(x: np.ndarray, n_surr: int, rng: np.random.Generator, max_iter: int = 1000) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogates (Schreiber & Schmitz 1996).

    Returns an (n_surr, N) array. Each surrogate has exactly the marginal of x (final step is the
    rank-ordering) and approximately its periodogram. Iteration stops per surrogate when the rank
    order no longer changes.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    amp = np.abs(np.fft.rfft(x))
    xs = np.sort(x)
    s = np.stack([rng.permutation(x) for _ in range(n_surr)])
    prev_rank = np.full((n_surr, n), -1, dtype=np.int64)
    active = np.ones(n_surr, dtype=bool)
    for _ in range(max_iter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        S = np.fft.rfft(s[idx], axis=1)
        z = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n, axis=1)
        rank = np.argsort(np.argsort(z, axis=1, kind="stable"), axis=1, kind="stable")
        s[idx] = xs[rank]
        done = np.all(rank == prev_rank[idx], axis=1)
        prev_rank[idx] = rank
        active[idx[done]] = False
    return s


def shuffle(x: np.ndarray, n_surr: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.stack([rng.permutation(x) for _ in range(n_surr)])


def endpoint_matched_segment(x: np.ndarray, length: int) -> tuple[int, int]:
    """Start index of the length-`length` segment minimising the end-point mismatch
    (x_0 - x_{L-1})^2 + ((x_1 - x_0) - (x_{L-1} - x_{L-2}))^2, which limits the spurious
    high-frequency content that a periodic Fourier surrogate would otherwise introduce."""
    x = np.asarray(x, dtype=np.float64)
    length = min(length, x.size)
    starts = np.arange(0, x.size - length + 1)
    a, b = x[starts], x[starts + length - 1]
    da, db = x[starts + 1] - a, b - x[starts + length - 2]
    cost = (a - b) ** 2 + (da - db) ** 2
    i = int(starts[np.argmin(cost)])
    return i, i + length


@dataclass
class ARModel:
    p: int
    const: float
    coefs: np.ndarray
    sigma: float
    aic_table: dict


def fit_ar_aic(y: np.ndarray, p_max: int) -> ARModel:
    """AR(p) with p in 1..p_max selected by AIC; all candidates fitted on the same sample
    (first p_max values held out) so AICs are comparable. OLS with intercept."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    target = y[p_max:]
    lags = np.column_stack([y[p_max - k: n - k] for k in range(1, p_max + 1)])
    best, table = None, {}
    for p in range(1, p_max + 1):
        Z = np.column_stack([np.ones(target.size), lags[:, :p]])
        beta, *_ = np.linalg.lstsq(Z, target, rcond=None)
        resid = target - Z @ beta
        s2 = resid @ resid / target.size
        aic = target.size * np.log(s2) + 2 * (p + 2)
        table[p] = float(aic)
        if best is None or aic < best[0]:
            best = (aic, p, beta, np.sqrt(resid @ resid / (target.size - p - 1)))
    _, p, beta, sig = best
    return ARModel(p=p, const=float(beta[0]), coefs=beta[1:].copy(), sigma=float(sig),
                   aic_table=table)


@njit(cache=True)
def _ar_sim(const, coefs, sigma, init, eps):
    n_series, total = eps.shape
    p = coefs.shape[0]
    out = np.empty((n_series, total))
    for s in range(n_series):
        buf = init[s].copy()  # most recent last
        for t in range(total):
            v = const
            for k in range(p):
                v += coefs[k] * buf[p - 1 - k]
            v += sigma * eps[s, t]
            for k in range(p - 1):
                buf[k] = buf[k + 1]
            buf[p - 1] = v
            out[s, t] = v
    return out


def simulate_ar(model: ARModel, y_train: np.ndarray, n_series: int, length: int, burn: int,
                rng: np.random.Generator) -> np.ndarray:
    """Simulate from the fitted AR(p), initialised from random train segments, with burn-in."""
    p = model.p
    starts = rng.integers(0, y_train.size - p, size=n_series)
    init = np.stack([y_train[s: s + p] for s in starts]).astype(np.float64)
    eps = rng.standard_normal((n_series, burn + length))
    sims = _ar_sim(model.const, model.coefs.astype(np.float64), model.sigma, init, eps)
    return sims[:, burn:]


def ar_forecast_ensemble(model: ARModel, history: np.ndarray, horizon: int, n_members: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Monte-Carlo predictive ensemble (n_members, horizon) from the last p values of history."""
    p = model.p
    init = np.repeat(np.asarray(history[-p:], dtype=np.float64)[None], n_members, axis=0)
    eps = rng.standard_normal((n_members, horizon))
    return _ar_sim(model.const, model.coefs.astype(np.float64), model.sigma, init, eps)
