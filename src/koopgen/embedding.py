"""Delay embedding and selection of (m, tau): AMI, Cao's method, false nearest neighbours.

Convention: x_t = (y_t, y_{t-tau}, ..., y_{t-(m-1)tau}); valid for t >= (m-1)*tau.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


def delay_embed(y: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Return the (N - (m-1)tau, m) matrix whose row r is x_{r + (m-1)tau}."""
    y = np.asarray(y, dtype=np.float64)
    span = (m - 1) * tau
    n = y.shape[0] - span
    if n <= 0:
        raise ValueError(f"series of length {y.shape[0]} too short for m={m}, tau={tau}")
    cols = [y[span - j * tau: span - j * tau + n] for j in range(m)]
    return np.column_stack(cols)


def regression_pairs(y: np.ndarray, m: int, tau: int):
    """States x_t, targets y_{t+1} and next states x_{t+1} for t = (m-1)tau .. N-2."""
    X_all = delay_embed(y, m, tau)
    X, X_next = X_all[:-1], X_all[1:]
    target = X_next[:, 0].copy()
    return X, target, X_next


def acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation at lags 0..max_lag (FFT based)."""
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    n = y.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(y, nfft)
    r = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return r / r[0]


def ami(y: np.ndarray, max_lag: int, bins: int = 16) -> np.ndarray:
    """Average mutual information I(y_t; y_{t+k}) in nats for k = 0..max_lag (histogram estimator,
    equiprobable-in-range bins over the min..max of the series)."""
    y = np.asarray(y, dtype=np.float64)
    lo, hi = y.min(), y.max()
    idx = np.minimum(((y - lo) / (hi - lo) * bins).astype(np.int64), bins - 1)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        a, b = idx[: len(y) - k], idx[k:]
        joint = np.zeros((bins, bins))
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        pa, pb = joint.sum(1), joint.sum(0)
        nz = joint > 0
        out[k] = np.sum(joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz]))
    return out


def first_local_minimum(v: np.ndarray) -> int | None:
    for k in range(1, len(v) - 1):
        if v[k] < v[k - 1] and v[k] <= v[k + 1]:
            return k
    return None


def first_zero_crossing(r: np.ndarray) -> int | None:
    idx = np.nonzero(r[1:] <= 0)[0]
    return int(idx[0] + 1) if idx.size else None


@njit(parallel=True, cache=True)
def _cao_fnn_kernel(y, tau, d_max, ref_idx, n_cand, theiler, rtol, atol):
    """For d = 1..d_max: Cao's E(d), E*(d) and the Kennel FNN fraction.

    Forward embedding v_i^d = (y_i, y_{i+tau}, ..., y_{i+(d-1)tau}); candidate neighbours are
    i in [0, n_cand), where n_cand is chosen so that index i + d_max*tau is valid.
    """
    n_ref = ref_idx.shape[0]
    E = np.zeros(d_max)
    Estar = np.zeros(d_max)
    fnn = np.zeros(d_max)
    sd = np.std(y)
    for d in range(1, d_max + 1):
        a_vals = np.zeros(n_ref)
        e_vals = np.zeros(n_ref)
        f_vals = np.zeros(n_ref)
        for r in prange(n_ref):
            i = ref_idx[r]
            best_inf = 1e300
            j_inf = -1
            best_euc = 1e300
            j_euc = -1
            for j in range(n_cand):
                if abs(i - j) <= theiler:
                    continue
                dinf = 0.0
                deuc = 0.0
                for k in range(d):
                    diff = abs(y[i + k * tau] - y[j + k * tau])
                    if diff > dinf:
                        dinf = diff
                    deuc += diff * diff
                if dinf > 0.0 and dinf < best_inf:
                    best_inf = dinf
                    j_inf = j
                if deuc > 0.0 and deuc < best_euc:
                    best_euc = deuc
                    j_euc = j
            # Cao
            extra = abs(y[i + d * tau] - y[j_inf + d * tau])
            a_vals[r] = max(best_inf, extra) / best_inf
            e_vals[r] = extra
            # Kennel FNN
            rd = np.sqrt(best_euc)
            extra_e = abs(y[i + d * tau] - y[j_euc + d * tau])
            rd1 = np.sqrt(best_euc + extra_e * extra_e)
            if extra_e / rd > rtol or rd1 / sd > atol:
                f_vals[r] = 1.0
        E[d - 1] = np.mean(a_vals)
        Estar[d - 1] = np.mean(e_vals)
        fnn[d - 1] = np.mean(f_vals)
    return E, Estar, fnn


@dataclass
class EmbeddingChoice:
    tau: int
    m: int
    tau_method: str
    m_method: str
    ami: np.ndarray = field(repr=False)
    acf: np.ndarray = field(repr=False)
    dims: np.ndarray = field(repr=False)
    cao_E1: np.ndarray = field(repr=False)
    cao_E2: np.ndarray = field(repr=False)
    fnn: np.ndarray = field(repr=False)
    ami_floor: float = 0.0
    notes: list = field(default_factory=list)


def cao_fnn(y, tau, d_max=10, max_points=3000, theiler=None, rtol=15.0, atol=2.0, seed=0):
    """Return dims (1..d_max-1), E1, E2, FNN fraction."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_cand = y.shape[0] - (d_max + 1) * tau
    if n_cand < 50:
        raise ValueError("series too short for Cao/FNN at this d_max and tau")
    if theiler is None:
        theiler = tau
    rng = np.random.default_rng(seed)
    if n_cand > max_points:
        ref_idx = np.sort(rng.choice(n_cand, size=max_points, replace=False)).astype(np.int64)
    else:
        ref_idx = np.arange(n_cand, dtype=np.int64)
    E, Estar, fnn = _cao_fnn_kernel(y, tau, d_max + 1, ref_idx, n_cand, theiler, rtol, atol)
    E1 = E[1:] / E[:-1]
    E2 = Estar[1:] / Estar[:-1]
    dims = np.arange(1, d_max + 1)
    return dims, E1, E2, fnn[:-1]


def choose_embedding(y_train: np.ndarray, cfg: dict, seed: int = 0) -> EmbeddingChoice:
    """Select tau (first AMI minimum, ACF-zero fallback) and m (Cao E1 threshold) on train data."""
    max_lag = int(cfg["max_lag"])
    notes = []
    bins = int(cfg["ami_bins"])
    a = ami(y_train, max_lag, bins=bins)
    r = acf(y_train, max_lag)
    # Histogram AMI of independent variables is biased upward by ~ (B-1)^2 / (2N) nats; a
    # "minimum" inside a small multiple of that floor is estimator noise, not structure.
    floor = (bins - 1) ** 2 / (2.0 * y_train.size)
    tau = first_local_minimum(a)
    tau_method = "first AMI minimum"
    if tau is not None and a[tau] <= float(cfg["ami_floor_factor"]) * floor:
        notes.append(f"First AMI minimum (lag {tau}, {a[tau]:.4f} nats) lies within "
                     f"{cfg['ami_floor_factor']}x the estimator bias floor ({floor:.4f}); rejected.")
        tau = None
    if tau is None:
        tau = first_zero_crossing(r)
        tau_method = "first ACF zero crossing (fallback: no AMI minimum above the bias floor)"
        notes.append("Used the first ACF zero crossing for tau (fallback).")
    if tau is None:
        raise RuntimeError(
            f"Neither an AMI minimum nor an ACF zero crossing was found within {max_lag} lags; "
            "cannot choose tau defensibly (series may be dominated by a trend).")
    if cfg.get("tau_override"):
        tau = int(cfg["tau_override"])
        tau_method = "override from config"
    def cao_choice(tau_):
        dims_, E1_, E2_, fnn_ = cao_fnn(y_train, tau_, d_max=int(cfg["m_max"]),
                                        max_points=int(cfg["cao_max_points"]), seed=seed)
        thr = float(cfg["cao_threshold"])
        sat_tol, sat_min = float(cfg["cao_saturation_tol"]), float(cfg["cao_saturation_min"])
        dE = np.append(np.abs(np.diff(E1_)), np.inf)
        hit = np.nonzero((E1_ >= thr) | ((E1_ >= sat_min) & (dE <= sat_tol)))[0]
        if hit.size:
            m_ = int(dims_[hit[0]])
            how = (f"Cao: smallest d with E1(d) >= {thr}, or E1(d) >= {sat_min} and "
                   f"|E1(d+1)-E1(d)| <= {sat_tol} (saturation)")
        else:
            m_ = int(dims_[-1])
            how = f"Cao E1 never reached {thr}; used m_max"
        return dims_, E1_, E2_, fnn_, m_, how

    dims, E1, E2, fnn, m, m_method = cao_choice(tau)
    # Delay-time selection unfolds a deterministic attractor. If Cao's E2 test finds no
    # determinism (E2(d) ~ 1 for every d), there is nothing to unfold and the Markov state of the
    # process is a block of consecutive lags, so tau = 1 is used (rule fixed before the benchmark
    # runs; see CHANGELOG_METHOD.md).
    stochastic = bool(np.all(np.abs(E2 - 1.0) < float(cfg["cao_e2_tol"])))
    if stochastic:
        notes.append(f"Cao E2(d) within {cfg['cao_e2_tol']} of 1 for all d: no detectable determinism "
                     "(consistent with a stochastic process).")
        if tau != 1 and not cfg.get("tau_override"):
            notes.append(f"Stochastic-series rule: tau {tau} -> 1 (consecutive-lag state).")
            tau, tau_method = 1, "1 (stochastic-series rule: Cao E2 ~ 1 for all d)"
            dims, E1, E2, fnn, m, m_method = cao_choice(tau)
    if m_method.startswith("Cao E1 never"):
        notes.append(m_method)
    m = max(m, int(cfg.get("m_min", 2)))
    if cfg.get("m_override"):
        m = int(cfg["m_override"])
        m_method = "override from config"
    return EmbeddingChoice(tau=int(tau), m=int(m), tau_method=tau_method, m_method=m_method,
                           ami_floor=floor, ami=a, acf=r, dims=dims, cao_E1=E1, cao_E2=E2, fnn=fnn, notes=notes)
