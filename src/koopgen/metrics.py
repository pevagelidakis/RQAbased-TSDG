"""Linear and nonlinear time-series statistics, implemented from scratch (numba where hot).

All functions take a 1-D float array. Nonlinear statistics take the metric embedding (m, tau)
explicitly so that every generator is measured identically.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy import signal

from .embedding import acf as _acf
from .embedding import delay_embed


# ----------------------------------------------------------------------------- distances
@njit(parallel=True, cache=True)
def pairwise_distances(X):
    n, m = X.shape
    D = np.empty((n, n))
    for i in prange(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            s = 0.0
            for k in range(m):
                d = X[i, k] - X[j, k]
                s += d * d
            D[i, j] = math.sqrt(s)
    for i in prange(n):
        for j in range(i):
            D[i, j] = D[j, i]
    return D


@njit(cache=True)
def _upper_distances(D, theiler):
    """Distances d_ij for j - i >= theiler (theiler >= 1)."""
    n = D.shape[0]
    cnt = 0
    for i in range(n):
        if n - (i + theiler) > 0:
            cnt += n - (i + theiler)
    out = np.empty(cnt)
    c = 0
    for i in range(n):
        for j in range(i + theiler, n):
            out[c] = D[i, j]
            c += 1
    return out


# ----------------------------------------------------------------------------- RQA
@njit(cache=True)
def _line_hist_diagonal(R, theiler):
    """Histogram of diagonal line lengths in the upper triangle (offsets k >= theiler).
    Because the RP is symmetric, ratios (DET, Lmean, ENTR) are identical to the full-matrix ones."""
    n = R.shape[0]
    hist = np.zeros(n + 1, dtype=np.int64)
    for k in range(theiler, n):
        run = 0
        for i in range(n - k):
            if R[i, i + k]:
                run += 1
            else:
                if run > 0:
                    hist[run] += 1
                run = 0
        if run > 0:
            hist[run] += 1
    return hist


@njit(cache=True)
def _line_hist_vertical(R):
    """Histograms of black vertical lines and of white vertical lines (recurrence times).
    White lines touching the matrix border are excluded (their length is censored)."""
    n = R.shape[0]
    black = np.zeros(n + 1, dtype=np.int64)
    white = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        run = 0
        wrun = 0
        seen_black = False
        for i in range(n):
            if R[i, j]:
                if wrun > 0 and seen_black:
                    white[wrun] += 1
                wrun = 0
                run += 1
                seen_black = True
            else:
                if run > 0:
                    black[run] += 1
                run = 0
                wrun += 1
        if run > 0:
            black[run] += 1
    return black, white


def rqa_from_rp(R: np.ndarray, theiler: int = 1, lmin: int = 2, vmin: int = 2) -> dict:
    """RQA measures from a boolean recurrence matrix.

    The Theiler band |i-j| < theiler is excluded from black-line statistics. For recurrence
    times (white vertical lines) the band is treated as recurrent, so that the trajectory point
    itself counts as a return (standard recurrence-time definition).
    """
    R = np.asarray(R, dtype=np.bool_)
    n = R.shape[0]
    Rb = R.copy()
    Rw = R.copy()
    for k in range(-(theiler - 1), theiler):
        idx = np.arange(max(0, -k), min(n, n - k))
        Rb[idx, idx + k] = False
        Rw[idx, idx + k] = True
    dh = _line_hist_diagonal(Rb, theiler)
    vb = _line_hist_vertical(Rb)[0]
    vw = _line_hist_vertical(Rw)[1]
    lengths = np.arange(n + 1)

    def ratio(h, lo):
        tot = np.sum(lengths * h)
        return float(np.sum(lengths[lo:] * h[lo:]) / tot) if tot > 0 else np.nan

    def mean_len(h, lo):
        c = h[lo:].sum()
        return float(np.sum(lengths[lo:] * h[lo:]) / c) if c > 0 else np.nan

    def entropy(h, lo):
        c = h[lo:].astype(float)
        s = c.sum()
        if s <= 0:
            return np.nan
        p = c[c > 0] / s
        return float(-np.sum(p * np.log(p)))

    nz = np.nonzero(dh)[0]
    off = np.ones((n, n), dtype=bool)
    for k in range(-(theiler - 1), theiler):
        idx = np.arange(max(0, -k), min(n, n - k))
        off[idx, idx + k] = False
    return {
        "rr": float(Rb[off].mean()),
        "det": ratio(dh, lmin),
        "lmean": mean_len(dh, lmin),
        "lmax": float(nz.max()) if nz.size else 0.0,
        "entr": entropy(dh, lmin),
        "lam": ratio(vb, vmin),
        "tt": mean_len(vb, vmin),
        "rte": entropy(vw, 1),
    }


def recurrence_plot(X: np.ndarray, rr: float, theiler: int = 1, D: np.ndarray | None = None):
    """Recurrence matrix at a fixed recurrence rate: eps = rr-quantile of off-band distances."""
    if D is None:
        D = pairwise_distances(np.ascontiguousarray(X, dtype=np.float64))
    d = _upper_distances(D, theiler)
    k = int(np.clip(round(rr * d.size), 1, d.size - 1))
    eps = float(np.partition(d, k)[k])
    return D <= eps, eps


def rqa(y: np.ndarray, m: int, tau: int, rr: float = 0.05, theiler: int = 1,
        lmin: int = 2, vmin: int = 2) -> dict:
    X = delay_embed(y, m, tau)
    R, eps = recurrence_plot(X, rr, theiler)
    out = rqa_from_rp(R, theiler, lmin, vmin)
    out["eps"] = eps
    return out


# ----------------------------------------------------------------------------- D2
def correlation_sum(X: np.ndarray, theiler: int, n_r: int = 40, D: np.ndarray | None = None):
    """Grassberger-Procaccia correlation sum C(r) on log-spaced radii spanning the distance range."""
    if D is None:
        D = pairwise_distances(np.ascontiguousarray(X, dtype=np.float64))
    d = np.sort(_upper_distances(D, max(1, theiler + 1)))
    d = d[d > 0]
    r = np.logspace(np.log10(d[max(0, int(1e-5 * d.size))]), np.log10(d[-1]), n_r)
    C = np.searchsorted(d, r, side="right") / d.size
    return r, C


def correlation_dimension(y: np.ndarray, m: int, tau: int, theiler: int,
                          c_lo: float = 1e-3, c_hi: float = 5e-2, n_r: int = 40,
                          return_curve: bool = False):
    """D2 = slope of log C(r) vs log r over the pre-registered scaling region c_lo <= C <= c_hi.
    The region is defined by correlation-sum level (equivalently, distance quantiles), which
    makes it automatic and identical for every series. Requires >= 4 radii in the region."""
    X = delay_embed(y, m, tau)
    r, C = correlation_sum(X, theiler, n_r)
    sel = (C >= c_lo) & (C <= c_hi) & (C > 0)
    if sel.sum() < 4:
        d2 = np.nan
        fit = (np.nan, np.nan)
    else:
        lr, lc = np.log(r[sel]), np.log(C[sel])
        d2 = float(np.polyfit(lr, lc, 1)[0])
        fit = (float(r[sel][0]), float(r[sel][-1]))
    if return_curve:
        return d2, {"r": r, "C": C, "fit_range": fit}
    return d2


# ----------------------------------------------------------------------------- Rosenstein
@njit(parallel=True, cache=True)
def _rosenstein_curve(X, theiler, k_max):
    n = X.shape[0]
    m = X.shape[1]
    nn = np.full(n, -1, dtype=np.int64)
    for i in prange(n - k_max):
        best = 1e300
        bj = -1
        for j in range(n - k_max):
            if abs(i - j) <= theiler:
                continue
            s = 0.0
            for c in range(m):
                d = X[i, c] - X[j, c]
                s += d * d
            if s > 0.0 and s < best:
                best = s
                bj = j
        nn[i] = bj
    curve = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        tot = 0.0
        cnt = 0
        for i in range(n - k_max):
            j = nn[i]
            if j < 0:
                continue
            s = 0.0
            for c in range(m):
                d = X[i + k, c] - X[j + k, c]
                s += d * d
            if s > 0.0:
                tot += 0.5 * math.log(s)
                cnt += 1
        curve[k] = tot / cnt if cnt > 0 else np.nan
    return curve


def lyapunov_rosenstein(y: np.ndarray, m: int, tau: int, theiler: int, k_max: int,
                        rise_lo: float = 0.0, rise_hi: float = 0.5, return_curve: bool = False):
    """Largest Lyapunov exponent (per sample) by Rosenstein et al. (1993).

    Fit range (pre-registered, automatic): k from the first index where the mean log-divergence
    has risen by rise_lo of its total rise (max - initial) to the first index where it has risen
    by rise_hi. At least 3 points are used.
    """
    X = np.ascontiguousarray(delay_embed(y, m, tau))
    curve = _rosenstein_curve(X, int(theiler), int(k_max))
    c0 = curve[0]
    top = np.nanmax(curve)
    rise = top - c0
    if not np.isfinite(rise) or rise <= 0:
        lam, fit = np.nan, (0, 0)
    else:
        a = int(np.argmax(curve - c0 >= rise_lo * rise))
        b = int(np.argmax(curve - c0 >= rise_hi * rise))
        b = max(b, a + 2)
        k = np.arange(a, b + 1)
        lam = float(np.polyfit(k, curve[a: b + 1], 1)[0])
        fit = (a, b)
    if return_curve:
        return lam, {"k": np.arange(k_max + 1), "curve": curve, "fit_range": fit}
    return lam


def mean_period(y: np.ndarray) -> float:
    """Mean period 1 / (power-weighted mean frequency), in samples (Rosenstein's Theiler window)."""
    f, p = signal.periodogram(np.asarray(y) - np.mean(y))
    f, p = f[1:], p[1:]
    return float(1.0 / (np.sum(f * p) / np.sum(p)))


# ----------------------------------------------------------------------------- entropies
@njit(cache=True)
def _sampen_counts(y, m, r):
    n = y.shape[0]
    B = 0
    A = 0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            ok = True
            for k in range(m):
                if abs(y[i + k] - y[j + k]) > r:
                    ok = False
                    break
            if ok:
                B += 1
                if abs(y[i + m] - y[j + m]) <= r:
                    A += 1
    return A, B


def sample_entropy(y: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    y = np.ascontiguousarray(y, dtype=np.float64)
    A, B = _sampen_counts(y, m, r_frac * np.std(y))
    if A == 0 or B == 0:
        return np.nan
    return float(-np.log(A / B))


def permutation_entropy(y: np.ndarray, order: int = 4, delay: int = 1) -> float:
    """Normalised permutation entropy (Bandt & Pompe), in [0, 1]."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0] - (order - 1) * delay
    patt = np.column_stack([y[i * delay: i * delay + n] for i in range(order)])
    ranks = np.argsort(patt, axis=1, kind="stable")
    code = np.zeros(n, dtype=np.int64)
    for c in range(order):
        code = code * order + ranks[:, c]
    _, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / math.log(math.factorial(order)))


# ----------------------------------------------------------------------------- higher order
def time_reversal_asymmetry(y: np.ndarray, lags) -> np.ndarray:
    """<(y_{t+k} - y_t)^3> for each k (y already on the common train-standardised scale)."""
    y = np.asarray(y, dtype=np.float64)
    return np.array([np.mean((y[k:] - y[:-k]) ** 3) for k in lags])


def third_order_autocov(y: np.ndarray, lags) -> np.ndarray:
    """C3(k) = <(y_t - ybar)(y_{t-k} - ybar)(y_{t-2k} - ybar)>."""
    y = np.asarray(y, dtype=np.float64)
    z = y - y.mean()
    return np.array([np.mean(z[2 * k:] * z[k:-k] * z[:-2 * k]) for k in lags])


# ----------------------------------------------------------------------------- linear
def acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    return _acf(y, max_lag)[1:]


def pacf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF at lags 1..max_lag via Durbin-Levinson on the biased ACF."""
    r = _acf(y, max_lag)
    phi = np.zeros((max_lag + 1, max_lag + 1))
    out = np.zeros(max_lag)
    phi[1, 1] = r[1]
    out[0] = r[1]
    v = 1.0 - r[1] ** 2
    for k in range(2, max_lag + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[1:k][::-1])
        phi[k, k] = num / v if v > 0 else 0.0
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        v *= 1.0 - phi[k, k] ** 2
        out[k - 1] = phi[k, k]
    return out


def welch_log_psd(y: np.ndarray, nperseg: int):
    f, p = signal.welch(np.asarray(y, dtype=np.float64), nperseg=nperseg, detrend="constant")
    return f, np.log10(np.maximum(p, 1e-300))


def ks_distance(a: np.ndarray, b_sorted: np.ndarray) -> float:
    """Two-sample KS statistic; b_sorted must be sorted (the train reference)."""
    a = np.sort(np.asarray(a))
    grid = np.concatenate([a, b_sorted])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b_sorted, grid, side="right") / b_sorted.size
    return float(np.max(np.abs(fa - fb)))


def hankel_dmd_frequency(y: np.ndarray, rows: int, rank: int, f_min: float) -> float:
    """Dominant Koopman frequency (cycles/sample) from Hankel-DMD of the window: the frequency of
    the mode with the largest amplitude |b_i|·||phi_i|| among modes with frequency >= f_min."""
    y = np.asarray(y, dtype=np.float64)
    H = np.lib.stride_tricks.sliding_window_view(y, rows).T  # rows x n_cols
    X, Y = H[:, :-1], H[:, 1:]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, int(np.sum(s > s[0] * 1e-10)))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    At = U.T @ Y @ Vt.T / s
    mu, Wv = np.linalg.eig(At)
    modes = Y @ Vt.T / s @ Wv
    b = np.linalg.lstsq(modes, X[:, 0].astype(complex), rcond=None)[0]
    amp = np.abs(b) * np.linalg.norm(modes, axis=0)
    freq = np.abs(np.angle(mu)) / (2 * np.pi)
    ok = freq >= f_min
    if not ok.any():
        return 0.0
    return float(freq[ok][np.argmax(amp[ok])])


# ----------------------------------------------------------------------------- forecast scoring
def crps_ensemble(ens: np.ndarray, obs: float | np.ndarray) -> np.ndarray:
    """CRPS of an ensemble (last axis = members): E|X-y| - 0.5 E|X-X'| (sorted O(n log n) form)."""
    ens = np.sort(np.asarray(ens, dtype=np.float64), axis=-1)
    n = ens.shape[-1]
    obs = np.asarray(obs, dtype=np.float64)
    t1 = np.mean(np.abs(ens - obs[..., None]), axis=-1)
    i = np.arange(1, n + 1)
    t2 = np.sum(ens * (2 * i - n - 1), axis=-1) / (n * n)
    return t1 - t2
