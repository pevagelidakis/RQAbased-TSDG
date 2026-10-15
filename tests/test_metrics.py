"""Unit tests for the from-scratch metric implementations against known values."""
import math

import numpy as np
import pytest

from koopgen import data
from koopgen import metrics as M
from koopgen.embedding import delay_embed


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def henon():
    y = data.henon_x(4000, 500, np.random.default_rng(101), obs_noise=0.0)
    return (y - y.mean()) / y.std()


@pytest.fixture(scope="module")
def lorenz():
    y = data.lorenz_x(6000, 0.05, 50.0, np.random.default_rng(202))
    return (y - y.mean()) / y.std()


# ----------------------------------------------------------------------------- RQA on toy RPs
def test_rqa_hand_computed_toy():
    # 6x6 symmetric RP, LOI excluded (theiler=1). Upper triangle recurrences:
    # offset 1: (0,1),(1,2),(2,3) -> one diagonal line of length 3
    # offset 3: (0,3)            -> isolated point (length 1)
    # offset 2: (3,5)            -> isolated point (length 1)
    R = np.zeros((6, 6), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 3), (3, 5)]:
        R[i, j] = R[j, i] = True
    out = M.rqa_from_rp(R, theiler=1, lmin=2, vmin=2)
    assert out["det"] == pytest.approx(3 / 5)          # 3 of 5 upper-triangle points on lines >= 2
    assert out["lmean"] == pytest.approx(3.0)
    assert out["lmax"] == 3
    assert out["entr"] == pytest.approx(0.0)             # a single line length
    assert out["rr"] == pytest.approx(10 / 30)           # 10 recurrent of 30 off-diagonal cells
    # vertical black lines (full matrix, LOI removed): column 3 has rows 0,2 -> runs (0),(2) and
    # row 5 -> lengths 1,1,1; column 1 rows 0,2 -> 1,1; column 2 rows 1,3 -> 1,1; column 0 rows 1,3;
    # column 5 row 3. No run >= 2  -> LAM = 0.
    assert out["lam"] == 0.0
    assert math.isnan(out["tt"])


def test_rqa_vertical_lines_and_recurrence_times():
    # column 0 has a vertical line of length 3 (rows 2,3,4); everything else empty (LOI excluded).
    R = np.zeros((8, 8), dtype=bool)
    for i in (2, 3, 4):
        R[i, 0] = R[0, i] = True
    out = M.rqa_from_rp(R, theiler=1)
    # vertical: column 0 -> one run of 3; columns 2,3,4 each one point -> LAM = 3/6
    assert out["lam"] == pytest.approx(3 / 6)
    assert out["tt"] == pytest.approx(3.0)
    # recurrence times (LOI treated as recurrent): column 0 = rows 0,2,3,4 -> white gap of 1
    # (row 1); columns 2,3,4: rows {0,c} -> gaps 1, 2, 3. Entropy of {1:2, 2:1, 3:1}.
    p = np.array([2, 1, 1]) / 4
    assert out["rte"] == pytest.approx(-np.sum(p * np.log(p)))


def test_rqa_periodic_is_deterministic():
    t = np.arange(1200)
    y = np.sin(2 * np.pi * t / 25.0)
    out = M.rqa(y, m=2, tau=6, rr=0.05)
    assert out["rr"] == pytest.approx(0.05, abs=0.005)
    # exact recurrences are 4% of pairs; the remaining 1% are near-ties between neighbouring
    # phases, which form a few short broken lines, so DET is high but not exactly 1.
    assert out["det"] > 0.95
    assert out["lmax"] > 1000


def test_rqa_iid_noise_analytic_det():
    # m = 1, iid noise: R_{i+1,j+1} is independent of R_{ij}, so a recurrent point lies on a
    # line of length >= 2 with probability 1 - (1 - RR)^2 (either diagonal neighbour recurrent).
    y = np.random.default_rng(0).standard_normal(1500)
    out = M.rqa(y, m=1, tau=1, rr=0.05)
    assert out["det"] == pytest.approx(1 - 0.95 ** 2, abs=0.01)
    # (LAM has no such closed form: vertical neighbours share y_j and are positively correlated.)


def _naive_rqa(R, theiler, lmin):
    n = R.shape[0]
    lines = []
    for k in range(theiler, n):
        run = 0
        for i in range(n - k):
            if R[i, i + k]:
                run += 1
            else:
                if run:
                    lines.append(run)
                run = 0
        if run:
            lines.append(run)
    vlines = []
    for j in range(n):
        run = 0
        for i in range(n):
            if R[i, j] and abs(i - j) >= theiler:
                run += 1
            else:
                if run:
                    vlines.append(run)
                run = 0
        if run:
            vlines.append(run)
    lines, vlines = np.array(lines), np.array(vlines)
    det = lines[lines >= lmin].sum() / lines.sum()
    lam = vlines[vlines >= lmin].sum() / vlines.sum()
    return det, lam, lines.max()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rqa_matches_naive_reference(seed):
    rng = np.random.default_rng(seed)
    A = rng.random((40, 40)) < 0.3
    R = A | A.T
    for theiler in (1, 3):
        out = M.rqa_from_rp(R, theiler=theiler)
        det, lam, lmax = _naive_rqa(R, theiler, 2)
        assert out["det"] == pytest.approx(det)
        assert out["lam"] == pytest.approx(lam)
        assert out["lmax"] == lmax


def test_rqa_henon_vs_noise(henon):
    det_h = M.rqa(henon[:1000], 2, 1, rr=0.05)["det"]
    det_n = M.rqa(np.random.default_rng(1).standard_normal(1000), 2, 1, rr=0.05)["det"]
    assert det_h > 0.6 and det_h > det_n + 0.3


# ----------------------------------------------------------------------------- invariants
def test_d2_henon(henon):
    d2 = M.correlation_dimension(henon[:3000], 2, 1, theiler=4)
    assert d2 == pytest.approx(1.21, abs=0.08)


def test_d2_lorenz(lorenz):
    d2 = M.correlation_dimension(lorenz[:3000], 3, 4, theiler=33)
    assert d2 == pytest.approx(2.05, abs=0.2)


def test_d2_noise_fills_embedding():
    y = np.random.default_rng(3).standard_normal(2000)
    d2 = M.correlation_dimension(y, 3, 1, theiler=1)
    assert d2 > 2.6


def test_lyapunov_henon(henon):
    lam = M.lyapunov_rosenstein(henon[:3000], 2, 1, theiler=4, k_max=30, rise_lo=0.2, rise_hi=0.7)
    assert lam == pytest.approx(0.419, rel=0.08)


def test_lyapunov_lorenz(lorenz):
    mp = M.mean_period(lorenz)
    lam = M.lyapunov_rosenstein(lorenz[:4000], 3, 4, theiler=int(np.ceil(mp)),
                                k_max=int(np.ceil(5 * mp)), rise_lo=0.2, rise_hi=0.7)
    assert lam / 0.05 == pytest.approx(0.906, rel=0.15)


def test_lyapunov_periodic_near_zero():
    y = np.sin(2 * np.pi * np.arange(3000) / 40.0)
    y = y + 1e-6 * np.random.default_rng(0).standard_normal(y.size)
    lam = M.lyapunov_rosenstein(y, 2, 10, theiler=40, k_max=200, rise_lo=0.2, rise_hi=0.7)
    assert abs(lam) < 0.01


# ----------------------------------------------------------------------------- entropies
def test_sample_entropy_white_noise():
    # For iid N(0,1), SampEn(m, r) ~ -ln P(|x-y| <= r) with r = 0.2 sd: P = erf(0.2/2)... use
    # the exact value 2*Phi(r/sqrt2) - 1 for the difference of two N(0,1).
    from scipy.stats import norm
    r = 0.2
    p = 2 * norm.cdf(r / np.sqrt(2)) - 1
    y = np.random.default_rng(4).standard_normal(3000)
    assert M.sample_entropy(y, 2, 0.2) == pytest.approx(-np.log(p), rel=0.06)


def test_sample_entropy_matches_naive():
    y = data.henon_x(300, 100, np.random.default_rng(11), obs_noise=0.0)
    r = 0.2 * y.std()
    n, m = y.size, 2
    B = A = 0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            if np.max(np.abs(y[i:i + m] - y[j:j + m])) <= r:
                B += 1
                A += abs(y[i + m] - y[j + m]) <= r
    assert M.sample_entropy(y, 2, 0.2) == pytest.approx(-np.log(A / B))


def test_sample_entropy_regular_below_chaotic():
    sine = np.sin(2 * np.pi * np.arange(2000) / 100)
    hen = data.henon_x(2000, 100, np.random.default_rng(12), obs_noise=0.0)
    assert M.sample_entropy(sine, 2, 0.2) < 0.5 * M.sample_entropy(hen, 2, 0.2)


def test_permutation_entropy_bounds():
    noise = np.random.default_rng(5).standard_normal(20000)
    assert M.permutation_entropy(noise, 4, 1) == pytest.approx(1.0, abs=0.01)
    assert M.permutation_entropy(np.arange(500.0), 4, 1) == pytest.approx(0.0, abs=1e-12)


def test_permutation_entropy_hand():
    # order 2: patterns in [1,3,2,4,5] -> up, down, up, up -> p = (3/4, 1/4)
    p = np.array([0.75, 0.25])
    expected = -np.sum(p * np.log(p)) / np.log(2)
    assert M.permutation_entropy(np.array([1.0, 3.0, 2.0, 4.0, 5.0]), 2, 1) == pytest.approx(expected)


# ----------------------------------------------------------------------------- higher-order
def test_time_reversal_asymmetry():
    gauss = np.random.default_rng(6).standard_normal(20000)
    assert np.all(np.abs(M.time_reversal_asymmetry(gauss, [1, 2, 3])) < 0.1)
    saw = np.tile(np.r_[np.linspace(0, 1, 9, endpoint=False), [0.0]], 300)
    tra = M.time_reversal_asymmetry(saw, [1])[0]
    # slow rise (+1/9 nine times) then sudden drop (-0.888..): <d^3> is strongly negative
    d = np.diff(saw)
    assert tra == pytest.approx(np.mean(d ** 3)) and tra < -0.05
    # reversing time flips the sign
    assert M.time_reversal_asymmetry(saw[::-1], [1])[0] == pytest.approx(-tra)


def test_third_order_autocov_hand():
    y = np.array([1.0, 2.0, 0.0, 3.0, -1.0])
    z = y - y.mean()
    expected = np.mean(z[2:] * z[1:-1] * z[:-2])
    assert M.third_order_autocov(y, [1])[0] == pytest.approx(expected)


# ----------------------------------------------------------------------------- linear
def test_acf_pacf_ar1():
    rng = np.random.default_rng(7)
    e = rng.standard_normal(50000)
    y = np.zeros_like(e)
    for t in range(1, y.size):
        y[t] = 0.7 * y[t - 1] + e[t]
    a = M.acf(y, 3)
    assert a == pytest.approx([0.7, 0.49, 0.343], abs=0.02)
    p = M.pacf(y, 3)
    assert p[0] == pytest.approx(0.7, abs=0.02) and np.all(np.abs(p[1:]) < 0.02)


def test_pacf_matches_statsmodels():
    from statsmodels.tsa.stattools import pacf as sm_pacf
    y = data.ar2(3000, 100, np.random.default_rng(8))
    assert M.pacf(y, 10) == pytest.approx(sm_pacf(y, 10, method="ldb")[1:], abs=1e-10)


def test_ks_distance():
    a = np.array([0.0, 1.0, 2.0])
    b = np.sort(np.array([0.5, 1.5, 2.5, 3.5]))
    from scipy.stats import ks_2samp
    assert M.ks_distance(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_hankel_dmd_frequency_sinusoid():
    t = np.arange(1000)
    y = np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.13 * t)
    f = M.hankel_dmd_frequency(y, rows=40, rank=8, f_min=0.005)
    assert f == pytest.approx(0.05, abs=1e-3)


def test_crps_ensemble():
    ens = np.array([0.0, 1.0])
    # E|X-y| for y=0.25: (0.25 + 0.75)/2 = 0.5; 0.5 E|X-X'| = 0.5 * 0.5 = 0.25
    assert M.crps_ensemble(ens, 0.25) == pytest.approx(0.25)
    big = np.random.default_rng(9).standard_normal(200000)
    # CRPS of N(0,1) at 0 is 2 phi(0) - 1/sqrt(pi)
    assert M.crps_ensemble(big, 0.0) == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi), abs=5e-3)


def test_delay_embed_convention():
    y = np.arange(10.0)
    X = delay_embed(y, 3, 2)
    assert X.shape == (6, 3)
    assert X[0].tolist() == [4.0, 2.0, 0.0]   # (y_t, y_{t-tau}, y_{t-2tau}) at t = 4
