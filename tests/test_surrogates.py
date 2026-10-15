import numpy as np
import pytest

from koopgen import data, surrogates as S
from koopgen import metrics as M


def test_iaaft_marginal_exact_and_spectrum_close():
    rng = np.random.default_rng(0)
    y = data.henon_x(2048, 200, rng, obs_noise=0.0)
    surr = S.iaaft(y, 5, rng)
    amp = np.abs(np.fft.rfft(y))
    for s in surr:
        assert np.array_equal(np.sort(s), np.sort(y))
        a = np.abs(np.fft.rfft(s))
        assert np.linalg.norm(a - amp) / np.linalg.norm(amp) < 0.05
    # linear structure kept: ACF close
    assert np.max(np.abs(M.acf(surr[0], 10) - M.acf(y, 10))) < 0.05


def test_iaaft_destroys_henon_nonlinearity():
    rng = np.random.default_rng(1)
    y = data.henon_x(2000, 200, rng, obs_noise=0.0)
    det_real = M.rqa(y, 2, 1)["det"]
    det_s = [M.rqa(s, 2, 1)["det"] for s in S.iaaft(y, 5, rng)]
    assert det_real > max(det_s) + 0.2


def test_iaaft_gaussian_linear_null_not_distinguishable():
    rng = np.random.default_rng(2)
    y = data.ar2(2000, 200, rng)
    tra = M.time_reversal_asymmetry(y, [1])[0]
    tra_s = np.array([M.time_reversal_asymmetry(s, [1])[0] for s in S.iaaft(y, 99, rng)])
    rank = np.mean(tra_s < tra)
    assert 0.01 < rank < 0.99


def test_shuffle_preserves_marginal_destroys_acf():
    rng = np.random.default_rng(3)
    y = data.ar2(3000, 100, rng)
    sh = S.shuffle(y, 3, rng)
    assert np.array_equal(np.sort(sh[1]), np.sort(y))
    assert abs(M.acf(sh[0], 1)[0]) < 0.06 and M.acf(y, 1)[0] > 0.6


def test_fit_ar_aic_recovers_ar2():
    rng = np.random.default_rng(4)
    y = data.ar2(20000, 200, rng, 1.2, -0.6)
    mod = S.fit_ar_aic(y, 10)
    assert mod.p >= 2
    assert mod.coefs[:2] == pytest.approx([1.2, -0.6], abs=0.03)
    assert mod.sigma == pytest.approx(1.0, abs=0.03)
    sims = S.simulate_ar(mod, y, 3, 5000, 200, rng)
    assert sims.shape == (3, 5000)
    assert M.acf(sims[0], 2) == pytest.approx(M.acf(y, 2), abs=0.05)


def test_endpoint_matching():
    y = np.sin(np.arange(1000) * 0.1)
    i0, i1 = S.endpoint_matched_segment(y, 500)
    assert i1 - i0 == 500
    assert abs(y[i0] - y[i1 - 1]) < 0.1
