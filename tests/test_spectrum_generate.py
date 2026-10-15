import numpy as np
import pytest

from koopgen import embedding as E
from koopgen import features as Fe
from koopgen import generate as G
from koopgen import model as Mo
from koopgen import spectrum as S


def _linear_delay_series(n=600, a=1.6, b=-0.8):
    y = np.zeros(n)
    y[0], y[1] = 1.0, 0.5
    for t in range(2, n):
        y[t] = a * y[t - 1] + b * y[t - 2]
    return y / np.abs(y).max()


def test_linear_eigenvalues_recovered_with_zero_residual():
    a, b = 1.6, -0.81
    y = _linear_delay_series(400, a, b)
    X, _, Xn = E.regression_pairs(y, 2, 1)
    rff = Fe.draw_rff(2, 30, 0)
    theta = np.array([0.0, np.log(1e-10), np.log(1e6), np.log(1e-4)])  # rff weights shrunk
    d = S.draw_spectrum(theta, rff, X, Xn)
    roots = np.roots([1.0, -a, -b])
    for r in roots:
        k = np.argmin(np.abs(d.mu - r))
        assert abs(d.mu[k] - r) < 1e-6
        assert d.res[k] < 1e-5


def test_readout_column_equals_posterior_mean():
    rng = np.random.default_rng(1)
    y = np.sin(np.arange(300) / 5) + 0.05 * rng.standard_normal(300)
    X, tgt, Xn = E.regression_pairs(y, 3, 2)
    rff = Fe.draw_rff(3, 40, 1)
    theta = np.array([0.3, -1.0, 0.5, -2.0])
    d = S.draw_spectrum(theta, rff, X, Xn)
    assert d.readout == pytest.approx(Mo.posterior_mean_weights(theta, X, tgt, rff), abs=1e-8)


def test_resdmd_gram_equals_direct():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(500)
    X, _, Xn = E.regression_pairs(y, 2, 1)
    rff = Fe.draw_rff(2, 15, 2)
    theta = np.array([0.5, 0.0, 0.0, 0.0])
    K, PX, PY = S.koopman_matrix(theta, rff, X, Xn)
    mu, V = np.linalg.eig(K)
    assert S.resdmd_gram(mu, V, PX, PY) == pytest.approx(S.resdmd_direct(mu, V, PX, PY), rel=1e-6)


def test_generator_matches_numpy_relifting_loop():
    rng = np.random.default_rng(3)
    m, tau = 3, 2
    span = (m - 1) * tau
    rff = Fe.draw_rff(m, 25, 3)
    M, T = 4, 30
    thetas = np.column_stack([rng.normal(0, 0.2, M), np.zeros(M), np.zeros(M), np.full(M, -2.0)])
    weights = rng.normal(0, 0.3, (M, rff.P))
    weights[:, 1] = 0.5
    buffers = rng.standard_normal((M, span + 1))
    innov = 0.1 * rng.standard_normal((M, T))
    ys, div, _ = G.simulate(buffers, thetas, weights, innov, rff, m, tau, threshold=1e6)
    for i in range(M):
        buf = list(buffers[i])
        for t in range(T):
            x = np.array([buf[-1 - j * tau] for j in range(m)])
            nxt = G.readout_np(x[None], thetas[i], weights[i], rff)[0] + innov[i, t]
            assert ys[i, t] == pytest.approx(nxt, abs=1e-10)
            buf.append(nxt)
    assert not div.any()


def test_generator_flags_and_freezes_divergence():
    m, tau = 2, 1
    rff = Fe.draw_rff(m, 10, 4)
    weights = np.zeros((2, rff.P))
    weights[0, 1] = 2.0       # y_{t+1} = 2 y_t  -> explodes
    weights[1, 1] = 0.5       # contracts
    thetas = np.zeros((2, 4))
    ys, div, step = G.simulate(np.ones((2, 2)), thetas, weights, np.zeros((2, 40)), rff, m, tau, 10.0)
    assert div.tolist() == [True, False]
    assert step[0] == 3              # 2, 4, 8, 16 -> flagged at the 4th step (index 3)
    assert np.all(ys[0, 3:] == ys[0, 2])   # frozen at last valid value


def test_residual_bootstrap_innovations_resample_train_residuals():
    rng = np.random.default_rng(5)
    y = np.sin(np.arange(400) / 4) + 0.1 * rng.standard_normal(400)
    Z, tgt, _ = E.regression_pairs(y, 2, 2)
    rff = Fe.draw_rff(2, 20, 5)
    th = np.array([[0.0, 0.0, 0.0, -2.0]] * 3)
    w = np.stack([Mo.posterior_mean_weights(th[0], Z, tgt, rff)] * 3)
    inn = G.innovations("residual_bootstrap", th, w, Z, tgt, rff, 1000, rng)
    resid = tgt - Fe.features_np(Z, rff, 1.0) @ w[0]
    assert inn.shape == (3, 1000)
    assert np.all(np.isin(np.round(inn[0], 12), np.round(resid - resid.mean(), 12)))
    gauss = G.innovations("gaussian", th, w, Z, tgt, rff, 20000, rng)
    assert gauss.std() == pytest.approx(np.exp(-2.0), rel=0.03)
