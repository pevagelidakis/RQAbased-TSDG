import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from koopgen import embedding as E
from koopgen import features as Fe
from koopgen import model as Mo

jax.config.update("jax_enable_x64", True)


def _jargs(X, y, rff):
    return (jnp.asarray(X), jnp.asarray(X @ rff.W.T), jnp.asarray(y), jnp.asarray(rff.b))


def _small_problem(seed=0, N=60, F=20, m=2):
    rng = np.random.default_rng(seed)
    y_series = np.cumsum(rng.standard_normal(N + 10)) * 0.1 + np.sin(np.arange(N + 10) / 3)
    X, y, _ = E.regression_pairs(y_series, m, 2)
    X, y = X[:N], y[:N]
    rff = Fe.draw_rff(m, F, seed)
    return X, y, rff


@pytest.mark.parametrize("theta", [(0.0, 0.0, 0.0, -1.0), (0.7, -1.5, 2.0, -2.5), (-0.5, 1.0, -1.0, 0.3)])
def test_marginal_likelihood_matches_bruteforce(theta):
    X, y, rff = _small_problem()
    th = np.array(theta)
    ll = float(Mo.marginal_loglik(jnp.asarray(th), *_jargs(X, y, rff)))
    Phi = Fe.features_np(X, rff, np.exp(th[0]))
    lam = np.r_[np.full(rff.n_lin, np.exp(th[1])), np.full(rff.F, np.exp(th[2]))]
    cov = np.exp(2 * th[3]) * np.eye(len(y)) + Phi @ np.diag(1 / lam) @ Phi.T
    ref = multivariate_normal(mean=np.zeros(len(y)), cov=cov).logpdf(y)
    assert ll == pytest.approx(ref, abs=1e-8)


def test_features_jax_matches_numpy():
    X, _, rff = _small_problem()
    a = Fe.features_np(X, rff, 1.3)
    b = np.asarray(Fe.features_jax(jnp.asarray(X), jnp.asarray(X @ rff.W.T),
                                   jnp.asarray(rff.b), 1.3))
    assert np.allclose(a, b, atol=1e-13)
    assert a.shape == (X.shape[0], rff.P)


def test_rff_approximates_gaussian_kernel():
    rng = np.random.default_rng(1)
    rff = Fe.draw_rff(3, 20000, 2)
    x, z = rng.standard_normal(3), rng.standard_normal(3)
    s = 1.5
    k_hat = Fe.features_np(x, rff, s)[0, rff.n_lin:] @ Fe.features_np(z, rff, s)[0, rff.n_lin:]
    assert k_hat == pytest.approx(np.exp(-np.sum((x - z) ** 2) / (2 * s * s)), abs=0.02)


def test_posterior_weights_match_closed_form():
    X, y, rff = _small_problem(seed=3)
    th = np.array([0.2, -1.0, 0.5, -1.2])
    Phi = Fe.features_np(X, rff, np.exp(th[0]))
    lam = np.r_[np.full(rff.n_lin, np.exp(th[1])), np.full(rff.F, np.exp(th[2]))]
    beta = np.exp(-2 * th[3])
    S = np.linalg.inv(np.diag(lam) + beta * Phi.T @ Phi)
    mN = beta * S @ Phi.T @ y
    assert Mo.posterior_mean_weights(th, X, y, rff) == pytest.approx(mN, abs=1e-9)
    draws = Mo.sample_weights(np.repeat(th[None], 4000, 0), X, y, rff, seed=0)
    assert draws.mean(0) == pytest.approx(mN, abs=6 * np.sqrt(np.diag(S).max() / 4000))
    emp_cov = np.cov(draws.T)
    assert np.max(np.abs(emp_cov - S)) < 0.1 * np.max(np.abs(S))


def test_gradient_finite():
    X, y, rff = _small_problem()
    g = jax.grad(Mo.marginal_loglik)(jnp.array([0.1, 0.0, 0.0, -1.0]), *_jargs(X, y, rff))
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.parametrize("theta", [(0.1, 0.0, 0.0, -1.0), (0.7, -1.5, 2.0, -2.5)])
def test_numpy_likelihood_and_callback_match_jax(theta):
    X, y, rff = _small_problem(seed=6, N=90, F=25, m=3)
    args = _jargs(X, y, rff)
    th = jnp.asarray(theta)
    v_ref, g_ref = jax.value_and_grad(Mo.marginal_loglik)(th, *args)
    lik = Mo.NumpyLikelihood(X, y, rff)
    v, g = lik(np.asarray(theta))
    assert v == pytest.approx(float(v_ref), abs=1e-9)
    assert g == pytest.approx(np.asarray(g_ref), rel=1e-7, abs=1e-8)
    f = Mo.make_callback_loglik(lik)
    v2, g2 = jax.jit(jax.value_and_grad(f))(th)
    assert float(v2) == pytest.approx(float(v_ref), abs=1e-9)
    assert np.asarray(g2) == pytest.approx(np.asarray(g_ref), rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("theta", [(0.1, 0.0, 0.0, -1.0), (0.3, -1.0, 1.0, -6.0)])
def test_noise_floor_value_and_gradient(theta):
    X, y, rff = _small_problem(seed=7, N=90, F=25, m=3)
    th = jnp.asarray(theta)
    v_ref, g_ref = jax.value_and_grad(Mo.marginal_loglik)(th, *_jargs(X, y, rff), 0.05)
    v, g = Mo.NumpyLikelihood(X, y, rff, floor=0.05)(np.asarray(theta))
    assert v == pytest.approx(float(v_ref), abs=1e-9)
    assert g == pytest.approx(np.asarray(g_ref), rel=1e-7, abs=1e-8)
    # floor = 0 recovers the brute-force Gaussian with sigma, floor > 0 the one with sigma_eff
    se = float(Mo.sigma_eff(np.float64(theta[3]), 0.05))
    Phi = Fe.features_np(X, rff, np.exp(theta[0]))
    lam = np.r_[np.full(rff.n_lin, np.exp(theta[1])), np.full(rff.F, np.exp(theta[2]))]
    cov = se ** 2 * np.eye(len(y)) + Phi @ np.diag(1 / lam) @ Phi.T
    assert v == pytest.approx(multivariate_normal(np.zeros(len(y)), cov).logpdf(y), abs=1e-8)
