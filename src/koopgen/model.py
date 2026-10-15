"""Bayesian readout model y_{t+1} = psi(x_t)^T w + eps with w collapsed analytically.

theta = (log s, log alpha_lin, log alpha_rff, log sigma). Prior w ~ N(0, Lambda^-1),
Lambda = diag(alpha_lin * 1_{1+m}, alpha_rff * 1_F). JAX reference functions take the states X and
the precomputed product XWt = X W^T.

    A      = Lambda + sigma^-2 Phi^T Phi           (= S_N^-1, Cholesky A = L L^T)
    m_N    = sigma^-2 A^-1 Phi^T y
    log p  = 1/2 sum log Lambda_jj - N/2 log sigma^2
             - 1/2 (sigma^-2 ||y - Phi m_N||^2 + m_N^T Lambda m_N) - 1/2 log|A| - N/2 log 2 pi
No explicit inverses: only Cholesky factorisations and triangular solves.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy.linalg as sla
from numba import njit, prange

from .features import RFF, features_jax

jax.config.update("jax_enable_x64", True)

THETA_NAMES = ("log_s", "log_alpha_lin", "log_alpha_rff", "log_sigma")


def sigma_eff(log_sigma, floor: float):
    """Noise SD used everywhere: sigma_eff^2 = sigma^2 + floor^2 (a fixed nugget in train-SD units).

    Noise-free simulations drive sigma towards 0 and A = Lambda + sigma^-2 Phi^T Phi past float64
    precision (Lorenz benchmark: sigma ~ 4e-5, cond(A) ~ 1e17). The floor acts as measurement
    resolution; it is negligible whenever the true noise exceeds it."""
    xp = jnp if isinstance(log_sigma, jnp.ndarray) and not isinstance(log_sigma, np.ndarray) else np
    return xp.sqrt(xp.exp(2.0 * log_sigma) + floor ** 2)


@dataclass
class Prior:
    log_s: tuple[float, float]
    log_alpha_lin: tuple[float, float]
    log_alpha_rff: tuple[float, float]
    log_sigma: tuple[float, float]

    def as_dict(self):
        return {k: list(getattr(self, k)) for k in THETA_NAMES}


def make_prior(X_train: np.ndarray, cfg_prior: dict, max_pairs: int, seed: int) -> Prior:
    """Weakly informative priors on standardised data. log s is centred at the log of the median
    pairwise distance between train states (train data only)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_train.shape[0], size=min(max_pairs, X_train.shape[0]), replace=False)
    Z = X_train[idx]
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    med = float(np.median(d[np.triu_indices_from(d, 1)]))
    return Prior(log_s=(float(np.log(med)), float(cfg_prior["log_s_sd"])),
                 log_alpha_lin=tuple(map(float, cfg_prior["log_alpha_lin"])),
                 log_alpha_rff=tuple(map(float, cfg_prior["log_alpha_rff"])),
                 log_sigma=tuple(map(float, cfg_prior["log_sigma"])))


def _precision_diag(log_alpha_lin, log_alpha_rff, n_lin, F):
    return jnp.concatenate([jnp.full((n_lin,), jnp.exp(log_alpha_lin)),
                            jnp.full((F,), jnp.exp(log_alpha_rff))])


def posterior_parts(theta, X, XWt, y, b, floor=0.0):
    """Return (L, m_N, Phi, lam) for the collapsed posterior at theta (JAX)."""
    log_s, log_al, log_ar, log_sig = theta[0], theta[1], theta[2], theta[3]
    Phi = features_jax(X, XWt, b, jnp.exp(log_s))
    lam = _precision_diag(log_al, log_ar, 1 + X.shape[1], XWt.shape[1])
    beta = 1.0 / (jnp.exp(2.0 * log_sig) + floor ** 2)
    A = jnp.diag(lam) + beta * (Phi.T @ Phi)
    L = jnp.linalg.cholesky(A)
    m_N = beta * jsl.cho_solve((L, True), Phi.T @ y)
    return L, m_N, Phi, lam


def marginal_loglik(theta, X, XWt, y, b, floor=0.0):
    """JAX reference (autodiff-able) version of the collapsed log marginal likelihood."""
    L, m_N, Phi, lam = posterior_parts(theta, X, XWt, y, b, floor)
    N = y.shape[0]
    beta = 1.0 / (jnp.exp(2.0 * theta[3]) + floor ** 2)
    resid = y - Phi @ m_N
    quad = beta * resid @ resid + m_N @ (lam * m_N)
    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return (0.5 * jnp.sum(jnp.log(lam)) + 0.5 * N * jnp.log(beta) - 0.5 * quad - 0.5 * logdet_A
            - 0.5 * N * jnp.log(2.0 * jnp.pi))


@njit(parallel=True, cache=True)
def _rff_cols(XWt, b, s, c, Phi, D, n_lin):
    """Phi[:, n_lin:] = c cos(XWt/s + b);  D = c sin(XWt/s + b) * XWt/s  (= d Phi_rff / d log s)."""
    N, F = XWt.shape
    for t in prange(N):
        for j in range(F):
            u = XWt[t, j] / s
            z = u + b[j]
            Phi[t, n_lin + j] = c * np.cos(z)
            D[t, j] = c * np.sin(z) * u


@njit(parallel=True, cache=True)
def _sum_prod(B, D):
    N, F = B.shape
    acc = np.zeros(N)
    for t in prange(N):
        a = 0.0
        for j in range(F):
            a += B[t, j] * D[t, j]
        acc[t] = a
    return acc.sum()


class NumpyLikelihood:
    """Value and analytic gradient of the collapsed log marginal likelihood (NumPy/BLAS + numba).

    Same value as the JAX reference `marginal_loglik`, with the analytic gradient in theta
    (envelope theorem): d ll / d Phi = beta [(y - Phi m_N) m_N^T - Phi A^-1]; d ll / d lambda_j =
    1/2 (1/lambda_j - m_j^2 - (A^-1)_jj); d ll / d log sigma = -N + beta (rss + tr(A^-1 G)).
    Used inside NUTS through `jax.pure_callback`: OpenBLAS SYRK/GEMM plus fused numba kernels are
    faster than reverse-mode autodiff through the Cholesky factorisation on XLA:CPU.
    Buffers are thread-local so concurrent callbacks are safe.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, rff: RFF, floor: float = 0.0):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.b = np.ascontiguousarray(rff.b, dtype=np.float64)
        self.XWt = np.ascontiguousarray(self.X @ rff.W.T)
        self.N, self.m = self.X.shape
        self.F = rff.F
        self.n_lin = 1 + self.m
        self.P = self.n_lin + self.F
        self.c = float(np.sqrt(2.0 / self.F))
        self.floor = float(floor)
        self._local = threading.local()

    def _buffers(self):
        loc = self._local
        if not hasattr(loc, "Phi"):
            loc.Phi = np.empty((self.N, self.P))
            loc.Phi[:, 0] = 1.0
            loc.Phi[:, 1:self.n_lin] = self.X
            loc.D = np.empty((self.N, self.F))
            loc.I = np.eye(self.P)
        return loc.Phi, loc.D, loc.I

    def __call__(self, theta) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        log_s, log_al, log_ar, log_sig = theta
        Phi, D, I = self._buffers()
        _rff_cols(self.XWt, self.b, np.exp(log_s), self.c, Phi, D, self.n_lin)
        lam = np.concatenate([np.full(self.n_lin, np.exp(log_al)), np.full(self.F, np.exp(log_ar))])
        s2 = np.exp(2.0 * log_sig)
        beta = 1.0 / (s2 + self.floor ** 2)
        r_sig = s2 * beta                     # d log beta / d log sigma = -2 r_sig
        G = Phi.T @ Phi
        A = beta * G
        A[np.diag_indices_from(A)] += lam
        try:
            L = sla.cholesky(A, lower=True, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf, np.zeros(4)
        r = Phi.T @ self.y
        m_N = beta * sla.cho_solve((L, True), r, check_finite=False)
        resid = self.y - Phi @ m_N
        rss = float(resid @ resid)
        ll = (0.5 * np.sum(np.log(lam)) + 0.5 * self.N * np.log(beta) - 0.5 * (beta * rss + m_N @ (lam * m_N))
              - np.sum(np.log(np.diag(L))) - 0.5 * self.N * np.log(2.0 * np.pi))
        Ainv = sla.cho_solve((L, True), I, check_finite=False)
        B = Phi @ Ainv[:, self.n_lin:]
        d_log_s = beta * (float(m_N[self.n_lin:] @ (D.T @ resid)) - _sum_prod(B, D))
        t = 0.5 - 0.5 * lam * m_N ** 2 - 0.5 * lam * np.diag(Ainv)
        grad = np.array([d_log_s, t[:self.n_lin].sum(), t[self.n_lin:].sum(),
                         r_sig * (-self.N + beta * rss + beta * float(np.sum(Ainv * G)))])
        if not (np.isfinite(ll) and np.all(np.isfinite(grad))):
            return -np.inf, np.zeros(4)
        return float(ll), grad


def laplace_approximation(lik: NumpyLikelihood, prior: Prior, h: float = 1e-2):
    """Approximate posterior mode and Laplace covariance of theta, used only to initialise NUTS
    (chain start points and the initial dense inverse mass matrix; warmup adaptation then
    proceeds as usual).

    Near-deterministic data make Phi^T Phi very ill-conditioned, so the Cholesky-based log
    likelihood carries rounding noise of ~1e-3 nats and its gradient ~1 unit. This is harmless
    for HMC (gradient scales are ~1e2), but it limits the optimiser to a gradient norm of a few,
    and it rules out small finite-difference steps. Hence: best of several L-BFGS runs,
    central differences with h = 1e-2, and Hessian eigenvalues floored at the smallest prior
    precision.
    """
    from scipy.optimize import minimize

    mu = np.array([getattr(prior, k)[0] for k in THETA_NAMES])
    sd = np.array([getattr(prior, k)[1] for k in THETA_NAMES])

    def neg(th):
        v, g = lik(th)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return 1e300, np.zeros(4)
        return -(v - 0.5 * np.sum(((th - mu) / sd) ** 2)), -(g - (th - mu) / sd ** 2)

    best = None
    starts = (mu, mu + np.array([0.0, 0.0, 0.0, -2.0]), mu + np.array([0.0, 0.0, -4.0, -4.0]),
              mu + np.array([0.0, -4.0, -8.0, -5.0]))
    for start in starts:
        r = minimize(neg, start, jac=True, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    th0 = best.x
    H = np.zeros((4, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        H[:, i] = (neg(th0 + e)[1] - neg(th0 - e)[1]) / (2 * h)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 1.0 / np.max(sd) ** 2)
    cov = (V / w) @ V.T
    return th0, cov, {"converged": bool(best.success), "nfev": int(best.nfev),
                      "grad_norm": float(np.linalg.norm(best.jac)),
                      "neg_log_post": float(best.fun)}


def make_callback_loglik(lik: NumpyLikelihood):
    """A JAX-traceable scalar function of theta whose value and gradient come from `lik`."""
    shapes = (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((4,), jnp.float64))

    def _host(th):
        v, g = lik(th)
        return np.asarray(v, dtype=np.float64), np.asarray(g, dtype=np.float64)

    def _both(theta):
        return jax.pure_callback(_host, shapes, theta, vmap_method="sequential")

    @jax.custom_vjp
    def f(theta):
        return _both(theta)[0]

    def f_fwd(theta):
        v, g = _both(theta)
        return v, g

    def f_bwd(g, ct):
        return (ct * g,)

    f.defvjp(f_fwd, f_bwd)
    return f


def numpyro_model_whitened(prior: Prior, loglik_fn, center, chol):
    """Posterior of theta (Normal priors x collapsed likelihood), sampled in whitened coordinates z with
    theta = center + chol @ z (an affine, constant-Jacobian reparameterisation; center/chol are
    the Laplace mode and Cholesky factor). In z the posterior is ~N(0, I), which matches NUTS'
    default initial mass matrix and its unit-scale mass-matrix regulariser. Without this, the
    posterior SD of log s (~1e-2 or smaller on long, near-deterministic series) is swamped by
    that regulariser and NUTS needs ~100 leapfrog steps per draw."""
    from numpyro.distributions import constraints

    z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (4,)))
    theta = center + chol @ z
    for i, k in enumerate(THETA_NAMES):
        numpyro.deterministic(k, theta[i])
    mu = jnp.array([getattr(prior, k)[0] for k in THETA_NAMES])
    sd = jnp.array([getattr(prior, k)[1] for k in THETA_NAMES])
    numpyro.factor("log_prior", jnp.sum(dist.Normal(mu, sd).log_prob(theta)))
    numpyro.factor("marginal_loglik", loglik_fn(theta))


_post_parts_jit = jax.jit(posterior_parts)


def sample_weights(theta_draws: np.ndarray, X: np.ndarray, y: np.ndarray, rff: RFF,
                   seed: int, floor: float = 0.0) -> np.ndarray:
    """Exact draws w ~ N(m_N, A^-1) for each theta (rows of theta_draws). w = m_N + L^-T z."""
    key = jax.random.PRNGKey(seed)
    Xj, Kj, yj, bj = map(jnp.asarray, (X, X @ rff.W.T, y, rff.b))
    out = np.empty((theta_draws.shape[0], rff.P))
    for i, th in enumerate(theta_draws):
        key, sub = jax.random.split(key)
        L, m_N, _, _ = _post_parts_jit(jnp.asarray(th), Xj, Kj, yj, bj, floor)
        z = jax.random.normal(sub, (rff.P,), dtype=jnp.float64)
        out[i] = np.asarray(m_N + jsl.solve_triangular(L.T, z, lower=False))
    return out


def posterior_mean_weights(theta: np.ndarray, X, y, rff: RFF, floor: float = 0.0) -> np.ndarray:
    _, m_N, _, _ = _post_parts_jit(jnp.asarray(theta), jnp.asarray(X),
                                   jnp.asarray(X @ rff.W.T), jnp.asarray(y),
                                   jnp.asarray(rff.b), floor)
    return np.asarray(m_N)
