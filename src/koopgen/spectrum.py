"""Regularised EDMD Koopman matrix per posterior draw, eigen-decomposition and ResDMD residuals.

K = (Psi_X^T Psi_X + sigma^2 Lambda)^-1 Psi_X^T Psi_Y, consistent with the readout model: the
column of K belonging to the first state coordinate equals the posterior mean weights m_N.
Koopman eigenfunctions are phi(x) = psi(x) g with K g = mu g.

ResDMD (Colbrook & Townsend 2023):
  res(mu, g)^2 = g^H (L - mu A^H - conj(mu) A + |mu|^2 G) g / (g^H G g)
              = ||Psi_Y g - mu Psi_X g||^2 / ||Psi_X g||^2,
with G = Psi_X^H Psi_X, A = Psi_X^H Psi_Y, L = Psi_Y^H Psi_Y. The second (algebraically
identical) form is evaluated in production because it avoids the squared condition number of the
Gram matrices; a test checks the two agree on well-conditioned problems.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .features import RFF, features_np


@dataclass
class DrawSpectrum:
    mu: np.ndarray          # (P,) complex eigenvalues
    res: np.ndarray         # (P,) ResDMD residuals
    readout: np.ndarray     # (P,) column of K for the first state coordinate


def koopman_matrix(theta: np.ndarray, rff: RFF, X: np.ndarray, Xn: np.ndarray, floor: float = 0.0):
    s = np.exp(theta[0])
    PX, PY = features_np(X, rff, s), features_np(Xn, rff, s)
    lam = np.r_[np.full(rff.n_lin, np.exp(theta[1])), np.full(rff.F, np.exp(theta[2]))]
    sigma2 = np.exp(2 * theta[3]) + floor ** 2
    G = PX.T @ PX
    M = G + sigma2 * np.diag(lam)
    K = sla.cho_solve(sla.cho_factor(M, lower=True, check_finite=False), PX.T @ PY,
                      check_finite=False)
    return K, PX, PY


def resdmd_gram(mu, V, PX, PY):
    """Spec form: g^H (L - mu A^H - conj(mu) A + |mu|^2 G) g / g^H G g (columns of V)."""
    G, A, L = PX.conj().T @ PX, PX.conj().T @ PY, PY.conj().T @ PY
    q = lambda Mx: np.sum(V.conj() * (Mx @ V), axis=0)  # noqa: E731
    num = q(L) - mu * q(A.conj().T) - mu.conj() * q(A) + np.abs(mu) ** 2 * q(G)
    return np.sqrt(np.maximum(num.real, 0.0) / q(G).real)


def resdmd_direct(mu, V, PX, PY):
    XV = PX @ V
    R = PY @ V - XV * mu[None, :]
    return np.linalg.norm(R, axis=0) / np.linalg.norm(XV, axis=0)


def draw_spectrum(theta, rff, X, Xn, floor: float = 0.0) -> DrawSpectrum:
    K, PX, PY = koopman_matrix(theta, rff, X, Xn, floor)
    mu, V = np.linalg.eig(K)
    res = resdmd_direct(mu, V, PX, PY)
    return DrawSpectrum(mu=mu, res=res, readout=K[:, 1].copy())


_W = {}


def _init(rff, X, Xn, floor):
    _W.update(rff=rff, X=X, Xn=Xn, floor=floor)


def _job(th):
    d = draw_spectrum(th, _W["rff"], _W["X"], _W["Xn"], _W["floor"])
    return d.mu, d.res


def posterior_spectra(thetas: np.ndarray, rff: RFF, X, Xn, threshold: float, delta: float,
                      log=print, n_jobs: int = 4, floor: float = 0.0):
    """Eigenvalues and ResDMD residuals for each draw, computed in n_jobs processes that split the
    CPU threads between them (each draw is an independent P x P eigenproblem)."""
    from concurrent.futures import ProcessPoolExecutor
    import os

    threads = str(max(1, (os.cpu_count() or 4) // n_jobs))
    keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: threads for k in keys})   # inherited by the spawned workers only
    try:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=(rff, X, Xn, floor)) as ex:
            out = list(ex.map(_job, list(thetas), chunksize=max(1, len(thetas) // (4 * n_jobs))))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    log(f"  spectrum: {len(thetas)} draws")
    mu, res = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    spurious = res > threshold
    unstable = (np.abs(mu) > 1 + delta) & ~spurious
    summary = {
        "n_draws": int(mu.shape[0]),
        "P": int(mu.shape[1]),
        "spurious_fraction": float(spurious.mean()),
        "unstable_draw_fraction": float(unstable.any(axis=1).mean()),
        "unstable_spurious_draw_fraction": float(((np.abs(mu) > 1 + delta) & spurious).any(axis=1).mean()),
        "max_nonspurious_modulus_median": float(np.median(np.where(~spurious, np.abs(mu), 0).max(axis=1))),
        "threshold": threshold, "delta": delta,
    }
    return mu, res, summary


def leading_frequencies(mu: np.ndarray, res: np.ndarray, threshold: float, min_modulus: float):
    """Frequencies (cycles/sample) of non-spurious eigenvalues with |mu| >= min_modulus and
    positive imaginary part, pooled over draws."""
    ok = (res <= threshold) & (np.abs(mu) >= min_modulus) & (mu.imag > 1e-12)
    return np.angle(mu[ok]) / (2 * np.pi)
