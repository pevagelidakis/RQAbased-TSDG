"""Random-Fourier-feature dictionary psi(x) = [1, x, sqrt(2/F) cos(W x / s + b)].

W and b are drawn once from a fixed seed and held constant through MCMC (common random numbers),
so the likelihood is a smooth function of the length-scale s.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RFF:
    W: np.ndarray   # (F, m)
    b: np.ndarray   # (F,)

    @property
    def F(self) -> int:
        return self.W.shape[0]

    @property
    def m(self) -> int:
        return self.W.shape[1]

    @property
    def n_lin(self) -> int:
        return 1 + self.m

    @property
    def P(self) -> int:
        return self.n_lin + self.F


def draw_rff(m: int, F: int, seed: int) -> RFF:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((F, m))
    b = rng.uniform(0.0, 2.0 * np.pi, F)
    return RFF(W=W, b=b)


def features_np(X: np.ndarray, rff: RFF, s: float) -> np.ndarray:
    X = np.atleast_2d(X)
    z = X @ rff.W.T / s + rff.b
    return np.concatenate([np.ones((X.shape[0], 1)), X, np.sqrt(2.0 / rff.F) * np.cos(z)], axis=1)


def features_jax(X, XWt, b, s):
    """JAX version given the precomputed product XWt = X @ W^T."""
    F = XWt.shape[-1]
    ones = jnp.ones(X.shape[:-1] + (1,), dtype=X.dtype)
    return jnp.concatenate([ones, X, jnp.sqrt(2.0 / F) * jnp.cos(XWt / s + b)], axis=-1)
