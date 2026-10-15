"""Data loading, validation, temporal split, standardisation, stationarity and benchmarks."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class DataValidationError(RuntimeError):
    """Raised for input problems that cannot be handled defensibly (no silent imputation)."""


@dataclass
class Series:
    name: str
    y_raw: np.ndarray            # validated raw values (after documented trimming only)
    dt: float                    # sampling interval in physical units (1.0 if unknown)
    time_unit: str
    findings: list = field(default_factory=list)
    truth: dict = field(default_factory=dict)   # benchmark ground truth (invariants)


@dataclass
class Split:
    y: np.ndarray                # standardised full series (train statistics)
    n_train: int
    mean: float
    sd: float

    @property
    def train(self):
        return self.y[: self.n_train]

    @property
    def test(self):
        return self.y[self.n_train:]


# ----------------------------------------------------------------------------- loading
def load_series(path: str | Path, column: str, time_col: str | None, min_length: int,
                dt_rel_tol: float = 0.01) -> Series:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"data file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise DataValidationError(f"unsupported file type '{path.suffix}' (use .csv or .parquet)")
    if column not in df.columns:
        raise DataValidationError(f"column '{column}' not in file; available: {list(df.columns)}")
    findings = [f"Loaded {len(df)} rows from {path.name}."]

    vals = pd.to_numeric(df[column], errors="coerce")
    bad = vals.isna() & df[column].notna()
    if bad.any():
        raise DataValidationError(
            f"{int(bad.sum())} non-numeric entries in '{column}' (first at row {int(np.argmax(bad))}).")

    dt, unit = 1.0, "sample"
    if time_col is not None:
        if time_col not in df.columns:
            raise DataValidationError(f"time column '{time_col}' not in file")
        t = df[time_col]
        if t.isna().any():
            raise DataValidationError(f"{int(t.isna().sum())} missing timestamps in '{time_col}'")
        if pd.api.types.is_numeric_dtype(t):
            tnum = t.to_numpy(dtype=np.float64)
            unit = f"{time_col} unit"
        else:
            try:
                tt = pd.to_datetime(t)
            except Exception as exc:  # noqa: BLE001
                raise DataValidationError(f"cannot parse '{time_col}' as datetimes: {exc}") from exc
            tnum = (tt - tt.iloc[0]).dt.total_seconds().to_numpy()
            unit = "s"
        dup = pd.Series(tnum).duplicated(keep=False)
        if dup.any():
            raise DataValidationError(
                f"{int(dup.sum())} rows share duplicate timestamps; cannot decide which value is "
                "correct, refusing to aggregate silently.")
        order = np.argsort(tnum, kind="stable")
        if np.any(order != np.arange(len(order))):
            findings.append("Timestamps were not sorted; rows were sorted by time (no values changed).")
            tnum, vals = tnum[order], vals.iloc[order].reset_index(drop=True)
        d = np.diff(tnum)
        med = float(np.median(d))
        irregular = np.abs(d - med) > dt_rel_tol * med
        if irregular.any():
            k = int(irregular.sum())
            raise DataValidationError(
                f"irregular sampling: {k} of {len(d)} intervals deviate by more than "
                f"{dt_rel_tol:.0%} from the median interval {med:g} {unit} (largest gap "
                f"{d.max():g}). A delay-embedding model needs uniform sampling; resample "
                "explicitly (and justify it) before running.")
        dt = med
        findings.append(f"Sampling is regular: dt = {dt:g} {unit} (tolerance {dt_rel_tol:.0%}).")
    else:
        findings.append("No time column given; uniform sampling assumed (dt = 1 sample).")

    y = vals.to_numpy(dtype=np.float64)
    isn = np.isnan(y)
    if isn.any():
        nz = np.nonzero(~isn)[0]
        if nz.size == 0:
            raise DataValidationError("column contains only missing values")
        lead, trail = int(nz[0]), int(len(y) - 1 - nz[-1])
        interior = int(isn[nz[0]: nz[-1] + 1].sum())
        if interior:
            raise DataValidationError(
                f"{interior} interior missing values (first at index "
                f"{int(nz[0] + np.argmax(isn[nz[0]:]))}); refusing to impute silently.")
        y = y[nz[0]: nz[-1] + 1]
        findings.append(f"Trimmed {lead} leading and {trail} trailing missing values (no imputation).")
    else:
        findings.append("No missing values.")
    if not np.all(np.isfinite(y)):
        raise DataValidationError("series contains infinite values")
    if np.std(y) == 0:
        raise DataValidationError("series is constant")
    if y.size < min_length:
        raise DataValidationError(f"series has {y.size} points; at least {min_length} are required")
    return Series(name=f"{path.stem}:{column}", y_raw=y, dt=dt, time_unit=unit, findings=findings)


def split_standardise(y_raw: np.ndarray, train_frac: float) -> Split:
    n_train = int(np.floor(train_frac * y_raw.size))
    tr = y_raw[:n_train]
    mu, sd = float(tr.mean()), float(tr.std(ddof=1))
    return Split(y=(y_raw - mu) / sd, n_train=n_train, mean=mu, sd=sd)


def stationarity(y_train: np.ndarray, alpha: float = 0.05) -> dict:
    """ADF (H0: unit root) and KPSS (H0: level-stationary) on the train series."""
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_stat, adf_p, *_ = adfuller(y_train, autolag="AIC", result_object=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kpss_stat, kpss_p, *_ = kpss(y_train, regression="c", nlags="auto")
        kpss_note = "; ".join(str(x.message) for x in w)
    nonstat = bool(adf_p > alpha or kpss_p < alpha)
    return {"adf_stat": float(adf_stat), "adf_p": float(adf_p), "kpss_stat": float(kpss_stat),
            "kpss_p": float(kpss_p), "kpss_note": kpss_note, "alpha": alpha,
            "nonstationary": nonstat}


# ----------------------------------------------------------------------------- benchmarks
def _lorenz_rhs(t, u, s, r, b):
    x, y, z = u
    return [s * (y - x), x * (r - z) - y, x * y - b * z]


def lorenz_x(n: int, dt: float, transient: float, rng: np.random.Generator,
             sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    u0 = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    t_end = transient + n * dt
    t_eval = transient + dt * np.arange(n)
    sol = solve_ivp(_lorenz_rhs, (0.0, t_end), u0, method="RK45", t_eval=t_eval,
                    args=(sigma, rho, beta), rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def henon_x(n: int, transient: int, rng: np.random.Generator, a: float = 1.4, b: float = 0.3,
            obs_noise: float = 0.01) -> np.ndarray:
    x, y = 0.1 * rng.standard_normal(), 0.1 * rng.standard_normal()
    out = np.empty(n)
    for i in range(n + transient):
        x, y = 1.0 - a * x * x + y, b * x
        if i >= transient:
            out[i - transient] = x
    return out + obs_noise * out.std() * rng.standard_normal(n)


def ar2(n: int, transient: int, rng: np.random.Generator, phi1: float = 1.2,
        phi2: float = -0.6) -> np.ndarray:
    e = rng.standard_normal(n + transient)
    y = np.zeros(n + transient)
    for t in range(2, n + transient):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + e[t]
    return y[transient:]


def benchmark_series(name: str, cfg: dict, rng: np.random.Generator) -> Series:
    b = cfg["benchmarks"][name]
    n = int(b["n"])
    if name == "lorenz":
        y = lorenz_x(n, float(b["dt"]), float(b["transient"]), rng)
        return Series(name="lorenz", y_raw=y, dt=float(b["dt"]), time_unit="Lorenz time unit",
                      findings=[f"Lorenz-63 x, RK45 (rtol 1e-10), sampled every {b['dt']}."],
                      truth={"lle": 0.906, "d2": 2.05, "lle_unit": "per time unit",
                             "nonlinear": True})
    if name == "henon":
        y = henon_x(n, int(b["transient"]), rng, obs_noise=float(b["obs_noise"]))
        return Series(name="henon", y_raw=y, dt=1.0, time_unit="iteration",
                      findings=[f"Henon map x (a=1.4, b=0.3) + {b['obs_noise']:.0%} observational noise."],
                      truth={"lle": 0.419, "d2": 1.21, "lle_unit": "per iteration",
                             "nonlinear": True})
    if name == "ar2":
        y = ar2(n, int(b["transient"]), rng, float(b["phi1"]), float(b["phi2"]))
        return Series(name="ar2", y_raw=y, dt=1.0, time_unit="sample",
                      findings=[f"Gaussian AR(2), phi=({b['phi1']}, {b['phi2']})."],
                      truth={"nonlinear": False})
    raise ValueError(f"unknown benchmark {name}")
