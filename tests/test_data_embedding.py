import numpy as np
import pandas as pd
import pytest

from koopgen import data as D
from koopgen import embedding as E

CFG_E = {"max_lag": 60, "ami_bins": 16, "ami_floor_factor": 3.0, "m_min": 2, "m_max": 8,
         "cao_threshold": 0.9, "cao_saturation_tol": 0.05, "cao_saturation_min": 0.8,
         "cao_max_points": 1500, "cao_e2_tol": 0.1}


def _csv(tmp_path, df):
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    return p


def _good(n=1200):
    t = np.arange(n) * 0.5
    return pd.DataFrame({"t": t, "y": np.sin(t) + 0.1 * np.random.default_rng(0).standard_normal(n)})


def test_load_valid(tmp_path):
    s = D.load_series(_csv(tmp_path, _good()), "y", "t", min_length=1000)
    assert s.dt == pytest.approx(0.5) and s.y_raw.size == 1200


def test_interior_nan_fails(tmp_path):
    df = _good()
    df.loc[500, "y"] = np.nan
    with pytest.raises(D.DataValidationError, match="interior missing"):
        D.load_series(_csv(tmp_path, df), "y", "t", min_length=1000)


def test_edge_nan_trimmed_and_reported(tmp_path):
    df = _good(1300)
    df.loc[:4, "y"] = np.nan
    s = D.load_series(_csv(tmp_path, df), "y", "t", min_length=1000)
    assert s.y_raw.size == 1295 and any("Trimmed 5 leading" in f for f in s.findings)


def test_duplicate_timestamps_fail(tmp_path):
    df = _good()
    df.loc[10, "t"] = df.loc[9, "t"]
    with pytest.raises(D.DataValidationError, match="duplicate"):
        D.load_series(_csv(tmp_path, df), "y", "t", min_length=1000)


def test_irregular_sampling_fails(tmp_path):
    df = _good().drop(index=[100, 101]).reset_index(drop=True)
    with pytest.raises(D.DataValidationError, match="irregular sampling"):
        D.load_series(_csv(tmp_path, df), "y", "t", min_length=1000)


def test_non_numeric_and_missing_column_fail(tmp_path):
    df = _good()
    df["y"] = df["y"].astype(object)
    df.loc[3, "y"] = "abc"
    with pytest.raises(D.DataValidationError, match="non-numeric"):
        D.load_series(_csv(tmp_path, df), "y", "t", min_length=1000)
    with pytest.raises(D.DataValidationError, match="not in file"):
        D.load_series(_csv(tmp_path, _good()), "z", None, min_length=1000)


def test_split_uses_train_statistics_only():
    y = np.r_[np.zeros(70) + 1.0, np.zeros(30) + 100.0] + np.random.default_rng(0).standard_normal(100)
    sp = D.split_standardise(y, 0.7)
    assert sp.n_train == 70
    assert sp.mean == pytest.approx(y[:70].mean()) and sp.sd == pytest.approx(y[:70].std(ddof=1))
    assert abs(sp.train.mean()) < 1e-12


def test_stationarity_flags_random_walk():
    rw = np.cumsum(np.random.default_rng(1).standard_normal(3000))
    assert D.stationarity(rw)["nonstationary"]
    assert not D.stationarity(D.ar2(3000, 200, np.random.default_rng(2)))["nonstationary"]


def test_embedding_henon_and_lorenz():
    h = D.henon_x(6000, 500, np.random.default_rng(3), obs_noise=0.0)
    e = E.choose_embedding(h, CFG_E)
    assert e.tau == 1 and e.m == 2 and "fallback" in e.tau_method
    lz = D.lorenz_x(6000, 0.05, 50, np.random.default_rng(4))
    e = E.choose_embedding(lz, CFG_E)
    assert 3 <= e.tau <= 5 and 3 <= e.m <= 5 and e.tau_method == "first AMI minimum"


def test_stochastic_series_gets_tau_one():
    y = D.ar2(7000, 500, np.random.default_rng(5))
    e = E.choose_embedding(y, CFG_E)
    assert e.tau == 1 and "stochastic" in e.tau_method
    assert any("E2" in n for n in e.notes)
