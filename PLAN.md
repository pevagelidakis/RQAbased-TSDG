# PLAN — Bayesian kernel-Koopman generator (`koopgen`)

## 0. Status of inputs

* The repository contains no user data file. Milestones 1–7 (code, tests, benchmarks) do not
  need it. Before milestone 8 the user is asked for `--data/--column/--time-col`.
* Environment: Python 3.11, virtualenv at `C:\Users\panos\.venvs\koopgen` (kept outside
  OneDrive). Packages: numpy, scipy, pandas, jax[cpu] (x64), numpyro, arviz (+h5netcdf for
  NetCDF), numba, statsmodels, matplotlib, pyyaml, pytest. `pyarrow` is needed by pandas for
  parquet I/O; it was already installed and is the only extra.

## 1. Modules (`src/koopgen/`)

| module | content |
|---|---|
| `data.py` | load csv/parquet, validation (NaN, duplicates, irregular Δt, constant, length), temporal 70/30 split, train-only standardisation, ADF+KPSS, benchmark generators (Lorenz-63, Hénon+1% obs. noise, AR(2)) |
| `embedding.py` | AMI (histogram), first-minimum τ with ACF-zero fallback, Cao E1/E2, FNN, `delay_embed` |
| `features.py` | RFF dictionary ψ(x) = [1, x, √(2/F) cos(Wx/s+b)] in JAX and NumPy; W, b from a fixed seed |
| `model.py` | collapsed marginal likelihood (Cholesky), posterior of w, NumPyro model, priors |
| `mcmc.py` | NUTS runner, ArviZ conversion, convergence gates, remediation loop (raise target_accept) |
| `spectrum.py` | regularised EDMD matrix per draw, eigen-decomposition, ResDMD residuals, instability fraction, Hankel-DMD dominant frequency (operator-level statistic) |
| `generate.py` | posterior-predictive generator with re-lifting (`jax.lax.scan`, vectorised over series), divergence flags, Gaussian / residual-bootstrap innovations, ensemble forecasts |
| `metrics.py` | RQA at fixed RR (numba), D₂ (Grassberger–Procaccia), Rosenstein λ₁, sample/permutation entropy, time-reversal asymmetry, 3rd-order autocovariance, ACF/PACF/Welch, CRPS |
| `surrogates.py` | IAAFT, shuffle, AR(p) by AIC (fit + simulate), end-point-matched segment selection |
| `validate.py` | statistic battery on windows, block-bootstrap null, W₁ decision rule, PPP-values, nonlinearity pre-test, controls, verdict, benchmark invariants |
| `plots.py` | the 12 figures (PNG 300 dpi + PDF) |
| `report.py` | `REPORT.md` rendered only from saved artefacts |
| `cli.py` | `koopgen run ...` orchestration, seeds, timing, `run_meta.json` |

## 2. Equations as implemented

**Embedding.** x_t = (y_t, y_{t−τ}, …, y_{t−(m−1)τ}) ∈ ℝᵐ, t ≥ (m−1)τ. Regression pairs
(x_t, y_{t+1}). τ = first local minimum of AMI(lag) (histogram estimator, bins fixed in config);
if none within `max_lag`, τ = first zero crossing of the ACF (reported as a fallback).
m = smallest d with Cao E1(d) ≥ `cao_threshold` (FNN curve shown alongside). Train data only.

**Features.** W ∈ ℝ^{F×m}, Wᵢⱼ ~ N(0,1); b ~ U(0,2π), drawn once from `seed.features`.
ψ(x) = [1, x, √(2/F)·cos(W x / s + b)] ∈ ℝᴾ, P = 1+m+F. In expectation,
ψ_rffᵀψ_rff → exp(−‖x−x'‖²/(2s²)), the Gaussian kernel ("soft RP").

**Stochastic-series rule (added after the AR(2) smoke benchmark; see CHANGELOG_METHOD.md).** If Cao's
E2(d) is within 0.10 of 1 for every d (no detectable determinism), τ = 1 and m is re-selected at
τ = 1. AMI delay selection serves attractor unfolding; a stochastic process's Markov state is a
block of consecutive lags.

**Model.** y_{t+1} = ψ(x_t)ᵀ w + ε, ε ~ N(0, σ_eff²) with σ_eff² = σ² + σ_min² (nugget, σ_min = 1e-3
train-SD units; added after the Lorenz benchmark, see CHANGELOG_METHOD.md; β below is σ_eff⁻²); w ~ N(0, Λ⁻¹),
Λ = diag(α_lin·1_{1+m}, α_rff·1_F).
θ = (log s, log α_lin, log α_rff, log σ), with priors
log s ~ N(log median‖x_i−x_j‖_train, 1), log α_lin ~ N(0, 3²), log α_rff ~ N(0, 3²),
log σ ~ N(log 0.1, 3²) (data standardised with train statistics).

**Collapsed marginal likelihood**, with Φ ∈ ℝ^{N×P} and β = σ⁻²:
  A = Λ + β ΦᵀΦ = L Lᵀ (Cholesky),  m_N = β A⁻¹ Φᵀy (two triangular solves),
  log p(y|θ) = ½Σ log Λ_jj − (N/2) log σ² − ½(β‖y−Φm_N‖² + m_NᵀΛm_N) − Σ log L_jj − (N/2) log 2π.
Tested against the brute-force log N(y; 0, σ²I + ΦΛ⁻¹Φᵀ) to 1e-8.
w | θ, y ~ N(m_N, A⁻¹) is sampled as m_N + L⁻ᵀ z.

**EDMD per draw.** K = (Ψ_XᵀΨ_X + σ²Λ)⁻¹ Ψ_XᵀΨ_Y (P×P), with Ψ_X = ψ(x_t), Ψ_Y = ψ(x_{t+1}). The readout row
of this K equals m_N (consistency check in a test). Eigenpairs Kg = μg.
ResDMD: res² = gᴴ(L − μAᴴ − μ̄A + |μ|²G)g / gᴴGg, with G = Ψ_XᴴΨ_X, A = Ψ_XᴴΨ_Y, L = Ψ_YᴴΨ_Y
(all scaled by 1/N). An eigenpair is spurious if res > `resdmd_threshold`. Instability:
any non-spurious |μ| > 1+δ. Frequency is arg μ / (2π) in cycles per sample.

**Generator (re-lifting).** The buffer holds the last (m−1)τ+1 values.
y_{t+1} = ψ(x_t)ᵀw⁽ⁱ⁾ + σ⁽ⁱ⁾ε_t, the buffer is shifted, and ψ is recomputed from the new
buffer. There is never any linear propagation in feature space. Each series i has its own
posterior draw (θ⁽ⁱ⁾, w⁽ⁱ⁾) and an initial buffer from a random train index. Burn-in is 500
steps, and the retained length is N_test. A series is diverged if |y| > 10 (train SD units);
it is then frozen, flagged and counted.

## 3. Validation design (interpretations are made explicit here)

**Windows.** All statistics are computed on windows of length W (`window_length` = 500,
capped at ⌊N_test/6⌋, at least 200),
standardised with **train** mean/SD, and with the same metric embedding (m, τ) and RQA settings
for every generator.
* Real: windows of the test set at stride `window_stride`, giving K_r overlapping windows.
* Generators: M series of length N_test each. One window per series at a random offset gives M
  independent values per statistic. IAAFT and shuffle are surrogates of the test set. AR(p) is
  fitted on train and simulated.

**Block bootstrap and null (revised in milestone 6, before any pipeline run).** The real distribution R is the set of window statistics along the
test set. A moving-block bootstrap of the test set is done at window granularity: blocks of
`block_windows` consecutive windows are resampled to length K_r. This is equivalent to
resampling contiguous test-set stretches, and it avoids the join artefacts that concatenating
raw blocks would put into recurrence and Lyapunov statistics.
Null discrepancy = W₁(R*_A, R*_B), where R*_A and R*_B are independent block-bootstrap
resamples drawn from the **disjoint first and second halves** of the test windows. It is
repeated `n_null` ≥ 1000 times. A statistic passes if W₁(R, G) ≤ q₉₅(null).

*Why disjoint halves?* My first argument was that W₁ between two resamples of all the windows
is conservative. A unit test (`test_decision_rule_size_and_power`) refuted it, and so did a
dedicated simulation: an AR(1) with N_test = 3000 and window statistics (variance, lag-1
autocorrelation). Rejection rates were measured under H0 (a perfect generator) and against a
φ 0.9 → 0.85 alternative:

| null | W=1000 size / power | W=500 size / power |
|---|---|---|
| two resamples of all windows (literal) | 0.53 / 1.00 | 0.13 / 1.00 |
| disjoint halves (adopted) | 0.33 / 0.97 | **0.03 / 0.90** |
| posterior-predictive (generator stretches) | 0.07 / 1.00 | 0.03 / 1.00 |

Resamples of the same few overlapping windows share blocks, so the null is far too narrow.
Disjoint halves make the two resamples genuinely independent stretches of real data. The
rule is calibrated once each half holds ≥ 3 window lengths, so W = min(500, ⌊N_test/6⌋).
The posterior-predictive null is reported as a secondary, non-gated check for the Koopman
generator.

**Vector statistics** (ACF lags 1..L, PACF, log-Welch PSD on a fixed grid, quantiles,
time-reversal asymmetry for k ∈ K, third-order autocovariance for k ∈ K) count as **one**
statistic each. The discrepancy is the mean over components of W₁ₖ / scaleₖ, where scaleₖ is the
SD of component k over the real windows. The same function is used for the observed and the
null discrepancy, so calibration is unaffected. This avoids 50+ separate lag tests, which would
make "all pass" impossible by chance alone.

**Statistics** (pre-registered list):
linear = {acf, pacf, psd, mean, variance, quantiles, ks_marginal};
nonlinear = {rqa_det, rqa_lam, rqa_lmean, rqa_lmax, rqa_entr, rqa_tt, rqa_rte, d2, lle,
sampen, permen, time_reversal, c3};
operator = {koopman_freq} (dominant Hankel-DMD frequency of the window, mode with the largest
amplitude excluding the zero frequency). This one is gated together with the linear group.
ks_marginal = KS distance between the window and the **train** marginal.

**Secondary.** PPP = P_G(T ≥ T_real), with T_real the statistic of the whole test set
(two-sided flag outside [0.025, 0.975]). Reported, not gated.

**Nonlinearity pre-test** (on the train set, so it is independent of the scorecard). An
end-point-matched train segment of length ≤ `pretest_length` is used with `n_surrogates`
IAAFT surrogates. Each nonlinear statistic gets a two-sided rank test, with a Holm correction
across statistics at α = 0.05. "Nonlinear dynamics detected" ⇔ at least one rejection.

**Controls.**
* Shuffle must fail all of {acf, pacf, psd}. Shuffle preserves the marginal, so it is expected
  to pass mean/variance/quantiles/ks.
* If the pre-test detected nonlinearity, IAAFT must fail ≥ 1 nonlinear statistic.
* If a control misbehaves, the verdict is INCONCLUSIVE.

**Verdict.**
* PRESERVED: all gated statistics pass, controls OK, MCMC gates OK, divergence ≤ 5%.
* PARTIALLY PRESERVED: controls OK and gates OK, but some statistics fail. All failing
  statistics are listed.
* NOT PRESERVED: the divergence rate is above 5%, or the MCMC gates still fail after
  remediation, or every gated group fails.
* INCONCLUSIVE: the controls lack power.
* If no nonlinearity is detected, only the linear claim is evaluated. The report states
  "no detectable nonlinear dynamics to preserve", and the nonlinear rows are shown but marked
  not applicable.

**Benchmarks** (pre-registered pass criteria, in `config.yaml → benchmarks`).
* Lorenz-63 and Hénon must (a) detect nonlinearity, (b) estimate λ₁ and D₂ on the real
  benchmark series within tolerance of theory, (c) give synthetic medians within the same
  tolerance, and (d) give verdict PRESERVED.
* AR(2) must report no detectable nonlinear dynamics and preserve the linear statistics.

## 4. Pre-registration mechanics

`config.yaml` holds every threshold. Estimator settings (D₂ fit range, Rosenstein fit range,
RQA Theiler window) are calibrated **only** in unit tests on long, independently generated
Hénon/Lorenz series (milestone 2). They are never calibrated on pipeline output. Before the
first benchmark pipeline run, `PREREGISTRATION.md` records the SHA-256 of `config.yaml`. Every
run recomputes the hash and puts it in `run_meta.json`, and the report flags any mismatch. Any
method change after the first benchmark run is logged in `CHANGELOG_METHOD.md` and summarised
in each report.

## 5. Risks

1. **MCMC cost.** Each gradient evaluation is O(N·P²) (Φ depends on s): ~22 evaluations/s with
   4 chain processes × 5 threads at N≈7000, P≈520. A fixed 4×(5000+25000) run would take
   ~10 h. Resolution (CHANGELOG_METHOD.md): whitened NUTS around the Laplace mode, plus a run
   length set by ESS (4× the gate) and capped at 25,000 draws per chain. Gates are unchanged.
2. **Free-run instability.** Iterated one-step maps can blow up or collapse onto a fixed point.
   The 5% divergence gate catches the first case. The scorecard (variance, ACF, D₂) catches the
   second.
3. **Estimator bias on short windows.** D₂/λ₁ from ≤ 2000 points are biased. The bias is the
   same across generators, since all use the same estimator and window, and the benchmark
   tolerances absorb the absolute bias.
4. **Hénon observational noise.** The fitted model is a dynamically noisy map, so small-scale
   invariants (D₂) may differ. This is a genuine finding and will be reported as such.
5. **Familywise error.** "All of ~21 statistics pass" has a size above 5% under a perfect
   generator. The window-level null is conservative, which partly offsets this. The issue is
   stated as a limitation, and thresholds are not changed.
6. **Non-stationarity** (user data): ADF/KPSS are reported. If either indicates
   non-stationarity, the report opens with a prominent warning.
