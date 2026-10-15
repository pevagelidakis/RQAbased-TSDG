# STATUS / handoff (paused 2026-10-15 ~15:40)

Read this first when resuming. Spec = the user's task prompt ("Bayesian Kernel-Koopman Generator…").
Design and equations: `PLAN.md`. Frozen thresholds: `PREREGISTRATION.md` (hash `b20120ba…`; every run
checks it). Method changes after freezing: `CHANGELOG_METHOD.md`.

## Environment
- venv: `C:\Users\panos\.venvs\koopgen` (outside OneDrive). Run `koopgen` via
  `/c/Users/panos/.venvs/koopgen/Scripts/koopgen`; tests via `.../Scripts/python -m pytest -q` (65 pass).
- Nothing is committed to git yet (the user has not asked for a commit).

## Milestones (spec execution plan)
| # | milestone | state |
|---|---|---|
| 1 | PLAN.md | done |
| 2 | metrics + surrogates + tests | done |
| 3 | embedding, features, collapsed likelihood (brute-force check 1e-8) | done |
| 4 | MCMC smoke + gate logic | done |
| 5 | spectrum + ResDMD + generator | done |
| 6 | validation protocol + scorecard | done (null rule redesigned; see PLAN §3) |
| 7 | **benchmarks end to end** | **IN PROGRESS: Lorenz run stopped mid-sampling (see below)** |
| 8 | full run on the user's data | blocked: **no data file provided yet** (need path, value column, time column) |
| 9 | report | code done (`report.py`); final reports pending runs |

## Where it stopped
`results/bench_lorenz/`: the non-smoke Lorenz benchmark (`--no-sensitivity`) was stopped during MCMC
sampling (4 chains, 500 of ~1800 draws each). **Chain 1 had 1 divergence**, so the zero-divergence gate
would have failed and triggered the remediation rerun. Embedding: τ=4, m=4. Laplace init converged
(mode [-0.099, -0.184, -4.564, -11.263]). Warmup was slow (≈35 min for 1000 iterations).
This directory is incomplete and should be deleted before rerunning.

Diagnosis: with the σ noise floor (σ_eff² = σ² + 1e-3²) the free log σ sits far below the floor
(σ ≈ 1.3e-5), where the likelihood is flat in log σ and only the prior constrains it, with a sharp
cliff near log(1e-3). That non-Gaussian geometry explains both the slow warmup and the divergence.

## NEXT STEP (proposed to the user, awaiting confirmation)
Sample the identified quantity instead: put θ₄ = log σ_eff (σ_eff > floor) in the sampler, or
equivalently reparameterise so the flat region disappears. Then:
1. Update `model.py` (NumpyLikelihood, reference `marginal_loglik`, `sigma_eff`, numpyro model,
   Laplace), `spectrum.py`, `generate.py`, `cli.py` (forecast). Keep θ's prior documented in
   PLAN/CHANGELOG. Unit test: value and gradient vs JAX autodiff and brute-force Gaussian.
2. Re-run `koopgen run --benchmark lorenz --no-sensitivity --run-id bench_lorenz`, then henon, then ar2.
   (Delete `results/bench_*` first.)
3. If all three pass: sensitivity. Ask the user whether to run it only on their data (saves ~5–7 h).
4. Get the user's data and run `koopgen run --data … --column … [--time-col …]`.

## Notes and evidence from this session
- Smoke runs (code paths only, NOT valid results): `results/smoke_henon` (20/21 statistics pass, D₂
  fails: dynamical vs observational noise), `results/smoke_ar2_prefix` (AR(2) failed linear statistics
  before the τ fix), `results/smoke_ar2` (made with the since-reverted buffer variant; all linear
  statistics passed). The current code (τ=1 stochastic rule + noise floor) has **not** yet been run end
  to end on AR(2) or Hénon.
- Measured cost: ~22 likelihood+gradient evaluations/s (4 chain processes × 5 threads, N≈7000, P≈517).
  In the Lorenz run, sampling cost ≈2.1 s per draw per chain, ESS ≈ 0.35–0.56 per draw.
- Kill helper (bash): `powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or
  Name='koopgen.exe'\" | Where-Object { \$_.CommandLine -match 'koopgen' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"`
  (do not include bash.exe in the filter: it kills the calling shell).
- README.md is UTF-16 encoded: edit it with the Edit tool, not Python scripts.
