# Pre-registration

Frozen on 2026-10-15, before any non-smoke pipeline run and before any scorecard had been looked at.

* SHA-256 of the canonical YAML dump of `config.yaml → preregistered` (decision rule, null construction,
  statistic list, control requirements, pre-test settings, MCMC gates, benchmark tolerances):

  `b20120ba46cbb3bc8b8aa5eba58c04f874c67ceb8062bc57410e6f5f8bd8b09d`

Every run recomputes this hash and reports whether it matches (`run_meta.json`, REPORT.md header).
Thresholds in that section are not to be changed after results are seen. Changes to method settings
outside that section are logged in `CHANGELOG_METHOD.md` with the reason.

Estimator settings (D2 correlation-sum range 1e-3..5e-2, Rosenstein rise window 0.2..0.7, Theiler window =
ceil(mean period), AMI bias-floor rule, Cao saturation rule) were fixed from unit-test calibration on
independently simulated Hénon/Lorenz series (milestone 2) and are part of the method settings.
The null construction (disjoint halves) and W = 500 were chosen from a simulation study on synthetic
AR(1) data (PLAN.md section 3), not from any benchmark output.
