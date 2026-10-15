"""REPORT.md - every number is read from the computed run artefacts (no hand-entered values)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _f(x, nd=3):
    if x is None:
        return "n/a"
    try:
        if not np.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}"


def _fig(name, captions, n):
    return f"![{name}](figures/{name}.png)\n\n*Figure {n}. {captions.get(name, '')}*\n"


def write_report(res: dict, out: Path, captions: dict):
    v = res["verdict"]
    g = res["gates"]
    emb = res["embedding"]
    st = res["stationarity"]
    sp = res["split"]
    spec = res["spectrum"]["summary"]
    rows = res["scorecard"]
    pr = res["cfg"]["preregistered"]
    L = []
    w = L.append
    w(f"# Dynamics-preservation report: `{res['name']}`\n")
    w(f"Run `{res['run_id']}` - config SHA-256 `{res['config_hash'][:16]}…`; pre-registered thresholds "
      f"SHA-256 `{res['prereg_hash'][:16]}…` (matches PREREGISTRATION.md: **{res['preregistered_match']}**)"
      + (" - **SMOKE MODE: reduced MCMC/generator settings; not a valid final result.**" if res["smoke"] else "") + "\n")
    if st["nonstationary"]:
        w("> **WARNING - non-stationarity.** ADF p = {:.3g}, KPSS p = {:.3g} on the train set. A single "
          "time-invariant Koopman operator is then a misspecified model; every conclusion below is "
          "conditional on approximate stationarity.\n".format(st["adf_p"], st["kpss_p"]))

    # 1 verdict
    w("## 1. Verdict\n")
    fails = v["failing_statistics"]
    sentence = f"**{v['verdict']}.** "
    if v["reasons"]:
        sentence += "Reasons: " + "; ".join(v["reasons"]) + ". "
    if fails:
        sentence += f"Failing gated statistics ({len(fails)} of {len(v['gated_statistics'])}): " + \
            ", ".join(f"`{s}`" for s in fails) + ". "
    else:
        sentence += f"All {len(v['gated_statistics'])} gated statistics pass the calibrated W1 rule. "
    sentence += (f"Nonlinearity pre-test: {'nonlinear dynamics detected' if v['nonlinear_detected'] else 'no detectable nonlinear dynamics - only the linear claim is evaluated'}. "
                 f"Controls: shuffle {'fails' if v['controls']['shuffle_ok'] else 'does NOT fail'} the required linear statistics; "
                 + (f"IAAFT fails {len(v['controls']['iaaft_failed_nonlinear'])} nonlinear statistic(s). " if v["nonlinear_detected"] else "IAAFT requirement not applicable. ")
                 + f"MCMC gates: {'passed' if g.passed else 'FAILED'}; divergence rate of synthetic series: "
                 f"{res['divergence_rate']:.1%} (limit {pr['decision']['max_divergence_rate']:.0%}).")
    w(sentence + "\n")
    if res.get("benchmark"):
        b = res["benchmark"]
        w(f"Benchmark criteria ({'ALL MET' if b['passed'] else 'NOT ALL MET'}): " +
          ", ".join(f"{k} = {'yes' if ok else 'NO'}" for k, ok in b["criteria"].items()) + "\n")

    # 2 data
    w("## 2. Data and embedding\n")
    for f in res["findings"]:
        w(f"- {f}")
    w(f"- Length {sp.y.size}; train {sp.n_train} / test {sp.test.size} (temporal 70/30). Standardised with "
      f"train mean {_f(sp.mean)} and SD {_f(sp.sd)} only.")
    w(f"- Stationarity (train): ADF stat {_f(st['adf_stat'])} (p = {_f(st['adf_p'])}); KPSS stat "
      f"{_f(st['kpss_stat'])} (p = {_f(st['kpss_p'])}){'; ' + st['kpss_note'] if st['kpss_note'] else ''}.")
    w(f"- τ = **{emb.tau}** ({emb.tau_method}); m = **{emb.m}** ({emb.m_method}).")
    for n in emb.notes:
        w(f"  - note: {n}")
    P = res["metric_params"]
    w(f"- Metric settings (fixed from train data for all generators): Theiler window {P.theiler_d2} samples "
      f"(⌈mean period⌉), Rosenstein k_max {P.k_max_lle}; windows of W = {res['window_length']} points, "
      f"stride {res['cfg']['metrics']['window_stride']}.\n")
    w(_fig("fig01_data_overview", captions, 1))

    # 3 MCMC
    w("## 3. MCMC\n")
    ms = res["mcmc_settings"]
    info = res["mcmc_info"]
    w(f"- Model: y_(t+1) = ψ(x_t)ᵀw + ε with F = {res['F']} random Fourier features (W, b fixed from a seed); "
      "w collapsed analytically; θ = (log s, log α_lin, log α_rff, log σ) sampled by NumPyro NUTS (JAX x64) "
      "with the exact marginal likelihood (Cholesky) and its analytic gradient.")
    w(f"- Settings ({res['mcmc_profile']}): {ms.num_chains} chains, {ms.num_warmup} warmup iterations each; draws "
      f"continue until every chain's min(bulk, tail) ESS over θ reaches {ms.target_ess}/{ms.num_chains} "
      f"(≥ {ms.min_samples}, ≤ {ms.max_samples} draws per chain); chains are truncated to a common length. "
      f"Draws per chain: {info['chain_lengths']} → retained {info['retained_per_chain']} each "
      f"({info['retained_per_chain'] * ms.num_chains} total; {info['n_gradient_evals']} gradient evaluations). "
      f"target_accept {ms.target_accept}, dense mass matrix, max tree depth {ms.max_tree_depth}. "
      f"NUTS runs in whitened coordinates θ = θ̂ + L z (Laplace mode/Cholesky; affine, exact) with chains "
      "initialised at z ~ 2·N(0, I).")
    w(f"- Wall time {info['wall_seconds']:.0f} s; mean leapfrog steps per draw {info['mean_leapfrog_steps']:.1f}; "
      f"step sizes {', '.join(_f(x) for x in info['step_sizes'])}.")
    w(f"- Gate results (thresholds: R-hat ≤ {pr['mcmc_gates']['rhat_max']}, bulk/tail ESS ≥ "
      f"{pr['mcmc_gates']['ess_bulk_min']}, divergences ≤ {pr['mcmc_gates']['divergences_max']}, E-BFMI > "
      f"{pr['mcmc_gates']['ebfmi_min']}): R-hat max {_f(g.rhat_max, 4)}, bulk ESS min {g.ess_bulk_min:.0f}, "
      f"tail ESS min {g.ess_tail_min:.0f}, divergences {g.divergences}, E-BFMI "
      f"{', '.join(_f(x) for x in g.ebfmi)} → **{'PASS' if g.passed else 'FAIL'}**.")
    if len(res["attempts"]) > 1 or not g.passed:
        w("- Remediation history:")
        for a in res["attempts"]:
            s = a["settings"]
            w(f"  - attempt {a['attempt']}: target_accept {s['target_accept']}, {s['num_chains']} chains, warmup "
              f"{s['num_warmup']}, target ESS {s['target_ess']}, draws per chain [{s['min_samples']}, {s['max_samples']}] → "
              + ("PASS" if a["gates"]["passed"] else "FAIL: " + "; ".join(a["gates"]["failures"])))
    w("\n| parameter | mean | sd | 94% HDI | R-hat | ESS bulk | ESS tail |\n|---|---|---|---|---|---|---|")
    for k, t in g.table.items():
        w(f"| {k} | {_f(t['mean'], 4)} | {_f(t['sd'], 4)} | [{_f(t['hdi_3%'], 4)}, {_f(t['hdi_97%'], 4)}] | "
          f"{_f(t['r_hat'], 4)} | {t['ess_bulk']:.0f} | {t['ess_tail']:.0f} |")
    w("")
    for i, n in enumerate(["fig02a_mcmc_trace", "fig02b_mcmc_rank", "fig02c_mcmc_autocorr", "fig02d_mcmc_pairs",
                           "fig02e_mcmc_energy", "fig02f_prior_posterior", "fig02g_rhat_ess_table"]):
        w(_fig(n, captions, f"2{'abcdefg'[i]}"))

    # 4 spectrum
    w("## 4. Koopman spectrum\n")
    w(f"- {spec['n_draws']} thinned posterior draws; EDMD matrix K = (Ψ_XᵀΨ_X + σ²Λ)⁻¹Ψ_XᵀΨ_Y of size "
      f"{spec['P']}×{spec['P']} per draw.")
    w(f"- Spurious eigenpairs (ResDMD residual > {spec['threshold']}): **{spec['spurious_fraction']:.1%}** of all eigenpairs.")
    w(f"- Draws with any non-spurious |μ| > 1 + {spec['delta']} (instability): **{spec['unstable_draw_fraction']:.1%}**; "
      f"draws with |μ| > 1 + δ only among spurious eigenpairs: {spec['unstable_spurious_draw_fraction']:.1%}. "
      f"Median over draws of the largest non-spurious |μ|: {_f(spec['max_nonspurious_modulus_median'], 4)}.")
    w("- In chaotic systems the Koopman operator has continuous spectrum; isolated EDMD eigenvalues are "
      "discretisation artefacts or approximate (pseudo)eigenvalues, which is why residuals are reported.\n")
    w(_fig("fig03_koopman_spectrum", captions, 3))

    # forecast
    fc = res["forecast"]
    w("### Forecast skill (test set)\n")
    hs = [1, 5, 10, len(fc["crps_koopman"])]
    w("| horizon [steps] | CRPS Koopman-MCMC | CRPS AR(p) | CRPS persistence |\n|---|---|---|---|")
    for h in hs:
        if h <= len(fc["crps_koopman"]):
            w(f"| {h} | {_f(fc['crps_koopman'][h - 1], 4)} | {_f(fc['crps_ar'][h - 1], 4)} | {_f(fc['crps_persistence'][h - 1], 4)} |")
    w("")
    w(_fig("fig04_forecast_skill", captions, 4))
    w(_fig("fig05_examples", captions, 5))
    w(_fig("fig06_phase_portraits", captions, 6))
    w(_fig("fig07_recurrence_plots", captions, 7))

    # 5 validation
    w("## 5. Validation\n")
    d = pr["decision"]
    w(f"Decision rule (pre-registered): a statistic passes if W1(real, generator) ≤ q{int(d['null_quantile'] * 100)} "
      f"of the null discrepancy, where the null is W1 between two independent moving-block-bootstrap "
      f"resamples drawn from the disjoint halves of the test windows ({d['n_null']} replicates; blocks of "
      f"{res['window_length'] // res['cfg']['metrics']['window_stride']} windows). Vector statistics use the mean "
      "over components of component-wise W1 scaled by the real-window SD. Secondary: posterior predictive "
      f"p-value (flag outside {d['ppp_interval']}).\n")
    gens = ["koopman", "iaaft", "ar", "shuffle"]
    idx = {(r["statistic"], r["generator"]): r for r in rows}
    names = list(dict.fromkeys(r["statistic"] for r in rows))
    lin = set(pr["statistics"]["linear"])
    w("| statistic | group | " + " | ".join(f"{gname} (W1/q95)" for gname in gens) + " | PPP (Koopman) |")
    w("|---|---|" + "---|" * len(gens) + "---|")
    for n in names:
        cells = []
        for gname in gens:
            r = idx[(n, gname)]
            cells.append(f"{'✓' if r['pass'] else '✗'} {_f(r['ratio'], 3)}" + (" (NaN)" if r["nan_fail"] else ""))
        ppp = idx[(n, "koopman")]["ppp"]
        flag = "" if not np.isfinite(ppp) or d["ppp_interval"][0] <= ppp <= d["ppp_interval"][1] else " ⚑"
        grp = "linear" if n in lin else ("nonlinear" if v["nonlinear_detected"] else "nonlinear (not gated)")
        w(f"| {n} | {grp} | " + " | ".join(cells) + f" | {_f(ppp, 3)}{flag} |")
    w("")
    if res.get("ppc"):
        npass = sum(r["ppc_pass"] for r in res["ppc"])
        w(f"Secondary posterior-predictive-calibrated null (not gated): {npass}/{len(res['ppc'])} statistics within "
          "the generator's own q95; failing: " + (", ".join(r["statistic"] for r in res["ppc"] if not r["ppc_pass"]) or "none") + ".\n")
    w(_fig("fig11_scorecard", captions, 11))
    pre = res["pretest"]
    w("### Nonlinearity pre-test (train segment vs IAAFT)\n")
    w(f"Segment [{pre['segment'][0]}, {pre['segment'][1]}) of the train set, {pre['n_surrogates']} IAAFT surrogates, "
      f"two-sided rank tests with Holm correction (α = {pr['pretest']['alpha']}): **{pre['n_rejections']} rejection(s)** → "
      f"{'nonlinear dynamics detected' if pre['nonlinear_detected'] else 'no detectable nonlinear dynamics'}.\n")
    w("| test | observed | surrogate median | p | reject |\n|---|---|---|---|---|")
    for t in pre["tests"]:
        w(f"| {t['test']} | {_f(t['observed'], 4)} | {_f(t['surrogate_median'], 4)} | {_f(t['p'], 3)} | {'yes' if t['reject'] else 'no'} |")
    w("")
    c = v["controls"]
    w("### Negative controls\n")
    w(f"- Shuffle must fail {list(c['shuffle_fails_required'])}: " +
      ", ".join(f"{k} {'failed ✓' if f else 'PASSED ✗'}" for k, f in c["shuffle_fails_required"].items()) + ".")
    if c["iaaft_required"]:
        w(f"- IAAFT must fail ≥ {pr['controls']['iaaft_must_fail_min_nonlinear']} nonlinear statistic: failed "
          f"{c['iaaft_failed_nonlinear'] or 'none'} → {'OK' if c['iaaft_ok'] else 'NOT MET (battery lacks power)'}.")
    else:
        w("- IAAFT requirement not applicable (no nonlinearity detected by the pre-test).")
    w("")
    inv = res["invariants"]
    w("### Invariants (estimated vs theoretical)\n")
    tr = res.get("truth") or {}
    unit = f"per {res['time_unit']}"
    w("| invariant | theory | real (test set) | synthetic median [IQR] |\n|---|---|---|---|")
    for key, lab in (("lle", f"λ₁ ({unit})"), ("d2", "D₂")):
        syn = np.asarray(inv["synthetic"][key])
        w(f"| {lab} | {_f(tr.get(key)) if tr.get(key) is not None else 'unknown'} | {_f(float(np.nanmedian(inv['real'][key])))} | "
          f"{_f(float(np.nanmedian(syn)))} [{_f(float(np.nanquantile(syn, 0.25)))}, {_f(float(np.nanquantile(syn, 0.75)))}] |")
    w(f"\n(Estimated on the full test set and on {len(inv['synthetic']['lle'])} synthetic series of the same length; "
      f"benchmark tolerances: |Δλ₁|/λ₁ ≤ {pr['benchmarks']['lle_rel_tol']}, |ΔD₂| ≤ {pr['benchmarks']['d2_abs_tol']}.)\n")
    for n, i in (("fig08_acf_psd_bands", 8), ("fig09_invariant_curves", 9), ("fig10_statistic_distributions", 10)):
        w(_fig(n, captions, i))

    # 6 sensitivity
    w("## 6. Sensitivity analysis\n")
    sens = res.get("sensitivity") or []
    if sens:
        w("| variant | MCMC | divergence rate | verdict | failing statistics |\n|---|---|---|---|---|")
        for s in sens:
            w(f"| {s['label']} | {s['mcmc']} | {s['divergence_rate']:.1%} | {s['verdict']} | {', '.join(s['failing']) or '-'} |")
        stable = len({s["verdict"] for s in sens}) == 1
        w(f"\nVerdict {'stable' if stable else 'NOT stable'} across variants.\n")
    else:
        w("Not run.\n")
    w(_fig("fig12_sensitivity", captions, 12))

    # 7 limitations
    w("## 7. Limitations\n")
    w("- **Finite-data invariants.** D₂ and λ₁ are estimated from windows of "
      f"{res['window_length']} points (and the {sp.test.size}-point test set for the invariants table). Both estimators are biased "
      "at these lengths (D₂ low, λ₁ dependent on the automatic fit range), noise inflates D₂ at small scales, and "
      "agreement is judged against the same estimator on real data rather than against exact values.")
    w("- **Continuous Koopman spectra.** For chaotic dynamics the Koopman operator has (partly) continuous spectrum. "
      "Finite EDMD eigenvalues are therefore approximations whose meaning is limited; ResDMD residuals flag spurious "
      "ones, but a non-spurious isolated eigenvalue is still not an exact eigenvalue of the infinite-dimensional operator.")
    w("- **Non-stationarity.** One time-invariant operator is fitted. " +
      ("The stationarity diagnostics flagged non-stationarity, so this assumption is questionable here." if st["nonstationary"]
       else "ADF/KPSS did not flag non-stationarity on the train set, but these tests have limited power against slow drifts and regime changes."))
    w(f"- **Dependence on (m, τ, F).** The model and the metrics depend on the delay embedding (m = {emb.m}, τ = {emb.tau}) "
      f"and the feature count (F = {res['F']}). Section 6 reports the verdict under m±1, F ∈ {{256, 1024}} and "
      "bootstrapped innovations; the sensitivity refits use shorter MCMC runs than the main fit.")
    w("- **Statistical scope.** Passing is failure to detect a difference with a calibrated test of demonstrated power "
      "(controls), not proof of identical dynamics. With ~20 gated statistics the probability that a perfect generator "
      "passes all of them is below 1 - 0.05 even though each test is calibrated; the rule was nevertheless kept as pre-registered.")
    w("- **Single test realisation.** The reference distribution comes from one held-out stretch; the null discrepancy "
      "compares its two halves, so power is limited by the test-set length.")
    t = res.get("timings", {})
    if t:
        w("\n## Appendix: wall time per stage\n")
        w("| stage | seconds |\n|---|---|")
        for k, s in t.items():
            w(f"| {k} | {s:.0f} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
