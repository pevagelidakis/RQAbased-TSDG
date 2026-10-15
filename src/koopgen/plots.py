"""The 12 report figures (matplotlib; 300-dpi PNG + vector PDF; consistent styling).

Colour roles (validated with the dataviz palette validator, all-pairs mode, light surface):
Koopman-MCMC = blue, IAAFT = orange, AR(p) = aqua (the three slots that pass all-pairs CVD and
normal-vision separation); Real = primary ink (black); Shuffle = neutral grey with a dashed
line / hatch as secondary encoding. Every multi-series panel carries a legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import metrics as Mx  # noqa: E402
from .embedding import delay_embed  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLORS = {"real": INK, "koopman": "#2a78d6", "iaaft": "#eb6834", "ar": "#1baf7a", "shuffle": MUTED}
LABELS = {"real": "Real (test)", "koopman": "Koopman-MCMC", "iaaft": "IAAFT", "ar": "AR(p)",
          "shuffle": "Shuffle"}
LS = {"real": "-", "koopman": "-", "iaaft": "-", "ar": "-", "shuffle": "--"}
ORDER = ["real", "koopman", "iaaft", "ar", "shuffle"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#0b3a75", "#2a78d6", "#9cc3ef", "#e3eefb"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d3", "#e34948"])


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
    })


def _save(fig, out: Path, name: str, captions: dict, caption: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    captions[name] = caption


def _tlabel(res):
    return f"t [{res['time_unit']}]" if res["dt"] != 1.0 or res["time_unit"] != "sample" else "t [samples]"


Y_LABEL = "y [train-SD units]"


# ----------------------------------------------------------------------------- fig 1
def fig_data(res, out, cap):
    sp, emb = res["split"], res["embedding"]
    t = np.arange(sp.y.size) * res["dt"]
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[: sp.n_train], sp.y[: sp.n_train], color=INK, lw=0.6, label="train (70%)")
    ax.plot(t[sp.n_train:], sp.y[sp.n_train:], color=COLORS["koopman"], lw=0.6, label="test (30%)")
    ax.axvline(t[sp.n_train], color=INK2, ls="--", lw=1)
    ax.set(xlabel=_tlabel(res), ylabel=Y_LABEL, title=f"{res['name']}: standardised series and temporal split")
    ax.legend(loc="upper right", ncol=2)
    ax = fig.add_subplot(gs[1, 0])
    lags = np.arange(emb.ami.size)
    ax.plot(lags, emb.ami, color=INK, marker="o", ms=2.5, label="AMI")
    ax.axhline(3 * emb.ami_floor, color=MUTED, ls=":", label="3× estimator bias floor")
    ax.axvline(emb.tau, color=COLORS["koopman"], ls="--", label=f"τ = {emb.tau}")
    ax.set(xlabel="lag [samples]", ylabel="mutual information [nats]", title=f"AMI ({emb.tau_method})")
    ax.set_xlim(0, min(lags[-1], max(30, 4 * emb.tau)))
    ax.legend()
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(emb.dims, emb.cao_E1, color=INK, marker="o", ms=3, label="Cao E1(d)")
    ax.plot(emb.dims, emb.cao_E2, color=COLORS["iaaft"], marker="s", ms=3, label="Cao E2(d)")
    ax.plot(emb.dims, emb.fnn, color=COLORS["ar"], marker="^", ms=3, label="FNN fraction")
    ax.axvline(emb.m, color=COLORS["koopman"], ls="--", label=f"m = {emb.m}")
    ax.set(xlabel="embedding dimension d", ylabel="ratio / fraction [-]", title="Cao and FNN")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "fig01_data_overview", cap,
          f"Standardised series with the 70/30 split; AMI (τ = {emb.tau}) and Cao/FNN curves (m = {emb.m}).")


# ----------------------------------------------------------------------------- fig 2
def fig_mcmc(res, out, cap):
    import arviz as az

    idata = res["idata"]
    names = ["log_s", "log_alpha_lin", "log_alpha_rff", "log_sigma"]
    az.style.use("default")
    style()
    axes = az.plot_trace(idata, var_names=names, compact=False, figsize=(10, 7))
    fig = axes.ravel()[0].figure
    for ax in axes[:, 1]:
        ax.set_xlabel("draw [-]")
    for ax in axes[:, 0]:
        ax.set_xlabel("value [log units]")
        ax.set_ylabel("density [-]")
    fig.suptitle("Trace plots (per chain)")
    fig.tight_layout()
    _save(fig, out, "fig02a_mcmc_trace", cap, "MCMC trace plots and marginal densities of θ, one line per chain.")
    axes = az.plot_rank(idata, var_names=names, figsize=(10, 3))
    fig = np.atleast_1d(axes).ravel()[0].figure
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xlabel("rank (all chains) [-]")
        ax.set_ylabel("chain [-]")
    fig.tight_layout()
    _save(fig, out, "fig02b_mcmc_rank", cap, "Rank plots: uniform bars indicate well-mixed chains.")
    axes = az.plot_autocorr(idata, var_names=names, combined=True, max_lag=100, figsize=(10, 3))
    fig = np.atleast_1d(axes).ravel()[0].figure
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xlabel("lag [draws]")
        ax.set_ylabel("autocorrelation [-]")
    fig.tight_layout()
    _save(fig, out, "fig02c_mcmc_autocorr", cap, "Autocorrelation of the posterior draws (chains combined).")
    sub = idata.sel(draw=slice(None, None, max(1, idata.posterior.sizes["draw"] // 2000)))
    axes = az.plot_pair(sub, var_names=names, divergences=True, kind="scatter",
                        scatter_kwargs={"alpha": 0.15, "s": 3, "color": COLORS["koopman"]},
                        divergences_kwargs={"color": "#e34948", "marker": "x", "markersize": 5},
                        figsize=(8, 8))
    fig = np.atleast_1d(axes).ravel()[0].figure
    fig.suptitle(f"Pair plot of θ (thinned); divergences as red × (total {res['gates'].divergences})")
    _save(fig, out, "fig02d_mcmc_pairs", cap, "Pair plot of θ (thinned) with divergent transitions overlaid.")
    ax = az.plot_energy(idata, figsize=(5, 3.5))
    fig = ax.figure
    ax.set_xlabel("energy (centred) [-]")
    ax.set_ylabel("density [-]")
    _save(fig, out, "fig02e_mcmc_energy", cap,
          "Energy plot: marginal energy vs energy transition distributions (E-BFMI per chain in the table).")
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, n in zip(axs, names):
        post = idata.posterior[n].values.ravel()
        pr = idata.prior[n].values.ravel()
        lo, hi = np.quantile(post, [0.0005, 0.9995])
        w = hi - lo
        grid = np.linspace(min(lo - w, np.quantile(pr, 0.01)), max(hi + w, np.quantile(pr, 0.99)), 400)
        from scipy.stats import gaussian_kde, norm
        mu, sd = res["prior"].as_dict()[n]
        ax.plot(grid, norm.pdf(grid, mu, sd), color=MUTED, ls="--", label="prior")
        ax.plot(grid, gaussian_kde(post[:: max(1, post.size // 20000)])(grid), color=COLORS["koopman"], label="posterior")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, None)
        ax.set(xlabel=f"{n} [log units]", ylabel="density [-]")
    axs[0].legend()
    fig.tight_layout()
    _save(fig, out, "fig02f_prior_posterior", cap, "Prior (dashed) vs posterior densities of θ (log y-axis).")
    g = res["gates"]
    fig, ax = plt.subplots(figsize=(9, 1.9))
    ax.axis("off")
    rows = [[n, f"{g.table[n]['mean']:.4f}", f"{g.table[n]['sd']:.4f}", f"{g.table[n]['r_hat']:.4f}",
             f"{g.table[n]['ess_bulk']:.0f}", f"{g.table[n]['ess_tail']:.0f}"] for n in names]
    tb = ax.table(cellText=rows, colLabels=["parameter", "mean", "sd", "R-hat", "ESS bulk", "ESS tail"],
                  loc="center", cellLoc="center")
    tb.scale(1, 1.3)
    ax.set_title(f"Gates: {'PASS' if g.passed else 'FAIL'} - divergences {g.divergences}, "
                 f"E-BFMI {', '.join(f'{v:.2f}' for v in g.ebfmi)}", fontsize=9)
    _save(fig, out, "fig02g_rhat_ess_table", cap, "R-hat / ESS / divergence / E-BFMI summary table.")


# ----------------------------------------------------------------------------- fig 3
def fig_spectrum(res, out, cap):
    from .spectrum import leading_frequencies

    sp = res["spectrum"]
    mu, rr, summ = sp["mu"], sp["res"], sp["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK2, lw=1)
    sub = slice(None, None, max(1, mu.shape[0] // 200))
    m, r = mu[sub].ravel(), rr[sub].ravel()
    order = np.argsort(-r)
    sc = ax.scatter(m.real[order], m.imag[order], c=np.log10(np.maximum(r[order], 1e-6)), cmap=SEQ,
                    vmin=-3, vmax=0.5, s=5, alpha=0.7, lw=0)
    ax.text(0.02, 0.02, f"dark = low residual; spurious if > {summ['threshold']}", transform=ax.transAxes,
            fontsize=7, color=INK2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10 ResDMD residual [-]")
    ax.set(xlabel="Re μ [-]", ylabel="Im μ [-]", aspect="equal",
           title=f"Posterior EDMD eigenvalues ({summ['n_draws']} draws)")
    ax = axs[1]
    f, p = Mx.welch_log_psd(res["split"].test, res["cfg"]["metrics"]["welch_nperseg"])
    fr = f / res["dt"]
    ax.plot(fr, p, color=INK, label="Welch PSD (real test)")
    ax.set(xlabel=f"frequency [cycles per {res['time_unit']}]", ylabel="log10 PSD [y² per frequency unit]")
    ax2 = ax.twinx()
    bins = np.linspace(0, fr.max(), 80)
    fq = leading_frequencies(mu, rr, summ["threshold"], 0.0) / res["dt"]
    nontriv = (mu.imag > 1e-12)
    rk = np.where(nontriv, rr, np.inf)
    best = np.argsort(rk, axis=1)[:, :5]
    fb = np.angle(np.take_along_axis(mu, best, 1)) / (2 * np.pi) / res["dt"]
    fb = fb[np.isfinite(np.take_along_axis(rk, best, 1))]
    ax2.hist(fb, bins=bins, color=COLORS["koopman"], alpha=0.35,
             label="5 lowest-residual complex eigenvalues per draw")
    if fq.size:
        ax2.hist(fq, bins=bins, color="#0b3a75", alpha=0.8,
                 label=f"non-spurious complex eigenvalues (res ≤ {summ['threshold']})")
    ax2.set_ylabel("eigenvalue count (pooled draws) [-]")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Koopman frequencies vs real PSD")
    fig.tight_layout()
    _save(fig, out, "fig03_koopman_spectrum", cap,
          f"EDMD eigenvalues coloured by ResDMD residual (spurious fraction {summ['spurious_fraction']:.2f}; "
          f"draws with non-spurious |μ|>1+δ: {summ['unstable_draw_fraction']:.2f}); frequencies of "
          "non-spurious eigenvalues over the real Welch PSD.")


# ----------------------------------------------------------------------------- fig 4
def fig_forecast(res, out, cap):
    fc = res["forecast"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    ens, truth, hist = fc["fan_ensemble"], fc["fan_truth"], fc["fan_history"]
    h = np.arange(1, ens.shape[1] + 1) * res["dt"]
    hh = np.arange(-hist.size + 1, 1) * res["dt"]
    q = np.quantile(ens, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.plot(hh, hist, color=INK, label="observed (history)")
    ax.fill_between(h, q[0], q[4], color=COLORS["koopman"], alpha=0.18, lw=0, label="90% PI")
    ax.fill_between(h, q[1], q[3], color=COLORS["koopman"], alpha=0.35, lw=0, label="50% PI")
    ax.plot(h, q[2], color=COLORS["koopman"], label="median")
    ax.plot(h, truth, color=INK, ls=":", label="observed (future)")
    ax.set(xlabel=f"lead time [{res['time_unit']}]", ylabel=Y_LABEL, title="Posterior predictive fan chart (test origin)")
    ax.legend(loc="upper left", fontsize=7)
    ax = axs[1]
    hz = np.arange(1, fc["crps_koopman"].size + 1) * res["dt"]
    ax.plot(hz, fc["crps_koopman"], color=COLORS["koopman"], label="Koopman-MCMC")
    ax.plot(hz, fc["crps_ar"], color=COLORS["ar"], label=f"AR({res['ar_model'].p})")
    ax.plot(hz, fc["crps_persistence"], color=MUTED, ls="--", label="persistence")
    ax.set(xlabel=f"lead time [{res['time_unit']}]", ylabel="mean CRPS [train-SD units]",
           title=f"CRPS vs horizon ({fc['n_origins']} test origins)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "fig04_forecast_skill", cap,
          "Posterior predictive fan chart from a test origin, and mean CRPS vs horizon against AR(p) and persistence.")


# ----------------------------------------------------------------------------- fig 5
def fig_examples(res, out, cap):
    y = res["split"].test
    syn = res["series"]["koopman"]
    ok = np.nonzero(~res["diverged"])[0][:3]
    L = min(y.size, 1000)
    t = np.arange(L) * res["dt"]
    fig, axs = plt.subplots(1 + len(ok), 1, figsize=(10, 1.6 * (1 + len(ok))), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    axs[0].plot(t, y[:L], color=INK, lw=0.7)
    axs[0].set_title("Real test segment", loc="left")
    for ax, i in zip(axs[1:], ok):
        ax.plot(t, syn[i, :L], color=COLORS["koopman"], lw=0.7)
        ax.set_title(f"Koopman-MCMC synthetic series #{i}", loc="left")
    for ax in axs:
        ax.set_ylabel(Y_LABEL)
    axs[-1].set_xlabel(_tlabel(res))
    fig.tight_layout()
    _save(fig, out, "fig05_examples", cap, "Real test segment and three synthetic series on the same axes.")


# ----------------------------------------------------------------------------- fig 6
def fig_phase(res, out, cap):
    m, tau = res["metric_params"].m, res["metric_params"].tau
    srcs = [("real", res["split"].test), ("koopman", res["series"]["koopman"][np.nonzero(~res["diverged"])[0][0]]),
            ("iaaft", res["series"]["iaaft"][0])]
    fig = plt.figure(figsize=(11, 10))
    for j, (k, y) in enumerate(srcs):
        y = y[:3000]
        X = delay_embed(y, max(3, m), tau)
        ax = fig.add_subplot(3, 3, 1 + j)
        ax.scatter(X[:, 1], X[:, 0], color=COLORS[k], s=0.6, lw=0, alpha=0.6)
        ax.set(xlabel="y(t−τ) [train-SD units]", ylabel="y(t) [train-SD units]", title=f"{LABELS[k]}: 2-D delay")
        ax = fig.add_subplot(3, 3, 4 + j, projection="3d")
        ax.scatter(X[:, 2], X[:, 1], X[:, 0], color=COLORS[k], s=0.4, lw=0, alpha=0.5)
        ax.set_xlabel("y(t−2τ)", fontsize=7)
        ax.set_ylabel("y(t−τ)", fontsize=7)
        ax.set_zlabel("y(t) [SD]", fontsize=7)
        ax.set_title(f"{LABELS[k]}: 3-D delay", fontsize=9)
        ax = fig.add_subplot(3, 3, 7 + j)
        ax.scatter(y[:-1], y[1:], s=1, color=COLORS[k], alpha=0.4, lw=0)
        ax.set(xlabel="y(t) [train-SD units]", ylabel="y(t+1) [train-SD units]", title=f"{LABELS[k]}: return map")
    fig.tight_layout()
    _save(fig, out, "fig06_phase_portraits", cap,
          f"Delay-embedding phase portraits (τ = {tau}) and one-step return maps: real vs Koopman-MCMC vs IAAFT.")


# ----------------------------------------------------------------------------- fig 7
def fig_rps(res, out, cap):
    P = res["metric_params"]
    W = res["window_length"]
    rr = res["cfg"]["metrics"]["rqa"]["rr"]
    srcs = [("real", res["split"].test[:W])]
    ok = np.nonzero(~res["diverged"])[0][0]
    srcs += [("koopman", res["series"]["koopman"][ok][:W]), ("iaaft", res["series"]["iaaft"][0][:W]),
             ("ar", res["series"]["ar"][0][:W]), ("shuffle", res["series"]["shuffle"][0][:W])]
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, (k, y) in zip(axs, srcs):
        R, _ = Mx.recurrence_plot(delay_embed(y, P.m, P.tau), rr, P.cfg["rqa"]["theiler"])
        ax.imshow(R, cmap=LinearSegmentedColormap.from_list("rp", ["white", COLORS[k] if k != "real" else INK]),
                  origin="lower", interpolation="nearest")
        ax.set(xlabel="i [samples]", ylabel="j [samples]", title=LABELS[k])
        ax.grid(False)
    fig.suptitle(f"Recurrence plots at fixed recurrence rate {rr:.0%} (m = {P.m}, τ = {P.tau}, {W} points)")
    fig.tight_layout()
    _save(fig, out, "fig07_recurrence_plots", cap, f"Recurrence plots at RR = {rr:.0%} for the five generators.")


# ----------------------------------------------------------------------------- fig 8
def fig_acf_psd(res, out, cap):
    st = res["stats"]
    L = res["cfg"]["metrics"]["acf_lags"]
    lags = np.arange(1, L + 1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    f = np.fft.rfftfreq(res["cfg"]["metrics"]["welch_nperseg"]) / res["dt"]
    for ax, key, x, xl in ((axs[0], "acf", lags * res["dt"], f"lag [{res['time_unit']}]"),
                           (axs[1], "psd", f, f"frequency [cycles per {res['time_unit']}]")):
        for g in ("koopman", "ar", "iaaft"):
            lo, hi = np.nanquantile(st[g][key], [0.025, 0.975], axis=0)
            ax.fill_between(x, lo, hi, color=COLORS[g], alpha=0.22, lw=0, label=f"{LABELS[g]} 95% band")
        rlo, rhi = np.nanquantile(st["real"][key], [0.025, 0.975], axis=0)
        ax.plot(x, np.nanmean(st["real"][key], axis=0), color=INK, label="real (mean of test windows)")
        ax.plot(x, rlo, color=INK, lw=0.6, ls=":")
        ax.plot(x, rhi, color=INK, lw=0.6, ls=":", label="real window range (2.5–97.5%)")
        ax.set_xlabel(xl)
        ax.legend(fontsize=7)
    axs[0].set(ylabel="autocorrelation [-]", title="ACF")
    axs[1].set(ylabel="log10 PSD [y² per frequency unit]", title="Welch PSD")
    fig.tight_layout()
    _save(fig, out, "fig08_acf_psd_bands", cap, "ACF and Welch PSD: 95% generator bands over the real curve.")


# ----------------------------------------------------------------------------- fig 9
def fig_invariant_curves(res, out, cap):
    from .validate import nonlinear_stats

    P = res["metric_params"]
    W = res["window_length"]
    ok = np.nonzero(~res["diverged"])[0][0]
    srcs = [("real", res["split"].test[:W]), ("koopman", res["series"]["koopman"][ok][:W]),
            ("iaaft", res["series"]["iaaft"][0][:W])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k, y in srcs:
        c = nonlinear_stats(y, P, curves=True)["_curves"]
        sel = c["C"] > 0
        axs[0].plot(np.log10(c["r"][sel]), np.log10(c["C"][sel]), color=COLORS[k], marker="o", ms=2,
                    label=f"{LABELS[k]}")
        if np.isfinite(c["d2_fit"][0]):
            axs[0].axvspan(np.log10(c["d2_fit"][0]), np.log10(c["d2_fit"][1]), color=COLORS[k], alpha=0.08)
        kk = c["lle_k"] * res["dt"]
        axs[1].plot(kk, c["lle_curve"], color=COLORS[k], label=LABELS[k])
        a, b = c["lle_fit"]
        axs[1].plot(kk[a:b + 1], c["lle_curve"][a:b + 1], color=COLORS[k], lw=3.5, alpha=0.4)
    axs[0].set(xlabel="log10 r [train-SD units]", ylabel="log10 C(r) [-]",
               title="Correlation sums (shaded: fitted scaling region)")
    axs[1].set(xlabel=f"time after separation [{res['time_unit']}]", ylabel="⟨ln divergence⟩ [-]",
               title="Rosenstein divergence curves (thick: fit range)")
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    _save(fig, out, "fig09_invariant_curves", cap,
          "Correlation-sum curves with scaling regions and Rosenstein divergence curves with fit ranges.")


# ----------------------------------------------------------------------------- fig 10
def fig_distributions(res, out, cap):
    st = res["stats"]
    names = list(res["cfg"]["preregistered"]["statistics"]["linear"]) + \
        list(res["cfg"]["preregistered"]["statistics"]["nonlinear"])
    comp_label = {"acf": "acf (lag 1)", "pacf": "pacf (lag 1)", "psd": "log PSD (lowest bin)",
                  "quantiles": "q05", "time_reversal": "TRA (k=1)", "c3": "C3 (k=1)"}
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(14, 2.4 * nrow))
    for ax, n in zip(axs.ravel(), names):
        data, cols = [], []
        for g in ORDER:
            v = st[g][n][:, 0 if n != "psd" else 1]
            data.append(v[np.isfinite(v)] if np.isfinite(v).any() else np.array([np.nan]))
            cols.append(COLORS[g])
        parts = ax.violinplot([d if d.size > 1 else np.r_[d, d] for d in data], showextrema=False,
                              showmedians=True)
        for b, c, g in zip(parts["bodies"], cols, ORDER):
            b.set_facecolor(c)
            b.set_alpha(0.55)
            b.set_edgecolor(c)
            if g == "shuffle":
                b.set_hatch("///")
        parts["cmedians"].set_color(INK)
        ax.set_xticks(range(1, 6), ["Real", "Koop", "IAAFT", "AR", "Shuf"], fontsize=7)
        ax.set_title(comp_label.get(n, n), fontsize=8)
        ax.set_ylabel("value [stat. units]", fontsize=7)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, out, "fig10_statistic_distributions", cap,
          "Window-statistic distributions for every statistic across the five generators (vector statistics: first component).")


# ----------------------------------------------------------------------------- fig 11
def fig_scorecard(res, out, cap):
    rows = res["scorecard"]
    names = list(dict.fromkeys(r["statistic"] for r in rows))
    gens = ["koopman", "iaaft", "ar", "shuffle"]
    lin = set(res["cfg"]["preregistered"]["statistics"]["linear"])
    M = np.full((len(names), len(gens)), np.nan)
    P = np.zeros_like(M, dtype=bool)
    for r in rows:
        i, j = names.index(r["statistic"]), gens.index(r["generator"])
        M[i, j] = r["ratio"]
        P[i, j] = r["pass"]
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(names) + 1.6))
    L2 = np.log2(np.clip(M, 1 / 64, 64))
    im = ax.imshow(L2, cmap=DIV, norm=TwoSlopeNorm(0, -6, 6), aspect="auto")
    for i in range(len(names)):
        for j in range(len(gens)):
            txt = f"{'✓' if P[i, j] else '✗'} {M[i, j]:.2f}" if np.isfinite(M[i, j]) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(gens)), [LABELS[g] for g in gens])
    ax.set_yticks(range(len(names)), [("[L] " if n in lin else "[NL] ") + n for n in names], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("log2( W1 / null q95 )  [≤ 0 passes]")
    nl_note = "" if res["verdict"]["nonlinear_detected"] else " (NL rows not gated: no nonlinearity detected)"
    ax.set_title(f"Scorecard - verdict: {res['verdict']['verdict']}{nl_note}", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig11_scorecard", cap,
          "Scorecard: pass (✓) / fail (✗) and W1/null-q95 ratio per statistic and generator (headline figure).")


# ----------------------------------------------------------------------------- fig 12
def fig_sensitivity(res, out, cap):
    sens = res.get("sensitivity") or []
    if not sens:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "Sensitivity analysis not run in this mode", ha="center")
        _save(fig, out, "fig12_sensitivity", cap, "Sensitivity analysis (not run in this mode).")
        return
    names = list(dict.fromkeys(r["statistic"] for r in sens[0]["rows"]))
    fig, ax = plt.subplots(figsize=(0.36 * len(names) + 3.5, 0.45 * len(sens) + 1.8))
    M = np.zeros((len(sens), len(names)))
    for i, s in enumerate(sens):
        d = {r["statistic"]: r for r in s["rows"] if r["generator"] == "koopman"}
        for j, n in enumerate(names):
            M[i, j] = np.log2(np.clip(d[n]["ratio"], 1 / 64, 64))
            ax.text(j, i, "✓" if d[n]["pass"] else "✗", ha="center", va="center", fontsize=8)
    ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -6, 6), aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=70, fontsize=7)
    ax.set_yticks(range(len(sens)), [f"{s['label']} → {s['verdict']}" for s in sens], fontsize=8)
    ax.grid(False)
    ax.set_title("Sensitivity of the Koopman-MCMC scorecard (colour: log2 W1/null q95; ✓ pass)")
    fig.tight_layout()
    _save(fig, out, "fig12_sensitivity", cap,
          "Verdict stability across (m±1), F ∈ {256, 1024} and the innovation scheme.")


def all_figures(res, out: Path) -> dict:
    style()
    cap = {}
    for f in (fig_data, fig_mcmc, fig_spectrum, fig_forecast, fig_examples, fig_phase, fig_rps,
              fig_acf_psd, fig_invariant_curves, fig_distributions, fig_scorecard, fig_sensitivity):
        style()
        f(res, out, cap)
    return cap
