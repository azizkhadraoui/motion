#!/usr/bin/env python
"""Builds the paper figures from results already measured (8c / 8d / 8f).
Vector PDF at NeurIPS column width. No cluster access needed."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_LFM, C_CDFM, C_REF, C_HL = "#1f4e79", "#c0504d", "#7f7f7f", "#2e8b57"
OUT = "/home/claude/fig"

# ---------------------------------------------------------------- Fig: in-process schedule ablation (8c)
def fig_inproc():
    windows = [10, 20, 50]
    strides = [1, 2, 4]
    fid = {(10,1):55.6,(10,2):42.5,(10,4):21.2,
           (20,1):55.8,(20,2):55.9,(20,4):43.5,
           (50,1):55.9,(50,2):55.0,(50,4):56.4}
    unc, post = 0.147, 0.142
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    w = 0.25
    x = np.arange(len(windows))
    for i, k in enumerate(strides):
        vals = [fid[(wi, k)] for wi in windows]
        ax.bar(x + (i-1)*w, vals, w, label=f"stride $k$={k}",
               color=plt.cm.Blues(0.45 + 0.22*i), edgecolor="white", linewidth=0.5)
    ax.axhline(unc, color=C_REF, ls="--", lw=0.9)
    ax.axhline(post, color=C_HL, ls="-", lw=0.9)
    ax.text(-0.48, unc*1.55, "unconstrained (0.147)", color=C_REF, fontsize=6, ha="left")
    ax.text(-0.48, post*0.40, "post-decode (0.142)", color=C_HL, fontsize=6, ha="left")
    ax.set_yscale("log")
    ax.set_ylim(0.04, 260)
    ax.set_xticks(x); ax.set_xticklabels([f"{wi}%" for wi in windows])
    ax.set_xlabel("projection window (final fraction of ODE trajectory)")
    ax.set_ylabel("FID $\\downarrow$ (log)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=1.0,
              handlelength=1.1, borderaxespad=0.2)
    ax.set_title("Hard in-process projection: no schedule recovers", pad=4)
    fig.savefig(f"{OUT}/fig_inproc_ablation.pdf")
    fig.savefig(f"{OUT}/fig_inproc_ablation.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig: soft-guidance sweep (8d)
def fig_guidance():
    w = np.array([0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00])
    lfm_fid  = np.array([0.147, 0.190, 0.337, 3.418, 15.159, 17.852, 20.812])
    cdfm_fid = np.array([0.2097, 0.2092, 0.2084, 0.2063, 0.2045, 0.2012, 0.1958])
    lfm_ble  = np.array([0.00533, 0.00512, 0.00494, 0.00467, 0.00451, 0.00434, 0.00423])
    cdfm_ble = np.array([0.00346, 0.00345, 0.00348, 0.00366, 0.00409, 0.00514, 0.00731])
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.95))
    a = axs[0]
    a.plot(w, lfm_fid, "o-", color=C_LFM, ms=3, lw=1.2, label="LFM (latent)")
    a.plot(w, cdfm_fid, "s--", color=C_CDFM, ms=3, lw=1.2, label="CDFM (direct)")
    a.set_yscale("log"); a.set_xlabel("guidance strength $w$"); a.set_ylabel("FID $\\downarrow$ (log)")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.annotate("LFM (latent)", xy=(0.55, 15.2), xytext=(0.62, 4.2), color=C_LFM, fontsize=7)
    a.annotate("CDFM (direct)", xy=(1.0, 0.201), xytext=(0.62, 0.42), color=C_CDFM, fontsize=7)
    a.set_ylim(0.09, 45)
    a.set_title("Generation quality", pad=3)
    b = axs[1]
    b.plot(w, lfm_ble*1e3, "o-", color=C_LFM, ms=3, lw=1.2)
    b.plot(w, cdfm_ble*1e3, "s--", color=C_CDFM, ms=3, lw=1.2)
    b.axhline(5.98, color=C_REF, ls=":", lw=0.9)
    b.text(0.02, 6.35, "frozen-AE round-trip floor", color=C_REF, fontsize=6, ha="left")
    b.axhline(0.0, color=C_HL, lw=0.9)
    b.text(0.02, 0.35, "exact satisfaction (post-decode projection)", color=C_HL, fontsize=6, ha="left")
    b.annotate("LFM", xy=(1.6, 4.3), color=C_LFM, fontsize=7)
    b.annotate("CDFM", xy=(1.42, 4.9), color=C_CDFM, fontsize=7)
    b.set_xlabel("guidance strength $w$"); b.set_ylabel("mean BLE $\\downarrow$ ($\\times10^{-3}$)")
    b.set_ylim(-0.6, 8.4)
    b.set_title("Constraint satisfaction", pad=3)
    fig.savefig(f"{OUT}/fig_guidance_sweep.pdf")
    fig.savefig(f"{OUT}/fig_guidance_sweep.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig: sampler-strategy ablation (8f)
def fig_sampler():
    labels = ["plain Euler\n(baseline)", "CFG resc. 0.7", "CFG resc. 1.0", "cosine steps",
              "power steps", "Heun", "Heun+cosine", "combined"]
    fid = [0.1473, 0.1613, 0.1708, 0.1516, 0.1651, 0.1503, 0.1515, 0.1637]
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    cols = [C_HL] + [C_REF]*(len(fid)-1)
    ax.bar(range(len(fid)), fid, 0.68, color=cols, edgecolor="white", linewidth=0.5)
    ax.axhline(fid[0], color=C_HL, ls="--", lw=0.9)
    ax.set_ylim(0.13, 0.178)
    ax.set_xticks(range(len(fid)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6)
    ax.set_ylabel("FID $\\downarrow$")
    ax.set_title("No sampling modification improves the baseline", pad=4)
    for i, v in enumerate(fid):
        ax.text(i, v + 0.0012, f"{v:.4f}", ha="center", fontsize=5.5)
    fig.savefig(f"{OUT}/fig_sampler_ablation.pdf")
    fig.savefig(f"{OUT}/fig_sampler_ablation.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig: replicated main result (8b)
def fig_main():
    labels = ["LFM", "LFM\n+post-decode", "CDFM", "CDFM\n+post-decode"]
    mean = [0.169, 0.154, 0.209, 0.172]
    ci   = [0.006, 0.005, 0.009, 0.008]
    ble  = [5.33, 0.0, 3.45, 0.0]
    cols = [C_LFM, C_LFM, C_CDFM, C_CDFM]
    hatch = ["", "//", "", "//"]
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.95))
    a = axs[0]
    a.bar(range(4), mean, 0.6, yerr=ci, capsize=3, color=cols, hatch=hatch,
          edgecolor="white", linewidth=0.6, error_kw=dict(lw=0.9, ecolor="0.25"))
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=6.5)
    a.set_ylabel("FID $\\downarrow$"); a.set_ylim(0, 0.24)
    a.set_title("FID, 20 sampling seeds (95% CI)", pad=3)
    b = axs[1]
    b.bar(range(4), ble, 0.6, color=cols, hatch=hatch, edgecolor="white", linewidth=0.6)
    b.set_xticks(range(4)); b.set_xticklabels(labels, fontsize=6.5)
    b.set_ylabel("mean BLE $\\downarrow$ ($\\times10^{-3}$)"); b.set_ylim(0, 6.4)
    for i, v in enumerate(ble):
        b.text(i, v + 0.15, "exact 0" if v == 0 else f"{v:.2f}", ha="center", fontsize=6)
    b.set_title("Bone-length error (exact by construction)", pad=3)
    fig.savefig(f"{OUT}/fig_main_replicated.pdf")
    fig.savefig(f"{OUT}/fig_main_replicated.png"); plt.close(fig)

for f in (fig_inproc, fig_guidance, fig_sampler, fig_main):
    f(); print("built", f.__name__)
