"""Diagnostic plots for a screened batch of designs."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def screen_dashboard(df: pd.DataFrame, out_path: str,
                     pae_max: float = 10.0, plddt_min: float = 80.0):
    """4-panel screening dashboard. Returns the Figure."""
    passed = df["pass"] if "pass" in df else pd.Series(True, index=df.index)
    colors = np.where(passed, "#2a7f3f", "#b0b0b0")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (1) the money plot: pLDDT(binder) vs pAE_interaction
    ax = axes[0, 0]
    ax.scatter(df["pae_interaction"], df["plddt_binder"], c=colors,
               s=40, edgecolor="k", linewidth=0.3, alpha=0.85)
    ax.axvline(pae_max, ls="--", c="crimson", lw=1)
    ax.axhline(plddt_min, ls="--", c="crimson", lw=1)
    ax.set_xlabel("pAE interaction (Å)")
    ax.set_ylabel("binder pLDDT")
    ax.set_title("Confidence vs interface error")
    ax.text(0.02, 0.03, "accept →", color="#2a7f3f", transform=ax.transAxes,
            fontsize=9, va="bottom")

    # (2) iptm distribution (or contacts if no iptm)
    ax = axes[0, 1]
    if "iptm" in df and df["iptm"].notna().any():
        ax.hist(df["iptm"].dropna(), bins=20, color="#3b6ea5", edgecolor="k")
        ax.axvline(0.5, ls="--", c="crimson", lw=1)
        ax.set_xlabel("ipTM")
        ax.set_title("Interface pTM distribution")
    else:
        ax.hist(df["n_contacts"], bins=20, color="#3b6ea5", edgecolor="k")
        ax.set_xlabel("interface contacts")
        ax.set_title("Interface contact distribution")
    ax.set_ylabel("designs")

    # (3) composite score ranking
    ax = axes[1, 0]
    if "composite_score" in df:
        s = df.sort_values("composite_score", ascending=True)
        ax.barh(range(len(s)), s["composite_score"],
                color=np.where(s["pass"], "#2a7f3f", "#c0c0c0"))
        ax.set_yticks([])
        ax.set_xlabel("composite score")
        ax.set_title(f"Ranked designs (n={len(s)})")
    else:
        ax.axis("off")

    # (4) binder length vs contacts, sized by score
    ax = axes[1, 1]
    sizes = 30
    if "composite_score" in df:
        sizes = 10 + (df["composite_score"] / df["composite_score"].max() * 120)
    ax.scatter(df["binder_len"], df["n_contacts"], c=colors, s=sizes,
               edgecolor="k", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("binder length (residues)")
    ax.set_ylabel("interface contacts")
    ax.set_title("Size vs interface engagement")

    n_pass = int(passed.sum())
    fig.suptitle(f"Mpro binder screen — {n_pass}/{len(df)} pass filters",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    return fig
