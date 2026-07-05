"""Acceptance filters and a composite score for binder designs.

Default thresholds follow the widely-used AF2/ColabFold binder-screening
convention (Bennett et al. 2023 and the RFdiffusion binder pipelines):
    pLDDT(binder)      >= 80
    pAE_interaction    <= 10  (Angstrom)
    iptm               >= 0.50  (when available)
    n_contacts         >= 3
These are deliberately editable — pass a Thresholds() to override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    plddt_binder_min: float = 80.0
    pae_interaction_max: float = 10.0
    iptm_min: float = 0.50
    plddt_interface_min: float = 80.0
    min_contacts: int = 3
    rmsd_to_design_max: float = 2.0   # only applied if column present & non-null


def apply_filters(df: pd.DataFrame, thr: Thresholds = Thresholds()
                  ) -> pd.DataFrame:
    """Add boolean pass columns + an overall `pass` and `fail_reasons`."""
    out = df.copy()
    checks: Dict[str, pd.Series] = {}

    checks["plddt_binder"] = out["plddt_binder"] >= thr.plddt_binder_min
    checks["pae_interaction"] = out["pae_interaction"] <= thr.pae_interaction_max
    if "iptm" in out and out["iptm"].notna().any():
        checks["iptm"] = (out["iptm"] >= thr.iptm_min) | out["iptm"].isna()
    checks["plddt_interface"] = out["plddt_interface"] >= thr.plddt_interface_min
    checks["contacts"] = out["n_contacts"] >= thr.min_contacts
    if "rmsd_to_design" in out and out["rmsd_to_design"].notna().any():
        checks["rmsd_to_design"] = (
            (out["rmsd_to_design"] <= thr.rmsd_to_design_max)
            | out["rmsd_to_design"].isna())

    for k, v in checks.items():
        out[f"pass_{k}"] = v.fillna(False)

    pass_cols = [f"pass_{k}" for k in checks]
    out["pass"] = out[pass_cols].all(axis=1)

    def reasons(row):
        return ";".join(k for k in checks if not row[f"pass_{k}"])
    out["fail_reasons"] = out.apply(reasons, axis=1)
    return out


def composite_score(df: pd.DataFrame) -> pd.Series:
    """A single 0-100 ranking score blending confidence, interface, geometry.

    Higher is better. Metrics are min-max normalised across the batch, so the
    score is relative to the current set of designs (good for ranking, not an
    absolute quality bar — use the filters for that).
    """
    def norm(col, invert=False):
        x = df[col].astype(float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or hi - lo < 1e-9:
            z = pd.Series(0.5, index=df.index)
        else:
            z = (x - lo) / (hi - lo)
        return (1 - z) if invert else z

    parts = []
    weights = []
    parts.append(norm("plddt_binder"));        weights.append(0.30)
    parts.append(norm("pae_interaction", invert=True)); weights.append(0.30)
    parts.append(norm("plddt_interface"));      weights.append(0.15)
    parts.append(norm("n_contacts"));           weights.append(0.15)
    if "iptm" in df and df["iptm"].notna().any():
        parts.append(norm("iptm").fillna(0.5)); weights.append(0.10)
    w = np.array(weights); w = w / w.sum()
    # A missing component (e.g. no interface -> NaN interface pLDDT) scores 0
    # for that term rather than nulling the whole design.
    stack = np.vstack([np.nan_to_num(p.values, nan=0.0) for p in parts])
    score = (w[:, None] * stack).sum(axis=0) * 100
    return pd.Series(np.round(score, 2), index=df.index)
