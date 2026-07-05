"""End-to-end screening pipeline: discover -> score -> filter -> rank -> report."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .discover import find_predictions, Prediction
from .metrics import score_complex
from .filters import Thresholds, apply_filters, composite_score
from .plots import screen_dashboard


@dataclass
class ScreenConfig:
    target_chain: str = "A"
    binder_chain: str = "B"
    interface_cutoff: float = 5.0
    best_rank_only: bool = True
    design_dir: Optional[str] = None       # RFdiffusion backbones for RMSD
    thresholds: Thresholds = None

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = Thresholds()


def _find_design_pdb(design_dir: Optional[str], name: str) -> Optional[str]:
    if not design_dir:
        return None
    import glob
    hits = glob.glob(os.path.join(design_dir, f"{name}*.pdb"))
    return hits[0] if hits else None


def run_screen(pred_dir: str, out_dir: str,
               config: ScreenConfig = None) -> pd.DataFrame:
    """Screen all ColabFold predictions under `pred_dir`.

    Writes: metrics.csv, ranked.csv, passing.csv, dashboard.png, summary.md
    Returns the ranked DataFrame.
    """
    config = config or ScreenConfig()
    os.makedirs(out_dir, exist_ok=True)

    preds: List[Prediction] = find_predictions(pred_dir, config.best_rank_only)
    if not preds:
        raise FileNotFoundError(f"No PDB predictions found under {pred_dir!r}")

    rows = []
    errors = []
    for p in preds:
        design_pdb = _find_design_pdb(config.design_dir, p.name)
        try:
            m = score_complex(
                pdb_path=p.pdb,
                scores_path=p.scores,
                target_chain=config.target_chain,
                binder_chain=config.binder_chain,
                design_pdb=design_pdb,
                interface_cutoff=config.interface_cutoff,
                name=p.name,
            )
            row = m.as_row()
            row["pdb"] = os.path.relpath(p.pdb, pred_dir)
            row["has_scores_json"] = p.scores is not None
            rows.append(row)
        except Exception as e:  # keep going; record the failure
            errors.append({"name": p.name, "pdb": p.pdb, "error": repr(e)})

    if not rows:
        raise RuntimeError(f"All {len(preds)} predictions failed to score. "
                           f"First error: {errors[0]['error'] if errors else '?'}")

    df = pd.DataFrame(rows)
    df = apply_filters(df, config.thresholds)
    df["composite_score"] = composite_score(df)
    df = df.sort_values(["pass", "composite_score"],
                        ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "rank_overall", range(1, len(df) + 1))

    # outputs
    front = ["rank_overall", "name", "pass", "composite_score",
             "plddt_binder", "pae_interaction", "iptm", "plddt_interface",
             "n_contacts", "binder_len", "fail_reasons"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "ranked.csv"), index=False)
    df[df["pass"]].to_csv(os.path.join(out_dir, "passing.csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(out_dir, "errors.csv"),
                                    index=False)

    screen_dashboard(df, os.path.join(out_dir, "dashboard.png"),
                     pae_max=config.thresholds.pae_interaction_max,
                     plddt_min=config.thresholds.plddt_binder_min)

    _write_summary(df, errors, config, os.path.join(out_dir, "summary.md"))
    return df


def _md_table(frame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no deps)."""
    cols = list(frame.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                v = f"{v:.2f}" if v == v else ""  # NaN -> blank
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def _write_summary(df, errors, config, path):
    thr = config.thresholds
    n = len(df); npass = int(df["pass"].sum())
    lines = []
    lines.append("# Mpro binder screening report\n")
    lines.append(f"- Designs scored: **{n}**")
    lines.append(f"- Passing all filters: **{npass}** ({100*npass/n:.0f}%)")
    if errors:
        lines.append(f"- Failed to score: {len(errors)} (see errors.csv)")
    lines.append("\n## Acceptance thresholds\n")
    lines.append(f"- binder pLDDT ≥ {thr.plddt_binder_min}")
    lines.append(f"- pAE interaction ≤ {thr.pae_interaction_max} Å")
    lines.append(f"- ipTM ≥ {thr.iptm_min} (if available)")
    lines.append(f"- interface pLDDT ≥ {thr.plddt_interface_min}")
    lines.append(f"- interface contacts ≥ {thr.min_contacts}")
    lines.append("\n## Top designs\n")
    top = df.head(10)
    show = ["rank_overall", "name", "pass", "composite_score",
            "plddt_binder", "pae_interaction", "iptm", "n_contacts"]
    show = [c for c in show if c in top.columns]
    lines.append(_md_table(top[show]))
    if npass:
        lines.append("\n## Most common failure reasons\n")
    from collections import Counter
    c = Counter()
    for r in df.loc[~df["pass"], "fail_reasons"]:
        for tok in str(r).split(";"):
            if tok:
                c[tok] += 1
    for reason, cnt in c.most_common():
        lines.append(f"- `{reason}`: {cnt} designs")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
