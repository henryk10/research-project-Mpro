"""Locate ColabFold prediction outputs and pair each PDB with its scores JSON.

ColabFold typical naming:
    <name>_unrelaxed_rank_001_alphafold2_multimer_v3_model_3_seed_000.pdb
    <name>_scores_rank_001_alphafold2_multimer_v3_model_3_seed_000.json
    <name>_predicted_aligned_error_v1.json           (older)
We match a PDB to the scores JSON that shares the most of its filename stem.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Prediction:
    name: str          # design name (before _unrelaxed/_relaxed/_rank)
    pdb: str
    scores: Optional[str]
    rank: Optional[int]


_RANK_RE = re.compile(r"rank[_-]?(\d+)")


def _rank_of(fname: str) -> Optional[int]:
    m = _RANK_RE.search(fname)
    return int(m.group(1)) if m else None


def _design_name(fname: str) -> str:
    base = os.path.basename(fname)
    base = re.sub(r"\.(pdb|cif|json)$", "", base)
    # cut at the first ColabFold suffix marker
    for marker in ("_unrelaxed", "_relaxed", "_scores",
                   "_predicted_aligned_error", "_rank"):
        i = base.find(marker)
        if i != -1:
            base = base[:i]
            break
    return base


def _tokens(fname: str) -> set:
    base = re.sub(r"\.(pdb|cif|json)$", "", os.path.basename(fname))
    return set(re.split(r"[_\-.]", base))


def find_predictions(root: str,
                     best_rank_only: bool = True) -> List[Prediction]:
    """Walk `root`, return paired predictions.

    If `best_rank_only`, keep only the top-ranked model per design (rank 1,
    or the single model if unranked).
    """
    pdbs = []
    for pat in ("**/*.pdb", "**/*.cif"):
        pdbs.extend(glob.glob(os.path.join(root, pat), recursive=True))
    jsons = [j for j in glob.glob(os.path.join(root, "**/*.json"), recursive=True)
             if ("scores" in os.path.basename(j).lower()
                 or "predicted_aligned_error" in os.path.basename(j).lower()
                 or "pae" in os.path.basename(j).lower())]

    preds: List[Prediction] = []
    for pdb in sorted(pdbs):
        # skip obvious non-prediction files
        low = os.path.basename(pdb).lower()
        if low.startswith("relaxed_") and any(p.endswith(low[8:]) for p in pdbs):
            pass
        ptoks = _tokens(pdb)
        best_j, best_overlap = None, -1
        pdir = os.path.dirname(pdb)
        for j in jsons:
            if os.path.dirname(j) != pdir:
                continue
            ov = len(ptoks & _tokens(j))
            if ov > best_overlap:
                best_j, best_overlap = j, ov
        preds.append(Prediction(
            name=_design_name(pdb), pdb=pdb, scores=best_j, rank=_rank_of(pdb)))

    if best_rank_only:
        by_name = {}
        for p in preds:
            key = p.name
            cur = by_name.get(key)
            r = p.rank if p.rank is not None else 1
            if cur is None or r < (cur.rank if cur.rank is not None else 1):
                by_name[key] = p
        preds = list(by_name.values())

    return sorted(preds, key=lambda p: p.name)
