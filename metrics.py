"""Interface and confidence metrics for two-chain binder:target complexes.

Designed for ColabFold outputs but works on any PDB + optional scores JSON.
By convention chain B (second chain) is the designed binder and chain A is
the target (SARS-CoV-2 Mpro). This can be overridden per call.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pdbio import Structure, Residue, read_pdb


# ----------------------------- geometry helpers -----------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between paired point sets after optimal superposition."""
    if P.shape != Q.shape or P.shape[0] == 0:
        return float("nan")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    Pr = Pc @ R
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def radius_of_gyration(coords: np.ndarray) -> float:
    if coords.shape[0] == 0:
        return float("nan")
    c = coords - coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(c ** 2, axis=1))))


# ----------------------------- scores JSON ----------------------------------

@dataclass
class Scores:
    plddt: Optional[np.ndarray] = None   # per-residue, file order
    pae: Optional[np.ndarray] = None     # (N,N)
    ptm: Optional[float] = None
    iptm: Optional[float] = None


def load_scores_json(path: str) -> Scores:
    """Read a ColabFold scores/PAE JSON. Tolerant to the several key spellings."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        data = data[0]
    plddt = data.get("plddt")
    pae = (data.get("pae") or data.get("predicted_aligned_error")
           or data.get("PAE"))
    ptm = data.get("ptm", data.get("pTM"))
    iptm = data.get("iptm", data.get("ipTM"))
    return Scores(
        plddt=np.asarray(plddt, float) if plddt is not None else None,
        pae=np.asarray(pae, float) if pae is not None else None,
        ptm=float(ptm) if ptm is not None else None,
        iptm=float(iptm) if iptm is not None else None,
    )


# ----------------------------- interface -------------------------------------

def _min_heavy_dist(a: Residue, b: Residue) -> float:
    A = a.heavy_atoms(); B = b.heavy_atoms()
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("inf")
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return float(d.min())


def interface_residues(struct: Structure, target: str, binder: str,
                       cutoff: float = 5.0) -> Tuple[List[int], List[int]]:
    """Return file-order indices of interface residues on (target, binder).

    A residue is 'interface' if any heavy atom is within `cutoff` A of any
    heavy atom of the partner chain. Uses a CA prefilter for speed.
    """
    order = struct.residue_order()
    idx_of = {id(r): i for i, r in enumerate(order)}
    tgt = struct.chain_residues(target)
    bnd = struct.chain_residues(binder)
    if not tgt or not bnd:
        return [], []
    tca = np.vstack([r.ca if r.ca is not None else np.full(3, 1e6) for r in tgt])
    bca = np.vstack([r.ca if r.ca is not None else np.full(3, 1e6) for r in bnd])
    ca_d = np.linalg.norm(tca[:, None, :] - bca[None, :, :], axis=2)
    near = ca_d < (cutoff + 12.0)  # CA-CA generous prefilter
    t_hit, b_hit = set(), set()
    ti, bi = np.where(near)
    for i, j in zip(ti, bi):
        if _min_heavy_dist(tgt[i], bnd[j]) <= cutoff:
            t_hit.add(idx_of[id(tgt[i])])
            b_hit.add(idx_of[id(bnd[j])])
    return sorted(t_hit), sorted(b_hit)


def interface_contacts(struct: Structure, target: str, binder: str,
                       cutoff: float = 5.0) -> int:
    """Count binder-target residue pairs in contact (heavy-atom < cutoff)."""
    tgt = struct.chain_residues(target)
    bnd = struct.chain_residues(binder)
    tca = np.vstack([r.ca if r.ca is not None else np.full(3, 1e6) for r in tgt])
    bca = np.vstack([r.ca if r.ca is not None else np.full(3, 1e6) for r in bnd])
    ca_d = np.linalg.norm(tca[:, None, :] - bca[None, :, :], axis=2)
    ti, bi = np.where(ca_d < (cutoff + 12.0))
    n = 0
    for i, j in zip(ti, bi):
        if _min_heavy_dist(tgt[i], bnd[j]) <= cutoff:
            n += 1
    return n


# ----------------------------- top-level scoring -----------------------------

@dataclass
class DesignMetrics:
    name: str
    binder_len: int
    binder_seq: str
    # confidence
    plddt_mean: float
    plddt_binder: float
    plddt_interface: float
    ptm: Optional[float]
    iptm: Optional[float]
    # interface
    n_interface_binder: int
    n_interface_target: int
    n_contacts: int
    pae_interaction: float          # mean cross-chain PAE (symmetrised)
    pae_int_binder_to_target: float
    # geometry
    binder_rg: float
    rmsd_to_design: Optional[float] = None

    def as_row(self) -> Dict:
        return asdict(self)


def score_complex(pdb_path: str,
                  scores_path: Optional[str] = None,
                  target_chain: str = "A",
                  binder_chain: str = "B",
                  design_pdb: Optional[str] = None,
                  interface_cutoff: float = 5.0,
                  name: Optional[str] = None) -> DesignMetrics:
    struct = read_pdb(pdb_path)
    chains = struct.chains()
    # If the requested chains aren't present, fall back to first two by order,
    # treating the SHORTER chain as the binder.
    if target_chain not in chains or binder_chain not in chains:
        if len(chains) >= 2:
            c0, c1 = chains[0], chains[1]
            if struct.chain_length(c1) <= struct.chain_length(c0):
                target_chain, binder_chain = c0, c1
            else:
                target_chain, binder_chain = c1, c0
        else:
            raise ValueError(f"{pdb_path}: need 2 chains, found {chains}")

    order = struct.residue_order()
    binder_res = struct.chain_residues(binder_chain)
    binder_seq = struct.sequence(binder_chain)

    t_iface, b_iface = interface_residues(struct, target_chain, binder_chain,
                                          interface_cutoff)
    n_contacts = interface_contacts(struct, target_chain, binder_chain,
                                    interface_cutoff)

    # --- pLDDT: prefer scores JSON, else B-factor column ---
    scores = load_scores_json(scores_path) if scores_path else Scores()
    if scores.plddt is not None and len(scores.plddt) == len(order):
        plddt = scores.plddt
    else:
        plddt = np.array([r.ca_bfactor if r.ca_bfactor is not None else np.nan
                          for r in order])
    binder_idx = [i for i, r in enumerate(order) if r.chain == binder_chain]
    plddt_mean = float(np.nanmean(plddt))
    plddt_binder = float(np.nanmean(plddt[binder_idx])) if binder_idx else float("nan")
    iface_all = t_iface + b_iface
    plddt_interface = (float(np.nanmean(plddt[iface_all]))
                       if iface_all else float("nan"))

    # --- PAE interaction ---
    pae_interaction = float("nan")
    pae_b2t = float("nan")
    if scores.pae is not None and scores.pae.shape[0] == len(order):
        tgt_idx = [i for i, r in enumerate(order) if r.chain == target_chain]
        pae = scores.pae
        if binder_idx and tgt_idx:
            block_bt = pae[np.ix_(binder_idx, tgt_idx)]
            block_tb = pae[np.ix_(tgt_idx, binder_idx)]
            pae_interaction = float((block_bt.mean() + block_tb.mean()) / 2)
            pae_b2t = float(block_bt.mean())

    # --- geometry ---
    binder_rg = radius_of_gyration(struct.ca_coords(binder_chain))

    # --- RMSD to design backbone (optional) ---
    rmsd = None
    if design_pdb:
        try:
            dstruct = read_pdb(design_pdb)
            dchain = _match_binder_chain(dstruct, len(binder_res))
            P = struct.ca_coords(binder_chain)
            Q = dstruct.ca_coords(dchain)
            n = min(len(P), len(Q))
            if n > 3:
                rmsd = kabsch_rmsd(P[:n], Q[:n])
        except Exception:
            rmsd = None

    return DesignMetrics(
        name=name or _stem(pdb_path),
        binder_len=len(binder_res),
        binder_seq=binder_seq,
        plddt_mean=round(plddt_mean, 3),
        plddt_binder=round(plddt_binder, 3),
        plddt_interface=round(plddt_interface, 3),
        ptm=scores.ptm,
        iptm=scores.iptm,
        n_interface_binder=len(b_iface),
        n_interface_target=len(t_iface),
        n_contacts=n_contacts,
        pae_interaction=round(pae_interaction, 3),
        pae_int_binder_to_target=round(pae_b2t, 3),
        binder_rg=round(binder_rg, 3),
        rmsd_to_design=round(rmsd, 3) if rmsd is not None else None,
    )


def _match_binder_chain(struct: Structure, length_hint: int) -> str:
    best, bestdiff = struct.chains()[0], 1e9
    for c in struct.chains():
        diff = abs(struct.chain_length(c) - length_hint)
        if diff < bestdiff:
            best, bestdiff = c, diff
    return best


def _stem(path: str) -> str:
    import os
    b = os.path.basename(path)
    for ext in (".pdb", ".cif"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b
