"""Minimal, dependency-free PDB/mmCIF-lite reader for ColabFold outputs.

Only ATOM/HETATM records are parsed. This is deliberately small so the
pipeline has no hard dependency on Biopython/biotite and runs anywhere
numpy is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Standard amino-acid three-to-one map (plus common variants).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, np.ndarray] = field(default_factory=dict)  # atom_name -> xyz
    bfactors: Dict[str, float] = field(default_factory=dict)    # atom_name -> B

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.resname, "X")

    @property
    def ca(self) -> Optional[np.ndarray]:
        return self.atoms.get("CA")

    @property
    def ca_bfactor(self) -> Optional[float]:
        # ColabFold writes per-residue pLDDT into the CA (all-atom) B-factor.
        if "CA" in self.bfactors:
            return self.bfactors["CA"]
        if self.bfactors:
            return float(np.mean(list(self.bfactors.values())))
        return None

    def heavy_atoms(self) -> np.ndarray:
        """(N,3) array of non-hydrogen atom coordinates."""
        coords = [xyz for name, xyz in self.atoms.items()
                  if not name.startswith("H") and not name[0].isdigit()]
        if not coords:
            return np.empty((0, 3))
        return np.vstack(coords)


@dataclass
class Structure:
    residues: List[Residue]
    source: str = ""

    def chains(self) -> List[str]:
        seen: List[str] = []
        for r in self.residues:
            if r.chain not in seen:
                seen.append(r.chain)
        return seen

    def chain_residues(self, chain: str) -> List[Residue]:
        return [r for r in self.residues if r.chain == chain]

    def chain_length(self, chain: str) -> int:
        return len(self.chain_residues(chain))

    def sequence(self, chain: str) -> str:
        return "".join(r.one_letter for r in self.chain_residues(chain))

    def ca_coords(self, chain: str) -> np.ndarray:
        cas = [r.ca for r in self.chain_residues(chain) if r.ca is not None]
        return np.vstack(cas) if cas else np.empty((0, 3))

    def ca_plddt(self, chain: str) -> np.ndarray:
        vals = [r.ca_bfactor for r in self.chain_residues(chain)
                if r.ca_bfactor is not None]
        return np.asarray(vals, dtype=float)

    def residue_order(self) -> List[Residue]:
        """Residues in file order — matches ColabFold's PAE/pLDDT indexing."""
        return list(self.residues)


def read_pdb(path: str) -> Structure:
    """Parse a PDB file into a Structure. First model only."""
    residues: List[Residue] = []
    index: Dict[Tuple[str, int, str], Residue] = {}
    with open(path, "r") as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE and rec == "HETATM":
                continue  # skip ligands/waters
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            atom_name = line[12:16].strip()
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            try:
                bfac = float(line[60:66])
            except ValueError:
                bfac = 0.0
            key = (chain, resseq, icode)
            res = index.get(key)
            if res is None:
                res = Residue(chain=chain, resseq=resseq, icode=icode, resname=resname)
                index[key] = res
                residues.append(res)
            res.atoms[atom_name] = np.array([x, y, z])
            res.bfactors[atom_name] = bfac
    return Structure(residues=residues, source=path)
