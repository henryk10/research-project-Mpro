"""Generate synthetic ColabFold-style outputs to exercise the pipeline.

Creates a target chain A (~150 aa stand-in for Mpro domain) and, for each
design, a binder chain B placed near or far from the target with pLDDT/PAE
tuned to be a 'good' or 'bad' binder. NOT real structures — just enough
geometry and score files to validate discovery/metrics/filters/plots.
"""
import json
import os
import numpy as np

rng = np.random.default_rng(7)

AA = list("ACDEFGHIKLMNPQRSTVWY")


def helix_coords(n, start, axis=np.array([0, 0, 1.0]), radius=2.3, rise=1.5):
    """Crude CA trace of an alpha helix."""
    pts = []
    for i in range(n):
        ang = i * 100 * np.pi / 180
        off = radius * np.array([np.cos(ang), np.sin(ang), 0]) + axis * rise * i
        pts.append(start + off)
    return np.array(pts)


def write_pdb(path, chains):
    """chains: list of (chain_id, coords(N,3), plddt(N,))"""
    with open(path, "w") as fh:
        serial = 1
        for cid, coords, plddt in chains:
            for i, (xyz, b) in enumerate(zip(coords, plddt), start=1):
                res = AA[(i * 7) % 20]
                three = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU",
                         "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
                         "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
                         "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
                         "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}[res]
                for aname, disp in (("N", [-1.2, 0, 0]), ("CA", [0, 0, 0]),
                                    ("C", [1.2, 0.3, 0]), ("O", [1.5, 1.4, 0])):
                    p = xyz + np.array(disp)
                    # Strict PDB fixed-column format (chainID at col 22).
                    fh.write(
                        f"ATOM  {serial:5d}  {aname:<3s} {three:>3s} {cid:1s}"
                        f"{i:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                        f"  1.00{b:6.2f}          {aname[0]:>2s}\n")
                    serial += 1
        fh.write("END\n")


def make_scores(path, plddt, n, iptm, pae_inter, tgt_len):
    """Write a ColabFold-style scores JSON with a block PAE matrix."""
    N = n
    pae = rng.uniform(2, 6, size=(N, N))  # intra-chain low error
    # cross-chain block gets pae_inter (+noise)
    pae[:tgt_len, tgt_len:] = rng.normal(pae_inter, 1.0, (tgt_len, N - tgt_len))
    pae[tgt_len:, :tgt_len] = rng.normal(pae_inter, 1.0, (N - tgt_len, tgt_len))
    pae = np.clip(pae, 0.2, 31.75)
    obj = {
        "plddt": [round(float(x), 2) for x in plddt],
        "pae": np.round(pae, 2).tolist(),
        "ptm": round(float(rng.uniform(0.6, 0.85)), 3),
        "iptm": round(float(iptm), 3),
    }
    with open(path, "w") as fh:
        json.dump(obj, fh)


def main(out="fixtures/colabfold_out"):
    os.makedirs(out, exist_ok=True)
    tgt_len = 150
    # Target: a compact bundle of short helices
    tgt = []
    base = np.array([0.0, 0, 0])
    for h in range(10):
        seg = helix_coords(15, base + np.array([h % 5 * 6.0, (h // 5) * 6.0, 0]))
        tgt.append(seg)
    tgt = np.vstack(tgt)[:tgt_len]

    specs = [
        # (name, quality, binder_len)
        ("design_0001", "good", 65),
        ("design_0002", "good", 58),
        ("design_0003", "good", 72),
        ("design_0004", "marginal", 60),
        ("design_0005", "marginal", 55),
        ("design_0006", "bad_far", 63),
        ("design_0007", "bad_lowconf", 68),
        ("design_0008", "bad_lowconf", 50),
        ("design_0009", "good", 61),
        ("design_0010", "bad_far", 57),
        ("design_0011", "marginal", 66),
        ("design_0012", "bad_lowconf", 59),
    ]

    for name, qual, blen in specs:
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        # target pLDDT high & stable
        tgt_plddt = rng.normal(88, 4, tgt_len).clip(50, 98)

        if qual == "good":
            # binder packed against target surface
            start = tgt.mean(0) + np.array([0, 0, -6.0])
            bcoords = helix_coords(blen, start, axis=np.array([0.2, 0.1, -1]))
            bcoords += rng.normal(0, 0.4, bcoords.shape)
            bplddt = rng.normal(87, 4, blen).clip(55, 97)
            iptm, pae_i = rng.uniform(0.62, 0.82), rng.uniform(4, 8)
        elif qual == "marginal":
            start = tgt.mean(0) + np.array([0, 0, -9.0])
            bcoords = helix_coords(blen, start, axis=np.array([0.3, 0.2, -1]))
            bcoords += rng.normal(0, 0.8, bcoords.shape)
            bplddt = rng.normal(80, 6, blen).clip(45, 95)
            iptm, pae_i = rng.uniform(0.45, 0.58), rng.uniform(9, 13)
        elif qual == "bad_far":
            start = tgt.mean(0) + np.array([40.0, 20, 30])  # not touching
            bcoords = helix_coords(blen, start)
            bplddt = rng.normal(84, 5, blen).clip(50, 96)
            iptm, pae_i = rng.uniform(0.15, 0.35), rng.uniform(20, 29)
        else:  # bad_lowconf
            start = tgt.mean(0) + np.array([0, 0, -6.0])
            bcoords = helix_coords(blen, start) + rng.normal(0, 2.5, (blen, 3))
            bplddt = rng.normal(58, 10, blen).clip(25, 85)
            iptm, pae_i = rng.uniform(0.3, 0.5), rng.uniform(12, 20)

        full_plddt = np.concatenate([tgt_plddt, bplddt])
        pdb = os.path.join(
            d, f"{name}_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")
        js = os.path.join(
            d, f"{name}_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json")
        write_pdb(pdb, [("A", tgt, tgt_plddt), ("B", bcoords, bplddt)])
        make_scores(js, full_plddt, tgt_len + blen, iptm, pae_i, tgt_len)

    print(f"Wrote {len(specs)} synthetic designs to {out}/")


if __name__ == "__main__":
    main()
