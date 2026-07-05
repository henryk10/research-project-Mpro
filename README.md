# mpro_screen

A reusable pipeline for the **screening** step of a binder-design campaign:
take a folder of ColabFold predictions (RFdiffusion → ProteinMPNN → ColabFold),
compute the interface and confidence metrics that separate plausible binders
from junk, apply literature-standard acceptance filters, rank the survivors,
and emit CSVs + a diagnostic dashboard + a summary report.

Built for **SARS-CoV-2 main protease (Mpro)** binder screening but works on any
two-chain (target + binder) complex.

## Why

The generate step (RFdiffusion/ProteinMPNN/ColabFold) needs GPUs, but the
*screening* step — parsing every predicted complex, measuring the interface,
deciding pass/fail, and ranking — is CPU work you were doing by hand for each
batch. This packages that judgment into one command so every batch is scored
the same way.

## Install / requirements

Pure Python — **numpy, pandas, matplotlib only**. No Biopython, no PyRosetta,
no compiled deps. PDB parsing and Kabsch RMSD are built in, so it runs on a
laptop, a Colab cell, or a cluster login node with nothing to install.

```bash
pip install numpy pandas matplotlib
```

## Usage

Command line:

```bash
python -m mpro_screen path/to/colabfold_out/ -o screen_results/
```

Python API:

```python
from mpro_screen import run_screen, ScreenConfig, Thresholds

df = run_screen(
    "colabfold_out/", "screen_results/",
    ScreenConfig(
        target_chain="A", binder_chain="B",
        design_dir="rfdiffusion_backbones/",   # optional: enables RMSD-to-design
        thresholds=Thresholds(plddt_binder_min=80, pae_interaction_max=10),
    ),
)
df.head()
```

### Inputs

Point it at the directory ColabFold wrote. It walks recursively, pairs each
`*.pdb` with its `*_scores_*.json` (falls back to the B-factor column for
pLDDT if no JSON is found), and by default keeps only the **rank-1** model per
design (`--all-ranks` to score every model).

### Outputs (in `-o` dir)

| file | contents |
|------|----------|
| `metrics.csv` / `ranked.csv` | every design, all metrics, sorted best-first |
| `passing.csv` | only designs passing all filters |
| `dashboard.png` | 4-panel screening dashboard |
| `summary.md` | pass rate, thresholds, top-10 table, failure breakdown |
| `errors.csv` | any predictions that failed to parse (only if some did) |

## Metrics computed

Per design (chain B = binder, chain A = target by convention):

- **plddt_binder** — mean pLDDT over binder residues (fold confidence)
- **plddt_interface** — mean pLDDT over interface residues
- **pae_interaction** — mean cross-chain predicted aligned error (Å); the
  single best AF2 discriminator of a real vs. spurious interface
- **iptm** — interface pTM (when present in the scores JSON)
- **n_interface_binder / n_interface_target / n_contacts** — interface size
  (heavy-atom contacts within 5 Å)
- **binder_rg** — binder radius of gyration (compactness sanity check)
- **rmsd_to_design** — CA RMSD of the predicted binder to its RFdiffusion
  backbone (only if `design_dir` is given) — how far ColabFold moved from the
  intended design

## Acceptance filters (defaults)

Following the AF2/ColabFold binder-screening convention (Bennett et al. 2023
and the RFdiffusion binder pipelines):

| metric | threshold |
|--------|-----------|
| binder pLDDT | ≥ 80 |
| pAE interaction | ≤ 10 Å |
| ipTM | ≥ 0.50 (if available) |
| interface pLDDT | ≥ 80 |
| interface contacts | ≥ 3 |

All editable via `Thresholds(...)` or CLI flags. A design must clear every
active filter to be marked `pass`. `fail_reasons` records which it missed.

The **composite_score** (0–100) is a batch-relative blend of these metrics for
*ranking* — use the filters for the absolute quality bar, the score to order
what survives.

## Testing

```bash
python tests/test_pipeline.py          # or: python -m pytest -q
```

`make_fixtures.py` writes 12 synthetic ColabFold-style designs (good / marginal
/ non-contacting / low-confidence) — a self-contained way to see what passing
vs. failing looks like without real GPU output.

## Note

Metrics are computed from geometry and AF2/ColabFold confidence — they measure
*structural plausibility of the predicted complex*, not experimental binding.
That is exactly the filter this stage is for: cut the designs not worth
ordering. Wet-lab validation is the next step, not something these numbers
replace.
