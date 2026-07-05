"""Command-line interface: python -m mpro_screen <pred_dir> -o <out_dir>"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run_screen, ScreenConfig
from .filters import Thresholds


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mpro_screen",
        description="Screen ColabFold binder predictions against a target.")
    ap.add_argument("pred_dir", help="directory of ColabFold outputs")
    ap.add_argument("-o", "--out", default="screen_results",
                    help="output directory (default: screen_results)")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--interface-cutoff", type=float, default=5.0)
    ap.add_argument("--design-dir", default=None,
                    help="RFdiffusion backbone dir for RMSD-to-design")
    ap.add_argument("--all-ranks", action="store_true",
                    help="score every ranked model, not just rank 1")
    ap.add_argument("--plddt-binder-min", type=float, default=80.0)
    ap.add_argument("--pae-interaction-max", type=float, default=10.0)
    ap.add_argument("--iptm-min", type=float, default=0.50)
    ap.add_argument("--min-contacts", type=int, default=3)
    args = ap.parse_args(argv)

    cfg = ScreenConfig(
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        interface_cutoff=args.interface_cutoff,
        best_rank_only=not args.all_ranks,
        design_dir=args.design_dir,
        thresholds=Thresholds(
            plddt_binder_min=args.plddt_binder_min,
            pae_interaction_max=args.pae_interaction_max,
            iptm_min=args.iptm_min,
            min_contacts=args.min_contacts,
        ),
    )
    df = run_screen(args.pred_dir, args.out, cfg)
    npass = int(df["pass"].sum())
    print(f"Scored {len(df)} designs; {npass} pass. "
          f"Results in {args.out}/ (see summary.md, dashboard.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
