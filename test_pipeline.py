"""Smoke + correctness tests. Run: python -m pytest -q   (or python tests/test_pipeline.py)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mpro_screen.metrics import kabsch_rmsd, radius_of_gyration
from mpro_screen.pipeline import run_screen, ScreenConfig
from mpro_screen.discover import find_predictions
import make_fixtures


def test_kabsch_identity():
    P = np.random.default_rng(0).normal(size=(20, 3))
    assert kabsch_rmsd(P, P.copy()) < 1e-9
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert kabsch_rmsd(P, P @ R.T) < 1e-9  # rotation-invariant


def test_rg_positive():
    coords = np.random.default_rng(1).normal(size=(30, 3)) * 5
    assert radius_of_gyration(coords) > 0


def test_end_to_end(tmp_path=None):
    d = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    fix = os.path.join(d, "cf")
    make_fixtures.main(fix)
    preds = find_predictions(fix)
    assert len(preds) == 12
    assert all(p.scores is not None for p in preds)

    out = os.path.join(d, "out")
    df = run_screen(fix, out, ScreenConfig())
    assert len(df) == 12
    assert df["pass"].sum() >= 3
    assert df["composite_score"].notna().all()
    zero = df[df["n_contacts"] == 0]
    assert len(zero) >= 1 and not zero["pass"].any()
    assert df.iloc[0]["pass"]
    for f in ("metrics.csv", "ranked.csv", "passing.csv",
              "dashboard.png", "summary.md"):
        assert os.path.exists(os.path.join(out, f)), f
    print("end-to-end OK:", int(df["pass"].sum()), "/", len(df), "pass")


if __name__ == "__main__":
    test_kabsch_identity()
    test_rg_positive()
    test_end_to_end()
    print("All tests passed.")
