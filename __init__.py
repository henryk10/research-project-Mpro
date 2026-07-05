"""mpro_screen — reusable screening pipeline for RFdiffusion/ProteinMPNN/ColabFold
binder designs against SARS-CoV-2 main protease (or any two-chain complex).

Quick start
-----------
    from mpro_screen import run_screen, ScreenConfig, Thresholds
    df = run_screen("colabfold_out/", "screen_results/")

Or from the command line:
    python -m mpro_screen colabfold_out/ -o screen_results/
"""
from .pipeline import run_screen, ScreenConfig
from .filters import Thresholds, apply_filters, composite_score
from .metrics import score_complex, DesignMetrics
from .discover import find_predictions

__version__ = "0.1.0"
__all__ = [
    "run_screen", "ScreenConfig", "Thresholds",
    "apply_filters", "composite_score",
    "score_complex", "DesignMetrics", "find_predictions",
]
