#!/usr/bin/env python3
"""
demo_synthetic.py — End-to-end pipeline dry-run without real model weights.

This script exercises every component of the Digital Empathy pipeline using
synthetic activations, so you can verify the full stack (scoring + heatmap
rendering) before committing to the ~30-second TRIBE v2 model load.

It produces:
  1. A Cognitive Friction Score + explanation for each simulated scenario
  2. A brain surface heatmap PNG saved to ./output/

Usage:
    python scripts/demo_synthetic.py

Expected output (approximate):
    [SCENARIO] Effortless UI
      Score      : 2.41 / 10
      Label      : Effortless
      CLR        : 0.53
      Heatmap    : /path/to/output/demo_effortless_brain_heatmap.png

    [SCENARIO] Confusing UI
      Score      : 8.14 / 10
      Label      : High Friction
      CLR        : 3.12
      Heatmap    : /path/to/output/demo_confusing_brain_heatmap.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the src/ directory is on the path when running the script directly
# (not needed when installed via pip install -e .)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from digital_empathy.brain_regions import BrainMasks
from digital_empathy.scoring import compute_friction_score
from digital_empathy.visualization import render_brain_heatmap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_VERTICES:  int = 20_484
N_TIMESTEPS: int = 30       # ~30 second recording
OUTPUT_DIR:  Path = _repo_root / "output"
RNG = np.random.default_rng(seed=7)

# Synthetic PFC + visual masks (positional proxies — no nilearn download needed)
_pfc_mask   = np.zeros(N_VERTICES, dtype=bool)
_pfc_mask[:1_400]         = True   # left  PFC proxy
_pfc_mask[10_242:11_642]  = True   # right PFC proxy

_visual_mask = np.zeros(N_VERTICES, dtype=bool)
_visual_mask[9_242:10_242]  = True  # left  visual proxy
_visual_mask[19_484:20_484] = True  # right visual proxy

SYNTHETIC_MASKS = BrainMasks(
    pfc=_pfc_mask,
    visual=_visual_mask,
    pfc_labels=["frontal_proxy (synthetic)"],
    visual_labels=["occipital_proxy (synthetic)"],
)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _make_effortless_activations() -> np.ndarray:
    """Low and flat — like a user breezing through a well-designed onboarding."""
    acts = RNG.normal(0.0, 0.15, (N_TIMESTEPS, N_VERTICES)).astype(np.float32)
    # Strong visual response (engaged perception)
    acts[:, 9_242:10_242]  += 1.8
    acts[:, 19_484:20_484] += 1.8
    # Minimal PFC effort
    acts[:, :1_400]         -= 0.3
    acts[:, 10_242:11_642]  -= 0.3
    return acts


def _make_moderate_activations() -> np.ndarray:
    """Balanced — like a user navigating a standard dashboard."""
    acts = RNG.normal(0.0, 0.2, (N_TIMESTEPS, N_VERTICES)).astype(np.float32)
    acts[:, :1_400]         += 0.8
    acts[:, 10_242:11_642]  += 0.8
    acts[:, 9_242:10_242]   += 0.9
    acts[:, 19_484:20_484]  += 0.9
    return acts


def _make_confusing_activations() -> np.ndarray:
    """High PFC, suppressed visual — like a user lost in a multi-step wizard."""
    acts = RNG.normal(0.0, 0.25, (N_TIMESTEPS, N_VERTICES)).astype(np.float32)
    acts[:, :1_400]         += 2.8   # heavy prefrontal load
    acts[:, 10_242:11_642]  += 2.8
    acts[:, 9_242:10_242]   -= 0.8   # visual suppression (not just looking around)
    acts[:, 19_484:20_484]  -= 0.8
    return acts


SCENARIOS: list[tuple[str, np.ndarray]] = [
    ("demo_effortless", _make_effortless_activations()),
    ("demo_moderate",   _make_moderate_activations()),
    ("demo_confusing",  _make_confusing_activations()),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Digital Empathy — Synthetic Pipeline Demo")
    print("=" * 60)
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Activation shape : ({N_TIMESTEPS}, {N_VERTICES})")
    print(f"  PFC vertices     : {SYNTHETIC_MASKS.n_pfc_vertices}")
    print(f"  Visual vertices  : {SYNTHETIC_MASKS.n_visual_vertices}")
    print("=" * 60 + "\n")

    for scenario_name, activations in SCENARIOS:
        print(f"[SCENARIO] {scenario_name.replace('demo_', '').replace('_', ' ').title()}")

        # 1 — Score
        result = compute_friction_score(activations, SYNTHETIC_MASKS)

        # 2 — Heatmap (uses nilearn; downloads fsaverage5 mesh on first run)
        try:
            heatmap_path = render_brain_heatmap(
                activations=activations,
                video_path=f"/synthetic/{scenario_name}.mp4",
                output_dir=OUTPUT_DIR,
                pfc_mask=SYNTHETIC_MASKS.pfc,
            )
        except Exception as exc:
            heatmap_path = f"[render failed: {exc}]"

        # 3 — Print results
        bar_filled = int(result.score - 1) * "█"
        bar_empty  = (9 - int(result.score - 1)) * "░"
        score_bar  = f"[{bar_filled}{bar_empty}]"

        print(f"  Score      : {result.score:>5.2f} / 10  {score_bar}")
        print(f"  Label      : {result.label}")
        print(f"  CLR        : {result.cognitive_load_ratio:.4f}")
        print(f"  PFC mean   : {result.pfc_mean_activation:+.4f}")
        print(f"  Visual mean: {result.visual_mean_activation:+.4f}")
        print(f"  Heatmap    : {heatmap_path}")
        print(f"\n  → {result.explanation}\n")
        print("-" * 60 + "\n")

    print("Demo complete.  Open the PNG files in ./output/ to inspect brain activation.\n")


if __name__ == "__main__":
    main()
