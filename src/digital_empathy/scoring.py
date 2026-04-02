"""
scoring.py — Cognitive Friction Score calculator.

Algorithm overview
------------------
TRIBE v2 outputs z-scored BOLD activations per cortical vertex over time:
    preds shape: (T, 20_484)   T = number of 1-second TR windows

We reduce this tensor to a single 1-10 "Friction Score" as follows:

  1. Temporal mean pooling
     Compute the mean activation per vertex across time:
       mean_activation = preds.mean(axis=0)  →  shape (20_484,)

  2. ROI extraction
     Apply the PFC and visual cortex masks from brain_regions.py:
       pfc_mean   = mean_activation[pfc_mask].mean()
       visual_mean = mean_activation[visual_mask].mean()

  3. Cognitive Load Ratio (CLR)
     CLR = pfc_mean / (visual_mean + ε)
     where ε = 1e-6 prevents division by zero.

     Interpretation:
       - High PFC + low visual  → high friction (user is confused / working hard)
       - Low PFC + high visual  → low friction (user is passively watching)
       - Balanced               → moderate friction (normal engaged browsing)

  4. Normalization to 1-10
     CLR is an unbounded ratio; we map it to [1, 10] using a logistic function:
       score = 1 + 9 * sigmoid((CLR - CLR_MIDPOINT) * SLOPE)
     where CLR_MIDPOINT and SLOPE are empirically chosen constants.

     Alternatively, if we have a population of CLR values, we could use
     percentile-based scaling — that's a Phase 3 improvement.

Score interpretation table:
  1.0 – 2.5  : Effortless      — UI is intuitive; minimal cognitive overhead
  2.5 – 4.0  : Comfortable     — Slight attention required; well-designed flow
  4.0 – 5.5  : Moderate        — Noticeable effort; some friction points
  5.5 – 7.0  : Strained        — Confusing layout or interaction patterns
  7.0 – 8.5  : High Friction   — Users likely struggle; refactor recommended
  8.5 – 10.0 : Critical        — Severe cognitive overload; major UX issues
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .brain_regions import BrainMasks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants  (adjust as empirical data is collected)
# ---------------------------------------------------------------------------

# The CLR value that maps to a score of ~5.5 (midpoint of the scale)
# For z-scored fMRI data, a ratio near 1.0 is typical for passive viewing.
# UI interactions generally push this above 1.0; complex UIs toward 2-3.
_CLR_MIDPOINT: float = 1.5

# Controls steepness of the sigmoid mapping.
# Higher → sharper cliff; lower → more gradual spread.
_SLOPE: float = 2.5

# Small constant to avoid division by zero in the CLR calculation
_EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FrictionResult:
    """Full output from the friction scoring pipeline."""

    # Final 1-10 score
    score: float

    # Intermediate values for transparency
    pfc_mean_activation: float
    visual_mean_activation: float
    cognitive_load_ratio: float

    # Human-readable interpretation
    label: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "explanation": self.explanation,
            "debug": {
                "pfc_mean_activation": round(self.pfc_mean_activation, 4),
                "visual_mean_activation": round(self.visual_mean_activation, 4),
                "cognitive_load_ratio": round(self.cognitive_load_ratio, 4),
            },
        }


# ---------------------------------------------------------------------------
# Score band definitions
# ---------------------------------------------------------------------------

# Each entry: (upper_bound_exclusive, label, explanation_template)
_SCORE_BANDS: list[tuple[float, str, str]] = [
    (
        2.5,
        "Effortless",
        "The UI feels highly intuitive. Neural activity shows minimal prefrontal "
        "engagement relative to visual cortex, indicating users process the "
        "interface with little cognitive effort. No refactoring needed.",
    ),
    (
        4.0,
        "Comfortable",
        "The UI requires slight attention but flows naturally. Prefrontal activation "
        "is moderate, suggesting users engage without significant mental effort. "
        "Minor polish may improve the experience further.",
    ),
    (
        5.5,
        "Moderate",
        "Noticeable cognitive friction detected. The prefrontal cortex is working "
        "meaningfully harder than the visual cortex. Consider reviewing navigation "
        "hierarchy, label clarity, or interaction affordances.",
    ),
    (
        7.0,
        "Strained",
        "The UI is creating measurable strain. High prefrontal activation suggests "
        "users are holding multiple mental models simultaneously. Audit for "
        "confusing layouts, ambiguous CTAs, or information overload.",
    ),
    (
        8.5,
        "High Friction",
        "Users are likely struggling with this interface. Neural predictions show "
        "strong cognitive overload signals. Significant UX refactoring is "
        "recommended — focus on reducing decision points and clarifying visual hierarchy.",
    ),
    (
        11.0,  # catch-all upper bound
        "Critical",
        "Severe cognitive overload detected. The interface is highly confusing. "
        "Predicted brain activity shows peak prefrontal strain. This UI needs a "
        "fundamental redesign — consider usability testing and a UX audit.",
    ),
]


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_friction_score(
    activations: np.ndarray,
    masks: BrainMasks,
) -> FrictionResult:
    """Compute the Cognitive Friction Score from raw TRIBE v2 predictions.

    Parameters
    ----------
    activations:
        np.ndarray of shape (T, 20_484) — z-scored BOLD predictions from TRIBE.
        T = number of TR timesteps; 20 484 = fsaverage5 cortical vertices.
    masks:
        BrainMasks from brain_regions.load_brain_masks().

    Returns
    -------
    FrictionResult
        Contains the 1-10 score, label, explanation, and intermediate values.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be 2-D (T, V), got shape {activations.shape}"
        )
    if activations.shape[1] != 20_484:
        raise ValueError(
            f"Expected 20 484 vertices (fsaverage5), got {activations.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Step 1 — Temporal mean pooling
    # ------------------------------------------------------------------
    mean_activation = activations.mean(axis=0)  # shape (20_484,)

    # ------------------------------------------------------------------
    # Step 2 — ROI extraction
    # ------------------------------------------------------------------
    pfc_activation = mean_activation[masks.pfc]
    visual_activation = mean_activation[masks.visual]

    pfc_mean = float(pfc_activation.mean()) if len(pfc_activation) > 0 else 0.0
    visual_mean = float(visual_activation.mean()) if len(visual_activation) > 0 else 0.0

    logger.debug(
        "ROI means — PFC: %.4f  Visual: %.4f",
        pfc_mean,
        visual_mean,
    )

    # ------------------------------------------------------------------
    # Step 3 — Cognitive Load Ratio
    # ------------------------------------------------------------------
    # We use absolute PFC mean (BOLD z-scores can be negative for deactivation).
    # Elevated PFC and suppressed visual → high ratio → high friction.
    clr = pfc_mean / (abs(visual_mean) + _EPSILON)

    logger.debug("Cognitive Load Ratio (CLR): %.4f", clr)

    # ------------------------------------------------------------------
    # Step 4 — Sigmoid normalization to [1, 10]
    # ------------------------------------------------------------------
    score = _sigmoid_scale(clr)

    logger.info("Friction score: %.2f  (CLR=%.4f)", score, clr)

    # ------------------------------------------------------------------
    # Step 5 — Band lookup for human-readable output
    # ------------------------------------------------------------------
    label, explanation = _lookup_band(score)

    return FrictionResult(
        score=score,
        pfc_mean_activation=pfc_mean,
        visual_mean_activation=visual_mean,
        cognitive_load_ratio=clr,
        label=label,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid_scale(clr: float) -> float:
    """Map an unbounded CLR value to a 1-10 score via logistic function.

    score = 1 + 9 * sigmoid((clr - midpoint) * slope)

    At clr == _CLR_MIDPOINT the score is exactly 5.5.
    The slope controls how sharply scores change around the midpoint.
    """
    x = (clr - _CLR_MIDPOINT) * _SLOPE
    # Standard sigmoid: 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + math.exp(-x))
    score = 1.0 + 9.0 * sig
    return round(float(np.clip(score, 1.0, 10.0)), 2)


def _lookup_band(score: float) -> tuple[str, str]:
    """Return (label, explanation) for the given score."""
    for upper_bound, label, explanation in _SCORE_BANDS:
        if score < upper_bound:
            return label, explanation
    # Fallback (should never happen after clip)
    return _SCORE_BANDS[-1][1], _SCORE_BANDS[-1][2]
