"""
brain_regions.py — fsaverage5 cortical vertex masks.

TRIBE v2 outputs predictions on the **fsaverage5** cortical surface mesh,
which has 20 484 vertices total (10 242 per hemisphere × 2 hemispheres).

This module uses nilearn's built-in Destrieux atlas (aparc.a2009s) parcellation
projected onto fsaverage5 to build boolean masks isolating two regions of
interest (ROIs):

  PFC  — Prefrontal Cortex  → cognitive load / working memory strain
  V1V2 — Visual Cortex      → baseline sensory processing (normalization term)

The friction score is computed as PFC activation / (V1V2 activation + ε),
so higher PFC relative to visual baseline → higher cognitive friction.

Vertex layout for fsaverage5 (nilearn convention):
  indices   0 .. 10 241  → LEFT  hemisphere
  indices 10 242 .. 20 483 → RIGHT hemisphere

Destrieux label strings used (partial match, case-insensitive):
  PFC  labels containing: "frontal", "orbital", "rectus", "cingulate_ant"
  V1V2 labels containing: "occipital", "calcarine", "cuneus", "lingual"

Reference: https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_surf_destrieux.html
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Total vertices in fsaverage5 (both hemispheres combined)
FSAVERAGE5_N_VERTICES: int = 20_484
# Vertices per hemisphere
HEMI_N_VERTICES: int = FSAVERAGE5_N_VERTICES // 2  # 10 242

# Destrieux label substrings that define each ROI (lowercase, partial match)
_PFC_LABEL_KEYWORDS: tuple[str, ...] = (
    "frontal",
    "orbital",
    "rectus",
    "cingulate_ant",  # anterior cingulate — involved in error/conflict monitoring
)

_VISUAL_LABEL_KEYWORDS: tuple[str, ...] = (
    "occipital",
    "calcarine",    # primary visual cortex (V1)
    "cuneus",       # V2/V3 medial
    "lingual",      # ventral visual stream
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class BrainMasks(NamedTuple):
    """Boolean vertex masks for our two ROIs across the full fsaverage5 mesh."""

    pfc: np.ndarray       # shape (20 484,) dtype bool
    visual: np.ndarray    # shape (20 484,) dtype bool

    # Human-readable label sets (for logging / debugging)
    pfc_labels: list[str]
    visual_labels: list[str]

    @property
    def n_pfc_vertices(self) -> int:
        return int(self.pfc.sum())

    @property
    def n_visual_vertices(self) -> int:
        return int(self.visual.sum())


# ---------------------------------------------------------------------------
# Atlas loader (cached — nilearn fetches once then hits disk)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_brain_masks() -> BrainMasks:
    """Fetch the Destrieux surface atlas and build PFC / visual cortex masks.

    Uses nilearn's `fetch_atlas_surf_destrieux()` which downloads the
    parcellation labels for fsaverage5 and caches them locally.

    Returns
    -------
    BrainMasks
        Two boolean numpy arrays, each of length 20 484, marking the vertices
        that belong to PFC or visual cortex respectively.

    Notes
    -----
    The Destrieux atlas is hemisphere-specific.  We fetch both hemispheres,
    offset the right-hemisphere indices by HEMI_N_VERTICES, and concatenate.
    """
    try:
        from nilearn import datasets as nl_datasets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for brain region masking.  "
            "Install it with:  pip install nilearn"
        ) from exc

    logger.info("Fetching Destrieux surface atlas (fsaverage5) …")
    destrieux = nl_datasets.fetch_atlas_surf_destrieux()
    # destrieux.labels : list[bytes] — 149 region label strings
    # destrieux.map_left  : np.ndarray[int32] shape (10 242,) — per-vertex label index
    # destrieux.map_right : np.ndarray[int32] shape (10 242,)

    # Decode label bytes → str, lowercase for matching
    labels: list[str] = [
        lbl.decode("utf-8").lower() if isinstance(lbl, bytes) else lbl.lower()
        for lbl in destrieux.labels
    ]
    logger.debug("Destrieux labels (%d): %s", len(labels), labels[:10])

    left_map: np.ndarray = destrieux.map_left.astype(int)    # (10 242,)
    right_map: np.ndarray = destrieux.map_right.astype(int)  # (10 242,)

    # ------------------------------------------------------------------
    # Build PFC mask
    # ------------------------------------------------------------------
    pfc_label_indices = _find_label_indices(labels, _PFC_LABEL_KEYWORDS)
    pfc_labels_matched = [labels[i] for i in pfc_label_indices]
    logger.info(
        "PFC labels matched (%d): %s",
        len(pfc_labels_matched),
        pfc_labels_matched,
    )

    pfc_left = np.isin(left_map, pfc_label_indices)     # (10 242,) bool
    pfc_right = np.isin(right_map, pfc_label_indices)   # (10 242,) bool
    pfc_full = np.concatenate([pfc_left, pfc_right])    # (20 484,) bool

    # ------------------------------------------------------------------
    # Build visual cortex mask
    # ------------------------------------------------------------------
    visual_label_indices = _find_label_indices(labels, _VISUAL_LABEL_KEYWORDS)
    visual_labels_matched = [labels[i] for i in visual_label_indices]
    logger.info(
        "Visual cortex labels matched (%d): %s",
        len(visual_labels_matched),
        visual_labels_matched,
    )

    visual_left = np.isin(left_map, visual_label_indices)
    visual_right = np.isin(right_map, visual_label_indices)
    visual_full = np.concatenate([visual_left, visual_right])  # (20 484,)

    masks = BrainMasks(
        pfc=pfc_full,
        visual=visual_full,
        pfc_labels=pfc_labels_matched,
        visual_labels=visual_labels_matched,
    )

    logger.info(
        "Brain masks ready — PFC: %d vertices, Visual: %d vertices",
        masks.n_pfc_vertices,
        masks.n_visual_vertices,
    )
    return masks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_label_indices(
    labels: list[str],
    keywords: tuple[str, ...],
) -> list[int]:
    """Return label indices whose names contain any of the given keywords."""
    matched: list[int] = []
    for idx, label in enumerate(labels):
        if any(kw in label for kw in keywords):
            matched.append(idx)
    return matched
