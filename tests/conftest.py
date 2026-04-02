"""
conftest.py — shared pytest fixtures for digital_empathy tests.

All fixtures produce synthetic data that matches the exact shapes and dtypes
that TRIBE v2 produces, so tests run instantly without GPU or model weights.

fsaverage5 shape convention:
    activations: (T, 20_484)  — T timesteps, 20 484 cortical vertices
    vertices 0       .. 10 241  → left  hemisphere
    vertices 10 242  .. 20 483  → right hemisphere
"""

from __future__ import annotations

import numpy as np
import pytest

# Total cortical vertices in fsaverage5 (both hemispheres)
N_VERTICES: int = 20_484
# Default number of TR timesteps for synthetic videos
N_TIMESTEPS: int = 30


# ---------------------------------------------------------------------------
# Synthetic activation arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator — deterministic across runs."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def flat_activations(rng: np.random.Generator) -> np.ndarray:
    """Uniform low-level activation — represents a very simple, calm UI.

    Mean activation is near zero (z-scored baseline).
    Expected score: low friction (≈ 1.5–3.5).
    """
    return rng.normal(loc=0.0, scale=0.1, size=(N_TIMESTEPS, N_VERTICES)).astype(
        np.float32
    )


@pytest.fixture
def high_pfc_activations(rng: np.random.Generator) -> np.ndarray:
    """Elevated PFC activation, suppressed visual — represents a confusing UI.

    PFC vertices (first ~1 400 of left hemi + equivalent right) are pushed
    strongly positive; visual vertices held near zero.
    Expected score: high friction (≈ 7.0–9.5).

    We approximate PFC vertex positions by taking the first 1 400 vertices
    of each hemisphere — a rough but deterministic stand-in for the real
    Destrieux frontal labels (which we cannot load in unit tests without
    nilearn network access).
    """
    acts = rng.normal(loc=0.0, scale=0.15, size=(N_TIMESTEPS, N_VERTICES)).astype(
        np.float32
    )
    # Pump up "frontal-ish" vertices (arbitrary positional proxy)
    acts[:, :1_400] += 2.5           # left PFC proxy
    acts[:, 10_242:11_642] += 2.5    # right PFC proxy
    # Suppress "visual-ish" vertices (last ~1 000 of each hemi)
    acts[:, 9_242:10_242] -= 1.5     # left visual proxy
    acts[:, 19_484:20_484] -= 1.5    # right visual proxy
    return acts


@pytest.fixture
def high_visual_activations(rng: np.random.Generator) -> np.ndarray:
    """Suppressed PFC, elevated visual — user is passively watching.

    Expected score: low friction (≈ 1.0–3.0).
    """
    acts = rng.normal(loc=0.0, scale=0.15, size=(N_TIMESTEPS, N_VERTICES)).astype(
        np.float32
    )
    acts[:, :1_400] -= 1.0           # suppress PFC proxy
    acts[:, 10_242:11_642] -= 1.0
    acts[:, 9_242:10_242] += 2.5     # elevate visual proxy
    acts[:, 19_484:20_484] += 2.5
    return acts


# ---------------------------------------------------------------------------
# Synthetic BrainMasks
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_masks() -> "BrainMasks":
    """BrainMasks using positional proxies instead of the real atlas.

    Uses the same vertex index ranges as the high_pfc / high_visual fixtures
    so the scoring tests produce predictable outcomes.
    """
    from digital_empathy.brain_regions import BrainMasks

    pfc = np.zeros(N_VERTICES, dtype=bool)
    pfc[:1_400] = True            # left PFC proxy
    pfc[10_242:11_642] = True     # right PFC proxy

    visual = np.zeros(N_VERTICES, dtype=bool)
    visual[9_242:10_242] = True   # left visual proxy
    visual[19_484:20_484] = True  # right visual proxy

    return BrainMasks(
        pfc=pfc,
        visual=visual,
        pfc_labels=["frontal_proxy"],
        visual_labels=["occipital_proxy"],
    )
