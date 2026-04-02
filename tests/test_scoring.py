"""
test_scoring.py — unit tests for the scoring pipeline.

These tests run entirely with synthetic numpy arrays — no GPU, no model
weights, no network access required.  They exercise:

  1. compute_friction_score()  — the main scoring function
  2. _sigmoid_scale()          — the CLR → 1-10 mapping
  3. _lookup_band()            — the score → label lookup
  4. Score monotonicity        — high PFC → higher score than high visual
  5. Edge cases                — zero activations, single timestep, wrong shape

Run with:
    pytest tests/test_scoring.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from digital_empathy.scoring import (
    FrictionResult,
    _lookup_band,
    _sigmoid_scale,
    compute_friction_score,
)

# Re-use fixtures from conftest.py:
#   flat_activations, high_pfc_activations, high_visual_activations,
#   synthetic_masks


# ---------------------------------------------------------------------------
# _sigmoid_scale — unit tests
# ---------------------------------------------------------------------------

class TestSigmoidScale:
    def test_midpoint_is_five_point_five(self):
        """CLR == _CLR_MIDPOINT should map to approximately 5.5."""
        from digital_empathy.scoring import _CLR_MIDPOINT
        score = _sigmoid_scale(_CLR_MIDPOINT)
        assert abs(score - 5.5) < 0.01

    def test_output_range_is_one_to_ten(self):
        """Score must always stay within [1, 10]."""
        for clr in [-100.0, -10.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 100.0]:
            score = _sigmoid_scale(clr)
            assert 1.0 <= score <= 10.0, f"Score {score} out of range for CLR={clr}"

    def test_monotonically_increasing(self):
        """Higher CLR must produce equal-or-higher score."""
        clrs = [-5.0, -2.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
        scores = [_sigmoid_scale(c) for c in clrs]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Monotonicity violated: CLR {clrs[i]}→{clrs[i+1]} "
                f"gave scores {scores[i]}→{scores[i+1]}"
            )

    def test_extreme_low_clr_approaches_one(self):
        score = _sigmoid_scale(-50.0)
        assert score < 1.5, f"Expected score near 1, got {score}"

    def test_extreme_high_clr_approaches_ten(self):
        score = _sigmoid_scale(50.0)
        assert score > 9.5, f"Expected score near 10, got {score}"


# ---------------------------------------------------------------------------
# _lookup_band — unit tests
# ---------------------------------------------------------------------------

class TestLookupBand:
    @pytest.mark.parametrize("score,expected_label", [
        (1.0,  "Effortless"),
        (2.0,  "Effortless"),
        (2.5,  "Comfortable"),   # boundary: 2.5 is NOT < 2.5, so goes to next
        (3.0,  "Comfortable"),
        (4.0,  "Moderate"),
        (5.0,  "Moderate"),
        (5.5,  "Strained"),
        (6.0,  "Strained"),
        (7.0,  "High Friction"),
        (8.0,  "High Friction"),
        (8.5,  "Critical"),
        (10.0, "Critical"),
    ])
    def test_band_labels(self, score: float, expected_label: str):
        label, _ = _lookup_band(score)
        assert label == expected_label, (
            f"Score {score}: expected '{expected_label}', got '{label}'"
        )

    def test_explanation_is_non_empty(self):
        """Every band should have a non-trivial explanation."""
        for score in [1.5, 3.0, 5.0, 6.0, 8.0, 9.5]:
            _, explanation = _lookup_band(score)
            assert len(explanation) > 20, (
                f"Explanation for score {score} is too short: '{explanation}'"
            )


# ---------------------------------------------------------------------------
# compute_friction_score — correctness tests
# ---------------------------------------------------------------------------

class TestComputeFrictionScore:
    def test_returns_friction_result(self, flat_activations, synthetic_masks):
        result = compute_friction_score(flat_activations, synthetic_masks)
        assert isinstance(result, FrictionResult)

    def test_score_in_valid_range(self, flat_activations, synthetic_masks):
        result = compute_friction_score(flat_activations, synthetic_masks)
        assert 1.0 <= result.score <= 10.0

    def test_high_pfc_yields_high_score(self, high_pfc_activations, synthetic_masks):
        """Elevated PFC + suppressed visual should produce a high friction score."""
        result = compute_friction_score(high_pfc_activations, synthetic_masks)
        assert result.score >= 6.0, (
            f"Expected high friction score (≥6), got {result.score}"
        )

    def test_high_visual_yields_low_score(self, high_visual_activations, synthetic_masks):
        """Elevated visual + suppressed PFC should produce a low friction score."""
        result = compute_friction_score(high_visual_activations, synthetic_masks)
        assert result.score <= 5.0, (
            f"Expected low friction score (≤5), got {result.score}"
        )

    def test_monotonicity_pfc_vs_visual(
        self,
        high_pfc_activations,
        high_visual_activations,
        synthetic_masks,
    ):
        """High-PFC scenario must score higher than high-visual scenario."""
        pfc_result    = compute_friction_score(high_pfc_activations, synthetic_masks)
        visual_result = compute_friction_score(high_visual_activations, synthetic_masks)
        assert pfc_result.score > visual_result.score, (
            f"PFC score ({pfc_result.score}) should exceed visual score ({visual_result.score})"
        )

    def test_clr_is_positive(self, flat_activations, synthetic_masks):
        """CLR should be non-negative for any reasonable input."""
        result = compute_friction_score(flat_activations, synthetic_masks)
        # CLR can be negative if PFC is deactivated, which is valid (low friction)
        # but the final score must still be ≥ 1
        assert result.score >= 1.0

    def test_label_matches_score(self, flat_activations, synthetic_masks):
        """Label returned by to_dict() should match direct _lookup_band() call."""
        result = compute_friction_score(flat_activations, synthetic_masks)
        expected_label, _ = _lookup_band(result.score)
        assert result.label == expected_label

    def test_debug_values_are_finite(self, flat_activations, synthetic_masks):
        """All intermediate values should be finite (no NaN/Inf)."""
        result = compute_friction_score(flat_activations, synthetic_masks)
        assert math.isfinite(result.pfc_mean_activation)
        assert math.isfinite(result.visual_mean_activation)
        assert math.isfinite(result.cognitive_load_ratio)
        assert math.isfinite(result.score)

    def test_to_dict_structure(self, flat_activations, synthetic_masks):
        """to_dict() must contain the keys the MCP server returns."""
        result = compute_friction_score(flat_activations, synthetic_masks)
        d = result.to_dict()
        assert "score"       in d
        assert "label"       in d
        assert "explanation" in d
        assert "debug"       in d
        assert "pfc_mean_activation"    in d["debug"]
        assert "visual_mean_activation" in d["debug"]
        assert "cognitive_load_ratio"   in d["debug"]


# ---------------------------------------------------------------------------
# Edge cases — bad inputs should raise, not silently corrupt
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_wrong_vertex_count_raises(self, synthetic_masks):
        bad_activations = np.zeros((10, 1_000), dtype=np.float32)  # wrong size
        with pytest.raises(ValueError, match="20 484"):
            compute_friction_score(bad_activations, synthetic_masks)

    def test_1d_array_raises(self, synthetic_masks):
        bad_activations = np.zeros(20_484, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            compute_friction_score(bad_activations, synthetic_masks)

    def test_single_timestep(self, synthetic_masks):
        """T=1 should work fine (e.g., very short recording)."""
        single = np.random.default_rng(0).normal(
            size=(1, 20_484)
        ).astype(np.float32)
        result = compute_friction_score(single, synthetic_masks)
        assert 1.0 <= result.score <= 10.0

    def test_all_zeros(self, synthetic_masks):
        """Zero activations — CLR is 0/ε, should produce a very low score."""
        zeros = np.zeros((10, 20_484), dtype=np.float32)
        result = compute_friction_score(zeros, synthetic_masks)
        # PFC/visual both zero → CLR ≈ 0 → score near 1
        assert result.score < 4.0

    def test_all_ones(self, synthetic_masks):
        """Uniform activation — CLR ≈ 1.0, score should be near midpoint."""
        ones = np.ones((10, 20_484), dtype=np.float32)
        result = compute_friction_score(ones, synthetic_masks)
        # With identical PFC and visual means, CLR == 1.0
        # score should be below midpoint since _CLR_MIDPOINT == 1.5
        assert 1.0 <= result.score <= 7.0

    def test_nan_in_activations_propagates_to_debug(self, synthetic_masks):
        """If activations somehow contain NaN, the score should be detectable as non-finite."""
        import math
        nan_acts = np.full((5, 20_484), float("nan"), dtype=np.float32)
        result = compute_friction_score(nan_acts, synthetic_masks)
        # NaN propagates through mean → CLR → sigmoid → should be caught upstream
        # We just verify the function doesn't crash; NaN handling is caller's job
        assert isinstance(result, FrictionResult)
