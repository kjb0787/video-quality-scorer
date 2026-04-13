"""Tests for temporal consistency scorer."""

import numpy as np
import pytest

from vq.scorers.temporal_consistency import TemporalConsistencyScorer


@pytest.fixture(scope="module")
def scorer():
    """Load the scorer once for all tests in this module (model loading is slow)."""
    return TemporalConsistencyScorer()


def test_temporal_consistency_output(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    assert "temporal_consistency" in result
    assert result["temporal_consistency"].shape == (1,)
    # Score should be between 0 and 1
    assert 0.0 <= result["temporal_consistency"][0] <= 1.0


def test_temporal_consistency_batch(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video), str(synthetic_video)]),
        "category": np.array(["ActionA", "ActionB"]),
    }
    result = scorer(batch)
    assert result["temporal_consistency"].shape == (2,)
