"""Tests for aesthetic scorer."""

import numpy as np
import pytest

from vq.scorers.aesthetic import AestheticScorer


@pytest.fixture(scope="module")
def scorer():
    return AestheticScorer()


def test_aesthetic_output(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    assert "aesthetic_score" in result
    assert result["aesthetic_score"].shape == (1,)
    # Aesthetic scores are typically 1-10
    assert result["aesthetic_score"][0] > 0


def test_aesthetic_batch(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video), str(synthetic_video)]),
        "category": np.array(["ActionA", "ActionB"]),
    }
    result = scorer(batch)
    assert result["aesthetic_score"].shape == (2,)
