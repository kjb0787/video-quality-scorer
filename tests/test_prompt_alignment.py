"""Tests for prompt alignment scorer."""

import numpy as np
import pytest

from vq.scorers.prompt_alignment import PromptAlignmentScorer, label_to_prompt


@pytest.fixture(scope="module")
def scorer():
    return PromptAlignmentScorer()


def test_label_to_prompt():
    assert label_to_prompt("ApplyEyeMakeup") == "a video of apply eye makeup"
    assert label_to_prompt("BasketballDunk") == "a video of basketball dunk"
    assert label_to_prompt("Bowling") == "a video of bowling"


def test_prompt_alignment_output(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    assert "prompt_alignment" in result
    assert result["prompt_alignment"].shape == (1,)


def test_prompt_alignment_batch(scorer, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video), str(synthetic_video)]),
        "category": np.array(["ActionA", "ActionB"]),
    }
    result = scorer(batch)
    assert result["prompt_alignment"].shape == (2,)
