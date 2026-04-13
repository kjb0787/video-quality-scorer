"""Tests for scene cut detector."""

import numpy as np
import pytest

from vq.scorers.scene_cut import SceneCutDetector


@pytest.fixture(scope="module")
def detector():
    return SceneCutDetector()


def test_scene_cut_output_columns(detector, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = detector(batch)
    assert "scene_cut_count" in result
    assert "has_scene_cut" in result
    assert "min_frame_similarity" in result


def test_scene_cut_count_type(detector, synthetic_video):
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = detector(batch)
    assert result["scene_cut_count"].dtype == np.int32
    assert result["scene_cut_count"][0] >= 0


def test_smooth_video_no_cuts(detector, synthetic_video_static):
    """A static video should have zero scene cuts."""
    batch = {
        "video_path": np.array([str(synthetic_video_static)]),
        "category": np.array(["TestAction"]),
    }
    result = detector(batch)
    assert result["scene_cut_count"][0] == 0
    assert result["has_scene_cut"][0] == False
    assert result["min_frame_similarity"][0] > 0.9
