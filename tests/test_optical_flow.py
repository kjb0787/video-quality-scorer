"""Tests for optical flow scorer."""

import numpy as np

from vq.scorers.optical_flow import OpticalFlowScorer


def test_optical_flow_scorer(synthetic_video):
    scorer = OpticalFlowScorer()
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    assert "optical_flow_mean" in result
    assert "optical_flow_std" in result
    assert result["optical_flow_mean"].shape == (1,)
    assert result["optical_flow_mean"][0] >= 0


def test_optical_flow_static_video(synthetic_video_static):
    scorer = OpticalFlowScorer()
    batch = {
        "video_path": np.array([str(synthetic_video_static)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    # Static video should have very low flow
    assert result["optical_flow_mean"][0] < 1.0
