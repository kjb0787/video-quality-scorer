"""Tests for perceptual hash dedup scorer."""

import shutil

import numpy as np

from vq.scorers.phash_dedup import PHashDedupScorer


def test_phash_produces_hash(synthetic_video):
    scorer = PHashDedupScorer()
    batch = {
        "video_path": np.array([str(synthetic_video)]),
        "category": np.array(["TestAction"]),
    }
    result = scorer(batch)
    assert "phash" in result
    assert "is_near_duplicate" in result
    assert len(result["phash"][0]) > 0  # non-empty hash string


def test_phash_detects_duplicate(synthetic_video, tmp_path):
    # Copy the same video — should be detected as duplicate
    dup_path = tmp_path / "duplicate.avi"
    shutil.copy2(synthetic_video, dup_path)

    scorer = PHashDedupScorer()
    batch = {
        "video_path": np.array([str(synthetic_video), str(dup_path)]),
        "category": np.array(["TestAction", "TestAction"]),
    }
    result = scorer(batch)
    assert result["is_near_duplicate"][0] == False  # first is never a dup
    assert result["is_near_duplicate"][1] == True   # second is a dup


def test_phash_different_categories_not_duplicate(synthetic_video, tmp_path):
    dup_path = tmp_path / "duplicate.avi"
    shutil.copy2(synthetic_video, dup_path)

    scorer = PHashDedupScorer()
    batch = {
        "video_path": np.array([str(synthetic_video), str(dup_path)]),
        "category": np.array(["ActionA", "ActionB"]),  # different categories
    }
    result = scorer(batch)
    # Same video but different categories — not flagged as duplicate
    assert result["is_near_duplicate"][0] == False
    assert result["is_near_duplicate"][1] == False
