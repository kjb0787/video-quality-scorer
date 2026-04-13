"""Tests for manifest builder."""

import pandas as pd
import pytest

from vq.config import Settings
from vq.manifest import _compute_composite_quality, _resolve_cross_actor_duplicates, build_manifest


@pytest.fixture
def scored_df():
    """Mock scored DataFrame as if pipeline already ran."""
    return pd.DataFrame({
        "video_id": ["A/v1", "A/v2", "A/v3", "B/v1", "B/v2"],
        "video_path": ["/tmp/a1", "/tmp/a2", "/tmp/a3", "/tmp/b1", "/tmp/b2"],
        "category": ["ActionA", "ActionA", "ActionA", "ActionB", "ActionB"],
        "filename": ["v1.avi", "v2.avi", "v3.avi", "v1.avi", "v2.avi"],
        "temporal_consistency": [0.9, 0.7, 0.5, 0.8, 0.6],
        "aesthetic_score": [6.0, 5.0, 3.0, 7.0, 4.0],
        "prompt_alignment": [0.8, 0.6, 0.4, 0.7, 0.5],
        "optical_flow_mean": [2.0, 3.0, 1.0, 2.5, 4.0],
        "optical_flow_std": [0.5, 0.3, 0.2, 0.4, 0.6],
        "phash": ["abcd1234abcd1234", "abcd1235abcd1235", "ffff0000ffff0000", "1111aaaa1111aaaa", "2222bbbb2222bbbb"],
        "is_near_duplicate": [False, False, False, False, False],
    })


def test_composite_quality_shape(scored_df):
    quality = _compute_composite_quality(scored_df)
    assert len(quality) == len(scored_df)
    assert all(0.0 <= q <= 1.0 for q in quality)


def test_build_manifest_from_parquet(scored_df, tmp_path):
    # Write mock scored data
    scored_dir = tmp_path / "output" / "scored"
    scored_dir.mkdir(parents=True)
    scored_df.to_parquet(scored_dir / "part-0.parquet")

    settings = Settings(output_dir=tmp_path / "output")
    manifest = build_manifest(settings)

    assert len(manifest) == 2  # ActionA, ActionB
    assert "mean_quality" in manifest.columns
    assert "p50_quality" in manifest.columns
    assert "filtered_count" in manifest.columns

    row_a = manifest[manifest["category"] == "ActionA"].iloc[0]
    assert row_a["count"] == 3


def test_manifest_filters_low_quality(scored_df, tmp_path):
    # Make one video very low quality
    scored_df.loc[2, "temporal_consistency"] = 0.0
    scored_df.loc[2, "aesthetic_score"] = 0.0
    scored_df.loc[2, "prompt_alignment"] = 0.0

    scored_dir = tmp_path / "output" / "scored"
    scored_dir.mkdir(parents=True)
    scored_df.to_parquet(scored_dir / "part-0.parquet")

    settings = Settings(output_dir=tmp_path / "output", min_quality_score=0.4)
    manifest = build_manifest(settings)

    row_a = manifest[manifest["category"] == "ActionA"].iloc[0]
    assert row_a["filtered_count"] <= row_a["count"]


def test_cross_actor_dedup_catches_split_duplicates():
    """Two similar hashes processed by different actors (both marked non-duplicate)."""
    df = pd.DataFrame({
        "video_id": ["A/v1", "A/v2", "A/v3"],
        "category": ["ActionA", "ActionA", "ActionA"],
        # v1 and v2 have very similar hashes (1 bit different), v3 is different
        "phash": ["abcd1234abcd1234", "abcd1234abcd1235", "ffff0000ffff0000"],
        "is_near_duplicate": [False, False, False],  # actors didn't see each other
    })
    result = _resolve_cross_actor_duplicates(df, hamming_threshold=8)
    # v2 should be flagged as duplicate of v1
    assert result["is_near_duplicate"].iloc[0] == False  # v1 is the original
    assert result["is_near_duplicate"].iloc[1] == True   # v2 is a dup
    assert result["is_near_duplicate"].iloc[2] == False   # v3 is different


def test_cross_actor_dedup_no_cascade():
    """Duplicates should not be used as reference for further matching.

    v2 is similar to v1 (dup). v3 is similar to v2 but NOT to v1.
    Without cascade prevention: v3 flagged (matches v2 in refs).
    With cascade prevention: v3 NOT flagged (only v1 in refs, too far).
    """
    # Carefully chosen hashes:
    # h1 = 0x0000000000000000
    # h2 = 0x00000000000000ff  (hamming from h1 = 8, within threshold 8)
    # h3 = 0x000000000000ffff  (hamming from h2 = 8, within threshold 8)
    #                          (hamming from h1 = 16, OUTSIDE threshold 8)
    df = pd.DataFrame({
        "video_id": ["A/v1", "A/v2", "A/v3"],
        "category": ["ActionA", "ActionA", "ActionA"],
        "phash": ["0000000000000000", "00000000000000ff", "000000000000ffff"],
        "is_near_duplicate": [False, False, False],
    })
    result = _resolve_cross_actor_duplicates(df, hamming_threshold=8)
    assert result["is_near_duplicate"].iloc[0] == False  # v1 original
    assert result["is_near_duplicate"].iloc[1] == True   # v2 dup of v1
    # v3: only compared against v1 (v2 excluded as dup). dist(v1,v3)=16 > 8
    assert result["is_near_duplicate"].iloc[2] == False   # NOT a dup
