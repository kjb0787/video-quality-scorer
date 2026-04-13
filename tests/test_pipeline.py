"""Integration test for the full Ray pipeline."""

import pytest

from vq.config import Settings
from vq.pipeline import run_pipeline


@pytest.mark.slow
def test_pipeline_end_to_end(sample_index_parquet, tmp_path):
    """Run the full pipeline on 3 synthetic videos."""
    settings = Settings(
        index_path=sample_index_parquet,
        output_dir=tmp_path / "output",
        batch_size=2,
        num_workers=1,
        ray_num_cpus=6,
    )

    output_path = run_pipeline(settings)

    # Check output exists
    import pyarrow.parquet as pq
    df = pq.read_table(output_path).to_pandas()

    assert len(df) == 3
    expected_cols = [
        "video_id", "video_path", "category", "filename",
        "temporal_consistency", "aesthetic_score", "prompt_alignment",
        "optical_flow_mean", "optical_flow_std",
        "phash", "is_near_duplicate",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
