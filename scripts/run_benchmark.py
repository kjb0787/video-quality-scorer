"""
Benchmark all 6 scorers across diverse UCF-101 categories.

Runs one category at a time to stay within RAM limits (~200 videos max per run).
Collects per-category statistics and writes a summary CSV.

Usage:
    python scripts/run_benchmark.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vq.config import Settings
from vq.pipeline import run_pipeline

CATEGORIES = [
    "Basketball",
    "Bowling",
    "CricketShot",
    "Drumming",
    "HorseRiding",
    "IceDancing",
    "RockClimbingIndoor",
    "Skijet",
    "SoccerPenalty",
    "Typing",
]


def run_single_category(category: str, base_settings: Settings) -> pd.DataFrame:
    """Filter index to one category, run pipeline, return scored DataFrame."""
    # Read full index, filter to this category
    full_index = pd.read_parquet(base_settings.index_path)
    cat_df = full_index[full_index["category"] == category]
    print(f"\n{'='*60}")
    print(f"  {category}: {len(cat_df)} videos")
    print(f"{'='*60}")

    if len(cat_df) == 0:
        print(f"  SKIP: no videos found for {category}")
        return pd.DataFrame()

    # Write temporary per-category index (reset_index to avoid pandas index duplication in Ray)
    tmp_index = base_settings.data_dir / f"index_{category}.parquet"
    cat_df.reset_index(drop=True).to_parquet(tmp_index, index=False)

    # Create settings pointing to the filtered index
    cat_settings = Settings(
        index_path=tmp_index,
        output_dir=base_settings.output_dir / f"benchmark_{category}",
        batch_size=base_settings.batch_size,
        num_workers=base_settings.num_workers,
    )

    try:
        run_pipeline(cat_settings)
        scored = pd.read_parquet(cat_settings.output_dir / "scored")
        return scored
    finally:
        # Clean up temp index
        tmp_index.unlink(missing_ok=True)


def compute_category_stats(scored: pd.DataFrame, category: str) -> dict:
    """Compute summary statistics for one category."""
    return {
        "category": category,
        "n_videos": len(scored),
        "temporal_consistency_mean": scored["temporal_consistency"].mean(),
        "temporal_consistency_std": scored["temporal_consistency"].std(),
        "scene_cut_count": int(scored["has_scene_cut"].sum()),
        "min_frame_similarity_mean": scored["min_frame_similarity"].mean(),
        "aesthetic_score_mean": scored["aesthetic_score"].mean(),
        "aesthetic_score_std": scored["aesthetic_score"].std(),
        "prompt_alignment_mean": scored["prompt_alignment"].mean(),
        "prompt_alignment_std": scored["prompt_alignment"].std(),
        "optical_flow_mean": scored["optical_flow_mean"].mean(),
        "optical_flow_std": scored["optical_flow_mean"].std(),
        "duplicate_count": int(scored["is_near_duplicate"].sum()),
    }


def main():
    base_settings = Settings(num_workers=1, batch_size=8)

    all_scored = []
    category_stats = []

    for category in CATEGORIES:
        # Resume support: reuse existing scored data if available
        cached_path = base_settings.output_dir / f"benchmark_{category}" / "scored"
        if cached_path.exists():
            try:
                scored = pd.read_parquet(cached_path)
                if len(scored) > 0 and "is_near_duplicate" in scored.columns:
                    print(f"\n  {category}: reusing {len(scored)} cached results")
                    all_scored.append(scored)
                    stats = compute_category_stats(scored, category)
                    category_stats.append(stats)
                    print(f"  temporal_consistency: {stats['temporal_consistency_mean']:.4f}")
                    print(f"  aesthetic_score:      {stats['aesthetic_score_mean']:.4f}")
                    print(f"  prompt_alignment:     {stats['prompt_alignment_mean']:.4f}")
                    print(f"  optical_flow:         {stats['optical_flow_mean']:.2f}")
                    print(f"  scene_cuts:           {stats['scene_cut_count']}")
                    print(f"  duplicates:           {stats['duplicate_count']}")
                    continue
            except Exception:
                pass  # Fall through to re-run

        scored = run_single_category(category, base_settings)
        if scored.empty:
            continue
        all_scored.append(scored)
        stats = compute_category_stats(scored, category)
        category_stats.append(stats)

        # Print progress
        print(f"  temporal_consistency: {stats['temporal_consistency_mean']:.4f}")
        print(f"  aesthetic_score:      {stats['aesthetic_score_mean']:.4f}")
        print(f"  prompt_alignment:     {stats['prompt_alignment_mean']:.4f}")
        print(f"  optical_flow:         {stats['optical_flow_mean']:.2f}")
        print(f"  scene_cuts:           {stats['scene_cut_count']}")
        print(f"  duplicates:           {stats['duplicate_count']}")

    # Build summary table
    summary = pd.DataFrame(category_stats)
    output_path = base_settings.output_dir / "benchmark_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"\n\nBenchmark summary written to {output_path}")
    print(f"\n{summary.to_string(index=False)}")

    # Also save all scored data combined
    if all_scored:
        combined = pd.concat(all_scored, ignore_index=True)
        combined.to_parquet(base_settings.output_dir / "benchmark_all_scored.parquet")
        print(f"\nCombined scored data: {len(combined)} videos")


if __name__ == "__main__":
    main()
