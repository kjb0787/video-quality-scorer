"""
Main entry point. Run the full quality scoring pipeline.

Usage:
    python scripts/run_pipeline.py [--subset-n 100] [--batch-size 4] [--num-workers 2]

Environment variables (via pydantic-settings):
    VQ_SUBSET_N=100
    VQ_BATCH_SIZE=4
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vq.config import Settings
from vq.manifest import build_manifest
from vq.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Video Quality Scoring Pipeline")
    parser.add_argument("--subset-n", type=int, default=None, help="Process only first N videos")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    settings = Settings(
        subset_n=args.subset_n,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Running pipeline on {settings.subset_n or 'all'} videos...")
    print(f"  Index: {settings.index_path}")
    print(f"  Output: {settings.output_dir}")
    print()

    output_path = run_pipeline(settings)
    print(f"\nScored data written to {output_path}")

    print("\nBuilding manifest...")
    manifest = build_manifest(settings)

    print(f"\nManifest ({len(manifest)} categories):\n")
    display_cols = [
        "category", "count", "mean_quality", "p50_quality",
        "filtered_count", "duplicate_count",
    ]
    print(manifest[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
