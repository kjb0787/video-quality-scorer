"""
Walk the extracted UCF-101 directory and build a Parquet index table.

Output schema:
    video_id:   str  (e.g., "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01")
    video_path: str  (absolute path to .avi file)
    category:   str  (e.g., "ApplyEyeMakeup")
    filename:   str  (e.g., "v_ApplyEyeMakeup_g01_c01.avi")

Usage:
    python scripts/build_index.py [--data-dir data/UCF-101] [--output data/index.parquet]
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_VIDEO_EXTS = ("*.avi", "*.mp4", "*.mkv", "*.mov", "*.webm")


def build_index(ucf_dir: Path, output_path: Path) -> int:
    records = []
    for category_dir in sorted(ucf_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        video_files = sorted(
            f for ext in _VIDEO_EXTS for f in category_dir.glob(ext)
        )
        for video_file in video_files:
            records.append({
                "video_id": f"{category}/{video_file.stem}",
                "video_path": str(video_file.resolve()),
                "category": category,
                "filename": video_file.name,
            })

    table = pa.Table.from_pylist(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Build Parquet index for UCF-101")
    parser.add_argument("--data-dir", type=Path, default=Path("data/UCF-101"))
    parser.add_argument("--output", type=Path, default=Path("data/index.parquet"))
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist. Run download_ucf101.py first.")
        return

    n = build_index(args.data_dir, args.output)
    print(f"Wrote {n} entries to {args.output}")


if __name__ == "__main__":
    main()
