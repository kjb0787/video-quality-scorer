"""Shared test fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a small synthetic .avi video (20 frames, 64x64, color gradient)."""
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (64, 64))

    for i in range(20):
        # Create frames with gradually changing colors
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * i / 19)  # Blue gradient
        frame[:, :, 1] = 128
        frame[:, :, 2] = int(255 * (19 - i) / 19)  # Red inverse gradient
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def synthetic_video_static(tmp_path: Path) -> Path:
    """Create a static video (all identical frames)."""
    video_path = tmp_path / "static_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (64, 64))

    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    for _ in range(20):
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_index_parquet(tmp_path: Path, synthetic_video: Path) -> Path:
    """Create a minimal Parquet index pointing to synthetic videos."""
    # Create 3 copies with different categories
    videos = []
    for i, cat in enumerate(["ActionA", "ActionB", "ActionA"]):
        dst = tmp_path / cat / f"video_{i}.avi"
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy the synthetic video
        import shutil
        shutil.copy2(synthetic_video, dst)
        videos.append({
            "video_id": f"{cat}/video_{i}",
            "video_path": str(dst),
            "category": cat,
            "filename": f"video_{i}.avi",
        })

    index_path = tmp_path / "index.parquet"
    table = pa.Table.from_pylist(videos)
    pq.write_table(table, index_path)
    return index_path
