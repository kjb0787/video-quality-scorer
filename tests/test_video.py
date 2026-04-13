"""Tests for video frame extraction."""

from pathlib import Path

import numpy as np
from PIL import Image

from vq.video import extract_frame_pairs, extract_frames


def test_extract_frames_count(synthetic_video: Path):
    frames = extract_frames(synthetic_video, num_frames=8)
    assert len(frames) == 8


def test_extract_frames_type(synthetic_video: Path):
    frames = extract_frames(synthetic_video, num_frames=4)
    for f in frames:
        assert isinstance(f, Image.Image)
        assert f.mode == "RGB"


def test_extract_frames_resize(synthetic_video: Path):
    frames = extract_frames(synthetic_video, num_frames=4, resize=(112, 112))
    for f in frames:
        assert f.size == (112, 112)


def test_extract_frames_no_resize(synthetic_video: Path):
    frames = extract_frames(synthetic_video, num_frames=4, resize=None)
    for f in frames:
        assert f.size == (64, 64)


def test_extract_frame_pairs_count(synthetic_video: Path):
    pairs = extract_frame_pairs(synthetic_video, num_pairs=5)
    assert len(pairs) == 5


def test_extract_frame_pairs_shape(synthetic_video: Path):
    pairs = extract_frame_pairs(synthetic_video, num_pairs=3)
    for prev_frame, next_frame in pairs:
        assert isinstance(prev_frame, np.ndarray)
        assert isinstance(next_frame, np.ndarray)
        assert prev_frame.shape == next_frame.shape
        assert len(prev_frame.shape) == 3  # HWC


def test_extract_frames_nonexistent():
    frames = extract_frames("/nonexistent/path/video.avi")
    assert frames == []


def test_extract_frames_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.avi"
    empty.touch()
    frames = extract_frames(str(empty))
    assert frames == []


def test_extract_frame_pairs_nonexistent():
    pairs = extract_frame_pairs("/nonexistent/path/video.avi")
    assert pairs == []
