"""Video frame extraction utilities using OpenCV."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_frames(
    video_path: str | Path,
    num_frames: int = 16,
    resize: tuple[int, int] | None = (224, 224),
) -> list[Image.Image]:
    """
    Uniformly sample `num_frames` from a video file.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            if resize:
                img = img.resize(resize, Image.LANCZOS)
            frames.append(img)
        return frames
    finally:
        cap.release()


def extract_frame_pairs(
    video_path: str | Path,
    num_pairs: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract consecutive frame pairs for optical flow computation.
    Returns list of (frame_t, frame_t+1) as numpy uint8 HWC arrays (BGR).
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            return []

        # Sample num_pairs+1 evenly spaced frames, then pair them
        n_frames = min(num_pairs + 1, total)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)

        pairs = []
        for i in range(len(frames) - 1):
            pairs.append((frames[i], frames[i + 1]))
        return pairs
    finally:
        cap.release()
