"""Perceptual hash deduplication scorer."""

from collections import defaultdict

import imagehash
import numpy as np

from vq.config import Settings
from vq.scorer_base import BaseScorer
from vq.video import extract_frames


class PHashDedupScorer(BaseScorer):
    """
    Compute perceptual hash (pHash) on a keyframe per video.
    Flag near-duplicates within a category using Hamming distance.

    This scorer is stateful across batches within an actor — it accumulates
    hashes to detect duplicates across batches processed by the same actor.
    Cross-actor duplicates are resolved in the manifest builder.
    """

    score_columns = ["phash", "is_near_duplicate"]

    def __init__(self):
        settings = Settings()
        self.hamming_threshold = settings.phash_hamming_threshold
        self._seen: dict[str, list[imagehash.ImageHash]] = defaultdict(list)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        hashes = []
        dup_flags = []

        for video_path, category in zip(batch["video_path"], batch["category"]):
            # Extract a single middle keyframe
            frames = extract_frames(video_path, num_frames=1, resize=(224, 224))
            if not frames:
                hashes.append("")
                dup_flags.append(False)
                continue

            h = imagehash.phash(frames[0])

            # Check against seen hashes for this category
            is_dup = any(
                (h - existing) <= self.hamming_threshold
                for existing in self._seen[category]
            )

            if not is_dup:
                # Only use non-duplicate hashes as reference to avoid cascade
                self._seen[category].append(h)
            hashes.append(str(h))
            dup_flags.append(is_dup)

        batch["phash"] = np.array(hashes)
        batch["is_near_duplicate"] = np.array(dup_flags)
        return batch
