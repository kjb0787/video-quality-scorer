"""Optical flow scorer: Farneback mean magnitude on consecutive frame pairs."""

import cv2
import numpy as np

from vq.scorer_base import BaseScorer
from vq.video import extract_frame_pairs


class OpticalFlowScorer(BaseScorer):
    """
    Farneback optical flow on consecutive frame pairs.
    Computes mean and std of flow magnitude.
    Filters static clips (too low) and extreme motion blur (too high).
    """

    score_columns = ["optical_flow_mean", "optical_flow_std"]

    def __init__(self):
        # Farneback params
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        means = []
        stds = []
        for video_path in batch["video_path"]:
            pairs = extract_frame_pairs(video_path, num_pairs=8)
            if not pairs:
                means.append(0.0)
                stds.append(0.0)
                continue

            magnitudes = []
            for prev_frame, next_frame in pairs:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray, None,
                    pyr_scale=self.pyr_scale,
                    levels=self.levels,
                    winsize=self.winsize,
                    iterations=self.iterations,
                    poly_n=self.poly_n,
                    poly_sigma=self.poly_sigma,
                    flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                magnitudes.append(float(mag.mean()))

            means.append(float(np.mean(magnitudes)))
            stds.append(float(np.std(magnitudes)))

        batch["optical_flow_mean"] = np.array(means, dtype=np.float32)
        batch["optical_flow_std"] = np.array(stds, dtype=np.float32)
        return batch
