"""Temporal consistency scorer: CLIP frame-to-frame cosine similarity."""

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from vq.config import Settings
from vq.device import get_device, get_torch_dtype
from vq.scorer_base import BaseScorer
from vq.video import extract_frames


class TemporalConsistencyScorer(BaseScorer):
    """
    Encode consecutive frames with CLIP, compute mean pairwise cosine similarity.
    Low score = flickery/incoherent video — bad training signal.
    """

    score_columns = ["temporal_consistency"]

    def __init__(self):
        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)
        settings = Settings()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained, device=self.device,
        )
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scores = []
        for video_path in batch["video_path"]:
            frames = extract_frames(video_path, num_frames=16, resize=None)
            if len(frames) < 2:
                scores.append(0.0)
                continue

            # Preprocess and stack
            tensors = torch.stack([self.preprocess(f) for f in frames])
            tensors = tensors.to(device=self.device, dtype=self.dtype)

            # Encode all frames in one forward pass
            embeddings = self.model.encode_image(tensors)
            embeddings = F.normalize(embeddings, dim=-1)

            # Cosine similarity between consecutive frames
            cos_sims = (embeddings[:-1] * embeddings[1:]).sum(dim=-1)
            scores.append(float(cos_sims.mean().cpu()))

        batch["temporal_consistency"] = np.array(scores, dtype=np.float32)
        return batch
