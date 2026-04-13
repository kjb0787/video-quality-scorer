"""Aesthetic scorer: LAION aesthetic predictor (MLP on CLIP ViT-L-14 embeddings)."""

from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

from vq.config import Settings
from vq.device import get_device, get_torch_dtype
from vq.scorer_base import BaseScorer
from vq.video import extract_frames


class AestheticMLP(nn.Module):
    """Reimplementation of the LAION aesthetic predictor MLP (no pytorch-lightning dep)."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer(BaseScorer):
    """
    LAION aesthetic predictor: linear probe on CLIP ViT-L-14 embeddings.
    Predicts a 1-10 aesthetic rating per frame, averaged across sampled frames.
    """

    score_columns = ["aesthetic_score"]

    def __init__(self):
        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)
        settings = Settings()

        # Load CLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained, device=self.device,
        )
        self.model.eval()

        # Load aesthetic MLP
        weights_path = settings.model_cache_dir / settings.aesthetic_weights_filename
        if not weights_path.exists():
            self._download_weights(settings.aesthetic_weights_url, weights_path)

        self.mlp = AestheticMLP(input_dim=768).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.mlp.eval()

    @staticmethod
    def _download_weights(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading aesthetic predictor weights to {dest}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    @torch.inference_mode()
    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scores = []
        for video_path in batch["video_path"]:
            frames = extract_frames(video_path, num_frames=8, resize=None)
            if not frames:
                scores.append(0.0)
                continue

            tensors = torch.stack([self.preprocess(f) for f in frames])
            tensors = tensors.to(device=self.device, dtype=self.dtype)

            # CLIP embeddings, L2-normalized (critical for aesthetic predictor)
            embeddings = self.model.encode_image(tensors)
            embeddings = F.normalize(embeddings, dim=-1).float()

            # MLP predicts aesthetic score per frame
            frame_scores = self.mlp(embeddings).squeeze(-1)
            scores.append(float(frame_scores.mean().cpu()))

        batch["aesthetic_score"] = np.array(scores, dtype=np.float32)
        return batch
