"""Prompt-video alignment scorer: CLIP text-video cosine similarity."""

import re

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from vq.config import Settings
from vq.device import get_device, get_torch_dtype
from vq.scorer_base import BaseScorer
from vq.video import extract_frames


def label_to_prompt(label: str) -> str:
    """Convert CamelCase UCF-101 label to natural language prompt.

    'ApplyEyeMakeup' -> 'a video of apply eye makeup'
    'BasketballDunk' -> 'a video of basketball dunk'
    """
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", label).lower()
    return f"a video of {words}"


class PromptAlignmentScorer(BaseScorer):
    """
    CLIP cosine similarity between text=category_label and image=sampled_frames.
    Measures whether video content matches its annotated category.
    """

    score_columns = ["prompt_alignment"]

    def __init__(self):
        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)
        settings = Settings()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scores = []
        for video_path, category in zip(batch["video_path"], batch["category"]):
            frames = extract_frames(video_path, num_frames=8, resize=None)
            if not frames:
                scores.append(0.0)
                continue

            # Encode frames
            tensors = torch.stack([self.preprocess(f) for f in frames])
            tensors = tensors.to(device=self.device, dtype=self.dtype)
            image_features = self.model.encode_image(tensors)
            image_features = F.normalize(image_features, dim=-1)

            # Encode text prompt
            prompt = label_to_prompt(category)
            text_tokens = self.tokenizer([prompt]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            # Mean cosine similarity across frames
            cos_sim = (image_features @ text_features.T).mean()
            scores.append(float(cos_sim.cpu()))

        batch["prompt_alignment"] = np.array(scores, dtype=np.float32)
        return batch
