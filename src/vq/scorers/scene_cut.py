"""Scene cut detector: flags videos with abrupt content changes between frames."""

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from vq.config import Settings
from vq.device import get_device, get_torch_dtype
from vq.scorer_base import BaseScorer
from vq.video import extract_frames


# A scene cut is different from low temporal consistency.
# A slow crossfade has high temporal consistency but IS a scene cut.
# A fast camera pan has low temporal consistency but is NOT a scene cut.
# The key difference: scene cuts produce a sharp DROP in frame-to-frame
# similarity, not a gradual decline. We detect this by looking for
# consecutive CLIP similarity values that fall below a hard threshold,
# rather than averaging across the whole video.

SCENE_CUT_THRESHOLD = 0.75


class SceneCutDetector(BaseScorer):
    """
    Detect scene cuts by finding sharp drops in CLIP frame-to-frame similarity.

    Unlike TemporalConsistencyScorer (which averages similarity), this flags
    individual frame transitions where similarity drops below a threshold —
    indicating a hard cut, flash, or abrupt scene change.

    For training video generation models, scene cuts are especially harmful:
    the model learns that abrupt visual discontinuities are normal, producing
    flickery or incoherent generated video.
    """

    score_columns = ["scene_cut_count", "has_scene_cut", "min_frame_similarity"]

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
        cut_counts = []
        has_cuts = []
        min_sims = []

        for video_path in batch["video_path"]:
            frames = extract_frames(video_path, num_frames=16, resize=None)
            if len(frames) < 2:
                cut_counts.append(0)
                has_cuts.append(False)
                min_sims.append(0.0)
                continue

            tensors = torch.stack([self.preprocess(f) for f in frames])
            tensors = tensors.to(device=self.device, dtype=self.dtype)

            embeddings = self.model.encode_image(tensors)
            embeddings = F.normalize(embeddings, dim=-1)

            # Per-transition similarity scores
            cos_sims = (embeddings[:-1] * embeddings[1:]).sum(dim=-1)
            sims = cos_sims.cpu().numpy()

            # Count transitions below threshold
            n_cuts = int((sims < SCENE_CUT_THRESHOLD).sum())
            cut_counts.append(n_cuts)
            has_cuts.append(n_cuts > 0)
            min_sims.append(float(sims.min()))

        batch["scene_cut_count"] = np.array(cut_counts, dtype=np.int32)
        batch["has_scene_cut"] = np.array(has_cuts)
        batch["min_frame_similarity"] = np.array(min_sims, dtype=np.float32)
        return batch
