# Video Quality Scorer

Ray-based pipeline for scoring video training data quality. Computes 6 quality signals per video and produces a per-category manifest for ML training data curation.

## Why This Exists

Training video generation models (like Luma's Uni-1) requires high-quality, well-curated training data. Not all videos are equal — some are flickery, static, aesthetically poor, or mislabeled. This pipeline automates quality assessment at scale, producing a manifest that feeds directly into a category-weighted training data sampler.

## Architecture

```
Video Dataset → Parquet Index → Ray Data Pipeline → Scored Parquet → Category Manifest
                                       │
                     ┌─────────────────┼──────────────────┐
                     │                 │                   │
               CLIP-based (GPU)   CPU-based           Stateful
               ┌──────────┐    ┌──────────┐       ┌──────────┐
               │Temporal   │    │Optical   │       │pHash     │
               │Consistency│    │Flow      │       │Dedup     │
               ├──────────┤    └──────────┘       └──────────┘
               │Scene Cut  │
               │Detector   │
               ├──────────┤
               │Aesthetic  │
               │Score      │
               ├──────────┤
               │Prompt     │
               │Alignment  │
               └──────────┘
```

## Quality Signals

| Signal | Method | Why It Matters |
|--------|--------|----------------|
| **Temporal Consistency** | CLIP frame-to-frame cosine similarity | Low score = flickery, incoherent video — bad training signal for temporal generation |
| **Scene Cut Detection** | Per-transition CLIP similarity thresholding | Flags hard cuts, flashes, and abrupt scene changes that teach the model visual discontinuity is normal |
| **Aesthetic Score** | LAION aesthetic predictor (MLP on CLIP ViT-L-14) | Filters low-visual-quality frames that degrade generation aesthetics |
| **Prompt-Video Alignment** | CLIP text-video cosine similarity | Ensures video content matches its category label — critical for conditional generation |
| **Optical Flow Magnitude** | Farneback (OpenCV) | Filters static clips (no motion) and extreme motion blur |
| **Perceptual Hash Dedup** | pHash + Hamming distance | Removes near-duplicate videos within categories to prevent training bias |

## Quick Start

```bash
# Setup
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Download UCF-101 (~6.5 GB)
.venv/bin/python scripts/download_ucf101.py

# Build index
.venv/bin/python scripts/build_index.py

# Run pipeline (50 videos for quick test)
.venv/bin/python scripts/run_pipeline.py --subset-n 50 --num-workers 1

# Run on all ~13K videos
.venv/bin/python scripts/run_pipeline.py --num-workers 2
```

## Cross-Category Benchmark

Scored 1,456 videos across 10 diverse UCF-101 categories. Each video is processed through all 6 quality signals. Results validate that each scorer produces sensible rankings across categories with fundamentally different motion and visual characteristics.

### Per-Category Scorer Means

| Category | Videos | Temporal Consistency | Aesthetic Score | Prompt Alignment | Optical Flow | Scene Cuts | Duplicates |
|----------|-------:|---------------------:|----------------:|-----------------:|-------------:|-----------:|-----------:|
| Typing | 136 | **0.980** | 3.99 | 0.263 | **1.61** | 0 | 91 |
| SoccerPenalty | 137 | 0.978 | **4.36** | 0.261 | 2.66 | 0 | 45 |
| Drumming | 161 | 0.964 | 3.97 | 0.246 | 2.53 | 0 | 73 |
| CricketShot | 167 | 0.963 | 4.07 | 0.267 | 2.34 | 1 | 87 |
| Bowling | 155 | 0.958 | 4.02 | 0.274 | 5.90 | 4 | 13 |
| Basketball | 134 | 0.955 | 3.83 | 0.258 | 3.17 | 1 | 40 |
| RockClimbingIndoor | 144 | 0.954 | 4.02 | **0.299** | 5.91 | 4 | 2 |
| HorseRiding | 164 | 0.946 | 3.45 | 0.270 | 6.60 | 1 | 3 |
| Skijet | 100 | 0.939 | 3.64 | 0.221 | 6.90 | 0 | 0 |
| IceDancing | 158 | **0.918** | 4.08 | 0.274 | **7.07** | 1 | 0 |

### Key Observations

- **Optical flow tracks real-world motion**: Typing (1.61) < CricketShot (2.34) < Basketball (3.17) < HorseRiding (6.60) < IceDancing (7.07). Static indoor activities score low; fast outdoor sports score high.
- **Temporal consistency inversely correlates with motion**: IceDancing (0.918) has the most frame-to-frame visual change; Typing (0.980) has the least. This confirms the scorer distinguishes content stability from visual quality.
- **Aesthetic scores reflect visual composition**: SoccerPenalty (4.36) — well-lit stadiums, clean backgrounds — scores highest. HorseRiding (3.45) — variable outdoor conditions, motion blur — scores lowest.
- **Prompt alignment is highest for visually distinctive categories**: RockClimbingIndoor (0.299) has unique visual features (walls, harnesses, chalk). Skijet (0.221) is harder for CLIP to distinguish from generic water/boat scenes.
- **Scene cuts are rare in UCF-101**: Only 12 flagged across 1,456 videos. Bowling and RockClimbingIndoor (4 each) have the most, likely from camera angle changes and quick motion.
- **Duplicate detection varies by recording setup**: Typing (91/136 = 67%) — many near-identical webcam angles. Skijet and IceDancing (0 each) — diverse viewpoints and environments.

## Configuration

All settings can be set via environment variables (prefix `VQ_`) or constructor args:

| Setting | Default | Description |
|---------|---------|-------------|
| `subset_n` | None (all) | Process only first N videos |
| `batch_size` | 8 | Batch size for Ray map_batches |
| `num_workers` | 2 | Ray actor pool size |
| `frames_per_video` | 16 | Frames sampled per video |
| `phash_hamming_threshold` | 8 | pHash near-duplicate threshold |
| `min_quality_score` | 0.5 | Composite quality filter threshold |
| `clip_model_name` | ViT-L-14 | CLIP model variant |

## Scaling to GPU Cluster

The pipeline runs locally on Mac (MPS/CPU) by default. To scale to a Ray GPU cluster:

```python
# In pipeline.py, change:
gpu_kwargs = dict(
    compute=ray.data.ActorPoolStrategy(size=8),  # more workers
    batch_size=32,                                 # larger batches
    num_gpus=0.5,                                  # fractional GPU per worker
    num_cpus=1,
)
```

Ray handles scheduling, fault tolerance, and data movement. The scorer interface is unchanged — each scorer calls `get_device()` which returns CUDA on a GPU cluster.

## Design Decisions

- **Modular scorers**: Each scorer is a standalone class implementing `BaseScorer`. They load independently and can be composed in any order.
- **OpenCV for video reading**: Decord is faster but lacks Python 3.13 support. OpenCV `VideoCapture` is the universal fallback.
- **Plain `nn.Module` for aesthetic MLP**: The LAION aesthetic predictor is 5 layers of `nn.Sequential`. No need for the full pytorch-lightning dependency.
- **Farneback over RAFT**: CPU-only, zero extra dependencies, sufficient precision for quality scoring.
- **Two-phase dedup with cascade prevention**: In-actor pHash catches most duplicates during streaming; manifest builder resolves cross-actor duplicates in a second pass. Both phases only use non-duplicate hashes as reference to prevent false-positive cascading.
- **Per-category quality normalization**: Composite quality score normalizes optical flow relative to each category's median, so categories with inherently different motion levels (e.g., "Typing" vs "Basketball") are scored fairly.

## Tests

```bash
# All tests
.venv/bin/pytest tests/ -v

# Fast tests only (no model download)
.venv/bin/pytest tests/test_video.py tests/test_optical_flow.py tests/test_phash_dedup.py tests/test_manifest.py -v

# CLIP-based scorer tests (downloads ~900MB model on first run)
.venv/bin/pytest tests/test_temporal_consistency.py tests/test_aesthetic.py tests/test_prompt_alignment.py -v
```
