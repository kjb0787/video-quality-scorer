# Video Quality Scorer

Ray-based pipeline for scoring video training data quality. Computes 5 quality signals per video and produces a per-category manifest for ML training data curation.

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

## Sample Output

### Scored Data (per-video)

| video_id | temporal_consistency | aesthetic_score | prompt_alignment | optical_flow_mean | is_near_duplicate |
|----------|---------------------|-----------------|------------------|-------------------|-------------------|
| ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 | 0.97 | 3.9 | 0.29 | 3.8 | False |
| ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 | 0.98 | 4.1 | 0.30 | 2.2 | False |
| ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 | 0.97 | 4.0 | 0.28 | 3.4 | False |

### Category Manifest

| category | count | mean_quality | p50_quality | filtered_count | duplicate_count |
|----------|-------|-------------|-------------|----------------|-----------------|
| ApplyEyeMakeup | 145 | 0.59 | 0.59 | 91 | 29 |
| ApplyLipstick | 55 | 0.49 | 0.48 | 14 | 15 |

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
