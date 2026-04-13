"""Per-category manifest builder from scored Parquet output."""

import imagehash
import pandas as pd

from vq.config import Settings


def _resolve_cross_actor_duplicates(df: pd.DataFrame, hamming_threshold: int) -> pd.DataFrame:
    """Second-pass dedup: catch duplicates that were split across Ray actors."""
    dup_mask = df["is_near_duplicate"].copy()

    for _category, group in df.groupby("category"):
        hashes = []
        for idx, row in group.iterrows():
            if not row["phash"] or pd.isna(row["phash"]):
                continue
            h = imagehash.hex_to_hash(row["phash"])
            is_dup = any((h - existing) <= hamming_threshold for existing in hashes)
            if is_dup:
                dup_mask.at[idx] = True
            else:
                # Only use non-duplicate hashes as reference to avoid cascading
                hashes.append(h)

    df["is_near_duplicate"] = dup_mask
    return df


def _compute_composite_quality(df: pd.DataFrame) -> pd.Series:
    """Weighted combination of normalized quality signals."""

    def _norm(series: pd.Series) -> pd.Series:
        s = series.fillna(0.0)
        smin, smax = s.min(), s.max()
        if smax - smin < 1e-8:
            return pd.Series(0.5, index=s.index)
        return (s - smin) / (smax - smin)

    temporal = _norm(df["temporal_consistency"])
    aesthetic = _norm(df["aesthetic_score"])
    alignment = _norm(df["prompt_alignment"])

    # Optical flow: penalize both too-low (static) and too-high (blur)
    # Use per-category median so categories with different motion levels aren't biased
    flow = df["optical_flow_mean"].fillna(0.0)
    category_median = df.groupby("category")["optical_flow_mean"].transform("median").fillna(0.0)
    flow_score = 1.0 - _norm((flow - category_median).abs())

    return (
        0.30 * temporal
        + 0.25 * aesthetic
        + 0.25 * alignment
        + 0.20 * flow_score
    )


def build_manifest(settings: Settings | None = None) -> pd.DataFrame:
    """
    Read scored Parquet, compute per-category aggregate statistics,
    resolve cross-actor duplicates, and produce the manifest.
    """
    settings = settings or Settings()
    scored_dir = settings.output_dir / "scored"
    df = pd.read_parquet(scored_dir)

    # Phase 1: Cross-actor dedup resolution
    df = _resolve_cross_actor_duplicates(df, settings.phash_hamming_threshold)

    # Phase 2: Composite quality score
    df["quality_score"] = _compute_composite_quality(df)

    # Phase 3: Per-category aggregation
    manifest = (
        df.groupby("category")
        .agg(
            count=("video_id", "count"),
            mean_temporal_consistency=("temporal_consistency", "mean"),
            mean_aesthetic_score=("aesthetic_score", "mean"),
            mean_prompt_alignment=("prompt_alignment", "mean"),
            mean_optical_flow=("optical_flow_mean", "mean"),
            duplicate_count=("is_near_duplicate", "sum"),
            mean_quality=("quality_score", "mean"),
            p50_quality=("quality_score", "median"),
        )
        .reset_index()
    )

    # Phase 4: Filtered sample IDs
    filtered = df[
        (df["quality_score"] >= settings.min_quality_score) & (~df["is_near_duplicate"])
    ]
    filtered_ids = (
        filtered.groupby("category")["video_id"]
        .apply(list)
        .reset_index(name="filtered_sample_ids")
    )
    manifest = manifest.merge(filtered_ids, on="category", how="left")
    manifest["filtered_count"] = manifest["filtered_sample_ids"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Write outputs
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(settings.output_dir / "manifest.parquet")
    manifest.drop(columns=["filtered_sample_ids"]).to_csv(
        settings.output_dir / "manifest.csv", index=False
    )

    return manifest
