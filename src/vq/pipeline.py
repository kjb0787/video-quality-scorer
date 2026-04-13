"""Ray Data pipeline orchestration — chains all quality scorers."""

import ray
import ray.data

from vq.config import Settings
from vq.scorers.aesthetic import AestheticScorer
from vq.scorers.optical_flow import OpticalFlowScorer
from vq.scorers.phash_dedup import PHashDedupScorer
from vq.scorers.prompt_alignment import PromptAlignmentScorer
from vq.scorers.temporal_consistency import TemporalConsistencyScorer


def run_pipeline(settings: Settings | None = None) -> str:
    """
    Execute the full quality scoring pipeline.

    1. Read index Parquet
    2. Optionally slice to subset_n rows
    3. Chain scorers via map_batches
    4. Write scored Parquet

    Returns the output directory path.
    """
    settings = settings or Settings()
    ray.init(num_cpus=settings.ray_num_cpus, ignore_reinit_error=True)

    try:
        ds = ray.data.read_parquet(str(settings.index_path))

        if ds.count() == 0:
            raise ValueError(
                f"Index Parquet at {settings.index_path} is empty — nothing to process."
            )

        if settings.subset_n is not None:
            ds = ds.limit(settings.subset_n)

        # On Mac: num_gpus=0 because Ray doesn't see MPS.
        # Scorers use get_device() internally for MPS/CUDA/CPU.
        # On a real GPU cluster, set num_gpus=1 (or 0.5 for fractional sharing).
        compute = ray.data.ActorPoolStrategy(size=settings.num_workers)
        gpu_kwargs = dict(
            compute=compute,
            batch_size=settings.batch_size,
            num_gpus=0,
            num_cpus=1,
        )
        cpu_kwargs = dict(
            compute=ray.data.ActorPoolStrategy(size=settings.num_workers),
            batch_size=settings.batch_size,
            num_gpus=0,
            num_cpus=1,
        )

        ds = (
            ds
            .map_batches(TemporalConsistencyScorer, **gpu_kwargs)
            .map_batches(AestheticScorer, **gpu_kwargs)
            .map_batches(PromptAlignmentScorer, **gpu_kwargs)
            .map_batches(OpticalFlowScorer, **cpu_kwargs)
            .map_batches(PHashDedupScorer, **cpu_kwargs)
        )

        output_path = settings.output_dir / "scored"
        # Remove stale output to prevent Parquet directory accumulation
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        ds.write_parquet(str(output_path))
        return str(output_path)
    finally:
        ray.shutdown()
