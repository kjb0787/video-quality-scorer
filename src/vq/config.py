from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Path("")
    ucf101_dir: Path = Path("")
    index_path: Path = Path("")
    output_dir: Path = Path("")
    model_cache_dir: Path = Path("")

    # Pipeline
    subset_n: int | None = None
    batch_size: int = 8
    num_workers: int = 2
    ray_num_cpus: int | None = None

    # Frame extraction
    frames_per_video: int = 16
    frame_size: tuple[int, int] = (224, 224)

    # Scorer thresholds
    phash_hamming_threshold: int = 8
    min_quality_score: float = 0.5

    # Model
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    aesthetic_weights_url: str = (
        "https://github.com/christophschuhmann/improved-aesthetic-predictor"
        "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
    )
    aesthetic_weights_filename: str = "sac+logos+ava1-l14-linearMSE.pth"

    model_config = {"env_prefix": "VQ_"}

    def model_post_init(self, __context) -> None:
        if self.data_dir == Path(""):
            self.data_dir = self.project_root / "data"
        if self.ucf101_dir == Path(""):
            self.ucf101_dir = self.data_dir / "UCF-101"
        if self.index_path == Path(""):
            self.index_path = self.data_dir / "index.parquet"
        if self.output_dir == Path(""):
            self.output_dir = self.project_root / "output"
        if self.model_cache_dir == Path(""):
            self.model_cache_dir = self.project_root / "models"
