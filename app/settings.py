from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "knorus-price-api"
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    bundle_path: Path = Path(os.getenv("BUNDLE_PATH", "artifacts/preprocess_bundle.joblib"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/final_model.joblib"))
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "Qwen/Qwen3.5-0.8B-Base")
    embedding_cache_path: Path = Path(os.getenv("EMBEDDING_CACHE_PATH", "artifacts/cache/embeddings.sqlite"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    max_title_annotation_length: int = int(os.getenv("MAX_TITLE_ANN_LEN", "512"))
    max_disc_theme_length: int = int(os.getenv("MAX_DISC_THEME_LEN", "96"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    target_transform: str = os.getenv("TARGET_TRANSFORM", "log1p")


settings = Settings()