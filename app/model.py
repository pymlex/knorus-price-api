from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class LoadedModel:
    model_name: str
    raw_model: Any
    target_transform: str

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        pred = self.raw_model.predict(features)
        return np.asarray(pred, dtype=np.float32)


def load_artifacts(bundle_path: Path, model_path: Path) -> tuple[dict[str, Any], LoadedModel]:
    bundle = joblib.load(bundle_path)
    model = joblib.load(model_path)
    loaded = LoadedModel(
        model_name=type(model).__name__,
        raw_model=model,
        target_transform=str(bundle.get("target_transform", "log1p")),
    )
    return bundle, loaded