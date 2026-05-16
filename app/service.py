from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .embedders import TextEmbedder
from .features import (
    as_frame,
    build_series_vector,
    compose_title_annotation,
    cover_to_bin,
    extract_core_title,
    grif_to_bin,
    normalize_spaces,
    parse_format,
    split_labels,
)
from .model import load_artifacts


def _apply_target_inverse(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(values)
    return values


@dataclass
class PredictionService:
    bundle_path: Path
    model_path: Path
    embedding_model_id: str
    embedding_cache_path: Path
    max_title_annotation_length: int
    max_disc_theme_length: int
    embed_batch_size: int
    device: torch.device

    def __post_init__(self) -> None:
        self.bundle, self.loaded_model = load_artifacts(
            self.bundle_path,
            self.model_path,
        )
        self.embedder = TextEmbedder(
            model_id=self.embedding_model_id,
            cache_path=self.embedding_cache_path,
            max_length=self.max_title_annotation_length,
            batch_size=self.embed_batch_size,
        )

    def close(self) -> None:
        self.embedder.close()

    def _numeric_block(self, item: dict) -> np.ndarray:
        format_w, format_h, format_n = parse_format(item.get("format_text"))
        values = {
            "Год издания": item.get("year"),
            "Кол-во страниц": item.get("pages"),
            "format_w": format_w,
            "format_h": format_h,
            "format_n": format_n,
        }
        row = [[values.get(col) for col in self.bundle["numeric_cols"]]]
        matrix = np.asarray(row, dtype=np.float32)
        matrix = self.bundle["num_imputer"].transform(matrix)
        matrix = self.bundle["num_scaler"].transform(matrix)
        return matrix.astype(np.float32)

    def _categorical_block(self, item: dict) -> np.ndarray:
        edition_value = normalize_spaces(item.get("edition_type")) or "__missing__"
        publisher_value = normalize_spaces(item.get("publisher")) or "__missing__"

        edition = as_frame([{"Вид издания": edition_value}])
        publisher = as_frame([{"Издательство": publisher_value}])

        edition_ohe = self.bundle["edition_encoder"].transform(edition).astype(np.float32)
        publisher_ohe = self.bundle["publisher_encoder"].transform(publisher).astype(np.float32)

        grif_text = normalize_spaces(item.get("grif"))
        if grif_text:
            grif_value = float(grif_to_bin(grif_text))
        else:
            grif_value = float(self.bundle["grif_fill_value"])

        cover_text = normalize_spaces(item.get("cover"))
        cover_value = cover_to_bin(cover_text)
        if np.isnan(cover_value):
            cover_value = float(self.bundle["cover_fill_value"])

        grif = np.asarray([[grif_value]], dtype=np.float32)
        cover = np.asarray([[cover_value]], dtype=np.float32)

        series_labels = split_labels(item.get("series"))
        series_vector = build_series_vector(series_labels, self.bundle["series_index"]).reshape(1, -1)

        return np.hstack([edition_ohe, publisher_ohe, grif, cover, series_vector]).astype(np.float32)

    def _semantic_blocks(self, item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        title = extract_core_title(item.get("title"))
        annotation = normalize_spaces(item.get("annotation"))
        title_annotation_text = compose_title_annotation(title, annotation)

        title_vector = self.embedder.embed_single(title_annotation_text).reshape(1, -1)
        title_vector = self.bundle["title_pca"].transform(title_vector).astype(np.float32)

        discipline_labels = split_labels(item.get("discipline"))
        theme_labels = split_labels(item.get("theme"))

        if discipline_labels:
            discipline_vectors = self.embedder.embed_texts(discipline_labels).mean(axis=0, keepdims=True).astype(np.float32)
        else:
            discipline_vectors = np.zeros((1, self.bundle["discipline_pca"].n_features_in_), dtype=np.float32)

        if theme_labels:
            theme_vectors = self.embedder.embed_texts(theme_labels).mean(axis=0, keepdims=True).astype(np.float32)
        else:
            theme_vectors = np.zeros((1, self.bundle["theme_pca"].n_features_in_), dtype=np.float32)

        discipline_vector = self.bundle["discipline_pca"].transform(discipline_vectors).astype(np.float32)
        theme_vector = self.bundle["theme_pca"].transform(theme_vectors).astype(np.float32)

        return title_vector, discipline_vector, theme_vector

    def build_features(self, item: dict) -> np.ndarray:
        numeric = self._numeric_block(item)
        categorical = self._categorical_block(item)
        title_vector, discipline_vector, theme_vector = self._semantic_blocks(item)
        return np.hstack([numeric, categorical, title_vector, discipline_vector, theme_vector]).astype(np.float32)

    def predict_one(self, item: dict) -> dict:
        features = self.build_features(item)
        raw_pred = self.loaded_model.predict_raw(features).reshape(-1)
        price = _apply_target_inverse(raw_pred, self.loaded_model.target_transform)[0]
        return {
            "predicted_price": float(price),
            "predicted_log_price": float(raw_pred[0]),
            "model_name": self.loaded_model.model_name,
            "target_transform": self.loaded_model.target_transform,
        }

    def predict_many(self, items: list[dict]) -> list[dict]:
        return [self.predict_one(item) for item in items]