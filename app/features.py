from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd


def normalize_spaces(text: object) -> str:
    if text is None:
        return ""
    value = str(text)
    if value.strip().lower() == "nan":
        return ""
    value = value.replace("\n", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_core_title(text: object) -> str:
    value = normalize_spaces(text)
    if not value:
        return ""
    base = value.rstrip(".").strip()
    if base.count(".") >= 2:
        return base.rsplit(".", 2)[0].strip()
    return base


def compose_title_annotation(title: object, annotation: object) -> str:
    title_value = extract_core_title(title)
    annotation_value = normalize_spaces(annotation)
    parts = []
    if title_value:
        parts.append(f"Название: {title_value}")
    if annotation_value:
        parts.append(f"Аннотация: {annotation_value}")
    return " \n ".join(parts)


def split_labels(text: object) -> List[str]:
    value = normalize_spaces(text)
    if not value:
        return []
    labels = [normalize_spaces(item) for item in value.split(";")]
    return [item for item in labels if item]


def parse_format(text: object) -> List[float]:
    value = normalize_spaces(text).lower().replace("x", "х").replace("*", "х").replace(" ", "")
    if not value:
        return [np.nan, np.nan, np.nan]
    match = re.match(r"^(\d+)х(\d+)/(\d+)$", value)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return [np.nan, np.nan, np.nan]


def grif_to_bin(text: object) -> int:
    value = normalize_spaces(text).lower()
    if not value or value == "без грифа":
        return 0
    return 1


def cover_to_bin(text: object) -> float:
    value = normalize_spaces(text).lower()
    if value == "переплет":
        return 1.0
    if value == "обложка":
        return 0.0
    return np.nan


def build_series_vector(labels: Iterable[str], index: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(index), dtype=np.float32)
    for label in labels:
        pos = index.get(label)
        if pos is not None:
            vector[pos] = 1.0
    return vector


def as_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)