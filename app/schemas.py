from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BookRecord(BaseModel):
    title: str = Field(..., description="Book title")
    annotation: Optional[str] = Field(default=None, description="Book annotation")
    year: Optional[float] = Field(default=None, description="Publication year")
    grif: Optional[str] = Field(default=None, description="Grif text")
    edition_type: Optional[str] = Field(default=None, description="Edition type")
    series: Optional[str] = Field(default=None, description="Semicolon-separated series labels")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    discipline: Optional[str] = Field(default=None, description="Semicolon-separated discipline labels")
    theme: Optional[str] = Field(default=None, description="Semicolon-separated theme labels")
    pages: Optional[float] = Field(default=None, description="Page count")
    cover: Optional[str] = Field(default=None, description="Cover type")
    format_text: Optional[str] = Field(default=None, description="Format string like 60х90/16")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_price: float
    predicted_log_price: Optional[float] = None
    model_name: str
    target_transform: str


class BatchPredictionRequest(BaseModel):
    items: List[BookRecord]