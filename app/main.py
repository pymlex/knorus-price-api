from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from .schemas import BatchPredictionRequest, BookRecord, PredictionResponse
from .service import PredictionService
from .settings import settings


service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = PredictionService(
        bundle_path=settings.bundle_path,
        model_path=settings.model_path,
        embedding_model_id=settings.embedding_model_id,
        embedding_cache_path=settings.embedding_cache_path,
        max_title_annotation_length=settings.max_title_annotation_length,
        max_disc_theme_length=settings.max_disc_theme_length,
        embed_batch_size=settings.embed_batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    try:
        yield
    finally:
        if service is not None:
            service.close()
            service = None


app = FastAPI(title="Knorus Price API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(item: BookRecord) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    result = service.predict_one(item.model_dump())
    return PredictionResponse(**result)


@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    results = service.predict_many([item.model_dump() for item in request.items])
    return {"results": results}