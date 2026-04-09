from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from training.src.models.recommendation_requests import MultiRecommendationRequest
from training.src.models.recommendation_responses import (
    HealthResponse,
    MultiRecommendationResponse,
    RecommendationResponse,
)
from training.src.services.recommendation_service import RecommendationService


MODEL_ID = os.getenv("MODEL_ID", "diana_model_v1")
S3_BUCKET = os.getenv("PREDICTIONS_S3_BUCKET")
S3_KEY = os.getenv("PREDICTIONS_S3_KEY")
LOCAL_PREDICTIONS_PATH = os.getenv("LOCAL_PREDICTIONS_PATH", "/tmp/predictions.parquet")

_service: RecommendationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service

    if not S3_BUCKET or not S3_KEY:
        raise RuntimeError("PREDICTIONS_S3_BUCKET and PREDICTIONS_S3_KEY must be set")

    _service = RecommendationService.from_s3(
        bucket=S3_BUCKET,
        key=S3_KEY,
        local_path=LOCAL_PREDICTIONS_PATH,
        model_id=MODEL_ID,
    )
    yield


app = FastAPI(
    title="Diana Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _service is None:
        return HealthResponse(status="loading")
    return HealthResponse(status="ok")


@app.get("/recommendations", response_model=list[RecommendationResponse])
def get_recommendations(
    anchorId: str = Query(...),
    kioskId: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
) -> list[RecommendationResponse]:
    if _service is None:
        raise RuntimeError("Service not initialized")

    items = _service.get_recommendations(
        anchor_id=anchorId,
        kiosk_id=kioskId,
        limit=limit,
    )
    return [RecommendationResponse(**item.__dict__) for item in items]


@app.post("/recommendations/multi", response_model=list[MultiRecommendationResponse])
def get_multi_recommendations(
    requests: list[MultiRecommendationRequest],
) -> list[MultiRecommendationResponse]:
    if _service is None:
        raise RuntimeError("Service not initialized")

    items = _service.get_multi_recommendations(
        [request.model_dump() for request in requests]
    )
    return [MultiRecommendationResponse(**item.__dict__) for item in items]