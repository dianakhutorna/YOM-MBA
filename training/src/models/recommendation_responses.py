from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class RecommendationResponse(BaseModel):
    anchor_id: str
    kiosk_id: str
    product_id: str
    model_id: str
    recommendation_date: str


class MultiRecommendationResponse(BaseModel):
    anchor_id: str
    kiosk_id: str
    recs: list[str]
    model_id: str
    recommendation_date: str