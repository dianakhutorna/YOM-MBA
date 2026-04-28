from pydantic import BaseModel


class MultiRecommendationRequest(BaseModel):
    anchor_id: str
    kiosk_id: str