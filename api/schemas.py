from pydantic import BaseModel


class PredictResponse(BaseModel):
    probability: float
    label: str
    threshold: float
    heatmap_base64: str


class HealthResponse(BaseModel):
    status: str