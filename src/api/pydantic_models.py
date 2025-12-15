from pydantic import BaseModel
from typing import Optional


class CustomerFeatures(BaseModel):
    # Put only a few key features for now; you can extend later
    total_amount: float
    avg_amount: float
    std_amount: float
    tx_count: int
    Recency: float
    Frequency: float
    Monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: Optional[int] = None
