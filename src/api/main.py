from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(title="Credit Risk Model API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    # For now, use a dummy rule: higher total_amount + frequency -> lower risk
    score = 0.5  # dummy base probability

    if features.tx_count > 5 and features.total_amount > 1000:
        score = 0.2
    elif features.tx_count < 2 and features.total_amount < 100:
        score = 0.8

    # Clamp between 0 and 1
    score = max(0.0, min(1.0, score))

    return PredictionResponse(
        risk_probability=score,
        is_high_risk=int(score > 0.5),
    )
