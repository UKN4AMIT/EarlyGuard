from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.schemas import KidneyPatientInput, KidneyPredictionResponse
from app.predictor import predict_kidney_disease

router = APIRouter(prefix="/api", tags=["prediction"])

@router.post("/predict/kidney", response_model=KidneyPredictionResponse)
def predict_kidney(payload: KidneyPatientInput):
    try:
        return predict_kidney_disease(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
