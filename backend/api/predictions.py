from fastapi import APIRouter, HTTPException

from backend.schemas.prediction import PredictionRequest
from backend.services.model1_runner import run_model1
from backend.inference.model2 import run_model2
from backend.services.prediction_compare import (
    run_prediction_comparison,
)


router = APIRouter(
    prefix="/api/predict",
    tags=["Predictions"],
)


@router.post("/model1")
def predict_model1(
    request: PredictionRequest,
):
    """
    Run the existing Model 1 pipeline.
    """

    try:
        result = run_model1(
            latitude=request.latitude,
            longitude=request.longitude,
        )

        return {
            "success": True,
            "model": "model1",
            "result": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@router.post("/model2")
def predict_model2(
    request: PredictionRequest,
):
    """
    Run the existing Model 2 pipeline.
    """

    try:
        result = run_model2(
            latitude=request.latitude,
            longitude=request.longitude,
        )

        return {
            "success": True,
            "model": "model2",
            "result": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@router.post("/compare")
def compare_predictions(
    request: PredictionRequest,
):
    """
    Run Model 1 and Model 2 independently on the same
    latitude and longitude.

    The final prediction is selected using the highest
    confidence score.
    """

    try:
        return run_prediction_comparison(
            latitude=request.latitude,
            longitude=request.longitude,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error