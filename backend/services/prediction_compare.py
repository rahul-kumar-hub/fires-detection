from __future__ import annotations

from typing import Any

from backend.inference.model1 import (
    load_model_package,
    predict_fire_source,
)

from backend.inference.model2 import run_model2
from backend.services.model1_runner import run_model1


def _extract_model1_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the existing Model 1 response into a common format.
    """

    prediction = result.get("prediction", {})

    if not isinstance(prediction, dict):
        prediction = {}

    predicted_class = (
        prediction.get("predicted_class_name")
        or prediction.get("class_name")
        or prediction.get("predicted_class")
        or "Unknown"
    )

    confidence = float(
        prediction.get("confidence", 0.0)
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    return {
        "predicted_class": str(predicted_class),
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_result": result,
    }


def _extract_model2_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the existing Model 2 response into a common format.
    """

    predicted_class = (
        result.get("final_class")
        or result.get("class_name")
        or "Unknown"
    )

    confidence = float(
        result.get("confidence", 0.0)
    )

    probabilities = result.get(
        "probabilities",
        {},
    )

    return {
        "predicted_class": str(predicted_class),
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_result": result,
    }


def run_prediction_comparison(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """
    Run Model 1 and Model 2 independently on the same location.

    The model with the highest confidence is selected as
    the final prediction.
    """

    # --------------------------------------------------------
    # Run Model 1
    # --------------------------------------------------------

    model1_raw = run_model1(
        latitude=latitude,
        longitude=longitude,
    )

    model1 = _extract_model1_result(
        model1_raw
    )

    # --------------------------------------------------------
    # Run Model 2
    # --------------------------------------------------------

    model2_raw = run_model2(
        latitude=latitude,
        longitude=longitude,
    )

    model2 = _extract_model2_result(
        model2_raw
    )

    # --------------------------------------------------------
    # Compare confidence
    # --------------------------------------------------------

    if (
        model1["confidence"]
        >= model2["confidence"]
    ):
        selected_model = "model1"
        final_prediction = model1
    else:
        selected_model = "model2"
        final_prediction = model2

    return {
        "success": True,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "model1": model1,
        "model2": model2,
        "final_prediction": {
            "predicted_class": final_prediction[
                "predicted_class"
            ],
            "confidence": final_prediction[
                "confidence"
            ],
            "selected_model": selected_model,
            "reason": (
                f"{selected_model.upper()} produced "
                "the highest confidence."
            ),
        },
    }
