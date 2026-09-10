# src/predict.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from backend.config import (
    CLASS_MAPPING,
    EXPECTED_FEATURE_COUNT,
    MODEL_PATH,
)
from backend.services.feature_builder import (
    prepare_model_input,
)


# ============================================================
# LOAD TRAINED MODEL PACKAGE
# ============================================================

def load_model_package(
    model_path=MODEL_PATH,
):
    """
    Load and validate the trained Random Forest package.

    Expected package contents:

        model
        features
        class_mapping
        train_medians
        classes
        model_type
        random_state
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model package not found:\n"
            f"{model_path}"
        )

    package = joblib.load(
        model_path
    )

    if not isinstance(
        package,
        dict,
    ):
        raise RuntimeError(
            "Loaded model package is not a dictionary."
        )

    # --------------------------------------------------------
    # Required package keys
    # --------------------------------------------------------

    required_keys = [
        "model",
        "features",
        "train_medians",
        "classes",
    ]

    missing = [
        key
        for key in required_keys
        if key not in package
    ]

    if missing:
        raise RuntimeError(
            "Model package is missing required keys:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing
            )
        )

    model = package["model"]
    features = list(
        package["features"]
    )
    train_medians = package[
        "train_medians"
    ]
    classes = np.asarray(
        package["classes"]
    )

    # --------------------------------------------------------
    # Feature schema validation
    # --------------------------------------------------------

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Stored feature list has "
            f"{len(features)} features; "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    if len(set(features)) != len(features):
        raise RuntimeError(
            "Stored feature list contains duplicate "
            "feature names."
        )

    # --------------------------------------------------------
    # Validate trained model feature count
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_",
    ):

        model_feature_count = (
            int(
                model.n_features_in_
            )
        )

        if (
            model_feature_count
            != EXPECTED_FEATURE_COUNT
        ):
            raise RuntimeError(
                "Trained model expects "
                f"{model_feature_count} features; "
                f"expected {EXPECTED_FEATURE_COUNT}."
            )

    # --------------------------------------------------------
    # Validate model classes against package classes
    # --------------------------------------------------------

    if hasattr(
        model,
        "classes_",
    ):

        model_classes = np.asarray(
            model.classes_
        )

        if not np.array_equal(
            model_classes,
            classes,
        ):
            raise RuntimeError(
                "Model classes_ do not match "
                "the classes stored in the model package.\n"
                f"Model classes: {model_classes}\n"
                f"Package classes: {classes}"
            )

    class_mapping = package.get(
        "class_mapping",
        CLASS_MAPPING,
    )

    return {
        "model": model,
        "features": features,
        "train_medians": train_medians,
        "classes": classes,
        "class_mapping": class_mapping,
        "model_type": package.get(
            "model_type"
        ),
        "random_state": package.get(
            "random_state"
        ),
    }


# ============================================================
# PREDICT FIRE SOURCE
# ============================================================

def predict_fire_source(
    feature_dict: dict,
    model_package: dict,
):
    """
    Prepare the exact 74-feature model input and
    run the trained Random Forest.

    Returns a prediction dictionary containing:

        predicted_class
        predicted_class_name
        confidence
        probabilities
        feature_count
    """

    # --------------------------------------------------------
    # Extract trained objects
    # --------------------------------------------------------

    model = model_package[
        "model"
    ]

    model_features = list(
        model_package["features"]
    )

    train_medians = model_package[
        "train_medians"
    ]

    class_mapping = model_package[
        "class_mapping"
    ]

    # --------------------------------------------------------
    # Verify exact feature schema
    # --------------------------------------------------------

    if len(model_features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Model feature schema contains "
            f"{len(model_features)} features; "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    # --------------------------------------------------------
    # Build exact Cell 2C model input
    # --------------------------------------------------------

    X_live = prepare_model_input(
        feature_dict=feature_dict,
        model_features=model_features,
        train_medians=train_medians,
    )

    # --------------------------------------------------------
    # Final shape validation
    # --------------------------------------------------------

    if X_live.shape != (
        1,
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            "Unexpected live model-input shape: "
            f"{X_live.shape}. "
            f"Expected (1, {EXPECTED_FEATURE_COUNT})."
        )

    # --------------------------------------------------------
    # Final finite-value validation
    # --------------------------------------------------------

    X_values = X_live.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        X_values
    ).all():

        raise RuntimeError(
            "Final model input contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Model feature-count validation
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_",
    ):

        if (
            int(model.n_features_in_)
            != X_live.shape[1]
        ):
            raise RuntimeError(
                "Model and live input feature counts do not match."
            )

    # ========================================================
    # PREDICTION
    # ========================================================

    predicted_class = model.predict(
        X_live
    )[0]

    # Convert numpy integer to normal Python int
    try:
        predicted_class_int = int(
            predicted_class
        )
    except (
        TypeError,
        ValueError,
    ):
        predicted_class_int = predicted_class

    # --------------------------------------------------------
    # Validate predicted class
    # --------------------------------------------------------

    if hasattr(
        model,
        "classes_",
    ):

        model_classes = np.asarray(
            model.classes_
        )

        if not np.any(
            model_classes
            == predicted_class
        ):
            raise RuntimeError(
                "Model returned a class that is "
                "not present in model.classes_."
            )

    # --------------------------------------------------------
    # Class name
    # --------------------------------------------------------

    predicted_class_name = (
        class_mapping.get(
            predicted_class_int,
            f"Class {predicted_class_int}",
        )
    )

    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    probability_dict = {}
    probabilities = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model
            .predict_proba(
                X_live
            )[0]
        )

        model_classes = np.asarray(
            model.classes_
        )

        if len(probabilities) != len(
            model_classes
        ):
            raise RuntimeError(
                "Probability count does not match "
                "model class count."
            )

        # ----------------------------------------------------
        # Build class-probability mapping
        # ----------------------------------------------------

        for class_value, probability in zip(
            model_classes,
            probabilities,
        ):

            try:
                class_int = int(
                    class_value
                )
            except (
                TypeError,
                ValueError,
            ):
                class_int = class_value

            class_name = (
                class_mapping.get(
                    class_int,
                    f"Class {class_int}",
                )
            )

            probability_dict[
                class_name
            ] = float(
                probability
            )

        # ----------------------------------------------------
        # Top-class confidence
        # ----------------------------------------------------

        predicted_probability = float(
            np.max(
                probabilities
            )
        )

    else:

        predicted_probability = None

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "predicted_class":
            predicted_class_int,

        "predicted_class_name":
            predicted_class_name,

        "confidence":
            predicted_probability,

        "probabilities":
            probability_dict,

        "feature_count":
            int(X_live.shape[1]),
    }


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def print_prediction(
    prediction: dict,
):
    """
    Print a clean final classification summary.
    """

    print("\n" + "=" * 60)
    print(
        "FIRE SOURCE CLASSIFICATION"
    )
    print("=" * 60)

    print(
        f"Predicted source : "
        f"{prediction['predicted_class_name']}"
    )

    confidence = prediction.get(
        "confidence"
    )

    if confidence is not None:

        print(
            f"Confidence       : "
            f"{confidence * 100:.2f}%"
        )

    else:

        print(
            "Confidence       : unavailable"
        )

    print(
        f"Feature count    : "
        f"{prediction['feature_count']}"
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    if probabilities:

        print(
            "\nClass probabilities:"
        )

        sorted_probabilities = sorted(
            probabilities.items(),
            key=lambda item:
            item[1],
            reverse=True,
        )

        for (
            class_name,
            probability,
        ) in sorted_probabilities:

            print(
                f"  {class_name:22s} "
                f"{probability * 100:8.2f}%"
            )

    print("=" * 60)

    print(
        "\nNOTE:"
    )

    print(
        "This is a predicted source class "
        "from the 2025-trained Random Forest."
    )

    print(
        "It is NOT a NASA FIRMS source label."
    )