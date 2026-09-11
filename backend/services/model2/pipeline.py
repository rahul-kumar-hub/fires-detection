from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config
from .firms import fetch_current_firms, fetch_historical_firms
from .events import build_event_context
from .history import prepare_historical_firms
from .temporal import build_temporal_features
from .event_features import build_event_features
from .flare import build_flare_features
from .industrial import load_industrial_evidence, build_industrial_features
from .mining import load_mining_evidence, build_mining_features
from .agriculture import load_agriculture_evidence, build_agriculture_features
from .wildfire import load_wildfire_evidence, build_wildfire_features
from .osm import build_osm_features, OSM_CATEGORIES
from .dynamic_world import (
    build_dynamic_world_features,
    aggregate_dynamic_world_features,
)
from .features import (
    build_dynamic_world_base_features,
    combine_features,
    build_model_matrix,
    feature_quality,
    apply_imputer,
)


def run_model2_pipeline(
    latitude: float,
    longitude: float,
    fire_datetime,
    firms_map_key: str | None = None,
) -> dict:

    map_key = (
        firms_map_key
        or os.getenv("NASA_FIRMS_MAP_KEY")
    )

    if not map_key:
        raise RuntimeError(
            "NASA FIRMS MAP KEY is required."
        )

    # ------------------------------------------------------------
    # 1. CURRENT FIRMS
    # ------------------------------------------------------------

    firms_df, bbox, current_requests = fetch_current_firms(
        map_key=map_key,
        latitude=latitude,
        longitude=longitude,
        current_days=config.CURRENT_DAYS,
    )

    if firms_df is None or len(firms_df) == 0:
        raise RuntimeError(
            "No current FIRMS detections found."
        )

    # ------------------------------------------------------------
    # 2. EVENT CONSTRUCTION
    # ------------------------------------------------------------

    selected = firms_df.iloc[0]

    selected_datetime = selected["acq_datetime"]

    event_df = build_event_context(
        firms_df,
        selected,
        selected_datetime,
    )

    if event_df is None or len(event_df) == 0:
        raise RuntimeError(
            "No FIRMS event could be constructed."
        )

    # ------------------------------------------------------------
    # 3. HISTORICAL FIRMS
    # ------------------------------------------------------------

    historical_df, historical_requests, total_historical_requests = (
        fetch_historical_firms(
            map_key=map_key,
            bbox=bbox,
            selected_date=pd.Timestamp(selected_datetime).normalize(),
        )
    )

    historical_df = prepare_historical_firms(
        historical_df=historical_df,
        selected=selected,
        selected_datetime=selected_datetime,
    )

    # ------------------------------------------------------------
    # 4. TEMPORAL FEATURES
    # ------------------------------------------------------------

    temporal_features = build_temporal_features(
        historical_df=historical_df,
        selected=selected,
        selected_datetime=selected_datetime,
        history_available=historical_requests > 0,
    )

    # ------------------------------------------------------------
    # 5. EVENT FEATURES
    # ------------------------------------------------------------

    event_features = build_event_features(
        event_df
    )

    # ------------------------------------------------------------
    # 6. WORLD BANK FLARE
    # ------------------------------------------------------------

    flare_features = build_flare_features(
        selected=selected,
    )

    # ------------------------------------------------------------
    # 7. INDUSTRIAL
    # ------------------------------------------------------------

    industrial_df = load_industrial_evidence()

    industrial_features = (
        build_industrial_features(
            industrial_df=industrial_df,
            event_df=event_df,
        )
    )

    # ------------------------------------------------------------
    # 8. MINING
    # ------------------------------------------------------------

    mining_df = load_mining_evidence()

    mining_features = build_mining_features(
        mining_df=mining_df,
        event_df=event_df,
    )

    # ------------------------------------------------------------
    # 9. AGRICULTURE
    # ------------------------------------------------------------

    agriculture_df = load_agriculture_evidence()

    # ------------------------------------------------------------
    # 10. OSM
    # ------------------------------------------------------------

    osm_nearest, agri_osm_distance = (
        build_osm_features(
            latitude=latitude,
            longitude=longitude,
        )
    )

    agriculture_features = (
        build_agriculture_features(
            agriculture_df=agriculture_df,
            event_df=event_df,
            agri_osm_distance=agri_osm_distance,
        )
    )

    # ------------------------------------------------------------
    # 11. WILDFIRE
    # ------------------------------------------------------------

    wildfire_df = load_wildfire_evidence()

    wildfire_features = build_wildfire_features(
        wildfire_df=wildfire_df,
        event_df=event_df,
    )

    # ------------------------------------------------------------
    # 12. DYNAMIC WORLD
    # ------------------------------------------------------------

    dw_df = build_dynamic_world_features(
        event_df=event_df,
        project=config.EE_PROJECT,
        area_m=config.DW_AREA_M,
        search_before_days=config.DW_SEARCH_BEFORE_DAYS,
        search_after_days=config.DW_SEARCH_AFTER_DAYS,
        bands=config.DW_BANDS,
        class_names=config.DW_CLASS_NAMES,
    )

    dw_event_features = (
        aggregate_dynamic_world_features(
            dw_df
        )
    )

    dw_base_features = (
        build_dynamic_world_base_features(
            dw_df=dw_df,
            bands=config.DW_BANDS,
        )
    )

    # ------------------------------------------------------------
    # 13. COMBINE ALL FEATURES
    # ------------------------------------------------------------

    features = combine_features(
        event_features=event_features,
        flare_features=flare_features,
        industrial_features=industrial_features,
        temporal_features=temporal_features,
        mining_features=mining_features,
        agriculture_features=agriculture_features,
        wildfire_features=wildfire_features,
        dw_event_features=dw_event_features,
        dw_base_features=dw_base_features,
        osm_nearest=osm_nearest,
        agriculture_osm_distance=agri_osm_distance,
        osm_categories=OSM_CATEGORIES,
    )

    # ------------------------------------------------------------
    # 14. EXACT 137-FEATURE MATRIX
    # ------------------------------------------------------------

    model_features = config.MODEL_FEATURES

    X_test = build_model_matrix(
        features=features,
        model_features=model_features,
    )

    quality = feature_quality(
        X_test=X_test,
        model_features=model_features,
    )

    # ------------------------------------------------------------
    # 15. IMPUTATION
    # ------------------------------------------------------------

    imputer = config.load_model2_imputer()

    X_imputed = apply_imputer(
        X_test=X_test,
        imputer=imputer,
    )

    # ------------------------------------------------------------
    # 16. MODEL PREDICTION
    # ------------------------------------------------------------

    model = config.load_model2_model()

    probabilities = model.predict_proba(
        X_imputed
    )[0]

    prediction = int(
        model.predict(
            X_imputed
        )[0]
    )

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            model.classes_,
            probabilities,
        )
    }

    confidence = float(
        max(probabilities)
    )

    class_name = config.MODEL2_CLASS_MAPPING.get(
        prediction,
        str(prediction),
    )

    final_class = (
        class_name
        if confidence >= config.CONFIDENCE_THRESHOLD
        else "Unknown"
    )

    status = (
        "Classified"
        if confidence >= config.CONFIDENCE_THRESHOLD
        else "Unknown"
    )

    # ------------------------------------------------------------
    # 17. RESULT
    # ------------------------------------------------------------

    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "prediction": prediction,
        "class_name": class_name,
        "final_class": final_class,
        "status": status,
        "confidence": confidence,
        "probabilities": probability_map,
        "feature_count": X_test.shape[1],
        "available_features": quality[
            "available_features"
        ],
        "missing_features": quality[
            "missing_features"
        ],
        "feature_coverage": quality[
            "coverage"
        ],
        "firms_detection_count": len(firms_df),
        "event_count": len(event_df),
        "historical_detection_count": len(
            historical_df
        ),
        "dynamic_world_records": len(dw_df),
        "dynamic_world_valid_records": int(
            (
                dw_df["status"] == "OK"
            ).sum()
            if len(dw_df)
            and "status" in dw_df.columns
            else 0
        ),
    }
