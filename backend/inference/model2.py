from __future__ import annotations

import os

from backend.services.model2.pipeline import run_model2_pipeline


def run_model2(
    latitude: float,
    longitude: float,
    timeout_seconds: int = 900,
) -> dict:
    """Run the complete Model 2 evidence pipeline for one location."""
    del timeout_seconds

    map_key = os.getenv("NASA_FIRMS_MAP_KEY")

    if not map_key:
        raise RuntimeError(
            "NASA_FIRMS_MAP_KEY is not available to the FastAPI process. "
            "Export it before starting the backend."
        )

    return run_model2_pipeline(
        latitude=latitude,
        longitude=longitude,
        fire_datetime=None,
        firms_map_key=map_key,
    )