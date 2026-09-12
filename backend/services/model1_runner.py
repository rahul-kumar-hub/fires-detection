from __future__ import annotations

import os
from getpass import getpass


from backend.services.firms import (
    find_live_hotspot,
    build_firms_features,
)

from backend.services.feature_builder import (
    assemble_feature_dict,
    add_dynamic_world_features,
)

from backend.services.persistence import calculate_persistence

from backend.services.osm import get_osm_features

from backend.services.dynamic_world import (
    get_dynamic_world_features,
)

from backend.inference.model1 import (
    load_model_package,
    predict_fire_source,
)


def run_model1(
    latitude: float,
    longitude: float,
) -> dict:
    """
    Run the existing Model 1 pipeline.

    This function preserves the same processing order
    used by backend/main.py.
    """

    # --------------------------------------------------------
    # NASA FIRMS API key
    # --------------------------------------------------------

    map_key = os.environ.get(
        "NASA_FIRMS_MAP_KEY"
    )

    if not map_key:
        map_key = getpass(
            "Enter NASA FIRMS MAP_KEY: "
        ).strip()

    if not map_key:
        raise RuntimeError(
            "NASA FIRMS MAP_KEY is required."
        )

    # --------------------------------------------------------
    # Load trained Model 1 package
    # --------------------------------------------------------

    model_package = load_model_package()

    # --------------------------------------------------------
    # Find live hotspot
    # --------------------------------------------------------

    live_hotspot = find_live_hotspot(
        lat=latitude,
        lon=longitude,
        map_key=map_key,
    )

    if not live_hotspot:
        raise RuntimeError(
            "No live NASA FIRMS hotspot found "
            "for the requested location."
        )

    # --------------------------------------------------------
    # Extract hotspot coordinates and time
    # --------------------------------------------------------

    hotspot_lat = live_hotspot.get(
        "latitude",
        live_hotspot.get("lat"),
    )

    hotspot_lon = live_hotspot.get(
        "longitude",
        live_hotspot.get("lon"),
    )

    hotspot_datetime = live_hotspot.get(
        "hotspot_datetime"
    )

    if hotspot_lat is None or hotspot_lon is None:
        raise RuntimeError(
            "Live hotspot does not contain valid "
            "latitude and longitude values."
        )

    if hotspot_datetime is None:
        raise RuntimeError(
            "Live hotspot does not contain a datetime."
        )

    # --------------------------------------------------------
    # Normalize datetime if necessary
    # --------------------------------------------------------

    # if isinstance(
    #     hotspot_datetime,
    #     str,
    # ):
    #     hotspot_datetime = datetime.fromisoformat(
    #         hotspot_datetime.replace(
    #             "Z",
    #             "+00:00",
    #         )
    #     )

    # --------------------------------------------------------
    # FIRMS features
    # --------------------------------------------------------

    firms_features = build_firms_features(
        live_hotspot
    )

    firms_features.update(
        live_hotspot
    )

    # --------------------------------------------------------
    # Persistence features
    # --------------------------------------------------------

    persistence_features = calculate_persistence(
        map_key=map_key,
        hotspot_lat=hotspot_lat,
        hotspot_lon=hotspot_lon,
        hotspot_datetime=hotspot_datetime,
    )

    # --------------------------------------------------------
    # OpenStreetMap features
    # --------------------------------------------------------

    osm_distances, osm_features = get_osm_features(
        lat=hotspot_lat,
        lon=hotspot_lon,
    )

    # Keep the same feature assembly used by the CLI.
    feature_dict = assemble_feature_dict(
        firms_features=firms_features,
        persistence_features=persistence_features,
        osm_features=osm_features,
    )

    # --------------------------------------------------------
    # Dynamic World features
    # --------------------------------------------------------

    dynamic_world_features = get_dynamic_world_features(
        live_hotspot_lat=hotspot_lat,
        live_hotspot_lon=hotspot_lon,
        live_hotspot_datetime=hotspot_datetime,
    )

    feature_dict = add_dynamic_world_features(
        feature_dict=feature_dict,
        dynamic_world_features=dynamic_world_features,
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = predict_fire_source(
        feature_dict=feature_dict,
        model_package=model_package,
    )

    # --------------------------------------------------------
    # API-safe response
    # --------------------------------------------------------

    return {
        "query_latitude": latitude,
        "query_longitude": longitude,
        "hotspot_latitude": hotspot_lat,
        "hotspot_longitude": hotspot_lon,
        "hotspot_datetime": hotspot_datetime.isoformat(),
        "firms_source": live_hotspot.get(
            "firms_source"
        ),
        "prediction": prediction,
    }