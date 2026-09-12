# backend/main.py

from __future__ import annotations

import os
from getpass import getpass

from backend.inference.model2 import run_model2
from backend.services.model2 import config as model2_config
from backend.services.firms import (
    build_firms_features,
    find_live_hotspot,
)
from backend.services.persistence import (
    calculate_persistence,
)
from backend.services.osm import (
    get_osm_features,
)
from backend.services.dynamic_world import (
    get_dynamic_world_features,
)
from backend.services.feature_builder import (
    assemble_feature_dict,
    add_dynamic_world_features,
)
from backend.inference.model1 import (
    load_model_package,
    predict_fire_source,
    print_prediction,
)


# ============================================================
# FIRE DETECTION MODEL 1
# ============================================================

def main():

    selected_model = os.environ.get("FIRE_MODEL", "1")

    print("\n" + "=" * 70)
    print(f"FIRE DETECTION SYSTEM — MODEL {selected_model}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. ENTER LOCATION
    # --------------------------------------------------------

    try:

        latitude = float(
            input(
                "Enter latitude : "
            ).strip()
        )

        longitude = float(
            input(
                "Enter longitude: "
            ).strip()
        )

    except ValueError as exc:

        raise ValueError(
            "Latitude and longitude must be valid numbers."
        ) from exc

    if not (
        -90 <= latitude <= 90
    ):
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not (
        -180 <= longitude <= 180
    ):
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    if os.environ.get("FIRE_MODEL", "1") == "2":
        result = run_model2(
            latitude=latitude,
            longitude=longitude,
        )
        print("\n" + "=" * 60)
        print("FIRE SOURCE CLASSIFICATION")
        print("=" * 60)
        print(f"Predicted source : {result['class_name']}")
        print(f"Confidence       : {result['confidence'] * 100:.2f}%")
        print(f"Feature count    : {result['feature_count']}")
        print("\nClass probabilities:")

        sorted_probabilities = sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for class_id, probability in sorted_probabilities:
            class_name = model2_config.MODEL2_CLASS_MAPPING.get(
                int(class_id),
                f"Class {class_id}",
            )
            print(f"  {class_name:22s} {probability * 100:8.2f}%")

        print("=" * 60)
        print("\nNOTE:")
        print(
            "This is a predicted source class "
            "from the 2025-trained Random Forest."
        )
        print("It is NOT a NASA FIRMS source label.")
        return result

    print("\nQuery location")
    print(
        f"Latitude : {latitude}"
    )
    print(
        f"Longitude: {longitude}"
    )

    # --------------------------------------------------------
    # 2. NASA FIRMS MAP KEY
    # --------------------------------------------------------

    map_key = os.environ.get(
        "NASA_FIRMS_MAP_KEY"
    )

    if not map_key:

        map_key = getpass(
            "\nEnter NASA FIRMS MAP_KEY: "
        ).strip()

    if not map_key:

        raise RuntimeError(
            "NASA FIRMS MAP_KEY is required."
        )

    # --------------------------------------------------------
    # 3. LOAD MODEL 1
    # --------------------------------------------------------

    print(
        "\nLoading Model 1..."
    )

    model_package = (
        load_model_package()
    )

    print(
        "✓ Model 1 loaded successfully"
    )

    print(
        f"✓ Model features: "
        f"{len(model_package['features'])}"
    )

    print(
        f"✓ Model classes: "
        f"{model_package['classes']}"
    )

    # --------------------------------------------------------
    # 4. FIND LIVE FIRMS HOTSPOT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "1. LIVE NASA FIRMS HOTSPOT"
    )
    print("=" * 70)

    live_hotspot = (
        find_live_hotspot(
            lat=latitude,
            lon=longitude,
            map_key=map_key,
        )
    )

    hotspot_lat = float(
        live_hotspot["latitude"]
    )

    hotspot_lon = float(
        live_hotspot["longitude"]
    )

    hotspot_datetime = (
        live_hotspot[
            "hotspot_datetime"
        ]
    )

    print(
        "\n✓ Live hotspot found"
    )

    print(
        f"Hotspot latitude : "
        f"{hotspot_lat:.6f}"
    )

    print(
        f"Hotspot longitude: "
        f"{hotspot_lon:.6f}"
    )

    print(
        f"Hotspot datetime : "
        f"{hotspot_datetime}"
    )

    print(
        f"FIRMS source     : "
        f"{live_hotspot['firms_source']}"
    )

    # --------------------------------------------------------
    # 5. FIRMS FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "2. FIRMS FEATURES"
    )
    print("=" * 70)

    firms_engineered = (
        build_firms_features(
            live_hotspot
        )
    )

    # Combine raw FIRMS fields with the engineered fields.
    firms_features = dict(
        live_hotspot
    )

    firms_features.update(
        firms_engineered
    )

    print(
        "✓ FIRMS features prepared"
    )

    # --------------------------------------------------------
    # 6. FIRMS PERSISTENCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "3. FIRMS PERSISTENCE"
    )
    print("=" * 70)

    persistence_features = (
        calculate_persistence(
            map_key=map_key,
            hotspot_lat=hotspot_lat,
            hotspot_lon=hotspot_lon,
            hotspot_datetime=hotspot_datetime,
        )
    )

    print(
        "✓ Persistence features prepared"
    )

    # --------------------------------------------------------
    # 7. OSM
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "4. OPENSTREETMAP FEATURES"
    )
    print("=" * 70)

    (
        osm_distances,
        osm_features,
    ) = get_osm_features(
        lat=hotspot_lat,
        lon=hotspot_lon,
    )

    print(
        f"✓ OSM distances: "
        f"{len(osm_distances)}"
    )

    print(
        f"✓ OSM model features: "
        f"{len(osm_features)}"
    )

    # --------------------------------------------------------
    # 8. BUILD NON-DW FEATURE DICTIONARY
    # --------------------------------------------------------

    feature_dict = (
        assemble_feature_dict(
            firms_features=firms_features,
            persistence_features=(
                persistence_features
            ),
            osm_features=osm_features,
        )
    )

    print(
        "\n✓ Non-Dynamic-World "
        f"features: {len(feature_dict)}"
    )

    # --------------------------------------------------------
    # 9. DYNAMIC WORLD
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "5. GOOGLE DYNAMIC WORLD"
    )
    print("=" * 70)

    dynamic_world_features = (
        get_dynamic_world_features(
            live_hotspot_lat=hotspot_lat,
            live_hotspot_lon=hotspot_lon,
            live_hotspot_datetime=(
                hotspot_datetime
            ),
        )
    )

    print(
        "✓ Dynamic World processing complete"
    )

    # --------------------------------------------------------
    # 10. FINAL FEATURE DICTIONARY
    # --------------------------------------------------------

    feature_dict = (
        add_dynamic_world_features(
            feature_dict=feature_dict,
            dynamic_world_features=(
                dynamic_world_features
            ),
        )
    )

    print("\n" + "=" * 70)
    print(
        "6. FINAL MODEL FEATURE SET"
    )
    print("=" * 70)

    print(
        f"Feature dictionary size: "
        f"{len(feature_dict)}"
    )

    # --------------------------------------------------------
    # 11. MODEL 1 PREDICTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "7. MODEL 1 PREDICTION"
    )
    print("=" * 70)

    prediction = (
        predict_fire_source(
            feature_dict=feature_dict,
            model_package=model_package,
        )
    )

    print_prediction(
        prediction
    )

    # --------------------------------------------------------
    # 12. RETURN
    # --------------------------------------------------------

    return {
        "query_latitude":
            latitude,

        "query_longitude":
            longitude,

        "hotspot_latitude":
            hotspot_lat,

        "hotspot_longitude":
            hotspot_lon,

        "hotspot_datetime":
            hotspot_datetime.isoformat(),

        "firms_source":
            live_hotspot.get(
                "firms_source"
            ),

        "prediction":
            prediction,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()