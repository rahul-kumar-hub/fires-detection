# src/dynamic_world.py

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd


# ============================================================
# DYNAMIC WORLD CONSTANTS
# ============================================================

EE_PROJECT = (
    os.getenv("EE_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or ""
).strip()

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

SEARCH_DAYS_BEFORE = 15
SEARCH_DAYS_AFTER = 1

DW_PROBABILITY_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_FEATURES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
    "dw_date_difference_days",
    "month",
    "dw_max_probability",
    "dw_available",
]


# ============================================================
# EARTH ENGINE IMPORT
# ============================================================

def ensure_earth_engine():
    """
    Make sure the Earth Engine Python API is available.
    """

    if importlib.util.find_spec("ee") is None:

        print("Earth Engine API not installed.")
        print("Installing...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "earthengine-api",
        ])

    import ee

    return ee


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

def initialize_earth_engine():
    """
    Initialize Google Earth Engine using the configured Cloud project.
    """

    ee = ensure_earth_engine()

    try:

        if EE_PROJECT:
            ee.Initialize(project=EE_PROJECT)
        else:
            ee.Initialize()

    except Exception as exc:

        raise RuntimeError(
            "\nEarth Engine initialization failed.\n\n"
            + str(exc)
            + "\n\nSet EE_PROJECT (or GOOGLE_CLOUD_PROJECT) to a Google Cloud "
            "project where Earth Engine is enabled and the authenticated "
            "caller has serviceusage.services.use permission."
        ) from exc

    return ee


# ============================================================
# NORMALIZE TARGET DATETIME
# ============================================================

def normalize_target_datetime(
    hotspot_datetime,
) -> pd.Timestamp:
    """
    Convert hotspot datetime into a UTC-aware pandas Timestamp.
    """

    target_dt = pd.Timestamp(
        hotspot_datetime
    )

    if target_dt.tzinfo is None:

        target_dt = target_dt.tz_localize(
            "UTC"
        )

    else:

        target_dt = target_dt.tz_convert(
            "UTC"
        )

    return target_dt


# ============================================================
# MISSING DYNAMIC WORLD FEATURES
# ============================================================

def missing_dynamic_world_features(
    target_dt: pd.Timestamp,
) -> dict:
    """
    Training-compatible missing-DW state.

    Numerical DW values remain NaN so that the final
    prediction stage can apply:

        fillna(train_medians).fillna(0)
    """

    feature_dict = {}

    for band in DW_PROBABILITY_BANDS:
        feature_dict[band] = np.nan

    feature_dict[
        "dw_date_difference_days"
    ] = np.nan

    feature_dict[
        "month"
    ] = int(target_dt.month)

    feature_dict[
        "dw_max_probability"
    ] = np.nan

    feature_dict[
        "dw_available"
    ] = 0

    return feature_dict


# ============================================================
# FIND CANDIDATE DYNAMIC WORLD IMAGES
# ============================================================

def get_dynamic_world_candidates(
    ee,
    point,
    target_dt: pd.Timestamp,
):
    """
    Search Dynamic World using the exact Cell 2B window:

        15 days before
        through
        1 day after
    """

    window_start = (
        target_dt
        - pd.Timedelta(
            days=SEARCH_DAYS_BEFORE
        )
    ).strftime("%Y-%m-%d")

    window_end = (
        target_dt
        + pd.Timedelta(
            days=SEARCH_DAYS_AFTER
        )
    ).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection(
            DW_COLLECTION
        )
        .filterBounds(point)
        .filterDate(
            window_start,
            window_end,
        )
    )

    count = collection.size().getInfo()

    candidates = []

    if count == 0:
        return collection, candidates, count

    image_list = collection.toList(
        count
    )

    for i in range(count):

        img = ee.Image(
            image_list.get(i)
        )

        try:

            timestamp = (
                img
                .get("system:time_start")
                .getInfo()
            )

            if timestamp is None:
                continue

            image_dt = pd.to_datetime(
                timestamp,
                unit="ms",
                utc=True,
            )

            difference_days = (
                abs(
                    (
                        image_dt
                        - target_dt
                    ).total_seconds()
                )
                / 86400.0
            )

            image_id = (
                img
                .get("system:index")
                .getInfo()
            )

            candidates.append({
                "image": img,
                "image_dt": image_dt,
                "difference_days": difference_days,
                "image_id": image_id,
            })

        except Exception:
            continue

    candidates.sort(
        key=lambda item:
        item["difference_days"]
    )

    return collection, candidates, count


# ============================================================
# FIND FIRST VALID PIXEL
# ============================================================

def find_valid_dynamic_world_pixel(
    ee,
    candidates,
    point,
):
    """
    Examine candidate images in nearest-date order.

    The first image for which all nine probability bands
    are available at the exact hotspot pixel is selected.
    """

    for candidate in candidates:

        try:

            sample = (
                candidate["image"]
                .select(
                    DW_PROBABILITY_BANDS
                )
                .reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=10,
                    bestEffort=True,
                )
            )

            values = sample.getInfo()

            if values is None:
                continue

            missing_bands = [
                band
                for band
                in DW_PROBABILITY_BANDS
                if values.get(band) is None
            ]

            if missing_bands:
                continue

            return {
                "image": candidate["image"],
                "image_dt": candidate["image_dt"],
                "difference_days": candidate[
                    "difference_days"
                ],
                "image_id": candidate[
                    "image_id"
                ],
                "values": values,
            }

        except Exception:
            continue

    return None


# ============================================================
# BUILD VALID DW FEATURES
# ============================================================

def build_valid_dynamic_world_features(
    valid_dw: dict,
    target_dt: pd.Timestamp,
) -> dict:
    """
    Convert a valid Dynamic World pixel into the exact
    Cell 2B feature representation.
    """

    best_dt = valid_dw[
        "image_dt"
    ]

    dw_values = valid_dw[
        "values"
    ]

    # --------------------------------------------------------
    # Nine probability bands
    # --------------------------------------------------------

    dw_probs = {
        band: float(
            dw_values[band]
        )
        for band in DW_PROBABILITY_BANDS
    }

    # --------------------------------------------------------
    # Maximum probability
    # --------------------------------------------------------

    dw_max_probability = max(
        dw_probs.values()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Training uses the signed calendar-date difference,
    # not the absolute timestamp difference.
    # --------------------------------------------------------

    target_date_only = (
        target_dt.normalize()
    )

    dw_date_only = (
        best_dt.normalize()
    )

    dw_date_difference_days = (
        target_date_only
        - dw_date_only
    ).total_seconds() / 86400.0

    # --------------------------------------------------------
    # Build feature dictionary
    # --------------------------------------------------------

    feature_dict = {}

    for band in DW_PROBABILITY_BANDS:

        feature_dict[band] = (
            dw_probs[band]
        )

    feature_dict[
        "dw_date_difference_days"
    ] = float(
        dw_date_difference_days
    )

    feature_dict[
        "month"
    ] = int(
        target_dt.month
    )

    feature_dict[
        "dw_max_probability"
    ] = float(
        dw_max_probability
    )

    feature_dict[
        "dw_available"
    ] = 1

    return feature_dict


# ============================================================
# MAIN DYNAMIC WORLD FUNCTION
# ============================================================

def get_dynamic_world_features(
    live_hotspot_lat: float,
    live_hotspot_lon: float,
    live_hotspot_datetime,
) -> dict:
    """
    Retrieve Dynamic World features for the live FIRMS
    hotspot using the exact Cell 2B workflow.

    Returns only the 13 DW-related model features:

        9 probabilities
        dw_date_difference_days
        month
        dw_max_probability
        dw_available
    """

    print("\n" + "=" * 70)
    print(
        "LIVE DYNAMIC WORLD RETRIEVAL"
    )
    print("=" * 70)

    target_dt = normalize_target_datetime(
        live_hotspot_datetime
    )

    print(
        f"Hotspot latitude : "
        f"{float(live_hotspot_lat):.6f}"
    )

    print(
        f"Hotspot longitude: "
        f"{float(live_hotspot_lon):.6f}"
    )

    print(
        f"Hotspot datetime : "
        f"{target_dt}"
    )

    # --------------------------------------------------------
    # Initialize Earth Engine
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "INITIALIZING GOOGLE EARTH ENGINE"
    )
    print("-" * 70)

    ee = initialize_earth_engine()

    print(
        "✓ Earth Engine initialized successfully"
    )

    print(
        "✓ Project ID:",
        EE_PROJECT,
    )

    # --------------------------------------------------------
    # Create hotspot point
    # --------------------------------------------------------

    point = ee.Geometry.Point([
        float(live_hotspot_lon),
        float(live_hotspot_lat),
    ])

    print(
        "\n✓ Hotspot point created"
    )

    # --------------------------------------------------------
    # Load Dynamic World collection
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "LOADING DYNAMIC WORLD"
    )
    print("-" * 70)

    print(
        f"Collection: {DW_COLLECTION}"
    )

    # --------------------------------------------------------
    # Search collection
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "SEARCHING DYNAMIC WORLD"
    )
    print("-" * 70)

    window_start = (
        target_dt
        - pd.Timedelta(
            days=SEARCH_DAYS_BEFORE
        )
    ).strftime("%Y-%m-%d")

    window_end = (
        target_dt
        + pd.Timedelta(
            days=SEARCH_DAYS_AFTER
        )
    ).strftime("%Y-%m-%d")

    print(
        "Search start:",
        window_start,
    )

    print(
        "Search end  :",
        window_end,
    )

    try:

        (
            collection,
            candidates,
            count,
        ) = get_dynamic_world_candidates(
            ee=ee,
            point=point,
            target_dt=target_dt,
        )

    except Exception as exc:

        raise RuntimeError(
            "Dynamic World collection query failed: "
            + str(exc)
        ) from exc

    print(
        "Dynamic World images found:",
        count,
    )

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if count == 0:

        print("\n" + "=" * 70)
        print(
            "NO DYNAMIC WORLD IMAGE FOUND"
        )
        print("=" * 70)

        print(
            "Dynamic World is unavailable "
            "for this hotspot."
        )

        return missing_dynamic_world_features(
            target_dt
        )

    # --------------------------------------------------------
    # Find first valid pixel
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "CHECKING FOR VALID DYNAMIC WORLD PIXEL"
    )
    print("-" * 70)

    valid_dw = find_valid_dynamic_world_pixel(
        ee=ee,
        candidates=candidates,
        point=point,
    )

    # --------------------------------------------------------
    # No valid pixel
    # --------------------------------------------------------

    if valid_dw is None:

        print("\n" + "=" * 70)
        print(
            "NO VALID DYNAMIC WORLD PIXEL"
        )
        print("=" * 70)

        print(
            f"Checked {len(candidates)} "
            f"candidate image(s)."
        )

        print(
            "The hotspot is "
            "masked/unavailable."
        )

        result = (
            missing_dynamic_world_features(
                target_dt
            )
        )

        print(
            "\n✓ Dynamic World marked unavailable"
        )

        print(
            "✓ dw_available = 0"
        )

        print(
            "✓ DW numerical features = NaN"
        )

        print(
            "✓ Final prediction stage "
            "will use train_medians"
        )

        return result

    # --------------------------------------------------------
    # Valid pixel found
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "VALID DYNAMIC WORLD IMAGE SELECTED"
    )
    print("=" * 70)

    print(
        "Image ID:",
        valid_dw["image_id"],
    )

    print(
        "DW datetime:",
        valid_dw["image_dt"],
    )

    print(
        "Absolute date gap:",
        f"{valid_dw['difference_days']:.6f}",
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    result = build_valid_dynamic_world_features(
        valid_dw=valid_dw,
        target_dt=target_dt,
    )

    print(
        "\nDynamic World probabilities:"
    )

    for band in DW_PROBABILITY_BANDS:

        print(
            f"{band:<22}: "
            f"{result[band]:.6f}"
        )

    probability_sum = sum(
        result[band]
        for band in DW_PROBABILITY_BANDS
    )

    print(
        "\nProbability sum:",
        f"{probability_sum:.6f}",
    )

    print(
        "Maximum probability:",
        f"{result['dw_max_probability']:.6f}",
    )

    print(
        "Signed date difference:",
        f"{result['dw_date_difference_days']:.6f}",
    )

    print(
        "\n✓ Dynamic World features added"
    )

    return result