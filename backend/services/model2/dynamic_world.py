from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import ee
import pandas as pd


def initialize_earth_engine(project: str) -> None:
    """
    Initialize Google Earth Engine with the configured project.
    """

    try:
        ee.Initialize(project=project)
        print(
            f"Earth Engine initialized with project: {project}"
        )
        return

    except Exception as first_error:

        print(
            "Earth Engine authentication required..."
        )

        print(
            f"Initial initialization error: {first_error}"
        )

        try:

            ee.Authenticate(
                auth_mode="notebook",
                force=True,
            )

            ee.Initialize(
                project=project
            )

            print(
                f"Earth Engine initialized with project: {project}"
            )

        except Exception as auth_error:

            raise RuntimeError(
                "Unable to initialize Google Earth Engine."
            ) from auth_error


def _normalise_fire_date(
    fire_date: Any,
) -> datetime:
    """
    Normalize the FIRMS acquisition date to midnight.
    """

    if isinstance(
        fire_date,
        pd.Timestamp,
    ):

        fire_date = fire_date.to_pydatetime()

    if isinstance(
        fire_date,
        datetime,
    ):

        return fire_date.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    if hasattr(
        fire_date,
        "to_pydatetime",
    ):

        value = fire_date.to_pydatetime()

        return value.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    if isinstance(
        fire_date,
        str,
    ):

        parsed = pd.to_datetime(
            fire_date
        )

        if isinstance(
            parsed,
            pd.Timestamp,
        ):

            parsed = parsed.to_pydatetime()

        return parsed.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    raise TypeError(
        f"Unsupported fire_date type: {type(fire_date)}"
    )


def _safe_float(
    value: Any,
) -> float | None:
    """
    Convert a value to float.
    """

    if value is None:
        return None

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


def get_dynamic_world_for_point(
    latitude: float,
    longitude: float,
    fire_date: Any,
    project: str,
    area_m: int = 300,
    search_before_days: int = 120,
    search_after_days: int = 3,
    bands: list[str] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Retrieve Dynamic World features for one FIRMS detection.

    The closest Dynamic World image is attempted first.

    If the closest image has all probability pixels masked,
    continue to the next closest image until a valid image
    is found.
    """

    if bands is None:

        bands = [
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

    if class_names is None:

        class_names = {
            0: "water",
            1: "trees",
            2: "grass",
            3: "flooded_vegetation",
            4: "crops",
            5: "shrub_and_scrub",
            6: "built",
            7: "bare",
            8: "snow_and_ice",
        }

    fire_dt = _normalise_fire_date(
        fire_date
    )

    fire_date_str = fire_dt.strftime(
        "%Y-%m-%d"
    )

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    point = ee.Geometry.Point(
        [
            longitude,
            latitude,
        ]
    )

    area = (
        point
        .buffer(
            area_m / 2
        )
        .bounds()
    )

    search_start = (
        fire_dt
        - timedelta(
            days=search_before_days
        )
    )

    search_end = (
        fire_dt
        + timedelta(
            days=search_after_days + 1
        )
    )

    collection = (
        ee.ImageCollection(
            "GOOGLE/DYNAMICWORLD/V1"
        )
        .filterBounds(
            point
        )
        .filterDate(
            search_start.strftime(
                "%Y-%m-%d"
            ),
            search_end.strftime(
                "%Y-%m-%d"
            ),
        )
    )

    count = int(
        collection
        .size()
        .getInfo()
    )

    print(
        "DW images in search window:",
        count,
    )

    if count == 0:

        return {
            "status": "NO_IMAGE",
            "latitude": latitude,
            "longitude": longitude,
            "fire_date": fire_date_str,
        }

    image_list = collection.toList(
        count
    )

    candidates: list[
        tuple[
            float,
            str,
            Any,
        ]
    ] = []

    # ------------------------------------------------------------
    # Get all Dynamic World image dates.
    # ------------------------------------------------------------

    for index in range(
        count
    ):

        try:

            image = ee.Image(
                image_list.get(
                    index
                )
            )

            image_date = (
                ee.Date(
                    image.get(
                        "system:time_start"
                    )
                )
                .format(
                    "YYYY-MM-dd"
                )
                .getInfo()
            )

            image_date_ts = datetime.strptime(
                image_date,
                "%Y-%m-%d",
            )

            difference_days = abs(
                (
                    image_date_ts
                    - fire_dt
                ).total_seconds()
            ) / 86400.0

            candidates.append(
                (
                    difference_days,
                    image_date,
                    image,
                )
            )

        except Exception as error:

            print(
                "Unable to read Dynamic World "
                f"image date at index {index}: {error}"
            )

    if not candidates:

        return {
            "status": "NO_IMAGE",
            "latitude": latitude,
            "longitude": longitude,
            "fire_date": fire_date_str,
        }

    candidates.sort(
        key=lambda item: item[0]
    )

    # ------------------------------------------------------------
    # Try each candidate.
    #
    # If all probability pixels are masked, continue.
    # ------------------------------------------------------------

    for candidate_number, (
        difference_days,
        image_date,
        image,
    ) in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"Trying DW image: {image_date} "
            f"| difference: {difference_days:.1f} days "
            f"| candidate {candidate_number}/{len(candidates)}"
        )

        try:

            probability_image = (
                image.select(
                    bands
                )
            )

            probability_result = (
                probability_image
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=area,
                    scale=10,
                    bestEffort=True,
                    maxPixels=100000,
                )
                .getInfo()
            )

            if not probability_result:

                print(
                    "  → no probability statistics returned"
                )

                if candidate_number < len(
                    candidates
                ):

                    print(
                        "  → trying next Dynamic World image"
                    )

                continue

            # ----------------------------------------------------
            # Read Dynamic World probabilities.
            #
            # Keep the original Kaggle column names.
            # ----------------------------------------------------

            probability_values: dict[
                str,
                float | None,
            ] = {}

            for band in bands:

                probability_values[
                    band
                ] = _safe_float(
                    probability_result.get(
                        band
                    )
                )

            valid_probability_values = [
                value
                for value in probability_values.values()
                if value is not None
            ]

            if not valid_probability_values:

                print(
                    "  → all probability pixels masked"
                )

                if candidate_number < len(
                    candidates
                ):

                    print(
                        "  → trying next Dynamic World image"
                    )

                continue

            # ----------------------------------------------------
            # Dynamic World label.
            # ----------------------------------------------------

            label_image = image.select(
                "label"
            )

            histogram_result = (
                label_image
                .reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=area,
                    scale=10,
                    bestEffort=True,
                    maxPixels=100000,
                )
                .getInfo()
            )

            histogram = {}

            if histogram_result:

                histogram = histogram_result.get(
                    "label",
                    {},
                )

            dominant_label = None
            dominant_count = None

            if histogram:

                converted_histogram = {}

                for key, value in histogram.items():

                    try:

                        label_key = int(
                            key
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    try:

                        count_value = float(
                            value
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    converted_histogram[
                        label_key
                    ] = count_value

                if converted_histogram:

                    dominant_label = max(
                        converted_histogram,
                        key=converted_histogram.get,
                    )

                    dominant_count = (
                        converted_histogram[
                            dominant_label
                        ]
                    )

            total_label_pixels = None

            if histogram:

                try:

                    total_label_pixels = sum(
                        float(value)
                        for value in histogram.values()
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    total_label_pixels = None

            dominant_probability = None

            if (
                dominant_count is not None
                and total_label_pixels
                and total_label_pixels > 0
            ):

                dominant_probability = (
                    dominant_count
                    / total_label_pixels
                )

            dominant_class = None

            if dominant_label is not None:

                dominant_class = class_names.get(
                    dominant_label
                )

            # ----------------------------------------------------
            # Natural vegetation probability.
            #
            # Exact Kaggle calculation:
            #
            # trees
            # + grass
            # + shrub_and_scrub
            # + flooded_vegetation
            # ----------------------------------------------------

            natural_components = [
                probability_values.get(
                    "trees"
                ),
                probability_values.get(
                    "grass"
                ),
                probability_values.get(
                    "shrub_and_scrub"
                ),
                probability_values.get(
                    "flooded_vegetation"
                ),
            ]

            natural_vegetation_probability = None

            if all(
                value is not None
                for value in natural_components
            ):

                natural_vegetation_probability = sum(
                    natural_components
                )

            print(
                f"  ✓ valid Dynamic World image: "
                f"{image_date}"
            )

            # ----------------------------------------------------
            # Return exact Kaggle-style Dynamic World columns.
            # ----------------------------------------------------

            return {
                "status": "OK",

                "latitude": latitude,

                "longitude": longitude,

                "fire_date": fire_date_str,

                "dw_image_date": image_date,

                "dw_date_difference_days": (
                    difference_days
                ),

                "dw_dominant_label": (
                    dominant_label
                ),

                "dw_dominant_class": (
                    dominant_class
                ),

                "dw_dominant_probability": (
                    dominant_probability
                ),

                "dw_water": probability_values.get(
                    "water"
                ),

                "dw_trees": probability_values.get(
                    "trees"
                ),

                "dw_grass": probability_values.get(
                    "grass"
                ),

                "dw_flooded_vegetation": (
                    probability_values.get(
                        "flooded_vegetation"
                    )
                ),

                "dw_crops": probability_values.get(
                    "crops"
                ),

                "dw_shrub_and_scrub": (
                    probability_values.get(
                        "shrub_and_scrub"
                    )
                ),

                "dw_built": probability_values.get(
                    "built"
                ),

                "dw_bare": probability_values.get(
                    "bare"
                ),

                "dw_snow_and_ice": (
                    probability_values.get(
                        "snow_and_ice"
                    )
                ),

                "dw_natural_vegetation_probability": (
                    natural_vegetation_probability
                ),
            }

        except Exception as error:

            print(
                "  → Dynamic World reduction failed: "
                f"{error}"
            )

            if candidate_number < len(
                candidates
            ):

                print(
                    "  → trying next Dynamic World image"
                )

            continue

    print(
        "No valid Dynamic World image found "
        f"for {fire_date_str}."
    )

    return {
        "status": "NO_VALID_PIXELS",
        "latitude": latitude,
        "longitude": longitude,
        "fire_date": fire_date_str,
    }


def build_dynamic_world_features(
    event_df: pd.DataFrame,
    project: str,
    area_m: int = 300,
    search_before_days: int = 120,
    search_after_days: int = 3,
    bands: list[str] | None = None,
    class_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Build Dynamic World features for every detection
    in the selected event.
    """

    initialize_earth_engine(
        project
    )

    if (
        event_df is None
        or event_df.empty
    ):

        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    for event_index, (
        _,
        row,
    ) in enumerate(
        event_df.iterrows()
    ):

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        fire_datetime = row[
            "acq_datetime"
        ]

        result = get_dynamic_world_for_point(
            latitude=latitude,
            longitude=longitude,
            fire_date=fire_datetime,
            project=project,
            area_m=area_m,
            search_before_days=search_before_days,
            search_after_days=search_after_days,
            bands=bands,
            class_names=class_names,
        )

        result[
            "event_index"
        ] = event_index

        rows.append(
            result
        )

    if not rows:

        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


def aggregate_dynamic_world_features(
    dw_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Aggregate Dynamic World event features.

    This follows the original Kaggle implementation.

    IMPORTANT:
    The same source column ``dw_crops`` is intentionally
    used for BOTH:

        event_*_dw_crop_probability

    and:

        event_*_dw_crop_prob

    Therefore this MUST be represented as a list of mappings,
    not a dictionary with duplicate ``dw_crops`` keys.
    """

    if (
        dw_df is None
        or dw_df.empty
    ):

        return {}

    if "status" in dw_df.columns:

        valid_df = dw_df[
            dw_df["status"] == "OK"
        ].copy()

    else:

        valid_df = dw_df.copy()

    if valid_df.empty:

        return {}

    features: dict[
        str,
        float,
    ] = {}

    # ------------------------------------------------------------
    # EXACT Kaggle mappings.
    #
    # DO NOT convert this to a dictionary because ``dw_crops``
    # appears twice intentionally.
    # ------------------------------------------------------------

    feature_mappings = [
        (
            "dw_crops",
            "event_max_dw_crop_probability",
            "event_mean_dw_crop_probability",
        ),

        (
            "dw_dominant_probability",
            "event_max_dw_top_probability",
            "event_mean_dw_top_probability",
        ),

        (
            "dw_trees",
            "event_max_dw_trees_prob",
            "event_mean_dw_trees_prob",
        ),

        (
            "dw_grass",
            "event_max_dw_grass_prob",
            "event_mean_dw_grass_prob",
        ),

        (
            "dw_shrub_and_scrub",
            "event_max_dw_shrub_prob",
            "event_mean_dw_shrub_prob",
        ),

        (
            "dw_flooded_vegetation",
            "event_max_dw_flooded_vegetation_prob",
            "event_mean_dw_flooded_vegetation_prob",
        ),

        # IMPORTANT:
        # Same source column again.
        (
            "dw_crops",
            "event_max_dw_crop_prob",
            "event_mean_dw_crop_prob",
        ),

        (
            "dw_natural_vegetation_probability",
            "event_max_dw_natural_vegetation_prob",
            "event_mean_dw_natural_vegetation_prob",
        ),
    ]

    # ------------------------------------------------------------
    # Perform max + mean exactly as Kaggle.
    # ------------------------------------------------------------

    for (
        source_column,
        max_feature,
        mean_feature,
    ) in feature_mappings:

        if source_column not in valid_df.columns:

            continue

        values = pd.to_numeric(
            valid_df[
                source_column
            ],
            errors="coerce",
        ).dropna()

        if values.empty:

            continue

        features[
            max_feature
        ] = float(
            values.max()
        )

        features[
            mean_feature
        ] = float(
            values.mean()
        )

    return features