from __future__ import annotations

import numpy as np
import pandas as pd


def initialize_earth_engine(
    project: str,
):
    import ee

    try:
        ee.Initialize(
            project=project
        )

    except Exception:
        print(
            "Earth Engine authentication required..."
        )

        ee.Authenticate(
            auth_mode="notebook",
            force=True
        )

        ee.Initialize(
            project=project
        )

    return ee


def get_dynamic_world_for_point(
    ee,
    fire_lat: float,
    fire_lon: float,
    fire_date,
    area_m: float,
    search_before_days: int,
    search_after_days: int,
    bands: list[str],
    class_names: dict[int, str],
):
    fire_date = pd.Timestamp(
        fire_date
    ).normalize()

    point = ee.Geometry.Point(
        [
            float(fire_lon),
            float(fire_lat),
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
        fire_date
        -
        pd.Timedelta(
            days=search_before_days
        )
    )

    search_end = (
        fire_date
        +
        pd.Timedelta(
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

    timestamps = (
        collection
        .aggregate_array(
            "system:time_start"
        )
        .getInfo()
    )

    if not timestamps:
        return {
            "status": "NO_IMAGE"
        }

    count = len(timestamps)

    timestamp_array = np.asarray(
        timestamps,
        dtype="int64"
    )

    image_dates = pd.to_datetime(
        timestamp_array,
        unit="ms",
        utc=True
    ).tz_convert(
        None
    ).normalize()

    differences = np.abs(
        (
            image_dates
            -
            fire_date
        )
        /
        pd.Timedelta(days=1)
    )

    closest_index = int(
        np.argmin(
            differences
        )
    )

    closest_difference = float(
        differences[
            closest_index
        ]
    )

    closest_date = image_dates[
        closest_index
    ]

    print(
        "Closest DW image:",
        closest_date.strftime(
            "%Y-%m-%d"
        ),
        "| difference:",
        round(
            closest_difference,
            2
        ),
        "days"
    )

    image_list = collection.toList(
        count
    )

    image = ee.Image(
        image_list.get(
            closest_index
        )
    )

    means = (
        image
        .select(
            bands
        )
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=10,
            bestEffort=True,
            maxPixels=100000,
        )
        .getInfo()
    )

    if not means:
        print(
            "  → no probability result"
        )

        return {
            "status":
                "NO_VALID_PIXELS"
        }

    valid_probability_values = [
        means.get(
            band
        )
        for band in bands
        if means.get(
            band
        ) is not None
    ]

    if len(
        valid_probability_values
    ) == 0:

        print(
            "  → all probability pixels masked"
        )

        return {
            "status":
                "NO_VALID_PIXELS"
        }

    histogram_result = (
        image
        .select(
            "label"
        )
        .reduceRegion(
            reducer=(
                ee.Reducer
                .frequencyHistogram()
            ),
            geometry=area,
            scale=10,
            bestEffort=True,
            maxPixels=100000,
        )
        .getInfo()
    )

    histogram = (
        histogram_result.get(
            "label",
            {}
        )
        if histogram_result
        else {}
    )

    histogram = {
        int(float(key)): int(value)
        for key, value in histogram.items()
    }

    if histogram:

        dominant_label = max(
            histogram,
            key=histogram.get
        )

        total_pixels = sum(
            histogram.values()
        )

        dominant_probability = (
            histogram[
                dominant_label
            ]
            /
            total_pixels
        )

        dominant_class = (
            class_names.get(
                dominant_label,
                "unknown"
            )
        )

    else:

        dominant_label = None
        dominant_probability = np.nan
        dominant_class = None

    natural_values = [
        means.get("trees"),
        means.get("grass"),
        means.get("shrub_and_scrub"),
        means.get("flooded_vegetation"),
    ]

    if all(
        value is not None
        for value in natural_values
    ):
        natural_probability = sum(
            float(value)
            for value in natural_values
        )
    else:
        natural_probability = np.nan

    print(
        "  ✓ VALID DW IMAGE FOUND"
    )

    return {
        "status":
            "OK",

        "dynamic_world_date":
            closest_date.strftime(
                "%Y-%m-%d"
            ),

        "date_difference_days":
            closest_difference,

        "dw_water":
            means.get("water"),

        "dw_trees":
            means.get("trees"),

        "dw_grass":
            means.get("grass"),

        "dw_flooded_vegetation":
            means.get(
                "flooded_vegetation"
            ),

        "dw_crops":
            means.get("crops"),

        "dw_shrub_and_scrub":
            means.get(
                "shrub_and_scrub"
            ),

        "dw_built":
            means.get("built"),

        "dw_bare":
            means.get("bare"),

        "dw_snow_and_ice":
            means.get(
                "snow_and_ice"
            ),

        "dw_dominant_class":
            dominant_class,

        "dw_dominant_probability":
            dominant_probability,

        "dw_crop_probability":
            means.get("crops"),

        "dw_top_probability":
            dominant_probability,

        "dw_trees_prob":
            means.get("trees"),

        "dw_grass_prob":
            means.get("grass"),

        "dw_shrub_prob":
            means.get(
                "shrub_and_scrub"
            ),

        "dw_flooded_vegetation_prob":
            means.get(
                "flooded_vegetation"
            ),

        "dw_crop_prob":
            means.get("crops"),

        "dw_natural_vegetation_prob":
            natural_probability,
    }


def build_dynamic_world_features(
    event_df: pd.DataFrame,
    project: str,
    area_m: float,
    search_before_days: int,
    search_after_days: int,
    bands: list[str],
    class_names: dict[int, str],
) -> pd.DataFrame:

    import pandas as pd

    dw_rows = []
    dw_cache = {}

    ee = initialize_earth_engine(
        project
    )

    for index, row in event_df.iterrows():

        try:
            cache_key = (
                round(
                    float(
                        row["latitude"]
                    ),
                    6
                ),
                round(
                    float(
                        row["longitude"]
                    ),
                    6
                ),
                str(
                    pd.Timestamp(
                        row["acq_date"]
                    ).date()
                ),
            )

            if cache_key in dw_cache:

                result = dict(
                    dw_cache[
                        cache_key
                    ]
                )

            else:

                result = (
                    get_dynamic_world_for_point(
                        ee=ee,
                        fire_lat=row[
                            "latitude"
                        ],
                        fire_lon=row[
                            "longitude"
                        ],
                        fire_date=row[
                            "acq_date"
                        ],
                        area_m=area_m,
                        search_before_days=(
                            search_before_days
                        ),
                        search_after_days=(
                            search_after_days
                        ),
                        bands=bands,
                        class_names=class_names,
                    )
                )

                dw_cache[
                    cache_key
                ] = dict(result)

            result[
                "event_index"
            ] = index

            dw_rows.append(
                result
            )

        except Exception as exc:

            print(
                "DW point error:",
                str(exc)[:200]
            )

    return pd.DataFrame(
        dw_rows
    )


def aggregate_dynamic_world_features(
    dw_df: pd.DataFrame,
) -> dict:

    features = {}

    def dw_max(
        column: str
    ):
        if column not in dw_df.columns:
            return np.nan

        values = pd.to_numeric(
            dw_df[column],
            errors="coerce",
        ).dropna()

        return (
            float(values.max())
            if len(values)
            else np.nan
        )

    def dw_mean(
        column: str
    ):
        if column not in dw_df.columns:
            return np.nan

        values = pd.to_numeric(
            dw_df[column],
            errors="coerce",
        ).dropna()

        return (
            float(values.mean())
            if len(values)
            else np.nan
        )

    features[
        "event_max_dw_crop_probability"
    ] = dw_max(
        "dw_crops"
    )

    features[
        "event_mean_dw_crop_probability"
    ] = dw_mean(
        "dw_crops"
    )

    features[
        "event_max_dw_top_probability"
    ] = dw_max(
        "dw_dominant_probability"
    )

    features[
        "event_mean_dw_top_probability"
    ] = dw_mean(
        "dw_dominant_probability"
    )

    features[
        "event_max_dw_trees_prob"
    ] = dw_max(
        "dw_trees"
    )

    features[
        "event_mean_dw_trees_prob"
    ] = dw_mean(
        "dw_trees"
    )

    features[
        "event_max_dw_grass_prob"
    ] = dw_max(
        "dw_grass"
    )

    features[
        "event_mean_dw_grass_prob"
    ] = dw_mean(
        "dw_grass"
    )

    features[
        "event_max_dw_shrub_prob"
    ] = dw_max(
        "dw_shrub_and_scrub"
    )

    features[
        "event_mean_dw_shrub_prob"
    ] = dw_mean(
        "dw_shrub_and_scrub"
    )

    features[
        "event_max_dw_flooded_vegetation_prob"
    ] = dw_max(
        "dw_flooded_vegetation"
    )

    features[
        "event_mean_dw_flooded_vegetation_prob"
    ] = dw_mean(
        "dw_flooded_vegetation"
    )

    features[
        "event_max_dw_crop_prob"
    ] = dw_max(
        "dw_crops"
    )

    features[
        "event_mean_dw_crop_prob"
    ] = dw_mean(
        "dw_crops"
    )

    features[
        "event_max_dw_natural_vegetation_prob"
    ] = dw_max(
        "dw_natural_vegetation_prob"
    )

    features[
        "event_mean_dw_natural_vegetation_prob"
    ] = dw_mean(
        "dw_natural_vegetation_prob"
    )

    return features
