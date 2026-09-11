from __future__ import annotations

import numpy as np
import pandas as pd

from .events import haversine_m


def safe_stat(
    series,
    function,
    default=np.nan,
):
    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(series) == 0:
        return default

    if function == "mean":
        return float(series.mean())

    if function == "median":
        return float(series.median())

    if function == "min":
        return float(series.min())

    if function == "max":
        return float(series.max())

    if function == "std":
        if len(series) >= 2:
            return float(series.std())
        return 0.0

    return default


def build_event_features(
    event_df: pd.DataFrame,
) -> dict:

    ev = event_df.copy()

    ev["acq_date"] = pd.to_datetime(
        ev["acq_date"],
        errors="coerce",
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
    ]

    for column in numeric_columns:
        if column in ev.columns:
            ev[column] = pd.to_numeric(
                ev[column],
                errors="coerce",
            )

    n = len(ev)

    active_days = (
        ev["acq_date"]
        .dt.date
        .nunique()
    )

    active_months = (
        ev["acq_date"]
        .dt.month
        .nunique()
    )

    frp = pd.to_numeric(
        ev["frp"],
        errors="coerce",
    )

    brightness = pd.to_numeric(
        ev.get(
            "bright_ti4",
            pd.Series(
                np.nan,
                index=ev.index,
            ),
        ),
        errors="coerce",
    )

    mean_latitude = float(
        ev["latitude"].mean()
    )

    mean_longitude = float(
        ev["longitude"].mean()
    )

    min_latitude = float(
        ev["latitude"].min()
    )

    max_latitude = float(
        ev["latitude"].max()
    )

    min_longitude = float(
        ev["longitude"].min()
    )

    max_longitude = float(
        ev["longitude"].max()
    )

    latitude_span_deg = (
        max_latitude -
        min_latitude
    )

    longitude_span_deg = (
        max_longitude -
        min_longitude
    )

    latitude_span_km = (
        latitude_span_deg *
        111.32
    )

    longitude_span_km = (
        longitude_span_deg *
        111.32 *
        np.cos(
            np.radians(
                mean_latitude
            )
        )
    )

    # ------------------------------------------------------------
    # 500 m / 1 km spatial blocks
    # ------------------------------------------------------------

    lat_step_500 = (
        0.5 / 111.32
    )

    lat_step_1km = (
        1.0 / 111.32
    )

    ev["_lat500"] = np.floor(
        ev["latitude"] /
        lat_step_500
    )

    ev["_lon500"] = np.floor(
        ev["longitude"] /
        (
            0.5 /
            (
                111.32 *
                np.cos(
                    np.radians(
                        ev["latitude"]
                    )
                )
            )
        )
    )

    ev["_lat1km"] = np.floor(
        ev["latitude"] /
        lat_step_1km
    )

    ev["_lon1km"] = np.floor(
        ev["longitude"] /
        (
            1.0 /
            (
                111.32 *
                np.cos(
                    np.radians(
                        ev["latitude"]
                    )
                )
            )
        )
    )

    block_500_count = (
        ev[
            [
                "_lat500",
                "_lon500",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    block_1km_count = (
        ev[
            [
                "_lat1km",
                "_lon1km",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # ------------------------------------------------------------
    # Temporal behavior inside event
    # ------------------------------------------------------------

    times = (
        ev["acq_datetime"]
        .dropna()
        .sort_values()
    )

    if len(times) >= 2:
        duration_hours = (
            (
                times.max() -
                times.min()
            ).total_seconds()
            /
            3600
        )
    else:
        duration_hours = 0.0

    duration_days = (
        duration_hours / 24
    )

    if len(times) >= 2:
        gaps_hours = (
            times
            .diff()
            .dt.total_seconds()
            /
            3600
        ).dropna()
    else:
        gaps_hours = pd.Series(
            dtype=float
        )

    unique_days = (
        ev["acq_date"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    if len(unique_days) >= 2:
        day_gaps = (
            unique_days
            .diff()
            .dt.total_seconds()
            /
            86400
        ).dropna()
    else:
        day_gaps = pd.Series(
            dtype=float
        )

    daily_counts = (
        ev
        .groupby(
            ev["acq_date"].dt.date
        )
        .size()
    )

    if len(daily_counts):
        max_daily = int(
            daily_counts.max()
        )

        mean_daily = float(
            daily_counts.mean()
        )

        median_daily = float(
            daily_counts.median()
        )
    else:
        max_daily = 0
        mean_daily = 0
        median_daily = 0

    monthly_counts = (
        ev["acq_date"]
        .dt.month
        .value_counts()
    )

    if len(monthly_counts):
        peak_month_detections = int(
            monthly_counts.max()
        )

        peak_month = int(
            monthly_counts.idxmax()
        )

        monthly_concentration = (
            peak_month_detections /
            n
        )
    else:
        peak_month_detections = 0
        peak_month = np.nan
        monthly_concentration = np.nan

    hours = (
        pd.to_numeric(
            ev["acq_time"],
            errors="coerce",
        )
        // 100
    ).dropna()

    # ------------------------------------------------------------
    # Spatial extent
    # ------------------------------------------------------------

    spatial_extent_km = (
        haversine_m(
            min_latitude,
            min_longitude,
            max_latitude,
            max_longitude,
        )
        /
        1000
    )

    # ------------------------------------------------------------
    # FRP behavior
    # ------------------------------------------------------------

    frp_valid = frp.dropna()

    if (
        len(frp_valid) >= 2
        and frp_valid.mean() != 0
    ):
        frp_cv = float(
            frp_valid.std() /
            frp_valid.mean()
        )
    else:
        frp_cv = 0.0

    # ------------------------------------------------------------
    # Final event feature dictionary
    # ------------------------------------------------------------

    return {
        "event_detection_count": n,

        "event_active_days": active_days,

        "event_mean_frp": safe_stat(
            frp,
            "mean",
        ),

        "event_median_frp": safe_stat(
            frp,
            "median",
        ),

        "event_max_frp": safe_stat(
            frp,
            "max",
        ),

        "event_std_frp": safe_stat(
            frp,
            "std",
            0.0,
        ),

        "event_min_frp": safe_stat(
            frp,
            "min",
        ),

        "event_sum_frp": (
            float(
                frp_valid.sum()
            )
            if len(frp_valid)
            else np.nan
        ),

        "event_mean_brightness": safe_stat(
            brightness,
            "mean",
        ),

        "event_max_brightness": safe_stat(
            brightness,
            "max",
        ),

        "event_median_brightness": safe_stat(
            brightness,
            "median",
        ),

        "event_std_brightness": safe_stat(
            brightness,
            "std",
            0.0,
        ),

        "event_min_brightness": safe_stat(
            brightness,
            "min",
        ),

        "event_mean_latitude":
            mean_latitude,

        "event_mean_longitude":
            mean_longitude,

        "event_min_latitude":
            min_latitude,

        "event_max_latitude":
            max_latitude,

        "event_min_longitude":
            min_longitude,

        "event_max_longitude":
            max_longitude,

        "event_500m_block_count":
            block_500_count,

        "event_1km_block_count":
            block_1km_count,

        "event_duration_hours":
            duration_hours,

        "event_duration_days":
            duration_days,

        "event_mean_detection_gap_hours": (
            float(gaps_hours.mean())
            if len(gaps_hours)
            else 0
        ),

        "event_median_detection_gap_hours": (
            float(gaps_hours.median())
            if len(gaps_hours)
            else 0
        ),

        "event_max_detection_gap_hours": (
            float(gaps_hours.max())
            if len(gaps_hours)
            else 0
        ),

        "event_min_detection_gap_hours": (
            float(gaps_hours.min())
            if len(gaps_hours)
            else 0
        ),

        "event_mean_active_day_gap": (
            float(day_gaps.mean())
            if len(day_gaps)
            else 0
        ),

        "event_median_active_day_gap": (
            float(day_gaps.median())
            if len(day_gaps)
            else 0
        ),

        "event_max_active_day_gap": (
            float(day_gaps.max())
            if len(day_gaps)
            else 0
        ),

        "event_min_active_day_gap": (
            float(day_gaps.min())
            if len(day_gaps)
            else 0
        ),

        "event_latitude_span_deg":
            latitude_span_deg,

        "event_longitude_span_deg":
            longitude_span_deg,

        "event_latitude_span_km":
            latitude_span_km,

        "event_longitude_span_km":
            longitude_span_km,

        "event_spatial_extent_km":
            spatial_extent_km,

        "event_spatial_extent_per_active_day_km": (
            spatial_extent_km / active_days
            if active_days
            else 0
        ),

        "event_detections_per_500m_block": (
            n / block_500_count
            if block_500_count
            else 0
        ),

        "event_detections_per_1km_block": (
            n / block_1km_count
            if block_1km_count
            else 0
        ),

        "event_detections_per_active_day": (
            n / active_days
            if active_days
            else 0
        ),

        "event_detections_per_hour": (
            n / duration_hours
            if duration_hours > 0
            else float(n)
        ),

        "event_peak_month_detections":
            peak_month_detections,

        "event_max_detections_one_day":
            max_daily,

        "event_mean_detections_active_day":
            mean_daily,

        "event_median_detections_active_day":
            median_daily,

        "event_active_months":
            active_months,

        "event_peak_month":
            peak_month,

        "event_monthly_concentration":
            monthly_concentration,

        "event_frp_range": (
            float(
                frp_valid.max()
                -
                frp_valid.min()
            )
            if len(frp_valid)
            else np.nan
        ),

        "event_frp_cv":
            frp_cv,

        "event_mean_acquisition_hour": (
            float(hours.mean())
            if len(hours)
            else np.nan
        ),

        "event_median_acquisition_hour": (
            float(hours.median())
            if len(hours)
            else np.nan
        ),

        "event_min_acquisition_hour": (
            float(hours.min())
            if len(hours)
            else np.nan
        ),

        "event_max_acquisition_hour": (
            float(hours.max())
            if len(hours)
            else np.nan
        ),

        "event_acquisition_hour_count":
            int(hours.nunique()),

        "event_satellite_count": (
            int(
                ev["satellite"].nunique()
            )
            if "satellite" in ev.columns
            else np.nan
        ),

        "event_day_fraction": (
            float(
                (
                    ev["daynight"] == "D"
                ).mean()
            )
            if "daynight" in ev.columns
            else np.nan
        ),

        "event_night_fraction": (
            float(
                (
                    ev["daynight"] == "N"
                ).mean()
            )
            if "daynight" in ev.columns
            else np.nan
        ),
    }
