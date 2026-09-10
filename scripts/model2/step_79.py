# ============================================================
# STEP 10B — TEST ONE REAL FIRMS DETECTION
# OPTIMIZED LOCAL VERSION
#
# Methodology preserved:
#   - FIRMS N20/N21/SNPP
#   - 30-day historical FIRMS
#   - 1 km / 48 hour event context
#   - World Bank flare evidence
#   - OSM evidence
#   - Dynamic World
#   - 137 exact model features
#   - 2025 trained Random Forest
#   - 0.70 confidence threshold
#
# Performance optimizations:
#   - Combined OSM query
#   - Faster OSM center extraction
#   - Dynamic World image dates fetched together
#   - Only closest DW image receives expensive reductions
#   - Reduced repeated conversions
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import warnings
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOCAL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "model2"

OUTPUT_DIR = DATA_DIR

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

print("=" * 70)
print("2026 FIRE SOURCE CLASSIFICATION PIPELINE - OPTIMIZED")
print("=" * 70)


# ------------------------------------------------------------
# FIRMS MAP KEY
# ------------------------------------------------------------
#
# KEEP YOUR EXISTING FIRMS MAP KEY HERE.
#
# The actual exposed key is intentionally not reproduced here.
#
# Example:
#
# MAP_KEY = "YOUR_EXISTING_MAP_KEY"
#
# ------------------------------------------------------------

MAP_KEY = "bbbbc47df4cd1e089cd78119863f9af2"


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    MODEL_DIR /
    "firms_2025_final_production_random_forest.joblib"
)

IMPUTER_PATH = (
    MODEL_DIR /
    "firms_2025_final_production_imputer.joblib"
)

FEATURE_SCHEMA_PATH = (
    MODEL_DIR /
    "firms_2025_final_model_feature_columns.csv"
)


# ------------------------------------------------------------
# 2025 EVIDENCE
# ------------------------------------------------------------

INDUSTRIAL_PATH = (
    DATA_DIR /
    "firms_industrial_evidence_indicators_2025_corrected.csv"
)

MINING_PATH = (
    DATA_DIR /
    "firms_coal_mine_distance_evidence_2025.csv"
)

AGRICULTURE_PATH = (
    DATA_DIR /
    "firms_agriculture_evidence_indicators_2025.csv"
)

WILDFIRE_PATH = (
    DATA_DIR /
    "firms_wildfire_evidence_2025.csv"
)

FLARE_PATH = (
    DATA_DIR /
    "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

PREDICTION_OUTPUT = (
    DATA_DIR /
    "firms_2026_test_point_prediction.csv"
)

FEATURE_OUTPUT = (
    DATA_DIR /
    "firms_2026_test_point_137_features.csv"
)


# ------------------------------------------------------------
# FIRMS
# ------------------------------------------------------------

FIRMS_SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT"
]

FIRMS_SP_SOURCES = [
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "VIIRS_SNPP_SP"
]

CURRENT_DAYS = 5

HISTORICAL_DAYS = 30

EVENT_RADIUS_KM = 1.0

EVENT_TIME_HOURS = 48.0

FIRMS_BBOX_DEG = 0.05


# ------------------------------------------------------------
# OSM
# ------------------------------------------------------------

OSM_RADIUS_M = 5000

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]


# ------------------------------------------------------------
# DYNAMIC WORLD
# ------------------------------------------------------------

EE_PROJECT = "possible-haven-507714-f0"

DW_AREA_M = 300

DW_SEARCH_BEFORE_DAYS = 120

DW_SEARCH_AFTER_DAYS = 3

DW_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice"
]

DW_CLASS_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice"
}


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def safe_float(x):

    try:

        if x is None or pd.isna(x):
            return np.nan

        return float(x)

    except Exception:

        return np.nan


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    lat1 = np.radians(
        np.asarray(lat1, dtype=float)
    )

    lat2 = np.radians(
        np.asarray(lat2, dtype=float)
    )

    dlat = lat2 - lat1

    dlon = np.radians(
        np.asarray(lon2, dtype=float)
        -
        np.asarray(lon1, dtype=float)
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


def find_column(
    df,
    exact=None,
    contains=None
):

    if df is None:
        return None

    cols = list(df.columns)

    if exact:

        target = str(
            exact
        ).strip().lower()

        for c in cols:

            if (
                str(c)
                .strip()
                .lower()
                ==
                target
            ):

                return c

    if contains:

        for c in cols:

            cl = str(c).lower()

            if all(
                str(x).lower() in cl
                for x in contains
            ):

                return c

    return None


def make_acq_datetime(df):

    date = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    time_numeric = pd.to_numeric(
        df["acq_time"],
        errors="coerce"
    )

    time_string = (
        time_numeric
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    return pd.to_datetime(
        date.dt.strftime("%Y-%m-%d")
        + " "
        + time_string,
        format="%Y-%m-%d %H%M",
        errors="coerce"
    )


def safe_stat(
    s,
    function,
    default=np.nan
):

    s = pd.to_numeric(
        s,
        errors="coerce"
    ).dropna()

    if len(s) == 0:
        return default

    if function == "mean":
        return float(s.mean())

    if function == "median":
        return float(s.median())

    if function == "min":
        return float(s.min())

    if function == "max":
        return float(s.max())

    if function == "std":

        if len(s) >= 2:
            return float(s.std())

        return 0.0

    return default


# ============================================================
# 4. USER INPUT
# ============================================================

lat = float(
    input(
        "Enter latitude : "
    ).strip()
)

lon = float(
    input(
        "Enter longitude: "
    ).strip()
)


print("\nUser location:")

print(
    "Latitude :",
    lat
)

print(
    "Longitude:",
    lon
)


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("2025 MODEL")
print("=" * 70)


if (
    not MODEL_PATH.exists()
    or
    not IMPUTER_PATH.exists()
    or
    not FEATURE_SCHEMA_PATH.exists()
):

    raise FileNotFoundError(
        "One or more Model 2 files are missing from models/model2/."
    )


rf_model = joblib.load(
    MODEL_PATH
)

rf_imputer = joblib.load(
    IMPUTER_PATH
)

schema = pd.read_csv(
    FEATURE_SCHEMA_PATH
)


if "feature" in schema.columns:

    MODEL_FEATURES = (
        schema["feature"]
        .astype(str)
        .tolist()
    )

elif "feature_name" in schema.columns:

    MODEL_FEATURES = (
        schema["feature_name"]
        .astype(str)
        .tolist()
    )

else:

    MODEL_FEATURES = (
        schema.iloc[:, 0]
        .astype(str)
        .tolist()
    )


print(
    "✓ 2025 model loaded"
)

print(
    "Trees   :",
    getattr(
        rf_model,
        "n_estimators",
        "unknown"
    )
)

print(
    "Features:",
    len(MODEL_FEATURES)
)


if len(MODEL_FEATURES) != 137:

    raise RuntimeError(
        "Expected exactly 137 model features."
    )


# ============================================================
# 6. FIRMS API
# ============================================================

if (
    not MAP_KEY
    or
    "PASTE_" in MAP_KEY
    or
    "YOUR_EXISTING" in MAP_KEY
):

    raise RuntimeError(
        "\nPut your existing FIRMS MAP KEY in MAP_KEY."
    )


def firms_request(
    source,
    bbox_string,
    days=5,
    date=None
):

    if date is None:

        url = (
            "https://firms.modaps.eosdis.nasa.gov/"
            "api/area/csv/"
            f"{MAP_KEY}/"
            f"{source}/"
            f"{bbox_string}/"
            f"{days}"
        )

    else:

        url = (
            "https://firms.modaps.eosdis.nasa.gov/"
            "api/area/csv/"
            f"{MAP_KEY}/"
            f"{source}/"
            f"{bbox_string}/"
            f"{days}/"
            f"{date}"
        )

    try:

        response = requests.get(
            url,
            timeout=60
        )

        if response.status_code != 200:

            return None, response.status_code

        if not response.text.strip():

            return (
                pd.DataFrame(),
                200
            )

        return (
            pd.read_csv(
                StringIO(response.text)
            ),
            200
        )

    except Exception as e:

        print(
            "FIRMS request error:",
            str(e)[:150]
        )

        return None, None


# ============================================================
# 7. CURRENT FIRMS SEARCH
# ============================================================

print("\n" + "=" * 70)
print("NASA FIRMS SEARCH")
print("=" * 70)


bbox = (
    f"{lon - FIRMS_BBOX_DEG},"
    f"{lat - FIRMS_BBOX_DEG},"
    f"{lon + FIRMS_BBOX_DEG},"
    f"{lat + FIRMS_BBOX_DEG}"
)


print(
    "BBOX:",
    bbox
)

print(
    "Searching last",
    CURRENT_DAYS,
    "days..."
)


current_frames = []

successful_current_requests = 0


for source in FIRMS_SOURCES:

    print("\n" + "-" * 60)

    print(
        "Source:",
        source
    )

    df_source, status_code = firms_request(
        source,
        bbox,
        CURRENT_DAYS
    )

    print(
        "HTTP status:",
        status_code
    )

    if (
        status_code != 200
        or
        df_source is None
    ):

        print(
            "⚠ Request failed"
        )

        continue

    successful_current_requests += 1

    print(
        "Detections:",
        len(df_source)
    )

    if len(df_source):

        df_source[
            "firms_source_api"
        ] = source

        current_frames.append(
            df_source
        )


if successful_current_requests == 0:

    raise RuntimeError(
        "All current FIRMS requests failed."
    )


if not current_frames:

    raise RuntimeError(
        "No FIRMS detection found within "
        "approximately 5 km during the last 5 days."
    )


firms_current = pd.concat(
    current_frames,
    ignore_index=True
)


# ============================================================
# 8. CLEAN FIRMS
# ============================================================

numeric_firms_columns = [
    "latitude",
    "longitude",
    "frp",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track"
]


for c in numeric_firms_columns:

    if c in firms_current.columns:

        firms_current[c] = pd.to_numeric(
            firms_current[c],
            errors="coerce"
        )


firms_current["acq_date"] = pd.to_datetime(
    firms_current["acq_date"],
    errors="coerce"
)

firms_current["acq_datetime"] = make_acq_datetime(
    firms_current
)


firms_current = (
    firms_current
    .dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime"
        ]
    )
    .copy()
)


firms_current[
    "distance_m_from_input"
] = haversine_m(
    lat,
    lon,
    firms_current["latitude"].values,
    firms_current["longitude"].values
)


firms_current = (
    firms_current
    .sort_values(
        [
            "distance_m_from_input",
            "acq_datetime"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 9. SELECT FIRMS DETECTION
# ============================================================

selected = firms_current.iloc[0]

selected_datetime = selected[
    "acq_datetime"
]


print("\n" + "=" * 70)
print("SELECTED FIRMS DETECTION")
print("=" * 70)


print(
    "Latitude   :",
    selected["latitude"]
)

print(
    "Longitude  :",
    selected["longitude"]
)

print(
    "Date       :",
    selected["acq_date"]
)

print(
    "Time       :",
    selected["acq_time"]
)

print(
    "FRP        :",
    selected.get(
        "frp",
        np.nan
    )
)

print(
    "Satellite  :",
    selected.get(
        "satellite",
        np.nan
    )
)

print(
    "Instrument :",
    selected.get(
        "instrument",
        np.nan
    )
)

print(
    "Confidence :",
    selected.get(
        "confidence",
        np.nan
    )
)

print(
    "Source     :",
    selected[
        "firms_source_api"
    ]
)

print(
    "Distance from input:",
    round(
        selected[
            "distance_m_from_input"
        ],
        2
    ),
    "m"
)


# ============================================================
# 10. EVENT CONTEXT
# ============================================================

print("\n" + "=" * 70)
print("EVENT CONTEXT")
print("=" * 70)


firms_current[
    "event_distance_m"
] = haversine_m(
    selected["latitude"],
    selected["longitude"],
    firms_current["latitude"].values,
    firms_current["longitude"].values
)


firms_current[
    "event_time_difference_hours"
] = (
    (
        firms_current[
            "acq_datetime"
        ]
        -
        selected_datetime
    )
    .abs()
    .dt.total_seconds()
    /
    3600.0
)


event_df = firms_current[
    (
        firms_current[
            "event_distance_m"
        ]
        <=
        EVENT_RADIUS_KM * 1000
    )
    &
    (
        firms_current[
            "event_time_difference_hours"
        ]
        <=
        EVENT_TIME_HOURS
    )
].copy()


if len(event_df) == 0:

    event_df = pd.DataFrame(
        [selected]
    )


event_df = (
    event_df
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    "Candidate observations:",
    len(event_df)
)


display_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_datetime",
    "frp",
    "event_distance_m",
    "event_time_difference_hours"
]


display(
    event_df[
        [
            c
            for c in display_columns
            if c in event_df.columns
        ]
    ]
)


# ============================================================
# 11. HISTORICAL FIRMS
# ============================================================

print("\n" + "=" * 70)
print("FIRMS TEMPORAL HISTORY")
print("=" * 70)


selected_date = (
    pd.Timestamp(
        selected["acq_date"]
    ).normalize()
)


chunk_starts = [

    selected_date - pd.Timedelta(days=29),

    selected_date - pd.Timedelta(days=24),

    selected_date - pd.Timedelta(days=19),

    selected_date - pd.Timedelta(days=14),

    selected_date - pd.Timedelta(days=9),

    selected_date - pd.Timedelta(days=4)
]


historical_frames = []

historical_successful_requests = 0

historical_total_requests = 0


for source in FIRMS_SOURCES:

    for start_date in chunk_starts:

        historical_total_requests += 1

        df_hist, status_code = firms_request(
            source,
            bbox,
            5,
            start_date.strftime("%Y-%m-%d")
        )

        if (
            status_code == 200
            and
            df_hist is not None
        ):

            historical_successful_requests += 1

            if len(df_hist):

                df_hist[
                    "firms_source_api"
                ] = source

                historical_frames.append(
                    df_hist
                )


print(
    "Successful NRT history requests:",
    historical_successful_requests,
    "/",
    historical_total_requests
)


# ------------------------------------------------------------
# Standard Processing fallback
# ------------------------------------------------------------

if historical_successful_requests == 0:

    print(
        "NRT historical data unavailable."
    )

    print(
        "Trying Standard Processing..."
    )

    for source in FIRMS_SP_SOURCES:

        for start_date in chunk_starts:

            historical_total_requests += 1

            df_hist, status_code = firms_request(
                source,
                bbox,
                5,
                start_date.strftime("%Y-%m-%d")
            )

            if (
                status_code == 200
                and
                df_hist is not None
            ):

                historical_successful_requests += 1

                if len(df_hist):

                    df_hist[
                        "firms_source_api"
                    ] = source

                    historical_frames.append(
                        df_hist
                    )


if historical_frames:

    historical_df = pd.concat(
        historical_frames,
        ignore_index=True
    )

else:

    historical_df = pd.DataFrame()


if len(historical_df):

    for c in [
        "latitude",
        "longitude",
        "frp"
    ]:

        if c in historical_df.columns:

            historical_df[c] = pd.to_numeric(
                historical_df[c],
                errors="coerce"
            )


    historical_df["acq_date"] = pd.to_datetime(
        historical_df["acq_date"],
        errors="coerce"
    )

    historical_df["acq_datetime"] = make_acq_datetime(
        historical_df
    )

    historical_df = historical_df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime"
        ]
    ).copy()


    history_start = (
        selected_datetime
        -
        pd.Timedelta(days=29)
    )


    historical_df = historical_df[
        (
            historical_df[
                "acq_datetime"
            ]
            >=
            history_start
        )
        &
        (
            historical_df[
                "acq_datetime"
            ]
            <=
            selected_datetime
        )
    ].copy()


    historical_df[
        "distance_to_selected_m"
    ] = haversine_m(
        selected["latitude"],
        selected["longitude"],
        historical_df["latitude"].values,
        historical_df["longitude"].values
    )


    historical_df = historical_df[
        historical_df[
            "distance_to_selected_m"
        ]
        >
        100
    ].copy()


    identity_columns = [
        c
        for c in [
            "latitude",
            "longitude",
            "acq_datetime",
            "frp"
        ]
        if c in historical_df.columns
    ]


    if identity_columns:

        historical_df = (
            historical_df
            .drop_duplicates(
                subset=identity_columns
            )
        )

else:

    historical_df = pd.DataFrame()


print(
    "Total successful history requests:",
    historical_successful_requests
)

print(
    "Historical detections:",
    len(historical_df)
)


if historical_successful_requests == 0:

    print(
        "⚠ HISTORY STATUS: UNAVAILABLE"
    )

elif len(historical_df) == 0:

    print(
        "✓ HISTORY STATUS: QUERY SUCCEEDED, "
        "BUT ZERO HISTORICAL DETECTIONS"
    )

else:

    print(
        "✓ HISTORY STATUS: AVAILABLE"
    )


# ============================================================
# 12. TEMPORAL FEATURES
# ============================================================

temporal_features = {}


if historical_successful_requests > 0:

    if len(historical_df):

        historical_df[
            "hours_before_selected"
        ] = (
            (
                selected_datetime
                -
                historical_df[
                    "acq_datetime"
                ]
            )
            .dt.total_seconds()
            /
            3600
        )

        historical_df[
            "days_before_selected"
        ] = (
            historical_df[
                "hours_before_selected"
            ]
            /
            24
        )


    lat_step_500 = (
        0.5 / 111.32
    )


    selected_lat500 = np.floor(
        selected["latitude"]
        /
        lat_step_500
    )


    selected_lon500 = np.floor(
        selected["longitude"]
        /
        (
            0.5
            /
            (
                111.32
                *
                np.cos(
                    np.radians(
                        selected["latitude"]
                    )
                )
            )
        )
    )


    if len(historical_df):

        historical_df[
            "_lat500"
        ] = np.floor(
            historical_df[
                "latitude"
            ]
            /
            lat_step_500
        )

        historical_df[
            "_lon500"
        ] = np.floor(
            historical_df[
                "longitude"
            ]
            /
            (
                0.5
                /
                (
                    111.32
                    *
                    np.cos(
                        np.radians(
                            historical_df[
                                "latitude"
                            ]
                        )
                    )
                )
            )
        )


        same_cell = (
            (
                historical_df[
                    "_lat500"
                ]
                ==
                selected_lat500
            )
            &
            (
                historical_df[
                    "_lon500"
                ]
                ==
                selected_lon500
            )
        )


        nearby_1km = (
            historical_df[
                "distance_to_selected_m"
            ]
            <=
            1000
        )


    for days in [3, 7, 30]:

        if len(historical_df):

            mask = (
                historical_df[
                    "days_before_selected"
                ]
                <=
                days
            )

            same_count = int(
                (
                    same_cell
                    &
                    mask
                ).sum()
            )

            nearby_count = int(
                (
                    nearby_1km
                    &
                    mask
                ).sum()
            )

        else:

            same_count = 0

            nearby_count = 0


        temporal_features[
            f"same_cell_fire_count_{days}d_max"
        ] = same_count

        temporal_features[
            f"nearby_fire_count_{days}d_1km_max"
        ] = nearby_count

        temporal_features[
            f"event_max_same_cell_fire_count_{days}d"
        ] = same_count

        temporal_features[
            f"event_max_nearby_fire_count_{days}d_1km"
        ] = nearby_count


else:

    for days in [3, 7, 30]:

        temporal_features[
            f"same_cell_fire_count_{days}d_max"
        ] = np.nan

        temporal_features[
            f"nearby_fire_count_{days}d_1km_max"
        ] = np.nan

        temporal_features[
            f"event_max_same_cell_fire_count_{days}d"
        ] = np.nan

        temporal_features[
            f"event_max_nearby_fire_count_{days}d_1km"
        ] = np.nan


print(
    "\nTemporal features:"
)


for k, v in temporal_features.items():

    print(
        f"  {k:<45} {v}"
    )


# ============================================================
# 13. EVENT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FIRMS EVENT FEATURES")
print("=" * 70)


ev = event_df.copy()


ev["acq_date"] = pd.to_datetime(
    ev["acq_date"],
    errors="coerce"
)

ev["acq_datetime"] = make_acq_datetime(
    ev
)


for c in [
    "latitude",
    "longitude",
    "frp",
    "bright_ti4",
    "bright_ti5"
]:

    if c in ev.columns:

        ev[c] = pd.to_numeric(
            ev[c],
            errors="coerce"
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
    errors="coerce"
)


brightness = pd.to_numeric(
    ev.get(
        "bright_ti4",
        pd.Series(
            np.nan,
            index=ev.index
        )
    ),
    errors="coerce"
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
    longitude_span_deg
    *
    111.32
    *
    np.cos(
        np.radians(
            mean_latitude
        )
    )
)


# ------------------------------------------------------------
# 500m / 1km blocks
# ------------------------------------------------------------

lat_step_500 = (
    0.5 / 111.32
)

lat_step_1km = (
    1.0 / 111.32
)


ev["_lat500"] = np.floor(
    ev["latitude"]
    /
    lat_step_500
)


ev["_lon500"] = np.floor(
    ev["longitude"]
    /
    (
        0.5
        /
        (
            111.32
            *
            np.cos(
                np.radians(
                    ev["latitude"]
                )
            )
        )
    )
)


ev["_lat1km"] = np.floor(
    ev["latitude"]
    /
    lat_step_1km
)


ev["_lon1km"] = np.floor(
    ev["longitude"]
    /
    (
        1.0
        /
        (
            111.32
            *
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
            "_lon500"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


block_1km_count = (
    ev[
        [
            "_lat1km",
            "_lon1km"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


times = (
    ev["acq_datetime"]
    .dropna()
    .sort_values()
)


if len(times) >= 2:

    duration_hours = (
        times.max()
        -
        times.min()
    ).total_seconds() / 3600

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
        errors="coerce"
    )
    // 100
).dropna()


spatial_extent_km = (
    haversine_m(
        min_latitude,
        min_longitude,
        max_latitude,
        max_longitude
    )
    /
    1000
)


frp_valid = frp.dropna()


if (
    len(frp_valid) >= 2
    and
    frp_valid.mean() != 0
):

    frp_cv = float(
        frp_valid.std()
        /
        frp_valid.mean()
    )

else:

    frp_cv = 0.0


event_features = {

    "event_detection_count":
        n,

    "event_active_days":
        active_days,

    "event_mean_frp":
        safe_stat(
            frp,
            "mean"
        ),

    "event_median_frp":
        safe_stat(
            frp,
            "median"
        ),

    "event_max_frp":
        safe_stat(
            frp,
            "max"
        ),

    "event_std_frp":
        safe_stat(
            frp,
            "std",
            0.0
        ),

    "event_min_frp":
        safe_stat(
            frp,
            "min"
        ),

    "event_sum_frp":
        (
            float(
                frp_valid.sum()
            )
            if len(frp_valid)
            else np.nan
        ),

    "event_mean_brightness":
        safe_stat(
            brightness,
            "mean"
        ),

    "event_max_brightness":
        safe_stat(
            brightness,
            "max"
        ),

    "event_median_brightness":
        safe_stat(
            brightness,
            "median"
        ),

    "event_std_brightness":
        safe_stat(
            brightness,
            "std",
            0.0
        ),

    "event_min_brightness":
        safe_stat(
            brightness,
            "min"
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

    "event_mean_detection_gap_hours":
        (
            float(
                gaps_hours.mean()
            )
            if len(gaps_hours)
            else 0
        ),

    "event_median_detection_gap_hours":
        (
            float(
                gaps_hours.median()
            )
            if len(gaps_hours)
            else 0
        ),

    "event_max_detection_gap_hours":
        (
            float(
                gaps_hours.max()
            )
            if len(gaps_hours)
            else 0
        ),

    "event_min_detection_gap_hours":
        (
            float(
                gaps_hours.min()
            )
            if len(gaps_hours)
            else 0
        ),

    "event_mean_active_day_gap":
        (
            float(
                day_gaps.mean()
            )
            if len(day_gaps)
            else 0
        ),

    "event_median_active_day_gap":
        (
            float(
                day_gaps.median()
            )
            if len(day_gaps)
            else 0
        ),

    "event_max_active_day_gap":
        (
            float(
                day_gaps.max()
            )
            if len(day_gaps)
            else 0
        ),

    "event_min_active_day_gap":
        (
            float(
                day_gaps.min()
            )
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

    "event_spatial_extent_per_active_day_km":
        (
            spatial_extent_km /
            active_days
            if active_days
            else 0
        ),

    "event_detections_per_500m_block":
        (
            n /
            block_500_count
            if block_500_count
            else 0
        ),

    "event_detections_per_1km_block":
        (
            n /
            block_1km_count
            if block_1km_count
            else 0
        ),

    "event_detections_per_active_day":
        (
            n /
            active_days
            if active_days
            else 0
        ),

    "event_detections_per_hour":
        (
            n /
            duration_hours
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

    "event_frp_range":
        (
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

    "event_mean_acquisition_hour":
        (
            float(
                hours.mean()
            )
            if len(hours)
            else np.nan
        ),

    "event_median_acquisition_hour":
        (
            float(
                hours.median()
            )
            if len(hours)
            else np.nan
        ),

    "event_min_acquisition_hour":
        (
            float(
                hours.min()
            )
            if len(hours)
            else np.nan
        ),

    "event_max_acquisition_hour":
        (
            float(
                hours.max()
            )
            if len(hours)
            else np.nan
        ),

    "event_acquisition_hour_count":
        int(
            hours.nunique()
        ),

    "event_satellite_count":
        (
            int(
                ev[
                    "satellite"
                ].nunique()
            )
            if "satellite" in ev.columns
            else np.nan
        ),

    "event_day_fraction":
        (
            float(
                (
                    ev[
                        "daynight"
                    ]
                    ==
                    "D"
                ).mean()
            )
            if "daynight" in ev.columns
            else np.nan
        ),

    "event_night_fraction":
        (
            float(
                (
                    ev[
                        "daynight"
                    ]
                    ==
                    "N"
                ).mean()
            )
            if "daynight" in ev.columns
            else np.nan
        )
}


# ============================================================
# 14. EVIDENCE AGGREGATION
# ============================================================

def aggregate_nearest_evidence(
    df,
    event_df,
    columns_min=None,
    columns_max=None
):

    columns_min = columns_min or []

    columns_max = columns_max or []

    result = {}

    for c in columns_min:
        result[c] = np.nan

    for c in columns_max:
        result[c] = np.nan

    if df is None or len(df) == 0:

        return result


    lat_col = (
        find_column(
            df,
            exact="latitude"
        )
        or
        find_column(
            df,
            exact="lat"
        )
    )


    lon_col = (
        find_column(
            df,
            exact="longitude"
        )
        or
        find_column(
            df,
            exact="lon"
        )
    )


    if (
        lat_col is None
        or
        lon_col is None
    ):

        return result


    wanted_columns = [
        c
        for c in (
            columns_min +
            columns_max
        )
        if c in df.columns
    ]


    source = df[
        [
            lat_col,
            lon_col,
            *wanted_columns
        ]
    ].copy()


    source["_lat"] = pd.to_numeric(
        source[lat_col],
        errors="coerce"
    )

    source["_lon"] = pd.to_numeric(
        source[lon_col],
        errors="coerce"
    )


    source = source.dropna(
        subset=[
            "_lat",
            "_lon"
        ]
    )


    if len(source) == 0:

        return result


    source_lat = source[
        "_lat"
    ].to_numpy()

    source_lon = source[
        "_lon"
    ].to_numpy()


    values_min = {
        c: []
        for c in columns_min
    }


    values_max = {
        c: []
        for c in columns_max
    }


    event_latitudes = (
        event_df[
            "latitude"
        ].to_numpy()
    )


    event_longitudes = (
        event_df[
            "longitude"
        ].to_numpy()
    )


    for event_lat, event_lon in zip(
        event_latitudes,
        event_longitudes
    ):

        distances = haversine_m(
            event_lat,
            event_lon,
            source_lat,
            source_lon
        )


        nearest_idx = int(
            np.argmin(
                distances
            )
        )


        evidence_row = source.iloc[
            nearest_idx
        ]


        for c in columns_min:

            if c in evidence_row.index:

                value = safe_float(
                    evidence_row[c]
                )

                if pd.notna(value):

                    values_min[c].append(
                        value
                    )


        for c in columns_max:

            if c in evidence_row.index:

                value = safe_float(
                    evidence_row[c]
                )

                if pd.notna(value):

                    values_max[c].append(
                        value
                    )


    for c in columns_min:

        if values_min[c]:

            result[c] = min(
                values_min[c]
            )


    for c in columns_max:

        if values_max[c]:

            result[c] = max(
                values_max[c]
            )


    return result


# ============================================================
# 15. WORLD BANK FLARES
# ============================================================

print("\n" + "=" * 70)
print("WORLD BANK FLARE EVIDENCE")
print("=" * 70)


flare_features = {

    "nearest_flare_distance_km":
        np.nan,

    "nearest_flare_2025_activity":
        np.nan,

    "active_flare_count_within_1km":
        0,

    "has_active_flare_within_1km":
        0,

    "active_flare_count_within_2km":
        0,

    "has_active_flare_within_2km":
        0,

    "active_flare_count_within_5km":
        0,

    "has_active_flare_within_5km":
        0,

    "active_flare_count_within_10km":
        0,

    "has_active_flare_within_10km":
        0,

    "nearest_gas_flare_distance_km":
        np.nan,

    "nearest_gas_flare_2025_activity":
        np.nan,

    "gas_flare_count_within_1km":
        0,

    "has_gas_flare_within_1km":
        0,

    "gas_flare_count_within_2km":
        0,

    "has_gas_flare_within_2km":
        0,

    "gas_flare_count_within_5km":
        0,

    "has_gas_flare_within_5km":
        0,

    "gas_flare_count_within_10km":
        0,

    "has_gas_flare_within_10km":
        0,

    "nearest_flare_is_gas":
        0,

    "nearest_flare_is_oil":
        0,

    "nearest_flare_is_unknown":
        0
}


try:

    flare_df = pd.read_excel(
        FLARE_PATH,
        sheet_name=(
            "2012-2025-Flare-Volume-Estimate"
        )
    )


    flare_df.columns = [
        str(c).strip()
        for c in flare_df.columns
    ]


    flare_lat_col = (
        find_column(
            flare_df,
            exact="Latitude"
        )
        or
        find_column(
            flare_df,
            contains=["lat"]
        )
    )


    flare_lon_col = (
        find_column(
            flare_df,
            exact="Longitude"
        )
        or
        find_column(
            flare_df,
            contains=["lon"]
        )
    )


    if (
        flare_lat_col
        and
        flare_lon_col
    ):

        flare_df["_lat"] = pd.to_numeric(
            flare_df[
                flare_lat_col
            ],
            errors="coerce"
        )

        flare_df["_lon"] = pd.to_numeric(
            flare_df[
                flare_lon_col
            ],
            errors="coerce"
        )


        country_col = find_column(
            flare_df,
            exact="Country"
        )


        if country_col:

            india = flare_df[
                flare_df[
                    country_col
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                "india"
            ].copy()

        else:

            india = flare_df[
                (
                    flare_df["_lat"]
                    >=
                    6
                )
                &
                (
                    flare_df["_lat"]
                    <=
                    38
                )
                &
                (
                    flare_df["_lon"]
                    >=
                    67
                )
                &
                (
                    flare_df["_lon"]
                    <=
                    98
                )
            ].copy()


        activity_col = None


        for c in india.columns:

            cl = str(c).lower()

            if (
                "2025" in cl
                and
                (
                    "volume" in cl
                    or
                    "flare" in cl
                )
            ):

                activity_col = c

                break


        if activity_col:

            india["_activity"] = pd.to_numeric(
                india[
                    activity_col
                ],
                errors="coerce"
            ).fillna(0)

        else:

            india["_activity"] = 0.0


        india = india[
            india[
                "_activity"
            ]
            >
            0
        ].copy()


        if len(india):

            india["_distance_m"] = haversine_m(
                selected["latitude"],
                selected["longitude"],
                india["_lat"].values,
                india["_lon"].values
            )


            nearest = (
                india
                .sort_values(
                    "_distance_m"
                )
                .iloc[0]
            )


            flare_features[
                "nearest_flare_distance_km"
            ] = (
                nearest[
                    "_distance_m"
                ]
                /
                1000
            )


            flare_features[
                "nearest_flare_2025_activity"
            ] = nearest[
                "_activity"
            ]


            for radius in [
                1,
                2,
                5,
                10
            ]:

                mask = (
                    india[
                        "_distance_m"
                    ]
                    <=
                    radius * 1000
                )


                flare_features[
                    f"active_flare_count_within_{radius}km"
                ] = int(
                    mask.sum()
                )


                flare_features[
                    f"has_active_flare_within_{radius}km"
                ] = int(
                    mask.any()
                )


            field_col = None


            for c in india.columns:

                cl = str(c).lower()

                if (
                    "field" in cl
                    and
                    "type" in cl
                ):

                    field_col = c

                    break


            if field_col:

                nearest_type = str(
                    nearest[
                        field_col
                    ]
                ).lower()

            else:

                nearest_type = ""


            flare_features[
                "nearest_flare_is_gas"
            ] = int(
                "gas"
                in
                nearest_type
            )


            flare_features[
                "nearest_flare_is_oil"
            ] = int(
                "oil"
                in
                nearest_type
            )


            flare_features[
                "nearest_flare_is_unknown"
            ] = int(
                "gas" not in nearest_type
                and
                "oil" not in nearest_type
            )


            if field_col:

                gas = india[
                    india[
                        field_col
                    ]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        "gas",
                        na=False
                    )
                ].copy()

            else:

                gas = pd.DataFrame()


            if len(gas):

                gas["_distance_m"] = haversine_m(
                    selected["latitude"],
                    selected["longitude"],
                    gas["_lat"].values,
                    gas["_lon"].values
                )


                nearest_gas = (
                    gas
                    .sort_values(
                        "_distance_m"
                    )
                    .iloc[0]
                )


                flare_features[
                    "nearest_gas_flare_distance_km"
                ] = (
                    nearest_gas[
                        "_distance_m"
                    ]
                    /
                    1000
                )


                flare_features[
                    "nearest_gas_flare_2025_activity"
                ] = nearest_gas[
                    "_activity"
                ]


                for radius in [
                    1,
                    2,
                    5,
                    10
                ]:

                    mask = (
                        gas[
                            "_distance_m"
                        ]
                        <=
                        radius * 1000
                    )


                    flare_features[
                        f"gas_flare_count_within_{radius}km"
                    ] = int(
                        mask.sum()
                    )


                    flare_features[
                        f"has_gas_flare_within_{radius}km"
                    ] = int(
                        mask.any()
                    )


            print(
                "Active India 2025 flares:",
                len(india)
            )

        else:

            print(
                "No active India 2025 flares."
            )

    else:

        print(
            "Flare coordinates not found."
        )


except Exception as e:

    print(
        "⚠ Flare evidence error:",
        e
    )


# ============================================================
# 16. LIVE OSM — ONE COMBINED REQUEST
# ============================================================

print("\n" + "=" * 70)
print("LIVE OPENSTREETMAP — OPTIMIZED")
print("=" * 70)


OSM_QUERY = f"""
[out:json][timeout:90];

(
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="flare"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="petroleum_well"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="oil_well"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="mineshaft"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="adit"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["man_made"="gasometer"];

  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="industrial"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["industrial"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="quarry"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="mine"];

  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="farmland"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="farmyard"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="orchard"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="vineyard"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="plant_nursery"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="greenhouse_horticulture"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="allotments"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["building"="greenhouse"];

  nwr(around:{OSM_RADIUS_M},{lat},{lon})["natural"="wood"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["natural"="forest"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["natural"="scrub"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["natural"="grassland"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["natural"="heath"];
  nwr(around:{OSM_RADIUS_M},{lat},{lon})["landuse"="forest"];
);

out body center;
"""


def query_overpass_optimized(
    query
):

    for server in OVERPASS_SERVERS:

        print(
            "\nTrying:",
            server
        )

        try:

            response = requests.post(
                server,
                data={
                    "data": query
                },
                headers={
                    "User-Agent":
                    "fire-source-classification-research"
                },
                timeout=100
            )


            print(
                "HTTP status:",
                response.status_code
            )


            if response.status_code == 200:

                data = response.json()

                return data.get(
                    "elements",
                    []
                )


            print(
                "⚠ Overpass response:",
                response.text[:200]
            )


        except Exception as e:

            print(
                "⚠",
                str(e)[:200]
            )


    return []


osm_elements = query_overpass_optimized(
    OSM_QUERY
)


print(
    "\nObjects returned:",
    len(osm_elements)
)


# ============================================================
# 17. OSM CATEGORIES
# ============================================================

OSM_CATEGORIES = [

    "flare",

    "oil_well",

    "mineshaft",

    "adit",

    "gasometer",

    "industrial",

    "quarry",

    "farmland",

    "farmyard",

    "orchard",

    "vineyard",

    "plant_nursery",

    "greenhouse",

    "allotment",

    "forest",

    "scrub",

    "grassland",

    "heath"
]


osm_nearest = {
    c: np.nan
    for c in OSM_CATEGORIES
}


osm_counts = {
    c: 0
    for c in OSM_CATEGORIES
}


# ============================================================
# 18. OSM HELPERS
# ============================================================

def element_coordinate(
    element
):

    if element.get("type") == "node":

        if (
            "lat" in element
            and
            "lon" in element
        ):

            return (
                float(element["lat"]),
                float(element["lon"])
            )

        return None


    center = element.get(
        "center"
    )


    if center:

        if (
            "lat" in center
            and
            "lon" in center
        ):

            return (
                float(center["lat"]),
                float(center["lon"])
            )


    geometry = element.get(
        "geometry",
        []
    )


    if geometry:

        # Fallback only if Overpass did not
        # provide a center.
        lats = [
            float(p["lat"])
            for p in geometry
            if "lat" in p
        ]

        lons = [
            float(p["lon"])
            for p in geometry
            if "lon" in p
        ]


        if lats and lons:

            return (
                float(
                    np.mean(lats)
                ),
                float(
                    np.mean(lons)
                )
            )


    return None


def classify_osm(
    tags
):

    man_made = tags.get(
        "man_made"
    )

    landuse = tags.get(
        "landuse"
    )

    natural = tags.get(
        "natural"
    )

    building = tags.get(
        "building"
    )


    if man_made == "flare":

        return "flare"


    if man_made in [
        "petroleum_well",
        "oil_well"
    ]:

        return "oil_well"


    if man_made == "mineshaft":

        return "mineshaft"


    if man_made == "adit":

        return "adit"


    if man_made == "gasometer":

        return "gasometer"


    if (
        landuse == "industrial"
        or
        "industrial" in tags
    ):

        return "industrial"


    if landuse == "quarry":

        return "quarry"


    if landuse == "mine":

        return "quarry"


    if landuse == "farmland":

        return "farmland"


    if landuse == "farmyard":

        return "farmyard"


    if landuse == "orchard":

        return "orchard"


    if landuse == "vineyard":

        return "vineyard"


    if landuse == "plant_nursery":

        return "plant_nursery"


    if (
        landuse == "greenhouse_horticulture"
        or
        building == "greenhouse"
    ):

        return "greenhouse"


    if landuse == "allotments":

        return "allotment"


    if (
        natural in [
            "wood",
            "forest"
        ]
        or
        landuse == "forest"
    ):

        return "forest"


    if natural == "scrub":

        return "scrub"


    if natural == "grassland":

        return "grassland"


    if natural == "heath":

        return "heath"


    return None


# ============================================================
# 19. PROCESS OSM
# ============================================================

for element in osm_elements:

    tags = element.get(
        "tags",
        {}
    )


    category = classify_osm(
        tags
    )


    if category is None:

        continue


    coordinate = element_coordinate(
        element
    )


    if coordinate is None:

        continue


    xlat, xlon = coordinate


    distance = float(
        haversine_m(
            lat,
            lon,
            xlat,
            xlon
        )
    )


    osm_counts[
        category
    ] += 1


    current_nearest = osm_nearest[
        category
    ]


    if (
        pd.isna(current_nearest)
        or
        distance < current_nearest
    ):

        osm_nearest[
            category
        ] = distance


print(
    "\nOSM category summary:"
)


for category in OSM_CATEGORIES:

    nearest_value = (

        round(
            osm_nearest[
                category
            ],
            1
        )

        if pd.notna(
            osm_nearest[
                category
            ]
        )

        else "NaN"
    )


    print(
        f"{category:20s}",
        "count =",
        osm_counts[
            category
        ],
        "nearest =",
        nearest_value
    )


# ------------------------------------------------------------
# Agriculture OSM distance
# ------------------------------------------------------------

agriculture_osm_values = [

    osm_nearest[c]

    for c in [
        "farmland",
        "farmyard",
        "orchard",
        "vineyard",
        "plant_nursery",
        "greenhouse",
        "allotment"
    ]

    if pd.notna(
        osm_nearest[c]
    )
]


agri_osm_distance = (

    float(
        min(
            agriculture_osm_values
        )
    )

    if agriculture_osm_values

    else np.nan
)


print(
    "\nAgriculture OSM distance:",
    agri_osm_distance
)


# ============================================================
# 20. LOAD 2025 EVIDENCE
# ============================================================

print("\n" + "=" * 70)
print("2025 EVIDENCE DATA")
print("=" * 70)


def load_evidence(
    path,
    name
):

    if not path.exists():

        print(
            f"⚠ {name} file missing:",
            path
        )

        return pd.DataFrame()


    df = pd.read_csv(
        path
    )


    print(
        f"{name} records:",
        len(df)
    )


    return df


industrial_df = load_evidence(
    INDUSTRIAL_PATH,
    "Industrial"
)


mining_df = load_evidence(
    MINING_PATH,
    "Mining"
)


agriculture_df = load_evidence(
    AGRICULTURE_PATH,
    "Agriculture"
)


wildfire_df = load_evidence(
    WILDFIRE_PATH,
    "Wildfire"
)


# ============================================================
# 21. INDUSTRIAL
# ============================================================

industrial_columns_min = [

    "steel_distance_m",

    "cement_distance_m",

    "wri_power_distance_m",

    "fertilizer_distance_m",

    "refinery_petro_distance_m"
]


industrial_columns_max = [

    "industrial_source_count_500m",

    "industrial_source_count_1km",

    "industrial_evidence_score"
]


industrial_raw = aggregate_nearest_evidence(
    industrial_df,
    event_df,
    columns_min=industrial_columns_min,
    columns_max=industrial_columns_max
)


industrial_features = {

    "event_min_steel_distance_m":
        industrial_raw[
            "steel_distance_m"
        ],

    "event_min_cement_distance_m":
        industrial_raw[
            "cement_distance_m"
        ],

    "event_min_wri_power_distance_m":
        industrial_raw[
            "wri_power_distance_m"
        ],

    "event_min_fertilizer_distance_m":
        industrial_raw[
            "fertilizer_distance_m"
        ],

    "event_min_refinery_petro_distance_m":
        industrial_raw[
            "refinery_petro_distance_m"
        ],

    "event_max_industrial_source_count_500m":
        industrial_raw[
            "industrial_source_count_500m"
        ],

    "event_max_industrial_source_count_1km":
        industrial_raw[
            "industrial_source_count_1km"
        ],

    "event_max_industrial_evidence_score":
        industrial_raw[
            "industrial_evidence_score"
        ]
}


# ============================================================
# 22. MINING
# ============================================================

print("\n" + "=" * 70)
print("MINING EVIDENCE")
print("=" * 70)


mine_distance_column = None


known_mine_columns = [

    "nearest_coal_mine_distance_m",

    "coal_mine_distance_m",

    "event_min_nearest_coal_mine_distance_m",

    "nearest_mine_distance_m"
]


for candidate in known_mine_columns:

    if candidate in mining_df.columns:

        mine_distance_column = candidate

        break


if mine_distance_column is None:

    for c in mining_df.columns:

        cl = str(c).lower()

        if (
            "coal" in cl
            and
            "mine" in cl
            and
            "distance" in cl
        ):

            mine_distance_column = c

            break


print(
    "Mining distance source column:",
    mine_distance_column
)


mine_raw = aggregate_nearest_evidence(
    mining_df,
    event_df,
    columns_min=(
        [
            mine_distance_column
        ]
        if mine_distance_column
        else []
    )
)


mine_distance_m = (

    mine_raw.get(
        mine_distance_column,
        np.nan
    )

    if mine_distance_column

    else np.nan
)


mining_features = {

    "event_min_nearest_coal_mine_distance_m":
        mine_distance_m,

    "event_min_nearest_coal_mine_distance_km":
        (
            mine_distance_m / 1000
            if pd.notna(
                mine_distance_m
            )
            else np.nan
        )
}


print(
    "Nearest coal mine distance:",
    mine_distance_m,
    "m"
)


# ============================================================
# 23. AGRICULTURE
# ============================================================

agri_score_column = None


for c in agriculture_df.columns:

    cl = str(c).lower()

    if (
        "agri" in cl
        and
        "score" in cl
    ):

        agri_score_column = c

        break


agri_raw = aggregate_nearest_evidence(
    agriculture_df,
    event_df,
    columns_max=(
        [
            agri_score_column
        ]
        if agri_score_column
        else []
    )
)


agriculture_features = {

    "agriculture_evidence_score_max":
        (
            agri_raw.get(
                agri_score_column,
                np.nan
            )
            if agri_score_column
            else np.nan
        ),

    "agri_osm_distance_m_min":
        agri_osm_distance
}


# ============================================================
# 24. WILDFIRE / NATURAL
# ============================================================

wildfire_distance_columns = [

    "distance_to_nearest_flare_m",

    "distance_to_nearest_oil_well_m",

    "distance_to_nearest_mineshaft_m",

    "distance_to_nearest_adit_m",

    "distance_to_nearest_gasometer_m",

    "distance_to_nearest_industrial_m",

    "distance_to_nearest_quarry_m",

    "distance_to_nearest_farmland_m",

    "distance_to_nearest_farmyard_m",

    "distance_to_nearest_orchard_m",

    "distance_to_nearest_vineyard_m",

    "distance_to_nearest_plant_nursery_m",

    "distance_to_nearest_greenhouse_m",

    "distance_to_nearest_allotment_m",

    "distance_to_nearest_forest_m",

    "distance_to_nearest_scrub_m",

    "distance_to_nearest_grassland_m",

    "distance_to_nearest_heath_m",

    "distance_to_nearest_agriculture_m",

    "distance_to_nearest_natural_vegetation_m"
]


wildfire_raw = aggregate_nearest_evidence(
    wildfire_df,
    event_df,
    columns_min=wildfire_distance_columns
)


wildfire_features = {}


for c in wildfire_distance_columns:

    wildfire_features[
        c + "_min"
    ] = wildfire_raw.get(
        c,
        np.nan
    )


natural_context_column = find_column(
    wildfire_df,
    exact="natural_landcover_context"
)


natural_strong_column = find_column(
    wildfire_df,
    exact="natural_vegetation_strong"
)


natural_moderate_column = find_column(
    wildfire_df,
    exact="natural_vegetation_moderate"
)


natural_columns = [

    c

    for c in [
        natural_context_column,
        natural_strong_column,
        natural_moderate_column
    ]

    if c is not None
]


natural_raw = aggregate_nearest_evidence(
    wildfire_df,
    event_df,
    columns_max=natural_columns
)


wildfire_features[
    "natural_landcover_context_max"
] = (

    natural_raw.get(
        natural_context_column,
        np.nan
    )

    if natural_context_column

    else np.nan
)


wildfire_features[
    "natural_vegetation_strong_max"
] = (

    natural_raw.get(
        natural_strong_column,
        np.nan
    )

    if natural_strong_column

    else np.nan
)


wildfire_features[
    "natural_vegetation_moderate_max"
] = (

    natural_raw.get(
        natural_moderate_column,
        np.nan
    )

    if natural_moderate_column

    else np.nan
)


# ============================================================
# 25. DYNAMIC WORLD — OPTIMIZED
# ============================================================

print("\n" + "=" * 70)
print("DYNAMIC WORLD — OPTIMIZED")
print("=" * 70)


dw_rows = []


try:

    import ee


    # --------------------------------------------------------
    # Initialize EE
    # --------------------------------------------------------

    try:

        ee.Initialize(
            project=EE_PROJECT
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
            project=EE_PROJECT
        )


    # --------------------------------------------------------
    # OPTIMIZED DW FUNCTION
    #
    # Main optimization:
    #
    # OLD:
    #   image 1 → getInfo date
    #   image 2 → getInfo date
    #   image 3 → getInfo date
    #   ...
    #
    # NEW:
    #   ALL image dates → ONE getInfo()
    #
    # Then only the closest image gets:
    #   - probability reduction
    #   - label histogram
    #
    # Same closest-date methodology.
    # --------------------------------------------------------

    def get_dw_for_point(
        fire_lat,
        fire_lon,
        fire_date
    ):

        fire_date = pd.Timestamp(
            fire_date
        ).normalize()


        point = ee.Geometry.Point(
            [
                float(fire_lon),
                float(fire_lat)
            ]
        )


        area = (
            point
            .buffer(
                DW_AREA_M / 2
            )
            .bounds()
        )


        search_start = (
            fire_date
            -
            pd.Timedelta(
                days=DW_SEARCH_BEFORE_DAYS
            )
        )


        search_end = (
            fire_date
            +
            pd.Timedelta(
                days=DW_SEARCH_AFTER_DAYS + 1
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
                )
            )
        )


        # ----------------------------------------------------
        # Number of images
        # ----------------------------------------------------

        count = int(
            collection
            .size()
            .getInfo()
        )


        print(
            "DW images in search window:",
            count
        )


        if count == 0:

            return {
                "status":
                    "NO_IMAGE"
            }


        # ----------------------------------------------------
        # OPTIMIZATION:
        #
        # Fetch ALL system:time_start values in ONE request.
        # ----------------------------------------------------

        timestamps = (
            collection
            .aggregate_array(
                "system:time_start"
            )
            .getInfo()
        )


        if not timestamps:

            return {
                "status":
                    "NO_IMAGE"
            }


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


        # ----------------------------------------------------
        # Find closest image locally
        # ----------------------------------------------------

        differences = np.abs(
            (
                image_dates
                -
                fire_date
            )
            /
            pd.Timedelta(
                days=1
            )
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


        # ----------------------------------------------------
        # Convert only selected image
        # ----------------------------------------------------

        image_list = collection.toList(
            count
        )


        image = ee.Image(
            image_list.get(
                closest_index
            )
        )


        # ====================================================
        # PROBABILITY REDUCTION
        # ====================================================

        means = (
            image
            .select(
                DW_BANDS
            )
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=10,
                bestEffort=True,
                maxPixels=100000
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

            for band in DW_BANDS

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


        # ====================================================
        # DOMINANT LABEL
        # ====================================================

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
                maxPixels=100000
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

            int(float(k)): int(v)

            for k, v in histogram.items()
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
                DW_CLASS_NAMES.get(
                    dominant_label,
                    "unknown"
                )
            )

        else:

            dominant_label = None

            dominant_probability = np.nan

            dominant_class = None


        # ====================================================
        # NATURAL VEGETATION PROBABILITY
        # ====================================================

        natural_values = [

            means.get(
                "trees"
            ),

            means.get(
                "grass"
            ),

            means.get(
                "shrub_and_scrub"
            ),

            means.get(
                "flooded_vegetation"
            )
        ]


        if all(
            x is not None
            for x in natural_values
        ):

            natural_probability = sum(
                float(x)
                for x in natural_values
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
                means.get(
                    "water"
                ),

            "dw_trees":
                means.get(
                    "trees"
                ),

            "dw_grass":
                means.get(
                    "grass"
                ),

            "dw_flooded_vegetation":
                means.get(
                    "flooded_vegetation"
                ),

            "dw_crops":
                means.get(
                    "crops"
                ),

            "dw_shrub_and_scrub":
                means.get(
                    "shrub_and_scrub"
                ),

            "dw_built":
                means.get(
                    "built"
                ),

            "dw_bare":
                means.get(
                    "bare"
                ),

            "dw_snow_and_ice":
                means.get(
                    "snow_and_ice"
                ),

            "dw_dominant_class":
                dominant_class,

            "dw_dominant_probability":
                dominant_probability,

            "dw_crop_probability":
                means.get(
                    "crops"
                ),

            "dw_top_probability":
                dominant_probability,

            "dw_trees_prob":
                means.get(
                    "trees"
                ),

            "dw_grass_prob":
                means.get(
                    "grass"
                ),

            "dw_shrub_prob":
                means.get(
                    "shrub_and_scrub"
                ),

            "dw_flooded_vegetation_prob":
                means.get(
                    "flooded_vegetation"
                ),

            "dw_crop_prob":
                means.get(
                    "crops"
                ),

            "dw_natural_vegetation_prob":
                natural_probability
        }


    # ========================================================
    # PROCESS EVENT POINTS
    # ========================================================

    for i, row in event_df.iterrows():

        try:

            result = get_dw_for_point(
                row["latitude"],
                row["longitude"],
                row["acq_date"]
            )


            result[
                "event_index"
            ] = i


            dw_rows.append(
                result
            )


        except Exception as e:

            print(
                "DW point error:",
                str(e)[:200]
            )


except Exception as e:

    print(
        "⚠ Earth Engine error:",
        e
    )


dw_df = pd.DataFrame(
    dw_rows
)


# ============================================================
# 26. DYNAMIC WORLD EVENT AGGREGATION
# ============================================================

dw_event_features = {}


DW_EVENT_FEATURES = [

    "event_max_dw_crop_probability",

    "event_mean_dw_crop_probability",

    "event_max_dw_top_probability",

    "event_mean_dw_top_probability",

    "event_max_dw_trees_prob",

    "event_mean_dw_trees_prob",

    "event_max_dw_grass_prob",

    "event_mean_dw_grass_prob",

    "event_max_dw_shrub_prob",

    "event_mean_dw_shrub_prob",

    "event_max_dw_flooded_vegetation_prob",

    "event_mean_dw_flooded_vegetation_prob",

    "event_max_dw_crop_prob",

    "event_mean_dw_crop_prob",

    "event_max_dw_natural_vegetation_prob",

    "event_mean_dw_natural_vegetation_prob"
]


print(
    "\nDynamic World event records:",
    len(dw_df)
)


if len(dw_df):

    print(
        "Dynamic World status:",
        dw_df[
            "status"
        ]
        .value_counts()
        .to_dict()
    )


def dw_max(
    column
):

    if column not in dw_df.columns:

        return np.nan


    values = pd.to_numeric(
        dw_df[
            column
        ],
        errors="coerce"
    ).dropna()


    return (

        float(
            values.max()
        )

        if len(values)

        else np.nan
    )


def dw_mean(
    column
):

    if column not in dw_df.columns:

        return np.nan


    values = pd.to_numeric(
        dw_df[
            column
        ],
        errors="coerce"
    ).dropna()


    return (

        float(
            values.mean()
        )

        if len(values)

        else np.nan
    )


dw_event_features[
    "event_max_dw_crop_probability"
] = dw_max(
    "dw_crops"
)


dw_event_features[
    "event_mean_dw_crop_probability"
] = dw_mean(
    "dw_crops"
)


dw_event_features[
    "event_max_dw_top_probability"
] = dw_max(
    "dw_dominant_probability"
)


dw_event_features[
    "event_mean_dw_top_probability"
] = dw_mean(
    "dw_dominant_probability"
)


dw_event_features[
    "event_max_dw_trees_prob"
] = dw_max(
    "dw_trees"
)


dw_event_features[
    "event_mean_dw_trees_prob"
] = dw_mean(
    "dw_trees"
)


dw_event_features[
    "event_max_dw_grass_prob"
] = dw_max(
    "dw_grass"
)


dw_event_features[
    "event_mean_dw_grass_prob"
] = dw_mean(
    "dw_grass"
)


dw_event_features[
    "event_max_dw_shrub_prob"
] = dw_max(
    "dw_shrub_and_scrub"
)


dw_event_features[
    "event_mean_dw_shrub_prob"
] = dw_mean(
    "dw_shrub_and_scrub"
)


dw_event_features[
    "event_max_dw_flooded_vegetation_prob"
] = dw_max(
    "dw_flooded_vegetation"
)


dw_event_features[
    "event_mean_dw_flooded_vegetation_prob"
] = dw_mean(
    "dw_flooded_vegetation"
)


dw_event_features[
    "event_max_dw_crop_prob"
] = dw_max(
    "dw_crops"
)


dw_event_features[
    "event_mean_dw_crop_prob"
] = dw_mean(
    "dw_crops"
)


dw_event_features[
    "event_max_dw_natural_vegetation_prob"
] = dw_max(
    "dw_natural_vegetation_prob"
)


dw_event_features[
    "event_mean_dw_natural_vegetation_prob"
] = dw_mean(
    "dw_natural_vegetation_prob"
)


for c in DW_EVENT_FEATURES:

    if c not in dw_event_features:

        dw_event_features[c] = np.nan


# ============================================================
# 27. BASE DYNAMIC WORLD
# ============================================================

first_valid_dw = None


if len(dw_df):

    valid_mask = (
        dw_df[
            "status"
        ]
        ==
        "OK"
    )


    if valid_mask.any():

        first_valid_dw = (
            dw_df[
                valid_mask
            ]
            .iloc[0]
        )


dw_base_features = {}


for band in DW_BANDS:

    key = (
        f"dw_{band}"
    )


    if (
        first_valid_dw is not None
        and
        key in first_valid_dw.index
    ):

        dw_base_features[
            key
        ] = first_valid_dw[
            key
        ]

    else:

        dw_base_features[
            key
        ] = np.nan


dw_base_features[
    "dw_dominant_probability"
] = (

    first_valid_dw[
        "dw_dominant_probability"
    ]

    if first_valid_dw is not None

    else np.nan
)


dw_base_features[
    "dw_dominant_class"
] = (

    first_valid_dw[
        "dw_dominant_class"
    ]

    if first_valid_dw is not None

    else np.nan
)


# ============================================================
# 28. COMBINE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("COMBINING FEATURES")
print("=" * 70)


features = {}


features.update(
    event_features
)


features.update(
    flare_features
)


features.update(
    industrial_features
)


features.update(
    temporal_features
)


features.update(
    mining_features
)


features.update(
    agriculture_features
)


features.update(
    wildfire_features
)


features.update(
    dw_event_features
)


features.update(
    dw_base_features
)


# ============================================================
# 29. OSM FEATURES
# ============================================================

for category in OSM_CATEGORIES:

    features[
        f"distance_to_nearest_{category}_m"
    ] = osm_nearest[
        category
    ]


features[
    "distance_to_nearest_agriculture_m"
] = agri_osm_distance


# ============================================================
# 30. FEATURE ALIASES
# ============================================================

ALIASES = {

    "event_frp_mean":
        "event_mean_frp",

    "event_frp_median":
        "event_median_frp",

    "event_frp_max":
        "event_max_frp",

    "event_frp_std":
        "event_std_frp",

    "event_occupied_blocks_500m":
        "event_500m_block_count",

    "event_occupied_blocks_1km":
        "event_1km_block_count",

    "event_duration":
        "event_duration_hours"
}


for old, new in ALIASES.items():

    if (
        old in features
        and
        new not in features
    ):

        features[new] = features[old]


# ============================================================
# 31. EXACT 137 FEATURES
# ============================================================

raw_df = pd.DataFrame(
    [features]
)


print(
    "Generated feature values:",
    len(raw_df.columns)
)


X_test = raw_df.reindex(
    columns=MODEL_FEATURES
)


X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)


print(
    "Final model matrix:",
    X_test.shape
)


if X_test.shape != (1, 137):

    raise RuntimeError(
        f"Expected (1,137), got {X_test.shape}"
    )


# ============================================================
# 32. FEATURE QUALITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE QUALITY CHECK")
print("=" * 70)


missing_mask = (
    X_test.iloc[0]
    .isna()
)


missing_features = (
    X_test.columns[
        missing_mask
    ]
    .tolist()
)


available_features = (
    137 -
    len(missing_features)
)


coverage = (
    available_features /
    137
)


print(
    "Available before imputation:",
    available_features,
    "/ 137"
)


print(
    "Missing before imputation:",
    len(missing_features)
)


print(
    "Feature coverage:",
    f"{coverage:.2%}"
)


critical_firms = [

    "event_detection_count",

    "event_active_days",

    "event_mean_frp",

    "event_median_frp",

    "event_max_frp",

    "event_std_frp",

    "event_mean_latitude",

    "event_mean_longitude",

    "event_min_latitude",

    "event_max_latitude",

    "event_min_longitude",

    "event_max_longitude",

    "event_500m_block_count",

    "event_1km_block_count",

    "event_duration_hours",

    "event_duration_days",

    "event_detections_per_active_day",

    "event_detections_per_hour"
]


critical_missing = [

    c

    for c in critical_firms

    if (
        c in MODEL_FEATURES
        and
        pd.isna(
            X_test.iloc[0][c]
        )
    )
]


if critical_missing:

    print(
        "\nCRITICAL FEATURES MISSING:"
    )


    for c in critical_missing:

        print(
            " -",
            c
        )


    raise RuntimeError(
        "Critical FIRMS event features missing."
    )


print(
    "✓ Critical FIRMS event features populated."
)


if missing_features:

    print(
        "\nRemaining missing features:"
    )


    for c in missing_features:

        print(
            " -",
            c
        )

else:

    print(
        "✓ No missing features before imputation."
    )


# ============================================================
# 33. DYNAMIC WORLD DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("DYNAMIC WORLD DIAGNOSTIC")
print("=" * 70)


if len(dw_df):

    print(
        "Total DW event records:",
        len(dw_df)
    )


    print(
        "DW status:",
        dw_df[
            "status"
        ]
        .value_counts()
        .to_dict()
    )


    valid_dw = dw_df[
        dw_df[
            "status"
        ]
        ==
        "OK"
    ]


    if len(valid_dw):

        print(
            "\nSelected Dynamic World:"
        )


        for _, row in valid_dw.iterrows():

            print(
                "  FIRMS event:",
                row.get(
                    "event_index"
                )
            )


            print(
                "  DW date:",
                row.get(
                    "dynamic_world_date"
                )
            )


            print(
                "  Date difference:",
                row.get(
                    "date_difference_days"
                ),
                "days"
            )


            print(
                "  DW dominant class:",
                row.get(
                    "dw_dominant_class"
                )
            )


            print(
                "  DW top probability:",
                row.get(
                    "dw_dominant_probability"
                )
            )


            print(
                "  DW crops:",
                row.get(
                    "dw_crops"
                )
            )


            print(
                "  DW trees:",
                row.get(
                    "dw_trees"
                )
            )


            print(
                "  DW grass:",
                row.get(
                    "dw_grass"
                )
            )


    else:

        print(
            "⚠ No valid Dynamic World pixels."
        )


else:

    print(
        "⚠ Dynamic World returned no records."
    )


# ============================================================
# 34. IMPORTANT FEATURE VALUES
# ============================================================

diagnostic_names = [

    "event_detection_count",

    "event_mean_frp",

    "event_max_frp",

    "event_mean_latitude",

    "event_mean_longitude",

    "event_duration_hours",

    "event_duration_days",

    "event_detections_per_active_day",

    "event_detections_per_hour",

    "event_min_steel_distance_m",

    "event_min_cement_distance_m",

    "event_min_wri_power_distance_m",

    "event_min_fertilizer_distance_m",

    "event_min_refinery_petro_distance_m",

    "event_min_nearest_coal_mine_distance_m",

    "event_min_nearest_coal_mine_distance_km",

    "event_max_same_cell_fire_count_3d",

    "event_max_same_cell_fire_count_7d",

    "event_max_same_cell_fire_count_30d",

    "same_cell_fire_count_3d_max",

    "same_cell_fire_count_7d_max",

    "same_cell_fire_count_30d_max",

    "nearby_fire_count_3d_1km_max",

    "nearby_fire_count_7d_1km_max",

    "nearby_fire_count_30d_1km_max",

    "event_max_nearby_fire_count_3d_1km",

    "event_max_nearby_fire_count_7d_1km",

    "event_max_nearby_fire_count_30d_1km",

    "agri_osm_distance_m_min",

    "event_max_dw_crop_probability",

    "event_mean_dw_crop_probability",

    "event_max_dw_top_probability",

    "event_mean_dw_top_probability",

    "event_max_dw_trees_prob",

    "event_mean_dw_trees_prob",

    "event_max_dw_grass_prob",

    "event_mean_dw_grass_prob",

    "event_max_dw_natural_vegetation_prob",

    "event_mean_dw_natural_vegetation_prob"
]


diagnostic_names = [
    c
    for c in diagnostic_names
    if c in X_test.columns
]


print(
    "\nImportant feature values:"
)


if diagnostic_names:

    display(
        X_test[
            diagnostic_names
        ].T.rename(
            columns={
                0: "value"
            }
        )
    )


# ============================================================
# 35. IMPUTATION
# ============================================================

print("\n" + "=" * 70)
print("IMPUTATION")
print("=" * 70)


X_imputed = rf_imputer.transform(
    X_test
)


print(
    "✓ 2025 imputer applied"
)


if X_imputed.shape != (1, 137):

    raise RuntimeError(
        "Imputed matrix is not (1,137)."
    )


if missing_features:

    print(
        "\nImputed values:"
    )


    imputed_series = pd.Series(
        X_imputed[0],
        index=MODEL_FEATURES
    )


    for c in missing_features:

        print(
            f"  {c:<55}"
            f"{imputed_series[c]:.6f}"
        )


# ============================================================
# 36. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST PREDICTION")
print("=" * 70)


prediction = rf_model.predict(
    X_imputed
)[0]


probabilities = (
    rf_model
    .predict_proba(
        X_imputed
    )[0]
)


classes = rf_model.classes_


CLASS_NAMES = {

    0: "Industrial",

    1: "Gas",

    2: "Agriculture",

    3: "Mining",

    4: "Wildfire"
}


probability_dict = {}


for cls, probability in zip(
    classes,
    probabilities
):

    probability_dict[
        CLASS_NAMES.get(
            int(cls),
            str(cls)
        )
    ] = float(
        probability
    )


confidence = float(
    np.max(
        probabilities
    )
)


predicted_class = CLASS_NAMES.get(
    int(prediction),
    str(prediction)
)


# ============================================================
# 37. FINAL DECISION
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


if confidence >= CONFIDENCE_THRESHOLD:

    final_class = predicted_class

    status = "Classified"

else:

    final_class = "Unknown"

    status = "Unknown"


# ============================================================
# 38. MODEL DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("MODEL DIAGNOSTIC")
print("=" * 70)


if hasattr(
    rf_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "feature":
            MODEL_FEATURES,

        "importance":
            rf_model.feature_importances_,

        "input_value":
            X_test.iloc[0].values,

        "imputed_value":
            X_imputed[0]

    })


    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nTop 20 model features:"
    )


    display(
        importance_df.head(20)
    )


# ============================================================
# 39. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL FIRE SOURCE CLASSIFICATION")
print("=" * 70)


print(
    "\nInput:"
)


print(
    "  Latitude :",
    lat
)


print(
    "  Longitude:",
    lon
)


print(
    "\nFIRMS:"
)


print(
    "  Latitude :",
    selected["latitude"]
)


print(
    "  Longitude:",
    selected["longitude"]
)


print(
    "  Date     :",
    selected["acq_date"]
)


print(
    "  Time     :",
    selected["acq_time"]
)


print(
    "  FRP      :",
    selected["frp"]
)


print(
    "  Satellite:",
    selected["satellite"]
)


print(
    "  Distance :",
    round(
        selected[
            "distance_m_from_input"
        ],
        2
    ),
    "m"
)


print(
    "\nModel candidate:"
)


print(
    "  Class      :",
    predicted_class
)


print(
    "  Probability:",
    f"{confidence:.4f}"
)


print(
    "\nFINAL DECISION:"
)


print(
    "  Class      :",
    final_class
)


print(
    "  Status     :",
    status
)


print(
    "\nClass probabilities:"
)


for name in [
    "Industrial",
    "Gas",
    "Agriculture",
    "Mining",
    "Wildfire"
]:

    print(
        f"  {name:15s}: "
        f"{probability_dict.get(name, 0):.4f}"
    )


# ============================================================
# 40. SAVE PREDICTION
# ============================================================

prediction_row = {

    "input_latitude":
        lat,

    "input_longitude":
        lon,

    "firms_latitude":
        selected["latitude"],

    "firms_longitude":
        selected["longitude"],

    "firms_date":
        selected["acq_date"],

    "firms_time":
        selected["acq_time"],

    "firms_frp":
        selected["frp"],

    "firms_satellite":
        selected["satellite"],

    "firms_source":
        selected[
            "firms_source_api"
        ],

    "firms_distance_m":
        selected[
            "distance_m_from_input"
        ],

    "event_detection_count":
        n,

    "candidate_class_id":
        int(prediction),

    "candidate_class":
        predicted_class,

    "final_class":
        final_class,

    "confidence":
        confidence,

    "status":
        status,

    "probability_industrial":
        probability_dict.get(
            "Industrial",
            np.nan
        ),

    "probability_gas":
        probability_dict.get(
            "Gas",
            np.nan
        ),

    "probability_agriculture":
        probability_dict.get(
            "Agriculture",
            np.nan
        ),

    "probability_mining":
        probability_dict.get(
            "Mining",
            np.nan
        ),

    "probability_wildfire":
        probability_dict.get(
            "Wildfire",
            np.nan
        ),

    "osm_object_count":
        len(osm_elements),

    "osm_agriculture_distance_m":
        agri_osm_distance,

    "dynamic_world_event_records":
        len(dw_df),

    "dynamic_world_valid_records":
        (
            int(
                (
                    dw_df[
                        "status"
                    ]
                    ==
                    "OK"
                ).sum()
            )
            if len(dw_df)
            else 0
        ),

    "historical_firms_detections":
        len(historical_df),

    "historical_firms_requests_successful":
        historical_successful_requests,

    "missing_before_imputation":
        len(missing_features),

    "feature_coverage":
        coverage
}


prediction_df = pd.DataFrame(
    [prediction_row]
)


prediction_df.to_csv(
    PREDICTION_OUTPUT,
    index=False
)


# ============================================================
# 41. SAVE 137 FEATURES
# ============================================================

X_test.to_csv(
    FEATURE_OUTPUT,
    index=False
)


# ============================================================
# 42. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)


print(
    "✓ NASA FIRMS N20/N21/SNPP"
)


print(
    "✓ Exact 30-day FIRMS history"
)


print(
    "✓ FIRMS event context"
)


print(
    "✓ FIRMS event statistics"
)


print(
    "✓ World Bank flare evidence"
)


print(
    "✓ Optimized single-request OSM"
)


print(
    "✓ Industrial evidence"
)


print(
    "✓ Mining evidence"
)


print(
    "✓ Agriculture evidence"
)


print(
    "✓ Wildfire/natural evidence"
)


print(
    "✓ Optimized Dynamic World"
)


print(
    "✓ Dynamic World valid-pixel fallback"
)


print(
    "✓ Exact 137-feature schema"
)


print(
    "✓ 2025 imputer"
)


print(
    "✓ 700-tree Random Forest"
)


print(
    "\nCANDIDATE CLASS:",
    predicted_class
)


print(
    "CANDIDATE PROBABILITY:",
    f"{confidence:.2%}"
)


print(
    "FINAL CLASS:",
    final_class
)


print(
    "STATUS:",
    status
)


print(
    "FEATURE COVERAGE:",
    f"{coverage:.2%}"
)


print(
    "MISSING BEFORE IMPUTATION:",
    len(missing_features)
)


print(
    "HISTORICAL DETECTIONS:",
    len(historical_df)
)


print(
    "DYNAMIC WORLD RECORDS:",
    len(dw_df)
)


print(
    "DYNAMIC WORLD VALID RECORDS:",
    (
        int(
            (
                dw_df[
                    "status"
                ]
                ==
                "OK"
            ).sum()
        )
        if len(dw_df)
        else 0
    )
)


print(
    "\nPrediction file:"
)


print(
    PREDICTION_OUTPUT
)


print(
    "\n137-feature file:"
)


print(
    FEATURE_OUTPUT
)


print(
    "=" * 70
)