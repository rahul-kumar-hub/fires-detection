from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL2_SCRIPT = PROJECT_ROOT / "scripts" / "model2" / "step_79.py"
PREDICTION_OUTPUT = PROJECT_ROOT / "data" / "firms_2026_test_point_prediction.csv"

PROGRESS_MARKERS = (
    "NASA FIRMS SEARCH",
    "SELECTED FIRMS DETECTION",
    "FIRMS TEMPORAL HISTORY",
    "WORLD BANK FLARE EVIDENCE",
    "LIVE OPENSTREETMAP",
    "DYNAMIC WORLD",
    "COMBINING FEATURES",
    "FINAL FIRE SOURCE CLASSIFICATION",
    "PIPELINE COMPLETE",
)


def run_model2(
    latitude: float,
    longitude: float,
    timeout_seconds: int = 900,
) -> dict:
    """Run the complete Model 2 evidence pipeline for one location."""

    if not MODEL2_SCRIPT.is_file():
        raise FileNotFoundError(f"Model 2 script not found: {MODEL2_SCRIPT}")

    try:
        process = subprocess.Popen(
            [sys.executable, str(MODEL2_SCRIPT)],
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(f"{latitude}\n{longitude}\n")
        process.stdin.close()

        output = []
        for line in process.stdout:
            output.append(line)
            if any(marker in line for marker in PROGRESS_MARKERS):
                print(line.rstrip(), flush=True)

        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise TimeoutError("Model 2 pipeline timed out.") from exc

    if return_code != 0:
        raise RuntimeError(
            "Model 2 pipeline failed:\n" + "".join(output[-20:]).strip()
        )

    if not PREDICTION_OUTPUT.is_file():
        raise RuntimeError(
            f"Model 2 completed without creating {PREDICTION_OUTPUT}"
        )

    prediction = pd.read_csv(PREDICTION_OUTPUT)
    if prediction.empty:
        raise RuntimeError("Model 2 produced an empty prediction file.")

    return prediction.iloc[-1].to_dict()