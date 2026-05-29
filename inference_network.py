"""Inference helpers for lead probability,
opportunity scoring,
engagement score,
priority ranking.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils import BASE_DIR, MODEL_FEATURES, add_model_features

DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:

    model_path = Path(model_path)

    if not model_path.exists():

        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def infer_lead_signals(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:

    result = frame.copy()

    if result.empty:

        return result

    if not set(MODEL_FEATURES).issubset(result.columns):

        result = add_model_features(result)

    result[MODEL_FEATURES] = result[MODEL_FEATURES].fillna(0)

    probability = artifact["pipeline"].predict_proba(result[MODEL_FEATURES])[:, 1]

    result["lead_probability"] = (probability * 100).round(1)

    baseline = (
        result["lead_score"]
        if "lead_score" in result.columns
        else result["lead_probability"]
    )

    result["business_opportunity_score"] = (
        result["lead_probability"] * 0.68 + baseline * 0.32
    ).round(1)

    result["engagement_index"] = (
        result["lead_probability"] * 0.55 + result["business_opportunity_score"] * 0.45
    ).round(1)

    result["lead_strength"] = pd.cut(
        result["lead_probability"],
        bins=[-1, 39, 69, 100],
        labels=["Emerging", "Promising", "Priority"],
    ).astype(str)

    result["growth_potential"] = np.select(
        [
            result["business_opportunity_score"] >= 75,
            result["business_opportunity_score"] >= 50,
        ],
        ["High", "Moderate"],
        default="Developing",
    )

    result["confidence_score"] = (abs(result["lead_probability"] - 50) * 2).round(1)

    result["priority_rank"] = (
        result["business_opportunity_score"]
        .rank(method="dense", ascending=False)
        .fillna(0)
        .astype(int)
    )

    return result


def predict_single(record: pd.DataFrame, artifact: dict) -> dict:

    result = infer_lead_signals(record, artifact)

    if result.empty:

        return {
            "lead_probability": 0,
            "lead_strength": "Unknown",
            "business_opportunity_score": 0,
            "growth_potential": "Unknown",
            "engagement_index": 0,
            "confidence_score": 0,
        }

    row = result.iloc[0]

    return {
        "lead_probability": float(row["lead_probability"]),
        "lead_strength": row["lead_strength"],
        "business_opportunity_score": float(row["business_opportunity_score"]),
        "growth_potential": row["growth_potential"],
        "engagement_index": float(row["engagement_index"]),
        "confidence_score": float(row["confidence_score"]),
    }
