"""Train and persist the proxy-qualified lead inference model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    calculate_lead_scores,
    load_company_data,
)

DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "training_metrics.json"


def train_and_save_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict:
    frame = calculate_lead_scores(load_company_data(data_path))
    features = frame[MODEL_FEATURES]
    target = frame["qualified_signal"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=260,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    pipeline = Pipeline([("processor", processor), ("classifier", classifier)])
    pipeline.fit(x_train, y_train)

    prediction = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "classification_report": classification_report(
            y_test, prediction, output_dict=True
        ),
        "records": int(len(frame)),
        "hot_lead_rate": round(float(target.mean()), 4),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_definition": "Proxy target: top cohort-relative lead-score segment (Hot).",
        "data_period": "2019-05-01 to 2019-05-31",
    }
    artifact = {
        "pipeline": pipeline,
        "model_features": MODEL_FEATURES,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
        "target": "qualified_signal",
        "disclosure": (
            "Lead probability models a rule-derived qualification signal because the MCA "
            "registration source contains no observed conversions."
        ),
    }
    model_path = Path(model_path)
    report_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    results = train_and_save_model()
    print(
        f"Saved model.pkl | records={results['records']:,} | "
        f"ROC AUC={results['roc_auc']:.3f}"
    )
