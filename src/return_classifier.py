"""
Module: return_classifier.py
Stage:  Day 5 — Return-Probability Classifier (XGBoost)

INPUT:
    master_df : pd.DataFrame  (data/master_df.parquet)
        Required columns:
            review_score, delivery_days, freight_value, price,
            product_category_name_english, payment_type, customer_state,
            seller_state, returned (bool target)

OUTPUT:
    return_prob column on master_df  — float32 ∈ [0, 1]
    models/return_classifier.pkl     — fitted XGBClassifier + calibrator
    outputs/return_classifier_metrics.json
        {
            "roc_auc": float,
            "pr_auc": float,
            "brier_score": float,
            "threshold_0.3": {"precision": float, "recall": float, "f1": float}
        }

INTERFACE:
    build_features(df)                    -> pd.DataFrame   # encode categoricals
    train(df, target_col, test_size)      -> (model, X_test, y_test)
    evaluate(model, X_test, y_test)       -> dict
    predict_proba(model, df)              -> np.ndarray
    save_model(model, path)               -> None
    load_model(path)                      -> model
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CAT_COLS = ["product_category_name_english", "payment_type", "customer_state", "seller_state"]
NUM_COLS = ["review_score", "delivery_days", "freight_value", "price"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals and select modelling columns.
    Returns a copy with CAT_COLS label-encoded in-place (integers).
    """
    out = df[CAT_COLS + NUM_COLS].copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].fillna("unknown").astype(str))
    out[NUM_COLS] = out[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    out[NUM_COLS] = out[NUM_COLS].fillna(out[NUM_COLS].median())
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    target_col: str = "returned",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train an XGBClassifier with Platt calibration.

    Returns
    -------
    (calibrated_model, X_test, y_test)
    """
    X = build_features(df)
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / max(pos, 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.3) -> dict:
    """Compute ROC-AUC, PR-AUC, Brier score, and threshold-based metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "brier_score": round(brier_score_loss(y_test, proba), 4),
        f"threshold_{threshold}": {
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        },
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Return return-probability scores for all rows in df."""
    X = build_features(df)
    return model.predict_proba(X)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str | Path = "models/return_classifier.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved → {path}")


def load_model(path: str | Path = "models/return_classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet("data/master_df.parquet")
    model, X_test, y_test = train(df)
    metrics = evaluate(model, X_test, y_test)
    print(json.dumps(metrics, indent=2))
    save_model(model)

    df["return_prob"] = predict_proba(model, df)
    print(df[["order_id", "return_prob"]].head())
