"""
Module: return_classifier.py
Stage:  Return-Probability Classifier (XGBoost)

DEPENDS ON:
    data/master_df_v2.parquet  — produced by clustering.py
        Required columns:
            review_score, days_late, freight_value, order_value,
            product_weight_g, product_category_name_english,
            payment_type, customer_state, seller_state, is_return

OUTPUT:
    data/master_df_v3.parquet            — master_df_v2 + return_prob + return_flag
    outputs/return_clf_v1.pkl            — fitted XGBClassifier + calibrator
    outputs/return_classifier_metrics.json
        {
            "roc_auc": float,
            "pr_auc": float,
            "brier_score": float,
            "threshold_0.3": {"precision": float, "recall": float, "f1": float, "flagged_pct": float}
        }

INTERFACE:
    build_features(df)                         -> pd.DataFrame
    train(df, target_col, test_size)           -> (model, X_test, y_test)
    evaluate(model, X_test, y_test, threshold) -> dict
    predict_proba(model, df)                   -> np.ndarray
    add_return_prob(model, df)                 -> pd.DataFrame
    save_model(model, path)                    -> None
    load_model(path)                           -> model
    save_metrics(metrics, path)                -> None
    run_full_pipeline(parquet_path, out_dir,
                      data_dir)                -> dict
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAT_COLS = [
    "product_category_name_english",
    "payment_type",
    "seller_state",
]
NUM_COLS = [
    "review_score",
    "freight_value",
    "order_value",
    "product_weight_g",
]
TARGET = "is_return"
RETURN_PROB_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals and select modelling columns.

    - days_late NaN → 0  (on-time delivery, not late)
    - Categoricals   → LabelEncoder (integer codes)
    - Numerics        → median imputation

    Returns a copy with all features as numeric dtype.
    """
    out = df[CAT_COLS + NUM_COLS].copy()

    for col in CAT_COLS:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].fillna("unknown").astype(str))

    out[NUM_COLS] = out[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    out[NUM_COLS] = out[NUM_COLS].fillna(out[NUM_COLS].median())

    return out


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    target_col: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train an XGBClassifier with Platt calibration (sigmoid, cv=3).

    scale_pos_weight is computed from the training split so it reflects
    the actual imbalance in training data, not the full dataset.

    Returns
    -------
    (calibrated_model, X_test, y_test)
    """
    X = build_features(df)
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / max(pos, 1)

    print(f"[train] Train: {len(X_train):,} rows | "
          f"positives: {pos:,} ({pos/len(y_train)*100:.1f}%) | "
          f"scale_pos_weight: {scale:.1f}")

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    print(f"[train] Model fitted.")
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = RETURN_PROB_THRESHOLD,
) -> dict:
    """
    Compute ROC-AUC, PR-AUC, Brier score, and threshold-based metrics.

    Target: ROC-AUC > 0.70
    PR-AUC is the primary metric for imbalanced data — reported alongside.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc":           round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc":            round(float(average_precision_score(y_test, proba)), 4),
        "brier_score":       round(float(brier_score_loss(y_test, proba)), 4),
        "n_test":            int(len(y_test)),
        "n_positives_test":  int(y_test.sum()),
        f"threshold_{threshold}": {
            "precision":   round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "flagged_pct": round(float(preds.mean() * 100), 2),
        },
    }

    print(f"[evaluate] ROC-AUC : {metrics['roc_auc']}  "
          f"({'✓ target met' if metrics['roc_auc'] >= 0.70 else '✗ below 0.70 target'})")
    print(f"[evaluate] PR-AUC  : {metrics['pr_auc']}")
    print(f"[evaluate] Brier   : {metrics['brier_score']}")
    thr = metrics[f"threshold_{threshold}"]
    print(f"[evaluate] @ threshold {threshold} — "
          f"precision={thr['precision']}  recall={thr['recall']}  "
          f"f1={thr['f1']}  flagged={thr['flagged_pct']}% of orders")

    return metrics


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Return return-probability scores for all rows in df."""
    X = build_features(df)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def add_return_prob(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return_prob and return_flag columns to df.

    return_flag = 1 if return_prob >= RETURN_PROB_THRESHOLD.
    These are the orders pre-assigned pickup slots in the SDVRP.
    """
    df = df.copy()
    df["return_prob"] = predict_proba(model, df)
    df["return_flag"] = (df["return_prob"] >= RETURN_PROB_THRESHOLD).astype(int)

    flagged = df["return_flag"].sum()
    print(f"[add_return_prob] return_prob added | "
          f"{flagged:,} orders flagged for pickup "
          f"({flagged/len(df)*100:.1f}% at threshold {RETURN_PROB_THRESHOLD})")
    return df


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

def save_model(
    model,
    path: str | Path = "outputs/return_clf_v1.pkl",
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[save_model] Model saved → {path}")


def load_model(path: str | Path = "outputs/return_clf_v1.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(
    metrics: dict,
    path: str | Path = "outputs/return_classifier_metrics.json",
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save_metrics] Metrics saved → {path}")


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    parquet_path: str | Path = "data/master_df_v2.parquet",
    out_dir: str | Path = "outputs",
    data_dir: str | Path = "data",
) -> dict:
    """
    End-to-end pipeline: load → train → evaluate → add return_prob → save.

    Writes:
        {data_dir}/master_df_v3.parquet
        {out_dir}/return_clf_v1.pkl
        {out_dir}/return_classifier_metrics.json

    Returns
    -------
    dict with keys: model, metrics, master_df_v3
    """
    print("=" * 60)
    print("  RETURN CLASSIFIER PIPELINE")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = pd.read_parquet(parquet_path)
    pos_rate = df[TARGET].mean()
    print(f"       {len(df):,} rows | positive rate: {pos_rate*100:.2f}%")

    print("\n[2/4] Training XGBoost + Platt calibration...")
    model, X_test, y_test = train(df)

    print("\n[3/4] Evaluating...")
    metrics = evaluate(model, X_test, y_test)

    print("\n[4/4] Adding return_prob and saving outputs...")
    df_v3 = add_return_prob(model, df)

    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df_v3.to_parquet(data_dir / "master_df_v3.parquet", index=False)
    print(f"[run_full_pipeline] master_df_v3.parquet saved "
          f"({len(df_v3):,} rows, {df_v3.shape[1]} cols)")

    save_model(model, path=out_dir / "return_clf_v1.pkl")
    save_metrics(metrics, path=out_dir / "return_classifier_metrics.json")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print(f"  PR-AUC   : {metrics['pr_auc']}")
    print(f"  Outputs  : {out_dir}/")
    print("=" * 60)

    return {"model": model, "metrics": metrics, "master_df_v3": df_v3}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_full_pipeline(
        parquet_path="data/master_df_v2.parquet",
        out_dir="outputs",
        data_dir="data",
    )
    print(json.dumps(results["metrics"], indent=2))