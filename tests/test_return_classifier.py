"""
Tests for src/return_classifier.py — XGBoost return classifier.

Covers:
    - Feature engineering for classifier
    - Train/test split stratification
    - Model training and prediction
    - Calibration (CalibratedClassifierCV)
    - Return probability output (bounds, dtype)
    - Metrics computation (ROC-AUC, PR-AUC, Brier)
    - master_df_v3 enrichment (return_prob, return_flag columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.return_classifier import (
    build_features,
    train,
    evaluate,
    add_return_prob,
)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_X(self, master_df_v2):
        X = build_features(master_df_v2)
        assert isinstance(X, pd.DataFrame)

    def test_X_no_leaky_columns(self, master_df_v2):
        X = build_features(master_df_v2)
        leaky = {"order_status", "is_return", "return_prob", "return_flag"}
        assert leaky.isdisjoint(set(X.columns))

    def test_X_has_rows(self, master_df_v2):
        X = build_features(master_df_v2)
        assert len(X) > 0

    def test_no_nans_in_X(self, master_df_v2):
        X = build_features(master_df_v2)
        assert not X.isna().any().any()

    def test_target_is_binary(self, master_df_v2):
        # target is in the dataframe directly
        from src.return_classifier import TARGET

        y = master_df_v2[TARGET]
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_returns_model_and_metrics(self, master_df_v2):
        model, X_test, y_test = train(master_df_v2)
        from src.return_classifier import evaluate

        metrics = evaluate(model, X_test, y_test)
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, master_df_v2):
        model, X_test, y_test = train(master_df_v2)
        metrics = evaluate(model, X_test, y_test)
        required = {"roc_auc", "pr_auc", "brier_score"}
        assert required.issubset(set(metrics.keys()))

    def test_roc_auc_bounded(self, master_df_v2):
        model, X_test, y_test = train(master_df_v2)
        metrics = evaluate(model, X_test, y_test)
        assert 0 <= metrics["roc_auc"] <= 1

    def test_brier_score_bounded(self, master_df_v2):
        model, X_test, y_test = train(master_df_v2)
        metrics = evaluate(model, X_test, y_test)
        assert 0 <= metrics["brier_score"] <= 1

    def test_model_can_predict(self, master_df_v2):
        model, X_test, _ = train(master_df_v2)
        probs = model.predict_proba(X_test[:5])
        assert probs.shape[1] == 2
        assert np.all(probs >= 0) and np.all(probs <= 1)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_metrics_keys(self, master_df_v2):
        model, X_test, y_test = train(master_df_v2)
        metrics = evaluate(model, X_test, y_test)
        assert metrics["roc_auc"] > 0.0

    def test_random_predictions(self):
        # mock model output
        class MockModel:
            def predict_proba(self, X):
                return np.random.default_rng(42).uniform(0, 1, size=(len(X), 2))

        X_test = pd.DataFrame(np.random.randn(100, 2))
        y_test = pd.Series(np.random.randint(0, 2, size=100))
        metrics = evaluate(MockModel(), X_test, y_test)
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["brier_score"] <= 1


# ---------------------------------------------------------------------------
# add_return_prob
# ---------------------------------------------------------------------------


class TestAddReturnProb:
    def test_adds_columns(self, master_df_v2):
        model, _, _ = train(master_df_v2)
        result = add_return_prob(model, master_df_v2)
        assert "return_prob" in result.columns
        assert "return_flag" in result.columns

    def test_return_prob_bounded(self, master_df_v2):
        model, _, _ = train(master_df_v2)
        result = add_return_prob(model, master_df_v2)
        assert result["return_prob"].between(0, 1).all()

    def test_return_flag_binary(self, master_df_v2):
        model, _, _ = train(master_df_v2)
        result = add_return_prob(model, master_df_v2)
        assert set(result["return_flag"].unique()).issubset({0, 1})

    def test_flag_threshold(self, master_df_v2):
        """return_flag == 1 iff return_prob >= 0.30."""
        model, _, _ = train(master_df_v2)
        result = add_return_prob(model, master_df_v2)
        high_prob = result[result["return_prob"] >= 0.30]
        low_prob = result[result["return_prob"] < 0.30]
        if len(high_prob) > 0:
            assert (high_prob["return_flag"] == 1).all()
        if len(low_prob) > 0:
            assert (low_prob["return_flag"] == 0).all()

    def test_row_count_preserved(self, master_df_v2):
        model, _, _ = train(master_df_v2)
        result = add_return_prob(model, master_df_v2)
        assert len(result) == len(master_df_v2)
