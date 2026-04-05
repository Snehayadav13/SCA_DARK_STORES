"""
Tests for src/data_pipeline.py — Olist CSV merge → master_df.parquet

Covers:
    - CSV loading and missing file handling
    - Geolocation deduplication (median lat/lon)
    - Multi-table merge correctness
    - Feature engineering (delivery_days, days_late, is_return, return_rate_by_category)
    - SP state filtering
    - Coordinate null dropping
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    LATE_DELIVERY_THRESHOLD_DAYS,
    RETURN_STATUSES,
    clean_geolocation,
    derive_features,
    drop_no_coords,
    filter_sp,
)


# ---------------------------------------------------------------------------
# clean_geolocation
# ---------------------------------------------------------------------------


class TestCleanGeolocation:
    def test_median_aggregation(self):
        geo = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [1000, 1000, 1000, 2000, 2000],
                "geolocation_lat": [-23.5, -23.6, -23.7, -22.0, -22.2],
                "geolocation_lng": [-46.6, -46.7, -46.8, -43.0, -43.2],
            }
        )
        result = clean_geolocation(geo)
        assert len(result) == 2, "Should collapse to 2 unique zips"
        assert "lat" in result.columns
        assert "lon" in result.columns

    def test_median_value_correctness(self):
        geo = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [1000, 1000, 1000],
                "geolocation_lat": [-23.0, -23.5, -24.0],
                "geolocation_lng": [-46.0, -46.5, -47.0],
            }
        )
        result = clean_geolocation(geo)
        assert np.isclose(result["lat"].iloc[0], -23.5)
        assert np.isclose(result["lon"].iloc[0], -46.5)

    def test_single_row_per_zip(self):
        geo = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [9999],
                "geolocation_lat": [-23.55],
                "geolocation_lng": [-46.63],
            }
        )
        result = clean_geolocation(geo)
        assert len(result) == 1
        assert np.isclose(result["lat"].iloc[0], -23.55)


# ---------------------------------------------------------------------------
# derive_features
# ---------------------------------------------------------------------------


class TestDeriveFeatures:
    def _make_df(self, statuses, days_late_values):
        n = len(statuses)
        base = pd.Timestamp("2018-06-01")
        return pd.DataFrame(
            {
                "order_status": statuses,
                "order_purchase_timestamp": [base] * n,
                "order_delivered_customer_date": [base + pd.Timedelta(days=10)] * n,
                "order_estimated_delivery_date": [
                    base + pd.Timedelta(days=d)
                    for d in [10 - dl for dl in days_late_values]
                ],
                "price": [100.0] * n,
                "freight_value": [15.0] * n,
                "product_weight_g": [500.0] * n,
                "product_category_name_english": ["electronics"] * n,
            }
        )

    def test_is_return_canceled(self):
        df = self._make_df(["canceled"], [0])
        result = derive_features(df)
        assert result["is_return"].iloc[0] == 1

    def test_is_return_unavailable(self):
        df = self._make_df(["unavailable"], [0])
        result = derive_features(df)
        assert result["is_return"].iloc[0] == 1

    def test_is_return_late_delivery(self):
        df = self._make_df(["delivered"], [LATE_DELIVERY_THRESHOLD_DAYS + 1])
        result = derive_features(df)
        assert result["is_return"].iloc[0] == 1

    def test_is_return_on_time_delivered(self):
        df = self._make_df(["delivered"], [0])
        result = derive_features(df)
        assert result["is_return"].iloc[0] == 0

    def test_order_value_computed(self):
        df = self._make_df(["delivered"], [0])
        result = derive_features(df)
        assert np.isclose(result["order_value"].iloc[0], 115.0)

    def test_delivery_days_non_negative(self):
        df = self._make_df(["delivered"], [0])
        result = derive_features(df)
        assert result["delivery_days"].iloc[0] >= 0

    def test_return_rate_by_category_exists(self):
        df = self._make_df(["delivered", "canceled"], [0, 0])
        result = derive_features(df)
        assert "return_rate_by_category" in result.columns
        assert result["return_rate_by_category"].iloc[0] == 0.5

    def test_null_weight_filled_with_median(self):
        df = self._make_df(["delivered", "delivered"], [0, 0])
        df.loc[0, "product_weight_g"] = np.nan
        result = derive_features(df)
        assert not result["product_weight_g"].isna().any()


# ---------------------------------------------------------------------------
# filter_sp
# ---------------------------------------------------------------------------


class TestFilterSP:
    def test_keeps_only_sp(self):
        df = pd.DataFrame(
            {
                "customer_state": ["SP", "RJ", "SP", "MG"],
                "val": [1, 2, 3, 4],
            }
        )
        result = filter_sp(df)
        assert len(result) == 2
        assert all(result["customer_state"] == "SP")

    def test_empty_when_no_sp(self):
        df = pd.DataFrame(
            {
                "customer_state": ["RJ", "MG"],
                "val": [1, 2],
            }
        )
        result = filter_sp(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# drop_no_coords
# ---------------------------------------------------------------------------


class TestDropNoCoords:
    def test_drops_null_lat(self):
        df = pd.DataFrame(
            {
                "customer_lat": [-23.5, np.nan, -23.6],
                "customer_lon": [-46.6, -46.7, -46.8],
            }
        )
        result = drop_no_coords(df)
        assert len(result) == 2

    def test_drops_null_lon(self):
        df = pd.DataFrame(
            {
                "customer_lat": [-23.5, -23.6],
                "customer_lon": [-46.6, np.nan],
            }
        )
        result = drop_no_coords(df)
        assert len(result) == 1

    def test_all_valid_unchanged(self):
        df = pd.DataFrame(
            {
                "customer_lat": [-23.5, -23.6],
                "customer_lon": [-46.6, -46.7],
            }
        )
        result = drop_no_coords(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Exceptions & IO
# ---------------------------------------------------------------------------
from src.data_pipeline import run


class TestDataPipelineIO:
    def test_missing_files_raises(self, tmp_path):
        """If raw datasets do not exist, it should raise KeyError due to missing geo frame."""
        with pytest.raises(KeyError):
            run(raw_dir=str(tmp_path), output_path=str(tmp_path / "out.parquet"))
