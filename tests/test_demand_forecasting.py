"""
Tests for src/demand_forecasting.py — Prophet-based demand forecasting.

Covers:
    - preprocess (column creation, type cast)
    - create_weekly_demand (aggregation)
    - forecast_demand (Prophet fitting + prediction)
    - save_outputs (file creation)
    - Edge cases (empty zones, insufficient data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.demand_forecasting import (
    preprocess,
    create_weekly_demand,
    forecast_demand,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def forecasting_df():
    """Synthetic DataFrame for forecasting — 3 zones, 40+ weeks each."""
    rng = np.random.default_rng(42)
    rows = []
    for zone in range(3):
        base = pd.Timestamp("2017-06-01")
        for w in range(50):
            n_orders = rng.integers(5, 20)
            for _ in range(n_orders):
                rows.append(
                    {
                        "order_purchase_timestamp": base
                        + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7))),
                        "dark_store_id": zone,
                        "order_id": f"ord_{len(rows)}",
                    }
                )
    df = pd.DataFrame(rows)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_adds_order_date(self, forecasting_df):
        result = preprocess(forecasting_df)
        assert "order_date" in result.columns

    def test_dark_store_id_is_int(self, forecasting_df):
        result = preprocess(forecasting_df)
        assert result["dark_store_id"].dtype in [np.int32, np.int64, int]

    def test_drops_null_dark_store(self, forecasting_df):
        df = forecasting_df.copy()
        df.loc[0, "dark_store_id"] = np.nan
        result = preprocess(df)
        assert not result["dark_store_id"].isna().any()


# ---------------------------------------------------------------------------
# create_weekly_demand
# ---------------------------------------------------------------------------


class TestCreateWeeklyDemand:
    def test_output_columns(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        assert "dark_store_id" in weekly.columns
        assert "order_date" in weekly.columns
        assert "orders" in weekly.columns

    def test_orders_positive(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        assert (weekly["orders"] > 0).all()

    def test_all_zones_present(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        assert set(weekly["dark_store_id"].unique()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# forecast_demand
# ---------------------------------------------------------------------------


class TestForecastDemand:
    def test_output_columns(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        result = forecast_demand(weekly)
        assert "ds" in result.columns
        assert "yhat" in result.columns
        assert "dark_store_id" in result.columns

    def test_yhat_reasonable(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        result = forecast_demand(weekly)
        # yhat should not be extremely negative
        assert result["yhat"].min() > -1000

    def test_all_zones_forecasted(self, forecasting_df):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        result = forecast_demand(weekly)
        assert set(result["dark_store_id"].unique()) == {0, 1, 2}

    def test_insufficient_data_skipped(self):
        """Zone with <10 weeks should be skipped."""
        df = pd.DataFrame(
            {
                "order_purchase_timestamp": pd.date_range(
                    "2018-01-01", periods=5, freq="W"
                ),
                "dark_store_id": [0] * 5,
                "order_id": range(5),
            }
        )
        df["order_date"] = df["order_purchase_timestamp"]
        weekly = create_weekly_demand(df)
        with pytest.raises(ValueError, match="No forecasts"):
            forecast_demand(weekly)


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def test_writes_files(self, forecasting_df, tmp_path):
        df = preprocess(forecasting_df)
        weekly = create_weekly_demand(df)
        result = forecast_demand(weekly)
        save_outputs(result, out_dir=tmp_path)
        assert (tmp_path / "forecasted_demand_by_zone.csv").exists()
        assert (tmp_path / "zone_total_demand.csv").exists()
