"""
Tests for src/demand_baseline.py — demand profile, baseline KPIs, master_df_v2.

Covers:
    - Haversine distance calculation (scalars + arrays)
    - Demand profile aggregation (weekly × zip)
    - Zip demand summary
    - Baseline KPI computation
    - master_df enrichment (demand_per_zip column)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.demand_baseline import (
    haversine_km,
    build_demand_profile,
    build_zip_demand_summary,
    compute_baseline_kpis,
    enrich_master_df,
)


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------


class TestHaversineKm:
    def test_same_point_zero_distance(self):
        d = haversine_km(-23.55, -46.63, -23.55, -46.63)
        assert np.isclose(d, 0.0, atol=1e-6)

    def test_known_distance_sp_to_rj(self):
        """SP (-23.55, -46.63) → RJ (-22.91, -43.17) ≈ 358 km."""
        d = haversine_km(-23.55, -46.63, -22.91, -43.17)
        assert 340 < d < 380, f"SP→RJ should be ~358 km, got {d:.1f}"

    def test_vectorised_arrays(self):
        lat1 = np.array([-23.55, -23.50])
        lon1 = np.array([-46.63, -46.60])
        lat2 = np.array([-23.60, -23.55])
        lon2 = np.array([-46.70, -46.65])
        result = haversine_km(lat1, lon1, lat2, lon2)
        assert result.shape == (2,)
        assert all(result > 0)

    def test_symmetry(self):
        d1 = haversine_km(-23.55, -46.63, -23.60, -46.70)
        d2 = haversine_km(-23.60, -46.70, -23.55, -46.63)
        assert np.isclose(d1, d2)

    def test_always_non_negative(self):
        rng = np.random.default_rng(42)
        lats = rng.uniform(-30, 0, size=100)
        lons = rng.uniform(-50, -40, size=100)
        result = haversine_km(lats, lons, lats[::-1], lons[::-1])
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# build_demand_profile
# ---------------------------------------------------------------------------


class TestBuildDemandProfile:
    def test_output_columns(self, master_df):
        result = build_demand_profile(master_df)
        expected = {
            "customer_zip_code_prefix",
            "year",
            "week",
            "order_count",
            "total_value",
            "total_weight_g",
            "return_count",
            "return_rate",
        }
        assert expected.issubset(set(result.columns))

    def test_return_rate_bounded(self, master_df):
        result = build_demand_profile(master_df)
        assert result["return_rate"].between(0, 1).all()

    def test_order_count_positive(self, master_df):
        result = build_demand_profile(master_df)
        assert (result["order_count"] > 0).all()

    def test_total_weight_non_negative(self, master_df):
        result = build_demand_profile(master_df)
        assert (result["total_weight_g"] >= 0).all()


# ---------------------------------------------------------------------------
# build_zip_demand_summary
# ---------------------------------------------------------------------------


class TestBuildZipDemandSummary:
    def test_output_columns(self, master_df):
        result = build_zip_demand_summary(master_df)
        expected = {
            "customer_zip_code_prefix",
            "demand_order_count",
            "demand_total_value",
            "mean_lat",
            "mean_lon",
        }
        assert expected.issubset(set(result.columns))

    def test_sorted_descending(self, master_df):
        result = build_zip_demand_summary(master_df)
        assert result["demand_order_count"].is_monotonic_decreasing

    def test_no_duplicate_zips(self, master_df):
        result = build_zip_demand_summary(master_df)
        assert result["customer_zip_code_prefix"].nunique() == len(result)


# ---------------------------------------------------------------------------
# compute_baseline_kpis
# ---------------------------------------------------------------------------


class TestComputeBaselineKPIs:
    def test_returns_dict(self, master_df):
        kpis = compute_baseline_kpis(master_df)
        assert isinstance(kpis, dict)

    def test_required_keys(self, master_df):
        kpis = compute_baseline_kpis(master_df)
        required = [
            "mean_cust_seller_dist_km",
            "mean_delivery_days",
            "return_rate_pct",
            "total_orders",
            "mean_order_value",
        ]
        for key in required:
            assert key in kpis, f"Missing key: {key}"

    def test_total_orders_matches_input(self, master_df):
        kpis = compute_baseline_kpis(master_df)
        assert kpis["total_orders"] == len(master_df)

    def test_return_rate_bounded(self, master_df):
        kpis = compute_baseline_kpis(master_df)
        assert 0 <= kpis["return_rate_pct"] <= 100

    def test_dark_store_coverage_zero(self, master_df):
        kpis = compute_baseline_kpis(master_df)
        assert kpis["dark_store_coverage_pct"] == 0.0


# ---------------------------------------------------------------------------
# enrich_master_df
# ---------------------------------------------------------------------------


class TestEnrichMasterDf:
    def test_adds_demand_per_zip(self, master_df):
        zip_summary = build_zip_demand_summary(master_df)
        result = enrich_master_df(master_df, zip_summary)
        assert "demand_per_zip" in result.columns

    def test_demand_per_zip_positive(self, master_df):
        zip_summary = build_zip_demand_summary(master_df)
        result = enrich_master_df(master_df, zip_summary)
        assert (result["demand_per_zip"] >= 1).all()

    def test_adds_cust_seller_dist(self, master_df):
        zip_summary = build_zip_demand_summary(master_df)
        result = enrich_master_df(master_df, zip_summary)
        assert "cust_seller_dist_km" in result.columns

    def test_row_count_preserved(self, master_df):
        zip_summary = build_zip_demand_summary(master_df)
        result = enrich_master_df(master_df, zip_summary)
        assert len(result) == len(master_df)
