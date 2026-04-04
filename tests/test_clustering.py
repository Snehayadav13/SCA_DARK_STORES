"""
Tests for src/clustering.py — K-Means + p-Median + Voronoi + coverage.

Covers:
    - haversine_km (scalar + vectorised)
    - load_sp_data validation
    - build_zip_level_coords aggregation
    - run_kmeans sweep
    - pick_optimal_k (silhouette max)
    - pick_k_by_coverage (coverage target)
    - run_p_median MILP formulation
    - build_p_median_locations_df
    - assign_voronoi (nearest-neighbour)
    - compute_coverage
    - build_dark_stores_df
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    haversine_km,
    build_zip_level_coords,
    run_kmeans,
    pick_optimal_k,
    pick_k_by_coverage,
    run_p_median,
    build_p_median_locations_df,
    assign_voronoi,
    compute_coverage,
    build_dark_stores_df,
)


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------


class TestClusteringHaversine:
    def test_zero_distance(self):
        assert np.isclose(haversine_km(-23.5, -46.6, -23.5, -46.6), 0.0)

    def test_symmetry(self):
        d1 = haversine_km(-23.5, -46.6, -22.9, -43.2)
        d2 = haversine_km(-22.9, -43.2, -23.5, -46.6)
        assert np.isclose(d1, d2)

    def test_vectorised(self):
        lats1 = np.array([-23.5, -23.6])
        lons1 = np.array([-46.6, -46.7])
        lats2 = np.array([-23.55, -23.65])
        lons2 = np.array([-46.65, -46.75])
        result = haversine_km(lats1, lons1, lats2, lons2)
        assert result.shape == (2,)
        assert np.all(result >= 0)

    def test_known_distance(self):
        """SP metro: 0.1° lat ≈ 11.1 km"""
        d = haversine_km(-23.5, -46.6, -23.4, -46.6)
        assert 10 < d < 12


# ---------------------------------------------------------------------------
# build_zip_level_coords
# ---------------------------------------------------------------------------


class TestBuildZipLevelCoords:
    def test_output_columns(self, master_df_v2):
        result = build_zip_level_coords(master_df_v2)
        assert set(result.columns) >= {
            "customer_zip_code_prefix",
            "lat",
            "lon",
            "demand_weight",
        }

    def test_one_row_per_zip(self, master_df_v2):
        result = build_zip_level_coords(master_df_v2)
        assert result["customer_zip_code_prefix"].nunique() == len(result)

    def test_demand_weight_positive(self, master_df_v2):
        result = build_zip_level_coords(master_df_v2)
        assert (result["demand_weight"] > 0).all()


# ---------------------------------------------------------------------------
# run_kmeans
# ---------------------------------------------------------------------------


class TestRunKMeans:
    def test_returns_dict(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        assert isinstance(results, dict)
        assert set(results.keys()) == {2, 3, 4}

    def test_result_keys(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 4))
        for k, res in results.items():
            assert "inertia" in res
            assert "silhouette" in res
            assert "centroids" in res
            assert "labels" in res

    def test_centroids_shape(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        assert results[3]["centroids"].shape == (3, 2)

    def test_labels_length(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        assert len(results[3]["labels"]) == len(small_coords)

    def test_inertia_decreases(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 6))
        inertias = [results[k]["inertia"] for k in range(2, 6)]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_with_weights(self, small_coords):
        weights = np.array([10, 5, 3, 8, 6, 4])
        results = run_kmeans(small_coords, weights=weights, k_range=range(2, 4))
        assert len(results) == 2

    def test_silhouette_bounded(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        for k, res in results.items():
            assert -1 <= res["silhouette"] <= 1


# ---------------------------------------------------------------------------
# pick_optimal_k
# ---------------------------------------------------------------------------


class TestPickOptimalK:
    def test_picks_max_silhouette(self):
        results = {
            3: {"silhouette": 0.4},
            4: {"silhouette": 0.6},
            5: {"silhouette": 0.5},
        }
        assert pick_optimal_k(results) == 4

    def test_single_option(self):
        results = {7: {"silhouette": 0.3}}
        assert pick_optimal_k(results) == 7


# ---------------------------------------------------------------------------
# pick_k_by_coverage
# ---------------------------------------------------------------------------


class TestPickKByCoverage:
    def test_picks_minimum_k_above_target(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 6))
        k, cov_by_k = pick_k_by_coverage(
            results,
            small_coords,
            target_coverage=0.0,
            radius_km=100.0,  # very generous
        )
        assert k == 2  # minimum K that hits 0% target

    def test_returns_coverage_dict(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        _, cov_by_k = pick_k_by_coverage(results, small_coords)
        assert isinstance(cov_by_k, dict)
        assert len(cov_by_k) == 3

    def test_coverage_values_bounded(self, small_coords):
        results = run_kmeans(small_coords, k_range=range(2, 5))
        _, cov_by_k = pick_k_by_coverage(results, small_coords)
        for v in cov_by_k.values():
            assert 0 <= v <= 1


# ---------------------------------------------------------------------------
# run_p_median
# ---------------------------------------------------------------------------


class TestRunPMedian:
    def test_opens_correct_number(self):
        distances = np.array(
            [
                [0, 10, 20],
                [10, 0, 15],
                [20, 15, 0],
            ],
            dtype=float,
        )
        demands = np.array([1, 1, 1], dtype=float)
        opened = run_p_median(distances, demands, p=2)
        assert len(opened) == 2

    def test_opened_indices_valid(self):
        distances = np.random.default_rng(42).uniform(1, 50, (5, 5))
        np.fill_diagonal(distances, 0)
        demands = np.ones(5)
        opened = run_p_median(distances, demands, p=3)
        assert all(0 <= idx < 5 for idx in opened)

    def test_p_equals_n_opens_all(self):
        n = 4
        distances = np.random.default_rng(7).uniform(1, 50, (n, n))
        np.fill_diagonal(distances, 0)
        demands = np.ones(n)
        opened = run_p_median(distances, demands, p=n)
        assert len(opened) == n

    def test_p_equals_one(self):
        distances = np.array(
            [
                [0, 10, 20],
                [10, 0, 15],
                [20, 15, 0],
            ],
            dtype=float,
        )
        demands = np.array([1, 1, 1], dtype=float)
        opened = run_p_median(distances, demands, p=1)
        assert len(opened) == 1


# ---------------------------------------------------------------------------
# build_p_median_locations_df
# ---------------------------------------------------------------------------


class TestBuildPMedianLocationsDf:
    def test_output_columns(self, small_coords):
        df = build_p_median_locations_df(small_coords, [0, 2, 4])
        assert set(df.columns) >= {
            "p_median_store_id",
            "source_centroid_idx",
            "lat",
            "lon",
        }

    def test_output_length(self, small_coords):
        df = build_p_median_locations_df(small_coords, [0, 2])
        assert len(df) == 2

    def test_sorted_by_index(self, small_coords):
        df = build_p_median_locations_df(small_coords, [4, 0, 2])
        assert df["source_centroid_idx"].tolist() == [0, 2, 4]


# ---------------------------------------------------------------------------
# assign_voronoi
# ---------------------------------------------------------------------------


class TestAssignVoronoi:
    def test_output_length(self, small_coords):
        centroids = small_coords[:2]
        labels = assign_voronoi(small_coords, centroids)
        assert len(labels) == len(small_coords)

    def test_labels_in_range(self, small_coords):
        centroids = small_coords[:3]
        labels = assign_voronoi(small_coords, centroids)
        assert all(0 <= l < 3 for l in labels)

    def test_depot_assigned_to_self(self, small_coords):
        centroids = small_coords.copy()
        labels = assign_voronoi(small_coords, centroids)
        for i in range(len(small_coords)):
            assert labels[i] == i


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_coverage_large_radius(self, small_coords):
        centroids = small_coords[:3]
        cov = compute_coverage(small_coords, centroids, radius_km=1000)
        assert np.isclose(cov, 1.0)

    def test_zero_coverage_tiny_radius(self, small_coords):
        centroids = np.array([[-20.0, -40.0]])  # very far away
        cov = compute_coverage(small_coords, centroids, radius_km=0.001)
        assert cov == 0.0

    def test_coverage_bounded(self, small_coords):
        centroids = small_coords[:2]
        cov = compute_coverage(small_coords, centroids, radius_km=5.0)
        assert 0 <= cov <= 1

    def test_more_centroids_increases_coverage(self, small_coords):
        cov_2 = compute_coverage(small_coords, small_coords[:2], radius_km=5.0)
        cov_4 = compute_coverage(small_coords, small_coords[:4], radius_km=5.0)
        assert cov_4 >= cov_2


# ---------------------------------------------------------------------------
# build_dark_stores_df
# ---------------------------------------------------------------------------


class TestBuildDarkStoresDf:
    def test_output_columns(self, master_df_v2):
        centroids = master_df_v2[["customer_lat", "customer_lon"]].values[:5]
        result = build_dark_stores_df(centroids, master_df_v2)
        expected = {
            "dark_store_id",
            "lat",
            "lon",
            "n_orders",
            "capacity_orders",
            "coverage_5km_pct",
        }
        assert expected.issubset(set(result.columns))

    def test_output_length_equals_k(self, master_df_v2):
        k = 5
        centroids = master_df_v2[["customer_lat", "customer_lon"]].values[:k]
        result = build_dark_stores_df(centroids, master_df_v2)
        assert len(result) == k

    def test_coverage_bounded(self, master_df_v2):
        centroids = master_df_v2[["customer_lat", "customer_lon"]].values[:5]
        result = build_dark_stores_df(centroids, master_df_v2)
        assert result["coverage_5km_pct"].between(0, 100).all()
