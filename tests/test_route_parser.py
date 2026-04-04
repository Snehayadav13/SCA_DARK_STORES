"""
Tests for src/route_parser.py — shared VRP utilities.

Covers:
    - Constants sanity
    - build_distance_matrix (Haversine in metres)
    - build_vrp_nodes (forward)
    - build_reverse_vrp_nodes
    - parse_solution (mock OR-Tools objects)
    - compute_routing_cost
    - nodes_to_csv
    - save_routes
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.route_parser import (
    VEHICLE_CAPACITY_G,
    VEHICLE_SPEED_KMH,
    FIXED_COST_PER_ROUTE,
    VAR_COST_PER_KM,
    SERVICE_TIME_MIN,
    SOLVER_TIME_LIMIT_S,
    MAX_CUSTOMERS_PER_ZONE,
    NUM_VEHICLES,
    RETURN_PROB_THRESHOLD,
    build_distance_matrix,
    build_vrp_nodes,
    build_reverse_vrp_nodes,
    compute_routing_cost,
    nodes_to_csv,
    save_routes,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_vehicle_capacity_positive(self):
        assert VEHICLE_CAPACITY_G > 0

    def test_vehicle_speed_positive(self):
        assert VEHICLE_SPEED_KMH > 0

    def test_fixed_cost_non_negative(self):
        assert FIXED_COST_PER_ROUTE >= 0

    def test_var_cost_non_negative(self):
        assert VAR_COST_PER_KM >= 0

    def test_service_time_positive(self):
        assert SERVICE_TIME_MIN > 0

    def test_solver_time_limit_positive(self):
        assert SOLVER_TIME_LIMIT_S > 0

    def test_max_customers_positive(self):
        assert MAX_CUSTOMERS_PER_ZONE > 0

    def test_num_vehicles_positive(self):
        assert NUM_VEHICLES > 0

    def test_return_prob_threshold_bounded(self):
        assert 0 < RETURN_PROB_THRESHOLD < 1


# ---------------------------------------------------------------------------
# build_distance_matrix (route_parser version — metres, float)
# ---------------------------------------------------------------------------


class TestRouteParserDistMatrix:
    def test_shape_is_square(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert mat.shape == (6, 6)

    def test_diagonal_zero(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert np.allclose(np.diag(mat), 0)

    def test_symmetric(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert np.allclose(mat, mat.T, atol=1)

    def test_off_diagonal_positive(self, small_coords):
        mat = build_distance_matrix(small_coords)
        mask = ~np.eye(6, dtype=bool)
        assert np.all(mat[mask] > 0)

    def test_units_are_metres(self, small_coords):
        """Central SP to ~6 km north → ~6000 metres."""
        mat = build_distance_matrix(small_coords)
        d_metres = mat[0, 1]
        assert 4000 < d_metres < 10000


# ---------------------------------------------------------------------------
# build_vrp_nodes (forward)
# ---------------------------------------------------------------------------


class TestBuildVrpNodes:
    def test_returns_list_of_dicts(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df)
        assert isinstance(zones, list)
        if len(zones) > 0:
            assert isinstance(zones[0], dict)

    def test_zone_dict_keys(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df)
        for z in zones:
            assert "zone_id" in z
            assert "node_coords" in z
            assert "demands" in z
            assert "time_windows" in z
            assert "node_ids" in z
            assert "n_customers" in z

    def test_node_0_is_depot(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df)
        for z in zones:
            assert z["node_ids"][0] == "depot"

    def test_depot_demand_is_zero(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df)
        for z in zones:
            assert z["demands"][0] == 0

    def test_depot_time_window_all_day(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df)
        for z in zones:
            assert z["time_windows"][0] == [0, 1440]

    def test_max_customers_respected(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=10)
        for z in zones:
            assert z["n_customers"] <= 10

    def test_node_coords_shape(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=10)
        for z in zones:
            n = z["n_customers"] + 1  # customers + depot
            assert z["node_coords"].shape == (n, 2)

    def test_demands_length_matches_nodes(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=10)
        for z in zones:
            assert len(z["demands"]) == len(z["node_coords"])

    def test_time_windows_length_matches_nodes(self, master_df_v3, dark_stores_df):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=10)
        for z in zones:
            assert len(z["time_windows"]) == len(z["node_coords"])

    def test_empty_zone_skipped(self, dark_stores_df):
        """If no orders map to a store, that zone should be skipped."""
        empty_df = pd.DataFrame(
            {
                "dark_store_id": [],
                "customer_lat": [],
                "customer_lon": [],
                "product_weight_g": [],
                "order_purchase_timestamp": [],
            }
        )
        zones = build_vrp_nodes(empty_df, dark_stores_df)
        assert len(zones) == 0


# ---------------------------------------------------------------------------
# build_reverse_vrp_nodes
# ---------------------------------------------------------------------------


class TestBuildReverseVrpNodes:
    def test_returns_list(self, master_df_v3, dark_stores_df):
        return_df = master_df_v3[master_df_v3["return_flag"] == 1].copy()
        zones = build_reverse_vrp_nodes(return_df, dark_stores_df)
        assert isinstance(zones, list)

    def test_n_pickups_key(self, master_df_v3, dark_stores_df):
        return_df = master_df_v3[master_df_v3["return_flag"] == 1].copy()
        zones = build_reverse_vrp_nodes(return_df, dark_stores_df)
        for z in zones:
            assert "n_pickups" in z

    def test_depot_is_first_node(self, master_df_v3, dark_stores_df):
        return_df = master_df_v3[master_df_v3["return_flag"] == 1].copy()
        zones = build_reverse_vrp_nodes(return_df, dark_stores_df)
        for z in zones:
            assert z["node_ids"][0] == "depot"

    def test_no_zones_for_empty_returns(self, dark_stores_df):
        empty_return_df = pd.DataFrame(
            {
                "dark_store_id": [],
                "customer_lat": [],
                "customer_lon": [],
                "product_weight_g": [],
                "order_purchase_timestamp": [],
            }
        )
        zones = build_reverse_vrp_nodes(empty_return_df, dark_stores_df)
        assert len(zones) == 0


# ---------------------------------------------------------------------------
# compute_routing_cost
# ---------------------------------------------------------------------------


class TestComputeRoutingCost:
    def test_formula(self):
        cost = compute_routing_cost(n_vehicles=3, total_dist_km=100.0)
        expected = FIXED_COST_PER_ROUTE * 3 + VAR_COST_PER_KM * 100.0
        assert np.isclose(cost, expected)

    def test_zero_distance(self):
        cost = compute_routing_cost(n_vehicles=1, total_dist_km=0.0)
        assert np.isclose(cost, FIXED_COST_PER_ROUTE)

    def test_zero_vehicles(self):
        cost = compute_routing_cost(n_vehicles=0, total_dist_km=50.0)
        expected = VAR_COST_PER_KM * 50.0
        assert np.isclose(cost, expected)

    def test_monotone_in_distance(self):
        c1 = compute_routing_cost(1, 100)
        c2 = compute_routing_cost(1, 200)
        assert c2 > c1


# ---------------------------------------------------------------------------
# nodes_to_csv
# ---------------------------------------------------------------------------


class TestNodesToCsv:
    def test_writes_csv(self, master_df_v3, dark_stores_df, tmp_path):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=5)
        nodes_to_csv(zones, tmp_path, "test_nodes.csv")
        out = tmp_path / "test_nodes.csv"
        assert out.exists()
        df = pd.read_csv(out)
        assert "zone_id" in df.columns
        assert "node_idx" in df.columns
        assert "is_depot" in df.columns

    def test_depot_row_exists(self, master_df_v3, dark_stores_df, tmp_path):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=5)
        nodes_to_csv(zones, tmp_path, "test_nodes.csv")
        df = pd.read_csv(tmp_path / "test_nodes.csv")
        depots = df[df["is_depot"] == 1]
        assert len(depots) == len(zones)

    def test_demand_is_integer(self, master_df_v3, dark_stores_df, tmp_path):
        zones = build_vrp_nodes(master_df_v3, dark_stores_df, max_per_zone=5)
        nodes_to_csv(zones, tmp_path, "test_nodes.csv")
        df = pd.read_csv(tmp_path / "test_nodes.csv")
        assert df["demand_g"].dtype in [np.int64, np.int32, int]


# ---------------------------------------------------------------------------
# save_routes
# ---------------------------------------------------------------------------


class TestSaveRoutes:
    def _make_zone_results(self):
        """Synthetic solved zone result."""
        routes_df = pd.DataFrame(
            {
                "vehicle_id": [0, 0, 0],
                "node_idx": [0, 1, 2],
                "node_id": ["depot", "c1", "c2"],
                "lat": [-23.55, -23.56, -23.57],
                "lon": [-46.63, -46.64, -46.65],
                "cumulative_distance_km": [0.0, 3.5, 7.0],
            }
        )
        return {
            0: {
                "zone_id": 0,
                "solved": True,
                "routes_df": routes_df,
                "total_dist_km": 7.0,
                "n_vehicles": 1,
                "routing_cost_R$": 60.5,
                "summary": {
                    "total_distance_km": 7.0,
                    "n_vehicles_used": 1,
                    "max_route_km": 7.0,
                    "min_route_km": 7.0,
                },
            }
        }

    def _make_zones(self):
        return [{"zone_id": 0, "n_customers": 2}]

    def test_writes_csv_and_json(self, tmp_path):
        results = self._make_zone_results()
        zones = self._make_zones()
        save_routes(results, zones, tmp_path, prefix="forward")
        assert (tmp_path / "forward_routes.csv").exists()
        assert (tmp_path / "forward_routes.json").exists()
        assert (tmp_path / "forward_kpi_summary.csv").exists()

    def test_kpi_summary_columns(self, tmp_path):
        results = self._make_zone_results()
        zones = self._make_zones()
        _, kpi_df = save_routes(results, zones, tmp_path, prefix="forward")
        assert "zone_id" in kpi_df.columns
        assert "routing_cost_R$" in kpi_df.columns
        assert "total_dist_km" in kpi_df.columns

    def test_json_valid(self, tmp_path):
        results = self._make_zone_results()
        zones = self._make_zones()
        save_routes(results, zones, tmp_path, prefix="test")
        with open(tmp_path / "test_routes.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["zone_id"] == 0

    def test_reverse_prefix(self, tmp_path):
        results = self._make_zone_results()
        zones = [{"zone_id": 0, "n_pickups": 2}]
        save_routes(results, zones, tmp_path, prefix="reverse")
        assert (tmp_path / "reverse_routes.csv").exists()
        kpi = pd.read_csv(tmp_path / "reverse_kpi_summary.csv")
        assert "n_pickups" in kpi.columns

    def test_unsolved_zone_skipped(self, tmp_path):
        results = {0: {"zone_id": 0, "solved": False}}
        zones = [{"zone_id": 0, "n_customers": 5}]
        routes_df, kpi_df = save_routes(results, zones, tmp_path, prefix="forward")
        assert routes_df.empty
        assert kpi_df.empty
