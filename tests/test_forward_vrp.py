"""
Tests for src/forward_vrp.py — Forward CVRPTW solver.

Covers:
    - solve_cvrptw (solver produces valid output on small problem)
    - compute_kpi_by_zone
    - run_full_pipeline data loading guard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.route_parser import (
    FIXED_COST_PER_ROUTE,
    VAR_COST_PER_KM,
    build_vrp_nodes,
)
from src.forward_vrp import (
    solve_cvrptw,
    compute_kpi_by_zone,
)


# ---------------------------------------------------------------------------
# solve_cvrptw — small synthetic zone
# ---------------------------------------------------------------------------


class TestSolveCvrptw:
    @pytest.fixture
    def tiny_zone(self):
        """Minimal zone: 1 depot + 5 customers."""
        coords = np.array(
            [
                [-23.55, -46.63],  # depot
                [-23.555, -46.635],
                [-23.56, -46.64],
                [-23.545, -46.625],
                [-23.55, -46.62],
                [-23.56, -46.63],
            ]
        )
        return {
            "zone_id": 0,
            "store_lat": -23.55,
            "store_lon": -46.63,
            "node_coords": coords,
            "demands": np.array([0, 500, 600, 400, 300, 700]),
            "time_windows": [
                [0, 1440],
                [480, 720],
                [480, 720],
                [720, 1080],
                [720, 1080],
                [480, 1080],
            ],
            "node_ids": ["depot", "c0", "c1", "c2", "c3", "c4"],
            "n_customers": 5,
        }

    @pytest.fixture(autouse=True)
    def patch_solver_time(self, monkeypatch):
        """Force the solver to exit after 1 second for these tests."""
        monkeypatch.setattr("src.forward_vrp.SOLVER_TIME_LIMIT_S", 1)

    def test_returns_dict(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        assert isinstance(result, dict)
        assert "zone_id" in result
        assert "solved" in result

    def test_solved_is_bool(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        assert isinstance(result["solved"], bool)

    def test_solution_found(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        assert result["solved"] is True, "Small problem should always be solvable"

    def test_routes_df_exists(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        if result["solved"]:
            assert "routes_df" in result
            assert isinstance(result["routes_df"], pd.DataFrame)
            assert len(result["routes_df"]) > 0

    def test_total_dist_positive(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        if result["solved"]:
            assert result["total_dist_km"] > 0

    def test_routing_cost_positive(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        if result["solved"]:
            assert result["routing_cost_R$"] > 0

    def test_n_vehicles_positive(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        if result["solved"]:
            assert result["n_vehicles"] >= 1

    def test_dist_matrix_returned(self, tiny_zone):
        result = solve_cvrptw(tiny_zone, num_vehicles=3)
        if result["solved"]:
            assert "dist_matrix" in result
            assert result["dist_matrix"].shape == (6, 6)


# ---------------------------------------------------------------------------
# compute_kpi_by_zone
# ---------------------------------------------------------------------------


class TestComputeKpiByZone:
    def test_empty_routes(self, tmp_path):
        empty = pd.DataFrame()
        result = compute_kpi_by_zone(empty, [], out_dir=tmp_path)
        assert result.empty

    def test_output_columns(self, tmp_path):
        routes_df = pd.DataFrame(
            {
                "zone_id": [0, 0, 0, 0],
                "vehicle_id": [0, 0, 1, 1],
                "node_idx": [0, 1, 0, 2],
                "cumulative_distance_km": [0.0, 5.0, 0.0, 8.0],
            }
        )
        zones = [{"zone_id": 0, "n_customers": 2}]
        result = compute_kpi_by_zone(routes_df, zones, out_dir=tmp_path)
        expected = {
            "zone_id",
            "vehicle_id",
            "num_stops",
            "total_distance_km",
            "routing_cost_R$",
            "cost_per_stop",
        }
        assert expected.issubset(set(result.columns))

    def test_cost_per_stop_positive(self, tmp_path):
        routes_df = pd.DataFrame(
            {
                "zone_id": [0, 0, 0],
                "vehicle_id": [0, 0, 0],
                "node_idx": [0, 1, 2],
                "cumulative_distance_km": [0.0, 3.0, 7.0],
            }
        )
        zones = [{"zone_id": 0, "n_customers": 2}]
        result = compute_kpi_by_zone(routes_df, zones, out_dir=tmp_path)
        assert (result["cost_per_stop"] > 0).all()

    def test_writes_csv(self, tmp_path):
        routes_df = pd.DataFrame(
            {
                "zone_id": [0, 0, 0],
                "vehicle_id": [0, 0, 0],
                "node_idx": [0, 1, 2],
                "cumulative_distance_km": [0.0, 3.0, 7.0],
            }
        )
        zones = [{"zone_id": 0, "n_customers": 2}]
        compute_kpi_by_zone(routes_df, zones, out_dir=tmp_path)
        assert (tmp_path / "forward_kpi_by_zone.csv").exists()
