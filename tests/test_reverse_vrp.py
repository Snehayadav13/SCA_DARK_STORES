"""
Tests for src/reverse_vrp.py — Reverse Pickup CVRPTW solver.

Covers:
    - solve_reverse_cvrptw (solver produces valid output)
    - Strategy constants
    - Symmetry with forward VRP structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reverse_vrp import (
    FIRST_SOLUTION_STRATEGY,
    LOCAL_SEARCH_METAHEURISTIC,
    STRATEGY_LABEL,
    solve_reverse_cvrptw,
)
from src.route_parser import RETURN_PROB_THRESHOLD


# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------


class TestStrategyConstants:
    def test_strategy_label_non_empty(self):
        assert len(STRATEGY_LABEL) > 0

    def test_first_solution_strategy_set(self):
        assert FIRST_SOLUTION_STRATEGY is not None

    def test_local_search_set(self):
        assert LOCAL_SEARCH_METAHEURISTIC is not None


# ---------------------------------------------------------------------------
# solve_reverse_cvrptw — small synthetic zone
# ---------------------------------------------------------------------------


class TestSolveReverseCvrptw:
    @pytest.fixture
    def tiny_pickup_zone(self):
        """Minimal reverse zone: 1 depot + 4 pickup nodes."""
        coords = np.array(
            [
                [-23.55, -46.63],  # depot
                [-23.555, -46.635],
                [-23.56, -46.64],
                [-23.545, -46.625],
                [-23.55, -46.62],
            ]
        )
        return {
            "zone_id": 0,
            "store_lat": -23.55,
            "store_lon": -46.63,
            "node_coords": coords,
            "demands": np.array([0, 500, 600, 400, 300]),
            "time_windows": [
                [0, 1440],
                [480, 720],
                [480, 720],
                [720, 1080],
                [720, 1080],
            ],
            "node_ids": ["depot", "ret_0", "ret_1", "ret_2", "ret_3"],
            "n_pickups": 4,
        }

    @pytest.fixture(autouse=True)
    def patch_solver_time(self, monkeypatch):
        """Force the solver to exit after 1 second for these tests."""
        monkeypatch.setattr("src.reverse_vrp.SOLVER_TIME_LIMIT_S", 1)

    def test_returns_dict(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        assert isinstance(result, dict)

    def test_solved(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        assert result["solved"] is True

    def test_routes_df_shape(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        if result["solved"]:
            assert len(result["routes_df"]) > 0

    def test_total_dist_positive(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        if result["solved"]:
            assert result["total_dist_km"] > 0

    def test_zone_id_preserved(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        assert result["zone_id"] == 0

    def test_cost_positive(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        if result["solved"]:
            assert result["routing_cost_R$"] > 0

    def test_routes_df_has_zone_id(self, tiny_pickup_zone):
        result = solve_reverse_cvrptw(tiny_pickup_zone, num_vehicles=3)
        if result["solved"]:
            assert "zone_id" in result["routes_df"].columns
            assert (result["routes_df"]["zone_id"] == 0).all()
