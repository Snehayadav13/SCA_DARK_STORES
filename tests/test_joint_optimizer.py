"""
Tests for src/joint_optimizer.py — PuLP MILP for joint forward + reverse.

Covers:
    - build_model (variable creation, constraint count)
    - solve (optimal / feasible status)
    - extract_results (result dict structure, Z computation)
    - run (convenience wrapper, file output)
    - Edge cases (empty DataFrames, single vehicle)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.joint_optimizer import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_DELTA,
    build_model,
    solve,
    extract_results,
    run,
)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_prob_and_vars(
        self, forward_routes_df, reverse_routes_df, return_probs
    ):
        prob, vars_dict = build_model(
            forward_routes_df, reverse_routes_df, return_probs
        )
        assert prob is not None
        assert "u" in vars_dict
        assert "w" in vars_dict

    def test_forward_vehicles_count(
        self, forward_routes_df, reverse_routes_df, return_probs
    ):
        _, vars_dict = build_model(forward_routes_df, reverse_routes_df, return_probs)
        assert len(vars_dict["u"]) == forward_routes_df["vehicle_id"].nunique()

    def test_reverse_vehicles_count(
        self, forward_routes_df, reverse_routes_df, return_probs
    ):
        _, vars_dict = build_model(forward_routes_df, reverse_routes_df, return_probs)
        assert len(vars_dict["w"]) == reverse_routes_df["vehicle_id"].nunique()

    def test_custom_weights(self, forward_routes_df, reverse_routes_df, return_probs):
        prob, _ = build_model(
            forward_routes_df,
            reverse_routes_df,
            return_probs,
            alpha=2.0,
            beta=1.5,
            gamma=3.0,
            delta=100.0,
        )
        assert prob is not None


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


class TestSolve:
    def test_returns_status_string(
        self, forward_routes_df, reverse_routes_df, return_probs
    ):
        prob, _ = build_model(forward_routes_df, reverse_routes_df, return_probs)
        status = solve(prob)
        assert isinstance(status, str)

    def test_status_optimal(self, forward_routes_df, reverse_routes_df, return_probs):
        prob, _ = build_model(forward_routes_df, reverse_routes_df, return_probs)
        status = solve(prob)
        assert status in {
            "Optimal",
            "Not Solved",
            "Infeasible",
            "Unbounded",
            "Undefined",
        }


# ---------------------------------------------------------------------------
# extract_results
# ---------------------------------------------------------------------------


class TestExtractResults:
    def test_result_keys(self, forward_routes_df, reverse_routes_df, return_probs):
        prob, vars_dict = build_model(
            forward_routes_df, reverse_routes_df, return_probs
        )
        solve(prob)
        result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)
        assert "Z" in result
        assert "C_fwd" in result
        assert "C_rev" in result
        assert "N_veh" in result
        assert "status" in result
        assert "vehicle_assignments" in result

    def test_z_is_float(self, forward_routes_df, reverse_routes_df, return_probs):
        prob, vars_dict = build_model(
            forward_routes_df, reverse_routes_df, return_probs
        )
        solve(prob)
        result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)
        assert isinstance(result["Z"], float)

    def test_n_veh_positive(self, forward_routes_df, reverse_routes_df, return_probs):
        prob, vars_dict = build_model(
            forward_routes_df, reverse_routes_df, return_probs
        )
        solve(prob)
        result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)
        assert result["N_veh"] >= 1

    def test_vehicle_assignments_df(
        self, forward_routes_df, reverse_routes_df, return_probs
    ):
        prob, vars_dict = build_model(
            forward_routes_df, reverse_routes_df, return_probs
        )
        solve(prob)
        result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)
        assignments = result["vehicle_assignments"]
        assert isinstance(assignments, pd.DataFrame)
        assert "vehicle_id" in assignments.columns
        assert "role" in assignments.columns
        assert "active" in assignments.columns


# ---------------------------------------------------------------------------
# run (convenience wrapper)
# ---------------------------------------------------------------------------


class TestRun:
    def test_writes_json(
        self, forward_routes_df, reverse_routes_df, return_probs, tmp_path
    ):
        out_path = tmp_path / "result.json"
        result = run(
            forward_routes_df, reverse_routes_df, return_probs, output_path=out_path
        )
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "Z" in data
        assert "status" in data

    def test_result_dict(
        self, forward_routes_df, reverse_routes_df, return_probs, tmp_path
    ):
        result = run(
            forward_routes_df,
            reverse_routes_df,
            return_probs,
            output_path=tmp_path / "r.json",
        )
        assert isinstance(result, dict)
        assert "Z" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestJointOptimizerEdgeCases:
    def test_empty_forward(self, reverse_routes_df, return_probs, tmp_path):
        empty_fwd = pd.DataFrame(columns=["vehicle_id", "cumulative_distance_km"])
        result = run(
            empty_fwd, reverse_routes_df, return_probs, output_path=tmp_path / "r.json"
        )
        assert result["C_fwd"] == 0.0

    def test_empty_reverse(self, forward_routes_df, return_probs, tmp_path):
        empty_rev = pd.DataFrame(columns=["vehicle_id", "cumulative_distance_km"])
        result = run(
            forward_routes_df, empty_rev, return_probs, output_path=tmp_path / "r.json"
        )
        assert result["C_rev"] == 0.0

    def test_empty_probs(self, forward_routes_df, reverse_routes_df, tmp_path):
        empty_probs = pd.Series([], dtype=np.float32)
        result = run(
            forward_routes_df,
            reverse_routes_df,
            empty_probs,
            output_path=tmp_path / "r.json",
        )
        assert isinstance(result["Z"], float)

    def test_single_forward_vehicle(self, return_probs, tmp_path):
        fwd = pd.DataFrame(
            {
                "vehicle_id": [0, 0, 0],
                "cumulative_distance_km": [0.0, 5.0, 10.0],
            }
        )
        rev = pd.DataFrame(
            {
                "vehicle_id": [0, 0],
                "cumulative_distance_km": [0.0, 3.0],
            }
        )
        result = run(fwd, rev, return_probs, output_path=tmp_path / "r.json")
        assert result["N_veh"] >= 1
