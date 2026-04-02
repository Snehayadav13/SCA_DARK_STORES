"""
Module: joint_optimizer.py
Stage:  Joint Forward + Reverse Logistics Optimisation (MILP via PuLP)

Objective (minimise):
    Z = α·C_fwd + β·C_rev + γ·T_pen + δ·N_veh

    C_fwd  = total forward route distance (km)
    C_rev  = total reverse route distance (km)
    T_pen  = sum of late-delivery penalties
    N_veh  = number of vehicles deployed (forward + reverse)

INPUT:
    forward_routes_df  : pd.DataFrame  (output of route_parser.py, Day 3)
        Columns: vehicle_id, stop_seq, node_idx, node_id, lat, lon,
                 cumulative_distance_km, load_after_stop
    reverse_routes_df  : pd.DataFrame  (output of route_parser.py, Day 4)
        Same schema.
    return_probs       : pd.Series indexed by order_id  — float32 ∈ [0,1]
        Output of return_classifier.predict_proba()
    alpha, beta, gamma, delta : float — objective weights

OUTPUT:
    joint_result : dict
        {
            "Z": float,
            "C_fwd": float, "C_rev": float, "T_pen": float, "N_veh": int,
            "status": str ("Optimal" | "Feasible" | "Infeasible"),
            "vehicle_assignments": pd.DataFrame
                Columns: vehicle_id, role (forward|reverse), active (bool)
        }
    outputs/joint_optimizer_result.json

INTERFACE:
    build_model(forward_routes_df, reverse_routes_df, return_probs, alpha, beta, gamma, delta)
        -> (prob, decision_vars)
    solve(prob)
        -> str  # status string
    extract_results(prob, decision_vars, forward_routes_df, reverse_routes_df)
        -> dict
    run(...)
        -> dict  # convenience: build + solve + extract
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# Default objective weights (tuned on Day 6)
DEFAULT_ALPHA = 1.0   # forward cost weight
DEFAULT_BETA  = 0.8   # reverse cost weight (slightly cheaper per km)
DEFAULT_GAMMA = 2.0   # penalty weight for late delivery
DEFAULT_DELTA = 50.0  # fixed cost equivalent per vehicle
# ---------------------------------------------------------------------------


def build_model(
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
) -> tuple:
    """
    Construct the PuLP MILP for joint optimisation.

    Decision variables:
        u_v ∈ {0,1}  — vehicle v is active for forward routing
        w_v ∈ {0,1}  — vehicle v is active for reverse routing
        (Routes themselves are fixed from OR-Tools Day 3/4; this MILP chooses
         which vehicles to activate and handles shared-capacity trade-offs.)

    Returns
    -------
    (prob, vars_dict)
    """
    prob = pulp.LpProblem("joint_fwd_rev_optimisation", pulp.LpMinimize)

    fwd_vehicles = forward_routes_df["vehicle_id"].unique().tolist() if not forward_routes_df.empty else []
    rev_vehicles = reverse_routes_df["vehicle_id"].unique().tolist() if not reverse_routes_df.empty else []

    # Per-vehicle route cost (total km)
    fwd_cost = (
        forward_routes_df.groupby("vehicle_id")["cumulative_distance_km"].max().to_dict()
        if not forward_routes_df.empty else {}
    )
    rev_cost = (
        reverse_routes_df.groupby("vehicle_id")["cumulative_distance_km"].max().to_dict()
        if not reverse_routes_df.empty else {}
    )

    # Binary activation variables
    u = {v: pulp.LpVariable(f"u_{v}", cat="Binary") for v in fwd_vehicles}
    w = {v: pulp.LpVariable(f"w_{v}", cat="Binary") for v in rev_vehicles}

    # Expected return penalty: high return_prob orders that are on forward routes
    # incur a latency penalty if the reverse trip is not activated same day.
    expected_returns = float(return_probs.sum()) if not return_probs.empty else 0.0
    T_pen_expr = gamma * expected_returns * (1 - pulp.lpSum(w.values()) / max(len(rev_vehicles), 1))

    # Objective
    C_fwd_expr = alpha * pulp.lpSum(fwd_cost.get(v, 0) * u[v] for v in fwd_vehicles)
    C_rev_expr = beta  * pulp.lpSum(rev_cost.get(v, 0)  * w[v] for v in rev_vehicles)
    N_veh_expr = delta * (pulp.lpSum(u.values()) + pulp.lpSum(w.values()))

    prob += C_fwd_expr + C_rev_expr + T_pen_expr + N_veh_expr, "total_cost"

    # Constraints: at least one active vehicle of each type (if routes exist)
    if fwd_vehicles:
        prob += pulp.lpSum(u.values()) >= 1, "min_one_fwd_vehicle"
    if rev_vehicles:
        prob += pulp.lpSum(w.values()) >= 1, "min_one_rev_vehicle"

    return prob, {"u": u, "w": w, "fwd_cost": fwd_cost, "rev_cost": rev_cost,
                  "expected_returns": expected_returns}


def solve(prob: pulp.LpProblem, time_limit_s: int = 60) -> str:
    """Solve the MILP; returns status string."""
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit_s)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def extract_results(
    prob: pulp.LpProblem,
    vars_dict: dict,
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
) -> dict:
    """Parse solved MILP into a result dict."""
    u, w = vars_dict["u"], vars_dict["w"]
    fwd_cost, rev_cost = vars_dict["fwd_cost"], vars_dict["rev_cost"]

    active_fwd = [v for v, var in u.items() if pulp.value(var) and pulp.value(var) > 0.5]
    active_rev = [v for v, var in w.items() if pulp.value(var) and pulp.value(var) > 0.5]

    C_fwd = sum(fwd_cost.get(v, 0) for v in active_fwd)
    C_rev = sum(rev_cost.get(v, 0) for v in active_rev)
    N_veh = len(active_fwd) + len(active_rev)
    Z = pulp.value(prob.objective) or 0.0

    assignments = pd.DataFrame(
        [{"vehicle_id": v, "role": "forward",  "active": v in active_fwd} for v in u]
        + [{"vehicle_id": v, "role": "reverse", "active": v in active_rev} for v in w]
    )

    return {
        "Z": round(Z, 3),
        "C_fwd": round(C_fwd, 3),
        "C_rev": round(C_rev, 3),
        "T_pen": round(vars_dict["expected_returns"], 3),
        "N_veh": N_veh,
        "status": pulp.LpStatus[prob.status],
        "vehicle_assignments": assignments,
    }


def run(
    forward_routes_df: pd.DataFrame,
    reverse_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    output_path: str | Path = "outputs/joint_optimizer_result.json",
) -> dict:
    """Convenience: build → solve → extract → save result."""
    prob, vars_dict = build_model(forward_routes_df, reverse_routes_df, return_probs, alpha, beta, gamma, delta)
    status = solve(prob)
    result = extract_results(prob, vars_dict, forward_routes_df, reverse_routes_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in result.items() if k != "vehicle_assignments"}
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[INFO] Joint optimiser result → {output_path}")
    print(f"       Status={status}  Z={result['Z']}  Nveh={result['N_veh']}")
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data — just checks the MILP scaffolding runs
    fwd = pd.DataFrame({"vehicle_id": [0, 0, 1, 1], "cumulative_distance_km": [10, 20, 5, 15]})
    rev = pd.DataFrame({"vehicle_id": [0, 0], "cumulative_distance_km": [8, 16]})
    probs = pd.Series([0.3, 0.7, 0.1])
    result = run(fwd, rev, probs)
    print(result)
