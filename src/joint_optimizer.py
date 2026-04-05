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
    solve_sdvrp_hybrid(zone_id, fwd_zone, rev_zone, num_vehicles,
                       separate_cost_r, output_path)
        -> dict          # Day 5 (Pritam): SDVRP one-zone hybrid solve
    run_all_zones_sdvrp(fwd_zones, rev_zones, fwd_kpi_df, rev_kpi_df,
                        num_vehicles, output_dir)
        -> pd.DataFrame  # Day 5 (Pritam): run all zones, write hybrid_routes.json
                         #                 + hybrid_kpi_summary.csv
    z_sensitivity_sweep(fwd_routes_df, rev_routes_df, return_probs,
                        alpha_grid, beta_grid, output_path)
        -> pd.DataFrame  # Day 5 (Pritam): α/β grid, γ=δ=(1−α−β)/2
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import pulp
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_parser import (
    VEHICLE_CAPACITY_G,
    VEHICLE_SPEED_KMH,
    SERVICE_TIME_MIN,
    SOLVER_TIME_LIMIT_S,
    build_distance_matrix,
    compute_routing_cost,
)


# ---------------------------------------------------------------------------
# Default objective weights (tuned on Day 6)
DEFAULT_ALPHA = 1.0  # forward cost weight
DEFAULT_BETA = 0.8  # reverse cost weight (slightly cheaper per km)
DEFAULT_GAMMA = 2.0  # penalty weight for late delivery
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

    fwd_vehicles = (
        forward_routes_df["vehicle_id"].unique().tolist()
        if not forward_routes_df.empty
        else []
    )
    rev_vehicles = (
        reverse_routes_df["vehicle_id"].unique().tolist()
        if not reverse_routes_df.empty
        else []
    )

    # Per-vehicle route cost (total km)
    fwd_cost = (
        forward_routes_df.groupby("vehicle_id")["cumulative_distance_km"]
        .max()
        .to_dict()
        if not forward_routes_df.empty
        else {}
    )
    rev_cost = (
        reverse_routes_df.groupby("vehicle_id")["cumulative_distance_km"]
        .max()
        .to_dict()
        if not reverse_routes_df.empty
        else {}
    )

    # Binary activation variables
    u = {v: pulp.LpVariable(f"u_{v}", cat="Binary") for v in fwd_vehicles}
    w = {v: pulp.LpVariable(f"w_{v}", cat="Binary") for v in rev_vehicles}

    # Expected return penalty: high return_prob orders that are on forward routes
    # incur a latency penalty if the reverse trip is not activated same day.
    expected_returns = float(return_probs.sum()) if not return_probs.empty else 0.0
    T_pen_expr = (
        gamma
        * expected_returns
        * (1 - pulp.lpSum(w.values()) / max(len(rev_vehicles), 1))
    )

    # Objective
    C_fwd_expr = alpha * pulp.lpSum(fwd_cost.get(v, 0) * u[v] for v in fwd_vehicles)
    C_rev_expr = beta * pulp.lpSum(rev_cost.get(v, 0) * w[v] for v in rev_vehicles)
    N_veh_expr = delta * (pulp.lpSum(u.values()) + pulp.lpSum(w.values()))

    prob += C_fwd_expr + C_rev_expr + T_pen_expr + N_veh_expr, "total_cost"

    # Constraints: at least one active vehicle of each type (if routes exist)
    if fwd_vehicles:
        prob += pulp.lpSum(u.values()) >= 1, "min_one_fwd_vehicle"
    if rev_vehicles:
        prob += pulp.lpSum(w.values()) >= 1, "min_one_rev_vehicle"

    return prob, {
        "u": u,
        "w": w,
        "fwd_cost": fwd_cost,
        "rev_cost": rev_cost,
        "expected_returns": expected_returns,
    }


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

    active_fwd = [
        v for v, var in u.items() if pulp.value(var) and pulp.value(var) > 0.5
    ]
    active_rev = [
        v for v, var in w.items() if pulp.value(var) and pulp.value(var) > 0.5
    ]

    C_fwd = sum(fwd_cost.get(v, 0) for v in active_fwd)
    C_rev = sum(rev_cost.get(v, 0) for v in active_rev)
    N_veh = len(active_fwd) + len(active_rev)
    Z = pulp.value(prob.objective) or 0.0

    assignments = pd.DataFrame(
        [{"vehicle_id": v, "role": "forward", "active": v in active_fwd} for v in u]
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
    prob, vars_dict = build_model(
        forward_routes_df, reverse_routes_df, return_probs, alpha, beta, gamma, delta
    )
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
# SDVRP Hybrid Solver  (Pritam — Day 5)
# ---------------------------------------------------------------------------


def solve_sdvrp_hybrid(
    zone_id: int,
    fwd_zone: dict,
    rev_zone: dict,
    num_vehicles: int = 5,
    separate_cost_r: float | None = None,
    output_path: str | Path = "outputs/sdvrp_zone8_result.json",
) -> dict:
    """
    Simultaneous Delivery and Pickup VRP (SDVRP) for one zone.

    Merges forward delivery nodes and reverse pickup nodes into a single
    OR-Tools CVRPTW solve so vehicles can deliver and collect on the same trip.

    Capacity model — single "Load" dimension (correct SDVRP invariant):
        load(t) = initial_delivery_load − cumul_delivered(t) + cumul_picked(t)
        transit[i] = pickup_weight[i] − delivery_weight[i]
        fix_start_cumul_to_zero=False → OR-Tools sets start = total_delivery_wt
        Constraint: 0 ≤ Load_cumul[i] ≤ VEHICLE_CAPACITY_G for all nodes

    Parameters
    ----------
    zone_id         : zone ID (int, for labelling)
    fwd_zone        : zone dict from route_parser.build_vrp_nodes()
    rev_zone        : zone dict from route_parser.build_reverse_vrp_nodes()
    num_vehicles    : max vehicles available for hybrid solve
    separate_cost_r : known cost of running fwd+rev separately (R$), for saving calc
    output_path     : path to write result JSON

    Returns
    -------
    dict  — zone_id, solved, n_deliveries, n_pickups, total_dist_km,
            n_vehicles, hybrid_cost_R$, separate_cost_R$, saving_R$,
            saving_pct, strategy
    """
    # ---- 1. Build combined node list ---------------------------------- #
    # Node 0: depot  |  1..n_del: delivery  |  n_del+1..: pickup
    n_del = len(fwd_zone["node_coords"]) - 1
    n_pick = len(rev_zone["node_coords"]) - 1

    depot_coords = fwd_zone["node_coords"][[0]]
    del_coords = fwd_zone["node_coords"][1:]
    pick_coords = rev_zone["node_coords"][1:]
    node_coords = np.vstack([depot_coords, del_coords, pick_coords])

    del_weights = fwd_zone["demands"][1:].tolist()
    pick_weights = rev_zone["demands"][1:].tolist()

    del_demand_arr = [0] + del_weights + [0] * n_pick
    pick_demand_arr = [0] + [0] * n_del + pick_weights

    del_tw = fwd_zone["time_windows"][1:]
    pick_tw = rev_zone["time_windows"][1:]
    all_tw = [[0, 1440]] + del_tw + pick_tw
    n_nodes = 1 + n_del + n_pick

    # ---- 2. Distance & time matrices ---------------------------------- #
    dist_matrix = build_distance_matrix(node_coords)  # metres, float64
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60
    time_matrix = np.rint(dist_matrix / speed_m_per_min).astype(int)

    # ---- 3. OR-Tools model -------------------------------------------- #
    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def time_cb(i, j):
        ni = manager.IndexToNode(i)
        return int(time_matrix[ni][manager.IndexToNode(j)]) + (
            SERVICE_TIME_MIN if ni != 0 else 0
        )

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 60, 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node_idx, (open_t, close_t) in enumerate(all_tw):
        time_dim.CumulVar(manager.NodeToIndex(node_idx)).SetRange(open_t, close_t)

    # ---- 4. Single-dimension SDVRP load model ------------------------- #
    # Correct SDVRP invariant:
    #   load(t) = initial_delivery_load - delivered(t) + picked_up(t)
    # transit[i] = pickup_weight[i] - delivery_weight[i]  (net change per node)
    # fix_start_cumul_to_zero=False lets OR-Tools pick start = total_delivery_wt
    # for each vehicle; cumul bounded [0, VEHICLE_CAPACITY_G] enforces capacity.
    def load_transit_cb(i):
        node = manager.IndexToNode(i)
        return int(pick_demand_arr[node]) - int(del_demand_arr[node])

    load_idx = routing.RegisterUnaryTransitCallback(load_transit_cb)
    routing.AddDimensionWithVehicleCapacity(
        load_idx, 0, [VEHICLE_CAPACITY_G] * num_vehicles, False, "Load"
    )
    load_dim = routing.GetDimensionOrDie("Load")
    for node in range(n_nodes):
        idx = manager.NodeToIndex(node)
        if idx >= 0:
            load_dim.CumulVar(idx).SetMin(0)

    # Soft disjunction — large penalty for dropped nodes
    penalty = 100_000
    for node in range(1, n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # ---- 5. Solve ----------------------------------------------------- #
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    )
    params.time_limit.seconds = SOLVER_TIME_LIMIT_S

    assignment = routing.SolveWithParameters(params)

    if assignment is None:
        print(f"  [SDVRP] Zone {zone_id}: NO SOLUTION FOUND")
        return {"zone_id": zone_id, "solved": False, "routes": []}

    # ---- 6. Extract routes + KPIs ------------------------------------- #
    depot_id = fwd_zone.get("node_ids", ["depot"])[0]
    del_ids = list(fwd_zone.get("node_ids", [])[1:]) or [
        str(i) for i in range(1, n_del + 1)
    ]
    pick_ids = list(rev_zone.get("node_ids", [])[1:]) or [
        str(i) for i in range(1, n_pick + 1)
    ]
    all_node_ids = [depot_id] + del_ids + pick_ids
    node_types = ["depot"] + ["delivery"] * n_del + ["pickup"] * n_pick

    total_dist_m = 0.0
    n_veh_used = 0
    route_records: list[dict] = []
    veh_max_km: list[float] = []

    for v in range(num_vehicles):
        idx = routing.Start(v)
        cum_dist_km = 0.0
        stop_seq = 0
        veh_stops: list[dict] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            nxt = assignment.Value(routing.NextVar(idx))
            veh_stops.append(
                {
                    "vehicle_id": v,
                    "stop_seq": stop_seq,
                    "node_idx": node,
                    "node_id": all_node_ids[node],
                    "lat": float(node_coords[node][0]),
                    "lon": float(node_coords[node][1]),
                    "node_type": node_types[node],
                    "cumulative_distance_km": round(cum_dist_km, 3),
                    "zone_id": zone_id,
                }
            )
            cum_dist_km += dist_matrix[node][manager.IndexToNode(nxt)] / 1000.0
            stop_seq += 1
            idx = nxt
        if cum_dist_km > 0:
            total_dist_m += cum_dist_km * 1000.0
            n_veh_used += 1
            veh_max_km.append(cum_dist_km)
            route_records.extend(veh_stops)

    total_dist_km = total_dist_m / 1000.0
    hybrid_cost = compute_routing_cost(n_veh_used, total_dist_km)

    saving_r = round(separate_cost_r - hybrid_cost, 2) if separate_cost_r else None
    saving_pct = round(saving_r / separate_cost_r * 100, 1) if saving_r else None

    result = {
        "zone_id": zone_id,
        "solved": True,
        "n_deliveries": n_del,
        "n_pickups": n_pick,
        "total_dist_km": round(total_dist_km, 2),
        "n_vehicles": n_veh_used,
        "hybrid_cost_R$": round(hybrid_cost, 2),
        "separate_cost_R$": round(separate_cost_r, 2) if separate_cost_r else None,
        "saving_R$": saving_r,
        "saving_pct": saving_pct,
        "strategy": "PATH_CHEAPEST_ARC + SIMULATED_ANNEALING",
        "routes": route_records,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in result.items() if k != "routes"}
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    sep_str = (
        f"vs R${separate_cost_r:.2f} separate \u2192 {saving_pct:.1f}% saving"
        if saving_r
        else ""
    )
    print(
        f"[SDVRP] Zone {zone_id}: {n_veh_used} vehicles | "
        f"{total_dist_km:.2f} km | R${hybrid_cost:.2f}  {sep_str}"
    )
    return result


# ---------------------------------------------------------------------------
# All-Zone SDVRP Runner  (Pritam — Day 5)
# ---------------------------------------------------------------------------


def run_all_zones_sdvrp(
    fwd_zones: dict,
    rev_zones: dict,
    fwd_kpi_df: pd.DataFrame,
    rev_kpi_df: pd.DataFrame,
    num_vehicles: int = 5,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """
    Run solve_sdvrp_hybrid for every zone and write Vybhav-ready outputs.

    Parameters
    ----------
    fwd_zones    : zone dict map from route_parser.build_vrp_nodes()
    rev_zones    : zone dict map from route_parser.build_reverse_vrp_nodes()
    fwd_kpi_df   : forward_kpi_summary DataFrame (zone_id, routing_cost_R$)
    rev_kpi_df   : reverse_kpi_summary DataFrame (zone_id, routing_cost_R$)
    num_vehicles : max vehicles per zone hybrid solve
    output_dir   : directory to write outputs

    Returns
    -------
    pd.DataFrame — hybrid_kpi_summary (one row per solved zone)

    Writes
    ------
    <output_dir>/hybrid_routes.json       — matches forward_routes.json schema
    <output_dir>/hybrid_kpi_summary.csv   — per-zone KPIs + saving vs separate
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fwd_cost_map = fwd_kpi_df.set_index("zone_id")["routing_cost_R$"].to_dict()
    rev_cost_map = rev_kpi_df.set_index("zone_id")["routing_cost_R$"].to_dict()

    zone_ids = sorted(set(fwd_zones.keys()) & set(rev_zones.keys()))
    print(f"[SDVRP-all] Running {len(zone_ids)} zones: {zone_ids}")

    all_zone_routes: list[dict] = []
    kpi_rows: list[dict] = []

    for zone_id in zone_ids:
        fwd_zone = fwd_zones[zone_id]
        rev_zone = rev_zones[zone_id]
        if len(rev_zone.get("node_coords", [])) <= 1:
            print(f"  [SDVRP-all] Zone {zone_id}: no reverse nodes, skipping")
            continue

        separate_cost = fwd_cost_map.get(zone_id, 0.0) + rev_cost_map.get(zone_id, 0.0)
        zone_result = solve_sdvrp_hybrid(
            zone_id=zone_id,
            fwd_zone=fwd_zone,
            rev_zone=rev_zone,
            num_vehicles=num_vehicles,
            separate_cost_r=separate_cost if separate_cost > 0 else None,
            output_path=out / f"sdvrp_zone{zone_id}_result.json",
        )
        if not zone_result.get("solved"):
            continue

        zone_routes = zone_result.get("routes", [])
        veh_kms: dict[int, float] = {}
        for rec in zone_routes:
            veh_kms[rec["vehicle_id"]] = rec["cumulative_distance_km"]

        all_zone_routes.append(
            {
                "zone_id": zone_id,
                "n_vehicles": zone_result["n_vehicles"],
                "total_dist_km": zone_result["total_dist_km"],
                "routing_cost_R$": zone_result["hybrid_cost_R$"],
                "routes": zone_routes,
            }
        )
        kpi_rows.append(
            {
                "zone_id": zone_id,
                "n_deliveries": zone_result["n_deliveries"],
                "n_pickups": zone_result["n_pickups"],
                "n_vehicles_used": zone_result["n_vehicles"],
                "total_dist_km": zone_result["total_dist_km"],
                "routing_cost_R$": zone_result["hybrid_cost_R$"],
                "separate_cost_R$": zone_result.get("separate_cost_R$"),
                "saving_R$": zone_result.get("saving_R$"),
                "saving_pct": zone_result.get("saving_pct"),
                "max_route_km": max(veh_kms.values()) if veh_kms else 0.0,
                "min_route_km": min(veh_kms.values()) if veh_kms else 0.0,
            }
        )

    routes_path = out / "hybrid_routes.json"
    with open(routes_path, "w") as f:
        json.dump(all_zone_routes, f, indent=2)
    print(
        f"[SDVRP-all] hybrid_routes.json \u2192 {routes_path}  ({len(all_zone_routes)} zones)"
    )

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_path = out / "hybrid_kpi_summary.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"[SDVRP-all] hybrid_kpi_summary.csv \u2192 {kpi_path}")

    if "saving_R$" in kpi_df.columns:
        total_saving = kpi_df["saving_R$"].dropna().sum()
        print(f"[SDVRP-all] Total fleet saving vs separate: R${total_saving:.2f}")

    return kpi_df


# ---------------------------------------------------------------------------
# Z Weight Sensitivity Sweep  (Pritam — Day 5)
# ---------------------------------------------------------------------------


def z_sensitivity_sweep(
    fwd_routes_df: pd.DataFrame,
    rev_routes_df: pd.DataFrame,
    return_probs: pd.Series,
    alpha_grid: list[float] | None = None,
    beta_grid: list[float] | None = None,
    output_path: str | Path = "outputs/z_sensitivity.csv",
) -> pd.DataFrame:
    """
    Grid-search over (alpha, beta) with gamma=delta=(1-alpha-beta)/2.

    Iterates every (alpha, beta) pair in alpha_grid x beta_grid where
    alpha + beta <= 0.9.  For each valid pair:
        gamma = delta = (1 - alpha - beta) / 2

    Parameters
    ----------
    fwd_routes_df : forward routes DataFrame
    rev_routes_df : reverse routes DataFrame
    return_probs  : pd.Series of return probabilities
    alpha_grid    : C_fwd weight values; default [0.1, 0.2, ..., 0.8]
    beta_grid     : C_rev weight values; default same as alpha_grid
    output_path   : path to write CSV

    Returns
    -------
    pd.DataFrame — columns: alpha, beta, gamma, delta, Z,
                             C_fwd, C_rev, T_pen, N_veh, status
    """
    if alpha_grid is None:
        alpha_grid = [round(i * 0.1, 1) for i in range(1, 9)]  # 0.1 .. 0.8
    if beta_grid is None:
        beta_grid = alpha_grid

    combos = [(a, b) for a in alpha_grid for b in beta_grid if round(a + b, 10) <= 0.9]
    print(f"[Z-sweep] Running {len(combos)} (alpha, beta) combinations...")

    rows = []
    for i, (alpha, beta) in enumerate(combos):
        gamma = delta = round((1.0 - alpha - beta) / 2.0, 6)
        prob, vars_dict = build_model(
            fwd_routes_df,
            rev_routes_df,
            return_probs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        solve(prob, time_limit_s=30)
        res = extract_results(prob, vars_dict, fwd_routes_df, rev_routes_df)
        rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "Z": res["Z"],
                "C_fwd": res["C_fwd"],
                "C_rev": res["C_rev"],
                "T_pen": res["T_pen"],
                "N_veh": res["N_veh"],
                "status": res["status"],
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(combos)} done")

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Z-sweep] Saved {len(df)} rows → {output_path}")

    best = df.loc[df["Z"].idxmin()]
    print(
        f"[Z-sweep] Lowest Z={best['Z']:.3f} at "
        f"α={best['alpha']} β={best['beta']} "
        f"γ={best['gamma']} δ={best['delta']}"
    )
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data — just checks the MILP scaffolding runs
    fwd = pd.DataFrame(
        {"vehicle_id": [0, 0, 1, 1], "cumulative_distance_km": [10, 20, 5, 15]}
    )
    rev = pd.DataFrame({"vehicle_id": [0, 0], "cumulative_distance_km": [8, 16]})
    probs = pd.Series([0.3, 0.7, 0.1])
    result = run(fwd, rev, probs)
    print(result)
