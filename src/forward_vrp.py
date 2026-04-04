"""
Module: forward_vrp.py
Stage:  Forward Delivery Routing (CVRPTW)

DEPENDS ON:
    data/master_df_v3.parquet  — produced by return_classifier.py
    data/dark_stores_final.csv — produced by clustering.py
    src/route_parser.py        — shared VRP utilities

OUTPUT:
    data/vrp_nodes.csv              — node list for all zones
    outputs/forward_routes.csv      — flat stop-level route table
    outputs/forward_routes.json     — full route detail per zone
    outputs/forward_kpi_summary.csv — KPIs per zone

PUBLIC INTERFACE:
    solve_cvrptw(zone, num_vehicles) -> dict
    run_full_pipeline(parquet_path, stores_path,
                      out_dir, data_dir)     -> dict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

import sys
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_parser import (
    VEHICLE_CAPACITY_G,
    VEHICLE_SPEED_KMH,
    SOLVER_TIME_LIMIT_S,
    SERVICE_TIME_MIN,
    NUM_VEHICLES,
    build_distance_matrix,
    build_vrp_nodes,
    parse_solution,
    compute_routing_cost,
    nodes_to_csv,
    save_routes,
)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_cvrptw(
    zone: dict,
    num_vehicles: int = NUM_VEHICLES,
) -> dict:
    """
    Run OR-Tools CVRPTW for one forward delivery zone.

    Returns
    -------
    dict: zone_id, solved, routes_df, summary,
          total_dist_km, n_vehicles, routing_cost_R$, dist_matrix
    """
    n       = len(zone["node_coords"])
    demands = zone["demands"].tolist()
    tw      = zone["time_windows"]

    dist_matrix     = build_distance_matrix(zone["node_coords"])
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60
    time_matrix     = np.rint(dist_matrix / speed_m_per_min).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)])
    dist_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    def demand_cb(i):
        return int(demands[manager.IndexToNode(i)])
    dem_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_cb_idx, 0, [VEHICLE_CAPACITY_G] * num_vehicles, True, "Capacity",
    )

    def time_cb(i, j):
        node_i = manager.IndexToNode(i)
        return int(time_matrix[node_i][manager.IndexToNode(j)]) + (
            SERVICE_TIME_MIN if node_i != 0 else 0
        )
    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, 60, 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node_idx, (open_t, close_t) in enumerate(tw):
        time_dim.CumulVar(manager.NodeToIndex(node_idx)).SetRange(open_t, close_t)

    penalty = 100_000
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = SOLVER_TIME_LIMIT_S

    assignment = routing.SolveWithParameters(params)

    if assignment is None:
        print(f"  Zone {zone['zone_id']}: NO SOLUTION FOUND")
        return {"zone_id": zone["zone_id"], "solved": False}

    routes_df, summary = parse_solution(
        manager, routing, assignment,
        zone["node_coords"], zone["node_ids"], dist_matrix,
    )
    routes_df["zone_id"] = zone["zone_id"]

    total_dist_km = summary["total_distance_km"]
    n_veh         = summary["n_vehicles_used"]
    routing_cost  = compute_routing_cost(n_veh, total_dist_km)

    print(f"  Zone {zone['zone_id']:2d}: {n_veh} vehicles | "
          f"{total_dist_km:.1f} km | R${routing_cost:.0f}")

    return {
        "zone_id":         zone["zone_id"],
        "solved":          True,
        "routes_df":       routes_df,
        "summary":         summary,
        "total_dist_km":   total_dist_km,
        "n_vehicles":      n_veh,
        "routing_cost_R$": routing_cost,
        "dist_matrix":     dist_matrix,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    parquet_path: str | Path = "data/master_df_v3.parquet",
    stores_path:  str | Path = "data/dark_stores_final.csv",
    out_dir:      str | Path = "outputs",
    data_dir:     str | Path = "data",
) -> dict:
    """
    End-to-end forward VRP pipeline.

    Writes:
        data/vrp_nodes.csv
        outputs/forward_routes.csv
        outputs/forward_routes.json
        outputs/forward_kpi_summary.csv

    Returns
    -------
    dict: zones, zone_results, kpi_df, forward_routes_df
    """
    print("=" * 60)
    print("  FORWARD VRP PIPELINE")
    print("=" * 60)

    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)

    print("\n[1/4] Loading data...")
    master_df   = pd.read_parquet(parquet_path)
    dark_stores = pd.read_csv(stores_path)
    print(f"       {len(master_df):,} orders | {len(dark_stores)} zones")

    print("\n[2/4] Building VRP nodes...")
    zones = build_vrp_nodes(master_df, dark_stores)
    nodes_to_csv(zones, data_dir, "vrp_nodes.csv")

    print(f"\n[3/4] Solving CVRPTW ({SOLVER_TIME_LIMIT_S}s per zone)...")
    zone_results = {z["zone_id"]: solve_cvrptw(z) for z in zones}
    n_solved = sum(r["solved"] for r in zone_results.values())
    print(f"\n       {n_solved}/{len(zones)} zones solved")

    print("\n[4/4] Saving outputs...")
    routes_df, kpi_df = save_routes(zone_results, zones, out_dir, prefix="forward")

    print("\n" + "=" * 60)
    print("  FORWARD VRP COMPLETE")
    print(f"  Zones solved  : {n_solved}/{len(zones)}")
    print(f"  Total dist    : {kpi_df['total_dist_km'].sum():.1f} km")
    print(f"  Total cost    : R${kpi_df['routing_cost_R$'].sum():.0f}")
    print(f"  Total vehicles: {kpi_df['n_vehicles_used'].sum()}")
    print("=" * 60)

    return {
        "zones":             zones,
        "zone_results":      zone_results,
        "kpi_df":            kpi_df,
        "forward_routes_df": routes_df,
    }


if __name__ == "__main__":
    run_full_pipeline()
