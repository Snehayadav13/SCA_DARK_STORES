"""
Module: reverse_vrp.py
Stage:  Reverse Pickup Routing (CVRPTW)

DEPENDS ON:
    data/master_df_v3.parquet   — produced by return_classifier.py
        Required columns: customer_lat, customer_lon, product_weight_g,
                          dark_store_id, return_flag
    data/dark_stores_final.csv  — produced by clustering.py

OUTPUT:
    outputs/reverse_routes.json      — full route detail per zone
    outputs/reverse_routes.csv       — flat route table
    outputs/reverse_kpi_summary.csv  — KPIs per zone
    data/reverse_vrp_nodes.csv       — pickup node list

PUBLIC INTERFACE:
    build_distance_matrix(coords)                       -> np.ndarray
    parse_solution(manager, routing, assignment,
                   node_coords, node_ids,
                   distance_matrix)                     -> (pd.DataFrame, dict)
    build_reverse_vrp_nodes(return_df, dark_stores,
                            max_per_zone, seed)         -> list[dict]
    solve_reverse_cvrptw(zone, num_vehicles)            -> dict
    run_full_pipeline(parquet_path, stores_path,
                      out_dir, data_dir)                -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEHICLE_CAPACITY_G      = 500_000
VEHICLE_SPEED_KMH       = 40
FIXED_COST_PER_ROUTE    = 50.0
VAR_COST_PER_KM         = 1.5
SERVICE_TIME_MIN        = 5
SOLVER_TIME_LIMIT_S     = 30
MAX_CUSTOMERS_PER_ZONE  = 75
NUM_VEHICLES            = 10
RETURN_PROB_THRESHOLD   = 0.30


# ---------------------------------------------------------------------------
# 1. Distance matrix (identical to forward_vrp — both must stay in sync)
# ---------------------------------------------------------------------------

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Haversine distance matrix in metres.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2) — [lat, lon] in degrees

    Returns
    -------
    np.ndarray, shape (n, n) — distances in metres (float64)
    """
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    R   = 6_371_000.0
    n   = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        dlat = lat[i] - lat
        dlon = lon[i] - lon
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2)
        mat[i] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        mat[i, i] = 0
    return mat


# ---------------------------------------------------------------------------
# 2. Solution parser
# ---------------------------------------------------------------------------

def parse_solution(
    manager,
    routing,
    assignment,
    node_coords: np.ndarray,
    node_ids: list[str],
    distance_matrix: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """
    Extract routes from OR-Tools solution into a DataFrame + summary dict.

    Returns
    -------
    routes_df : pd.DataFrame — one row per stop
    summary   : dict — total_distance_km, n_vehicles_used, max/min_route_km
    """
    rows       = []
    total_dist = 0.0
    n_vehicles = 0

    for v in range(routing.vehicles()):
        idx = routing.Start(v)
        if routing.IsEnd(assignment.Value(routing.NextVar(idx))):
            continue
        n_vehicles += 1
        route_dist  = 0.0

        while not routing.IsEnd(idx):
            node      = manager.IndexToNode(idx)
            next_idx  = assignment.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(next_idx)
            route_dist += distance_matrix[node][next_node]
            rows.append({
                "vehicle_id":            v,
                "node_idx":              node,
                "node_id":               node_ids[node],
                "lat":                   float(node_coords[node][0]),
                "lon":                   float(node_coords[node][1]),
                "cumulative_distance_km": round(route_dist / 1000, 3),
            })
            idx = next_idx

        total_dist += route_dist

    routes_df = pd.DataFrame(rows)
    if len(routes_df):
        route_max = routes_df.groupby("vehicle_id")["cumulative_distance_km"].max()
        max_km = round(float(route_max.max()), 2)
        min_km = round(float(route_max.min()), 2)
    else:
        max_km = min_km = 0.0

    summary = {
        "total_distance_km": round(total_dist / 1000, 2),
        "n_vehicles_used":   n_vehicles,
        "max_route_km":      max_km,
        "min_route_km":      min_km,
    }
    return routes_df, summary


# ---------------------------------------------------------------------------
# 3. Reverse node builder
# ---------------------------------------------------------------------------

def build_reverse_vrp_nodes(
    return_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    max_per_zone: int = MAX_CUSTOMERS_PER_ZONE,
    seed: int = 42,
) -> list[dict]:
    """
    For each dark store zone, build a pickup node list for OR-Tools.

    Node 0  = depot (dark store — vehicles start and end here)
    Node 1+ = customers with return_flag=1 to collect from

    Returns
    -------
    list of zone dicts, each containing:
        zone_id, store_lat, store_lon, node_coords, demands,
        time_windows, node_ids, n_pickups
    """
    rng   = np.random.default_rng(seed)
    zones = []

    for _, store in dark_stores.iterrows():
        zid     = int(store["dark_store_id"])
        zone_df = return_df[return_df["dark_store_id"] == zid].copy()

        if zone_df.empty:
            continue

        if len(zone_df) > max_per_zone:
            zone_df = zone_df.sample(
                n=max_per_zone,
                random_state=int(rng.integers(0, 2**31)),
            )

        depot_coords = np.array([[store["lat"], store["lon"]]])
        cust_coords  = zone_df[["customer_lat", "customer_lon"]].values
        node_coords  = np.vstack([depot_coords, cust_coords])

        # Pickup weight = weight of item being returned
        if "product_weight_g" in zone_df.columns:
            cust_demands = (
                zone_df["product_weight_g"]
                .fillna(500)
                .clip(50, 30_000)
                .values
            )
        else:
            cust_demands = np.full(len(zone_df), 500)
        demands = np.concatenate([[0], cust_demands]).astype(int)

        # Time windows — customers available for pickup AM or PM
        tw = [[0, 1440]]
        if "order_purchase_timestamp" in zone_df.columns:
            for ts in zone_df["order_purchase_timestamp"]:
                try:
                    hour = pd.Timestamp(ts).hour
                    tw.append([480, 720] if hour < 12 else [720, 1080])
                except Exception:
                    tw.append([480, 1080])
        else:
            for i in range(len(zone_df)):
                tw.append([480, 720] if i % 2 == 0 else [720, 1080])

        node_ids = ["depot"] + zone_df.index.astype(str).tolist()

        zones.append({
            "zone_id":      zid,
            "store_lat":    store["lat"],
            "store_lon":    store["lon"],
            "node_coords":  node_coords,
            "demands":      demands,
            "time_windows": tw,
            "node_ids":     node_ids,
            "n_pickups":    len(zone_df),
        })

    print(f"[build_reverse_vrp_nodes] Built pickup nodes for {len(zones)} zones")
    for z in zones:
        print(f"  Zone {z['zone_id']:2d}: {z['n_pickups']} pickups | "
              f"total weight = {z['demands'].sum()/1000:.1f} kg")
    return zones


# ---------------------------------------------------------------------------
# 4. Solver
# ---------------------------------------------------------------------------

def solve_reverse_cvrptw(
    zone: dict,
    num_vehicles: int = NUM_VEHICLES,
) -> dict:
    """
    Run OR-Tools CVRPTW for one reverse pickup zone.

    Returns
    -------
    dict with keys: zone_id, solved, routes_df, summary,
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
        dem_cb_idx, 0,
        [VEHICLE_CAPACITY_G] * num_vehicles,
        True, "Capacity",
    )

    def time_cb(i, j):
        node_i = manager.IndexToNode(i)
        travel = int(time_matrix[node_i][manager.IndexToNode(j)])
        service = SERVICE_TIME_MIN if node_i != 0 else 0
        return travel + service
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
        zone["node_coords"], zone["node_ids"],
        distance_matrix=dist_matrix,
    )
    routes_df["zone_id"] = zone["zone_id"]

    total_dist_km = summary["total_distance_km"]
    n_veh         = summary["n_vehicles_used"]
    routing_cost  = FIXED_COST_PER_ROUTE * n_veh + VAR_COST_PER_KM * total_dist_km

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
# 5. Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    parquet_path: str | Path = "data/master_df_v3.parquet",
    stores_path:  str | Path = "data/dark_stores_final.csv",
    out_dir:      str | Path = "outputs",
    data_dir:     str | Path = "data",
) -> dict:
    """
    End-to-end reverse VRP pipeline.

    Writes:
        {data_dir}/reverse_vrp_nodes.csv
        {out_dir}/reverse_routes.json
        {out_dir}/reverse_routes.csv
        {out_dir}/reverse_kpi_summary.csv

    Returns
    -------
    dict with keys: reverse_zones, reverse_results, kpi_df, reverse_routes_df
    """
    print("=" * 60)
    print("  REVERSE VRP PIPELINE")
    print("=" * 60)

    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading data...")
    master_df   = pd.read_parquet(parquet_path)
    dark_stores = pd.read_csv(stores_path)
    return_df   = master_df[master_df["return_flag"] == 1].copy()
    print(f"       {len(master_df):,} orders | {len(return_df):,} returns | "
          f"{len(dark_stores)} zones")

    print("\n[2/4] Building reverse VRP nodes...")
    reverse_zones = build_reverse_vrp_nodes(return_df, dark_stores)

    # Save reverse_vrp_nodes.csv
    rows = []
    for z in reverse_zones:
        for i, (coords, demand, tw, nid) in enumerate(zip(
                z["node_coords"], z["demands"],
                z["time_windows"], z["node_ids"])):
            rows.append({
                "zone_id":  z["zone_id"], "node_idx": i,
                "node_id":  nid,
                "lat":      round(float(coords[0]), 6),
                "lon":      round(float(coords[1]), 6),
                "demand_g": int(demand),
                "tw_open":  tw[0], "tw_close": tw[1],
                "is_depot": int(i == 0),
            })
    pd.DataFrame(rows).to_csv(data_dir / "reverse_vrp_nodes.csv", index=False)
    print(f"       reverse_vrp_nodes.csv saved ({len(rows)} rows)")

    print(f"\n[3/4] Solving reverse CVRPTW ({SOLVER_TIME_LIMIT_S}s per zone)...")
    reverse_results = {}
    for z in reverse_zones:
        reverse_results[z["zone_id"]] = solve_reverse_cvrptw(z)

    n_solved = sum(r["solved"] for r in reverse_results.values())
    print(f"\n       {n_solved}/{len(reverse_zones)} zones solved")

    print("\n[4/4] Saving outputs...")
    all_dfs = [r["routes_df"] for r in reverse_results.values()
               if r["solved"] and "routes_df" in r]
    reverse_routes_df = pd.concat(all_dfs, ignore_index=True)
    reverse_routes_df.to_csv(out_dir / "reverse_routes.csv", index=False)

    rev_json = []
    kpi_rows = []
    for zid, r in reverse_results.items():
        if not r["solved"]:
            continue
        rev_json.append({
            "zone_id":         zid,
            "n_vehicles":      r["n_vehicles"],
            "total_dist_km":   r["total_dist_km"],
            "routing_cost_R$": r["routing_cost_R$"],
            "routes":          r["routes_df"].to_dict(orient="records"),
        })
        zone = next(z for z in reverse_zones if z["zone_id"] == zid)
        kpi_rows.append({
            "zone_id":         zid,
            "n_pickups":       zone["n_pickups"],
            "n_vehicles_used": r["n_vehicles"],
            "total_dist_km":   round(r["total_dist_km"], 2),
            "routing_cost_R$": round(r["routing_cost_R$"], 2),
            "max_route_km":    r["summary"]["max_route_km"],
            "min_route_km":    r["summary"]["min_route_km"],
        })

    with open(out_dir / "reverse_routes.json", "w") as f:
        json.dump(rev_json, f, indent=2)

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(out_dir / "reverse_kpi_summary.csv", index=False)

    print(f"       reverse_routes.json saved")
    print(f"       reverse_routes.csv  saved ({len(reverse_routes_df)} rows)")
    print(f"       reverse_kpi_summary.csv saved")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Zones solved  : {n_solved}/{len(reverse_zones)}")
    print(f"  Total dist    : {kpi_df['total_dist_km'].sum():.1f} km")
    print(f"  Total cost    : R${kpi_df['routing_cost_R$'].sum():.0f}")
    print(f"  Total vehicles: {kpi_df['n_vehicles_used'].sum()}")
    print("=" * 60)

    return {
        "reverse_zones":    reverse_zones,
        "reverse_results":  reverse_results,
        "kpi_df":           kpi_df,
        "reverse_routes_df": reverse_routes_df,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_pipeline(
        parquet_path="data/master_df_v3.parquet",
        stores_path="data/dark_stores_final.csv",
        out_dir="outputs",
        data_dir="data",
    )
    