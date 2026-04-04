"""
Module: route_parser.py
Stage:  Shared VRP Utilities

All shared logic for forward and reverse VRP lives here.
Both forward_vrp.py and reverse_vrp.py import from this module.
Notebooks also import from here for interactive exploration.

PUBLIC INTERFACE:
    build_distance_matrix(coords)                   -> np.ndarray
    build_vrp_nodes(master_df, dark_stores,
                    max_per_zone, seed)             -> list[dict]
    build_reverse_vrp_nodes(return_df, dark_stores,
                            max_per_zone, seed)     -> list[dict]
    parse_solution(manager, routing, assignment,
                   node_coords, node_ids,
                   distance_matrix)                 -> tuple[pd.DataFrame, dict]
    compute_routing_cost(n_vehicles,
                         total_dist_km)             -> float
    nodes_to_csv(zones, data_dir, filename)         -> None
    save_routes(zone_results, zones,
                out_dir, prefix)                    -> tuple[pd.DataFrame, pd.DataFrame]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — single source of truth for all VRP modules
# ---------------------------------------------------------------------------

VEHICLE_CAPACITY_G     = 500_000   # 500 kg in grams
VEHICLE_SPEED_KMH      = 40
FIXED_COST_PER_ROUTE   = 50.0      # R$
VAR_COST_PER_KM        = 1.5       # R$
SERVICE_TIME_MIN       = 5
SOLVER_TIME_LIMIT_S    = 30
MAX_CUSTOMERS_PER_ZONE = 75
NUM_VEHICLES           = 10
RETURN_PROB_THRESHOLD  = 0.30


# ---------------------------------------------------------------------------
# 1. Distance matrix
# ---------------------------------------------------------------------------

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Haversine distance matrix in metres (vectorised, float64).

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2) — [lat, lon] in degrees

    Returns
    -------
    np.ndarray, shape (n, n) — distances in metres
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
# 2. Forward VRP node builder
# ---------------------------------------------------------------------------

def build_vrp_nodes(
    master_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    max_per_zone: int = MAX_CUSTOMERS_PER_ZONE,
    seed: int = 42,
) -> list[dict]:
    """
    Build OR-Tools node list for forward delivery routing (one list per zone).

    Node 0  = depot (dark store centroid)
    Node 1+ = sampled customers in that zone

    Time windows are derived from order_purchase_timestamp if present,
    else alternating AM (480-720 min) / PM (720-1080 min).

    Returns
    -------
    list of zone dicts, each with keys:
        zone_id, store_lat, store_lon, node_coords (np.ndarray),
        demands (np.ndarray), time_windows (list), node_ids (list),
        n_customers (int)
    """
    rng   = np.random.default_rng(seed)
    zones = []

    for _, store in dark_stores.iterrows():
        zid     = int(store["dark_store_id"])
        zone_df = master_df[master_df["dark_store_id"] == zid].copy()

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

        if "product_weight_g" in zone_df.columns:
            cust_demands = (
                zone_df["product_weight_g"]
                .fillna(500).clip(50, 30_000).values
            )
        else:
            cust_demands = np.full(len(zone_df), 500)
        demands = np.concatenate([[0], cust_demands]).astype(int)

        tw = [[0, 1440]]  # depot: all day
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
            "n_customers":  len(zone_df),
        })

    print(f"[build_vrp_nodes] {len(zones)} zones")
    for z in zones:
        print(f"  Zone {z['zone_id']:2d}: {z['n_customers']} customers | "
              f"demand = {z['demands'].sum()/1000:.1f} kg")
    return zones


# ---------------------------------------------------------------------------
# 3. Reverse VRP node builder
# ---------------------------------------------------------------------------

def build_reverse_vrp_nodes(
    return_df: pd.DataFrame,
    dark_stores: pd.DataFrame,
    max_per_zone: int = MAX_CUSTOMERS_PER_ZONE,
    seed: int = 42,
) -> list[dict]:
    """
    Build OR-Tools node list for reverse pickup routing (one list per zone).

    Node 0  = depot (dark store — vehicles start and end here)
    Node 1+ = customers with return_flag=1 to collect returns from

    Returns
    -------
    list of zone dicts, each with keys:
        zone_id, store_lat, store_lon, node_coords (np.ndarray),
        demands (np.ndarray), time_windows (list), node_ids (list),
        n_pickups (int)
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

        if "product_weight_g" in zone_df.columns:
            cust_demands = (
                zone_df["product_weight_g"]
                .fillna(500).clip(50, 30_000).values
            )
        else:
            cust_demands = np.full(len(zone_df), 500)
        demands = np.concatenate([[0], cust_demands]).astype(int)

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

    print(f"[build_reverse_vrp_nodes] {len(zones)} zones")
    for z in zones:
        print(f"  Zone {z['zone_id']:2d}: {z['n_pickups']} pickups | "
              f"weight = {z['demands'].sum()/1000:.1f} kg")
    return zones


# ---------------------------------------------------------------------------
# 4. Solution parser
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
    Extract OR-Tools solution into a flat DataFrame + summary dict.

    Parameters
    ----------
    manager         : RoutingIndexManager
    routing         : RoutingModel
    assignment      : OR-Tools Assignment (solution)
    node_coords     : np.ndarray, shape (n, 2) — [lat, lon]
    node_ids        : list[str] — label per node index
    distance_matrix : np.ndarray, shape (n, n) — metres

    Returns
    -------
    routes_df : pd.DataFrame — one row per stop visited
        Columns: vehicle_id, node_idx, node_id, lat, lon,
                 cumulative_distance_km
    summary   : dict
        total_distance_km, n_vehicles_used, max_route_km, min_route_km
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
                "vehicle_id":             v,
                "node_idx":               node,
                "node_id":                node_ids[node],
                "lat":                    float(node_coords[node][0]),
                "lon":                    float(node_coords[node][1]),
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
# 5. Cost calculator
# ---------------------------------------------------------------------------

def compute_routing_cost(n_vehicles: int, total_dist_km: float) -> float:
    """
    R$ cost = fixed cost per vehicle + variable cost per km.
    Used identically in forward and reverse VRP.
    """
    return FIXED_COST_PER_ROUTE * n_vehicles + VAR_COST_PER_KM * total_dist_km


# ---------------------------------------------------------------------------
# 6. Node CSV writer
# ---------------------------------------------------------------------------

def nodes_to_csv(
    zones: list[dict],
    data_dir: str | Path,
    filename: str,
) -> None:
    """
    Flatten a zone list into a CSV of nodes.
    Used by both forward (vrp_nodes.csv) and reverse (reverse_vrp_nodes.csv).
    """
    rows = []
    for z in zones:
        size_key = "n_customers" if "n_customers" in z else "n_pickups"
        for i, (coords, demand, tw, nid) in enumerate(zip(
                z["node_coords"], z["demands"],
                z["time_windows"], z["node_ids"])):
            rows.append({
                "zone_id":  z["zone_id"],
                "node_idx": i,
                "node_id":  nid,
                "lat":      round(float(coords[0]), 6),
                "lon":      round(float(coords[1]), 6),
                "demand_g": int(demand),
                "tw_open":  tw[0],
                "tw_close": tw[1],
                "is_depot": int(i == 0),
            })
    out = Path(data_dir) / filename
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[nodes_to_csv] {out} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 7. Route + KPI saver
# ---------------------------------------------------------------------------

def save_routes(
    zone_results: dict,
    zones: list[dict],
    out_dir: str | Path,
    prefix: str,   # "forward" or "reverse"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save route outputs for one VRP run (forward or reverse).

    Writes:
        {out_dir}/{prefix}_routes.csv
        {out_dir}/{prefix}_routes.json
        {out_dir}/{prefix}_kpi_summary.csv

    Parameters
    ----------
    zone_results : dict  — {zone_id: result_dict} from solver
    zones        : list  — zone dicts from node builder
    out_dir      : str | Path
    prefix       : "forward" or "reverse"

    Returns
    -------
    (routes_df, kpi_df)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_reverse  = prefix == "reverse"
    count_key   = "n_pickups" if is_reverse else "n_customers"
    zone_map    = {z["zone_id"]: z for z in zones}

    all_dfs  = []
    json_out = []
    kpi_rows = []

    for zid, r in zone_results.items():
        if not r.get("solved"):
            continue

        df = r["routes_df"].copy()
        df["zone_id"] = zid
        all_dfs.append(df)

        json_out.append({
            "zone_id":         zid,
            "n_vehicles":      r["n_vehicles"],
            "total_dist_km":   r["total_dist_km"],
            "routing_cost_R$": r["routing_cost_R$"],
            "routes":          df.to_dict(orient="records"),
        })

        z = zone_map[zid]
        kpi_rows.append({
            "zone_id":         zid,
            count_key:         z[count_key],
            "n_vehicles_used": r["n_vehicles"],
            "total_dist_km":   round(r["total_dist_km"], 2),
            "routing_cost_R$": round(r["routing_cost_R$"], 2),
            "max_route_km":    r["summary"]["max_route_km"],
            "min_route_km":    r["summary"]["min_route_km"],
        })

    routes_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    kpi_df    = pd.DataFrame(kpi_rows)

    routes_df.to_csv(out_dir / f"{prefix}_routes.csv", index=False)
    kpi_df.to_csv(out_dir / f"{prefix}_kpi_summary.csv", index=False)
    with open(out_dir / f"{prefix}_routes.json", "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"[save_routes] {prefix}_routes.csv     ({len(routes_df)} rows)")
    print(f"[save_routes] {prefix}_routes.json    ({len(json_out)} zones)")
    print(f"[save_routes] {prefix}_kpi_summary.csv ({len(kpi_df)} zones)")

    return routes_df, kpi_df
    