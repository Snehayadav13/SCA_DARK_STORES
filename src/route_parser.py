"""
Module: route_parser.py
Owner:  Pranav builds the OR-Tools models; Pritam's pipeline CONSUMES this output.

INPUT  (produced by Pranav's VRP solver):
    routing_model   : ortools.constraint_solver.RoutingModel
    routing_manager : ortools.constraint_solver.RoutingIndexManager
    assignment      : ortools.constraint_solver.Assignment
    node_coords     : np.ndarray, shape (n_nodes, 2)  — (lat, lon) per node index
    node_ids        : list[str]  — order_id or stop label per node index

OUTPUT:
    routes_df : pd.DataFrame
        Columns:
            vehicle_id (int), stop_seq (int), node_idx (int),
            node_id (str), lat (float), lon (float),
            cumulative_distance_km (float), load_after_stop (float)
    summary : dict
        total_distance_km, n_vehicles_used, max_route_km, min_route_km

INTERFACE:
    parse_solution(manager, routing, assignment, node_coords, node_ids)
        -> (routes_df, summary)
    extract_route(manager, routing, assignment, vehicle)
        -> list[int]   # ordered list of node indices for one vehicle
    route_to_df(manager, routing, assignment, node_coords, node_ids, vehicle)
        -> pd.DataFrame  # per-stop details for one vehicle
    save_routes(routes_df, path)
        -> None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core parsing helpers
# ---------------------------------------------------------------------------

def extract_route(manager, routing, assignment, vehicle: int) -> list[int]:
    """
    Walk the assignment for vehicle and return ordered node indices
    (excluding the depot at start and end).
    """
    nodes: list[int] = []
    index = routing.Start(vehicle)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        nodes.append(node)
        index = assignment.Value(routing.NextVar(index))
    return nodes


def route_to_df(
    manager,
    routing,
    assignment,
    node_coords: np.ndarray,
    node_ids: list[str],
    vehicle: int,
    distance_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a per-stop DataFrame for one vehicle route.

    Columns: stop_seq, node_idx, node_id, lat, lon, cumulative_distance_km, load_after_stop.
    Note: load_after_stop requires a LoadDimension to be set in routing; set to -1 if absent.
    """
    route_nodes = extract_route(manager, routing, assignment, vehicle)
    rows = []
    cum_dist = 0.0

    for seq, node in enumerate(route_nodes):
        lat, lon = float(node_coords[node, 0]), float(node_coords[node, 1])

        # Cumulative distance via distance matrix (if provided)
        if distance_matrix is not None and seq > 0:
            prev_node = route_nodes[seq - 1]
            cum_dist += distance_matrix[prev_node, node] / 1000.0  # scaled ×1000 → km

        # Load after stop (if LoadDimension exists)
        load = -1
        try:
            dim = routing.GetDimensionOrDie("Load")
            index = routing.Start(vehicle)
            for _ in range(seq + 1):
                index = assignment.Value(routing.NextVar(index))
            load = assignment.Min(dim.CumulVar(index))
        except Exception:
            pass

        rows.append({
            "stop_seq": seq,
            "node_idx": node,
            "node_id": node_ids[node] if node < len(node_ids) else str(node),
            "lat": lat,
            "lon": lon,
            "cumulative_distance_km": round(cum_dist, 3),
            "load_after_stop": load,
        })
    return pd.DataFrame(rows)


def parse_solution(
    manager,
    routing,
    assignment,
    node_coords: np.ndarray,
    node_ids: list[str],
    distance_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Parse OR-Tools solution into routes_df and a summary dict.

    Parameters
    ----------
    manager         : RoutingIndexManager
    routing         : RoutingModel
    assignment      : Assignment (solution)
    node_coords     : (n, 2) array of (lat, lon)
    node_ids        : list of stop identifiers parallel to node_coords
    distance_matrix : optional (n, n) int array, scaled ×1000 km

    Returns
    -------
    (routes_df, summary)
    """
    all_routes: list[pd.DataFrame] = []

    for v in range(routing.vehicles()):
        if routing.IsVehicleUsed(assignment, v):
            df = route_to_df(manager, routing, assignment, node_coords, node_ids, v, distance_matrix)
            df.insert(0, "vehicle_id", v)
            all_routes.append(df)

    routes_df = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()

    if routes_df.empty:
        return routes_df, {"total_distance_km": 0, "n_vehicles_used": 0}

    route_totals = (
        routes_df.groupby("vehicle_id")["cumulative_distance_km"].max()
    )
    summary = {
        "total_distance_km": round(route_totals.sum(), 2),
        "n_vehicles_used": len(all_routes),
        "max_route_km": round(route_totals.max(), 2),
        "min_route_km": round(route_totals.min(), 2),
    }
    return routes_df, summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_routes(routes_df: pd.DataFrame, path: str | Path = "outputs/routes.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    routes_df.to_csv(path, index=False)
    print(f"[INFO] Routes saved → {path}  ({len(routes_df)} stops)")
