#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# --- robust repo root ---
REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))

DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- constants ---
VEHICLE_CAPACITY_G      = 500_000
VEHICLE_SPEED_KMH       = 40
FIXED_COST_PER_ROUTE    = 50
VAR_COST_PER_KM         = 1.5
SERVICE_TIME_MIN        = 5
SOLVER_TIME_LIMIT_S     = 30
MAX_CUSTOMERS_PER_ZONE  = 75
RETURN_PROB_THRESHOLD   = 0.30

# --- inline helpers (no src/ dependency) ---
def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    R = 6_371_000
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dlat = lat[i] - lat[j]
                dlon = lon[i] - lon[j]
                a = (np.sin(dlat/2)**2
                     + np.cos(lat[i]) * np.cos(lat[j]) * np.sin(dlon/2)**2)
                mat[i, j] = 2 * R * np.arcsin(np.sqrt(a))
    return mat

def parse_solution(manager, routing, assignment, node_coords, node_ids, distance_matrix):
    routes = []
    total_dist = 0
    n_vehicles = 0
    for v in range(routing.vehicles()):
        idx = routing.Start(v)
        if routing.IsEnd(assignment.Value(routing.NextVar(idx))):
            continue
        n_vehicles += 1
        route_dist = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            next_idx = assignment.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(next_idx)
            route_dist += distance_matrix[node][next_node]
            routes.append({
                "vehicle_id": v,
                "node_idx": node,
                "node_id": node_ids[node],
                "lat": float(node_coords[node][0]),
                "lon": float(node_coords[node][1]),
                "cumulative_distance_km": round(route_dist / 1000, 3),
            })
            idx = next_idx
        total_dist += route_dist
    routes_df = pd.DataFrame(routes)
    route_distances = routes_df.groupby("vehicle_id")["cumulative_distance_km"].max() if len(routes_df) else pd.Series(dtype=float)
    summary = {
        "total_distance_km": round(total_dist / 1000, 2),
        "n_vehicles_used":   n_vehicles,
        "max_route_km":      round(route_distances.max(), 2) if len(route_distances) else 0,
        "min_route_km":      round(route_distances.min(), 2) if len(route_distances) else 0,
    }
    return routes_df, summary

print("Setup complete. Repo root:", REPO_ROOT)


# In[13]:


master_df   = pd.read_parquet(DATA_DIR / "master_df_v2.parquet")
dark_stores = pd.read_csv(DATA_DIR / "dark_stores_final.csv")

# Use real return_prob if classifier has run, else use stub
if "return_prob" not in master_df.columns:
    print("WARNING: return_prob not found — using 5% random stub.")
    rng = np.random.default_rng(42)
    master_df["return_prob"] = rng.beta(0.5, 9.5, size=len(master_df)).clip(0, 1)
else:
    print("Using real return_prob from classifier.")

master_df["is_return_node"] = (master_df["return_prob"] > RETURN_PROB_THRESHOLD).astype(int)
return_df = master_df[master_df["is_return_node"] == 1].copy()

print(f"Total orders     : {len(master_df):,}")
print(f"Return nodes     : {len(return_df):,} ({len(return_df)/len(master_df)*100:.1f}%)")
print(f"Dark stores      : {len(dark_stores)}")


# In[14]:


def build_reverse_vrp_nodes(return_df, dark_stores, max_per_zone=MAX_CUSTOMERS_PER_ZONE, seed=42):
    """
    For each dark store zone, build reverse pickup node list:
      node 0  = depot (dark store)
      node 1+ = customers to pick up returns from
    """
    rng = np.random.default_rng(seed)
    zones = []

    for _, store in dark_stores.iterrows():
        zid     = int(store["dark_store_id"])
        zone_df = return_df[return_df["dark_store_id"] == zid].copy()

        if len(zone_df) == 0:
            continue

        if len(zone_df) > max_per_zone:
            zone_df = zone_df.sample(n=max_per_zone,
                                     random_state=int(rng.integers(0, 2**31)))

        depot_coords = np.array([[store["lat"], store["lon"]]])
        cust_coords  = zone_df[["customer_lat", "customer_lon"]].values
        node_coords  = np.vstack([depot_coords, cust_coords])

        if "product_weight_g" in zone_df.columns:
            demands = np.concatenate([[0],
                zone_df["product_weight_g"].fillna(500).clip(50, 30_000).values])
        else:
            demands = np.concatenate([[0], np.full(len(zone_df), 500)])

        # Pickup time windows — customers available AM or PM
        tw = [[0, 1440]]
        for i in range(len(zone_df)):
            tw.append([480, 720] if i % 2 == 0 else [720, 1080])

        node_ids = ["depot"] + zone_df.index.astype(str).tolist()

        zones.append({
            "zone_id":      zid,
            "store_lat":    store["lat"],
            "store_lon":    store["lon"],
            "node_coords":  node_coords,
            "demands":      demands.astype(int),
            "time_windows": tw,
            "node_ids":     node_ids,
            "n_pickups":    len(zone_df),
        })

    print(f"Built reverse VRP nodes for {len(zones)} zones")
    for z in zones:
        print(f"  Zone {z['zone_id']:2d}: {z['n_pickups']} pickups | "
              f"total weight = {z['demands'].sum()/1000:.1f} kg")
    return zones

reverse_zones = build_reverse_vrp_nodes(return_df, dark_stores)


# In[15]:


rows = []
for z in reverse_zones:
    for i, (coords, demand, tw, nid) in enumerate(zip(
            z["node_coords"], z["demands"], z["time_windows"], z["node_ids"])):
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

reverse_vrp_nodes = pd.DataFrame(rows)
reverse_vrp_nodes.to_csv(DATA_DIR / "reverse_vrp_nodes.csv", index=False)
print(f"Saved: {DATA_DIR / 'reverse_vrp_nodes.csv'}  ({len(reverse_vrp_nodes)} rows)")
reverse_vrp_nodes.head(6)


# In[16]:


def solve_reverse_cvrptw(zone: dict, num_vehicles: int = 10) -> dict:
    n       = len(zone["node_coords"])
    demands = zone["demands"].tolist()
    tw      = zone["time_windows"]

    dist_matrix  = build_distance_matrix(zone["node_coords"])
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60
    time_matrix  = np.rint(dist_matrix / speed_m_per_min).astype(int)

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
        True, "Capacity")

    def time_cb(i, j):
        node_i  = manager.IndexToNode(i)
        travel  = int(time_matrix[node_i][manager.IndexToNode(j)])
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
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = SOLVER_TIME_LIMIT_S

    assignment = routing.SolveWithParameters(params)

    if assignment is None:
        print(f"  Zone {zone['zone_id']}: NO SOLUTION FOUND")
        return {"zone_id": zone["zone_id"], "solved": False}

    routes_df, summary = parse_solution(
        manager, routing, assignment,
        zone["node_coords"], zone["node_ids"],
        distance_matrix=dist_matrix
    )
    routes_df["zone_id"] = zone["zone_id"]

    total_dist_km = summary["total_distance_km"]
    n_vehicles    = summary["n_vehicles_used"]
    routing_cost  = FIXED_COST_PER_ROUTE * n_vehicles + VAR_COST_PER_KM * total_dist_km

    print(f"  Zone {zone['zone_id']:2d}: {n_vehicles} vehicles | "
          f"{total_dist_km:.1f} km | R${routing_cost:.0f}")

    return {
        "zone_id":         zone["zone_id"],
        "solved":          True,
        "routes_df":       routes_df,
        "summary":         summary,
        "total_dist_km":   total_dist_km,
        "n_vehicles":      n_vehicles,
        "routing_cost_R$": routing_cost,
        "dist_matrix":     dist_matrix,
    }

print("Reverse solver function defined.")


# In[17]:


print("Running Reverse VRP (CVRPTW) — pickup routes...")
print(f"Time limit: {SOLVER_TIME_LIMIT_S}s per zone | "
      f"Vehicle capacity: {VEHICLE_CAPACITY_G/1000:.0f} kg\n")

reverse_results = {}
for z in reverse_zones:
    result = solve_reverse_cvrptw(z)
    reverse_results[z["zone_id"]] = result

n_solved = sum(r["solved"] for r in reverse_results.values())
print(f"\nCompleted: {n_solved}/{len(reverse_zones)} zones solved.")


# In[18]:


all_route_dfs = [r["routes_df"] for r in reverse_results.values()
                 if r["solved"] and "routes_df" in r]
reverse_routes_df = pd.concat(all_route_dfs, ignore_index=True)
reverse_routes_df.to_csv(OUTPUT_DIR / "reverse_routes.csv", index=False)

json_out = []
for zid, r in reverse_results.items():
    if not r["solved"]: continue
    json_out.append({
        "zone_id":         zid,
        "n_vehicles":      r["n_vehicles"],
        "total_dist_km":   r["total_dist_km"],
        "routing_cost_R$": r["routing_cost_R$"],
        "routes":          r["routes_df"].to_dict(orient="records"),
    })
with open(OUTPUT_DIR / "reverse_routes.json", "w") as f:
    json.dump(json_out, f, indent=2)

kpi_rows = []
for zid, r in reverse_results.items():
    if not r["solved"]: continue
    kpi_rows.append({
        "zone_id":         zid,
        "n_pickups":       next(z["n_pickups"] for z in reverse_zones if z["zone_id"] == zid),
        "n_vehicles_used": r["n_vehicles"],
        "total_dist_km":   round(r["total_dist_km"], 2),
        "routing_cost_R$": round(r["routing_cost_R$"], 2),
        "max_route_km":    r["summary"]["max_route_km"],
        "min_route_km":    r["summary"]["min_route_km"],
    })
kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv(OUTPUT_DIR / "reverse_kpi_summary.csv", index=False)

print("Saved:")
print(f"  {OUTPUT_DIR / 'reverse_routes.csv'}")
print(f"  {OUTPUT_DIR / 'reverse_routes.json'}")
print(f"  {OUTPUT_DIR / 'reverse_kpi_summary.csv'}")
print()
print(kpi_df.to_string(index=False))


# In[20]:


fwd_kpi = pd.read_csv(OUTPUT_DIR / "forward_kpi_summary.csv")
rev_kpi = kpi_df.copy()

merged = fwd_kpi.merge(rev_kpi, on="zone_id", suffixes=("_fwd", "_rev"))

print("=" * 60)
print("FORWARD vs REVERSE VRP — ZONE-LEVEL COMPARISON")
print("=" * 60)
print(merged[["zone_id",
              "n_customers", "n_pickups",
              "total_dist_km_fwd", "total_dist_km_rev",
              "routing_cost_R$_fwd", "routing_cost_R$_rev"]].to_string(index=False))

print("\n" + "=" * 60)
print("TOTALS")
print("=" * 60)
print(f"  Forward  — dist: {fwd_kpi['total_dist_km'].sum():.1f} km | "
      f"cost: R${fwd_kpi['routing_cost_R$'].sum():.0f} | "
      f"vehicles: {fwd_kpi['n_vehicles_used'].sum()}")
print(f"  Reverse  — dist: {rev_kpi['total_dist_km'].sum():.1f} km | "
      f"cost: R${rev_kpi['routing_cost_R$'].sum():.0f} | "
      f"vehicles: {rev_kpi['n_vehicles_used'].sum()}")
total_cost = fwd_kpi['routing_cost_R$'].sum() + rev_kpi['routing_cost_R$'].sum()
print(f"\n  Combined total cost: R${total_cost:.0f}")
print("=" * 60)


# In[ ]:




