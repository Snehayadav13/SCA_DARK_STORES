#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# --- robust repo root detection ---
REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent

# --- fix imports ---
sys.path.insert(0, str(REPO_ROOT))

# --- paths ---
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- constants ---
VEHICLE_CAPACITY_G  = 500_000   # 500 kg in grams
VEHICLE_SPEED_KMH   = 40
FIXED_COST_PER_ROUTE = 50       # R$
VAR_COST_PER_KM      = 1.5      # R$
SERVICE_TIME_MIN     = 5
SOLVER_TIME_LIMIT_S  = 30
MAX_CUSTOMERS_PER_ZONE = 75     # tractability cap
RETURN_PROB_THRESHOLD   = 0.30   # orders above 30% return probability get a return node

#from src.haversine_matrix import build_distance_matrix
#from src.route_parser import parse_solution, save_routes

#print("Imports OK:", build_distance_matrix)

print("Setup complete.")
print("Repo root:", REPO_ROOT)

from scipy.spatial.distance import cdist

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Haversine distance matrix in metres."""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    R = 6_371_000
    def hav(u, v):
        dlat = u[0] - v[0]; dlon = u[1] - v[1]
        a = np.sin(dlat/2)**2 + np.cos(u[0])*np.cos(v[0])*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = hav(
                    [lat[i], lon[i]],
                    [lat[j], lon[j]]
                )
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
        route_nodes = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
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

    import pandas as pd
    routes_df = pd.DataFrame(routes)
    route_distances = routes_df.groupby("vehicle_id")["cumulative_distance_km"].max()
    summary = {
        "total_distance_km": round(total_dist / 1000, 2),
        "n_vehicles_used": n_vehicles,
        "max_route_km": round(route_distances.max(), 2) if len(route_distances) else 0,
        "min_route_km": round(route_distances.min(), 2) if len(route_distances) else 0,
    }
    return routes_df, summary

print("✅ build_distance_matrix and parse_solution defined inline.")


# In[2]:


master_df   = pd.read_parquet(DATA_DIR / "master_df_v2.parquet")
dark_stores = pd.read_csv(DATA_DIR / "dark_stores_final.csv")

# Use real return_prob if classifier has run, else use stub (~5% returns)
if "return_prob" not in master_df.columns:
    print("WARNING: return_prob not found — using 5% random stub.")
    print("         Replace with real probabilities after running 03_return_ml.ipynb")

    rng = np.random.default_rng(42)
    master_df["return_prob"] = rng.beta(0.5, 9.5, size=len(master_df))
    master_df["return_prob"] = master_df["return_prob"].clip(0, 1)
else:
    print("Using real return_prob from classifier.")

# Flag return nodes
master_df["is_return_node"] = (master_df["return_prob"] > RETURN_PROB_THRESHOLD).astype(int)

n_returns = master_df["is_return_node"].sum()
print(f"Return nodes: {n_returns:,} ({n_returns/len(master_df)*100:.1f}% of orders)")


# In[3]:


from scipy.spatial import cKDTree

# Assign each customer to nearest dark store by lat/lon
store_coords = dark_stores[["lat", "lon"]].values
cust_coords  = master_df[["customer_lat", "customer_lon"]].values

tree = cKDTree(store_coords)
_, indices = tree.query(cust_coords)

master_df["dark_store_id"] = dark_stores["dark_store_id"].iloc[indices].values

# Re-save so all downstream notebooks have this column
master_df.to_parquet(DATA_DIR / "master_df_v2.parquet", index=False)

print(f"✅ Assigned {len(master_df):,} orders to {master_df['dark_store_id'].nunique()} dark stores")
print(master_df["dark_store_id"].value_counts().sort_index())


# In[4]:


def build_vrp_nodes(master_df, dark_stores, max_per_zone=MAX_CUSTOMERS_PER_ZONE, seed=42):
    """
    For each dark store zone, build a node list:
      node 0  = depot (dark store centroid)
      node 1+ = sampled customers in that zone
    Returns list of dicts, one per zone.
    """
    rng = np.random.default_rng(seed)
    zones = []
    for _, store in dark_stores.iterrows():
        zid = int(store["dark_store_id"])
        zone_df = master_df[master_df["dark_store_id"] == zid].copy()

        # Sample for tractability
        if len(zone_df) > max_per_zone:
            zone_df = zone_df.sample(n=max_per_zone,
                                     random_state=int(rng.integers(0, 2**31)))

        # Build node coords: depot first, then customers
        depot_coords = np.array([[store["lat"], store["lon"]]])
        cust_coords  = zone_df[["customer_lat","customer_lon"]].values
        node_coords  = np.vstack([depot_coords, cust_coords])

        # Demands in grams (depot = 0)
        if "product_weight_g" in zone_df.columns:
            demands = np.concatenate([[0], zone_df["product_weight_g"].fillna(500).clip(50, 30_000).values])
        else:
            demands = np.concatenate([[0], np.full(len(zone_df), 500)])

        # Time windows in minutes from midnight
        # Assign AM/PM alternately
        tw = [[0, 1440]]  # depot: all day
        for i in range(len(zone_df)):
            tw.append([480, 720] if i % 2 == 0 else [720, 1080])  # AM or PM

        node_ids = ["depot"] + zone_df.index.astype(str).tolist()

        zones.append({
            "zone_id":     zid,
            "store_lat":   store["lat"],
            "store_lon":   store["lon"],
            "node_coords": node_coords,
            "demands":     demands.astype(int),
            "time_windows": tw,
            "node_ids":    node_ids,
            "n_customers": len(zone_df),
        })

    print(f"Built VRP nodes for {len(zones)} zones")
    for z in zones:
        print(f"  Zone {z['zone_id']:2d}: {z['n_customers']} customers | "
              f"total demand = {z['demands'].sum()/1000:.1f} kg")
    return zones

zones = build_vrp_nodes(master_df, dark_stores)


# In[5]:


rows = []
for z in zones:
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

vrp_nodes = pd.DataFrame(rows)
vrp_nodes.to_csv(DATA_DIR / "vrp_nodes.csv", index=False)
print(f"Saved: {DATA_DIR / 'vrp_nodes.csv'}  ({len(vrp_nodes)} rows)")
vrp_nodes.head(6)


# In[6]:


def solve_cvrptw(zone: dict, num_vehicles: int = 10) -> dict:
    n       = len(zone["node_coords"])
    demands = zone["demands"].tolist()
    tw      = zone["time_windows"]

    dist_matrix = build_distance_matrix(zone["node_coords"])
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60
    time_matrix = np.rint(dist_matrix / speed_m_per_min).astype(int)

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
        idx = manager.NodeToIndex(node_idx)
        time_dim.CumulVar(idx).SetRange(open_t, close_t)

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

print("Solver function defined.")


# In[7]:


print("Running Forward VRP (CVRPTW) across all zones...")
print(f"Time limit: {SOLVER_TIME_LIMIT_S}s per zone | "
      f"Vehicle capacity: {VEHICLE_CAPACITY_G/1000:.0f} kg\n")

zone_results = {}
for z in zones:
    result = solve_cvrptw(z)
    zone_results[z["zone_id"]] = result

n_solved = sum(r["solved"] for r in zone_results.values())
print(f"\nCompleted: {n_solved}/{len(zones)} zones solved.")


# In[8]:


# Combine all route DataFrames
all_routes_dfs = [r["routes_df"] for r in zone_results.values()
                  if r["solved"] and "routes_df" in r]
forward_routes_df = pd.concat(all_routes_dfs, ignore_index=True)
forward_routes_df.to_csv("outputs/forward_routes.csv", index=False)

# Save JSON
forward_routes_json = []
for zid, r in zone_results.items():
    if not r["solved"]:
        continue
    forward_routes_json.append({
        "zone_id":          zid,
        "n_vehicles":       r["n_vehicles"],
        "total_dist_km":    r["total_dist_km"],
        "routing_cost_R$":  r["routing_cost_R$"],
        "routes":           r["routes_df"].to_dict(orient="records"),
    })
with open("outputs/forward_routes.json", "w") as f:
    json.dump(forward_routes_json, f, indent=2)

# KPI Summary
kpi_rows = []
for zid, r in zone_results.items():
    if not r["solved"]:
        continue
    kpi_rows.append({
        "zone_id":            zid,
        "n_customers":        zones[zid]["n_customers"],
        "n_vehicles_used":    r["n_vehicles"],
        "total_dist_km":      round(r["total_dist_km"], 2),
        "routing_cost_R$":    round(r["routing_cost_R$"], 2),
        "max_route_km":       r["summary"]["max_route_km"],
        "min_route_km":       r["summary"]["min_route_km"],
    })

kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv("outputs/forward_kpi_summary.csv", index=False)

print("Saved:")
print("  outputs/forward_routes.csv")
print("  outputs/forward_routes.json")
print("  outputs/forward_kpi_summary.csv")
print(kpi_df.to_string(index=False))


# In[9]:


# Naive baseline: direct seller-to-customer, no dark stores
# Use mean customer-to-nearest-seller distance from baseline_kpis.json
import json as _json

baseline_path = DATA_DIR / "baseline_kpis.json"
print(f"\nLoading baseline KPIs from: {baseline_path}")
with open(baseline_path) as f:
    baseline = _json.load(f)

naive_avg_dist = (
    baseline.get("mean_customer_seller_distance_km")
    or baseline.get("mean_dist_km")
    or baseline.get("avg_distance_km")
    or 150.0   # fallback
)

route_max_per_vehicle = (
    forward_routes_df
    .groupby(["zone_id", "vehicle_id"])["cumulative_distance_km"]
    .max()
)
dark_store_avg = route_max_per_vehicle.mean()


total_fwd_cost = kpi_df["routing_cost_R$"].sum()
total_vehicles = kpi_df["n_vehicles_used"].sum()
total_dist     = kpi_df["total_dist_km"].sum()

print("=" * 55)
print("FORWARD VRP — BASELINE COMPARISON")
print("=" * 55)
print(f"Naive avg distance (seller→customer) : {naive_avg_dist:.1f} km")
print(f"Dark store avg route distance        : {dark_store_avg:.1f} km")
print(f"Distance reduction                   : "
      f"{(1 - dark_store_avg / naive_avg_dist) * 100:.1f}%")
print(f"\nTotal fleet distance                 : {total_dist:.1f} km")
print(f"Total vehicles deployed              : {total_vehicles}")
print(f"Total forward routing cost           : R${total_fwd_cost:.0f}")
print(f"Zones solved                         : {len(kpi_df)}/{len(zones)}")
print("=" * 55)


# In[ ]:




