"""
Module: clustering.py
Stage:  Dark Store Placement (K-Means + p-Median)

INPUT:
    master_df : pd.DataFrame
        Columns required: customer_lat, customer_lon, customer_zip_code_prefix,
        order_id (for demand weighting).
        Source: data/master_df.parquet, filtered to customer_state='SP'.

OUTPUT:
    dark_store_candidates.csv  (Day 2) — K-Means centroids for K in {3..12}
    dark_stores_final.csv      (Day 3) — Chosen K locations with capacity + coverage
    master_df_v2 (modified)    — Adds dark_store_id column per customer

INTERFACE NOTES:
    - run_kmeans(coords, weights, k_range) -> dict[int, dict]
      Returns inertia and silhouette per K; centroids for each K.
    - pick_optimal_k(results) -> int
      Elbow + silhouette agreement heuristic.
    - run_p_median(distances, demands, p) -> list[int]
      Facility indices from PuLP MILP. Validates K-Means choice.
    - assign_voronoi(customers, centroids) -> pd.Series
      Returns dark_store_id for each customer row.
    - compute_coverage(master_df, centroids, radius_km=5.0) -> float
      % of customers within radius_km of their assigned dark store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(
    coords: np.ndarray,
    weights: np.ndarray | None = None,
    k_range: range = range(3, 13),
) -> dict:
    """
    Run K-Means for each K in k_range.

    Returns
    -------
    dict: {k: {"inertia": float, "silhouette": float, "centroids": np.ndarray}}
    """
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords, sample_weight=weights)
        sil = silhouette_score(coords, labels, sample_size=min(5000, len(coords)))
        results[k] = {
            "inertia": km.inertia_,
            "silhouette": sil,
            "centroids": km.cluster_centers_,
            "labels": labels,
        }
    return results


def pick_optimal_k(results: dict) -> int:
    """
    Choose K at the elbow/silhouette agreement point.
    Heuristic: highest silhouette score (primary), elbow as tie-breaker.
    """
    return max(results, key=lambda k: results[k]["silhouette"])


def run_p_median(distances: np.ndarray, demands: np.ndarray, p: int) -> list[int]:
    """
    Solve the p-median MILP using PuLP (CBC solver).

    Parameters
    ----------
    distances : np.ndarray, shape (n_customers, n_candidates)
    demands   : np.ndarray, shape (n_customers,)
    p         : int — number of facilities to open

    Returns
    -------
    List of facility indices (column indices) that are opened.
    """
    n_cust, n_fac = distances.shape
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_fac)] for i in range(n_cust)]
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_fac)]

    prob += pulp.lpSum(distances[i][j] * demands[i] * x[i][j]
                       for i in range(n_cust) for j in range(n_fac))
    for i in range(n_cust):
        prob += pulp.lpSum(x[i][j] for j in range(n_fac)) == 1
    for i in range(n_cust):
        for j in range(n_fac):
            prob += x[i][j] <= y[j]
    prob += pulp.lpSum(y[j] for j in range(n_fac)) == p

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [j for j in range(n_fac) if pulp.value(y[j]) > 0.5]


def assign_voronoi(customer_coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each customer to nearest centroid. Returns array of zone indices."""
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    _, indices = tree.query(customer_coords)
    return indices


def compute_coverage(customer_coords: np.ndarray, centroids: np.ndarray, radius_km: float = 5.0) -> float:
    """Return fraction of customers within radius_km of their assigned dark store."""
    from src.haversine_matrix import build_distance_matrix
    assigned = assign_voronoi(customer_coords, centroids)
    covered = 0
    for i, zone in enumerate(assigned):
        dist_scaled = build_distance_matrix(
            np.array([customer_coords[i], centroids[zone]])
        )[0, 1]
        if dist_scaled / 1000 <= radius_km:
            covered += 1
    return covered / len(customer_coords)
