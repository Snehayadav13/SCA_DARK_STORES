"""
Module: clustering.py
Stage:  Dark Store Placement (K-Means + p-Median)

DEPENDS ON:
    data/master_df_v2.parquet   — produced by demand_baseline.py
        Required columns: customer_lat, customer_lon,
        customer_zip_code_prefix, order_id, demand_per_zip,
        customer_unique_id, order_value, customer_state.

OUTPUT:
    data/dark_store_candidates.csv         — K-Means centroids for every K, with coverage column
    data/coverage_by_k.csv                 — K vs coverage sweep (K selection rationale)
    data/dark_stores_final.csv             — Chosen K locations with capacity + coverage KPIs
    data/master_df_v2.parquet              — Updated in-place: adds dark_store_id column
    data/p_median_locations.csv           — p-Median opened facility lat/lon
    data/KMeans_vs_pMedian_comparison.csv — Side-by-side weighted distance comparison

PUBLIC INTERFACE:
    load_sp_data(path)                              -> pd.DataFrame
    build_zip_level_coords(df)                      -> pd.DataFrame
    run_kmeans(coords, weights, k_range)            -> dict[int, dict]
    pick_optimal_k(results)                         -> int
    pick_k_by_coverage(kmeans_results,
                       customer_coords,
                       target_coverage,
                       radius_km)                   -> tuple[int, dict]
    run_p_median(distances, demands, p)             -> list[int]
    build_p_median_locations_df(centroids,
                                opened_indices)     -> pd.DataFrame
    save_p_median_outputs(p_median_df,
                          comparison_df, out_dir)  -> None
    assign_voronoi(customer_coords, centroids)      -> np.ndarray
    haversine_km(lat1, lon1, lat2, lon2)            -> np.ndarray | float
    compute_coverage(customer_coords, centroids,
                     radius_km)                     -> float
    build_dark_stores_df(centroids, master_df,
                         capacity_buffer)           -> pd.DataFrame
    save_outputs(kmeans_results, dark_stores_df,
                 master_df, coverage_by_k,
                 out_dir)                           -> None
    run_full_pipeline(parquet_path, out_dir,
                      k_range)                      -> dict
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts and pipelines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM: float = 6371.0

ZONE_COLOURS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653",
    "#F4A261", "#A8DADC", "#6D6875", "#B5838D", "#E07A5F",
    "#8338EC", "#FB5607",
]


# ---------------------------------------------------------------------------
# Haversine (vectorised) — mirrors demand_baseline.haversine_km signature
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM: float = 6371.0


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """
    Vectorised Haversine distance in km.
    Accepts scalars, Series, or ndarrays — mirrors demand_baseline.haversine_km.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_sp_data(path: str | Path) -> pd.DataFrame:
    """
    Load master_df_v2.parquet (output of demand_baseline.run()) and validate it.

    Expects the `demand_per_zip` column that demand_baseline adds.
    Falls back gracefully to master_df.parquet if v2 is not available yet
    (recomputes demand_per_zip from scratch in that case).

    Returns
    -------
    pd.DataFrame — SP-only rows with valid coordinates.
    """
    path = Path(path)
    df = pd.read_parquet(path)

    # If demand_per_zip is missing (e.g. master_df.parquet was passed directly)
    # derive it so the rest of the pipeline still works
    if "demand_per_zip" not in df.columns:
        print("[load_sp_data] WARNING: demand_per_zip not found — computing from scratch.")
        print("               Run demand_baseline.py first to produce master_df_v2.parquet.")
        zip_counts = df.groupby("customer_zip_code_prefix")["order_id"].count()
        df["demand_per_zip"] = df["customer_zip_code_prefix"].map(zip_counts).fillna(1).astype(int)

    # Guard: keep only SP rows (v2 should already be SP-only, but be explicit)
    if "customer_state" in df.columns:
        df = df[df["customer_state"] == "SP"].copy()

    # Filter to Greater São Paulo metro — dark stores are an urban concept;
    # full SP state spreads stores 400 km apart with near-zero coverage
    SP_METRO_CITIES = {
        "sao paulo", "guarulhos", "sao bernardo do campo",
        "santo andre", "osasco",
    }
    if "customer_city" in df.columns:
        before = len(df)
        df = df[df["customer_city"].isin(SP_METRO_CITIES)].copy()
        print(f"[load_sp_data] Metro filter: {before:,} → {len(df):,} rows "
              f"({len(df)/before*100:.1f}% of SP state retained)")

    # Drop rows missing coordinates
    df = df.dropna(subset=["customer_lat", "customer_lon"])

    # Sanity check: coordinates must fall inside São Paulo state bounding box
    sp_lat = (-25.3, -19.8)
    sp_lon = (-53.2, -44.0)
    before = len(df)
    df = df[
        df["customer_lat"].between(*sp_lat) &
        df["customer_lon"].between(*sp_lon)
    ].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[load_sp_data] Dropped {dropped} rows outside SP bounding box.")

    print(f"[load_sp_data] {len(df):,} rows | "
          f"{df['customer_zip_code_prefix'].nunique():,} zip codes | "
          f"demand_per_zip range: {df['demand_per_zip'].min()}–{df['demand_per_zip'].max()}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Zip-level coordinate table (for K-Means input)
# ---------------------------------------------------------------------------

def build_zip_level_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse order-level rows to one representative point per zip code.

    Uses the mean lat/lon per zip (consistent with demand_baseline.build_zip_demand_summary).
    Uses demand_per_zip as the demand weight (already computed by demand_baseline).

    We cluster at zip level — not order level — because:
    - A single address with 50 orders should not artificially dominate the centroid.
    - São Paulo has ~3,000 zip prefixes vs ~40,000 order rows: much faster to cluster.
    - demand_per_zip ensures high-demand zips still attract centroids.

    Returns
    -------
    pd.DataFrame with columns:
        customer_zip_code_prefix, lat, lon, demand_weight
    """
    zip_df = (
        df.groupby("customer_zip_code_prefix", as_index=False)
        .agg(
            lat=("customer_lat", "mean"),
            lon=("customer_lon", "mean"),
            demand_weight=("demand_per_zip", "first"),   # same value for all rows of a zip
        )
    )
    print(f"[build_zip_level_coords] {len(zip_df):,} zip codes | "
          f"demand_weight range: {zip_df['demand_weight'].min()}–{zip_df['demand_weight'].max()}")
    return zip_df


# ---------------------------------------------------------------------------
# 3. K-Means clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    coords: np.ndarray,
    weights: np.ndarray | None = None,
    k_range: range = range(3, 13),
) -> dict:
    """
    Run weighted K-Means for every K in k_range.

    Parameters
    ----------
    coords  : np.ndarray, shape (n, 2) — [lat, lon] per zip-code point
    weights : np.ndarray, shape (n,)   — demand_per_zip values (order counts)
    k_range : range                    — values of K to evaluate

    Returns
    -------
    dict: {
        k: {
            "inertia":    float,
            "silhouette": float,
            "centroids":  np.ndarray shape (k, 2),
            "labels":     np.ndarray shape (n,),
        }
    }
    """
    results: dict = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords, sample_weight=weights)
        sil = silhouette_score(coords, labels, sample_size=min(5000, len(coords)))
        results[k] = {
            "inertia":    km.inertia_,
            "silhouette": sil,
            "centroids":  km.cluster_centers_,
            "labels":     labels,
        }
        print(f"  K={k:2d}  inertia={km.inertia_:>12.2f}  silhouette={sil:.4f}")
    return results


def pick_optimal_k(results: dict) -> int:
    """
    Return the K with the highest silhouette score.

    Silhouette is preferred over the elbow alone because it balances both
    compactness (inertia) and cluster separation — the elbow is often ambiguous
    on real geographic data where the inertia curve declines smoothly.
    """
    return max(results, key=lambda k: results[k]["silhouette"])


def pick_k_by_coverage(
    kmeans_results: dict,
    customer_coords: np.ndarray,
    target_coverage: float = 0.70,
    radius_km: float = 5.0,
) -> tuple[int, dict]:
    """
    Return the minimum K that achieves target_coverage within radius_km,
    plus the full coverage sweep as a dict for saving.

    Coverage is the hard operational KPI — silhouette alone is insufficient
    because it optimises cluster separation, not service reach.
    pick_optimal_k (silhouette) is retained for academic comparison only.

    If no K hits the target, returns the K with the highest coverage + warning.

    Returns
    -------
    chosen_k     : int
    coverage_by_k: dict {k: float}  — fractional coverage for every K tried
    """
    coverage_by_k = {}
    for k, res in kmeans_results.items():
        c = compute_coverage(customer_coords, res["centroids"], radius_km=radius_km)
        coverage_by_k[k] = c
        print(f"  K={k:2d}  coverage={c * 100:.1f}%")

    viable = {k: v for k, v in coverage_by_k.items() if v >= target_coverage}
    if viable:
        chosen = min(viable)
        print(f"\n[pick_k_by_coverage] Minimum K hitting {target_coverage*100:.0f}% target: "
              f"K={chosen} ({coverage_by_k[chosen]*100:.1f}%)")
    else:
        chosen = max(coverage_by_k, key=coverage_by_k.get)
        print(f"\n[pick_k_by_coverage] WARNING: No K hits {target_coverage*100:.0f}% target. "
              f"Best available: K={chosen} ({coverage_by_k[chosen]*100:.1f}%)")
        print("  Consider: increase k_range upper bound, or tighten city filter.")
    return chosen, coverage_by_k


# ---------------------------------------------------------------------------
# 4. p-Median MILP (validation layer)
# ---------------------------------------------------------------------------

def run_p_median(
    distances: np.ndarray,
    demands: np.ndarray,
    p: int,
) -> list[int]:
    """
    Solve the p-median facility location problem with PuLP + CBC solver.

    Objective:  min  Σ_i Σ_j  d_ij · w_i · x_ij
    Constraints:
        Σ_j x_ij = 1       ∀ i   (each customer assigned to exactly one facility)
        x_ij ≤ y_j         ∀ i,j (only assign to open facilities)
        Σ_j y_j = p              (open exactly p facilities)
        x_ij, y_j ∈ {0, 1}

    Parameters
    ----------
    distances : np.ndarray, shape (n_customers, n_candidates)
        Haversine distance (km) from each customer/zip to each candidate facility.
    demands   : np.ndarray, shape (n_customers,)
        Demand weight per customer/zip (demand_per_zip).
    p         : int
        Number of facilities to open — set equal to optimal_k from K-Means.

    Returns
    -------
    list[int] — column indices (into distances) of the opened facilities.

    Notes
    -----
    Keep n_customers × n_candidates ≤ ~150×150 for CBC to solve in < 60 s.
    The notebook uses a 100-zip sample with K-Means centroids as candidates.
    """
    try:
        import pulp
    except ImportError:
        raise ImportError("PuLP is required. Install with: pip install pulp")

    n_cust, n_fac = distances.shape
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)

    x = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_fac)]
        for i in range(n_cust)
    ]
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_fac)]

    prob += pulp.lpSum(
        distances[i][j] * demands[i] * x[i][j]
        for i in range(n_cust) for j in range(n_fac)
    )
    for i in range(n_cust):
        prob += pulp.lpSum(x[i][j] for j in range(n_fac)) == 1
    for i in range(n_cust):
        for j in range(n_fac):
            prob += x[i][j] <= y[j]
    prob += pulp.lpSum(y[j] for j in range(n_fac)) == p

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    opened = [j for j in range(n_fac) if pulp.value(y[j]) > 0.5]
    print(f"[run_p_median] CBC status: {pulp.LpStatus[prob.status]} | "
          f"Opened facility indices: {opened}")
    return opened


def build_p_median_locations_df(
    centroids: np.ndarray,
    opened_indices: list[int],
) -> pd.DataFrame:
    """
    Build a DataFrame of p-Median facility locations from opened centroid indices.

    Returns
    -------
    pd.DataFrame with columns:
        p_median_store_id, source_centroid_idx, lat, lon
    """
    rows = []
    for rank, idx in enumerate(sorted(opened_indices)):
        rows.append({
            "p_median_store_id":   rank,
            "source_centroid_idx": idx,
            "lat":                 round(float(centroids[idx, 0]), 6),
            "lon":                 round(float(centroids[idx, 1]), 6),
        })
    return pd.DataFrame(rows)


def save_p_median_outputs(
    p_median_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_dir: str | Path = "data",
) -> None:
    """
    Write p-Median artefacts to disk.

    Files written
    -------------
    {out_dir}/p_median_locations.csv            — opened facility lat/lon
    {out_dir}/KMeans_vs_pMedian_comparison.csv  — side-by-side weighted distance comparison
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_median_df.to_csv(out_dir / "p_median_locations.csv", index=False)
    print(f"[save_p_median_outputs] p_median_locations.csv           ({len(p_median_df)} stores)")
    comparison_df.to_csv(out_dir / "KMeans_vs_pMedian_comparison.csv", index=False)
    print(f"[save_p_median_outputs] KMeans_vs_pMedian_comparison.csv ({len(comparison_df)} rows)")


# ---------------------------------------------------------------------------
# 5. Voronoi assignment
# ---------------------------------------------------------------------------

def assign_voronoi(
    customer_coords: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Assign each customer/zip to its nearest dark store centroid (nearest-neighbour).

    Uses scipy cKDTree on Euclidean (lat, lon). For São Paulo's ≈ 2° × 2° extent
    the flat-Earth distortion is < 1% — acceptable for zone assignment.

    Parameters
    ----------
    customer_coords : np.ndarray, shape (n, 2) — [lat, lon]
    centroids       : np.ndarray, shape (k, 2) — [lat, lon]

    Returns
    -------
    np.ndarray, shape (n,) — integer zone index 0 … k-1 per customer
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    _, indices = tree.query(customer_coords)
    return indices


# ---------------------------------------------------------------------------
# 6. Coverage
# ---------------------------------------------------------------------------

def compute_coverage(
    customer_coords: np.ndarray,
    centroids: np.ndarray,
    radius_km: float = 5.0,
) -> float:
    """
    Fraction of customers within radius_km of their assigned dark store.

    Uses true Haversine distances (not Euclidean) because coverage > 70%
    is a hard KPI reported in the final comparison table.

    Parameters
    ----------
    customer_coords : np.ndarray, shape (n, 2) — [lat, lon]
    centroids       : np.ndarray, shape (k, 2) — [lat, lon]
    radius_km       : float — default 5.0 km

    Returns
    -------
    float in [0, 1]
    """
    zone_ids = assign_voronoi(customer_coords, centroids)
    ds_lats = centroids[zone_ids, 0]
    ds_lons = centroids[zone_ids, 1]
    dists = haversine_km(
        customer_coords[:, 0], customer_coords[:, 1],
        ds_lats, ds_lons,
    )
    return float((dists <= radius_km).mean())


# ---------------------------------------------------------------------------
# 7. Build dark_stores_final DataFrame
# ---------------------------------------------------------------------------

def build_dark_stores_df(
    centroids: np.ndarray,
    master_df: pd.DataFrame,
    capacity_buffer: float = 1.3,
) -> pd.DataFrame:
    """
    Build the dark_stores_final.csv table from K-Means centroids.

    Columns:
        dark_store_id, lat, lon,
        n_unique_customers, n_orders, total_order_value,
        capacity_orders,          (= n_orders/K × buffer — max orders per store)
        coverage_5km_pct          (% of this zone's customers within 5 km)

    Parameters
    ----------
    centroids      : np.ndarray, shape (k, 2)
    master_df      : pd.DataFrame — must already have 'dark_store_id' column
    capacity_buffer: float — demand buffer (1.3 = 30% headroom for demand spikes)
    """
    k = len(centroids)
    total_orders = len(master_df)
    base_capacity = int((total_orders / k) * capacity_buffer)

    customer_coords = master_df[["customer_lat", "customer_lon"]].values
    zone_ids = assign_voronoi(customer_coords, centroids)

    rows = []
    for z in range(k):
        mask_df = master_df["dark_store_id"] == z
        mask_arr = zone_ids == z
        zone_coords = customer_coords[mask_arr]

        cov = compute_coverage(zone_coords, centroids[z:z + 1], radius_km=5.0) if mask_arr.sum() > 0 else 0.0

        rows.append({
            "dark_store_id":       z,
            "lat":                 round(float(centroids[z, 0]), 6),
            "lon":                 round(float(centroids[z, 1]), 6),
            "n_unique_customers":  int(master_df.loc[mask_df, "customer_unique_id"].nunique())
                                   if "customer_unique_id" in master_df.columns else int(mask_df.sum()),
            "n_orders":            int(mask_df.sum()),
            "total_order_value":   round(float(master_df.loc[mask_df, "order_value"].sum()), 2)
                                   if "order_value" in master_df.columns else 0.0,
            "capacity_orders":     base_capacity,
            "coverage_5km_pct":   round(cov * 100, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

def save_plots(
    kmeans_results: dict,
    optimal_k: int,
    master_df: pd.DataFrame,
    centroids: np.ndarray,
    coverage_overall: float,
    out_dir: str | Path = "outputs",
) -> None:
    """
    Generate and save the three standard clustering PNGs.

    Files written
    -------------
    {out_dir}/elbow_silhouette.png   — elbow curve + silhouette bars
    {out_dir}/kmeans_cluster_map.png — customer scatter coloured by zone
    {out_dir}/zone_demand_bars.png   — orders and value per zone
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list      = sorted(kmeans_results.keys())
    inertias    = [kmeans_results[k]["inertia"]    for k in k_list]
    silhouettes = [kmeans_results[k]["silhouette"] for k in k_list]

    # ── 1. Elbow + silhouette ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_list, inertias, marker="o", color="steelblue", linewidth=2)
    ax1.axvline(optimal_k, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Chosen K = {optimal_k}")
    ax1.set_xlabel("Number of dark stores (K)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Curve", fontsize=13, fontweight="bold")
    ax1.set_xticks(k_list)
    ax1.legend(fontsize=11)

    bar_cols = ["crimson" if k == optimal_k else "steelblue" for k in k_list]
    ax2.bar(k_list, silhouettes, color=bar_cols, edgecolor="white")
    ax2.set_xlabel("Number of dark stores (K)", fontsize=12)
    ax2.set_ylabel("Silhouette score", fontsize=12)
    ax2.set_title("Silhouette Score by K", fontsize=13, fontweight="bold")
    ax2.set_xticks(k_list)
    ax2.bar([], [], color="crimson", label=f"Coverage-optimal K = {optimal_k}")
    ax2.legend(fontsize=11)

    plt.suptitle(f"K selection — coverage-optimal K = {optimal_k}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save_plots] elbow_silhouette.png saved")

    # ── 2. Cluster map ───────────────────────────────────────────────────────
    np.random.seed(42)
    sample_idx = np.random.choice(len(master_df),
                                  size=min(10_000, len(master_df)), replace=False)
    sample_df  = master_df.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=(9, 9))
    for z in range(len(centroids)):
        mask = sample_df["dark_store_id"] == z
        ax.scatter(
            sample_df.loc[mask, "customer_lon"],
            sample_df.loc[mask, "customer_lat"],
            s=4, alpha=0.35, color=ZONE_COLOURS[z % len(ZONE_COLOURS)],
            label=f"Zone {z}",
        )
    ax.scatter(centroids[:, 1], centroids[:, 0],
               s=350, marker="*", color="black", zorder=6, label="Dark store")
    for z, (lat, lon) in enumerate(centroids):
        ax.annotate(f" DS-{z}", xy=(lon, lat), fontsize=8,
                    fontweight="bold", color="black", zorder=7)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        f"K-Means Dark Store Placement — SP Metro (K = {optimal_k})\n"
        f"★ = dark store  ·  5-km coverage = {coverage_overall * 100:.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "kmeans_cluster_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save_plots] kmeans_cluster_map.png saved")

    # ── 3. Zone demand bars ──────────────────────────────────────────────────
    zone_summary = (
        master_df.groupby("dark_store_id")
        .agg(n_orders=("order_id", "count"),
             total_value=("order_value", "sum"))
        .reset_index()
    )
    bar_cols = [ZONE_COLOURS[z % len(ZONE_COLOURS)]
                for z in zone_summary["dark_store_id"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(zone_summary["dark_store_id"], zone_summary["n_orders"],
            color=bar_cols, edgecolor="white")
    ax1.set_xlabel("Dark store zone", fontsize=12)
    ax1.set_ylabel("Orders", fontsize=12)
    ax1.set_title("Orders per zone", fontsize=13, fontweight="bold")
    ax1.set_xticks(zone_summary["dark_store_id"])

    ax2.bar(zone_summary["dark_store_id"], zone_summary["total_value"] / 1e3,
            color=bar_cols, edgecolor="white")
    ax2.set_xlabel("Dark store zone", fontsize=12)
    ax2.set_ylabel("Order value (R$ thousands)", fontsize=12)
    ax2.set_title("Order value per zone", fontsize=13, fontweight="bold")
    ax2.set_xticks(zone_summary["dark_store_id"])

    plt.suptitle("Demand distribution across dark store zones",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "zone_demand_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save_plots] zone_demand_bars.png saved")


# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    kmeans_results: dict,
    dark_stores_df: pd.DataFrame,
    master_df: pd.DataFrame,
    coverage_by_k: dict | None = None,
    out_dir: str | Path = "data",
) -> None:
    """
    Write all clustering artefacts to disk.

    Files written
    -------------
    {out_dir}/dark_store_candidates.csv     — centroids for all K, with coverage column
    {out_dir}/coverage_by_k.csv             — K vs coverage sweep (K selection rationale)
    {out_dir}/dark_stores_final.csv         — chosen-K table with KPIs
    {out_dir}/master_df_v2.parquet          — master_df with dark_store_id added
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # All candidate centroids — include coverage if provided
    rows = []
    for k, res in kmeans_results.items():
        cov = coverage_by_k[k] * 100 if coverage_by_k and k in coverage_by_k else None
        for z, (lat, lon) in enumerate(res["centroids"]):
            rows.append({
                "k":         k,
                "zone":      z,
                "lat":       lat,
                "lon":       lon,
                "inertia":   res["inertia"],
                "silhouette": res["silhouette"],
                "coverage_5km_pct": round(cov, 1) if cov is not None else None,
            })
    pd.DataFrame(rows).to_csv(out_dir / "dark_store_candidates.csv", index=False)
    print(f"[save_outputs] dark_store_candidates.csv  ({len(rows)} rows)")

    # Coverage sweep — K selection rationale
    if coverage_by_k:
        cov_df = pd.DataFrame([
            {
                "k":                  k,
                "coverage_5km_pct":   round(v * 100, 1),
                "silhouette":         round(kmeans_results[k]["silhouette"], 4),
                "inertia":            round(kmeans_results[k]["inertia"], 2),
            }
            for k, v in coverage_by_k.items()
        ])
        cov_df.to_csv(out_dir / "coverage_by_k.csv", index=False)
        print(f"[save_outputs] coverage_by_k.csv          ({len(cov_df)} rows)")

    # Final chosen dark stores
    dark_stores_df.to_csv(out_dir / "dark_stores_final.csv", index=False)
    print(f"[save_outputs] dark_stores_final.csv      ({len(dark_stores_df)} stores)")

    # master_df_v2 with dark_store_id
    master_df.to_parquet(out_dir / "master_df_v2.parquet", index=False)
    print(f"[save_outputs] master_df_v2.parquet       ({len(master_df):,} rows, "
          f"{master_df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# 9. Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def run_full_pipeline(
    parquet_path: str | Path = "data/master_df_v2.parquet",
    out_dir: str | Path = "data",
    plot_dir: str | Path = "outputs",
    k_range: range = range(3, 13),
) -> dict:
    """
    End-to-end clustering pipeline.

    Call from the notebook with one line:
        results = run_full_pipeline()

    Returns
    -------
    dict with keys:
        optimal_k, silhouette_k, centroids, dark_stores_df, master_df,
        kmeans_results, coverage_overall, coverage_by_k
    """
    print("=" * 60)
    print("  CLUSTERING PIPELINE — Dark Store Placement")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_sp_data(parquet_path)

    print("\n[2/5] Building zip-level coordinate table...")
    zip_df = build_zip_level_coords(df)
    coords  = zip_df[["lat", "lon"]].values
    weights = zip_df["demand_weight"].values

    print(f"\n[3/5] Running K-Means sweep (K = {k_range.start}–{k_range.stop - 1})...")
    kmeans_results = run_kmeans(coords, weights, k_range)

    print("\n[4/5] Selecting optimal K by coverage target (≥70% within 5 km)...")
    customer_coords = df[["customer_lat", "customer_lon"]].values
    optimal_k, coverage_by_k = pick_k_by_coverage(
        kmeans_results, customer_coords, target_coverage=0.70, radius_km=5.0
    )
    silhouette_k = pick_optimal_k(kmeans_results)
    if silhouette_k != optimal_k:
        print(f"      Note: silhouette preferred K={silhouette_k} — "
              f"overridden by coverage constraint (K={optimal_k})")
    centroids = kmeans_results[optimal_k]["centroids"]
    print(f"      Optimal K = {optimal_k}  "
          f"(silhouette = {kmeans_results[optimal_k]['silhouette']:.4f})")

    df["dark_store_id"] = assign_voronoi(customer_coords, centroids)

    coverage = compute_coverage(customer_coords, centroids, radius_km=5.0)
    print(f"      Coverage within 5 km: {coverage * 100:.1f}%")

    print("\n[5/5] Building output tables and saving...")
    dark_stores_df = build_dark_stores_df(centroids, df)
    save_outputs(kmeans_results, dark_stores_df, df,
                 coverage_by_k=coverage_by_k, out_dir=out_dir)
    save_plots(kmeans_results, optimal_k, df, centroids,
               coverage, out_dir=plot_dir)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Optimal K           : {optimal_k}")
    print(f"  Coverage (5 km)     : {coverage * 100:.1f}%")
    print(f"  Outputs             : {out_dir}/")
    print("=" * 60)

    return {
        "optimal_k":        optimal_k,
        "silhouette_k":     silhouette_k,
        "centroids":        centroids,
        "dark_stores_df":   dark_stores_df,
        "master_df":        df,
        "kmeans_results":   kmeans_results,
        "coverage_overall": coverage,
        "coverage_by_k":    coverage_by_k,
    }
if __name__ == "__main__":
    run_full_pipeline(
        parquet_path="data/master_df_v2.parquet",
        out_dir="data",
        plot_dir="outputs",
        k_range=range(3, 13),
    )
    