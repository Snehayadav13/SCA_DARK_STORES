"""
Module: demand_baseline.py
Demand Profile + Baseline KPIs + master_df_v2


INPUT:
    data/master_df.parquet  (Day 1 output, SP-filtered, 41K rows)

OUTPUT:
    data/demand_profile.csv       — orders per zip per week
    data/baseline_kpis.csv        — naive baseline metrics (pre-dark-store)
    data/master_df_v2.parquet     — master_df + demand_per_zip column

INTERFACE:
    build_demand_profile(df)      -> pd.DataFrame
    compute_baseline_kpis(df)     -> dict
    haversine_km(lat1, lon1, lat2, lon2) -> float  # row-wise, no matrix needed
    enrich_master_df(df, demand)  -> pd.DataFrame   # adds demand_per_zip
    run(input_path, output_dir)   -> dict           # full pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine (row-wise, vectorised) — independent of haversine_matrix.py
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Haversine distance in km between two arrays of (lat, lon).
    Works with scalars, Series, or ndarrays.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Demand profile
# ---------------------------------------------------------------------------

def build_demand_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate orders per customer_zip_code_prefix per ISO week.

    Returns DataFrame with columns:
        customer_zip_code_prefix, year, week, order_count, total_value,
        total_weight_g, return_count, return_rate
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["year"] = df["order_date"].dt.isocalendar().year.astype(int)
    df["week"] = df["order_date"].dt.isocalendar().week.astype(int)

    demand = (
        df.groupby(["customer_zip_code_prefix", "year", "week"], as_index=False)
        .agg(
            order_count=("order_id", "count"),
            total_value=("order_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            return_count=("is_return", "sum"),
        )
    )
    demand["return_rate"] = demand["return_count"] / demand["order_count"]
    demand = demand.sort_values(["customer_zip_code_prefix", "year", "week"])
    return demand


def build_zip_demand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total orders per zip (for demand weighting in K-Means).

    Returns DataFrame with columns:
        customer_zip_code_prefix, demand_order_count, demand_total_value,
        mean_lat, mean_lon
    """
    summary = (
        df.groupby("customer_zip_code_prefix", as_index=False)
        .agg(
            demand_order_count=("order_id", "count"),
            demand_total_value=("order_value", "sum"),
            mean_lat=("customer_lat", "mean"),
            mean_lon=("customer_lon", "mean"),
        )
    )
    return summary.sort_values("demand_order_count", ascending=False)


# ---------------------------------------------------------------------------
# Baseline KPIs (naive — no dark stores, no clustering)
# ---------------------------------------------------------------------------

def compute_baseline_kpis(df: pd.DataFrame) -> dict:
    """
    Compute the naive baseline metrics that our optimised solution
    will be compared against in the final report.

    Naive scenario: every order ships direct from seller to customer.
    No dark stores, no clustering, no route optimisation.
    """
    # Customer → seller Haversine distance (drop rows with null seller coords)
    valid = df.dropna(subset=["seller_lat", "seller_lon"]).copy()
    valid["cust_seller_dist_km"] = haversine_km(
        valid["customer_lat"].values, valid["customer_lon"].values,
        valid["seller_lat"].values, valid["seller_lon"].values,
    )

    kpis = {
        # --- Distance ---
        "mean_cust_seller_dist_km": round(valid["cust_seller_dist_km"].mean(), 2),
        "median_cust_seller_dist_km": round(valid["cust_seller_dist_km"].median(), 2),
        "p90_cust_seller_dist_km": round(valid["cust_seller_dist_km"].quantile(0.90), 2),
        "max_cust_seller_dist_km": round(valid["cust_seller_dist_km"].max(), 2),

        # --- Delivery time ---
        "mean_delivery_days": round(df["delivery_days"].mean(), 2),
        "median_delivery_days": round(df["delivery_days"].median(), 2),
        "mean_days_late": round(df["days_late"].mean(), 2),
        "pct_late_deliveries": round((df["days_late"] > 0).mean() * 100, 2),

        # --- Returns ---
        "return_rate_pct": round(df["is_return"].mean() * 100, 2),
        "total_returns": int(df["is_return"].sum()),
        "total_orders": len(df),

        # --- Cost proxies ---
        "mean_order_value": round(df["order_value"].mean(), 2),
        "mean_freight_value": round(df["freight_value"].mean(), 2),
        "total_freight_revenue": round(df["freight_value"].sum(), 2),

        # --- Coverage (naive = 0%, no dark stores) ---
        "dark_store_coverage_pct": 0.0,
        "num_dark_stores": 0,

        # --- Unique entities ---
        "unique_customers": df["customer_id"].nunique(),
        "unique_zips": df["customer_zip_code_prefix"].nunique(),
        "unique_sellers_in_sp_orders": valid["seller_zip_code_prefix"].nunique()
            if "seller_zip_code_prefix" in valid.columns else -1,
    }
    return kpis


# ---------------------------------------------------------------------------
# Enrich master_df → v2
# ---------------------------------------------------------------------------

def enrich_master_df(df: pd.DataFrame, zip_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add demand_per_zip (order count per zip) to master_df.
    This column is used as sample_weight in K-Means clustering.
    """
    df = df.merge(
        zip_summary[["customer_zip_code_prefix", "demand_order_count"]],
        on="customer_zip_code_prefix",
        how="left",
    )
    df["demand_per_zip"] = df["demand_order_count"].fillna(1).astype(int)
    df = df.drop(columns=["demand_order_count"], errors="ignore")

    # Also add customer→seller distance for each row
    valid_mask = df["seller_lat"].notna() & df["seller_lon"].notna()
    df.loc[valid_mask, "cust_seller_dist_km"] = haversine_km(
        df.loc[valid_mask, "customer_lat"].values,
        df.loc[valid_mask, "customer_lon"].values,
        df.loc[valid_mask, "seller_lat"].values,
        df.loc[valid_mask, "seller_lon"].values,
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: str | Path = "data/master_df.parquet",
    output_dir: str | Path = "data",
) -> dict:
    """Full Day 2 pipeline: demand profile + baseline KPIs + master_df_v2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DAY 2 — DEMAND PROFILE + BASELINE KPIs")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading master_df.parquet...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows")

    # Demand profile (weekly, per zip)
    print("\n[2/5] Building demand profile (weekly × zip)...")
    demand = build_demand_profile(df)
    demand_path = output_dir / "demand_profile.csv"
    demand.to_csv(demand_path, index=False)
    print(f"  Saved {len(demand):,} rows → {demand_path}")
    print(f"  Unique zips: {demand['customer_zip_code_prefix'].nunique():,}")
    print(f"  Week range:  {demand['week'].min()}–{demand['week'].max()}")

    # Zip-level demand summary
    print("\n[3/5] Building zip demand summary...")
    zip_summary = build_zip_demand_summary(df)
    zip_path = output_dir / "zip_demand_summary.csv"
    zip_summary.to_csv(zip_path, index=False)
    print(f"  Saved {len(zip_summary):,} zips → {zip_path}")
    print(f"  Top 5 zips by order count:")
    for _, row in zip_summary.head(5).iterrows():
        print(f"    ZIP {int(row['customer_zip_code_prefix']):>5d}  "
              f"{int(row['demand_order_count']):>4d} orders  "
              f"({row['mean_lat']:.4f}, {row['mean_lon']:.4f})")

    # Baseline KPIs
    print("\n[4/5] Computing baseline KPIs (naive — no dark stores)...")
    kpis = compute_baseline_kpis(df)
    kpi_path = output_dir / "baseline_kpis.csv"
    pd.DataFrame([kpis]).to_csv(kpi_path, index=False)
    print(f"  Saved → {kpi_path}")
    print(f"  Key metrics:")
    print(f"    Mean cust→seller distance: {kpis['mean_cust_seller_dist_km']} km")
    print(f"    Median cust→seller:        {kpis['median_cust_seller_dist_km']} km")
    print(f"    P90 cust→seller:           {kpis['p90_cust_seller_dist_km']} km")
    print(f"    Mean delivery days:        {kpis['mean_delivery_days']}")
    print(f"    % late deliveries:         {kpis['pct_late_deliveries']}%")
    print(f"    Return rate:               {kpis['return_rate_pct']}%")
    print(f"    Mean freight:              R$ {kpis['mean_freight_value']}")

    # Also save as JSON for easy loading downstream
    kpi_json_path = output_dir / "baseline_kpis.json"
    with open(kpi_json_path, "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"  Also saved JSON → {kpi_json_path}")

    # Enrich master_df → v2
    print("\n[5/5] Enriching master_df → v2...")
    df_v2 = enrich_master_df(df, zip_summary)
    v2_path = output_dir / "master_df_v2.parquet"
    df_v2.to_parquet(v2_path, index=False)
    print(f"  Saved {len(df_v2):,} rows × {df_v2.shape[1]} cols → {v2_path}")
    print(f"  New columns: demand_per_zip, cust_seller_dist_km")

    return kpis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/master_df.parquet"
    run(input_path=inp)