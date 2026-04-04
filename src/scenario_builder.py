"""
scenario_builder.py — Day 4 · Anurag
Dark Store + Integrated Logistics

Builds vrp_nodes.csv from master_df_v3.parquet (since Pranav's Day 2-3
file is not yet available), then produces three scenario variants:

    Scenario A (Base)          → outputs/vrp_nodes_A.csv   current demand
    Scenario B (Surge +30%)    → outputs/vrp_nodes_B.csv   all demand_kg × 1.3
    Scenario C (High Returns)  → outputs/vrp_nodes_C.csv   return_prob threshold halved (×2 flagged returners)

Intermediate output:
    data/vrp_nodes.csv         base node list (reusable by Pritam / Pranav)

Usage:
    python scenario_builder.py
    python scenario_builder.py --parquet data/master_df_v3.parquet
    python scenario_builder.py --parquet data/master_df_v3.parquet --out_dir outputs --vrp_dir data

Dependencies:
    pandas, numpy, scipy, pathlib  (all in requirements.txt)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (mirror roadmap specs)
# ---------------------------------------------------------------------------

VEHICLE_CAPACITY_KG = 500  # max load per vehicle
VEHICLE_SPEED_KMH = 40  # assumed average speed
VEHICLE_FIXED_COST = 50  # R$ per route
CUSTOMERS_PER_ZONE = 70  # sample 60-80 per dark store zone (roadmap)
SERVICE_TIME_MIN = 5  # minutes per stop
RETURN_PROB_THRESH = 0.30  # base threshold → pickup node if prob >= this

# Time windows (minutes from midnight)
TW_MORNING_OPEN = 480  # 08:00
TW_MORNING_CLOSE = 720  # 12:00
TW_AFTERNOON_OPEN = 720  # 12:00
TW_AFTERNOON_CLOSE = 1080  # 18:00


# ---------------------------------------------------------------------------
# Step 1: Load and validate master_df_v3
# ---------------------------------------------------------------------------


def load_master(parquet_path: str | Path) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"master_df_v3.parquet not found at {path}. "
            "Ensure Stage 3 (return_classifier) has completed."
        )
    # Support .csv files for testing; otherwise read parquet
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Could not read parquet file: {e}\n"
                "Ensure pyarrow or fastparquet is installed: pip install pyarrow"
            ) from e

    # Hard required columns (verified against real master_df_v3.parquet)
    required = {
        "customer_lat",
        "customer_lon",
        "seller_lat",
        "seller_lon",
        "dark_store_id",
        "product_weight_g",
        "return_prob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master_df_v3 is missing columns: {missing}")

    # Normalise order timestamp to 'order_date' for internal use.
    # Real master_df_v3 has: order_purchase_timestamp, order_approved_at, etc.
    DATE_ALIASES = [
        "order_purchase_timestamp",  # actual column in master_df_v3
        "order_approved_at",
        "order_delivered_customer_date",
        "purchase_timestamp",
        "order_timestamp",
        "timestamp",
    ]
    if "order_date" not in df.columns:
        for alias in DATE_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "order_date"})
                print(f"  Note: using '{alias}' as order_date")
                break
        else:
            print(
                "  Warning: no timestamp column found — all nodes default to morning window"
            )

    # Normalise price column if needed
    if "order_value" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "order_value"})

    print(
        f"  Loaded master_df_v3: {len(df):,} rows, {df['dark_store_id'].nunique()} zones"
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Assign time windows from order_date hour
# ---------------------------------------------------------------------------


def assign_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Morning window  (08:00-12:00) if order hour < 12, else afternoon (12:00-18:00).
    Falls back to morning if order_date can't be parsed.
    """
    df = df.copy()
    try:
        hours = pd.to_datetime(df["order_date"]).dt.hour
        is_morning = hours < 12
    except Exception:
        is_morning = pd.Series(True, index=df.index)

    df["tw_open_min"] = np.where(is_morning, TW_MORNING_OPEN, TW_AFTERNOON_OPEN)
    df["tw_close_min"] = np.where(is_morning, TW_MORNING_CLOSE, TW_AFTERNOON_CLOSE)
    return df


# ---------------------------------------------------------------------------
# Step 3: Build base vrp_nodes for a single zone
# ---------------------------------------------------------------------------


def _dark_store_row(store_id: int, store_lat: float, store_lon: float) -> dict:
    """Dark store depot node — demand = 0, full-day time window."""
    return {
        "node_id": f"DS_{store_id:02d}",
        "type": "depot",
        "dark_store_id": store_id,
        "lat": round(store_lat, 6),
        "lon": round(store_lon, 6),
        "demand_kg": 0.0,
        "tw_open_min": TW_MORNING_OPEN,
        "tw_close_min": TW_AFTERNOON_CLOSE,
        "service_time_min": 0,
        "return_prob": 0.0,
        "is_pickup": False,
    }


def build_vrp_nodes(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dark store zone:
      - Add 1 depot row (the dark store itself)
      - Sample up to CUSTOMERS_PER_ZONE customer rows
      - Assign node_id, type, demand_kg, time windows, is_pickup flag
    """
    df = assign_time_window(df)
    df = df.copy()
    df["demand_kg"] = (df["product_weight_g"] / 1000).clip(lower=0.1).round(3)

    # Build a lat/lon lookup for dark store centroids
    # stores_df must have: dark_store_id, lat, lon
    store_lookup = stores_df.set_index("dark_store_id")[["lat", "lon"]].to_dict("index")

    rows = []
    rng = np.random.default_rng(seed=42)

    for store_id, zone_df in df.groupby("dark_store_id"):
        # Depot row
        if store_id in store_lookup:
            s_lat = store_lookup[store_id]["lat"]
            s_lon = store_lookup[store_id]["lon"]
        else:
            # Fallback: centroid of zone customers
            s_lat = zone_df["customer_lat"].mean()
            s_lon = zone_df["customer_lon"].mean()

        rows.append(_dark_store_row(store_id, s_lat, s_lon))

        # Customer sample
        n_sample = min(CUSTOMERS_PER_ZONE, len(zone_df))
        sample = zone_df.sample(n=n_sample, random_state=int(rng.integers(0, 9999)))

        for seq, (_, row) in enumerate(sample.iterrows(), start=1):
            is_pickup = bool(row["return_prob"] >= RETURN_PROB_THRESH)
            rows.append(
                {
                    "node_id": f"C_{store_id:02d}_{seq:04d}",
                    "type": "pickup" if is_pickup else "delivery",
                    "dark_store_id": store_id,
                    "lat": round(row["customer_lat"], 6),
                    "lon": round(row["customer_lon"], 6),
                    "demand_kg": row["demand_kg"],
                    "tw_open_min": int(row["tw_open_min"]),
                    "tw_close_min": int(row["tw_close_min"]),
                    "service_time_min": SERVICE_TIME_MIN,
                    "return_prob": round(row["return_prob"], 4),
                    "is_pickup": is_pickup,
                }
            )

    nodes = pd.DataFrame(rows)
    return nodes


# ---------------------------------------------------------------------------
# Step 4: Compute dark store centroids directly from master_df_v3
# ---------------------------------------------------------------------------


def derive_stores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute zone depot locations purely from master_df_v3.
    Uses mean customer lat/lon per dark_store_id as the depot centroid.
    No external file dependency — master_df_v3 is the single source.
    """
    centroids = (
        df.groupby("dark_store_id")[["customer_lat", "customer_lon"]]
        .mean()
        .rename(columns={"customer_lat": "lat", "customer_lon": "lon"})
        .reset_index()
    )
    print(f"  Computed {len(centroids)} zone centroids from master_df_v3")
    return centroids


# ---------------------------------------------------------------------------
# Step 5: Three scenario transforms
# ---------------------------------------------------------------------------


def scenario_a(nodes: pd.DataFrame) -> pd.DataFrame:
    """Base — no change."""
    return nodes.copy()


def scenario_b(nodes: pd.DataFrame) -> pd.DataFrame:
    """Demand surge +30%: all delivery demand_kg × 1.3, capped at VEHICLE_CAPACITY_KG."""
    out = nodes.copy()
    mask = out["type"] == "delivery"
    out.loc[mask, "demand_kg"] = (
        (out.loc[mask, "demand_kg"] * 1.3).clip(upper=VEHICLE_CAPACITY_KG).round(3)
    )
    return out


def scenario_c(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    High returns × 2: halve the pickup threshold (0.30 → 0.15).
    Reclassifies delivery nodes with return_prob >= 0.15 as pickup nodes.
    """
    NEW_THRESH = RETURN_PROB_THRESH / 2  # 0.15
    out = nodes.copy()

    was_delivery = out["type"] == "delivery"
    now_pickup = was_delivery & (out["return_prob"] >= NEW_THRESH)

    out.loc[now_pickup, "type"] = "pickup"
    out.loc[now_pickup, "is_pickup"] = True

    n_extra = now_pickup.sum()
    print(
        f"    Scenario C: {n_extra} additional pickup nodes (threshold {NEW_THRESH:.2f})"
    )
    return out


# ---------------------------------------------------------------------------
# Step 6: Summary printer
# ---------------------------------------------------------------------------


def print_summary(label: str, nodes: pd.DataFrame) -> None:
    depots = (nodes["type"] == "depot").sum()
    deliveries = (nodes["type"] == "delivery").sum()
    pickups = (nodes["type"] == "pickup").sum()
    total_kg = nodes.loc[nodes["type"] == "delivery", "demand_kg"].sum()
    zones = nodes["dark_store_id"].nunique()

    print(f"\n  [{label}]")
    print(f"    Zones      : {zones}")
    print(f"    Depots     : {depots}")
    print(f"    Deliveries : {deliveries}")
    print(f"    Pickups    : {pickups}")
    print(f"    Total demand (delivery) : {total_kg:,.1f} kg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    parquet_path: str | Path = "data/master_df_v3.parquet",
    out_dir: str | Path = "outputs",
    vrp_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """
    Build vrp_nodes.csv and the three scenario files.

    Returns dict with keys: 'base', 'A', 'B', 'C'
    """
    out_dir = Path(out_dir)
    vrp_dir = Path(vrp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vrp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SCENARIO BUILDER — Day 4 (Anurag)")
    print("=" * 60)

    # 1. Load
    df = load_master(parquet_path)

    # 2. Dark store centroids — derived purely from master_df_v3
    stores = derive_stores(df)

    # 3. Build base vrp_nodes
    print("\n  Building base vrp_nodes …")
    base_nodes = build_vrp_nodes(df, stores)
    base_path = vrp_dir / "vrp_nodes.csv"
    base_nodes.to_csv(base_path, index=False)
    print(f"  Saved → {base_path}")
    print_summary("vrp_nodes (base)", base_nodes)

    # 4. Three scenarios
    scenarios = {
        "A": scenario_a(base_nodes),
        "B": scenario_b(base_nodes),
        "C": scenario_c(base_nodes),
    }

    for label, nodes in scenarios.items():
        path = out_dir / f"vrp_nodes_{label}.csv"
        nodes.to_csv(path, index=False)
        print(f"  Saved → {path}")
        print_summary(f"Scenario {label}", nodes)

    # 5. Quick sanity check
    _sanity_check(base_nodes, scenarios)

    print("\n" + "=" * 60)
    print("  SCENARIO BUILDER COMPLETE")
    print(f"  Base  : {vrp_dir}/vrp_nodes.csv")
    print(f"  Scen A: {out_dir}/vrp_nodes_A.csv")
    print(f"  Scen B: {out_dir}/vrp_nodes_B.csv")
    print(f"  Scen C: {out_dir}/vrp_nodes_C.csv")
    print("=" * 60 + "\n")

    return {"base": base_nodes, **scenarios}


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_check(base: pd.DataFrame, scenarios: dict[str, pd.DataFrame]) -> None:
    print("\n  — Sanity checks —")

    # B demand must be >= A demand for delivery nodes
    a_del = scenarios["A"].loc[scenarios["A"]["type"] == "delivery", "demand_kg"]
    b_del = scenarios["B"].loc[scenarios["B"]["type"] == "delivery", "demand_kg"]
    assert (b_del.values >= a_del.values).all(), "Scenario B demand should be >= A"
    print("  ✓ Scenario B demand ≥ Scenario A for all delivery nodes")

    # C must have >= as many pickups as A
    a_pick = (scenarios["A"]["type"] == "pickup").sum()
    c_pick = (scenarios["C"]["type"] == "pickup").sum()
    assert c_pick >= a_pick, "Scenario C should have >= pickups vs A"
    print(f"  ✓ Scenario C pickups ({c_pick}) ≥ Scenario A pickups ({a_pick})")

    # No node should exceed vehicle capacity
    for label, nodes in scenarios.items():
        over = (nodes["demand_kg"] > VEHICLE_CAPACITY_KG).sum()
        assert over == 0, f"Scenario {label}: {over} nodes exceed vehicle capacity"
    print(f"  ✓ No single node exceeds vehicle capacity ({VEHICLE_CAPACITY_KG} kg)")

    # All zones present in all scenarios
    base_zones = set(base["dark_store_id"].unique())
    for label, nodes in scenarios.items():
        assert (
            set(nodes["dark_store_id"].unique()) == base_zones
        ), f"Scenario {label} missing zones vs base"
    print(f"  ✓ All {len(base_zones)} zones present in all scenarios")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build vrp_nodes.csv + 3 scenario files from master_df_v3.parquet"
    )
    parser.add_argument(
        "--parquet",
        default="data/master_df_v3.parquet",
        help="Path to master_df_v3.parquet (default: data/master_df_v3.parquet)",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs",
        help="Directory for vrp_nodes_A/B/C.csv (default: outputs)",
    )
    parser.add_argument(
        "--vrp_dir",
        default="data",
        help="Directory for base vrp_nodes.csv (default: data)",
    )
    args = parser.parse_args()

    run(
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        vrp_dir=args.vrp_dir,
    )
