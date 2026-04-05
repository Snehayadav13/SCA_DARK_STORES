"""
all_zones_aggregator.py
Dark Store + Integrated Logistics

Collects forward + reverse KPIs into a unified DataFrame, computes joint
totals, and identifies the hardest zone (highest cost per delivery).

Inputs:
    outputs/forward_kpi_by_zone.csv     — from Pranav's route_parser
    outputs/reverse_kpi_summary.csv     — from Vybhav's reverse VRP

Output:
    outputs/all_zones_summary.csv       — joint totals per zone + hardest zone flagged

Usage:
    python all_zones_aggregator.py
    python all_zones_aggregator.py --fwd outputs/forward_kpi_by_zone.csv \
                                   --rev outputs/reverse_kpi_summary.csv \
                                   --out outputs/all_zones_summary.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Expected column aliases — tolerant to slight naming differences between
# Pranav's and Vybhav's outputs
# ---------------------------------------------------------------------------

# Forward KPI columns — verified against real forward_kpi_by_zone.csv
# Actual columns: zone_id, vehicle_id, total_distance_km, num_stops, routing_cost_R$, cost_per_stop
FWD_ALIASES = {
    "dark_store_id": ["zone_id", "dark_store_id", "store_id", "zone"],
    "total_distance_km": [
        "total_distance_km",
        "fwd_distance_km",
        "distance_km",
        "total_dist_km",
    ],
    "total_time_min": [
        "total_time_min",
        "fwd_time_min",
        "time_min",
        "total_duration_min",
    ],
    "total_load_kg": ["total_load_kg", "fwd_load_kg", "load_kg", "total_demand_kg"],
    "num_stops": ["num_stops", "n_stops", "num_customers", "n_customers"],
    "n_vehicles": [
        "vehicle_id",
        "n_vehicles",
        "num_vehicles",
        "n_routes",
        "num_routes",
    ],
    "total_cost": ["routing_cost_R$", "total_cost", "fwd_cost", "route_cost", "cost"],
    "cost_per_stop": ["cost_per_stop", "cost_per_delivery"],
}

# Reverse KPI columns — verified against real reverse_kpi_summary.csv
# Actual columns: zone_id, n_pickups, n_vehicles_used, total_dist_km, routing_cost_R$, max_route_km, min_route_km
REV_ALIASES = {
    "dark_store_id": ["zone_id", "dark_store_id", "store_id", "zone"],
    "total_pickup_distance": [
        "total_dist_km",
        "total_pickup_distance",
        "rev_distance_km",
        "pickup_distance_km",
    ],
    "total_pickup_cost": [
        "routing_cost_R$",
        "total_pickup_cost",
        "rev_cost",
        "pickup_cost",
    ],
    "avg_pickups_per_route": [
        "n_pickups",
        "avg_pickups_per_route",
        "avg_pickups",
        "mean_pickups_per_route",
    ],
    "n_pickup_vehicles": [
        "n_vehicles_used",
        "n_pickup_vehicles",
        "rev_vehicles",
        "num_pickup_vehicles",
    ],
    "max_route_km": ["max_route_km"],
    "min_route_km": ["min_route_km"],
}

# Vehicle fixed cost (R$ per route) — mirrors roadmap spec
VEHICLE_FIXED_COST = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_col(df: pd.DataFrame, aliases: list[str], fallback=None):
    """Return the first alias that exists as a column name, or fallback."""
    for a in aliases:
        if a in df.columns:
            return a
    return fallback


def _normalise(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    """Rename columns using alias_map; return df with canonical names only."""
    rename = {}
    for canonical, aliases in alias_map.items():
        found = _resolve_col(df, aliases)
        if found and found != canonical:
            rename[found] = canonical
    df = df.rename(columns=rename)
    # Keep only canonical columns that exist
    cols = [c for c in alias_map if c in df.columns]
    return df[cols].copy()


def _infer_cost(
    df: pd.DataFrame, dist_col: str, vehicles_col: str | None, cost_col: str
) -> pd.Series:
    """
    If cost column missing, estimate: distance_km * 2 (R$/km proxy) + n_vehicles * fixed_cost.
    This is a fallback — actual cost should come from OR-Tools output.
    """
    if cost_col in df.columns:
        return df[cost_col]
    cost = df[dist_col] * 2.0
    if vehicles_col and vehicles_col in df.columns:
        cost += df[vehicles_col] * VEHICLE_FIXED_COST
    return cost.round(2)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_forward(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Forward KPI file not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded forward KPIs: {len(df)} rows from {path.name}")
    print(f"  Forward columns: {df.columns.tolist()}")

    # forward_kpi_by_zone.csv has one row per (zone_id, vehicle_id).
    # Must aggregate to one row per zone before merging with reverse (one-row-per-zone).
    zone_col = _resolve_col(df, ["zone_id", "dark_store_id", "store_id", "zone"])
    if zone_col is None:
        raise ValueError("Cannot find zone identifier column in forward KPI file.")

    agg = {
        c: "sum" for c in df.select_dtypes(include="number").columns if c != zone_col
    }

    # cost_per_stop is a rate — take mean, not sum
    rate_cols = [c for c in ["cost_per_stop", "cost_per_delivery"] if c in agg]
    for c in rate_cols:
        agg[c] = "mean"

    # n_vehicles: count distinct vehicle_ids per zone
    vehicle_col = _resolve_col(df, ["vehicle_id", "n_vehicles", "num_vehicles"])
    if vehicle_col and vehicle_col in df.columns:
        agg[vehicle_col] = "nunique"

    df = df.groupby(zone_col, as_index=False).agg(agg)
    print(f"  Aggregated to {len(df)} zones (grouped by {zone_col})")

    df = _normalise(df, FWD_ALIASES)

    # Infer cost if missing
    if "total_cost" not in df.columns and "total_distance_km" in df.columns:
        df["total_cost"] = _infer_cost(
            df, "total_distance_km", "n_vehicles", "total_cost"
        )
        print("  Note: total_cost estimated from distance + vehicles")

    return df


def load_reverse(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Reverse KPI file not found: {path}\n"
            "Ensure Vybhav's reverse VRP has completed and saved reverse_kpi_summary.csv"
        )
    df = pd.read_csv(path)
    print(f"  Loaded reverse KPIs: {len(df)} zones from {path.name}")
    print(f"  Reverse columns: {df.columns.tolist()}")
    df = _normalise(df, REV_ALIASES)

    # Infer cost if missing
    if "total_pickup_cost" not in df.columns and "total_pickup_distance" in df.columns:
        df["total_pickup_cost"] = _infer_cost(
            df, "total_pickup_distance", "n_pickup_vehicles", "total_pickup_cost"
        )
        print(
            "  Note: total_pickup_cost not found in reverse file — estimated from distance + vehicles"
        )

    return df


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def aggregate(fwd: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """
    Merge forward + reverse KPIs on dark_store_id and compute joint totals:

        total_fleet_size     = fwd n_vehicles + rev n_pickup_vehicles
        total_distance_km    = fwd total_distance_km + rev total_pickup_distance
        total_cost           = fwd total_cost + rev total_pickup_cost
        cost_per_delivery    = total_cost / num_stops  (hardest zone metric)
        is_hardest_zone      = flag for zone with highest cost_per_delivery
    """
    merged = pd.merge(
        fwd, rev, on="dark_store_id", how="outer", suffixes=("_fwd", "_rev")
    )

    # Fill NaN for zones present in only one file (shouldn't happen but defensive)
    merged = merged.fillna(0)

    # ── Joint totals ─────────────────────────────────────────────────────────

    # Fleet size — n_vehicles from fwd (aliased from vehicle_id),
    # n_pickup_vehicles from rev (aliased from n_vehicles_used)
    n_fwd = merged.get("n_vehicles", pd.Series(0, index=merged.index))
    n_rev = merged.get("n_pickup_vehicles", pd.Series(0, index=merged.index))
    merged["total_fleet_size"] = (n_fwd + n_rev).astype(int)

    # Total distance
    d_fwd = merged.get("total_distance_km", pd.Series(0.0, index=merged.index))
    d_rev = merged.get("total_pickup_distance", pd.Series(0.0, index=merged.index))
    merged["total_distance_km_joint"] = (d_fwd + d_rev).round(3)

    # Total cost
    c_fwd = merged.get("total_cost", pd.Series(0.0, index=merged.index))
    c_rev = merged.get("total_pickup_cost", pd.Series(0.0, index=merged.index))
    merged["total_cost_joint"] = (c_fwd + c_rev).round(2)

    # Cost per delivery (hardest zone metric)
    # Prefer fwd cost_per_stop if present, else compute from joint cost / stops
    if "cost_per_stop" in merged.columns:
        merged["cost_per_delivery"] = merged["cost_per_stop"].round(4)
    else:
        stops = merged.get("num_stops", pd.Series(1, index=merged.index)).replace(0, 1)
        merged["cost_per_delivery"] = (merged["total_cost_joint"] / stops).round(4)

    # Hardest zone flag
    hardest_idx = merged["cost_per_delivery"].idxmax()
    merged["is_hardest_zone"] = False
    merged.loc[hardest_idx, "is_hardest_zone"] = True

    # ── Clean column order ────────────────────────────────────────────────────
    col_order = [
        "dark_store_id",
        # Forward
        "total_distance_km",
        "total_time_min",
        "total_load_kg",
        "num_stops",
        "n_vehicles",
        "total_cost",
        # Reverse
        "total_pickup_distance",
        "total_pickup_cost",
        "avg_pickups_per_route",
        "n_pickup_vehicles",
        # Joint
        "total_fleet_size",
        "total_distance_km_joint",
        "total_cost_joint",
        "cost_per_delivery",
        "is_hardest_zone",
    ]
    existing = [c for c in col_order if c in merged.columns]
    extra = [c for c in merged.columns if c not in existing]
    merged = merged[existing + extra]

    return merged.sort_values("dark_store_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    print("\n  ── Joint Totals (all zones) ──────────────────────────")

    def _get(col):
        return df[col].sum() if col in df.columns else "N/A"

    total_fleet = _get("total_fleet_size")
    total_dist = _get("total_distance_km_joint")
    total_cost = _get("total_cost_joint")
    total_stops = _get("num_stops")
    total_pickups = _get("avg_pickups_per_route")

    print(f"    Zones              : {len(df)}")
    print(f"    Total fleet size   : {total_fleet} vehicles")
    print(f"    Total distance     : {total_dist:,.1f} km  (fwd + rev)")
    print(f"    Total cost         : R$ {total_cost:,.2f}  (fwd + rev)")
    print(f"    Total deliveries   : {int(total_stops):,}")

    if "is_hardest_zone" in df.columns:
        hardest = df[df["is_hardest_zone"] == True].iloc[0]
        print(f"\n  ── Hardest Zone ──────────────────────────────────────")
        print(f"    Zone ID            : {int(hardest['dark_store_id'])}")
        print(f"    Cost per delivery  : R$ {hardest['cost_per_delivery']:.4f}")
        print(f"    Total cost         : R$ {hardest['total_cost_joint']:.2f}")
        if "num_stops" in hardest:
            print(f"    Deliveries         : {int(hardest['num_stops'])}")
        print(f"    → Flag this zone for scheduling review / extra vehicle allocation")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    fwd_path: str | Path = "outputs/forward_kpi_by_zone.csv",
    rev_path: str | Path = "outputs/reverse_kpi_summary.csv",
    out_path: str | Path = "outputs/all_zones_summary.csv",
) -> pd.DataFrame:

    fwd_path = Path(fwd_path)
    rev_path = Path(rev_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ALL-ZONES JOINT AGGREGATOR")
    print("=" * 60)

    # Load
    fwd = load_forward(fwd_path)
    rev = load_reverse(rev_path)

    # Merge + compute
    print("\n  Aggregating forward + reverse KPIs …")
    summary = aggregate(fwd, rev)

    # Save
    summary.to_csv(out_path, index=False)
    print(
        f"  Saved → {out_path}  ({len(summary)} zones, {len(summary.columns)} columns)"
    )

    # Print summary
    print_summary(summary)

    # Sanity checks
    _sanity_check(summary)

    print("=" * 60)
    print(f"  OUTPUT: {out_path}")
    print("=" * 60 + "\n")

    return summary


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_check(df: pd.DataFrame) -> None:
    print("  ── Sanity checks ─────────────────────────────────────")

    # Exactly one hardest zone
    if "is_hardest_zone" in df.columns:
        n_hardest = df["is_hardest_zone"].sum()
        assert n_hardest == 1, f"Expected 1 hardest zone, got {n_hardest}"
        print("  ✓ Exactly 1 hardest zone flagged")

    # Joint cost >= both components
    if all(
        c in df.columns for c in ["total_cost", "total_pickup_cost", "total_cost_joint"]
    ):
        assert (
            df["total_cost_joint"] >= df["total_cost"] - 0.01
        ).all(), "Joint cost should be >= forward cost"
        assert (
            df["total_cost_joint"] >= df["total_pickup_cost"] - 0.01
        ).all(), "Joint cost should be >= reverse cost"
        print("  ✓ Joint cost ≥ both forward and reverse costs")

    # Joint distance >= forward distance
    if all(c in df.columns for c in ["total_distance_km", "total_distance_km_joint"]):
        assert (
            df["total_distance_km_joint"] >= df["total_distance_km"] - 0.01
        ).all(), "Joint distance should be >= forward distance"
        print("  ✓ Joint distance ≥ forward distance")

    # Fleet size — warn only (vehicle columns may not be in all VRP outputs)
    if "total_fleet_size" in df.columns:
        zeros = (df["total_fleet_size"] == 0).sum()
        if zeros > 0:
            print(
                f"  ⚠ {zeros} zone(s) have fleet_size = 0 "
                "(n_vehicles / n_pickup_vehicles not found in input files — check column names)"
            )
        else:
            print("  ✓ All zones have fleet_size > 0")

    # No NaNs in key columns
    key_cols = ["dark_store_id", "total_cost_joint", "cost_per_delivery"]
    for c in key_cols:
        if c in df.columns:
            assert df[c].notna().all(), f"NaN values found in {c}"
    print("  ✓ No NaN values in key columns")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate forward + reverse VRP KPIs into all_zones_summary.csv"
    )
    parser.add_argument(
        "--fwd",
        default="outputs/forward_kpi_by_zone.csv",
        help="Path to forward_kpi_by_zone.csv (default: outputs/forward_kpi_by_zone.csv)",
    )
    parser.add_argument(
        "--rev",
        default="outputs/reverse_kpi_summary.csv",
        help="Path to reverse_kpi_summary.csv (default: outputs/reverse_kpi_summary.csv)",
    )
    parser.add_argument(
        "--out",
        default="outputs/all_zones_summary.csv",
        help="Output path for all_zones_summary.csv (default: outputs/all_zones_summary.csv)",
    )
    args = parser.parse_args()

    run(
        fwd_path=args.fwd,
        rev_path=args.rev,
        out_path=args.out,
    )
