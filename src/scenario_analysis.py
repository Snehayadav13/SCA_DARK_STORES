"""
Module: scenario_analysis.py
Stage:  3-Scenario Analysis (Day 6 — Vybhav)

Loads vrp_nodes_A/B/C.csv produced by scenario_builder.py,
runs forward + reverse VRP for each, feeds into joint optimizer,
and records a 3-row × 6-column KPI table.

DEPENDS ON:
    data/vrp_nodes_A.csv        — base scenario
    data/vrp_nodes_B.csv        — surge +30% demand
    data/vrp_nodes_C.csv        — high returns ×2
    outputs/forward_kpi_summary.csv   — for coverage baseline
    src/forward_vrp.py          — solve_cvrptw
    src/reverse_vrp.py          — solve_reverse_cvrptw
    src/joint_optimizer.py      — run
    src/route_parser.py         — shared utilities

OUTPUT:
    outputs/scenario_results_table.csv   — 3 rows × 6 KPI columns
    outputs/scenario_B_infeasible.csv    — zones infeasible under surge
    outputs/scenario_C_reverse_delta.csv — reverse cost growth vs base

INTERFACE:
    load_scenario_zones(csv_path)  -> tuple[list[dict], list[dict]]
    run_scenario(label, csv_path)  -> dict
    run_all_scenarios(...)         -> pd.DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_parser import (
    VEHICLE_CAPACITY_G,
    VEHICLE_SPEED_KMH,
    FIXED_COST_PER_ROUTE,
    VAR_COST_PER_KM,
    SERVICE_TIME_MIN,
    SOLVER_TIME_LIMIT_S,
    NUM_VEHICLES,
    build_distance_matrix,
    compute_routing_cost,
    save_routes,
)
from src.forward_vrp import solve_cvrptw
from src.reverse_vrp import solve_reverse_cvrptw
from src.joint_optimizer import run as run_joint


# ---------------------------------------------------------------------------
# Zone dict builder from scenario CSV
# ---------------------------------------------------------------------------


def load_scenario_zones(
    csv_path: str | Path,
) -> tuple[list[dict], list[dict]]:
    """
    Convert a scenario CSV (vrp_nodes_A/B/C.csv) into two lists of zone dicts:
        fwd_zones  — delivery nodes per dark store (for forward VRP)
        rev_zones  — pickup nodes per dark store   (for reverse VRP)

    Each zone dict matches the format expected by solve_cvrptw /
    solve_reverse_cvrptw:
        zone_id, node_coords (np.ndarray), demands (np.ndarray),
        time_windows (list of [open, close]), node_ids (list), n_customers/n_pickups
    """
    df = pd.read_csv(csv_path)
    speed_m_per_min = VEHICLE_SPEED_KMH * 1000 / 60

    fwd_zones: list[dict] = []
    rev_zones: list[dict] = []

    for store_id, zone_df in df.groupby("dark_store_id"):
        depot = zone_df[zone_df["type"] == "depot"].iloc[0]
        deliveries = zone_df[zone_df["type"] == "delivery"]
        pickups = zone_df[zone_df["type"] == "pickup"]

        # ---- Forward zone (depot + delivery nodes) ---- #
        fwd_rows = pd.concat([zone_df[zone_df["type"] == "depot"], deliveries])
        fwd_coords = fwd_rows[["lat", "lon"]].values.astype(float)
        fwd_demands = (fwd_rows["demand_kg"] * 1000).values.astype(int)  # kg → grams
        fwd_tw = list(
            zip(
                fwd_rows["tw_open_min"].astype(int),
                fwd_rows["tw_close_min"].astype(int),
            )
        )
        fwd_zones.append(
            {
                "zone_id": int(store_id),
                "node_coords": fwd_coords,
                "demands": fwd_demands,
                "time_windows": fwd_tw,
                "node_ids": fwd_rows["node_id"].tolist(),
                "n_customers": len(deliveries),
            }
        )

        # ---- Reverse zone (depot + pickup nodes) ---- #
        rev_rows = pd.concat([zone_df[zone_df["type"] == "depot"], pickups])
        rev_coords = rev_rows[["lat", "lon"]].values.astype(float)
        rev_demands = (rev_rows["demand_kg"] * 1000).values.astype(int)
        rev_tw = list(
            zip(
                rev_rows["tw_open_min"].astype(int),
                rev_rows["tw_close_min"].astype(int),
            )
        )
        rev_zones.append(
            {
                "zone_id": int(store_id),
                "node_coords": rev_coords,
                "demands": rev_demands,
                "time_windows": rev_tw,
                "node_ids": rev_rows["node_id"].tolist(),
                "n_pickups": len(pickups),
            }
        )

    return fwd_zones, rev_zones


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    label: str,
    csv_path: str | Path,
    out_dir: str | Path = "outputs",
    dark_store_radius_km: float = 5.0,
) -> dict:
    """
    Run forward VRP + reverse VRP + joint optimizer for one scenario.

    Returns
    -------
    dict with 6 KPI columns + infeasibility info:
        scenario, total_routing_cost_R$, total_distance_km,
        avg_delivery_time_min, coverage_pct, return_efficiency_pct,
        n_vehicles, n_infeasible_zones
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    print(f"\n{'='*60}")
    print(f"  SCENARIO {label} — {csv_path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    fwd_zones, rev_zones = load_scenario_zones(csv_path)

    # ------------------------------------------------------------------ #
    # 1. Forward VRP
    # ------------------------------------------------------------------ #
    print(f"\n[1/3] Forward VRP ({len(fwd_zones)} zones)...")
    fwd_results = {}
    fwd_infeasible = []
    for zone in fwd_zones:
        r = solve_cvrptw(zone)
        fwd_results[zone["zone_id"]] = r
        if not r["solved"]:
            fwd_infeasible.append(zone["zone_id"])

    n_fwd_solved = sum(r["solved"] for r in fwd_results.values())
    print(
        f"  {n_fwd_solved}/{len(fwd_zones)} zones solved"
        + (f" | INFEASIBLE: {fwd_infeasible}" if fwd_infeasible else "")
    )

    # ------------------------------------------------------------------ #
    # 2. Reverse VRP
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Reverse VRP ({len(rev_zones)} zones)...")
    rev_results = {}
    for zone in rev_zones:
        if zone["n_pickups"] == 0:
            rev_results[zone["zone_id"]] = {"solved": False, "zone_id": zone["zone_id"]}
            continue
        r = solve_reverse_cvrptw(zone)
        rev_results[zone["zone_id"]] = r

    n_rev_solved = sum(r["solved"] for r in rev_results.values())
    print(f"  {n_rev_solved}/{len(rev_zones)} zones solved")

    # ------------------------------------------------------------------ #
    # 3. Aggregate route DataFrames
    # ------------------------------------------------------------------ #
    fwd_route_dfs = [
        r["routes_df"]
        for r in fwd_results.values()
        if r.get("solved") and "routes_df" in r
    ]
    rev_route_dfs = [
        r["routes_df"]
        for r in rev_results.values()
        if r.get("solved") and "routes_df" in r
    ]

    fwd_routes_df = (
        pd.concat(fwd_route_dfs, ignore_index=True) if fwd_route_dfs else pd.DataFrame()
    )
    rev_routes_df = (
        pd.concat(rev_route_dfs, ignore_index=True) if rev_route_dfs else pd.DataFrame()
    )

    # ------------------------------------------------------------------ #
    # 4. Joint optimizer
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Joint optimizer...")
    return_probs = pd.Series(df[df["type"] == "pickup"]["return_prob"].values)
    joint_result = run_joint(
        fwd_routes_df,
        rev_routes_df,
        return_probs,
        output_path=out_dir / f"joint_result_scenario_{label}.json",
    )

    # ------------------------------------------------------------------ #
    # 5. Compute 6 KPIs
    # ------------------------------------------------------------------ #

    # Cost
    total_fwd_cost = sum(
        r["routing_cost_R$"] for r in fwd_results.values() if r.get("solved")
    )
    total_rev_cost = sum(
        r["routing_cost_R$"] for r in rev_results.values() if r.get("solved")
    )
    total_cost = total_fwd_cost + total_rev_cost

    # Distance
    total_fwd_dist = sum(
        r["total_dist_km"] for r in fwd_results.values() if r.get("solved")
    )
    total_rev_dist = sum(
        r["total_dist_km"] for r in rev_results.values() if r.get("solved")
    )
    total_dist = total_fwd_dist + total_rev_dist

    # Avg delivery time — total dist / speed
    speed_km_per_min = VEHICLE_SPEED_KMH / 60
    avg_delivery_time_min = round(
        (total_fwd_dist / max(n_fwd_solved, 1)) / speed_km_per_min, 1
    )

    # Coverage — % delivery nodes within dark_store_radius_km of their depot
    delivery_df = df[df["type"] == "delivery"].copy()
    depot_df = df[df["type"] == "depot"][["dark_store_id", "lat", "lon"]].rename(
        columns={"lat": "depot_lat", "lon": "depot_lon"}
    )
    delivery_df = delivery_df.merge(depot_df, on="dark_store_id", how="left")
    # Haversine approx (degrees → km, rough)
    delivery_df["dist_to_depot_km"] = np.sqrt(
        ((delivery_df["lat"] - delivery_df["depot_lat"]) * 111.0) ** 2
        + ((delivery_df["lon"] - delivery_df["depot_lon"]) * 92.0) ** 2
    )
    coverage_pct = round(
        (delivery_df["dist_to_depot_km"] <= dark_store_radius_km).mean() * 100, 1
    )

    # Return efficiency — % pickup nodes served (vs total pickup nodes)
    n_pickups_total = (df["type"] == "pickup").sum()
    n_pickups_served = sum(
        r.get("n_vehicles", 0) > 0 for r in rev_results.values() if r.get("solved")
    )
    return_efficiency_pct = round((n_rev_solved / max(len(rev_zones), 1)) * 100, 1)

    # Vehicles
    n_fwd_veh = sum(r["n_vehicles"] for r in fwd_results.values() if r.get("solved"))
    n_rev_veh = sum(r["n_vehicles"] for r in rev_results.values() if r.get("solved"))
    n_vehicles = n_fwd_veh + n_rev_veh

    result = {
        "scenario": label,
        "total_routing_cost_R$": round(total_cost, 2),
        "total_distance_km": round(total_dist, 2),
        "avg_delivery_time_min": avg_delivery_time_min,
        "coverage_pct": coverage_pct,
        "return_efficiency_pct": return_efficiency_pct,
        "n_vehicles": n_vehicles,
        "n_infeasible_zones": len(fwd_infeasible),
        "infeasible_zones": fwd_infeasible,
        "fwd_cost_R$": round(total_fwd_cost, 2),
        "rev_cost_R$": round(total_rev_cost, 2),
        "n_pickups_total": int(n_pickups_total),
        "joint_Z": joint_result["Z"],
    }

    print(
        f"\n  [{label}] Cost=R${total_cost:.0f} | Dist={total_dist:.1f}km | "
        f"Coverage={coverage_pct}% | ReturnEff={return_efficiency_pct}% | "
        f"Vehicles={n_vehicles} | Infeasible={len(fwd_infeasible)}"
    )

    return result


# ---------------------------------------------------------------------------
# All scenarios runner
# ---------------------------------------------------------------------------


def run_all_scenarios(
    data_dir: str | Path = "data",
    out_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """
    Run all 3 scenarios and produce scenario_results_table.csv.

    Also writes:
        outputs/scenario_B_infeasible.csv    — zones infeasible under surge
        outputs/scenario_C_reverse_delta.csv — reverse cost growth vs base
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "A": data_dir / "vrp_nodes_A.csv",
        "B": data_dir / "vrp_nodes_B.csv",
        "C": data_dir / "vrp_nodes_C.csv",
    }

    rows = []
    for label, csv_path in scenarios.items():
        result = run_scenario(label, csv_path, out_dir=out_dir)
        rows.append(result)

    # ------------------------------------------------------------------ #
    # Main KPI table — 3 rows × 6 columns
    # ------------------------------------------------------------------ #
    kpi_cols = [
        "scenario",
        "total_routing_cost_R$",
        "total_distance_km",
        "avg_delivery_time_min",
        "coverage_pct",
        "return_efficiency_pct",
        "n_vehicles",
    ]
    results_df = pd.DataFrame(rows)
    kpi_df = results_df[kpi_cols].copy()
    kpi_path = out_dir / "scenario_results_table.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"\n[scenario_analysis] scenario_results_table.csv saved → {kpi_path}")

    # ------------------------------------------------------------------ #
    # Scenario B insight — infeasible zones
    # ------------------------------------------------------------------ #
    b_row = results_df[results_df["scenario"] == "B"].iloc[0]
    if b_row["n_infeasible_zones"] > 0:
        infeasible_df = pd.DataFrame(
            {
                "zone_id": b_row["infeasible_zones"],
                "reason": "demand_surge_+30pct_exceeds_capacity",
            }
        )
        infeasible_path = out_dir / "scenario_B_infeasible.csv"
        infeasible_df.to_csv(infeasible_path, index=False)
        print(
            f"[scenario_analysis] Scenario B: {b_row['n_infeasible_zones']} infeasible zones → {infeasible_path}"
        )
    else:
        print("[scenario_analysis] Scenario B: all zones feasible under +30% surge ✓")

    # ------------------------------------------------------------------ #
    # Scenario C insight — reverse cost delta vs base
    # ------------------------------------------------------------------ #
    a_rev = results_df[results_df["scenario"] == "A"]["rev_cost_R$"].iloc[0]
    c_rev = results_df[results_df["scenario"] == "C"]["rev_cost_R$"].iloc[0]
    delta_abs = round(c_rev - a_rev, 2)
    delta_pct = round((delta_abs / max(a_rev, 1)) * 100, 1)
    delta_df = pd.DataFrame(
        [
            {
                "base_rev_cost_R$": a_rev,
                "high_return_rev_cost_R$": c_rev,
                "delta_R$": delta_abs,
                "delta_pct": delta_pct,
                "extra_pickups": results_df[results_df["scenario"] == "C"][
                    "n_pickups_total"
                ].iloc[0]
                - results_df[results_df["scenario"] == "A"]["n_pickups_total"].iloc[0],
            }
        ]
    )
    delta_path = out_dir / "scenario_C_reverse_delta.csv"
    delta_df.to_csv(delta_path, index=False)
    print(
        f"[scenario_analysis] Scenario C: reverse cost +{delta_pct}% vs base (R${delta_abs:+.2f}) → {delta_path}"
    )

    # ------------------------------------------------------------------ #
    # Print final table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("  SCENARIO ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(kpi_df.to_string(index=False))
    print(f"\n  Outputs:")
    print(f"    {kpi_path}")
    print(f"    {out_dir}/scenario_B_infeasible.csv")
    print(f"    {out_dir}/scenario_C_reverse_delta.csv")

    return kpi_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_scenarios(
        data_dir="data",
        out_dir="outputs",
    )
