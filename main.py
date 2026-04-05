"""
main.py — Full pipeline entry point for Dark Store + Integrated Logistics

Stage execution order (dependency-driven):
    Stage 0:  data_pipeline        → data/master_df.parquet
    Stage 1:  demand_baseline      → data/master_df_v2.parquet + baseline KPIs
    Stage 2:  haversine_matrix     → data/distance_matrix.npy + sp_customer_sample.csv
    Stage 3:  clustering           → data/dark_stores_final.csv + dark_store_id column added to master_df_v2
    Stage 4:  return_classifier    → data/master_df_v3.parquet (return_prob + return_flag)
    Stage 5:  demand_forecasting   → outputs/forecasted_demand_by_zone.csv  [needs dark_store_id → after clustering]
    Stage 6:  scenario_builder     → data/vrp_nodes.csv + data/vrp_nodes_A/B/C.csv  [needs master_df_v3]
    Stage 7:  forward_vrp          → outputs/forward_routes.json + forward_kpi_by_zone.csv
    Stage 8:  reverse_vrp          → outputs/reverse_routes.json + reverse_kpi_summary.csv
    Stage 9:  all_zones_aggregator → outputs/all_zones_summary.csv  [needs fwd + rev KPIs]
    Stage 10: joint_optimizer      → outputs/hybrid_routes.json + hybrid_kpi_summary.csv
    Stage 11: scenario_analysis    → outputs/scenario_results_table.csv 
Usage:
    python main.py                              # run all stages
    python main.py --from clustering            # resume from a specific stage
    python main.py --stage forward_vrp          # run one stage only
    python main.py --list                       # print all stage names and exit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Stage registry — ordered by dependency
# ---------------------------------------------------------------------------

STAGES = [
    "data_pipeline",  # 0 — Olist CSVs → master_df.parquet
    "demand_baseline",  # 1 — demand profile + baseline KPIs + master_df_v2
    "haversine_matrix",  # 2 — 500×500 distance matrix
    "clustering",  # 3 — K-Means + p-Median → dark_stores_final + dark_store_id
    "return_classifier",  # 4 — XGBoost → return_prob → master_df_v3
    "demand_forecasting",  # 5 — Prophet per zone → forecasted_demand_by_zone.csv
    "scenario_builder",  # 6 — vrp_nodes.csv + scenario A/B/C variants
    "forward_vrp",  # 7 — OR-Tools CVRPTW → forward_routes + KPIs
    "reverse_vrp",  # 8 — OR-Tools CVRPTW → reverse_routes + KPIs
    "all_zones_aggregator",  # 9 — merge fwd + rev KPIs → all_zones_summary.csv
    "joint_optimizer",  # 10 — SDVRP hybrid + Pareto sweep
    "scenario_analysis",  # 11 — 3-scenario A/B/C KPI table
]


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_data_pipeline():
    from src.data_pipeline import run

    run(raw_dir="data/raw", output_path="data/master_df.parquet")


def run_demand_baseline():
    from src.demand_baseline import run

    run(input_path="data/master_df.parquet", output_dir="data")


def run_haversine_matrix():
    from src.haversine_matrix import run

    run(
        parquet_path="data/master_df.parquet",
        matrix_path="data/distance_matrix.npy",
        sample_csv_path="data/sp_customer_sample.csv",
    )


def run_clustering():
    from src.clustering import run_full_pipeline

    run_full_pipeline(
        parquet_path="data/master_df_v2.parquet",
        out_dir="data",
        plot_dir="outputs",
        k_range=range(3, 13),
    )


def run_return_classifier():
    from src.return_classifier import run_full_pipeline

    run_full_pipeline(
        parquet_path="data/master_df_v2.parquet",
        out_dir="outputs",
        data_dir="data",
    )


def run_demand_forecasting():
    from src.demand_forecasting import run_pipeline

    run_pipeline(input_path="data/master_df_v2.parquet")


def run_scenario_builder():
    from src.scenario_builder import run

    run(
        parquet_path="data/master_df_v3.parquet",
        out_dir="data",
        vrp_dir="data",
    )


def run_forward_vrp():
    from src.forward_vrp import run_full_pipeline

    run_full_pipeline(
        parquet_path="data/master_df_v3.parquet",
        stores_path="data/dark_stores_final.csv",
        out_dir="outputs",
        data_dir="data",
    )


def run_reverse_vrp():
    from src.reverse_vrp import run_full_pipeline

    run_full_pipeline(
        parquet_path="data/master_df_v3.parquet",
        stores_path="data/dark_stores_final.csv",
        out_dir="outputs",
        data_dir="data",
    )


def run_all_zones_aggregator():
    from all_zones_aggregator import run

    run(
        fwd_path="outputs/forward_kpi_by_zone.csv",
        rev_path="outputs/reverse_kpi_summary.csv",
        out_path="outputs/all_zones_summary.csv",
    )


def run_joint_optimizer():
    """
    SDVRP hybrid all zones + joint Z computation.
    Reads forward + reverse zone dicts from already-solved KPI files.
    """
    import pandas as pd
    from src.joint_optimizer import run_all_zones_sdvrp
    from src.route_parser import build_vrp_nodes, build_reverse_vrp_nodes

    master = pd.read_parquet("data/master_df_v3.parquet")
    dark_stores = pd.read_csv("data/dark_stores_final.csv")
    return_df = master[master["return_flag"] == 1].copy()

    fwd_zones = build_vrp_nodes(master, dark_stores)
    rev_zones = build_reverse_vrp_nodes(return_df, dark_stores)

    fwd_kpi = pd.read_csv("outputs/forward_kpi_summary.csv")
    rev_kpi = pd.read_csv("outputs/reverse_kpi_summary.csv")

    run_all_zones_sdvrp(
        fwd_zones=fwd_zones,
        rev_zones=rev_zones,
        fwd_kpi_df=fwd_kpi,
        rev_kpi_df=rev_kpi,
        num_vehicles=10,
        output_dir="outputs",
    )


def run_scenario_analysis():
    from src.scenario_analysis import run_all_scenarios

    run_all_scenarios(data_dir="data", out_dir="outputs")


# ---------------------------------------------------------------------------
# Stage dispatch map
# ---------------------------------------------------------------------------

RUNNERS = {
    "data_pipeline": run_data_pipeline,
    "demand_baseline": run_demand_baseline,
    "haversine_matrix": run_haversine_matrix,
    "clustering": run_clustering,
    "return_classifier": run_return_classifier,
    "demand_forecasting": run_demand_forecasting,
    "scenario_builder": run_scenario_builder,
    "forward_vrp": run_forward_vrp,
    "reverse_vrp": run_reverse_vrp,
    "all_zones_aggregator": run_all_zones_aggregator,
    "joint_optimizer": run_joint_optimizer,
    "scenario_analysis": run_scenario_analysis,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Dark Store + Integrated Logistics — full pipeline runner"
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="Run a single named stage only",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        default=None,
        help="Resume pipeline from this stage (inclusive)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all stage names in execution order and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nPipeline stages (execution order):")
        for i, s in enumerate(STAGES):
            print(f"  {i:2d}  {s}")
        print()
        return

    # Determine which stages to run
    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        stages_to_run = STAGES[STAGES.index(args.from_stage) :]
    else:
        stages_to_run = STAGES

    print("=" * 60)
    print("  DARK STORE + INTEGRATED LOGISTICS — PIPELINE")
    print(f"  Stages ({len(stages_to_run)}): {' → '.join(stages_to_run)}")
    print("=" * 60)

    total_start = time.time()

    for stage in stages_to_run:
        print(f"\n{'─' * 60}")
        print(f"  STAGE: {stage}")
        print(f"{'─' * 60}")
        t0 = time.time()
        try:
            RUNNERS[stage]()
            print(f"\n  ✓ {stage} completed in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"\n  ✗ {stage} FAILED: {e}")
            print("  Aborting pipeline.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {len(stages_to_run)} stages in {total:.1f}s")
    print(f"  Outputs : outputs/")
    print(f"  Data    : data/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
