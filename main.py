"""
main.py — Full pipeline entry point for Dark Store + Integrated Logistics

Runs all stages in dependency order:
    Stage 0: Data pipeline        → data/master_df.parquet
    Stage 1: Demand baseline      → data/master_df_v2.parquet + baseline KPIs
    Stage 2: Clustering           → data/dark_stores_final.csv + master_df_v2 updated
    Stage 3: Return classifier    → data/master_df_v3.parquet
    Stage 4: Forward VRP          → outputs/forward_routes.json + KPIs
    Stage 5: Reverse VRP          → outputs/reverse_routes.json + KPIs
    Stage 6: Joint optimizer      → outputs/joint_optimizer_result.json  [Day 5-6]
    Stage 7: Demand forecasting   → outputs/forecasted_demand_by_zone.csv [Anurag]

Usage:
    python main.py                  # run all stages
    python main.py --from clustering  # resume from a specific stage
    python main.py --stage forward_vrp  # run one stage only

All paths are relative to project root.
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
    "data_pipeline",
    "demand_baseline",
    "clustering",
    "return_classifier",
    "forward_vrp",
    "reverse_vrp",
    "joint_optimizer",
    "demand_forecasting",
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


def run_joint_optimizer():
    import pandas as pd
    from src.joint_optimizer import run
    fwd = pd.read_csv("outputs/forward_routes.csv")
    rev = pd.read_csv("outputs/reverse_routes.csv")
    probs = pd.read_parquet("data/master_df_v3.parquet")["return_prob"]
    run(
        forward_routes_df=fwd,
        reverse_routes_df=rev,
        return_probs=probs,
        output_path="outputs/joint_optimizer_result.json",
    )


def run_demand_forecasting():
    from src.demand_forecasting import run_pipeline
    run_pipeline(input_path="data/master_df_v2.parquet")


# ---------------------------------------------------------------------------
# Stage dispatch map
# ---------------------------------------------------------------------------

RUNNERS = {
    "data_pipeline":      run_data_pipeline,
    "demand_baseline":    run_demand_baseline,
    "clustering":         run_clustering,
    "return_classifier":  run_return_classifier,
    "forward_vrp":        run_forward_vrp,
    "reverse_vrp":        run_reverse_vrp,
    "joint_optimizer":    run_joint_optimizer,
    "demand_forecasting": run_demand_forecasting,
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
        help="Run a single stage only",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        default=None,
        help="Resume pipeline from this stage (inclusive)",
    )
    args = parser.parse_args()

    # Determine which stages to run
    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        start_idx = STAGES.index(args.from_stage)
        stages_to_run = STAGES[start_idx:]
    else:
        stages_to_run = STAGES

    print("=" * 60)
    print("  DARK STORE + INTEGRATED LOGISTICS — PIPELINE")
    print(f"  Stages: {' → '.join(stages_to_run)}")
    print("=" * 60)

    total_start = time.time()

    for stage in stages_to_run:
        print(f"\n{'─' * 60}")
        print(f"  STAGE: {stage}")
        print(f"{'─' * 60}")
        t0 = time.time()
        try:
            RUNNERS[stage]()
            elapsed = time.time() - t0
            print(f"\n  ✓ {stage} completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n  ✗ {stage} FAILED: {e}")
            print("  Aborting pipeline.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {len(stages_to_run)} stages in {total:.1f}s")
    print(f"  Outputs: outputs/  |  Data: data/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    