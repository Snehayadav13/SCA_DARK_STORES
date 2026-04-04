from pathlib import Path
import pandas as pd


def create_scenarios():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    input_path = PROJECT_ROOT / "data" / "vrp_nodes.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"❌ File not found: {input_path}")

    df = pd.read_csv(input_path)

    print("✅ Columns:", df.columns.tolist())

    # ----------------------------------
    # Convert demand from grams → kg
    # ----------------------------------
    df["demand_kg"] = df["demand_g"] / 1000

    # ----------------------------------
    # Scenario A (base)
    # ----------------------------------
    scenario_A = df.copy()

    # ----------------------------------
    # Scenario B (+30% demand)
    # ----------------------------------
    scenario_B = df.copy()
    scenario_B["demand_kg"] *= 1.3

    # ----------------------------------
    # Scenario C (high returns)
    # ----------------------------------
    scenario_C = df.copy()

    # (you don't have return_prob yet — that's fine)
    print("⚠️ No return_prob column yet — Scenario C same as base")

    # ----------------------------------
    # Save
    # ----------------------------------
    scenario_A.to_csv(PROJECT_ROOT / "data" / "vrp_nodes_A.csv", index=False)
    scenario_B.to_csv(PROJECT_ROOT / "data" / "vrp_nodes_B.csv", index=False)
    scenario_C.to_csv(PROJECT_ROOT / "data" / "vrp_nodes_C.csv", index=False)

    print("✅ Scenario files created successfully!")


if __name__ == "__main__":
    create_scenarios()
