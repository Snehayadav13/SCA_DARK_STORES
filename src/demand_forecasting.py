"""
Module: demand_forecasting.py
Stage:  Temporal Demand Forecasting

DEPENDS ON:
    data/master_df_v2.parquet   — enriched dataset with dark_store_id

OUTPUT:
    outputs/forecasted_demand_by_zone.csv   — 4-week forecast per zone
    outputs/zone_total_demand.csv           — total predicted demand per zone
    data/zone_total_demand.csv              — (for VRP module)

PUBLIC INTERFACE:
    run_pipeline(input_path) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from prophet import Prophet


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[load_data] Loaded {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["order_date"] = pd.to_datetime(df["order_purchase_timestamp"])
    df = df.dropna(subset=["dark_store_id"])
    df["dark_store_id"] = df["dark_store_id"].astype(int)

    print(f"[preprocess] {df['dark_store_id'].nunique()} zones available")
    return df


# ---------------------------------------------------------------------------
# 3. Weekly Aggregation
# ---------------------------------------------------------------------------

def create_weekly_demand(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df
        .groupby([
            "dark_store_id",
            pd.Grouper(key="order_date", freq="W")
        ])
        .size()
        .reset_index(name="orders")
    )

    print(f"[weekly] Generated {len(weekly):,} rows")
    return weekly


# ---------------------------------------------------------------------------
# 4. Forecast per Zone
# ---------------------------------------------------------------------------

def forecast_demand(weekly_df: pd.DataFrame) -> pd.DataFrame:
    all_forecasts = []

    zones = weekly_df["dark_store_id"].unique()

    for zone in zones:
        zone_df = weekly_df[weekly_df["dark_store_id"] == zone].copy()

        zone_df = zone_df.rename(columns={
            "order_date": "ds",
            "orders": "y"
        })

        if len(zone_df) < 10:
            print(f"[forecast] Skipping zone {zone} (insufficient data)")
            continue

        try:
            model = Prophet(weekly_seasonality=True)
            model.fit(zone_df)

            future = model.make_future_dataframe(periods=4, freq="W")
            forecast = model.predict(future)

            forecast["dark_store_id"] = zone

            all_forecasts.append(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "dark_store_id"]]
            )

        except Exception as e:
            print(f"[forecast] Error in zone {zone}: {e}")

    if not all_forecasts:
        raise ValueError("No forecasts generated — check your data!")

    result = pd.concat(all_forecasts, ignore_index=True)
    print(f"[forecast] Final shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# 5. Save Outputs
# ---------------------------------------------------------------------------

def save_outputs(forecast_df: pd.DataFrame, out_dir: str | Path = "outputs") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full forecast
    forecast_path = out_dir / "forecasted_demand_by_zone.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Aggregate demand per zone
    zone_total = (
        forecast_df
        .groupby("dark_store_id")["yhat"]
        .sum()
        .reset_index()
    )

    # Save in outputs (for reporting)
    zone_total_outputs = out_dir / "zone_total_demand.csv"
    zone_total.to_csv(zone_total_outputs, index=False)

    # ALSO save in data (for VRP usage)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    zone_total_data = data_dir / "zone_total_demand.csv"
    zone_total.to_csv(zone_total_data, index=False)

    print(f"[save] Forecast → {forecast_path}")
    print(f"[save] Zone totals (outputs) → {zone_total_outputs}")
    print(f"[save] Zone totals (data) → {zone_total_data}")


# ---------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str | Path = "data/master_df_v2.parquet",
) -> pd.DataFrame:

    print("=" * 60)
    print(" DEMAND FORECASTING")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data(input_path)

    print("\n[2/4] Preprocessing...")
    df = preprocess(df)

    print("\n[3/4] Aggregating weekly demand...")
    weekly = create_weekly_demand(df)

    print("\n[4/4] Running forecasting...")
    forecast_df = forecast_demand(weekly)

    print("\nSaving outputs...")
    save_outputs(forecast_df)

    print("\n" + "=" * 60)
    print("  FORECASTING COMPLETE")
    print("=" * 60)

    return forecast_df


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()