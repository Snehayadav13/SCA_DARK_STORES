"""
conftest.py — Shared pytest fixtures for the Dark Store test suite.

Provides synthetic data fixtures that mirror the real Olist schema so every
test module can run without touching disk or real Parquet files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# São Paulo metro bounding box (realistic coordinates)
# ---------------------------------------------------------------------------
SP_LAT_RANGE = (-23.8, -23.4)
SP_LON_RANGE = (-46.9, -46.4)


def _random_sp_coords(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return n random lat/lon pairs inside the SP metro box."""
    lats = rng.uniform(*SP_LAT_RANGE, size=n)
    lons = rng.uniform(*SP_LON_RANGE, size=n)
    return lats, lons


# ---------------------------------------------------------------------------
# master_df (mimics data_pipeline output)
# ---------------------------------------------------------------------------


@pytest.fixture
def master_df() -> pd.DataFrame:
    """
    Synthetic master_df.parquet equivalent — 200 rows, SP-only,
    all columns needed by every downstream module.
    """
    rng = np.random.default_rng(42)
    n = 200
    lats, lons = _random_sp_coords(n, rng)
    seller_lats, seller_lons = _random_sp_coords(n, rng)

    categories = ["electronics", "furniture", "toys", "health_beauty", "sports"]
    payment_types = ["credit_card", "boleto", "debit_card", "voucher"]
    seller_states = ["SP", "MG", "RJ", "PR", "RS"]
    statuses = ["delivered"] * 180 + ["canceled"] * 10 + ["unavailable"] * 10

    base_ts = pd.Timestamp("2018-01-01")
    purchase_ts = [
        base_ts + pd.Timedelta(days=int(d)) for d in rng.integers(0, 365, size=n)
    ]
    delivery_ts = [
        ts + pd.Timedelta(days=int(d))
        for ts, d in zip(purchase_ts, rng.integers(3, 20, size=n))
    ]
    estimated_ts = [
        ts + pd.Timedelta(days=int(d))
        for ts, d in zip(purchase_ts, rng.integers(7, 14, size=n))
    ]

    df = pd.DataFrame(
        {
            "order_id": [f"ord_{i:04d}" for i in range(n)],
            "customer_id": [f"cust_{i:04d}" for i in range(n)],
            "customer_unique_id": [f"cu_{i:04d}" for i in range(n)],
            "customer_lat": lats,
            "customer_lon": lons,
            "customer_zip_code_prefix": rng.integers(1000, 1200, size=n),
            "customer_state": ["SP"] * n,
            "customer_city": rng.choice(["sao paulo", "guarulhos", "osasco"], size=n),
            "seller_lat": seller_lats,
            "seller_lon": seller_lons,
            "seller_state": rng.choice(seller_states, size=n),
            "seller_zip_code_prefix": rng.integers(1000, 1200, size=n),
            "product_id": [f"prod_{i:04d}" for i in range(n)],
            "product_category_name_english": rng.choice(categories, size=n),
            "product_weight_g": rng.integers(100, 5000, size=n).astype(float),
            "price": rng.uniform(10, 500, size=n).round(2),
            "freight_value": rng.uniform(5, 50, size=n).round(2),
            "order_value": rng.uniform(15, 550, size=n).round(2),
            "order_status": rng.choice(statuses, size=n),
            "order_purchase_timestamp": purchase_ts,
            "order_delivered_customer_date": delivery_ts,
            "order_estimated_delivery_date": estimated_ts,
            "delivery_days": rng.integers(3, 20, size=n).astype(float),
            "days_late": rng.uniform(-15, 10, size=n).round(1),
            "review_score": rng.integers(1, 6, size=n).astype(float),
            "payment_type": rng.choice(payment_types, size=n),
            "payment_value": rng.uniform(15, 550, size=n).round(2),
            "n_items": rng.integers(1, 4, size=n),
            "is_return": np.concatenate([np.zeros(180), np.ones(20)]).astype(int),
            "return_rate_by_category": rng.uniform(0.01, 0.15, size=n).round(4),
        }
    )
    return df


# ---------------------------------------------------------------------------
# master_df_v2 (adds demand_per_zip + dark_store_id)
# ---------------------------------------------------------------------------


@pytest.fixture
def master_df_v2(master_df: pd.DataFrame) -> pd.DataFrame:
    """master_df enriched with demand_per_zip and dark_store_id."""
    df = master_df.copy()
    zip_counts = df.groupby("customer_zip_code_prefix")["order_id"].transform("count")
    df["demand_per_zip"] = zip_counts.astype(int)
    df["dark_store_id"] = np.tile(np.arange(5), len(df) // 5 + 1)[: len(df)]
    return df


# ---------------------------------------------------------------------------
# master_df_v3 (adds return_prob + return_flag)
# ---------------------------------------------------------------------------


@pytest.fixture
def master_df_v3(master_df_v2: pd.DataFrame) -> pd.DataFrame:
    """master_df_v2 enriched with return_prob and return_flag."""
    rng = np.random.default_rng(99)
    df = master_df_v2.copy()
    df["return_prob"] = rng.uniform(0, 1, size=len(df)).round(4).astype(np.float32)
    df["return_flag"] = (df["return_prob"] >= 0.30).astype(int)
    return df


# ---------------------------------------------------------------------------
# Dark stores fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dark_stores_df() -> pd.DataFrame:
    """Synthetic dark_stores_final.csv — 5 stores in SP metro."""
    rng = np.random.default_rng(7)
    lats, lons = _random_sp_coords(5, rng)
    return pd.DataFrame(
        {
            "dark_store_id": range(5),
            "lat": lats.round(6),
            "lon": lons.round(6),
            "n_unique_customers": [40, 35, 42, 38, 45],
            "n_orders": [40, 35, 42, 38, 45],
            "total_order_value": [5000.0, 4200.0, 5500.0, 4800.0, 6000.0],
            "capacity_orders": [60] * 5,
            "coverage_5km_pct": [75.0, 72.0, 80.0, 70.0, 78.0],
        }
    )


# ---------------------------------------------------------------------------
# Small coordinate arrays for distance/clustering tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_coords() -> np.ndarray:
    """6 SP-area coordinate pairs — small enough for fast tests."""
    return np.array(
        [
            [-23.55, -46.63],  # central SP
            [-23.50, -46.60],  # ~6 km north
            [-23.60, -46.70],  # ~8 km SW
            [-23.52, -46.65],  # ~4 km NW
            [-23.58, -46.58],  # ~5 km SE
            [-23.48, -46.68],  # ~9 km NW
        ]
    )


# ---------------------------------------------------------------------------
# Synthetic route DataFrames for joint optimizer tests
# ---------------------------------------------------------------------------


@pytest.fixture
def forward_routes_df() -> pd.DataFrame:
    """Synthetic forward_routes.csv — 3 vehicles, depot + stops."""
    rows = []
    for v in range(3):
        for stop in range(5):
            rows.append(
                {
                    "vehicle_id": v,
                    "zone_id": 0,
                    "node_idx": stop,
                    "node_id": "depot" if stop == 0 else f"cust_{v}_{stop}",
                    "lat": -23.55 + stop * 0.01,
                    "lon": -46.63 + stop * 0.01,
                    "cumulative_distance_km": stop * 3.5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def reverse_routes_df() -> pd.DataFrame:
    """Synthetic reverse_routes.csv — 2 vehicles for pickups."""
    rows = []
    for v in range(2):
        for stop in range(4):
            rows.append(
                {
                    "vehicle_id": v,
                    "zone_id": 0,
                    "node_idx": stop,
                    "node_id": "depot" if stop == 0 else f"ret_{v}_{stop}",
                    "lat": -23.55 + stop * 0.005,
                    "lon": -46.63 + stop * 0.005,
                    "cumulative_distance_km": stop * 2.8,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def return_probs() -> pd.Series:
    """Synthetic return probabilities — 50 orders."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(0, 1, size=50).round(4).astype(np.float32))
