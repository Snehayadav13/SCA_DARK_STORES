"""
Module: data_pipeline.py
Stage:  Day 1 — Olist CSV merge  →  master_df.parquet

INPUT:
    data/raw/{filename}.csv  — Olist Brazilian E-Commerce (9 files)
        olist_orders_dataset.csv
        olist_order_items_dataset.csv
        olist_products_dataset.csv
        olist_sellers_dataset.csv
        olist_customers_dataset.csv
        olist_geolocation_dataset.csv
        olist_order_reviews_dataset.csv
        olist_order_payments_dataset.csv
        product_category_name_translation.csv

OUTPUT:
    data/master_df.parquet  — wide-format merged DataFrame
        Key columns:
            order_id, customer_id, customer_lat, customer_lon,
            customer_zip_code_prefix, customer_state, customer_city,
            seller_lat, seller_lon, seller_state,
            product_id, product_category_name_english, price, freight_value,
            order_purchase_timestamp, order_delivered_customer_date,
            review_score, payment_type, payment_value,
            delivery_days (derived), returned (derived, bool)

INTERFACE:
    load_csvs(raw_dir)          -> dict[str, pd.DataFrame]
    clean_geolocation(df)       -> pd.DataFrame   # deduplicate zips → mean lat/lon
    build_master(dfs)           -> pd.DataFrame   # multi-step merge
    derive_features(df)         -> pd.DataFrame   # delivery_days, returned, etc.
    save_master(df, path)       -> None
    run(raw_dir, output_path)   -> pd.DataFrame   # full pipeline in one call
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

REVIEW_RETURN_THRESHOLD = 2  # review_score <= this => treat as returned


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_csvs(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all Olist CSVs from raw_dir into a dict keyed by alias."""
    raw_dir = Path(raw_dir)
    dfs: dict[str, pd.DataFrame] = {}
    for alias, fname in RAW_FILES.items():
        path = raw_dir / fname
        if path.exists():
            dfs[alias] = pd.read_csv(path, low_memory=False)
        else:
            print(f"[WARN] Missing: {path}")
    return dfs


def clean_geolocation(geo_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate geolocation by zip code prefix — take mean lat/lon."""
    return (
        geo_df
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lon=("geolocation_lng", "mean"))
    )


def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all Olist tables into one wide DataFrame.

    Merge order:
        orders ← customers ← geo (customer zip)
               ← items ← products ← translation
                        ← sellers ← geo (seller zip)
               ← reviews (one row per order — latest review if multiple)
               ← payments (one row per order — primary payment)
    """
    geo = clean_geolocation(dfs["geo"])

    # --- customers + geo ---
    customers = dfs["customers"].merge(
        geo.rename(columns={"lat": "customer_lat", "lon": "customer_lon"}),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    # --- sellers + geo ---
    sellers = dfs["sellers"].merge(
        geo.rename(columns={"lat": "seller_lat", "lon": "seller_lon"}),
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    # --- items + products + translation + sellers ---
    translation = dfs["translation"]
    products = dfs["products"].merge(translation, on="product_category_name", how="left")
    items = (
        dfs["items"]
        .merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
        .merge(sellers[["seller_id", "seller_lat", "seller_lon", "seller_state"]], on="seller_id", how="left")
    )
    # Aggregate items per order (sum price/freight, keep first seller)
    items_agg = items.groupby("order_id", as_index=False).agg(
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        product_id=("product_id", "first"),
        product_category_name_english=("product_category_name_english", "first"),
        seller_lat=("seller_lat", "first"),
        seller_lon=("seller_lon", "first"),
        seller_state=("seller_state", "first"),
    )

    # --- reviews (keep latest review per order) ---
    reviews = (
        dfs["reviews"]
        .sort_values("review_answer_timestamp", na_position="last")
        .drop_duplicates("order_id", keep="last")
        [["order_id", "review_score"]]
    )

    # --- payments (primary = highest payment_value) ---
    payments = (
        dfs["payments"]
        .sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id", keep="first")
        [["order_id", "payment_type", "payment_value"]]
    )

    # --- final merge ---
    master = (
        dfs["orders"]
        .merge(customers, on="customer_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
    )
    return master


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: delivery_days (int), returned (bool)."""
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df.get("order_estimated_delivery_date"))

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["returned"] = (df["review_score"] <= REVIEW_RETURN_THRESHOLD).fillna(False)
    return df


def save_master(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[INFO] Saved {len(df):,} rows → {path}")


def run(raw_dir: str | Path = "data/raw", output_path: str | Path = "data/master_df.parquet") -> pd.DataFrame:
    """Full pipeline: load CSVs → merge → derive features → save parquet."""
    dfs = load_csvs(raw_dir)
    master = build_master(dfs)
    master = derive_features(master)
    save_master(master, output_path)
    return master


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = run(raw_dir=raw)
    print(df.shape)
    print(df.columns.tolist())
    print(df.head(2).to_string())
