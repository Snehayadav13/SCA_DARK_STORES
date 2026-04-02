"""
Module: data_pipeline.py
Stage:  Olist CSV merge  →  master_df.parquet

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
    data/master_df.parquet  — wide-format merged DataFrame filtered to SP
        Key columns:
            order_id, customer_id, customer_lat, customer_lon,
            customer_zip_code_prefix, customer_state, customer_city,
            seller_lat, seller_lon, seller_state,
            product_id, product_category_name_english,
            product_weight_g, price, freight_value, order_value,
            order_purchase_timestamp, order_delivered_customer_date,
            order_estimated_delivery_date,
            delivery_days, days_late, review_score,
            payment_type, payment_value,
            is_return (bool), return_rate_by_category (float)

INTERFACE:
    load_csvs(raw_dir)          -> dict[str, pd.DataFrame]
    clean_geolocation(df)       -> pd.DataFrame   # deduplicate zips → median lat/lon
    build_master(dfs)           -> pd.DataFrame   # multi-step merge
    derive_features(df)         -> pd.DataFrame   # delivery_days, is_return, etc.
    save_master(df, path)       -> None
    run(raw_dir, output_path)   -> pd.DataFrame   # full pipeline in one call
"""

from __future__ import annotations

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

# Days beyond estimated delivery that we treat as a "return" trigger
LATE_DELIVERY_THRESHOLD_DAYS = 7

# Statuses that count as returns
RETURN_STATUSES = {"canceled", "unavailable"}


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
            print(f"  [OK] {alias:>12s} — {len(dfs[alias]):>8,} rows  ← {fname}")
        else:
            print(f"  [WARN] Missing: {path}")
    return dfs


def clean_geolocation(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate geolocation by zip code prefix — take **median** lat/lon.
    (Session summary specifies median to reduce outlier influence.)
    """
    return (
        geo_df
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median"))
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
    print(f"\n  Geolocation deduplicated: {len(dfs['geo']):,} → {len(geo):,} zip prefixes")

    # --- customers + geo ---
    customers = dfs["customers"].merge(
        geo.rename(columns={"lat": "customer_lat", "lon": "customer_lon"}),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).drop(columns=["geolocation_zip_code_prefix"], errors="ignore")

    # --- sellers + geo ---
    sellers = dfs["sellers"].merge(
        geo.rename(columns={"lat": "seller_lat", "lon": "seller_lon"}),
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).drop(columns=["geolocation_zip_code_prefix"], errors="ignore")

    # --- products + translation ---
    translation = dfs["translation"]
    products = dfs["products"].merge(translation, on="product_category_name", how="left")

    # --- items + products + sellers ---
    items = (
        dfs["items"]
        .merge(
            products[["product_id", "product_category_name_english",
                       "product_weight_g", "product_length_cm",
                       "product_height_cm", "product_width_cm"]],
            on="product_id", how="left",
        )
        .merge(
            sellers[["seller_id", "seller_lat", "seller_lon",
                     "seller_state", "seller_city", "seller_zip_code_prefix"]],
            on="seller_id", how="left",
        )
    )

    # Aggregate items per order
    # - sum price, freight, weight
    # - keep first product category, seller info (most orders are single-item)
    items_agg = items.groupby("order_id", as_index=False).agg(
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        product_weight_g=("product_weight_g", "sum"),
        n_items=("product_id", "count"),
        product_id=("product_id", "first"),
        product_category_name_english=("product_category_name_english", "first"),
        seller_lat=("seller_lat", "first"),
        seller_lon=("seller_lon", "first"),
        seller_state=("seller_state", "first"),
        seller_city=("seller_city", "first"),
        seller_zip_code_prefix=("seller_zip_code_prefix", "first"),
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

    print(f"  Master merged: {len(master):,} rows × {master.shape[1]} cols")
    return master


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns needed by all downstream modules.

    Columns added:
        - delivery_days (int): actual delivery - purchase date
        - days_late (float): actual delivery - estimated delivery (positive = late)
        - order_value (float): price + freight_value
        - is_return (int 0/1): order_status in {canceled, unavailable}
                                OR delivered > estimated + 7 days
        - return_rate_by_category (float): category-level return rate
    """
    df = df.copy()

    # --- Parse timestamps ---
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- delivery_days: purchase → actual delivery ---
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # --- days_late: actual delivery − estimated delivery (positive = late) ---
    df["days_late"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # --- order_value ---
    df["order_value"] = df["price"].fillna(0) + df["freight_value"].fillna(0)

    # --- is_return (per session summary definition) ---
    # 1 if order_status ∈ {canceled, unavailable}
    # OR actual delivery > estimated delivery + 7 days
    status_return = df["order_status"].isin(RETURN_STATUSES)
    late_return = df["days_late"] > LATE_DELIVERY_THRESHOLD_DAYS
    df["is_return"] = (status_return | late_return.fillna(False)).astype(int)

    # --- return_rate_by_category ---
    cat_return = (
        df.groupby("product_category_name_english")["is_return"]
        .mean()
        .rename("return_rate_by_category")
    )
    df = df.merge(cat_return, on="product_category_name_english", how="left")

    # --- Fill NaN product_weight_g with median (needed for VRP capacity) ---
    median_weight = df["product_weight_g"].median()
    df["product_weight_g"] = df["product_weight_g"].fillna(median_weight)

    return df


def filter_sp(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to São Paulo state customers only (~41% of orders)."""
    sp = df[df["customer_state"] == "SP"].copy()
    print(f"  SP filter: {len(df):,} → {len(sp):,} rows ({len(sp)/len(df)*100:.1f}%)")
    return sp


def drop_no_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where customer lat/lon is missing (can't route without coords)."""
    before = len(df)
    df = df.dropna(subset=["customer_lat", "customer_lon"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing customer coordinates")
    return df


def save_master(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\n  [SAVED] {len(df):,} rows → {path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print key stats for validation."""
    print("\n" + "=" * 60)
    print("  MASTER_DF SUMMARY")
    print("=" * 60)
    print(f"  Shape:               {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Customer state:      {df['customer_state'].unique().tolist()}")
    print(f"  Unique orders:       {df['order_id'].nunique():,}")
    print(f"  Unique customers:    {df['customer_id'].nunique():,}")
    print(f"  Date range:          {df['order_purchase_timestamp'].min()} → "
          f"{df['order_purchase_timestamp'].max()}")
    print(f"  Null customer_lat:   {df['customer_lat'].isna().sum():,}")
    print(f"  Null seller_lat:     {df['seller_lat'].isna().sum():,}")
    print(f"  Lat range:           [{df['customer_lat'].min():.4f}, {df['customer_lat'].max():.4f}]")
    print(f"  Lon range:           [{df['customer_lon'].min():.4f}, {df['customer_lon'].max():.4f}]")
    print(f"  ---")
    print(f"  Return rate:         {df['is_return'].mean()*100:.2f}%")
    print(f"    - status-based:    {df['order_status'].isin(RETURN_STATUSES).sum():,}")
    print(f"    - late (>7d):      {(df['days_late'] > LATE_DELIVERY_THRESHOLD_DAYS).sum():,}")
    print(f"  Avg delivery_days:   {df['delivery_days'].mean():.1f}")
    print(f"  Avg days_late:       {df['days_late'].mean():.1f}")
    print(f"  Avg order_value:     R$ {df['order_value'].mean():.2f}")
    print(f"  Avg product_weight:  {df['product_weight_g'].mean():.0f} g")
    print(f"  Top 5 categories:")
    top5 = df["product_category_name_english"].value_counts().head(5)
    for cat, cnt in top5.items():
        print(f"    {cat:>30s}  {cnt:>6,}")
    print(f"  Order statuses:")
    for st, cnt in df["order_status"].value_counts().items():
        print(f"    {st:>20s}  {cnt:>6,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/master_df.parquet",
    filter_state: str = "SP",
) -> pd.DataFrame:
    """
    Full pipeline: load CSVs → merge → derive features → filter SP → save parquet.

    Parameters
    ----------
    raw_dir      : path to folder with 9 Olist CSVs
    output_path  : where to save the final parquet
    filter_state : Brazilian state code to filter (default 'SP' for São Paulo)
    """
    print("=" * 60)
    print("  DATA PIPELINE — Olist → master_df.parquet")
    print("=" * 60)

    print("\n[1/5] Loading CSVs...")
    dfs = load_csvs(raw_dir)
    if len(dfs) < 9:
        print(f"  WARNING: Only {len(dfs)}/9 files loaded. Some joins may fail.")

    print("\n[2/5] Merging tables...")
    master = build_master(dfs)

    print("\n[3/5] Deriving features...")
    master = derive_features(master)

    print(f"\n[4/5] Filtering to {filter_state}...")
    master = filter_sp(master)
    master = drop_no_coords(master)

    print("\n[5/5] Saving...")
    save_master(master, output_path)

    print_summary(master)
    return master


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/master_df.parquet"
    run(raw_dir=raw, output_path=out)
