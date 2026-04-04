"""
Tests for Full E2E Integration — testing pipeline stage connectivity.

Spins up dummy raw data, triggers each stage in exact sequence as main.py,
and verifies output paths and formats drop cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Main runner imports
from src import (
    data_pipeline,
    demand_baseline,
    clustering,
    return_classifier,
    route_parser,
    forward_vrp,
    reverse_vrp,
    joint_optimizer,
)

# ---------------------------------------------------------------------------
# Integration Setup
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_raw_data_dir(tmp_path):
    """Generates 5-row toy datasets in tmp_path to mimic the Olist structure."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n = 20

    # 1. orders
    statuses = ["delivered"] * 18 + ["canceled"] * 2
    orders = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(n)],
            "customer_id": [f"cust_{i}" for i in range(n)],
            "order_status": statuses,
            "order_purchase_timestamp": [pd.Timestamp("2018-01-01")] * n,
            "order_delivered_customer_date": [pd.Timestamp("2018-01-10")] * n,
            "order_estimated_delivery_date": [pd.Timestamp("2018-01-15")] * n,
        }
    )
    orders.to_csv(raw_dir / "olist_orders_dataset.csv", index=False)

    # 2. order_items
    items = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(n)],
            "product_id": [f"prod_{i}" for i in range(n)],
            "seller_id": [f"seller_{i}" for i in range(n)],
            "price": [10.0] * n,
            "freight_value": [5.0] * n,
        }
    )
    items.to_csv(raw_dir / "olist_order_items_dataset.csv", index=False)

    # 3. products
    products = pd.DataFrame(
        {
            "product_id": [f"prod_{i}" for i in range(n)],
            "product_category_name": ["electronics"] * n,
            "product_weight_g": [500.0] * n,
            "product_length_cm": [10.0] * n,
            "product_height_cm": [10.0] * n,
            "product_width_cm": [10.0] * n,
        }
    )
    products.to_csv(raw_dir / "olist_products_dataset.csv", index=False)

    # 4. customers
    customers = pd.DataFrame(
        {
            "customer_id": [f"cust_{i}" for i in range(n)],
            "customer_unique_id": [f"cu_{i}" for i in range(n)],
            "customer_zip_code_prefix": rng.integers(1000, 1005, size=n),
            "customer_state": ["SP"] * n,
            "customer_city": ["sao paulo"] * n,
        }
    )
    customers.to_csv(raw_dir / "olist_customers_dataset.csv", index=False)

    # 5. sellers
    sellers = pd.DataFrame(
        {
            "seller_id": [f"seller_{i}" for i in range(n)],
            "seller_zip_code_prefix": rng.integers(1000, 1005, size=n),
            "seller_state": ["SP"] * n,
            "seller_city": ["sao paulo"] * n,
        }
    )
    sellers.to_csv(raw_dir / "olist_sellers_dataset.csv", index=False)

    # 6. geolocation (need enough matches for the zips we randomly generated)
    unique_zips = pd.concat(
        [customers["customer_zip_code_prefix"], sellers["seller_zip_code_prefix"]]
    ).unique()
    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": unique_zips,
            "geolocation_lat": [-23.55 - (i * 0.01) for i in range(len(unique_zips))],
            "geolocation_lng": [-46.63 - (i * 0.01) for i in range(len(unique_zips))],
        }
    )
    geo.to_csv(raw_dir / "olist_geolocation_dataset.csv", index=False)

    # 7. reviews
    reviews = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(n)],
            "review_score": [5] * n,
            "review_creation_date": [pd.Timestamp("2018-01-11")] * n,
            "review_answer_timestamp": [pd.Timestamp("2018-01-12")] * n,
        }
    )
    reviews.to_csv(raw_dir / "olist_order_reviews_dataset.csv", index=False)

    # 8. payments
    payments = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(n)],
            "payment_type": ["credit_card"] * n,
            "payment_value": [15.0] * n,
        }
    )
    payments.to_csv(raw_dir / "olist_order_payments_dataset.csv", index=False)

    # 9. category translation
    translation = pd.DataFrame(
        {
            "product_category_name": ["electronics"],
            "product_category_name_english": ["electronics"],
        }
    )
    translation.to_csv(raw_dir / "product_category_name_translation.csv", index=False)

    return raw_dir


# ---------------------------------------------------------------------------
# Test E2E
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    @pytest.fixture(autouse=True)
    def patch_solver_time(self, monkeypatch):
        """Force the solvers to exit instantly for testing."""
        monkeypatch.setattr("src.forward_vrp.SOLVER_TIME_LIMIT_S", 1)
        monkeypatch.setattr("src.reverse_vrp.SOLVER_TIME_LIMIT_S", 1)

    def test_full_pipeline_end_to_end(self, mock_raw_data_dir, tmp_path):
        """
        Runs the exact pipeline steps as defined in main.py but points targets
        output and data directories to tmp_path to prove format compatibility.
        """
        raw_dir = mock_raw_data_dir
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "outputs"

        data_dir.mkdir()
        out_dir.mkdir()

        # [Stage 0]: Data pipeline
        data_pipeline.run(raw_dir=raw_dir, output_path=data_dir / "master_df.parquet")
        assert (data_dir / "master_df.parquet").exists()

        # [Stage 1]: Demand Baseline
        demand_baseline.run(
            input_path=data_dir / "master_df.parquet", output_dir=data_dir
        )
        assert (data_dir / "master_df_v2.parquet").exists()

        # [Stage 2]: Clustering
        # We limit K to very small numbers so coverage logic computes cleanly.
        clustering.run_full_pipeline(
            parquet_path=data_dir / "master_df_v2.parquet",
            out_dir=data_dir,
            plot_dir=out_dir,
            k_range=range(2, 4),
        )
        assert (data_dir / "dark_stores_final.csv").exists()

        # [Stage 3]: Return Classifier
        from src.return_classifier import TARGET

        df_v2 = pd.read_parquet(data_dir / "master_df_v2.parquet")
        # Ensure balanced target for 3-fold cross validation
        df_v2[TARGET] = np.random.default_rng(42).integers(0, 2, size=len(df_v2))
        df_v2.to_parquet(data_dir / "master_df_v2.parquet", index=False)

        return_classifier.run_full_pipeline(
            parquet_path=data_dir / "master_df_v2.parquet",
            out_dir=out_dir,
            data_dir=data_dir,
        )
        assert (data_dir / "master_df_v3.parquet").exists()

        # Check return flags were successfully created
        df_v3 = pd.read_parquet(data_dir / "master_df_v3.parquet")
        assert "return_prob" in df_v3.columns
        assert "return_flag" in df_v3.columns

        # [Stage 4]: Forward VRP
        forward_vrp.run_full_pipeline(
            parquet_path=data_dir / "master_df_v3.parquet",
            stores_path=data_dir / "dark_stores_final.csv",
            out_dir=out_dir,
            data_dir=data_dir,
        )
        assert (out_dir / "forward_routes.csv").exists()

        # [Stage 5]: Reverse VRP
        reverse_vrp.run_full_pipeline(
            parquet_path=data_dir / "master_df_v3.parquet",
            stores_path=data_dir / "dark_stores_final.csv",
            out_dir=out_dir,
            data_dir=data_dir,
        )
        assert (out_dir / "reverse_routes.csv").exists()

        # [Stage 6]: Joint Optimizer
        # If returns existed, we run joint opt
        fwd_routes = pd.read_csv(out_dir / "forward_routes.csv")
        rev_routes = pd.read_csv(out_dir / "reverse_routes.csv")

        if not fwd_routes.empty and not rev_routes.empty:
            joint_optimizer.run(
                forward_routes_df=fwd_routes,
                reverse_routes_df=rev_routes,
                return_probs=df_v3["return_prob"],
                output_path=out_dir / "joint_optimizer_result.json",
            )
            assert (out_dir / "joint_optimizer_result.json").exists()
