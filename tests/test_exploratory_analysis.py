"""
Tests for src/exploratory_analysis.py — SP Spatial EDA.

Covers:
    - compute_bounding_box (output keys, lat/lon range)
    - write_spatial_summary (file output, content)
    - save_bounding_box_csv (file output, columns)
    - plot_density_scatter (file creation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.exploratory_analysis import (
    compute_bounding_box,
    write_spatial_summary,
    save_bounding_box_csv,
    plot_density_scatter,
)


# ---------------------------------------------------------------------------
# Fixture — SP sample
# ---------------------------------------------------------------------------


@pytest.fixture
def sp_sample():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "customer_lat": rng.uniform(-23.8, -23.4, size=n),
            "customer_lng": rng.uniform(-46.9, -46.4, size=n),
            "customer_zip_code_prefix": rng.integers(1000, 1200, size=n),
            "order_id": range(n),
        }
    )


# ---------------------------------------------------------------------------
# compute_bounding_box
# ---------------------------------------------------------------------------


class TestComputeBoundingBox:
    def test_returns_dict(self, sp_sample):
        bb = compute_bounding_box(sp_sample)
        assert isinstance(bb, dict)

    def test_required_keys(self, sp_sample):
        bb = compute_bounding_box(sp_sample)
        required = [
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "lat_q1",
            "lat_q3",
            "lon_q1",
            "lon_q3",
            "lat_med",
            "lon_med",
            "n_orders",
        ]
        for key in required:
            assert key in bb, f"Missing key: {key}"

    def test_min_less_than_max(self, sp_sample):
        bb = compute_bounding_box(sp_sample)
        assert bb["lat_min"] < bb["lat_max"]
        assert bb["lon_min"] < bb["lon_max"]

    def test_quartile_ordering(self, sp_sample):
        bb = compute_bounding_box(sp_sample)
        assert bb["lat_q1"] <= bb["lat_med"] <= bb["lat_q3"]
        assert bb["lon_q1"] <= bb["lon_med"] <= bb["lon_q3"]

    def test_n_orders(self, sp_sample):
        bb = compute_bounding_box(sp_sample)
        assert bb["n_orders"] == len(sp_sample)


# ---------------------------------------------------------------------------
# write_spatial_summary
# ---------------------------------------------------------------------------


class TestWriteSpatialSummary:
    def test_writes_file(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        out = tmp_path / "summary.txt"
        write_spatial_summary(sp_sample, bb, out)
        assert out.exists()

    def test_content_mentions_sp(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        out = tmp_path / "summary.txt"
        text = write_spatial_summary(sp_sample, bb, out)
        assert "SP" in text or "Sao Paulo" in text

    def test_returns_string(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        text = write_spatial_summary(sp_sample, bb, tmp_path / "s.txt")
        assert isinstance(text, str)
        assert len(text) > 100


# ---------------------------------------------------------------------------
# save_bounding_box_csv
# ---------------------------------------------------------------------------


class TestSaveBoundingBoxCsv:
    def test_writes_csv(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        out = tmp_path / "bb.csv"
        save_bounding_box_csv(bb, out)
        assert out.exists()

    def test_csv_columns(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        out = tmp_path / "bb.csv"
        save_bounding_box_csv(bb, out)
        df = pd.read_csv(out)
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_csv_row_count(self, sp_sample, tmp_path):
        bb = compute_bounding_box(sp_sample)
        out = tmp_path / "bb.csv"
        save_bounding_box_csv(bb, out)
        df = pd.read_csv(out)
        assert len(df) == 11  # 8 bbox + n_orders + 2 suggested K


# ---------------------------------------------------------------------------
# plot_density_scatter
# ---------------------------------------------------------------------------


class TestPlotDensityScatter:
    def test_creates_png(self, sp_sample, tmp_path):
        out = tmp_path / "scatter.png"
        plot_density_scatter(sp_sample, out)
        assert out.exists()
        assert out.stat().st_size > 0
