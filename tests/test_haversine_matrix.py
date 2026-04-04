"""
Tests for src/haversine_matrix.py — stratified sampling + distance matrix.

Covers:
    - Vectorised Haversine distance matrix (symmetry, diagonal, scaling)
    - Matrix validation function
    - Stratified spatial sampling (quota logic, deduplication)
    - Save/load round-trip
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.haversine_matrix import (
    SCALE_FACTOR,
    build_distance_matrix,
    validate_matrix,
    save_distance_matrix,
    load_distance_matrix,
    stratified_sample,
)


# ---------------------------------------------------------------------------
# build_distance_matrix
# ---------------------------------------------------------------------------


class TestBuildDistanceMatrix:
    def test_shape_is_square(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert mat.shape == (6, 6)

    def test_dtype_is_int64(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert mat.dtype == np.int64

    def test_diagonal_is_zero(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert np.all(np.diag(mat) == 0)

    def test_symmetric(self, small_coords):
        mat = build_distance_matrix(small_coords)
        assert np.array_equal(mat, mat.T)

    def test_off_diagonal_positive(self, small_coords):
        mat = build_distance_matrix(small_coords)
        mask = ~np.eye(6, dtype=bool)
        assert np.all(mat[mask] > 0)

    def test_scale_factor_applied(self, small_coords):
        """
        Central SP to ~6 km north should be ~6000 in integer metres.
        """
        mat = build_distance_matrix(small_coords)
        d_01 = mat[0, 1]
        d_km = d_01 / SCALE_FACTOR
        assert 4 < d_km < 10, f"Expected ~6 km, got {d_km:.1f} km"

    def test_triangle_inequality(self, small_coords):
        mat = build_distance_matrix(small_coords)
        n = len(small_coords)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert mat[i, j] <= mat[i, k] + mat[k, j] + 1  # +1 for int rounding

    def test_single_point(self):
        coords = np.array([[-23.55, -46.63]])
        mat = build_distance_matrix(coords)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0

    def test_two_points(self):
        coords = np.array([[-23.55, -46.63], [-23.50, -46.60]])
        mat = build_distance_matrix(coords)
        assert mat.shape == (2, 2)
        assert mat[0, 1] == mat[1, 0]
        assert mat[0, 1] > 0


# ---------------------------------------------------------------------------
# validate_matrix
# ---------------------------------------------------------------------------


class TestValidateMatrix:
    def test_valid_matrix(self, small_coords):
        mat = build_distance_matrix(small_coords)
        stats = validate_matrix(mat)
        assert stats["is_symmetric"] is True
        assert stats["diagonal_zero"] is True
        assert stats["all_positive_off_diag"] is True
        assert stats["min_km"] > 0
        assert stats["max_km"] < 1000  # SP metro < 50 km across

    def test_non_square_raises(self):
        mat = np.ones((3, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="square"):
            validate_matrix(mat)

    def test_wrong_dtype_raises(self):
        mat = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="int64"):
            validate_matrix(mat)

    def test_non_symmetric_raises(self):
        mat = np.array([[0, 100, 200], [100, 0, 300], [200, 301, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="symmetric"):
            validate_matrix(mat)

    def test_nonzero_diagonal_raises(self):
        mat = np.array([[1, 100], [100, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="Diagonal"):
            validate_matrix(mat)

    def test_zero_off_diagonal_raises(self):
        mat = np.array([[0, 0], [0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="zeros"):
            validate_matrix(mat)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip(self, small_coords, tmp_path):
        mat = build_distance_matrix(small_coords)
        path = tmp_path / "test_matrix.npy"
        save_distance_matrix(mat, path)
        loaded = load_distance_matrix(path)
        assert np.array_equal(mat, loaded)
        assert loaded.dtype == np.int64


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_output_size(self, master_df):
        sample = stratified_sample(master_df, n=50)
        assert len(sample) == 50

    def test_output_columns(self, master_df):
        sample = stratified_sample(master_df, n=50)
        assert "node_id" in sample.columns
        assert "customer_lat" in sample.columns
        assert "customer_lon" in sample.columns
        assert "customer_zip_code_prefix" in sample.columns
        assert "order_count" in sample.columns

    def test_no_duplicate_coords(self, master_df):
        sample = stratified_sample(master_df, n=50)
        deduped = sample.drop_duplicates(subset=["customer_lat", "customer_lon"])
        assert len(deduped) == len(sample)

    def test_reproducible(self, master_df):
        s1 = stratified_sample(master_df, n=50, random_state=42)
        s2 = stratified_sample(master_df, n=50, random_state=42)
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seeds_differ(self, master_df):
        s1 = stratified_sample(master_df, n=50, random_state=42)
        s2 = stratified_sample(master_df, n=50, random_state=99)
        assert not s1["customer_lat"].equals(s2["customer_lat"])

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            stratified_sample(df, n=1)

    def test_sample_too_large_raises(self, master_df):
        large_n = len(master_df) + 1000
        with pytest.raises(ValueError):
            stratified_sample(master_df, n=large_n)
