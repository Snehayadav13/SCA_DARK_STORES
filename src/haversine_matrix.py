"""
Module: haversine_matrix.py
Stage:  Day 2 — Distance Matrix Construction

INPUT:
    coords : np.ndarray, shape (N, 2)
        Array of [latitude, longitude] pairs in decimal degrees.
        Typically sp_customer_sample.csv (500 stratified SP customers).

OUTPUT:
    distance_matrix : np.ndarray, shape (N, N), dtype int64
        Pairwise Haversine distances scaled by x1000 (metres as integers).
        Required by OR-Tools which only accepts integer cost matrices.
    Saved to: data/distance_matrix.npy

INTERFACE NOTES:
    - Row/column order matches vrp_nodes.csv index order exactly.
    - Call: build_distance_matrix(coords) -> np.ndarray
    - Call: save_distance_matrix(matrix, path) -> None
    - Call: validate_matrix(matrix) -> dict with min/mean/max km
      Expected range for SP: 0.5 km to 60 km.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


EARTH_RADIUS_KM = 6371.0
SCALE_FACTOR = 1000  # km -> integer metres for OR-Tools


def _haversine_element(u: np.ndarray, v: np.ndarray) -> float:
    """Haversine distance in km between two (lat, lon) points in radians."""
    diff = v - u
    a = np.sin(diff[0] / 2) ** 2 + np.cos(u[0]) * np.cos(v[0]) * np.sin(diff[1] / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Haversine distance matrix and scale to integers.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Columns: [latitude_deg, longitude_deg]

    Returns
    -------
    np.ndarray, shape (N, N), dtype int64
        Integer-scaled distances (km * 1000).
    """
    coords_rad = np.radians(coords)
    n = len(coords_rad)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_element(coords_rad[i], coords_rad[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return (matrix * SCALE_FACTOR).astype(np.int64)


def save_distance_matrix(matrix: np.ndarray, path: str = "data/distance_matrix.npy") -> None:
    np.save(path, matrix)


def validate_matrix(matrix: np.ndarray) -> dict:
    km = matrix.astype(np.float64) / SCALE_FACTOR
    mask = km > 0
    return {
        "min_km": float(km[mask].min()),
        "mean_km": float(km[mask].mean()),
        "max_km": float(km[mask].max()),
        "shape": matrix.shape,
    }


if __name__ == "__main__":
    # Smoke test with 5 SP points
    test_coords = np.array([
        [-23.5505, -46.6333],
        [-23.5489, -46.6388],
        [-23.5612, -46.6560],
        [-23.5200, -46.6100],
        [-23.5800, -46.6700],
    ])
    mat = build_distance_matrix(test_coords)
    stats = validate_matrix(mat)
    print("Distance matrix smoke test:", stats)
