"""
Module: haversine_matrix.py
Stage:  Day 2 — Distance Matrix Construction
Owner:  Pritam

INPUT:
    coords : np.ndarray, shape (N, 2)
        Array of [latitude, longitude] pairs in decimal degrees.
        Produced by stratified_sample() from master_df.parquet filtered to SP.

OUTPUT:
    distance_matrix : np.ndarray, shape (N, N), dtype int64
        Pairwise Haversine distances scaled by ×1000 (km stored as int metres).
        OR-Tools requires integer cost matrices — this scaling preserves 3 dp
        of km precision while satisfying that constraint.
    sp_customer_sample.csv : 500 rows with columns
        node_id, customer_lat, customer_lon, customer_zip_code_prefix, order_count
    data/distance_matrix.npy : saved matrix

INTERFACE:
    stratified_sample(df, n, random_state) -> pd.DataFrame
    build_distance_matrix(coords)          -> np.ndarray (N×N, int64)
    save_distance_matrix(matrix, path)     -> None
    load_distance_matrix(path)             -> np.ndarray
    validate_matrix(matrix)               -> dict[str, float|tuple]

ALGORITHM — vectorised Haversine via NumPy broadcasting:
    For N=500, vectorised broadcasting runs in ~10 ms vs ~8 s for a Python loop.
    The formula (all angles in radians):
        a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
        d = 2R · arcsin(√a),   R = 6 371 km

DESIGN NOTES:
    - Row/column order matches vrp_nodes.csv index exactly (node 0 = depot).
    - Diagonal is 0 (self-distance).
    - Matrix is symmetric by construction.
    - Expected SP range: min ~0.5 km, mean ~15 km, max ~60 km.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM: float = 6371.0
SCALE_FACTOR: int = 1000          # km → integer (km × 1000) for OR-Tools
SAMPLE_SIZE: int = 500             # default stratified sample size


# ─────────────────────────────────────────────────────────────────────────────
# 1. STRATIFIED SPATIAL SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Produce a stratified spatial sample of n customer locations from master_df.

    Strategy
    --------
    Group by customer_zip_code_prefix (each zip ≈ a small neighbourhood).
    Compute order volume per zip, then allocate sample slots proportionally.
    Within each zip, draw actual rows (not just the centroid) — this gives
    n distinct real customer coordinates that respect the demand geography
    of São Paulo.

    Deduplication: rows with identical (lat, lon) are removed first so the
    resulting distance matrix has strictly positive off-diagonal entries
    (required by OR-Tools and our validate_matrix check).

    Parameters
    ----------
    df : pd.DataFrame
        master_df filtered to customer_state == 'SP'.
        Must have columns: customer_lat, customer_lon,
        customer_zip_code_prefix, order_id.
    n : int
        Target sample size (default 500).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame with columns:
        node_id, customer_lat, customer_lon,
        customer_zip_code_prefix, order_count
    Indexed 0 … n-1 (node_id == index, used as VRP node index).
    """
    needed = ["customer_lat", "customer_lon", "customer_zip_code_prefix", "order_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"master_df missing columns: {missing}")

    # Drop nulls and deduplicate on exact (lat, lon) — prevents zero off-diagonals
    clean = (
        df.dropna(subset=["customer_lat", "customer_lon"])
        .drop_duplicates(subset=["customer_lat", "customer_lon"])
        .copy()
    )

    # Order-count weight per zip for proportional allocation
    zip_counts = clean.groupby("customer_zip_code_prefix")["order_id"].count()
    clean["zip_order_count"] = clean["customer_zip_code_prefix"].map(zip_counts)

    # Proportional quota per zip, floored to integers summing to n
    total = clean["zip_order_count"].sum()
    zip_df = (
        clean.groupby("customer_zip_code_prefix")
        .agg(zip_order_count=("zip_order_count", "first"))
        .reset_index()
    )
    zip_df["quota"] = (zip_df["zip_order_count"] / total * n).clip(lower=1)
    zip_df["quota_int"] = np.floor(zip_df["quota"]).astype(int)
    remainder = n - zip_df["quota_int"].sum()
    if remainder > 0:
        frac = zip_df["quota"] - zip_df["quota_int"]
        zip_df.loc[frac.nlargest(int(remainder)).index, "quota_int"] += 1

    # Sample actual rows within each zip (not centroids — real customer coords)
    rng = np.random.default_rng(random_state)
    parts: list[pd.DataFrame] = []
    for _, zrow in zip_df[zip_df["quota_int"] > 0].iterrows():
        z = zrow["customer_zip_code_prefix"]
        q = int(zrow["quota_int"])
        pool = clean[clean["customer_zip_code_prefix"] == z]
        drawn = pool.sample(n=min(q, len(pool)),
                            random_state=int(rng.integers(0, 2**31)),
                            replace=False)
        parts.append(drawn[["customer_lat", "customer_lon",
                             "customer_zip_code_prefix", "zip_order_count"]])

    sample_raw = pd.concat(parts, ignore_index=True).head(n)

    sample_df = sample_raw.rename(
        columns={"zip_order_count": "order_count"}
    ).reset_index(drop=True)
    sample_df.index.name = "node_id"
    sample_df = sample_df.reset_index()   # node_id becomes explicit column

    return sample_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTORISED HAVERSINE DISTANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute an N×N Haversine distance matrix, integer-scaled for OR-Tools.

    Uses fully vectorised NumPy broadcasting — no Python loops.
    For N=500: ~10 ms on a modern CPU (vs ~8 s for a double for-loop).

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Columns: [latitude_deg, longitude_deg].

    Returns
    -------
    np.ndarray, shape (N, N), dtype int64
        d[i, j] = Haversine(i, j) in km × SCALE_FACTOR (default 1000).
        Diagonal is 0.  Matrix is symmetric.

    Notes
    -----
    SCALE_FACTOR = 1000 means distances are stored as integer metres.
    1 km → 1000 units, 0.5 km → 500 units, preserving 3 decimal places.
    OR-Tools interprets these raw integers as arc costs.
    """
    coords_rad = np.radians(coords.astype(np.float64))   # (N, 2)
    lat = coords_rad[:, 0]   # (N,)
    lon = coords_rad[:, 1]   # (N,)

    # Broadcasting: (N,1) − (1,N) → (N,N) difference matrices
    dlat = lat[:, None] - lat[None, :]   # (N, N)
    dlon = lon[:, None] - lon[None, :]   # (N, N)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    # clip to [0,1] guards against floating-point noise producing a > 1
    d_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (d_km * SCALE_FACTOR).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_distance_matrix(
    matrix: np.ndarray,
    path: str | Path = "data/distance_matrix.npy",
) -> None:
    """Save integer distance matrix to .npy binary format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def load_distance_matrix(path: str | Path = "data/distance_matrix.npy") -> np.ndarray:
    """Load distance matrix from .npy file.  Returns int64 array."""
    return np.load(Path(path)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_matrix(matrix: np.ndarray) -> dict:
    """
    Compute summary statistics and run sanity assertions.

    Returns
    -------
    dict with keys:
        shape, dtype, min_km, mean_km, max_km,
        is_symmetric, diagonal_zero, all_positive_off_diag
    Raises AssertionError if any check fails.
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], \
        "Matrix must be square 2D array"
    assert matrix.dtype == np.int64, \
        f"Expected int64, got {matrix.dtype}"

    km = matrix.astype(np.float64) / SCALE_FACTOR
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)

    is_symmetric = bool(np.array_equal(matrix, matrix.T))
    diagonal_zero = bool(np.all(np.diag(matrix) == 0))
    all_positive = bool(np.all(matrix[off_diag_mask] > 0))

    assert is_symmetric,     "Matrix is not symmetric"
    assert diagonal_zero,    "Diagonal contains non-zero values"
    assert all_positive,     "Off-diagonal entries contain zeros or negatives"

    off_diag_km = km[off_diag_mask]
    return {
        "shape": matrix.shape,
        "dtype": str(matrix.dtype),
        "min_km": float(off_diag_km.min()),
        "mean_km": float(off_diag_km.mean()),
        "max_km": float(off_diag_km.max()),
        "is_symmetric": is_symmetric,
        "diagonal_zero": diagonal_zero,
        "all_positive_off_diag": all_positive,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    parquet_path: str | Path = "data/master_df.parquet",
    matrix_path: str | Path = "data/distance_matrix.npy",
    sample_csv_path: str | Path = "data/sp_customer_sample.csv",
    n: int = SAMPLE_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Full Day 2 pipeline: load → filter SP → stratified sample → matrix → save.

    Parameters
    ----------
    parquet_path      : path to master_df.parquet
    matrix_path       : output path for distance_matrix.npy
    sample_csv_path   : output path for sp_customer_sample.csv
    n                 : sample size (default 500)
    random_state      : reproducibility seed

    Returns
    -------
    (sample_df, matrix, stats)
        sample_df : pd.DataFrame — 500-row customer sample
        matrix    : np.ndarray  — (500, 500) int64 distance matrix
        stats     : dict        — validation statistics
    """
    print(f"[1/5] Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    print(f"      Total rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print("[2/5] Filtering to customer_state == 'SP' …")
    sp = df[df["customer_state"] == "SP"].copy()
    print(f"      SP rows: {len(sp):,}")

    print(f"[3/5] Stratified spatial sample → {n} representative points …")
    sample_df = stratified_sample(sp, n=n, random_state=random_state)
    print(f"      Sample shape: {sample_df.shape}  |  "
          f"Unique zips: {sample_df['customer_zip_code_prefix'].nunique()}")

    coords = sample_df[["customer_lat", "customer_lon"]].to_numpy()

    print(f"[4/5] Building {n}×{n} Haversine distance matrix (vectorised) …")
    matrix = build_distance_matrix(coords)
    stats = validate_matrix(matrix)
    print(f"      min={stats['min_km']:.2f} km  "
          f"mean={stats['mean_km']:.2f} km  "
          f"max={stats['max_km']:.2f} km  ✓")

    print(f"[5/5] Saving outputs …")
    save_distance_matrix(matrix, matrix_path)
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"      {matrix_path}  ({matrix.nbytes / 1024:.0f} KB)")
    print(f"      {sample_csv_path}  ({len(sample_df)} rows)")
    print("Done.")
    return sample_df, matrix, stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample, mat, stats = run()
    print("\nValidation stats:", stats)
