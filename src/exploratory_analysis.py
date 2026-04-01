"""
Module: exploratory_analysis.py
Stage:  Day 1-2 — SP Spatial EDA (Anurag)

INPUT:
    data/raw/  — olist_orders_dataset.csv, olist_customers_dataset.csv,
                 olist_geolocation_dataset.csv, olist_sellers_dataset.csv

OUTPUT:
    outputs/sp_density_scatter.png   — Customer density scatter plot
    outputs/sp_spatial_summary.txt   — 5-bullet spatial insight report
    outputs/sp_bounding_box.csv      — Bounding box coordinates
    visualisations/sp_eda_map.html   — Interactive Folium map (Day 2)

INTERFACE:
    load_sp_sample(raw_dir)                        -> pd.DataFrame
    compute_bounding_box(sp_sample)                -> dict
    plot_density_scatter(sp_sample, out)           -> None
    write_spatial_summary(sp_sample, bb, out)      -> str
    save_bounding_box_csv(bb, out)                 -> None
    build_folium_map(sp_sample, sellers_sp, out)   -> folium.Map
    run(raw_dir, output_dir, vis_dir)              -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR    = Path("data/raw")
OUTPUT_DIR = Path("outputs")
VIS_DIR    = Path("visualisations")


# ---------------------------------------------------------------------------
# Step 1 — Load & merge (orders + customers + geolocation) filtered to SP
# ---------------------------------------------------------------------------

def load_sp_sample(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """
    Load and merge orders, customers, geolocation for SP state.

    Returns
    -------
    pd.DataFrame with columns:
        order_id, customer_id, customer_lat, customer_lng,
        customer_zip_code_prefix, customer_state, order_status, ...
    """
    raw_dir = Path(raw_dir)

    orders    = pd.read_csv(raw_dir / "olist_orders_dataset.csv",     low_memory=False)
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv",  low_memory=False)
    geo       = pd.read_csv(raw_dir / "olist_geolocation_dataset.csv", low_memory=False)

    # Geolocation: median lat/lon per zip code prefix (removes noise / outliers)
    geo_clean = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        })
    )

    # Filter to SP, then attach lat/lon
    sp_customers = customers[customers["customer_state"] == "SP"].copy()
    sp_customers = sp_customers.merge(geo_clean, on="customer_zip_code_prefix", how="left")

    # Merge with orders
    sp_sample = orders.merge(sp_customers, on="customer_id", how="inner")
    sp_sample = sp_sample.dropna(subset=["customer_lat", "customer_lng"])

    print(f"[INFO] SP sample shape: {sp_sample.shape}")
    return sp_sample


# ---------------------------------------------------------------------------
# Step 2 — Load SP sellers (needed for Folium map)
# ---------------------------------------------------------------------------

def load_sp_sellers(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """
    Load sellers filtered to SP state with lat/lon attached.

    Returns
    -------
    pd.DataFrame with columns: seller_id, seller_lat, seller_lng, ...
    """
    raw_dir = Path(raw_dir)

    sellers = pd.read_csv(raw_dir / "olist_sellers_dataset.csv",      low_memory=False)
    geo     = pd.read_csv(raw_dir / "olist_geolocation_dataset.csv",  low_memory=False)

    geo_sell = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        })
    )

    sellers_sp = (
        sellers[sellers["seller_state"] == "SP"]
        .merge(geo_sell, on="seller_zip_code_prefix", how="left")
        .dropna(subset=["seller_lat", "seller_lng"])
    )

    print(f"[INFO] SP sellers shape: {sellers_sp.shape}")
    return sellers_sp


# ---------------------------------------------------------------------------
# Step 3 — Bounding box
# ---------------------------------------------------------------------------

def compute_bounding_box(sp_sample: pd.DataFrame) -> dict:
    """Return full-extent and IQR bounding box coordinates."""
    return {
        "lat_min":  float(sp_sample["customer_lat"].min()),
        "lat_max":  float(sp_sample["customer_lat"].max()),
        "lon_min":  float(sp_sample["customer_lng"].min()),
        "lon_max":  float(sp_sample["customer_lng"].max()),
        "lat_q1":   float(sp_sample["customer_lat"].quantile(0.25)),
        "lat_q3":   float(sp_sample["customer_lat"].quantile(0.75)),
        "lon_q1":   float(sp_sample["customer_lng"].quantile(0.25)),
        "lon_q3":   float(sp_sample["customer_lng"].quantile(0.75)),
        "lat_med":  float(sp_sample["customer_lat"].median()),
        "lon_med":  float(sp_sample["customer_lng"].median()),
        "n_orders": int(len(sp_sample)),
    }


# ---------------------------------------------------------------------------
# Step 4 — Density scatter plot
# ---------------------------------------------------------------------------

def plot_density_scatter(
    sp_sample: pd.DataFrame,
    output_path: str | Path = OUTPUT_DIR / "sp_density_scatter.png",
) -> None:
    """Save SP customer density scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        sp_sample["customer_lng"],
        sp_sample["customer_lat"],
        alpha=0.15, s=5, color="steelblue",
    )
    plt.title("SP Customer Density Scatter", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Scatter plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Step 5 — Written spatial summary
# ---------------------------------------------------------------------------

def write_spatial_summary(
    sp_sample: pd.DataFrame,
    bb: dict,
    output_path: str | Path = OUTPUT_DIR / "sp_spatial_summary.txt",
) -> str:
    """Generate and save the 5-bullet spatial summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lat_span_km = abs(bb["lat_q3"] - bb["lat_q1"]) * 111
    lon_span_km = abs(bb["lon_q3"] - bb["lon_q1"]) * 89

    summary = f"""
=== SP Customer Spatial Summary ===
Generated from: olist_orders + olist_customers + olist_geolocation
Total SP orders analysed: {bb['n_orders']:,}

5-BULLET SPATIAL SUMMARY:
1. Most customers are concentrated in the Sao Paulo metro region,
   centred around lat={bb['lat_med']:.3f}, lon={bb['lon_med']:.3f},
   confirming Sao Paulo city as the dominant demand hub.

2. Customer density shows a long-tail spread toward SP state interior
   (lat up to {bb['lat_max']:.2f}), indicating sparse but non-zero demand
   in smaller cities like Campinas, Ribeirao Preto, and Sao Jose dos Campos.

3. The dense core (IQR) spans only {lat_span_km:.0f} km x {lon_span_km:.0f} km,
   meaning ~50% of all SP customers live in a tightly packed metro zone.

4. Sparse demand outside the dense core suggests that placing dark stores
   purely at geographic centroid would under-serve the metro core --
   clustering is needed to separate high-density zones from low-density ones.

5. Recommended K for K-Means: 5-8 clusters, with at least 2-3 clusters
   covering the Sao Paulo metro core and remaining clusters absorbing
   interior cities. (To be confirmed by elbow + silhouette analysis.)

BOUNDING BOX:
  Full SP state  -> Lat: {bb['lat_min']:.4f} to {bb['lat_max']:.4f} | Lon: {bb['lon_min']:.4f} to {bb['lon_max']:.4f}
  Dense core IQR -> Lat: {bb['lat_q1']:.4f} to {bb['lat_q3']:.4f} | Lon: {bb['lon_q1']:.4f} to {bb['lon_q3']:.4f}
"""
    output_path.write_text(summary)
    print(f"[INFO] Spatial summary saved → {output_path}")
    return summary


# ---------------------------------------------------------------------------
# Step 6 — Save bounding box CSV
# ---------------------------------------------------------------------------

def save_bounding_box_csv(
    bb: dict,
    output_path: str | Path = OUTPUT_DIR / "sp_bounding_box.csv",
) -> None:
    """Save bounding box as structured CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"metric": "full_sp_lat_min",  "value": bb["lat_min"]},
        {"metric": "full_sp_lat_max",  "value": bb["lat_max"]},
        {"metric": "full_sp_lon_min",  "value": bb["lon_min"]},
        {"metric": "full_sp_lon_max",  "value": bb["lon_max"]},
        {"metric": "core_iqr_lat_min", "value": bb["lat_q1"]},
        {"metric": "core_iqr_lat_max", "value": bb["lat_q3"]},
        {"metric": "core_iqr_lon_min", "value": bb["lon_q1"]},
        {"metric": "core_iqr_lon_max", "value": bb["lon_q3"]},
        {"metric": "total_sp_orders",  "value": bb["n_orders"]},
        {"metric": "suggested_k_min",  "value": 5},
        {"metric": "suggested_k_max",  "value": 8},
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[INFO] Bounding box CSV saved → {output_path}")


# ---------------------------------------------------------------------------
# Step 7 — Interactive Folium map (Day 2 deliverable)
# ---------------------------------------------------------------------------

def build_folium_map(
    sp_sample: pd.DataFrame,
    sellers_sp: pd.DataFrame,
    output_path: str | Path = VIS_DIR / "sp_eda_map.html",
):
    """
    Build and save the interactive Folium map with:
      - Customer density heatmap layer
      - Seller MarkerCluster layer
      - LayerControl toggle

    Returns
    -------
    folium.Map object
    """
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
    except ImportError:
        print("[WARN] folium not installed. Run: pip install folium")
        print("[WARN] Skipping map generation.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base map centred on Sao Paulo
    m = folium.Map(location=[-23.5, -46.6], zoom_start=9, tiles="CartoDB positron")

    # Layer 1: Customer density heatmap
    heat_data = sp_sample[["customer_lat", "customer_lng"]].dropna().values.tolist()
    HeatMap(
        heat_data,
        radius=8, blur=12, min_opacity=0.3,
        name="Customer density",
    ).add_to(m)

    # Layer 2: Seller locations (MarkerCluster)
    seller_cluster = MarkerCluster(name="Sellers (SP)").add_to(m)
    for _, row in sellers_sp.iterrows():
        folium.CircleMarker(
            location=[row["seller_lat"], row["seller_lng"]],
            radius=4, color="red", fill=True, fill_opacity=0.6,
            tooltip=row["seller_id"],
        ).add_to(seller_cluster)

    folium.LayerControl().add_to(m)

    m.save(str(output_path))
    print(f"[INFO] Folium map saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# Full pipeline runner — called by main.py
# ---------------------------------------------------------------------------

def run(
    raw_dir:    str | Path = RAW_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    vis_dir:    str | Path = VIS_DIR,
) -> pd.DataFrame:
    """

    Steps
    -----
    1. Load & merge SP sample (orders + customers + geolocation)
    2. Compute bounding box
    3. Save density scatter plot  ->  outputs/sp_density_scatter.png
    4. Save spatial summary txt   ->  outputs/sp_spatial_summary.txt
    5. Save bounding box CSV      ->  outputs/sp_bounding_box.csv
    6. Build Folium map           ->  visualisations/sp_eda_map.html

    Returns
    -------
    sp_sample : pd.DataFrame  (passed downstream to clustering module)
    """
    output_dir = Path(output_dir)
    vis_dir    = Path(vis_dir)

    # Day 1
    sp_sample  = load_sp_sample(raw_dir)
    bb         = compute_bounding_box(sp_sample)
    plot_density_scatter(sp_sample,    output_dir / "sp_density_scatter.png")
    write_spatial_summary(sp_sample, bb, output_dir / "sp_spatial_summary.txt")
    save_bounding_box_csv(bb,          output_dir / "sp_bounding_box.csv")

    # Day 2 — Folium map (requires sellers table)
    sellers_sp = load_sp_sellers(raw_dir)
    build_folium_map(sp_sample, sellers_sp, vis_dir / "sp_eda_map.html")

    print("\n[DONE] EDA pipeline complete.")
    print(f"  Orders  : {bb['n_orders']:,}")
    print(f"  Sellers : {len(sellers_sp):,}")
    print(f"  Centre  : ({bb['lat_med']:.3f}, {bb['lon_med']:.3f})")
    print(f"  Core    : {abs(bb['lat_q3']-bb['lat_q1'])*111:.0f} km x "
          f"{abs(bb['lon_q3']-bb['lon_q1'])*89:.0f} km")

    return sp_sample


# ---------------------------------------------------------------------------
# Entry point (run directly: python src/exploratory_analysis.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()