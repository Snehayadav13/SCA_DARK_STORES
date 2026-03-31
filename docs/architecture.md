# System Architecture — Dark Store Placement & Integrated Forward/Reverse Logistics

## Overview

This project optimises dark store placement and end-to-end logistics for
Brazilian e-commerce using the Olist dataset (~100k orders, São Paulo state).

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER  (Day 1)                                │
│                                                                             │
│  data/raw/  ─── 9 Olist CSVs ──► src/data_pipeline.py ──► master_df.parquet│
│                                   (merge + feature eng.)                    │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ master_df.parquet
          ┌───────────────────┼────────────────────────────────┐
          ▼                   ▼                                ▼
┌─────────────────┐  ┌────────────────────┐  ┌───────────────────────────────┐
│  PLACEMENT      │  │  FORWARD VRP       │  │  REVERSE LOGISTICS            │
│  MODULE (Day 2) │  │  MODULE (Day 3)    │  │  MODULE (Day 4–5)             │
│                 │  │                    │  │                               │
│ clustering.py   │  │ (Pranav)           │  │ return_classifier.py          │
│ ─────────────   │  │ ─────────────────  │  │ ──────────────────────────    │
│ KMeans K=3..12  │  │ CVRPTW (OR-Tools)  │  │ XGBoost P(return | order)     │
│ p-Median MILP   │  │ Capacity + TW      │  │ Calibrated probabilities      │
│ Silhouette sel. │  │ Distance matrix    │  │                               │
│                 │  │ (haversine_matrix) │  │ route_parser.py               │
│ OUTPUT:         │  │                    │  │ (for reverse routes)          │
│ dark_stores.csv │  │ OUTPUT:            │  │                               │
│ (K centroids +  │  │ fwd_routes.csv     │  │ OUTPUT:                       │
│  coverage %)    │  │                    │  │ rev_routes.csv                │
└────────┬────────┘  └────────┬───────────┘  └───────────────┬───────────────┘
         │                   │                               │
         │  centroids        │ fwd_routes_df                 │ rev_routes_df
         │                   │                               │ return_probs
         └───────────────────┴───────────────┬───────────────┘
                                             ▼
                          ┌──────────────────────────────────────┐
                          │  JOINT OPTIMISER  (Day 6)            │
                          │                                      │
                          │  joint_optimizer.py                  │
                          │  ──────────────────────              │
                          │  Min Z = α·C_fwd + β·C_rev           │
                          │         + γ·T_pen + δ·N_veh          │
                          │                                      │
                          │  PuLP MILP (CBC solver)              │
                          │  Sensitivity: α, β, γ, δ             │
                          │                                      │
                          │  OUTPUT:                             │
                          │  joint_result.json                   │
                          │  vehicle_assignments.csv             │
                          └──────────────────┬───────────────────┘
                                             │
                                             ▼
                          ┌──────────────────────────────────────┐
                          │  VISUALISATION + REPORTING (Day 7–8) │
                          │                                      │
                          │  Folium  — interactive route map     │
                          │  Plotly  — Pareto frontier,          │
                          │            sensitivity plots         │
                          │  Seaborn — return classifier metrics │
                          │  outputs/  — all artefacts           │
                          │  report/   — LaTeX / Word writeup    │
                          └──────────────────────────────────────┘
```

---

## Module Interface Summary

| Module | Input | Output | Day |
|--------|-------|--------|-----|
| `data_pipeline.py` | `data/raw/*.csv` | `master_df.parquet` | 1 |
| `haversine_matrix.py` | `coords: np.ndarray (N×2)` | `dist_matrix: np.ndarray int64 (N×N)` | 2 |
| `clustering.py` | `master_df` | `dark_stores.csv`, `master_df_v2` | 2–3 |
| `ortools_toy_cvrptw.py` | (toy data) | console output — PASS/FAIL | 1 |
| `route_parser.py` | `RoutingModel, assignment, node_coords` | `routes_df.csv`, `summary dict` | 3–4 |
| `return_classifier.py` | `master_df` | `return_prob column`, `classifier.pkl` | 5 |
| `joint_optimizer.py` | `fwd_routes_df`, `rev_routes_df`, `return_probs` | `joint_result.json` | 6 |

---

## Data Flow (DAG)

```
master_df.parquet
    ├── clustering.py ──────────────────────────► dark_stores.csv
    │       │
    │       └── haversine_matrix.py ──► dist_matrix.npy
    │                   │
    │                   ├── [Pranav] Forward VRP (OR-Tools CVRPTW)
    │                   │           └── route_parser.py ──► fwd_routes.csv
    │                   │
    │                   └── [Pranav] Reverse VRP (OR-Tools SDVRP)
    │                               └── route_parser.py ──► rev_routes.csv
    │
    └── return_classifier.py ──────────────────► return_probs.csv
                │
                └── joint_optimizer.py (fwd_routes + rev_routes + return_probs)
                            └──► joint_result.json
                                        └── visualisations/ + report/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Data | pandas 3, numpy 2, geopandas, scipy |
| ML | scikit-learn, XGBoost 3, Prophet, SHAP |
| Optimisation | Google OR-Tools (CVRPTW, SDVRP), PuLP + CBC (p-Median, MILP) |
| Visualisation | Folium, Plotly Express, Matplotlib, Seaborn |
| Dev | uv, Jupyter Lab, VS Code + GitHub Copilot, Git/GitHub |
| AI | Claude Sonnet (Pritam), Gemini Pro + Copilot (team) |

---

## Sprint Calendar

| Day | Date | Focus | Owner |
|-----|------|-------|-------|
| 1 | Apr 1 | Repo scaffold, env setup, data pipeline, OR-Tools toy | **Pritam** + Vybhav |
| 2 | Apr 2 | Distance matrix, K-Means clustering, p-Median | **Pritam** |
| 3 | Apr 3 | Forward VRP (CVRPTW), route visualisation | Pranav + Pritam |
| 4 | Apr 4 | Reverse VRP (SDVRP), return route parsing | Pranav + Pritam |
| 5 | Apr 5 | Return classifier (XGBoost), SHAP analysis | **Pritam** + Bhargav |
| 6 | Apr 6 | Joint optimiser MILP, sensitivity analysis | **Pritam** + team |
| 7 | Apr 7 | Dashboard (Folium), Plotly Pareto, EDA figs | Sai + Pooja |
| 8 | Apr 8 | Report writing, final commit, presentation | All |

---

*Generated: Day 1 — Architecture committed as part of initial scaffold.*
