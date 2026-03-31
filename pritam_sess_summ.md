# Pritam's Progress Summary — Dark Store Project
> **Purpose:** Context handoff for a fresh Claude / ChatGPT session scoped to Pritam's work only.  
> Load this + `session_summary_v3.md` (master context) at the start of every new session.  
> **Last updated:** March 31, 2026 | **Day 1 complete. Day 2 starts April 1.**

---

## 1. WHO IS PRITAM

- **Role:** Lead Coder + Integration Architect (★★ heaviest load)
- **AI tool:** Claude Code (GitHub Copilot in VS Code)
- **Environment:** uv + Python 3.13.12 on WSL2 (`/mnt/d/Python-UV/SCA_DARK_STORES`)
- **Background:** PGDBA @ ISI Kolkata / IIM Calcutta. Mech Eng IIT Kharagpur. EXL Services (Citi Bank analytics), Aarti Industries. Strong: Python, OR/LP, ML, time series, PyTorch.

---

## 2. WHAT PRITAM HAS COMPLETED (Day 1 — March 31, 2026)

### 2.1 Environment & Repo

| Item | Status | Detail |
|------|--------|--------|
| uv environment | ✅ Done | Python 3.13.12 pinned; `.venv/` created; all packages installed via `uv add` |
| GitHub repo | ✅ Live | https://github.com/metaphorpritam/SCA_DARK_STORES · branch `main` |
| Initial commit | ✅ Pushed | Commit `d9993f4` — README only (first commit) |
| Day 1 scaffold commit | ✅ Pushed | Commit `9a8b257` — full folder tree + all stubs + OR-Tools toy |

### 2.2 Folder Structure Created

```
SCA_DARK_STORES/
├── data/
│   └── raw/                    # empty — Olist CSVs go here (Vybhav's job)
├── notebooks/                  # empty — notebooks created per day
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        ← stub
│   ├── haversine_matrix.py     ← stub
│   ├── clustering.py           ← stub
│   ├── route_parser.py         ← stub
│   ├── return_classifier.py    ← stub
│   ├── joint_optimizer.py      ← stub
│   └── ortools_toy_cvrptw.py   ← WORKING toy CVRPTW (verified)
├── outputs/                    # empty
├── report/                     # empty
├── visualisations/             # empty
├── docs/
│   └── architecture.md         ← ASCII diagram + module interfaces + DAG
├── requirements.txt            ← pip-compatible mirror of pyproject.toml deps
├── pyproject.toml              ← uv-managed; Python 3.13.12
├── uv.lock                     ← committed; 115 packages locked
├── .gitignore                  ← blocks data/raw CSVs, *.parquet, *.npy, *.pkl, outputs/
├── README.md                   ← full project README (committed)
└── session_summary_v3.md       ← master context document
```

### 2.3 OR-Tools Toy CVRPTW — VERIFIED WORKING

File: `src/ortools_toy_cvrptw.py`

**Test result (run on Day 1):**
```
[PASS] OR-Tools CVRPTW toy example solved successfully.
Total distance: 10.5 km across 2 vehicles, 10 customer nodes, 1 depot.
Strategy: PATH_CHEAPEST_ARC → GUIDED_LOCAL_SEARCH, 30s limit.
```

This confirms OR-Tools is installed and functional in the uv environment.

### 2.4 Architecture Diagram

File: `docs/architecture.md`

Contains:
- Full ASCII pipeline: Olist → Preprocessing → Clustering → Forward VRP → Reverse VRP → Joint Optimizer → Dashboard
- Module interface table: inputs, outputs, dependencies for every `src/` module
- Dependency DAG (who unblocks whom, Day 1–8)
- Sprint calendar (all 8 days, all 6 people)

### 2.5 Module Stubs Committed

Each stub has: module docstring, typed function signatures, parameter descriptions, return type annotations, and a `# TODO` body. Ready to be filled in starting Day 2/3.

| File | Key functions stubbed |
|------|----------------------|
| `src/data_pipeline.py` | `load_olist()`, `merge_master_df()`, `engineer_features()`, `filter_sp()` |
| `src/haversine_matrix.py` | `haversine_km()`, `build_distance_matrix()`, `stratified_spatial_sample()` |
| `src/clustering.py` | `run_kmeans_sweep()`, `pick_optimal_k()`, `assign_voronoi()`, `run_pmedian()` |
| `src/route_parser.py` | `extract_routes()`, `compute_route_kpis()`, `routes_to_dataframe()` |
| `src/return_classifier.py` | `build_features()`, `train_classifier()`, `predict_return_prob()` |
| `src/joint_optimizer.py` | `compute_Z()`, `pareto_sweep()`, `solve_sdvrp_hybrid()` |

### 2.6 Installed Packages (pyproject.toml)

```
numpy>=2.4.4          pandas>=3.0.2        jupyter>=1.1.1
ipykernel>=7.2.0      matplotlib>=3.10.8   seaborn>=0.13.2
scikit-learn>=1.8.0   ortools>=9.15.6755   pulp>=3.3.0
geopandas>=1.1.3      xgboost>=3.2.0       prophet>=1.3.0
shap>=0.51.0          folium>=0.20.0
```

All locked in `uv.lock` (115 packages). Verified with:
```bash
uv sync && uv run python -c "import numpy, pandas, sklearn, ortools, pulp; print('ok')"
```

---

## 3. WHAT PRITAM MUST DO NEXT (Day 2 — April 1)

Per the roadmap ([`Dark_Store_Logistics_Roadmap_v3.pdf`](Dark_Store_Logistics_Roadmap_v3.pdf)):

### Primary task: Haversine Distance Matrix
**Depends on:** `data/master_df.parquet` from Vybhav (Day 1 EOD)  
**File to implement:** `src/haversine_matrix.py`

Steps:
1. Load `master_df.parquet`; filter `customer_state == 'SP'`
2. Stratified spatial sample → 500 representative customer (lat, lon) points
3. Build 500×500 pairwise Haversine distance matrix
4. Scale to integers × 1000 for OR-Tools compatibility
5. Save `data/distance_matrix.npy` and `data/sp_customer_sample.csv`
6. Validate: print min/mean/max — expect ~0.5 km to ~60 km for SP

**Expected output files:**
- `data/distance_matrix.npy` — integer-scaled, 500×500
- `data/sp_customer_sample.csv` — 500 rows: `[node_id, lat, lon, zip_prefix, order_count]`

**Key implementation note:**
```python
# Haversine must be integer-scaled for OR-Tools
dist_matrix_int = (haversine_matrix_km * 1000).astype(int)
np.save("data/distance_matrix.npy", dist_matrix_int)
```

### Secondary (if master_df arrives early): Notebook skeleton
Create `notebooks/02_distance_matrix.ipynb` — interactive version of the above.

---

## 4. BLOCKING DEPENDENCIES ON OTHERS (Day 2)

| Who | What Pritam needs from them | When |
|-----|-----------------------------|------|
| Vybhav | `data/master_df.parquet` | EOD Day 1 (April 1) |
| Pranav | `vrp_nodes_schema.md` (already committed Day 1) | ✅ Done |
| Sneha | `dark_store_candidates.csv` | Mid Day 2 (needed for Day 3 VRP) |

Pritam is **not blocked on Day 2** — distance matrix only needs master_df which Vybhav targets by EOD Day 1.

---

## 5. PRITAM'S FULL 8-DAY SCHEDULE (SUMMARY)

| Day | Pritam's Task | Key Output |
|-----|--------------|------------|
| **Day 1 ✅** | Repo + scaffold + OR-Tools warmup + architecture | GitHub live, OR-Tools verified, all stubs |
| **Day 2** | Haversine 500×500 distance matrix | `distance_matrix.npy`, `sp_customer_sample.csv` |
| **Day 3** | Forward VRP — OR-Tools CVRPTW all K zones | `forward_routes.json` |
| **Day 4** | Forward VRP all zones + SDVRP prototype (1 zone) | `forward_kpi_summary.csv`, `sdvrp_prototype_v1.py` |
| **Day 5** | SDVRP hybrid all zones + `joint_optimizer.py` v1 | `hybrid_routes.json`, Z computable |
| **Day 6** | Weighted-sum Pareto sweep (25 combos) + report section | `pareto_results.csv`, `pareto_tradeoff.png` |
| **Day 7** | Full 10–12 page report + `run_all.sh` + pipeline test | `report_draft_v1.docx`, reproducible pipeline |
| **Day 8** | Final polish + `submission_package/` assembly | `project_final.zip`, submitted ★ |

---

## 6. ENVIRONMENT QUICK REFERENCE

```bash
# Activate environment
cd /mnt/d/Python-UV/SCA_DARK_STORES
source .venv/bin/activate

# Run a script
uv run python src/ortools_toy_cvrptw.py

# Add a new package
uv add <package-name>

# Push to GitHub
git add -A && git commit -m "message" && git push origin main
```

**GitHub:** https://github.com/metaphorpritam/SCA_DARK_STORES  
**Branch:** `main`  
**Current HEAD:** commit `9a8b257` (Day 1 scaffold complete)

---

## 7. OR-TOOLS CRITICAL NOTES (TO AVOID DAY 3 BUGS)

1. **Always integer-scale distances:** `int(km * 1000)` before passing to OR-Tools
2. **Use `manager.IndexToNode(index)` inside all callbacks** — raw index ≠ node index
3. **`AddDimensionWithVehicleCapacity` takes a list:** `[500000] * num_vehicles`, not a scalar
4. **SDVRP load invariant:** net load at each stop ≥ 0 and ≤ capacity — enforce via AddDimension with slack=0
5. **Search order:** `PATH_CHEAPEST_ARC` first solution → `GUIDED_LOCAL_SEARCH` improvement, 30s limit

---

*End of Pritam's session summary. Pair with `session_summary_v3.md` for full project context.*
