# Master Project Summary — Dark Store + Integrated Logistics Project
> **Purpose:** Canonical context handoff for any fresh Claude / ChatGPT session. Read fully before responding to any follow-up. This is the single source of truth for the whole project — team, architecture, methodology, file structure, and implementation decisions.  
> **Last updated:** March 31, 2026 | **Project deadline:** April 8, 2026 | **Sprint start:** April 1, 2026

---

## 1. WHO WE ARE

| Name | Tool | Role | Skill Level |
|------|------|------|-------------|
| **Pritam** ★★ | Claude Code | Lead Coder + Integration Architect | ★★★★★ |
| **Vybhav** ★ | Gemini Pro | Co-Lead Coder (Data Pipeline + Reverse Logistics) | ★★★★★ |
| **Anurag** | Gemini Pro | Data Support + EDA + Forecasting | ★★★ |
| **Sneha** | Gemini Pro | Clustering Specialist | ★★★ |
| **Pranav** | Gemini Pro | VRP Node Prep + Route Parsing | ★★★ |
| **Varsha** | Gemini Pro | Visualisation + Report + Deck | ★★★ |

**Context about Pritam:** PGDBA candidate at ISI Kolkata (IIM Calcutta coursework). Dual degree Mech Eng IIT Kharagpur. Prior: EXL Services (Citi Bank regulatory banking analytics), Aarti Industries (Manufacturing). Strong: Python, OR/LP, ML, time series, PyTorch, UV/WSL environment.

---

## 2. THE PROJECT

### 2.1 Final Project Title
> **"Dark Store Placement + Integrated Forward & Reverse Logistics Optimisation"**

### 2.2 Problem Statement
How can we **optimally place dark stores** (micro-fulfilment centres) and design **integrated delivery + return routes** to simultaneously minimise total cost, travel distance, and delivery time — treating forward and reverse logistics as a **single joint optimisation problem** rather than two separate systems?

### 2.3 Why This Is Strong
- **Real-world relevance:** Dark stores power Blinkit, Zepto, Rappi, Amazon Fresh
- **Academic novelty:** Most papers treat VRP (delivery) and reverse logistics separately; integrating both into a joint MILP is the gap being addressed
- **Technical depth:** 3-stage pipeline — Forecasting → MCDM/Clustering → Multi-Objective Optimisation — mirrors the canonical SCA framework
- **Implementable in 8 days** with Olist dataset (public, rich, real)

### 2.4 Architecture (Mental Model)
```
Olist Dataset (9 CSV tables)
        ↓
[Data Pipeline] master_df.parquet
        ↓
[Stage 1: Forecasting]
  - Demand per zip (Prophet per zone)
  - Return probability per order (XGBoost classifier)
        ↓
[Stage 2: Clustering / Dark Store Placement]
  - K-Means (elbow + silhouette → optimal K)
  - p-Median MILP (PuLP) — academic validation layer
  - Voronoi assignment → dark_store_id per customer
        ↓
[Stage 3: Multi-Objective Optimisation]
  - Forward VRP: OR-Tools CVRPTW (dark store → customers)
  - Reverse VRP: OR-Tools CVRPTW (customers → dark store, return pickups)
  - SDVRP Hybrid: same vehicle does delivery + pickup
  - Joint Objective: Z = α·C_fwd + β·C_rev + γ·T_penalty + δ·N_vehicles
  - Weighted-sum Pareto sweep: vary (α,β) over [0.1..0.9]² grid
        ↓
[Results + Visualisation]
  - Folium: customer heatmap, dark store map, Voronoi zones, route polylines
  - Plotly: Pareto front, cost comparison, scenario radar, KPI dashboard
  - Report (10-12 pages) + 14-slide deck
```

---

## 3. DATASET

**Primary dataset: Olist Brazilian E-Commerce Public Dataset**
- Source: `kaggle.com/datasets/olistbr/brazilian-ecommerce`
- Size: ~100K orders, 9 interlinked CSV tables
- Tables: `orders`, `order_items`, `products`, `customers`, `sellers`, `geolocation`, `order_payments`, `order_reviews`, `product_category_name_translation`

**Key join chain:**
```
orders (order_id) 
  ← order_items (order_id, product_id, seller_id) 
  ← products (product_id) 
  ← customers (customer_id) 
  ← sellers (seller_id) 
  + geolocation (zip_code_prefix → median lat/lon)
```

**Scope decision:** Filter to **São Paulo state** (`customer_state = 'SP'`) — ~41% of all orders, manageable size, natural geographic clustering.

**Key engineered columns:**
- `is_return`: 1 if `order_status IN ('canceled','unavailable')` OR `actual_delivery > estimated_delivery + 7 days`
- `return_rate_by_category`: return rate grouped by product category
- `dark_store_id`: Voronoi assignment of each customer to nearest dark store
- `return_prob`: XGBoost output — probability this order will be returned

---

## 4. METHODOLOGY (KEY DECISIONS MADE)

### 4.1 Clustering
| Method | What it minimises | Our choice |
|--------|------------------|------------|
| K-Means | Within-cluster variance (Euclidean) | **Primary** (fast, visual) |
| p-Median (PuLP MILP) | Weighted customer-to-facility distance | **Validation** (academic rigour) |

- **K selection:** Elbow curve (inertia) + silhouette score, typically K=5–8 for SP
- **Sensitivity check:** Perturb demand weights ±20%, re-run 10 times — centroids must stay stable
- **Voronoi assignment:** Each customer → nearest dark store centroid
- **Capacity:** total_daily_orders / K × 1.3 (30% buffer per dark store)
- **Coverage target:** > 70% of SP customers within 5 km of their assigned dark store

### 4.2 VRP Setup (OR-Tools CVRPTW)
```python
# Key parameters
vehicle_capacity_kg   = 500
vehicle_speed_kmh     = 40      # São Paulo urban
fixed_cost_per_route  = 50      # R$
variable_cost_per_km  = 1.5     # R$
time_window_morning   = (480, 720)    # minutes from midnight
time_window_afternoon = (720, 1080)
service_time_min      = 5       # per stop
max_solve_time_sec    = 30      # per zone
search_strategy       = GUIDED_LOCAL_SEARCH  # after PATH_CHEAPEST_ARC first solution
customer_sample       = 60-80   # per zone (tractability)
distance_matrix_nodes = 500     # stratified spatial sample, integer-scaled ×1000
```

**Three VRP variants built:**
1. **Forward CVRPTW:** dark store → customers (delivery)
2. **Reverse CVRPTW:** customers → dark store (return pickup)
3. **SDVRP Hybrid:** same vehicle, delivery + pickup combined (the integration edge)

### 4.3 Joint Objective
```
Z = α · C_forward + β · C_reverse + γ · T_penalty + δ · N_vehicles

where:
  C_forward  = total forward routing cost (R$)
  C_reverse  = total reverse routing cost (R$)
  T_penalty  = Σ time_window_violations × penalty_multiplier
  N_vehicles = total fleet size (encourages route consolidation)
  α = β = γ = δ = 0.25  (baseline equal weights)
```

**Pareto sweep:** Grid search over (α,β) ∈ [0.1, 0.3, 0.5, 0.7, 0.9]² → 25 combinations → 2D trade-off curve → identify knee point.

### 4.4 Return Probability Classifier
```python
Model: XGBClassifier(scale_pos_weight=15)  # ~5% positive rate → heavy imbalance
Features: product_category, product_weight_g, freight_value, seller_state,
          days_late (actual − estimated), review_score, payment_type, order_value
Target: is_return (binary)
Split: 80/20 stratified
Metrics: AUC-ROC target > 0.70, also report PR-AUC (better for imbalanced)
Usage: orders with return_prob > 0.30 → pre-assigned pickup slots in SDVRP
```

### 4.5 Scenario Analysis
| Scenario | Description | Purpose |
|----------|-------------|---------|
| A (Base) | Current demand, current return rate | Baseline |
| B (Surge +30%) | All demands × 1.3 (flash sale simulation) | Capacity stress test |
| C (High Returns ×2) | Return prob threshold halved | Reverse logistics stress |

---

## 5. DEPENDENCY GRAPH (CRITICAL PATH)

This is the most important operational insight — it determines who waits on whom:

```
Day 1 EOD: master_df.parquet (Vybhav)
  → unlocks: Anurag (SP EDA), Sneha (K-Means), Pranav (vrp_nodes), Vybhav (return flag)

Day 2 EOD: dark_store_candidates.csv (Sneha) + vrp_nodes.csv (Pranav) + distance_matrix.npy (Pritam)
  → unlocks: Pritam (Forward VRP), Vybhav (p-Median + Reverse VRP), Anurag (Prophet forecasting)

Day 3 EOD: forward_routes.json (Pritam) + return_prob column (Vybhav)
  → unlocks: Pranav (route parser), Pritam (SDVRP hybrid), joint_optimizer

Day 4 EOD: reverse_routes.json (Vybhav) + hybrid_prototype (Pritam)
  → unlocks: joint_optimizer.py full implementation (Day 5)

Day 5 EOD: all_zones_summary.csv (Anurag)
  → unlocks: Varsha (final dashboard), Pareto sweep (Day 6), Scenario analysis (Day 6)

Day 6 EOD: pareto_tradeoff.png + scenario_results_table.csv
  → unlocks: Report writing (Day 7), Final dashboard

Day 7 EOD: report_draft_v1.pdf + presentation_v1.pptx + v1.0-final GitHub tag
  → unlocks: Day 8 cross-review + submission
```

**Day 1 is FULLY PARALLEL** — all 6 tasks are independent, zero waiting.

---

## 6. TASK ALLOCATION BY DAY

### Day 1 — April 1 (Setup — all parallel)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | Repo + architecture + OR-Tools warmup | GitHub repo, OR-Tools confirmed, architecture diagram |
| Vybhav ★ | Olist download + master_df merge pipeline | master_df.parquet (EOD) |
| Anurag | Schema study + SP sample EDA | ER diagram, SP density scatter |
| Sneha | Literature review + K-Means toy prototype | K-Means vs p-Median comparison note |
| Pranav | OR-Tools tutorial + VRP node schema design | OR-Tools example running, vrp_nodes_schema.md |
| Varsha | Dashboard template + slide skeleton | base_map.html, presentation_skeleton.pptx |

### Day 2 — April 2 (Data complete → Clustering starts)
| Person | Task | Output |
|--------|------|--------|
| Vybhav ★ | Return flag + demand profile + baseline KPIs | master_df_v2.parquet, demand_profile.csv, baseline_kpis.csv |
| Pritam ★★ | Haversine distance matrix (500×500) | distance_matrix.npy, sp_customer_sample.csv |
| Sneha | K-Means (K=3..12), elbow+silhouette, pick optimal K | dark_store_candidates.csv, elbow_silhouette.png |
| Anurag | Full SP spatial EDA + interactive Folium map | sp_eda_map.html |
| Pranav | VRP node list construction | vrp_nodes.csv with time windows |
| Varsha | Clustering visualisation (Voronoi zones) | dark_store_map.html, clustering_comparison.png |

### Day 3 — April 3 (Clustering final + Forward VRP + ML start)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | OR-Tools CVRPTW forward VRP (full implementation, all zones) | forward_routes.json |
| Vybhav ★ | p-Median MILP (PuLP) + XGBoost return classifier start | p_median_locations.csv, return_clf_v1.pkl |
| Sneha | Dark store finalisation + Voronoi assignment + sensitivity | dark_stores_final.csv, sensitivity_stability.png |
| Anurag | Prophet demand forecasting per zone | forecasted_demand_by_zone.csv |
| Pranav | Baseline comparison + route_parser.py stub | baseline_kpis_naive.csv, route_parser.py |
| Varsha | Methodology section draft + architecture diagram | methodology_section_draft.docx |

### Day 4 — April 4 (VRP complete + Return ML final + Reverse VRP)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | Forward VRP all zones + SDVRP prototype (1 zone) | forward_kpi_summary.csv, sdvrp_prototype_v1.py |
| Vybhav ★ | Return classifier final + Reverse VRP | master_df_v3.parquet (return_prob), reverse_routes.json |
| Sneha | Resilience analysis (dark store failure simulation) | resilience_analysis.png |
| Anurag | Scenario node files A/B/C | vrp_nodes_A.csv, vrp_nodes_B.csv, vrp_nodes_C.csv |
| Pranav | Route parser complete + forward KPI by zone | forward_kpi_by_zone.csv |
| Varsha | Forward route map + return heatmap + feature importance | forward_routes_map.html, return_heatmap.html |

### Day 5 — April 5 (SDVRP hybrid + Joint optimizer begins)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | SDVRP hybrid all zones + joint_optimizer.py v1 (Z computable) | hybrid_routes.json, joint_optimizer.py |
| Vybhav ★ | Reverse VRP all zones + solver tuning | all_zones_reverse_results.csv, solver_tuning_results.csv |
| Anurag | All-zones results aggregation | all_zones_summary.csv |
| Sneha | Solver tuning support + literature positioning draft | literature_positioning.docx |
| Pranav | Final comparison table (naive/separate/hybrid) | final_comparison_table.csv |
| Varsha | Dashboard draft (with placeholders) | dashboard_draft.html |

### Day 6 — April 6 (Joint optimisation + Scenarios + Pareto)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | Weighted-sum Pareto sweep (25 combinations) + joint optim report section | pareto_results.csv, pareto_tradeoff.png |
| Vybhav ★ | 3-scenario analysis (A/B/C) | scenario_results_table.csv |
| Anurag | Cross-validation of all KPI numbers | master_results.json |
| Sneha | Sensitivity + robustness report section | sensitivity_and_robustness_section.docx |
| Pranav | RESULTS.md + GitHub v0.9-results tag | RESULTS.md |
| Varsha | Final dashboard (all real data, no placeholders) | final_dashboard.html, all_charts/ (6 PNGs) |

### Day 7 — April 7 (Polish + Full report + Cross-review)
**Cross-review pairs:** Sneha↔Pranav (clustering↔VRP), Anurag↔Varsha (data↔visuals), Pritam↔Vybhav (VRP↔optimizer)
| Person | Task | Output |
|--------|------|--------|
| Pritam ★★ | Full 10-12 page report + pipeline reproducibility test | report_draft_v1.docx, run_all.sh confirmed |
| Vybhav ★ | Code cleanup + docstrings + GitHub v1.0-final tag | v1.0-final tag, all notebooks clean |
| Sneha | Cross-review: VRP↔Clustering consistency | cross_review_clustering_vrp.md |
| Anurag | Cross-review: Data↔Results consistency | data_results_verification.md |
| Pranav | Cross-review: Dashboard↔CSVs + tools section | dashboard_vs_csv_verification.md |
| Varsha | Final 14-slide deck + report assembly + PDF exports | presentation_v1.pptx, report_draft_v1.pdf |

### Day 8 — April 8 (Submit)
| Person | Task |
|--------|------|
| Pritam ★★ | Final report polish + submission_package/ assembly |
| Vybhav ★ | Final README.md + last reproducibility check |
| Sneha | Q&A prep: clustering + optimisation questions |
| Anurag | Q&A prep: data + scenario questions |
| Pranav | Q&A prep: VRP + implementation questions |
| Varsha | Final deck polish + **lead the 12-minute dry run** |

**Speaker assignments (dry run):** Varsha (slides 1–4) → Sneha (5–6) → Pranav (7–9) → Vybhav (10–11) → Pritam (12–14)

---

## 7. FILE STRUCTURE (COMMITTED TO GITHUB)

```
project-root/
├── data/
│   ├── raw/                    # Olist CSVs (9 files)
│   ├── master_df_v3.parquet    # final merged dataset with all engineered columns
│   ├── dark_stores_final.csv   # K dark store locations with capacity
│   ├── vrp_nodes.csv           # full node list for OR-Tools
│   ├── vrp_nodes_A/B/C.csv     # scenario variants
│   ├── distance_matrix.npy     # 500×500 Haversine, integer-scaled
│   ├── demand_profile.csv
│   ├── forecasted_demand_by_zone.csv
│   └── baseline_kpis.csv
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_forward_vrp.ipynb
│   ├── 04_return_ml.ipynb
│   ├── 05_reverse_vrp.ipynb
│   ├── 06_joint_optimizer.ipynb
│   └── 07_scenarios_and_analysis.ipynb
├── src/
│   ├── route_parser.py         # OR-Tools solution → DataFrame
│   ├── joint_optimizer.py      # Z = α·Cfwd + β·Crev + γ·Tpen + δ·Nveh
│   ├── haversine_matrix.py
│   ├── clustering.py
│   └── return_classifier.py
├── outputs/
│   ├── forward_routes.json
│   ├── reverse_routes.json
│   ├── hybrid_routes.json
│   ├── forward_kpi_summary.csv
│   ├── reverse_kpi_summary.csv
│   ├── hybrid_kpi_summary.csv
│   ├── all_zones_summary.csv
│   ├── pareto_results.csv
│   ├── scenario_results_table.csv
│   └── master_results.json     # single source of truth for all KPIs
├── visualisations/
│   ├── sp_eda_map.html
│   ├── dark_store_map.html
│   ├── forward_routes_map.html
│   ├── return_heatmap.html
│   ├── final_dashboard.html
│   └── all_charts/             # 6 PNGs for report embedding
├── report/
│   ├── report_draft_v1.docx
│   └── report_final.pdf
├── presentation/
│   ├── presentation_v1.pptx
│   └── presentation_final.pdf
├── RESULTS.md                  # all final numerical KPIs
├── README.md
├── requirements.txt
└── run_all.sh
```

---

## 8. KEY METRICS TO REPORT

| Metric | Target | Comparison Baseline |
|--------|--------|---------------------|
| ↓ Total routing cost | 20–40% below naive | Naive: direct seller→customer, no clustering |
| ↓ Avg delivery distance | 25–35% reduction | Naive avg distance (km) |
| ↑ Return efficiency | 15–25% cost saving | Separate forward + reverse systems |
| ↑ Dark store coverage | > 70% within 5 km | 0% (no dark stores) |
| Return classifier AUC | > 0.70 | Random baseline = 0.50 |
| Route consolidation | > 30% routes with ≥1 pickup | 0% (separate systems) |
| Scenario B robustness | Solution feasible under +30% demand | N/A |

---

## 9. TECHNOLOGY STACK

```
Language:        Python 3.13.12 (pinned via uv)
Data:            pandas>=3.0.2, numpy>=2.4.4, scipy, geopandas>=1.1.3
ML:              scikit-learn>=1.8.0, xgboost>=3.2.0, prophet>=1.3.0, shap>=0.51.0
Optimisation:    Google OR-Tools>=9.15 (CVRPTW), PuLP>=3.3.0 (p-median + joint MILP)
Visualisation:   Folium>=0.20.0, Plotly Express, matplotlib>=3.10.8, seaborn>=0.13.2
Reporting:       python-docx, reportlab, python-pptx
Dev tools:       Jupyter Lab>=1.1.1, ipykernel>=7.2.0, VS Code + GitHub Copilot, Git/GitHub
AI — Pritam:     Claude Code (architecture, complex debugging, report, integration)
AI — Others:     Gemini Pro + GitHub Copilot
Environment:     uv on WSL2/Windows (see Section 9a below)
```

### 9a. UV Environment Setup (Pritam's machine)

```
OS:              Windows 11 + WSL2 (Ubuntu)
Path:            /mnt/d/Python-UV/SCA_DARK_STORES
Python:          3.13.12 (pinned with uv python pin 3.13.12)
Venv:            .venv/ managed by uv (created automatically on first uv add)
Package file:    pyproject.toml  (uv-native; requirements.txt is a pip-compatible mirror)
Lockfile:        uv.lock  (committed; reproducible installs)
GitHub repo:     https://github.com/metaphorpritam/SCA_DARK_STORES
```

**Commands to reproduce environment from scratch:**
```bash
# From WSL terminal
cd /mnt/d/Python-UV/SCA_DARK_STORES
uv python pin 3.13.12
uv sync                          # installs all deps from uv.lock
source .venv/bin/activate        # activate venv for notebook / script use
```

**pyproject.toml dependencies (as of Day 1):**
```toml
dependencies = [
    "numpy>=2.4.4",
    "pandas>=3.0.2",
    "jupyter>=1.1.1",
    "ipykernel>=7.2.0",
    "matplotlib>=3.10.8",
    "seaborn>=0.13.2",
    "scikit-learn>=1.8.0",
    "ortools>=9.15.6755",
    "pulp>=3.3.0",
    "geopandas>=1.1.3",
    "xgboost>=3.2.0",
    "prophet>=1.3.0",
    "shap>=0.51.0",
    "folium>=0.20.0",
]
```

**To add a new package:**
```bash
uv add <package>       # resolves, installs, updates pyproject.toml + uv.lock
```

---

## 10. RELEVANT PAPERS (FOR LITERATURE REVIEW SECTION)

### Core papers to cite:
1. **Dethloff (2001)** — "Vehicle routing and reverse logistics: The vehicle routing problem with simultaneous delivery and pick-up" — *OR Spektrum* — this is the canonical SDVRP paper; directly supports the hybrid route innovation
2. **Islam, Amin & Wardley (2022)** — "Supplier Selection & Order Allocation with Predictive Analytics and Multi-Objective Programming" — *Computers & Industrial Engineering* — 3-stage pipeline precedent (forecast → evaluate → optimise)
3. **Islam et al. (2023)** — "Supplier Selection Framework: Deep Learning, PCA and Optimisation" — *Expert Systems with Applications* — inter-product correlation in forecasting
4. **Mohammed et al. (2019)** — "Hybrid MCDM-FMOO for Sustainable Supplier Selection" — *IJPE* — Fuzzy AHP + TOPSIS + multi-objective model
5. **OSCM Journal (2025)** — "Forecasting and Multi-Objective Optimisation for SSOA at Lubricants Distributor" — exact same 3-stage pipeline, real industry data
6. **Wasi et al. (2024)** — "SupplyGraph: GNN Benchmark for Supply Chain Planning" — *arXiv:2401.15299* — graph-native supply chain dataset

### Literature positioning argument:
> "Most papers treat forward VRP (Toth & Vigo 2014) and reverse logistics / closed-loop supply chains (Guide & Van Wassenhove 2009) as independent problems. Our work follows Dethloff (2001) in integrating simultaneous delivery and pickup (SDVRP), and extends it with (a) ML-predicted return probability as a dynamic input, and (b) a joint multi-objective optimisation framework producing a Pareto front."

---

## 11. SCA PROBLEM LANDSCAPE (CONTEXT FROM SESSION)

This project was chosen after surveying 8 integrated SCA problem domains. The unified pipeline across all domains:

```
Stage 1: FORECASTING → Stage 2: EVALUATION/SCORING → Stage 3: MULTI-OBJECTIVE OPTIMISATION
```

| Topic | Forecasting | Evaluation | Optimisation |
|-------|------------|------------|--------------|
| **SSOA** (our adjacent) | ARIMA/LSTM demand | Fuzzy AHP + TOPSIS | MILP (5 objectives) |
| **This project (Dark Store+VRP)** | XGBoost returns + Prophet demand | K-Means + p-Median clustering | SDVRP + joint MILP |
| Inventory Optimisation | SKU demand | ABC-XYZ | Newsvendor / (s,S) MILP |
| Network Design | Regional demand | AHP on sites | Facility location MILP |
| Risk Management | Disruption probability | FMEA + AHP | Stochastic MILP |

Our project is rated **⭐⭐⭐ High Implementability** (same tier as SSOA and Inventory Optimisation).

---

## 12. OR-TOOLS IMPLEMENTATION NOTES (CRITICAL)

### Common bugs to watch for:
1. **Integer scaling:** Distance matrix MUST be integer. Scale: `int(haversine_km * 1000)`
2. **Callback index vs node index:** `manager.IndexToNode(index)` ≠ `index` — always use IndexToNode in callbacks
3. **Capacity dimension:** `AddDimensionWithVehicleCapacity` needs a list of capacities (one per vehicle), not a scalar
4. **SDVRP load constraint:** `vehicle_load_at_point = Σ deliveries_remaining - Σ pickups_collected` — must stay ≥ 0 and ≤ capacity at every stop
5. **Time windows:** Must be in integer minutes from midnight. `tw_open=480` = 8:00 AM
6. **First solution vs improvement:** Always use `AUTOMATIC` or `PATH_CHEAPEST_ARC` first, then switch to `GUIDED_LOCAL_SEARCH`

### Skeleton for forward CVRPTW:
```python
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
routing = pywrapcp.RoutingModel(manager)

# Distance callback
def dist_callback(i, j):
    return distance_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
transit_idx = routing.RegisterTransitCallback(dist_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

# Capacity
def demand_callback(i):
    return demands[manager.IndexToNode(i)]
demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [500000]*num_vehicles, True, "Capacity")

# Time windows
routing.AddDimension(transit_idx, 30, 1080, False, "Time")
time_dim = routing.GetDimensionOrDie("Time")
for node_idx, (open_t, close_t) in enumerate(time_windows):
    idx = manager.NodeToIndex(node_idx)
    time_dim.CumulVar(idx).SetRange(open_t, close_t)

# Solve
params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
params.time_limit.seconds = 30
solution = routing.SolveWithParameters(params)
```

---

## 13. P-MEDIAN MILP (PuLP) SKELETON

```python
import pulp

def p_median(distances, demands, p):
    """
    distances: (n_customers, n_facilities) numpy array
    demands:   (n_customers,) weights
    p:         number of facilities to open
    """
    n_cust, n_fac = distances.shape
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts("x", [(i,j) for i in range(n_cust) for j in range(n_fac)], cat="Binary")
    y = pulp.LpVariable.dicts("y", range(n_fac), cat="Binary")
    
    # Objective: minimise weighted distance
    prob += pulp.lpSum(distances[i][j] * demands[i] * x[i,j]
                       for i in range(n_cust) for j in range(n_fac))
    # Constraints
    for i in range(n_cust):
        prob += pulp.lpSum(x[i,j] for j in range(n_fac)) == 1   # assign to exactly 1
    for i in range(n_cust):
        for j in range(n_fac):
            prob += x[i,j] <= y[j]                               # only if facility open
    prob += pulp.lpSum(y[j] for j in range(n_fac)) == p          # exactly p open
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    open_facilities = [j for j in range(n_fac) if pulp.value(y[j]) > 0.5]
    return open_facilities
```

---

## 14. JOINT OPTIMIZER SKELETON

```python
def compute_Z(fwd_cost, rev_cost, time_violations, n_vehicles,
              alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    """Normalise each component to [0,1] before weighting."""
    Z = (alpha * fwd_cost + beta * rev_cost +
         gamma * time_violations + delta * n_vehicles)
    return Z

def pareto_sweep(zones, alpha_vals, beta_vals):
    results = []
    for alpha in alpha_vals:
        for beta in beta_vals:
            gamma = delta = (1 - alpha - beta) / 2  # remaining weight split equally
            if gamma < 0: continue
            total_cost, avg_time, coverage = solve_all_zones(zones, alpha, beta, gamma, delta)
            results.append({"alpha":alpha,"beta":beta,"cost":total_cost,
                            "time":avg_time,"coverage":coverage})
    return pd.DataFrame(results)
```

---

## 15. WHAT WAS PREVIOUSLY DISCUSSED (DOCUMENTS GENERATED)

Three PDF documents were created during this session:

1. **`SCA_Problem_Landscape.pdf`** — 8 SCA problem domains with 3-stage pipeline for each, objective functions, tools, datasets, difficulty ratings. Design: dark editorial, colour-coded per topic.

2. **`SCA_Datasets_and_Papers.pdf`** — 35+ datasets across Hugging Face (5), Kaggle (15), GitHub/ArXiv (4), Figshare (4), OR-Library (4), Gov/Open (6). Each dataset includes: description, key features, topic tags, and papers that used it. 30+ papers cited with one-line findings.

3. **`Dark_Store_Logistics_Roadmap.pdf`** — 8-day personalised roadmap with real names. Covers: cover page with critical path table, 8 day pages (one per day, April 1–8) with per-person task cards (coloured accent strips), end-of-day checklists, AI tips (Claude Code vs Gemini Pro split), risk register, and appendix (tech stack, daily deliverables, workload summary).

---

## 16. RISK REGISTER

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| master_df delayed past Day 1 EOD | HIGH | MEDIUM | Vybhav starts merge early morning; Anurag can build SP sample from orders+customers alone if full merge slips |
| OR-Tools infeasible solution on real data | HIGH | MEDIUM | Relax time windows by 30 min; reduce customer sample per zone from 80 to 50; fall back to AUTOMATIC strategy |
| XGBoost AUC below 0.60 | MEDIUM | LOW | Add NLP sentiment features from reviews; try LightGBM; accept lower AUC and note as limitation |
| Git merge conflicts | LOW | HIGH | Each person works in separate files/folders; use feature branches; Pritam resolves conflicts daily |
| Pareto sweep too slow (25 solves) | MEDIUM | MEDIUM | Reduce grid to [0.2, 0.5, 0.8]² = 9 combinations; parallelise with multiprocessing |
| Team member unavailable for 1 day | MEDIUM | LOW | All tasks have written specs; partner can pick up using AI tool + roadmap; Day 7 buffer absorbs delays |
| Dashboard renders differently across machines | LOW | MEDIUM | Use standalone HTML with CDN-hosted Plotly; test on 2 machines before Day 8 |

---

## 17. WORKLOAD DISTRIBUTION SUMMARY

| Name | Load | Tasks Owned |
|------|------|-------------|
| **Pritam ★★** | HEAVY | Repo · Architecture · Distance matrix · OR-Tools Forward VRP (all zones) · SDVRP hybrid · Joint optimizer · Pareto sweep · Full report · Pipeline reproducibility |
| **Vybhav ★** | HEAVY | Master data pipeline · Return flag + KPIs · p-Median MILP · Return ML (XGBoost) · Reverse VRP (all zones) · Solver tuning · 3-Scenario analysis · Code cleanup |
| Anurag | MEDIUM | Schema study + EDA · SP filter · Temporal demand (Prophet) · Scenario node files · Results aggregation · Data cross-verification |
| Sneha | MEDIUM | K-Means (all K) · Dark store selection · Voronoi assignment · Sensitivity + resilience analysis · Literature positioning · Clustering cross-review |
| Pranav | MEDIUM | VRP node list · Route parser · Baseline comparison · Forward KPI metrics · Solver verification · Tools section · VRP cross-review |
| Varsha | MEDIUM | All Folium maps · Plotly dashboard (all charts) · Return heatmap · Full 14-slide deck · Report assembly · Dry run lead |

---

## 18. OPEN QUESTIONS / POSSIBLE NEXT STEPS

The following topics were raised but not yet resolved — a new session might pick these up:

1. **Code implementation:** Begin actual Python code for any module (data pipeline, clustering, VRP)
2. **Literature review depth:** Find specific SP dark store papers or quick-commerce logistics papers
3. **Report template:** Create a LaTeX or Word template for the 10-12 page report
4. **Presentation template:** Build the 14-slide deck structure with placeholder charts
5. **OR-Tools debugging:** Any specific error messages from Pritam's implementation
6. **Return classifier feature engineering:** Additional features from Olist reviews (NLP sentiment?)
7. **Metric validation:** How to validate the Haversine matrix is correctly scaled before passing to OR-Tools
8. **SHAP analysis:** Adding SHAP explainability layer to the return classifier

---

## 19. TONE / RESPONSE PREFERENCES FOR THIS PROJECT

- **Be direct and technical.** Pritam has strong OR, ML, and Python background — no need to explain basics.
- **Code-first.** When asked for implementation help, lead with working code, then explain.
- **Cite papers concisely.** Author (year) format; one-sentence finding is enough.
- **Flag dependencies explicitly.** When giving a code module, state what it depends on.
- **Prefer OR-Tools over custom heuristics** for routing — already decided.
- **Prefer PuLP over scipy.optimize** for MILP — simpler API.
- **UV environment on WSL** — use `uv run` or `uv pip install` syntax if relevant.
- **Avoid re-explaining the project.** Read this document and treat all context as given.

---

*End of session summary. Token budget: this document is ~3,800 words / ~4,800 tokens. A fresh Claude session with 200K context has ~195K tokens remaining after loading this.*
