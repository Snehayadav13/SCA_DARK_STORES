# Dark Store Placement + Integrated Forward & Reverse Logistics Optimisation

> **Course:** Operations Management · Supply Chain Analytics  
> **Institution:** PGDBA — ISI Kolkata / IIM Calcutta / IIT Kharagpur  
> **Timeline:** April 1–8, 2026  
> **Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Dark Stores?](#2-why-dark-stores)
3. [Theoretical Foundations](#3-theoretical-foundations)
   - 3.1 [Facility Location: K-Means and p-Median](#31-facility-location-k-means-and-p-median)
   - 3.2 [Capacitated VRP with Time Windows (CVRPTW)](#32-capacitated-vrp-with-time-windows-cvrptw)
   - 3.3 [Simultaneous Delivery and Pickup (SDVRP)](#33-simultaneous-delivery-and-pickup-sdvrp)
   - 3.4 [Joint Multi-Objective Optimisation](#34-joint-multi-objective-optimisation)
   - 3.5 [Return Probability Prediction](#35-return-probability-prediction)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Step-by-Step Guide](#6-step-by-step-guide)
   - 6.1 [Environment Setup](#61-environment-setup)
   - 6.2 [Stage 0 — Data Pipeline](#62-stage-0--data-pipeline)
   - 6.3 [Stage 1 — Forecasting and Return Prediction](#63-stage-1--forecasting-and-return-prediction)
   - 6.4 [Stage 2 — Clustering and Dark Store Placement](#64-stage-2--clustering-and-dark-store-placement)
   - 6.5 [Stage 3 — Forward VRP](#65-stage-3--forward-vrp)
   - 6.6 [Stage 4 — Reverse VRP](#66-stage-4--reverse-vrp)
   - 6.7 [Stage 5 — SDVRP Hybrid](#67-stage-5--sdvrp-hybrid)
   - 6.8 [Stage 6 — Joint Optimisation and Pareto Analysis](#68-stage-6--joint-optimisation-and-pareto-analysis)
   - 6.9 [Stage 7 — Scenario Analysis](#69-stage-7--scenario-analysis)
   - 6.10 [Stage 8 — Visualisation and Dashboard](#610-stage-8--visualisation-and-dashboard)
7. [Key Metrics](#7-key-metrics)
8. [Project Structure](#8-project-structure)
9. [References](#9-references)
10. [Team](#10-team)

---

## 1. Problem Statement

E-commerce logistics in dense urban markets faces two intertwined challenges that are almost always treated as independent systems:

**Forward logistics** — how do you deliver orders from a warehouse (or dark store) to customers as cheaply and quickly as possible?

**Reverse logistics** — how do you collect returned or rejected orders back from customers without deploying a separate fleet?

Most academic literature and industry practice solve these sequentially: design delivery routes first, then design pickup routes separately. This is suboptimal because a delivery vehicle driving past a customer who needs to return a product could simply pick it up on the same trip.

This project asks a single integrated question:

> *Given a set of customer locations, order demands, and predicted return probabilities, how should we (a) place dark stores to minimise customer-to-facility distance, and (b) design vehicle routes that simultaneously deliver orders and collect returns, minimising a weighted combination of total cost, delivery time, and fleet size?*

The answer requires solving three coupled subproblems in sequence: **facility location**, **demand and return forecasting**, and **multi-objective vehicle routing** — unified through a joint objective function that produces a Pareto-optimal trade-off surface.

---

## 2. Why Dark Stores?

A dark store (also called a micro-fulfilment centre or MFC) is a small warehouse that serves only online orders — no walk-in customers. Companies like Blinkit, Zepto, Gopuff, Gorillas, and Amazon Fresh use dark stores to achieve sub-30-minute delivery in dense urban areas.

The core operations research question behind dark stores is a variant of the **p-median facility location problem**: given n customer locations with heterogeneous demand, open exactly p facilities to minimise the total weighted distance between customers and their assigned facility.

Dark stores introduce additional constraints not present in classical facility location:

- **Capacity limits** — each dark store can handle only a finite number of orders per day, determined by storage space and picker throughput.
- **Coverage requirements** — a target fraction of customers (e.g., 70%) must be within a service radius (e.g., 5 km) for the promise of fast delivery to hold.
- **Return handling** — dark stores also serve as collection points for returns, creating a reverse flow that interacts with forward delivery routes.

These constraints make the problem a natural fit for mixed-integer linear programming (MILP) and metaheuristic vehicle routing, which is exactly the methodology we adopt.

---

## 3. Theoretical Foundations

### 3.1 Facility Location: K-Means and p-Median

We use two complementary approaches to determine dark store locations.

**K-Means clustering** provides a fast, visual baseline. Given n customer points in ℝ² (latitude, longitude), K-Means partitions them into K clusters by minimising within-cluster sum of squared Euclidean distances:

$$
\min_{S_1, \ldots, S_K} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in S_k} \| \mathbf{x}_i - \boldsymbol{\mu}_k \|^2
$$

where μₖ is the centroid of cluster Sₖ. We weight each customer point by its order volume (orders per zip code) so that high-demand areas pull centroids toward them. The optimal K is selected at the agreement point between the **elbow curve** (inertia vs K) and the **silhouette score** (cohesion vs separation), typically yielding K ∈ {5, …, 8} for São Paulo.

**p-Median MILP** provides the academically rigorous validation layer. Define binary decision variables:

$$
x_{ij} = \begin{cases} 1 & \text{if customer } i \text{ is assigned to facility } j \\ 0 & \text{otherwise} \end{cases}
\qquad
y_j = \begin{cases} 1 & \text{if facility } j \text{ is opened} \\ 0 & \text{otherwise} \end{cases}
$$

The p-median formulation is:

$$
\min \sum_{i=1}^{n} \sum_{j=1}^{m} d_{ij} \cdot w_i \cdot x_{ij}
$$

subject to:

$$
\sum_{j=1}^{m} x_{ij} = 1 \quad \forall\, i \qquad \text{(each customer assigned to exactly one facility)}
$$

$$
x_{ij} \leq y_j \quad \forall\, i, j \qquad \text{(only assign to open facilities)}
$$

$$
\sum_{j=1}^{m} y_j = p \qquad \text{(open exactly } p \text{ facilities)}
$$

where dᵢⱼ is the Haversine distance between customer i and candidate facility j, and wᵢ is the demand weight of customer i. We solve this with PuLP (CBC solver). The p-median solution is compared against K-Means centroids: if both methods agree on locations within a tolerance (typically < 1 km), we have high confidence in the placement.

**Voronoi assignment** partitions all customers to their nearest dark store, creating non-overlapping service zones. Each zone becomes an independent subproblem for vehicle routing.

### 3.2 Capacitated VRP with Time Windows (CVRPTW)

The **Capacitated Vehicle Routing Problem with Time Windows** is a constrained combinatorial optimisation problem. Given a depot (dark store), a set of customers with demands, vehicle capacity limits, and per-customer delivery time windows, find the minimum-cost set of routes such that:

- Every customer is visited exactly once.
- Each route starts and ends at the depot.
- The total demand on any route does not exceed vehicle capacity Q.
- Each customer is served within their time window [aᵢ, bᵢ].

Formally, let G = (V, A) be a complete directed graph where V = {0, 1, …, n} (node 0 is the depot), and let cᵢⱼ be the travel cost from i to j. Define binary variables xᵢⱼₖ = 1 if vehicle k traverses arc (i, j). The objective is:

$$
\min \sum_{k=1}^{K} \sum_{(i,j) \in A} c_{ij} \cdot x_{ijk}
$$

subject to flow conservation, capacity, time window, and subtour elimination constraints.

CVRPTW is NP-hard, so exact solutions are intractable for instances beyond ~25 nodes. We use **Google OR-Tools**, which implements a two-phase approach:

1. **First solution heuristic** — `PATH_CHEAPEST_ARC` constructs an initial feasible solution by greedily extending the cheapest arc at each step.
2. **Local search metaheuristic** — `GUIDED_LOCAL_SEARCH` iteratively improves the solution by penalising frequently used arcs, escaping local optima over a 30-second time limit per zone.

**Key implementation parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Vehicle capacity | 500 kg | Standard urban delivery van |
| Vehicle speed | 40 km/h | São Paulo urban average |
| Fixed cost per route | R$50 | Driver + vehicle amortisation |
| Variable cost per km | R$1.50 | Fuel + maintenance |
| Morning time window | 08:00–12:00 (480–720 min) | Typical AM delivery slot |
| Afternoon time window | 12:00–18:00 (720–1080 min) | Typical PM delivery slot |
| Service time per stop | 5 min | Handoff time |
| Customer sample per zone | 60–80 | Tractability bound |
| Distance matrix scaling | ×1000 (to integer) | OR-Tools requires integer costs |

### 3.3 Simultaneous Delivery and Pickup (SDVRP)

The **Simultaneous Delivery and Pickup VRP** (Dethloff, 2001) extends CVRPTW by allowing each vehicle to both deliver goods and collect returns on the same route. This is the key innovation of our project — most papers and industry systems treat forward and reverse logistics as separate routing problems.

Each customer node i now has two demand attributes:

- dᵢ — delivery demand (order weight to drop off)
- pᵢ — pickup demand (return weight to collect, non-zero only if return_prob > 0.30)

The load on a vehicle at any point in the route must satisfy:

$$
0 \leq L_{\text{vehicle}}(t) \leq Q \quad \forall\, t
$$

where the load evolves as:

$$
L_{\text{vehicle}}(\text{after stop } i) = L_{\text{vehicle}}(\text{before stop } i) - d_i + p_i
$$

The vehicle departs the depot with load = Σdᵢ for all customers on its route, drops off deliveries and picks up returns along the way, and returns to the depot with load = Σpᵢ. The constraint is that at no point does the load exceed Q or drop below 0.

This is non-trivial because the order in which stops are visited affects whether the capacity constraint is satisfied. A route that works with deliveries only may become infeasible when pickups are added if a large pickup occurs before enough deliveries have reduced the load.

**Expected cost saving:** 15–25% compared to running separate forward and reverse fleets, because SDVRP eliminates redundant travel to customer locations that require both delivery and pickup.

### 3.4 Joint Multi-Objective Optimisation

Forward cost, reverse cost, delivery time, and fleet size are competing objectives. Minimising cost may require fewer vehicles with longer routes (higher delivery time). Minimising time may require more vehicles (higher fleet cost). We unify these into a single weighted objective:

$$
Z = \alpha \cdot C_{\text{fwd}} + \beta \cdot C_{\text{rev}} + \gamma \cdot T_{\text{penalty}} + \delta \cdot N_{\text{vehicles}}
$$

where:

- C_fwd = total forward routing cost (R$)
- C_rev = total reverse routing cost (R$)
- T_penalty = Σ time window violations × penalty multiplier
- N_vehicles = total fleet size across all zones
- α, β, γ, δ are normalised weights summing to 1

Each component is normalised to [0, 1] before weighting to ensure commensurability.

**Pareto sweep:** We vary (α, β) over a 5×5 grid {0.1, 0.3, 0.5, 0.7, 0.9}² with γ = δ = (1 − α − β)/2 (valid only when α + β ≤ 1), yielding up to 25 weight combinations. For each combination, we re-solve the routing problem and record (total_cost, avg_delivery_time, return_coverage). The resulting 2D scatter of cost vs. delivery time reveals the **Pareto front** — the set of solutions where no objective can be improved without worsening another.

The **knee point** of the Pareto front (the point of maximum curvature) represents the most balanced trade-off and is our recommended operating point for practical deployment.

### 3.5 Return Probability Prediction

Not all customers will return products, but we need to pre-assign pickup slots in the SDVRP before routes are finalised. We train an **XGBoost binary classifier** to predict the probability that an order will be returned.

**Target variable:** `is_return = 1` if `order_status ∈ {canceled, unavailable}` OR `actual_delivery_date > estimated_delivery_date + 7 days`.

**Feature set:**

| Feature | Type | Rationale |
|---------|------|-----------|
| product_category | Categorical | Electronics and fashion have higher return rates |
| product_weight_g | Continuous | Heavier items less likely to be returned (shipping cost) |
| freight_value | Continuous | High shipping cost correlates with delivery issues |
| seller_state | Categorical | Distant sellers have longer transit, more delays |
| days_late | Continuous | actual − estimated delivery date; strong predictor |
| review_score | Ordinal (1–5) | Low reviews correlate with returns |
| payment_type | Categorical | Credit card orders have easier return policies |
| order_value | Continuous | High-value orders may have stricter return behaviour |

**Class imbalance handling:** The return rate is approximately 5%, so we use `scale_pos_weight=15` in XGBoost to upweight the minority class. Evaluation uses AUC-ROC (target > 0.70) and PR-AUC (more informative for imbalanced data).

**Integration with routing:** Customers with `return_prob > 0.30` are flagged as pickup nodes in the SDVRP. Their expected return weight is estimated as `return_prob × product_weight_g`.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OLIST DATASET (9 CSVs)                      │
│  orders · items · products · customers · sellers · geolocation  │
│  payments · reviews · category_translation                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    DATA PIPELINE       │
              │  Merge → Filter (SP)   │
              │  → master_df.parquet   │
              └───────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   ┌─────────────────┐    ┌─────────────────────┐
   │   FORECASTING   │    │   RETURN PREDICTION  │
   │  Prophet / zone  │    │  XGBoost classifier  │
   │  weekly demand   │    │  return_prob / order  │
   └────────┬────────┘    └──────────┬───────────┘
            │                        │
            └───────────┬────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   CLUSTERING / PLACEMENT     │
         │  K-Means (primary)           │
         │  p-Median MILP (validation)  │
         │  → dark_stores_final.csv     │
         │  → Voronoi zone assignment   │
         └──────────────┬───────────────┘
                        │
         ┌──────────────┼───────────────┐
         ▼              ▼               ▼
  ┌────────────┐ ┌────────────┐ ┌──────────────┐
  │ FORWARD    │ │ REVERSE    │ │ SDVRP HYBRID │
  │ VRP        │ │ VRP        │ │ Delivery +   │
  │ OR-Tools   │ │ OR-Tools   │ │ Pickup       │
  │ CVRPTW     │ │ CVRPTW     │ │ combined     │
  └─────┬──────┘ └─────┬──────┘ └──────┬───────┘
        │              │               │
        └──────────────┼───────────────┘
                       ▼
          ┌──────────────────────────┐
          │   JOINT OPTIMISER        │
          │  Z = αCf + βCr + γT + δN │
          │  Pareto sweep (25 pts)   │
          │  → knee point solution   │
          └────────────┬─────────────┘
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
   ┌───────────┐ ┌──────────┐ ┌──────────────┐
   │ SCENARIOS │ │ DASHBOARD│ │ REPORT       │
   │ A / B / C │ │ Plotly   │ │ 10-12 pages  │
   │ stress    │ │ Folium   │ │ + 14 slides  │
   └───────────┘ └──────────┘ └──────────────┘
```

---

## 5. Dataset

**Olist Brazilian E-Commerce Public Dataset** — a real-world dataset of ~100,000 orders placed between 2016 and 2018 on the Olist marketplace in Brazil.

| Table | Rows (approx.) | Key Columns |
|-------|----------------|-------------|
| `orders` | 99,441 | order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_timestamp, order_estimated_delivery_date |
| `order_items` | 112,650 | order_id, product_id, seller_id, price, freight_value |
| `products` | 32,951 | product_id, product_category_name, product_weight_g, product_length/height/width_cm |
| `customers` | 99,441 | customer_id, customer_zip_code_prefix, customer_city, customer_state |
| `sellers` | 3,095 | seller_id, seller_zip_code_prefix, seller_city, seller_state |
| `geolocation` | 1,000,163 | geolocation_zip_code_prefix, geolocation_lat, geolocation_lng |
| `order_payments` | 103,886 | order_id, payment_type, payment_value |
| `order_reviews` | 99,224 | order_id, review_score, review_comment_message |
| `category_translation` | 71 | product_category_name, product_category_name_english |

**Scope:** We filter to `customer_state = 'SP'` (São Paulo), which contains ~41% of all orders. This provides a geographically coherent urban area suitable for dark store placement and last-mile routing, while keeping the problem computationally tractable.

**Join chain for master_df:**

```
orders
  ← order_items    ON order_id
  ← products       ON product_id
  ← customers      ON customer_id
  ← sellers        ON seller_id
  + geolocation    ON zip_code_prefix → median(lat, lon) per prefix
```

---

## 6. Step-by-Step Guide

### 6.1 Environment Setup

**Prerequisites:** Python 3.10+, pip (or uv)

```bash
# Clone the repository
git clone https://github.com/<your-org>/dark-store-logistics.git
cd dark-store-logistics

# Install dependencies
pip install -r requirements.txt

# Or with uv (recommended)
uv pip install -r requirements.txt
```

**requirements.txt contents:**
```
pandas>=2.0
numpy>=1.24
scipy>=1.11
geopandas>=0.14
scikit-learn>=1.3
xgboost>=2.0
prophet>=1.1
shap>=0.43
ortools>=9.8
pulp>=2.7
folium>=0.15
plotly>=5.18
matplotlib>=3.8
seaborn>=0.13
python-docx>=1.1
python-pptx>=0.6
openpyxl>=3.1
```

**Download the dataset:**
```bash
# Option 1: Kaggle CLI
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

# Option 2: Manual download
# Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Download and extract all 9 CSVs into data/raw/
```

**Verify OR-Tools installation:**
```python
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
print("OR-Tools version:", ortools.__version__)  # Should print >= 9.8
```

---

### 6.2 Stage 0 — Data Pipeline

**Script:** `notebooks/01_data_pipeline.ipynb` or `python src/data_pipeline.py`

**What it does:**

1. Loads all 9 Olist CSVs from `data/raw/`.
2. Merges them into a single denormalised DataFrame using the join chain described above.
3. Resolves geolocation by computing the median latitude and longitude per `zip_code_prefix` (the raw geolocation table has multiple entries per zip).
4. Filters to `customer_state = 'SP'`.
5. Engineers the `is_return` column.
6. Computes baseline KPIs: mean customer-to-nearest-seller distance, mean delivery days.

**Output:** `data/master_df.parquet` (~40K rows, ~15 columns)

**Key code:**
```python
master_df = (
    orders
    .merge(order_items, on="order_id")
    .merge(products, on="product_id")
    .merge(customers, on="customer_id")
    .merge(sellers, on="seller_id")
)

# Geolocation: median lat/lon per zip prefix
geo_median = (geolocation
    .groupby("geolocation_zip_code_prefix")
    .agg(lat=("geolocation_lat", "median"),
         lon=("geolocation_lng", "median"))
    .reset_index())

# Join customer and seller coordinates
master_df = master_df.merge(
    geo_median.rename(columns={"lat": "customer_lat", "lon": "customer_lon"}),
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix"
)

# Filter to São Paulo
master_df = master_df[master_df["customer_state"] == "SP"]

# Engineer return flag
master_df["is_return"] = (
    master_df["order_status"].isin(["canceled", "unavailable"]) |
    (master_df["delivery_days_late"] > 7)
).astype(int)
```

---

### 6.3 Stage 1 — Forecasting and Return Prediction

**Scripts:** `notebooks/04_return_ml.ipynb`, `src/return_classifier.py`

**Return classifier (XGBoost):**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

features = ["product_weight_g", "freight_value", "review_score",
            "days_late", "order_value"]
# product_category and seller_state are label-encoded

X_train, X_test, y_train, y_test = train_test_split(
    master_df[features], master_df["is_return"],
    test_size=0.2, stratify=master_df["is_return"], random_state=42
)

model = XGBClassifier(scale_pos_weight=15, max_depth=6,
                       learning_rate=0.1, n_estimators=200,
                       random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")  # Target > 0.70

# Add return_prob to master_df
master_df["return_prob"] = model.predict_proba(master_df[features])[:, 1]
```

**Demand forecasting (Prophet per zone):**

```python
from prophet import Prophet

# After dark store zones are assigned (Stage 2)
for zone_id in master_df["dark_store_id"].unique():
    zone_df = (master_df[master_df["dark_store_id"] == zone_id]
               .groupby(pd.Grouper(key="order_date", freq="W"))
               .size().reset_index(name="y")
               .rename(columns={"order_date": "ds"}))
    
    m = Prophet(weekly_seasonality=True)
    m.fit(zone_df)
    future = m.make_future_dataframe(periods=4, freq="W")
    forecast = m.predict(future)
```

**Output:** `data/master_df_v3.parquet` (with `return_prob` column), `data/forecasted_demand_by_zone.csv`

---

### 6.4 Stage 2 — Clustering and Dark Store Placement

**Scripts:** `notebooks/02_clustering.ipynb`, `src/clustering.py`

**K-Means with demand weighting:**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Weight by order volume per zip code
coords = master_df[["customer_lat", "customer_lon"]].values
weights = master_df.groupby("customer_zip_code_prefix").transform("count")["order_id"]

results = []
for k in range(3, 13):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords, sample_weight=weights)
    results.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(coords, labels)
    })

# Select K at elbow/silhouette agreement
optimal_k = ...  # Typically 5–8 for São Paulo
```

**p-Median validation (PuLP):**

```python
import pulp

def p_median(distances, demands, p):
    n_cust, n_fac = distances.shape
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts("x",
        [(i,j) for i in range(n_cust) for j in range(n_fac)], cat="Binary")
    y = pulp.LpVariable.dicts("y", range(n_fac), cat="Binary")
    
    prob += pulp.lpSum(
        distances[i][j] * demands[i] * x[i,j]
        for i in range(n_cust) for j in range(n_fac))
    
    for i in range(n_cust):
        prob += pulp.lpSum(x[i,j] for j in range(n_fac)) == 1
    for i in range(n_cust):
        for j in range(n_fac):
            prob += x[i,j] <= y[j]
    prob += pulp.lpSum(y[j] for j in range(n_fac)) == p
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [j for j in range(n_fac) if pulp.value(y[j]) > 0.5]
```

**Output:** `data/dark_stores_final.csv` (id, lat, lon, capacity, assigned_customers, coverage_pct), `data/dark_store_candidates.csv`

---

### 6.5 Stage 3 — Forward VRP

**Scripts:** `notebooks/03_forward_vrp.ipynb`, `src/haversine_matrix.py`

**Distance matrix construction:**

```python
from scipy.spatial.distance import cdist
import numpy as np

def haversine_matrix(coords):
    """Compute pairwise Haversine distances in km, return integer-scaled matrix."""
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    # ... Haversine formula ...
    return (distances_km * 1000).astype(int)  # OR-Tools needs integers

dist_matrix = haversine_matrix(sp_sample_coords)
np.save("data/distance_matrix.npy", dist_matrix)
```

**OR-Tools CVRPTW per zone:**

```python
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def solve_forward_vrp(dist_matrix, demands, time_windows,
                       num_vehicles, depot=0):
    n = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def dist_cb(i, j):
        return dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
    tid = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)

    # Capacity dimension
    def demand_cb(i):
        return demands[manager.IndexToNode(i)]
    did = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        did, 0, [500_000] * num_vehicles, True, "Capacity")

    # Time dimension
    routing.AddDimension(tid, 30, 1080, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node, (tw_open, tw_close) in enumerate(time_windows):
        if node == depot:
            continue
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(tw_open, tw_close)

    # Solve
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(params)
    return solution, routing, manager
```

Run `solve_forward_vrp` for each of the K dark store zones. Parse solutions into structured route DataFrames using `src/route_parser.py`.

**Output:** `outputs/forward_routes.json`, `outputs/forward_kpi_summary.csv`

---

### 6.6 Stage 4 — Reverse VRP

**Scripts:** `notebooks/05_reverse_vrp.ipynb`

Same CVRPTW structure as forward VRP, but:

- Only customers with `return_prob > 0.30` are included as pickup nodes.
- Demand represents **pickup weight** (product_weight_g of the expected return).
- Vehicles depart empty from the dark store and collect returns.
- Vehicle load increases at each stop (opposite of forward VRP).

**Output:** `outputs/reverse_routes.json`, `outputs/reverse_kpi_summary.csv`

---

### 6.7 Stage 5 — SDVRP Hybrid

**Scripts:** `notebooks/03_forward_vrp.ipynb` (extended section)

The SDVRP modifies the forward VRP by adding pickup demands at return-flagged nodes. The key constraint modification:

```python
# Each node has both delivery demand (positive) and pickup demand (negative load change)
# delivery_demand[i] = order_weight_g (dropped off)
# pickup_demand[i]   = return_weight_g (collected), non-zero only if return_prob > 0.30

def sdvrp_demand_cb(i):
    node = manager.IndexToNode(i)
    return delivery_demand[node]  # positive: adds to initial load

def sdvrp_pickup_cb(i):
    node = manager.IndexToNode(i)
    return pickup_demand[node]    # collected items

# Two dimensions: one for remaining deliveries, one for collected pickups
# Vehicle load at any point = initial_load - delivered_so_far + picked_up_so_far
# Must satisfy: 0 <= load <= Q at every stop
```

**Key comparison:** SDVRP total cost vs. (separate forward cost + separate reverse cost). The difference is the **integration saving**, expected at 15–25%.

**Output:** `outputs/hybrid_routes.json`, `outputs/hybrid_kpi_summary.csv`

---

### 6.8 Stage 6 — Joint Optimisation and Pareto Analysis

**Scripts:** `notebooks/06_joint_optimizer.ipynb`, `src/joint_optimizer.py`

```python
def compute_Z(fwd_cost, rev_cost, time_violations, n_vehicles,
              alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    # Normalise each component to [0, 1]
    max_cost = max(fwd_cost, rev_cost, 1)
    Z = (alpha * fwd_cost / max_cost +
         beta  * rev_cost / max_cost +
         gamma * time_violations / max(time_violations, 1) +
         delta * n_vehicles / max(n_vehicles, 1))
    return Z

def pareto_sweep(zones, solve_fn):
    alpha_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_vals  = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    for alpha in alpha_vals:
        for beta in beta_vals:
            gamma = delta = (1 - alpha - beta) / 2
            if gamma < 0:
                continue
            cost, time, coverage = solve_fn(zones, alpha, beta, gamma, delta)
            results.append({
                "alpha": alpha, "beta": beta,
                "total_cost": cost, "avg_time": time,
                "coverage": coverage
            })
    return pd.DataFrame(results)
```

Plot the 2D Pareto front (cost vs. delivery time) and identify the knee point using maximum curvature or the L-method.

**Output:** `outputs/pareto_results.csv`, `visualisations/pareto_tradeoff.png`

---

### 6.9 Stage 7 — Scenario Analysis

**Scripts:** `notebooks/07_scenarios_and_analysis.ipynb`

| Scenario | Modification | Purpose |
|----------|-------------|---------|
| A (Base) | No change | Baseline reference |
| B (Surge +30%) | All demand weights × 1.3 | Simulate a flash sale or seasonal peak |
| C (High Returns ×2) | Return probability threshold halved (0.15 instead of 0.30) | Stress-test reverse logistics capacity |

For each scenario, regenerate `vrp_nodes.csv`, re-run the joint optimizer, and record the KPI table.

**Key questions answered:**

- Does Scenario B make any zone infeasible (demand exceeds vehicle fleet capacity)?
- How much does reverse cost increase in Scenario C? Does the SDVRP saving still hold?
- Which dark store zone is the most sensitive to demand perturbation?

**Output:** `outputs/scenario_results_table.csv` (3 scenarios × 6 KPIs)

---

### 6.10 Stage 8 — Visualisation and Dashboard

**Scripts:** Varsha's visualisation notebooks, `visualisations/` folder

**Folium maps:**

- `sp_eda_map.html` — Customer density heatmap with seller markers
- `dark_store_map.html` — Dark store locations with Voronoi zone boundaries
- `forward_routes_map.html` — Vehicle routes as coloured polylines
- `return_heatmap.html` — Return probability density

**Plotly dashboard (2×3 subplot figure):**

1. Cost comparison bar chart (naive / separate / hybrid)
2. Pareto front with knee point highlighted
3. Return probability histogram
4. Coverage choropleth per zone
5. Route distance box plots per zone
6. Scenario radar chart (A/B/C across 6 KPIs)

**Output:** `visualisations/final_dashboard.html` (standalone, no server required)

---

## 7. Key Metrics

| Metric | Target | Baseline Comparison |
|--------|--------|---------------------|
| ↓ Total routing cost | 20–40% reduction | Naive: direct seller → customer, no clustering |
| ↓ Avg delivery distance | 25–35% reduction | Naive average distance (km) |
| ↑ Return efficiency | 15–25% cost saving | Separate forward + reverse fleets |
| ↑ Dark store coverage | > 70% within 5 km | 0% (no dark stores) |
| Return classifier AUC | > 0.70 | Random baseline = 0.50 |
| Route consolidation | > 30% routes with ≥1 pickup | 0% (separate systems) |
| Scenario B robustness | All zones feasible under +30% demand | N/A |

---

## 8. Project Structure

```
project-root/
├── data/
│   ├── raw/                         # 9 Olist CSVs
│   ├── master_df_v3.parquet         # Final merged dataset
│   ├── dark_stores_final.csv        # K dark store locations
│   ├── vrp_nodes.csv                # Full node list for OR-Tools
│   ├── vrp_nodes_A.csv              # Scenario A (base)
│   ├── vrp_nodes_B.csv              # Scenario B (surge +30%)
│   ├── vrp_nodes_C.csv              # Scenario C (high returns)
│   ├── distance_matrix.npy          # 500×500 Haversine (integer-scaled)
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
│   ├── data_pipeline.py
│   ├── clustering.py
│   ├── haversine_matrix.py
│   ├── return_classifier.py
│   ├── route_parser.py
│   └── joint_optimizer.py
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
│   └── master_results.json
├── visualisations/
│   ├── sp_eda_map.html
│   ├── dark_store_map.html
│   ├── forward_routes_map.html
│   ├── return_heatmap.html
│   ├── final_dashboard.html
│   └── all_charts/                  # 6 PNGs for report
├── report/
│   ├── report_draft_v1.docx
│   └── report_final.pdf
├── presentation/
│   ├── presentation_v1.pptx
│   └── presentation_final.pdf
├── RESULTS.md
├── README.md
├── requirements.txt
└── run_all.sh
```

**Full pipeline execution:**

```bash
chmod +x run_all.sh
./run_all.sh
# Runs all stages in dependency order
# Expected runtime: ~15-25 minutes on a modern laptop
```

---

## 9. References

1. **Dethloff, J. (2001).** Vehicle routing and reverse logistics: The vehicle routing problem with simultaneous delivery and pick-up. *OR Spektrum*, 23, 113–132. — Canonical SDVRP formulation; directly supports our hybrid routing approach.

2. **Toth, P. & Vigo, D. (2014).** *Vehicle Routing: Problems, Methods, and Applications.* SIAM. — Comprehensive VRP reference; CVRPTW formulations and solution methods.

3. **Guide, V.D.R. & Van Wassenhove, L.N. (2009).** The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10–18. — Foundational work on reverse logistics integration.

4. **Islam, S., Amin, S.H. & Wardley, L.J. (2022).** Supplier selection and order allocation with predictive analytics and multi-objective programming. *Computers & Industrial Engineering*, 174, 108825. — 3-stage pipeline precedent (forecast → evaluate → optimise).

5. **Mohammed, A. et al. (2019).** A hybrid MCDM-FMOO approach for sustainable supplier selection. *International Journal of Production Economics*, 217, 171–184. — Multi-criteria decision-making with multi-objective optimisation.

6. **Wasi, A.T. et al. (2024).** SupplyGraph: A benchmark dataset for graph-based supply chain planning. *arXiv:2401.15299*. — Graph-native supply chain dataset for benchmarking.

7. **Google OR-Tools.** *Vehicle Routing Problem with Time Windows.* [developers.google.com/optimization/routing/vrptw](https://developers.google.com/optimization/routing/vrptw) — Implementation reference for CVRPTW solver.

---

## 10. Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Pritam Sarkar** ★★ | Lead · Claude Code | Architecture · Forward VRP · SDVRP · Joint Optimizer · Pareto Sweep · Report · Pipeline |
| **Vybhav** ★ | Co-Lead Coder · Gemini Pro | Data Pipeline · p-Median · Return ML · Reverse VRP · Scenarios · Code Cleanup |
| **Anurag** | Support · Gemini Pro | EDA · SP Filtering · Prophet Forecasting · Scenario Data Prep · Results Aggregation |
| **Sneha** | Support · Gemini Pro | K-Means Clustering · Dark Store Selection · Sensitivity Analysis · Literature Review |
| **Pranav** | Support · Gemini Pro | VRP Node List · Route Parser · Baseline Comparison · Forward KPI Metrics |
| **Varsha** | Support · Gemini Pro | Folium Maps · Plotly Dashboard · Presentations · Report Assembly · Dry Run Lead |

---

*Built as part of the PGDBA programme (ISI Kolkata / IIM Calcutta / IIT Kharagpur). April 2026.*
