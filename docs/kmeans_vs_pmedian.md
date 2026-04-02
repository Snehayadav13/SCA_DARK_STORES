# K-Means vs p-Median: A Comparison Note for Dark Store Placement

**Author:** Sneha  
**Project:** Dark Store Placement + Integrated Forward & Reverse Logistics Optimisation

---

## 1. Why we need to place dark stores carefully

A dark store (micro-fulfilment centre) can only serve customers within a reasonable radius — usually 5 km for fast delivery. If we place it in the wrong location, we either leave large customer clusters unserved, or we open too many stores and waste money.

Facility location theory asks: *given a set of customer locations with different demand levels, where should we open exactly K warehouses to minimise the total distance customers must travel to their nearest warehouse?*

Two main methods address this:

---

## 2. Method 1 — K-Means Clustering

### What it does
K-Means partitions all customer points into K groups (clusters) by minimising the **within-cluster sum of squared distances**:

$$\min \sum_{k=1}^{K} \sum_{x_i \in S_k} \| x_i - \mu_k \|^2$$

The centroid of each cluster (its geographic centre of gravity) becomes the candidate dark store location.

### How we use it
- We weight each customer point by their order volume, so high-demand areas attract centroids toward them.
- We run K-Means for K = 2 through 10 and pick the K with the best agreement between the **elbow curve** (inertia) and the **silhouette score** (cluster separation quality).

### Strengths
| Strength | Why it matters for dark stores |
|---|---|
| Very fast | Runs in seconds even with 40,000 São Paulo customers |
| Demand-weighted | High-demand zip codes pull the centroid toward them |
| Intuitive output | Centroids map directly to "open a dark store here" |
| Easy to visualise | Colour-coded clusters on a map are easy to explain to stakeholders |

### Limitations
| Limitation | Impact |
|---|---|
| Minimises Euclidean distance, not actual road distance | Centroid might be placed in a river or inside a protected park |
| Sensitive to outliers | One cluster of customers very far away can pull a centroid off-centre |
| Assumes spherical clusters | Real demand patterns may be irregular (e.g. along a metro line) |
| Not mathematically optimal | Does not guarantee the minimum possible total weighted distance |

---

## 3. Method 2 — p-Median MILP (Mixed-Integer Linear Programming)

### What it does
The p-median problem is the academically rigorous formulation. We define:
- **n** candidate facility locations (we use a grid over São Paulo, or K-Means centroids as candidates)
- **m** customer demand points with weights $w_i$
- **$d_{ij}$** = Haversine distance from customer $i$ to candidate facility $j$

We solve:

$$\min \sum_{i=1}^{m} \sum_{j=1}^{n} d_{ij} \cdot w_i \cdot x_{ij}$$

subject to:
- Each customer is assigned to exactly one facility
- A customer can only be assigned to an open facility
- Exactly p facilities are opened

This is solved with **PuLP** (a Python LP library) using the CBC solver.

### Strengths
| Strength | Why it matters |
|---|---|
| Provably optimal | CBC solver finds the minimum total weighted distance |
| Uses real Haversine distance | More geographically accurate than Euclidean |
| Hard constraint on number of stores | Exactly p facilities, no more, no less |
| Academically defensible | Standard formulation in OR literature (Hakimi, 1964) |

### Limitations
| Limitation | Impact |
|---|---|
| Computationally expensive | With 40K customers and 100 candidates, solver can take minutes |
| Requires candidate set | We must pre-specify where a dark store *could* go |
| NP-hard in general | For very large instances, exact solution is impractical |
| Less interpretable | Binary decision variables harder to explain to non-technical stakeholders |

---

## 4. Side-by-side comparison

| Dimension | K-Means | p-Median MILP |
|---|---|---|
| **Speed** | Seconds | Minutes |
| **Solution quality** | Good heuristic | Provably optimal |
| **Distance metric** | Euclidean (or weighted) | Haversine (geographic) |
| **Number of stores** | Chosen via elbow/silhouette | Exact p is an input |
| **Use in our project** | Primary placement method | Validation layer |
| **Tool** | scikit-learn KMeans | PuLP (CBC solver) |
| **Output** | Centroids = dark store locations | Binary decision variables → open facilities |
| **Literature basis** | Lloyd (1982) | Hakimi (1964), ReVelle & Swain (1970) |

---

## 5. How we use both in this project

We use a **two-stage approach**:

1. **K-Means first** — fast, visual, demand-weighted. Gives us candidate dark store locations quickly. We use the elbow + silhouette method to pick K.

2. **p-Median second** — uses K-Means centroids as the candidate set and validates whether the optimal MILP solution agrees. If both methods place a dark store within ~1 km of each other, we have **high confidence** in the location.

This is the same "fast heuristic + rigorous validation" pattern used in operations research for real supply chain problems (see: Brandeau & Chiu, 1989; Daskin, 1995).

---

## 6. Literature positioning

> *"Most facility location papers treat K-Means and p-Median as competing methods. We treat them as complementary: K-Means for speed and stakeholder communication, p-Median for rigorous academic validation. This mirrors standard practice in applied OR (Farahani et al., 2010)."*

### Key references

1. **Hakimi, S.L. (1964).** Optimum locations of switching centres. *Operations Research*, 12(3), 450–459.  
   → Original formulation of the p-median problem.

2. **Lloyd, S. (1982).** Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.  
   → Standard K-Means algorithm proof.

3. **Farahani, R.Z. et al. (2010).** Covering problems in facility location: A review. *Computers & Industrial Engineering*, 62(1), 368–407.  
   → Reviews when to use coverage vs median models; directly supports our 5 km coverage target.

4. **Dethloff, J. (2001).** Vehicle routing and reverse logistics: The SDVRP with simultaneous delivery and pick-up. *OR Spektrum*, 23, 113–132.  
   → After facility placement, routes are designed using SDVRP — this paper is the foundation.

5. **Brandeau, M.L. & Chiu, S.S. (1989).** An overview of representative problems in location research. *Management Science*, 35(6), 645–674.  
   → Excellent survey; directly compares p-median to clustering heuristics.

---

## 7. Conclusion

For our project, **K-Means is the right primary tool** because:
- It is fast enough to run interactively during team sessions
- It produces intuitive, visual output (maps of zones) stakeholders can understand
- Demand-weighting ensures high-order zones get a dark store

**p-Median is the right validation tool** because:
- It gives us a provably optimal benchmark to compare against
- It is the academically accepted method in OR facility location literature
- If K-Means and p-Median agree within 1 km, we can confidently report the dark store placement in our final report

---

*This document is part of the Day 1 deliverables for Sneha's clustering workstream.*  
*For the full pipeline, see: `src/clustering.py` and `notebooks/02_clustering.ipynb`.*
