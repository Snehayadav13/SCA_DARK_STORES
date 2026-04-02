"""
OR-Tools CVRPTW Toy Example — 10-node problem
==============================================
confirm OR-Tools is installed and working end-to-end.

Problem:
    1 depot (node 0) + 9 customer stops
    2 vehicles, capacity 15 units each
    Time windows [earliest, latest] in minutes from midnight
    Travel times via Haversine distance (km → minutes at 30 km/h average)

Run:
    uv run python src/ortools_toy_cvrptw.py
    OR:
    python src/ortools_toy_cvrptw.py

Expected output (approximate):
    Objective: <total distance> m
    Route for vehicle 0: 0 -> ... -> 0
    Route for vehicle 1: 0 -> ... -> 0
    [PASS] OR-Tools CVRPTW confirmed working.
"""

from __future__ import annotations

import math
import sys

# ------------------------------------------------------------------
# Toy data
# ------------------------------------------------------------------
# Coordinates: (lat, lon) — Sao Paulo area
COORDS = [
    (-23.5505, -46.6333),   # 0: Depot (city centre)
    (-23.5475, -46.6361),   # 1
    (-23.5530, -46.6290),   # 2
    (-23.5460, -46.6200),   # 3
    (-23.5600, -46.6400),   # 4
    (-23.5520, -46.6500),   # 5
    (-23.5440, -46.6100),   # 6
    (-23.5580, -46.6250),   # 7
    (-23.5490, -46.6450),   # 8
    (-23.5550, -46.6180),   # 9
]

# Demand per node (depot = 0)
DEMANDS = [0, 3, 4, 2, 5, 3, 2, 4, 3, 2]

# Time windows [open, close] in minutes from midnight
TIME_WINDOWS = [
    (0,   1440),   # 0: Depot — always open
    (480,  540),   # 1
    (490,  560),   # 2
    (500,  600),   # 3
    (510,  620),   # 4
    (480,  580),   # 5
    (530,  650),   # 6
    (520,  640),   # 7
    (540,  660),   # 8
    (500,  600),   # 9
]

VEHICLE_CAPACITY = 15
NUM_VEHICLES = 2
DEPOT = 0
AVG_SPEED_KMH = 30       # for travel-time estimate
INT_SCALE = 1000         # km → integer metres equivalent for OR-Tools


# ------------------------------------------------------------------
# Haversine distance (km), integer-scaled
# ------------------------------------------------------------------

def _haversine_km(p1: tuple, p2: tuple) -> float:
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def build_distance_matrix(coords: list[tuple]) -> list[list[int]]:
    """Return integer distance matrix (km × INT_SCALE)."""
    n = len(coords)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = int(_haversine_km(coords[i], coords[j]) * INT_SCALE)
    return mat


def build_time_matrix(dist_matrix: list[list[int]]) -> list[list[int]]:
    """Convert integer distance matrix to integer travel time (minutes × INT_SCALE)."""
    n = len(dist_matrix)
    # dist in km = dist_matrix[i][j] / INT_SCALE
    # time in min = (km / speed_kmh) * 60
    factor = 60.0 / AVG_SPEED_KMH  # minutes per km
    return [
        [int(dist_matrix[i][j] / INT_SCALE * factor * INT_SCALE) for j in range(n)]
        for i in range(n)
    ]


# ------------------------------------------------------------------
# OR-Tools CVRPTW solver
# ------------------------------------------------------------------

def solve_cvrptw() -> bool:
    """
    Solve the toy CVRPTW problem and print the solution.
    Returns True if a solution is found, False otherwise.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        print("[FAIL] ortools not installed. Run: uv add ortools")
        return False

    dist_matrix = build_distance_matrix(COORDS)
    time_matrix = build_time_matrix(dist_matrix)
    n = len(COORDS)

    # --- Manager and Model ---
    manager = pywrapcp.RoutingIndexManager(n, NUM_VEHICLES, DEPOT)
    routing = pywrapcp.RoutingModel(manager)

    # --- Distance callback ---
    def distance_callback(from_index: int, to_index: int) -> int:
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- Capacity dimension ---
    def demand_callback(from_index: int) -> int:
        return DEMANDS[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                     # no slack
        [VEHICLE_CAPACITY] * NUM_VEHICLES,
        True,                  # start cumul at zero
        "Capacity",
    )

    # --- Time window dimension ---
    def time_callback(from_index: int, to_index: int) -> int:
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        30 * INT_SCALE,        # max wait time (30 min slack)
        1440 * INT_SCALE,      # max total time (24 h)
        False,                 # don't force start at 0
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        index = manager.NodeToIndex(node)
        open_t  = TIME_WINDOWS[node][0] * INT_SCALE
        close_t = TIME_WINDOWS[node][1] * INT_SCALE
        time_dim.CumulVar(index).SetRange(open_t, close_t)

    # --- Search parameters ---
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = 10

    # --- Solve ---
    solution = routing.SolveWithParameters(params)

    if not solution:
        print("[FAIL] No solution found.")
        return False

    # --- Print solution ---
    print(f"\nObjective (total scaled distance): {solution.ObjectiveValue()}")
    total_dist_km = 0.0

    for vehicle in range(NUM_VEHICLES):
        index = routing.Start(vehicle)
        route_str = ""
        route_dist = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_str += f"{node} -> "
            next_index = solution.Value(routing.NextVar(index))
            route_dist += routing.GetArcCostForVehicle(index, next_index, vehicle)
            index = next_index
        node = manager.IndexToNode(index)
        route_str += str(node)
        km = route_dist / INT_SCALE
        total_dist_km += km
        print(f"Route for vehicle {vehicle}: {route_str}   ({km:.3f} km)")

    print(f"Total distance: {total_dist_km:.3f} km")
    print("\n[PASS] OR-Tools CVRPTW confirmed working.")
    return True


if __name__ == "__main__":
    success = solve_cvrptw()
    sys.exit(0 if success else 1)
