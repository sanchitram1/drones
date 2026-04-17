from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer


DEFAULT_RESTAURANTS_CENSUS_CRS = "EPSG:4326"
DEFAULT_GRAPH_CRS = "EPSG:32610"


@dataclass
class RealDataBundle:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    census: pd.DataFrame
    restaurants: pd.DataFrame
    graph: nx.Graph
    topology: "RealDataTopology"


class RealDataTopology:
    """
    Lightweight adapter around the exported SF node/edge CSV graph that
    exposes the topology interface used by the optimizer.
    """

    def __init__(self, graph: nx.Graph):
        self.G = graph
        self.node_positions = {
            node: (float(data["x"]), float(data["y"]))
            for node, data in graph.nodes(data=True)
        }
        self.node_list = list(graph.nodes())
        self.n_nodes = len(self.node_list)

    def get_position(self, node_id) -> Tuple[float, float]:
        return self.node_positions[node_id]

    def compute_heading(self, from_node, to_node) -> float:
        x1, y1 = self.get_position(from_node)
        x2, y2 = self.get_position(to_node)
        dx, dy = x2 - x1, y2 - y1
        return float(np.degrees(np.arctan2(dx, dy)) % 360)

    def shortest_path(self, origin, destination) -> list:
        try:
            return nx.shortest_path(self.G, origin, destination, weight="length_m")
        except nx.NetworkXNoPath:
            return []


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def safe_corridor_height_ft(edges: pd.DataFrame) -> np.ndarray:
    if "corridor_height_ft" in edges.columns:
        return (
            pd.to_numeric(edges["corridor_height_ft"], errors="coerce")
            .fillna(40.0)
            .to_numpy()
        )
    if "corridor_height_m" in edges.columns:
        return (
            pd.to_numeric(edges["corridor_height_m"], errors="coerce")
            .fillna(12.0)
            .to_numpy()
            / 0.3048
        )
    return np.full(len(edges), 40.0)


def project_latlon_to_graph_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    restaurants_census_crs: str = DEFAULT_RESTAURANTS_CENSUS_CRS,
    graph_crs: str = DEFAULT_GRAPH_CRS,
) -> Tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        restaurants_census_crs,
        graph_crs,
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def load_real_data(
    data_dir: str | Path = "data",
    restaurants_census_crs: str = DEFAULT_RESTAURANTS_CENSUS_CRS,
    graph_crs: str = DEFAULT_GRAPH_CRS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    nodes_path = data_dir / "sf_nodes.csv"
    edges_path = data_dir / "sf_edges.csv"
    census_path = data_dir / "sf_census.csv"
    restaurants_path = data_dir / "sf_restaurants.csv"

    missing = [
        str(path)
        for path in [nodes_path, edges_path, census_path, restaurants_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required exported files:\n  - " + "\n  - ".join(missing)
        )

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    census = pd.read_csv(census_path)
    restaurants = pd.read_csv(restaurants_path)

    nodes["node_id"] = normalize_id_series(nodes["node_id"])
    edges["u"] = normalize_id_series(edges["u"])
    edges["v"] = normalize_id_series(edges["v"])

    node_set = set(nodes["node_id"])
    edges = edges[edges["u"].isin(node_set) & edges["v"].isin(node_set)].copy()
    edges["length_m"] = pd.to_numeric(edges["length_m"], errors="coerce")
    edges = edges.dropna(subset=["length_m"])
    edges = edges[edges["u"] != edges["v"]]

    if not {"lat", "lon"}.issubset(restaurants.columns):
        raise ValueError("sf_restaurants.csv must contain lat/lon columns")
    if not {"lat", "lon"}.issubset(census.columns):
        raise ValueError("sf_census.csv must contain lat/lon columns")

    restaurants["x"], restaurants["y"] = project_latlon_to_graph_xy(
        restaurants["lat"].to_numpy(),
        restaurants["lon"].to_numpy(),
        restaurants_census_crs=restaurants_census_crs,
        graph_crs=graph_crs,
    )
    census["x"], census["y"] = project_latlon_to_graph_xy(
        census["lat"].to_numpy(),
        census["lon"].to_numpy(),
        restaurants_census_crs=restaurants_census_crs,
        graph_crs=graph_crs,
    )

    return nodes, edges, census, restaurants


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.node_id, x=float(row.x), y=float(row.y))

    corridor_ft = safe_corridor_height_ft(edges)
    for row, height_ft in zip(edges.itertuples(index=False), corridor_ft):
        if graph.has_edge(row.u, row.v):
            if float(row.length_m) < graph[row.u][row.v]["length_m"]:
                graph[row.u][row.v].update(
                    length_m=float(row.length_m),
                    corridor_height_ft=float(height_ft),
                )
        else:
            graph.add_edge(
                row.u,
                row.v,
                length_m=float(row.length_m),
                corridor_height_ft=float(height_ft),
            )

    return graph


def load_real_data_bundle(
    data_dir: str | Path = "data",
    restaurants_census_crs: str = DEFAULT_RESTAURANTS_CENSUS_CRS,
    graph_crs: str = DEFAULT_GRAPH_CRS,
) -> RealDataBundle:
    nodes, edges, census, restaurants = load_real_data(
        data_dir=data_dir,
        restaurants_census_crs=restaurants_census_crs,
        graph_crs=graph_crs,
    )
    graph = build_graph(nodes, edges)
    topology = RealDataTopology(graph)
    return RealDataBundle(
        nodes=nodes,
        edges=edges,
        census=census,
        restaurants=restaurants,
        graph=graph,
        topology=topology,
    )


def nearest_node_ids(
    points_xy: np.ndarray,
    nodes_xy: np.ndarray,
    node_ids: np.ndarray,
) -> np.ndarray:
    out = []
    batch = 128
    for i in range(0, len(points_xy), batch):
        points = points_xy[i : i + batch]
        d2 = ((points[:, None, :] - nodes_xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        out.extend(node_ids[idx])
    return np.asarray(out)


def compute_sampling_weights(
    df: pd.DataFrame,
    column: Optional[str],
    label: str,
) -> np.ndarray:
    if column is None:
        return np.full(len(df), 1.0 / max(len(df), 1))
    if column not in df.columns:
        raise ValueError(
            f"{label} weight column '{column}' was not found in the input data."
        )

    weights = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy()
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            f"{label} weight column '{column}' has no positive numeric values."
        )
    return weights / total


def sample_od_records(
    census: pd.DataFrame,
    restaurants: pd.DataFrame,
    graph_or_topology,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    origin_weight_col: Optional[str] = None,
    dest_weight_col: Optional[str] = "pop_density",
    dest_jitter_m: float = 100.0,
) -> pd.DataFrame:
    if n_samples <= 0:
        return pd.DataFrame(
            columns=[
                "origin_node",
                "dest_node",
                "orig_x",
                "orig_y",
                "dest_x",
                "dest_y",
            ]
        )

    rng = rng or np.random.default_rng(42)
    graph = (
        graph_or_topology.G if hasattr(graph_or_topology, "G") else graph_or_topology
    )
    node_ids = np.array(list(graph.nodes()))
    nodes_xy = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in node_ids])

    origin_weights = compute_sampling_weights(
        restaurants,
        origin_weight_col,
        "Origin",
    )
    dest_weights = compute_sampling_weights(
        census,
        dest_weight_col,
        "Destination",
    )

    rest_idx = rng.choice(len(restaurants), size=n_samples, p=origin_weights)
    origin_xy = restaurants[["x", "y"]].to_numpy()[rest_idx]

    dest_idx = rng.choice(len(census), size=n_samples, p=dest_weights)
    dest_xy = census[["x", "y"]].to_numpy()[dest_idx].copy()
    dest_xy += rng.normal(0.0, dest_jitter_m, size=dest_xy.shape)

    origin_nodes = nearest_node_ids(origin_xy, nodes_xy, node_ids)
    dest_nodes = nearest_node_ids(dest_xy, nodes_xy, node_ids)

    same = origin_nodes == dest_nodes
    while np.any(same):
        resample = rng.choice(len(census), size=same.sum(), p=dest_weights)
        dest_xy[same] = census[["x", "y"]].to_numpy()[resample] + rng.normal(
            0.0,
            dest_jitter_m,
            size=(same.sum(), 2),
        )
        dest_nodes[same] = nearest_node_ids(dest_xy[same], nodes_xy, node_ids)
        same = origin_nodes == dest_nodes

    return pd.DataFrame(
        {
            "origin_node": origin_nodes,
            "dest_node": dest_nodes,
            "orig_x": origin_xy[:, 0],
            "orig_y": origin_xy[:, 1],
            "dest_x": dest_xy[:, 0],
            "dest_y": dest_xy[:, 1],
        }
    )


def generate_request_times(
    demand_model: str,
    sim_duration_s: int,
    dt_s: int,
    base_lambda_per_min: float,
    peak_multiplier: float,
    peak_window_s: Tuple[int, int],
    n_orders_override: Optional[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(42)

    if demand_model == "fixed-count":
        if n_orders_override is None or n_orders_override <= 0:
            raise ValueError("Fixed-count demand requires n_orders_override > 0.")
        return np.sort(rng.uniform(0, sim_duration_s, n_orders_override))

    if demand_model != "time-series":
        raise ValueError(f"Unsupported generated demand model '{demand_model}'.")

    n_steps = sim_duration_s // dt_s
    req_times = []
    for t_idx in range(n_steps):
        t0 = t_idx * dt_s
        lam_per_min = base_lambda_per_min
        if peak_window_s[0] <= t0 < peak_window_s[1]:
            lam_per_min *= peak_multiplier
        lam_per_min *= 1.0 + 0.25 * math.sin(2 * math.pi * t0 / sim_duration_s)
        arrivals = rng.poisson(max(lam_per_min, 0.05) * (dt_s / 60.0))
        if arrivals > 0:
            req_times.extend(t0 + rng.uniform(0, dt_s, arrivals))
    return np.sort(np.asarray(req_times))


def generate_orders(
    census: pd.DataFrame,
    restaurants: pd.DataFrame,
    graph_or_topology,
    demand_model: str = "time-series",
    sim_duration_s: int = 2 * 3600,
    dt_s: int = 30,
    base_lambda_per_min: float = 6.0,
    peak_multiplier: float = 1.8,
    peak_window_s: Tuple[int, int] = (45 * 60, 95 * 60),
    n_orders_override: Optional[int] = None,
    origin_weight_col: Optional[str] = None,
    dest_weight_col: Optional[str] = "pop_density",
    dest_jitter_m: float = 100.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    req_times = generate_request_times(
        demand_model=demand_model,
        sim_duration_s=sim_duration_s,
        dt_s=dt_s,
        base_lambda_per_min=base_lambda_per_min,
        peak_multiplier=peak_multiplier,
        peak_window_s=peak_window_s,
        n_orders_override=n_orders_override,
        rng=rng,
    )
    od_records = sample_od_records(
        census=census,
        restaurants=restaurants,
        graph_or_topology=graph_or_topology,
        n_samples=len(req_times),
        rng=rng,
        origin_weight_col=origin_weight_col,
        dest_weight_col=dest_weight_col,
        dest_jitter_m=dest_jitter_m,
    )
    od_records.insert(0, "request_time_s", req_times)
    od_records.insert(0, "order_id", np.arange(len(req_times)))
    return od_records


def load_orders_csv(orders_path: str | Path, graph_or_topology) -> pd.DataFrame:
    orders = pd.read_csv(orders_path).copy()

    if "request_time_s" not in orders.columns and "request_time" in orders.columns:
        orders = orders.rename(columns={"request_time": "request_time_s"})

    required = {"origin_node", "dest_node", "request_time_s"}
    missing = sorted(required - set(orders.columns))
    if missing:
        raise ValueError(
            f"Orders CSV is missing required columns: {', '.join(missing)}."
        )

    if "order_id" not in orders.columns:
        orders["order_id"] = np.arange(len(orders))

    orders["origin_node"] = normalize_id_series(orders["origin_node"])
    orders["dest_node"] = normalize_id_series(orders["dest_node"])
    orders["request_time_s"] = pd.to_numeric(orders["request_time_s"], errors="coerce")

    if orders["request_time_s"].isna().any():
        raise ValueError("Orders CSV contains non-numeric request_time_s values.")

    graph = (
        graph_or_topology.G if hasattr(graph_or_topology, "G") else graph_or_topology
    )
    graph_nodes = set(graph.nodes())
    unknown_origins = sorted(set(orders["origin_node"]) - graph_nodes)
    unknown_destinations = sorted(set(orders["dest_node"]) - graph_nodes)
    if unknown_origins or unknown_destinations:
        problems = []
        if unknown_origins:
            problems.append(f"unknown origin nodes: {', '.join(unknown_origins[:5])}")
        if unknown_destinations:
            problems.append(
                f"unknown destination nodes: {', '.join(unknown_destinations[:5])}"
            )
        raise ValueError(
            "Orders CSV references nodes not present in the graph: "
            + "; ".join(problems)
        )

    if not {"orig_x", "orig_y"}.issubset(orders.columns):
        orders["orig_x"] = orders["origin_node"].map(lambda n: graph.nodes[n]["x"])
        orders["orig_y"] = orders["origin_node"].map(lambda n: graph.nodes[n]["y"])
    if not {"dest_x", "dest_y"}.issubset(orders.columns):
        orders["dest_x"] = orders["dest_node"].map(lambda n: graph.nodes[n]["x"])
        orders["dest_y"] = orders["dest_node"].map(lambda n: graph.nodes[n]["y"])

    return orders
