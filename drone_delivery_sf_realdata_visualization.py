import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from realdata import (
    build_graph as build_realdata_graph,
    generate_orders as generate_realdata_orders,
    load_orders_csv as load_orders_csv_shared,
    load_real_data as load_realdata,
    project_latlon_to_graph_xy as project_latlon_to_graph_xy_shared,
)

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


@dataclass
class SimConfig:
    data_dir: str = "data"
    random_seed: int = 42
    turn_protocol: str = "normal"
    lane_type: str = "normal"
    demand_model: str = "time-series"

    sim_duration_s: int = 2 * 3600
    dt_s: int = 30
    base_lambda_per_min: float = 6.0
    peak_multiplier: float = 1.8
    peak_window_s: Tuple[int, int] = (45 * 60, 95 * 60)
    n_orders_override: Optional[int] = None
    origin_weight_col: Optional[str] = None
    dest_weight_col: Optional[str] = "pop_density"
    dest_jitter_m: float = 100.0

    cruise_speed_ft_s: float = 35.0
    climb_rate_ft_s: float = 8.0
    descend_rate_ft_s: float = 7.0
    turn_time_s: float = 4.0
    max_altitude_ft: float = 400.0
    building_clearance_ft: float = 5.0
    collision_radius_ft: float = 10.0

    alt_north_ft: float = 300.0
    alt_south_ft: float = 315.0
    alt_east_ft: float = 330.0
    alt_west_ft: float = 345.0
    alt_turn_ft: float = 370.0

    edge_time_headway_s: float = 3.0
    intersection_headway_s: float = 4.0
    max_demo_routes_3d: int = 8
    max_demo_routes_map: int = 120

    sphereabout_radius_m: float = 30.0
    sphereabout_points: int = 20
    intersection_lane_min_angle_deg: float = 35.0
    intersection_lane_max_angle_deg: float = 145.0
    intersection_lane_max_length_multiplier: float = 2.2

    restaurants_census_crs: str = "EPSG:4326"
    graph_crs: str = "EPSG:32610"


SUPPORTED_TURN_PROTOCOLS = {"normal", "sphereabout"}
SUPPORTED_LANE_TYPES = {"normal", "intersection"}
SUPPORTED_DEMAND_MODELS = {"time-series", "fixed-count", "csv"}


def ft_to_m(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x) * 0.3048 if isinstance(x, (list, tuple, np.ndarray)) else x * 0.3048


def m_to_ft(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x) / 0.3048 if isinstance(x, (list, tuple, np.ndarray)) else x / 0.3048


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def safe_corridor_height_ft(edges: pd.DataFrame) -> np.ndarray:
    if "corridor_height_ft" in edges.columns:
        return pd.to_numeric(edges["corridor_height_ft"], errors="coerce").fillna(40.0).to_numpy()
    if "corridor_height_m" in edges.columns:
        return m_to_ft(pd.to_numeric(edges["corridor_height_m"], errors="coerce").fillna(12.0).to_numpy())
    return np.full(len(edges), 40.0)


def project_latlon_to_graph_xy(lat: np.ndarray, lon: np.ndarray, cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    return project_latlon_to_graph_xy_shared(
        lat,
        lon,
        restaurants_census_crs=cfg.restaurants_census_crs,
        graph_crs=cfg.graph_crs,
    )


def load_real_data(cfg: SimConfig):
    return load_realdata(
        data_dir=cfg.data_dir,
        restaurants_census_crs=cfg.restaurants_census_crs,
        graph_crs=cfg.graph_crs,
    )


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    return build_realdata_graph(nodes, edges)


def nearest_node_ids(points_xy: np.ndarray, nodes_xy: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    out = []
    batch = 128
    for i in range(0, len(points_xy), batch):
        P = points_xy[i : i + batch]
        d2 = ((P[:, None, :] - nodes_xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        out.extend(node_ids[idx])
    return np.asarray(out)


def normalize_choice(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def resolve_turn_protocol(turn_protocol: str) -> str:
    normalized = normalize_choice(turn_protocol)
    if normalized not in SUPPORTED_TURN_PROTOCOLS:
        raise ValueError(f"Unsupported turn protocol '{turn_protocol}'. Expected one of: {', '.join(sorted(SUPPORTED_TURN_PROTOCOLS))}.")
    return normalized


def resolve_lane_type(lane_type: str) -> str:
    normalized = normalize_choice(lane_type)
    if normalized not in SUPPORTED_LANE_TYPES:
        raise ValueError(f"Unsupported lane type '{lane_type}'. Expected one of: {', '.join(sorted(SUPPORTED_LANE_TYPES))}.")
    return normalized


def resolve_demand_model(demand_model: str) -> str:
    normalized = normalize_choice(demand_model).replace("_", "-")
    if normalized not in SUPPORTED_DEMAND_MODELS:
        raise ValueError(f"Unsupported demand model '{demand_model}'. Expected one of: {', '.join(sorted(SUPPORTED_DEMAND_MODELS))}.")
    return normalized


def compute_sampling_weights(df: pd.DataFrame, column: Optional[str], label: str) -> np.ndarray:
    if column is None:
        return np.full(len(df), 1.0 / max(len(df), 1))
    if column not in df.columns:
        raise ValueError(f"{label} weight column '{column}' was not found in the input data.")
    weights = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy()
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"{label} weight column '{column}' has no positive numeric values.")
    return weights / total


def generate_orders(census: pd.DataFrame, restaurants: pd.DataFrame, G: nx.Graph, cfg: SimConfig) -> pd.DataFrame:
    demand_model = resolve_demand_model(cfg.demand_model)
    if demand_model == "csv":
        raise ValueError("CSV-driven demand must be loaded via load_orders_csv().")
    return generate_realdata_orders(
        census=census,
        restaurants=restaurants,
        graph_or_topology=G,
        demand_model=demand_model,
        sim_duration_s=cfg.sim_duration_s,
        dt_s=cfg.dt_s,
        base_lambda_per_min=cfg.base_lambda_per_min,
        peak_multiplier=cfg.peak_multiplier,
        peak_window_s=cfg.peak_window_s,
        n_orders_override=cfg.n_orders_override,
        origin_weight_col=cfg.origin_weight_col,
        dest_weight_col=cfg.dest_weight_col,
        dest_jitter_m=cfg.dest_jitter_m,
        random_seed=cfg.random_seed,
    )


def generate_time_series_orders(census: pd.DataFrame, restaurants: pd.DataFrame, G: nx.Graph, cfg: SimConfig) -> pd.DataFrame:
    return generate_orders(census, restaurants, G, cfg)


def load_orders_csv(orders_path: str | Path, G: nx.Graph) -> pd.DataFrame:
    return load_orders_csv_shared(orders_path, G)


def edge_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def segment_direction(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) >= abs(dy):
        return "E" if dx >= 0 else "W"
    return "N" if dy >= 0 else "S"


def heading_altitude_ft(direction: str, cfg: SimConfig) -> float:
    return {"N": cfg.alt_north_ft, "S": cfg.alt_south_ft, "E": cfg.alt_east_ft, "W": cfg.alt_west_ft}[direction]


def reserve_time_window(resource_map: Dict, key, start: float, duration: float, headway: float) -> float:
    bookings = resource_map.setdefault(key, [])
    proposed = start
    while True:
        shifted = False
        for s, e in bookings:
            if proposed < e + headway and proposed + duration > s - headway:
                proposed = e + headway
                shifted = True
        if not shifted:
            break
    bookings.append((proposed, proposed + duration))
    bookings.sort(key=lambda z: z[0])
    return proposed


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def build_intersection_lane_graph(base_G: nx.Graph, cfg: SimConfig) -> nx.Graph:
    H = nx.Graph()
    for n, attrs in base_G.nodes(data=True):
        H.add_node(n, **attrs)
    for center in base_G.nodes():
        nbrs = list(base_G.neighbors(center))
        if len(nbrs) < 2:
            continue
        cx = base_G.nodes[center]["x"]
        cy = base_G.nodes[center]["y"]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a = nbrs[i]
                b = nbrs[j]
                ax = base_G.nodes[a]["x"]
                ay = base_G.nodes[a]["y"]
                bx = base_G.nodes[b]["x"]
                by = base_G.nodes[b]["y"]
                va = np.array([ax - cx, ay - cy], dtype=float)
                vb = np.array([bx - cx, by - cy], dtype=float)
                turn_angle = angle_between_vectors_deg(va, vb)
                if not (cfg.intersection_lane_min_angle_deg <= turn_angle <= cfg.intersection_lane_max_angle_deg):
                    continue
                d_ac = float(base_G[a][center]["length_m"])
                d_cb = float(base_G[center][b]["length_m"])
                direct = float(np.hypot(ax - bx, ay - by))
                if direct <= 1e-6:
                    continue
                if direct > cfg.intersection_lane_max_length_multiplier * min(d_ac, d_cb):
                    continue
                corridor_ft = max(float(base_G[a][center].get("corridor_height_ft", 40.0)), float(base_G[center][b].get("corridor_height_ft", 40.0)))
                if H.has_edge(a, b):
                    if direct < H[a][b].get("length_m", np.inf):
                        H[a][b].update({"length_m": direct, "corridor_height_ft": corridor_ft, "center_node": center, "lane_type": "intersection"})
                else:
                    H.add_edge(a, b, length_m=direct, corridor_height_ft=corridor_ft, center_node=center, lane_type="intersection")
    isolates = list(nx.isolates(H))
    if isolates:
        H.remove_nodes_from(isolates)
    return H


def prepare_graph_variant(nodes: pd.DataFrame, edges: pd.DataFrame, cfg: SimConfig) -> nx.Graph:
    lane_type = resolve_lane_type(cfg.lane_type)
    base_G = build_graph(nodes, edges)
    if lane_type == "normal":
        return base_G
    return build_intersection_lane_graph(base_G, cfg)


def turn_duration_s(prev_dir: str, next_dir: str, cfg: SimConfig) -> float:
    if prev_dir == next_dir:
        return 0.0
    if cfg.turn_protocol == "normal":
        return max(cfg.alt_turn_ft - heading_altitude_ft(prev_dir, cfg), 0.0) / cfg.climb_rate_ft_s + cfg.turn_time_s + max(cfg.alt_turn_ft - heading_altitude_ft(next_dir, cfg), 0.0) / cfg.descend_rate_ft_s
    radius_ft = m_to_ft(cfg.sphereabout_radius_m)
    arc_len_ft = 0.5 * math.pi * radius_ft
    horiz_time = arc_len_ft / cfg.cruise_speed_ft_s
    delta_alt = abs(heading_altitude_ft(next_dir, cfg) - heading_altitude_ft(prev_dir, cfg))
    vert_rate = min(cfg.climb_rate_ft_s, cfg.descend_rate_ft_s)
    vert_time = delta_alt / max(vert_rate, 1e-9)
    return max(horiz_time, vert_time)


def simulate_orders(G: nx.Graph, orders: pd.DataFrame, cfg: SimConfig):
    edge_bookings: Dict[Tuple[Tuple[str, str], str], List[Tuple[float, float]]] = {}
    intersection_bookings: Dict[str, List[Tuple[float, float]]] = {}
    results = []
    used_edge_counts: Dict[Tuple[str, str], int] = {}
    demand_trace = []
    orders_sorted = orders.sort_values("request_time_s").reset_index(drop=True)
    for order in orders_sorted.itertuples(index=False):
        demand_trace.append((order.request_time_s, 1))
        try:
            path = nx.shortest_path(G, source=order.origin_node, target=order.dest_node, weight="length_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue
        path_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in path], dtype=float)
        euclid_m = float(np.hypot(path_xy[-1, 0] - path_xy[0, 0], path_xy[-1, 1] - path_xy[0, 1]))
        manhattan_like_m = 0.0
        directions = []
        seg_lengths_m = []
        seg_edge_heights_ft = []
        for a, b in zip(path[:-1], path[1:]):
            xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
            xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
            directions.append(segment_direction(xa, ya, xb, yb))
            seg_lengths_m.append(float(G[a][b]["length_m"]))
            manhattan_like_m += abs(xb - xa) + abs(yb - ya)
            seg_edge_heights_ft.append(float(G[a][b].get("corridor_height_ft", 40.0)))
        n_turns = sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))
        feasible = True
        for d, h_ft in zip(directions, seg_edge_heights_ft):
            cruise_alt = heading_altitude_ft(d, cfg)
            if h_ft + cfg.building_clearance_ft > cruise_alt or cruise_alt > cfg.max_altitude_ft:
                feasible = False
                break
        if cfg.turn_protocol == "normal" and cfg.alt_turn_ft > cfg.max_altitude_ft:
            feasible = False
        if not feasible:
            continue
        request_time = float(order.request_time_s)
        first_alt = heading_altitude_ft(directions[0], cfg)
        takeoff_time = first_alt / cfg.climb_rate_ft_s
        launch_time = request_time
        current_time = launch_time + takeoff_time
        launch_delay = 0.0
        seg_records = []
        prev_dir = None
        for a, b, d, seg_len_m in zip(path[:-1], path[1:], directions, seg_lengths_m):
            if prev_dir is not None and d != prev_dir:
                tdur = turn_duration_s(prev_dir, d, cfg)
                reserved_turn_start = reserve_time_window(intersection_bookings, a, current_time, max(tdur, 0.0), cfg.intersection_headway_s)
                launch_delay += reserved_turn_start - current_time
                current_time = reserved_turn_start + max(tdur, 0.0)
            seg_duration = m_to_ft(seg_len_m) / cfg.cruise_speed_ft_s
            ekey = edge_key(a, b)
            resource_key = (ekey, d)
            reserved_seg_start = reserve_time_window(edge_bookings, resource_key, current_time, seg_duration, cfg.edge_time_headway_s)
            launch_delay += reserved_seg_start - current_time
            seg_start = reserved_seg_start
            seg_end = seg_start + seg_duration
            current_time = seg_end
            prev_dir = d
            used_edge_counts[ekey] = used_edge_counts.get(ekey, 0) + 1
            seg_records.append({"u": a, "v": b, "direction": d, "alt_ft": heading_altitude_ft(d, cfg), "start_s": seg_start, "end_s": seg_end, "length_m": seg_len_m, "lane_type": cfg.lane_type})
        landing_time = heading_altitude_ft(directions[-1], cfg) / cfg.descend_rate_ft_s
        finish_time = current_time + landing_time
        results.append({
            "order_id": order.order_id,
            "request_time_s": request_time,
            "launch_time_s": launch_time,
            "finish_time_s": finish_time,
            "launch_delay_s": launch_delay,
            "total_time_s": finish_time - request_time,
            "origin_node": order.origin_node,
            "dest_node": order.dest_node,
            "path": path,
            "path_xy": path_xy,
            "seg_records": seg_records,
            "manhattan_like_m": manhattan_like_m,
            "euclidean_m": euclid_m,
            "detour_ratio": manhattan_like_m / max(euclid_m, 1e-9),
            "n_turns": n_turns,
        })
    conflict_count = 0
    for bookings in edge_bookings.values():
        bookings_sorted = sorted(bookings, key=lambda x: x[0])
        for i in range(len(bookings_sorted)):
            for j in range(i + 1, len(bookings_sorted)):
                s1, e1 = bookings_sorted[i]
                s2, e2 = bookings_sorted[j]
                overlap = min(e1, e2) - max(s1, s2)
                if overlap > 0:
                    gap_ft = abs(s2 - s1) * cfg.cruise_speed_ft_s
                    if gap_ft < cfg.collision_radius_ft:
                        conflict_count += 1
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"path", "path_xy", "seg_records"}} for r in results])
    return results, results_df, used_edge_counts, conflict_count


def _quadratic_bezier(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return ((1 - t)[:, None] ** 2) * P0 + 2 * ((1 - t)[:, None] * t[:, None]) * P1 + (t[:, None] ** 2) * P2


def _sphereabout_arc_points(p_prev, p_corner, p_next, z_start, z_end, n=40, radius_m=45.0):
    """
    Build a true quarter-circle arc for a 90-degree turn.

    p_prev   : point before intersection, shape (2,)
    p_corner : intersection point, shape (2,)
    p_next   : point after intersection, shape (2,)
    z_start  : altitude at start of turn
    z_end    : altitude at end of turn
    n        : number of samples along arc
    radius_m : visual turn radius in meters
    """
    import numpy as np

    p_prev = np.asarray(p_prev, dtype=float)
    p_corner = np.asarray(p_corner, dtype=float)
    p_next = np.asarray(p_next, dtype=float)

    # Unit incoming direction toward the corner
    v_in = p_corner - p_prev
    v_in = v_in / np.linalg.norm(v_in)

    # Unit outgoing direction away from the corner
    v_out = p_next - p_corner
    v_out = v_out / np.linalg.norm(v_out)

    # Clamp radius so we stay inside both segments
    max_r_in = 0.45 * np.linalg.norm(p_corner - p_prev)
    max_r_out = 0.45 * np.linalg.norm(p_next - p_corner)
    r = min(radius_m, max_r_in, max_r_out)

    # Tangency points on the two legs
    t1 = p_corner - r * v_in
    t2 = p_corner + r * v_out

    # Circle center for a perfect 90-degree arc:
    # move from t1 in the outgoing direction by r
    center = t1 + r * v_out

    # Angles from center to tangency points
    a1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
    a2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])

    # Determine clockwise vs counterclockwise so we take the shorter 90° arc
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]

    if cross > 0:
        # left turn / CCW
        if a2 < a1:
            a2 += 2 * np.pi
        theta = np.linspace(a1, a2, n)
    else:
        # right turn / CW
        if a2 > a1:
            a2 -= 2 * np.pi
        theta = np.linspace(a1, a2, n)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.linspace(z_start, z_end, n)

    return x, y, z, t1, t2


def build_protocol_waypoints(path_xy, segs, turn_type, cfg):
    """
    Build 3D waypoints for a route under the selected turn protocol.

    Parameters
    ----------
    path_xy : ndarray, shape (N, 2)
        Node coordinates along the route.
    segs : list of dict
        Segment records, each containing at least:
        - "direction"
        - "alt_ft"
    turn_type : str
        "normal" or "sphereabout"
    cfg : config object
        Used only if needed for defaults.

    Returns
    -------
    wx, wy, wz : lists
        3D waypoint coordinates for plotting.
    """
    import numpy as np

    def _sphereabout_arc_points(p_prev, p_corner, p_next, z_start, z_end, n=40, radius_m=45.0):
        """
        True quarter-circle arc for a 90-degree turn.
        Returns arc samples plus tangent points t1 and t2.
        """
        p_prev = np.asarray(p_prev, dtype=float)
        p_corner = np.asarray(p_corner, dtype=float)
        p_next = np.asarray(p_next, dtype=float)

        # Unit incoming direction toward the corner
        v_in = p_corner - p_prev
        v_in = v_in / np.linalg.norm(v_in)

        # Unit outgoing direction away from the corner
        v_out = p_next - p_corner
        v_out = v_out / np.linalg.norm(v_out)

        # Clamp radius so it fits on both adjacent segments
        max_r_in = 0.45 * np.linalg.norm(p_corner - p_prev)
        max_r_out = 0.45 * np.linalg.norm(p_next - p_corner)
        r = min(radius_m, max_r_in, max_r_out)

        # Tangency points
        t1 = p_corner - r * v_in
        t2 = p_corner + r * v_out

        # Circle center for a true 90-degree arc
        center = t1 + r * v_out

        a1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        a2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])

        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]

        if cross > 0:
            # CCW turn
            if a2 < a1:
                a2 += 2 * np.pi
            theta = np.linspace(a1, a2, n)
        else:
            # CW turn
            if a2 > a1:
                a2 -= 2 * np.pi
            theta = np.linspace(a1, a2, n)

        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = np.linspace(z_start, z_end, n)

        return x, y, z, t1, t2

    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) == 0 or len(segs) == 0:
        return [], [], []

    turn_type = str(turn_type).strip().lower()

    # Start on ground, then rise to first segment altitude
    wx = [float(path_xy[0, 0]), float(path_xy[0, 0])]
    wy = [float(path_xy[0, 1]), float(path_xy[0, 1])]
    wz = [0.0, float(segs[0]["alt_ft"])]

    prev_dir = segs[0]["direction"]
    prev_alt = float(segs[0]["alt_ft"])

    # Fly first segment endpoint at first altitude
    wx.append(float(path_xy[1, 0]))
    wy.append(float(path_xy[1, 1]))
    wz.append(prev_alt)

    for s_idx in range(1, len(segs)):
        cur_dir = segs[s_idx]["direction"]
        cur_alt = float(segs[s_idx]["alt_ft"])

        p_prev = np.array([path_xy[s_idx - 1, 0], path_xy[s_idx - 1, 1]], dtype=float)
        p_corner = np.array([path_xy[s_idx, 0], path_xy[s_idx, 1]], dtype=float)
        p_next = np.array([path_xy[s_idx + 1, 0], path_xy[s_idx + 1, 1]], dtype=float)

        turning = (cur_dir != prev_dir)

        if not turning:
            # Straight continuation
            wx.append(float(p_next[0]))
            wy.append(float(p_next[1]))
            wz.append(cur_alt)
            prev_dir = cur_dir
            prev_alt = cur_alt
            continue

        if turn_type == "sphereabout":
            arc_x, arc_y, arc_z, t1, t2 = _sphereabout_arc_points(
                p_prev,
                p_corner,
                p_next,
                prev_alt,
                cur_alt,
                n=40,
                radius_m=45.0,
            )

            # Replace hard-corner arrival with tangent approach if needed
            if len(wx) > 0 and np.isclose(wx[-1], p_corner[0]) and np.isclose(wy[-1], p_corner[1]):
                wx.pop()
                wy.pop()
                wz.pop()

            # Straight to first tangency point at old altitude
            wx.append(float(t1[0]))
            wy.append(float(t1[1]))
            wz.append(prev_alt)

            # Circular arc with gradual altitude change
            for k in range(len(arc_x)):
                wx.append(float(arc_x[k]))
                wy.append(float(arc_y[k]))
                wz.append(float(arc_z[k]))

            # Ensure continuation from second tangency point at new altitude
            wx.append(float(t2[0]))
            wy.append(float(t2[1]))
            wz.append(cur_alt)

            # Then continue to end of outgoing segment at new altitude
            wx.append(float(p_next[0]))
            wy.append(float(p_next[1]))
            wz.append(cur_alt)

        else:
            # "normal" turn:
            # sharp direction change at corner, then altitude change after direction change
            wx.append(float(p_corner[0]))
            wy.append(float(p_corner[1]))
            wz.append(prev_alt)

            wx.append(float(p_corner[0]))
            wy.append(float(p_corner[1]))
            wz.append(cur_alt)

            wx.append(float(p_next[0]))
            wy.append(float(p_next[1]))
            wz.append(cur_alt)

        prev_dir = cur_dir
        prev_alt = cur_alt

    # Land vertically at destination
    wx.append(float(path_xy[-1, 0]))
    wy.append(float(path_xy[-1, 1]))
    wz.append(0.0)

    return wx, wy, wz


def draw_3d_trajectories(ax, results: List[dict], cfg: SimConfig):
    ax.clear()
    ax.set_title(f"3D Trajectories — {cfg.turn_protocol.title()} turn")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Altitude (ft)")
    c3 = cm.get_cmap("tab10", cfg.max_demo_routes_3d)
    for i, r in enumerate(results[: cfg.max_demo_routes_3d]):
        wx, wy, wz = build_protocol_waypoints(r["path_xy"], r["seg_records"], cfg.turn_protocol, cfg)
        if len(wx) == 0:
            continue
        ax.plot(wx, wy, wz, color=c3(i), lw=1.6)
    ax.view_init(elev=24, azim=32)


def visualize_all(G: nx.Graph, orders: pd.DataFrame, results: List[dict], results_df: pd.DataFrame, used_edge_counts: Dict[Tuple[str, str], int], cfg: SimConfig, show: bool = True, save_prefix: Optional[str | Path] = None, return_figures: bool = False):
    if results_df.empty:
        raise RuntimeError("No feasible routes were simulated. Check altitude / clearance assumptions.")
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title(f"Top-Down Route Map — {cfg.lane_type.title()} lanes")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        ax1.plot([x1, x2], [y1, y2], color="0.88", lw=0.35, zorder=1)
    cmap = cm.get_cmap("viridis", min(len(results), max(cfg.max_demo_routes_map, 1)))
    for i, r in enumerate(results[: cfg.max_demo_routes_map]):
        xy = r["path_xy"]
        ax1.plot(xy[:, 0], xy[:, 1], color=cmap(i), lw=1.0, alpha=0.65, zorder=2)
    first = results[0]["path_xy"]
    ax1.scatter(first[0, 0], first[0, 1], c="g", s=30, label="Origin", zorder=3)
    ax1.scatter(first[-1, 0], first[-1, 1], c="r", s=30, marker="s", label="Destination", zorder=3)
    ax1.legend(loc="best")
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    draw_3d_trajectories(ax2, results, cfg)

    ax3 = fig.add_subplot(2, 3, 3)
    sc = ax3.scatter(results_df["euclidean_m"], results_df["manhattan_like_m"], c=results_df["n_turns"], s=28, cmap="hot", alpha=0.75)
    md = max(results_df["manhattan_like_m"].max(), results_df["euclidean_m"].max()) * 1.05
    ax3.plot([0, md], [0, md], "k--", lw=0.8)
    ax3.plot([0, md], [0, md * math.sqrt(2)], "r--", lw=0.8)
    ax3.set_title("Route Efficiency")
    ax3.set_xlabel("Euclidean distance (m)")
    ax3.set_ylabel("Grid-constrained path metric (m)")
    plt.colorbar(sc, ax=ax3, label="Turns")

    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.arange(0, cfg.sim_duration_s + cfg.dt_s, cfg.dt_s)
    counts, _ = np.histogram(orders["request_time_s"], bins=bins)
    ax4.plot(bins[:-1] / 60.0, counts, lw=1.8)
    ax4.set_title("Demand Over Time")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel(f"Orders per {cfg.dt_s}s")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(results_df["total_time_s"] / 60.0, bins=20, alpha=0.7, label="Total flight time (min)")
    ax5.hist(results_df["launch_delay_s"] / 60.0, bins=20, alpha=0.6, label="Launch delay (min)")
    ax5.set_title("Time Distributions")
    ax5.set_xlabel("Minutes")
    ax5.set_ylabel("Count")
    ax5.legend(loc="best")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Edge Utilization Heatmap")
    ax6.set_xlabel("x (m)")
    ax6.set_ylabel("y (m)")
    if used_edge_counts:
        vals = np.array(list(used_edge_counts.values()), dtype=float)
        vmin, vmax = vals.min(), vals.max()
        denom = max(vmax - vmin, 1e-9)
        for (u, v), c in used_edge_counts.items():
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            color = cm.plasma((c - vmin) / denom)
            ax6.plot([x1, x2], [y1, y2], color=color, lw=1.4 + 2.5 * (c - vmin) / denom)
    ax6.set_aspect("equal", adjustable="box")
    plt.suptitle(f"SF Drone Delivery Simulation — {cfg.turn_protocol.title()} turn, {cfg.lane_type.title()} lanes", fontsize=15, fontweight="bold")
    plt.tight_layout()

    fig2, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].hist(results_df["detour_ratio"], bins=20, color="#55aa77", alpha=0.85)
    axs[0].axvline(results_df["detour_ratio"].mean(), color="b", ls="--", lw=1.5)
    axs[0].set_title("Detour Ratio")
    axs[0].set_xlabel("Grid metric / Euclidean")
    axs[0].set_ylabel("Count")
    dir_counts = {"N": 0, "S": 0, "E": 0, "W": 0}
    for r in results:
        for seg in r["seg_records"]:
            dir_counts[seg["direction"]] += 1
    axs[1].bar([cfg.alt_north_ft, cfg.alt_south_ft, cfg.alt_east_ft, cfg.alt_west_ft], [dir_counts["N"], dir_counts["S"], dir_counts["E"], dir_counts["W"]], width=8)
    axs[1].set_title("Altitude Band Utilization")
    axs[1].set_xlabel("Altitude (ft)")
    axs[1].set_ylabel("Segments used")
    served_by_min = np.cumsum(np.histogram(results_df["finish_time_s"], bins=np.arange(0, cfg.sim_duration_s + 60, 60))[0])
    axs[2].plot(np.arange(len(served_by_min)), served_by_min, lw=2)
    axs[2].set_title("Cumulative Deliveries Completed")
    axs[2].set_xlabel("Time (min)")
    axs[2].set_ylabel("Completed orders")
    plt.tight_layout()

    saved_paths = {}
    if save_prefix is not None:
        prefix = resolve_output_prefix(save_prefix)
        overview_path = prefix.parent / f"{prefix.name}_overview.png"
        summary_path = prefix.parent / f"{prefix.name}_summary.png"
        fig.savefig(overview_path, dpi=150, bbox_inches="tight")
        fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
        saved_paths = {"overview_figure": str(overview_path), "summary_figure": str(summary_path)}
    if show:
        plt.show()
    if return_figures:
        return saved_paths, fig, fig2
    plt.close(fig)
    plt.close(fig2)
    return saved_paths


def build_summary(results_df: pd.DataFrame, conflict_count: int, orders: pd.DataFrame, G: nx.Graph, cfg: SimConfig, demand_source: str) -> Dict[str, float | int | str]:
    served = len(results_df)
    total = len(orders)
    if served == 0:
        avg_total_time_s = avg_launch_delay_s = avg_detour_ratio = avg_turns = max_total_time_s = 0.0
    else:
        avg_total_time_s = float(results_df["total_time_s"].mean())
        avg_launch_delay_s = float(results_df["launch_delay_s"].mean())
        avg_detour_ratio = float(results_df["detour_ratio"].mean())
        avg_turns = float(results_df["n_turns"].mean())
        max_total_time_s = float(results_df["total_time_s"].max())
    return {
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "orders_generated": int(total),
        "orders_served": int(served),
        "orders_served_pct": float(100 * served / max(total, 1)),
        "avg_total_time_s": avg_total_time_s,
        "avg_total_time_min": avg_total_time_s / 60.0,
        "avg_launch_delay_s": avg_launch_delay_s,
        "avg_detour_ratio": avg_detour_ratio,
        "avg_turns": avg_turns,
        "conflict_risk_count": int(conflict_count),
        "max_total_time_s": max_total_time_s,
        "turn_protocol": cfg.turn_protocol,
        "lane_type": cfg.lane_type,
        "demand_source": demand_source,
        "data_dir": cfg.data_dir,
        "random_seed": int(cfg.random_seed),
    }


def print_summary(summary: Dict[str, float | int | str]):
    print(f"Loaded graph: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges")
    print(f"Orders generated: {summary['orders_generated']}")
    print(f"Feasible / served: {summary['orders_served']} ({summary['orders_served_pct']:.1f}%)")
    print(f"Average total time: {summary['avg_total_time_s']:.1f} s ({summary['avg_total_time_min']:.2f} min)")
    print(f"Average launch delay: {summary['avg_launch_delay_s']:.1f} s")
    print(f"Average detour ratio: {summary['avg_detour_ratio']:.3f}")
    print(f"Average turns: {summary['avg_turns']:.2f}")
    print(f"Conflict-risk count: {summary['conflict_risk_count']}")
    print(f"Max total time: {summary['max_total_time_s']:.1f} s")
    print(f"Turn protocol: {summary['turn_protocol']}")
    print(f"Lane type: {summary['lane_type']}")


def resolve_output_prefix(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_orders_csv(orders: pd.DataFrame, output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(output, index=False)
    return str(output)


def save_run_outputs(results_df: pd.DataFrame, used_edge_counts: Dict[Tuple[str, str], int], cfg: SimConfig, summary: Dict[str, float | int | str], output_prefix: str | Path) -> Dict[str, str]:
    prefix = resolve_output_prefix(output_prefix)
    results_path = prefix.parent / f"{prefix.name}_results.csv"
    edge_usage_path = prefix.parent / f"{prefix.name}_edge_usage.csv"
    summary_path = prefix.parent / f"{prefix.name}_summary.json"
    config_path = prefix.parent / f"{prefix.name}_config.json"
    results_df.to_csv(results_path, index=False)
    pd.DataFrame([{"u": u, "v": v, "count": count} for (u, v), count in sorted(used_edge_counts.items())]).to_csv(edge_usage_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    return {"results_csv": str(results_path), "edge_usage_csv": str(edge_usage_path), "summary_json": str(summary_path), "config_json": str(config_path)}


def run_simulation(cfg: Optional[SimConfig] = None, orders: Optional[pd.DataFrame] = None, orders_csv: Optional[str | Path] = None, show: bool = True, save_figure: Optional[str | Path] = None, save_results: Optional[str | Path] = None, save_orders: Optional[str | Path] = None, return_figures: bool = False):
    cfg = cfg or SimConfig()
    cfg.turn_protocol = resolve_turn_protocol(cfg.turn_protocol)
    cfg.lane_type = resolve_lane_type(cfg.lane_type)
    cfg.demand_model = resolve_demand_model(cfg.demand_model)
    np.random.seed(cfg.random_seed)
    if orders is not None and orders_csv is not None:
        raise ValueError("Pass either orders or orders_csv, not both.")
    if cfg.demand_model == "csv" and orders_csv is None and orders is None:
        raise ValueError("Demand model 'csv' requires --orders-csv.")
    nodes, edges, census, restaurants = load_real_data(cfg)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges, {len(census)} census zones, {len(restaurants)} restaurants")
    G = prepare_graph_variant(nodes, edges, cfg)
    if orders_csv is not None:
        orders = load_orders_csv(orders_csv, G)
        demand_source = "csv"
    elif orders is None:
        orders = generate_orders(census, restaurants, G, cfg)
        demand_source = cfg.demand_model
    else:
        demand_source = "dataframe"
    saved_paths: Dict[str, str] = {}
    if save_orders is not None:
        saved_paths["orders_csv"] = save_orders_csv(orders, save_orders)
    results, results_df, used_edge_counts, conflict_count = simulate_orders(G, orders, cfg)
    if results_df.empty:
        raise RuntimeError("Simulation produced no feasible routes. Reduce altitude constraints or inspect edge corridor heights.")
    summary = build_summary(results_df, conflict_count, orders, G, cfg, demand_source)
    print_summary(summary)
    figs = ()
    if save_results is not None:
        saved_paths.update(save_run_outputs(results_df, used_edge_counts, cfg, summary, save_results))
    if show or save_figure is not None or return_figures:
        vis_out = visualize_all(G, orders, results, results_df, used_edge_counts, cfg, show=show, save_prefix=save_figure, return_figures=return_figures)
        if return_figures:
            fig_saved, fig1, fig2 = vis_out
            saved_paths.update(fig_saved)
            figs = (fig1, fig2)
        else:
            saved_paths.update(vis_out)
    for label, path in saved_paths.items():
        print(f"Saved {label}: {path}")
    return {"config": cfg, "graph": G, "orders": orders, "results": results, "results_df": results_df, "used_edge_counts": used_edge_counts, "conflict_count": conflict_count, "summary": summary, "saved_paths": saved_paths, "figures": figs}


class InteractiveSimulationApp:
    def __init__(self, base_cfg: Optional[SimConfig] = None):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter is not available in this Python environment.")
        self.base_cfg = base_cfg or SimConfig()
        self.root = tk.Tk()
        self.root.title("SF Drone Delivery Simulation Controls")
        self.root.geometry("1450x980")
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="Turn type:").pack(side=tk.LEFT, padx=(0, 6))
        self.turn_var = tk.StringVar(value=self.base_cfg.turn_protocol)
        self.turn_box = ttk.Combobox(controls, textvariable=self.turn_var, values=["normal", "sphereabout"], state="readonly", width=16)
        self.turn_box.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(controls, text="Lane type:").pack(side=tk.LEFT, padx=(0, 6))
        self.lane_var = tk.StringVar(value=self.base_cfg.lane_type)
        self.lane_box = ttk.Combobox(controls, textvariable=self.lane_var, values=["normal", "intersection"], state="readonly", width=16)
        self.lane_box.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(controls, text="Update visualization", command=self.refresh).pack(side=tk.LEFT, padx=(0, 10))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 0))
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas1 = None
        self.canvas2 = None
        self.turn_box.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.lane_box.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.refresh()

    def _clear_canvases(self):
        for canvas in [self.canvas1, self.canvas2]:
            if canvas is not None:
                canvas.get_tk_widget().destroy()
        self.canvas1 = None
        self.canvas2 = None

    def refresh(self):
        try:
            self.status_var.set("Running simulation...")
            self.root.update_idletasks()
            cfg = replace(self.base_cfg, turn_protocol=self.turn_var.get(), lane_type=self.lane_var.get())
            out = run_simulation(cfg, show=False, return_figures=True)
            fig1, fig2 = out["figures"]
            self._clear_canvases()
            self.canvas1 = FigureCanvasTkAgg(fig1, master=self.canvas_frame)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas2 = FigureCanvasTkAgg(fig2, master=self.canvas_frame)
            self.canvas2.draw()
            self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
            self.status_var.set(f"Showing {cfg.turn_protocol} turn + {cfg.lane_type} lanes | {out['summary']['orders_served']} served")
        except Exception as exc:
            self.status_var.set("Error")
            messagebox.showerror("Simulation error", str(exc))

    def run(self):
        self.root.mainloop()


def main():
    if TK_AVAILABLE:
        InteractiveSimulationApp(SimConfig()).run()
    else:
        run_simulation(SimConfig(), show=True)


if __name__ == "__main__":
    main()
