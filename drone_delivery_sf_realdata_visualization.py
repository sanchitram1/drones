import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyproj import Transformer

# ================================================================
# San Francisco Drone Delivery Simulation (Python version)
# ---------------------------------------------------------------
# Reads exported CSVs from the data/ folder and simulates
# time-varying demand, shortest-path routing, altitude-separated
# traffic lanes, launch delays from corridor/intersection occupancy,
# and several visualizations.
# ================================================================


@dataclass
class SimConfig:
    data_dir: str = "data"
    random_seed: int = 42

    # Demand / simulation horizon
    sim_duration_s: int = 2 * 3600  # 2 hours
    dt_s: int = 30  # demand discretization step
    base_lambda_per_min: float = 6.0  # average orders/min systemwide
    peak_multiplier: float = 1.8
    peak_window_s: Tuple[int, int] = (45 * 60, 95 * 60)
    n_orders_override: Optional[int] = None  # if set, ignore poisson time series

    # Drone / routing parameters
    cruise_speed_ft_s: float = 35.0
    climb_rate_ft_s: float = 8.0
    descend_rate_ft_s: float = 7.0
    turn_time_s: float = 4.0
    max_altitude_ft: float = 400.0
    building_clearance_ft: float = 5.0
    collision_radius_ft: float = 10.0

    # Directional altitude scheme (must all remain <= max altitude)
    alt_north_ft: float = 300.0
    alt_south_ft: float = 315.0
    alt_east_ft: float = 330.0
    alt_west_ft: float = 345.0
    alt_turn_ft: float = 370.0

    # Capacity model
    edge_time_headway_s: float = 3.0  # min time gap on same directional corridor
    intersection_headway_s: float = 4.0  # min time gap for turn/intersection use
    max_demo_routes_3d: int = 8
    max_demo_routes_map: int = 120

    # Projection assumptions from your loader pipeline
    restaurants_census_crs: str = "EPSG:4326"
    graph_crs: str = "EPSG:32610"  # UTM Zone 10N, meters


def ft_to_m(x: np.ndarray | float) -> np.ndarray | float:
    return (
        np.asarray(x) * 0.3048
        if isinstance(x, (list, tuple, np.ndarray))
        else x * 0.3048
    )


def m_to_ft(x: np.ndarray | float) -> np.ndarray | float:
    return (
        np.asarray(x) / 0.3048
        if isinstance(x, (list, tuple, np.ndarray))
        else x / 0.3048
    )


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
        return m_to_ft(
            pd.to_numeric(edges["corridor_height_m"], errors="coerce")
            .fillna(12.0)
            .to_numpy()
        )
    return np.full(len(edges), 40.0)


def project_latlon_to_graph_xy(
    lat: np.ndarray, lon: np.ndarray, cfg: SimConfig
) -> Tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        cfg.restaurants_census_crs, cfg.graph_crs, always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def load_real_data(cfg: SimConfig):
    data_dir = Path(cfg.data_dir)
    nodes_path = data_dir / "sf_nodes.csv"
    edges_path = data_dir / "sf_edges.csv"
    census_path = data_dir / "sf_census.csv"
    restaurants_path = data_dir / "sf_restaurants.csv"

    missing = [
        str(p)
        for p in [nodes_path, edges_path, census_path, restaurants_path]
        if not p.exists()
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

    # Harmonize lengths and clean edges
    node_set = set(nodes["node_id"])
    edges = edges[edges["u"].isin(node_set) & edges["v"].isin(node_set)].copy()
    edges["length_m"] = pd.to_numeric(edges["length_m"], errors="coerce")
    edges = edges.dropna(subset=["length_m"])
    edges = edges[edges["u"] != edges["v"]]

    # projected xy for nodes already in meters; restaurants/census likely lat/lon
    if not {"lat", "lon"}.issubset(restaurants.columns):
        raise ValueError("sf_restaurants.csv must contain lat/lon columns")
    if not {"lat", "lon"}.issubset(census.columns):
        raise ValueError("sf_census.csv must contain lat/lon columns")

    restaurants["x"], restaurants["y"] = project_latlon_to_graph_xy(
        restaurants["lat"].to_numpy(), restaurants["lon"].to_numpy(), cfg
    )
    census["x"], census["y"] = project_latlon_to_graph_xy(
        census["lat"].to_numpy(), census["lon"].to_numpy(), cfg
    )

    return nodes, edges, census, restaurants


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in nodes.itertuples(index=False):
        G.add_node(row.node_id, x=float(row.x), y=float(row.y))

    corridor_ft = safe_corridor_height_ft(edges)
    for row, height_ft in zip(edges.itertuples(index=False), corridor_ft):
        if G.has_edge(row.u, row.v):
            # keep the shortest parallel edge
            if float(row.length_m) < G[row.u][row.v]["length_m"]:
                G[row.u][row.v].update(
                    length_m=float(row.length_m), corridor_height_ft=float(height_ft)
                )
        else:
            G.add_edge(
                row.u,
                row.v,
                length_m=float(row.length_m),
                corridor_height_ft=float(height_ft),
            )
    return G


def nearest_node_ids(
    points_xy: np.ndarray, nodes_xy: np.ndarray, node_ids: np.ndarray
) -> np.ndarray:
    # Chunked nearest-neighbor using vectorized distance; good enough for a few thousand points.
    out = []
    batch = 128
    for i in range(0, len(points_xy), batch):
        P = points_xy[i : i + batch]
        d2 = ((P[:, None, :] - nodes_xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        out.extend(node_ids[idx])
    return np.asarray(out)


def generate_time_series_orders(
    census: pd.DataFrame, restaurants: pd.DataFrame, G: nx.Graph, cfg: SimConfig
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    node_ids = np.array(list(G.nodes()))
    nodes_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids])

    if cfg.n_orders_override is not None:
        req_times = np.sort(rng.uniform(0, cfg.sim_duration_s, cfg.n_orders_override))
        n_orders = cfg.n_orders_override
    else:
        n_steps = cfg.sim_duration_s // cfg.dt_s
        req_times = []
        for t_idx in range(n_steps):
            t0 = t_idx * cfg.dt_s
            lam_per_min = cfg.base_lambda_per_min
            if cfg.peak_window_s[0] <= t0 < cfg.peak_window_s[1]:
                lam_per_min *= cfg.peak_multiplier
            # slight sinusoid to avoid a flat trace
            lam_per_min *= 1.0 + 0.25 * math.sin(2 * math.pi * t0 / cfg.sim_duration_s)
            arrivals = rng.poisson(max(lam_per_min, 0.05) * (cfg.dt_s / 60.0))
            if arrivals > 0:
                req_times.extend(t0 + rng.uniform(0, cfg.dt_s, arrivals))
        req_times = np.sort(np.asarray(req_times))
        n_orders = len(req_times)

    rest_idx = rng.integers(0, len(restaurants), size=n_orders)
    origin_xy = restaurants[["x", "y"]].to_numpy()[rest_idx]

    weights = (
        pd.to_numeric(
            census.get("pop_density", pd.Series(np.ones(len(census)))), errors="coerce"
        )
        .fillna(1.0)
        .to_numpy()
    )
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()
    dest_idx = rng.choice(len(census), size=n_orders, p=weights)
    dest_xy = census[["x", "y"]].to_numpy()[dest_idx].copy()
    # add spatial jitter (~100m) so deliveries are not all at tract centroids
    dest_xy += rng.normal(0.0, 100.0, size=dest_xy.shape)

    origin_nodes = nearest_node_ids(origin_xy, nodes_xy, node_ids)
    dest_nodes = nearest_node_ids(dest_xy, nodes_xy, node_ids)

    # avoid same-node trips
    same = origin_nodes == dest_nodes
    while np.any(same):
        resample = rng.choice(len(census), size=same.sum(), p=weights)
        dest_xy[same] = census[["x", "y"]].to_numpy()[resample] + rng.normal(
            0.0, 100.0, size=(same.sum(), 2)
        )
        dest_nodes[same] = nearest_node_ids(dest_xy[same], nodes_xy, node_ids)
        same = origin_nodes == dest_nodes

    orders = pd.DataFrame(
        {
            "order_id": np.arange(n_orders),
            "request_time_s": req_times,
            "origin_node": origin_nodes,
            "dest_node": dest_nodes,
            "orig_x": origin_xy[:, 0],
            "orig_y": origin_xy[:, 1],
            "dest_x": dest_xy[:, 0],
            "dest_y": dest_xy[:, 1],
        }
    )
    return orders


def edge_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def segment_direction(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) >= abs(dy):
        return "E" if dx >= 0 else "W"
    return "N" if dy >= 0 else "S"


def heading_altitude_ft(direction: str, cfg: SimConfig) -> float:
    return {
        "N": cfg.alt_north_ft,
        "S": cfg.alt_south_ft,
        "E": cfg.alt_east_ft,
        "W": cfg.alt_west_ft,
    }[direction]


def reserve_time_window(
    resource_map: Dict, key, start: float, duration: float, headway: float
) -> float:
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
            path = nx.shortest_path(
                G, source=order.origin_node, target=order.dest_node, weight="length_m"
            )
        except nx.NetworkXNoPath:
            continue

        # Precompute path geometry / routing stats
        path_xy = np.array(
            [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in path], dtype=float
        )
        euclid_m = float(
            np.hypot(path_xy[-1, 0] - path_xy[0, 0], path_xy[-1, 1] - path_xy[0, 1])
        )
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
        n_turns = sum(
            directions[i] != directions[i - 1] for i in range(1, len(directions))
        )

        # Check altitude feasibility from building clearances
        feasible = True
        for d, h_ft in zip(directions, seg_edge_heights_ft):
            cruise_alt = heading_altitude_ft(d, cfg)
            if h_ft + cfg.building_clearance_ft > cruise_alt:
                feasible = False
                break
            if cruise_alt > cfg.max_altitude_ft:
                feasible = False
                break
        if cfg.alt_turn_ft > cfg.max_altitude_ft:
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
        for idx, (a, b, d, seg_len_m) in enumerate(
            zip(path[:-1], path[1:], directions, seg_lengths_m)
        ):
            if prev_dir is not None and d != prev_dir:
                turn_dur = (
                    (cfg.alt_turn_ft - heading_altitude_ft(prev_dir, cfg))
                    / cfg.climb_rate_ft_s
                    + cfg.turn_time_s
                    + (cfg.alt_turn_ft - heading_altitude_ft(d, cfg))
                    / cfg.descend_rate_ft_s
                )
                key_inter = (
                    b if False else a
                )  # turn occurs at current intersection (start node of new segment)
                reserved_turn_start = reserve_time_window(
                    intersection_bookings,
                    key_inter,
                    current_time,
                    max(turn_dur, 0.0),
                    cfg.intersection_headway_s,
                )
                launch_delay += reserved_turn_start - current_time
                current_time = reserved_turn_start + max(turn_dur, 0.0)

            seg_duration = m_to_ft(seg_len_m) / cfg.cruise_speed_ft_s
            ekey = edge_key(a, b)
            resource_key = (ekey, d)
            reserved_seg_start = reserve_time_window(
                edge_bookings,
                resource_key,
                current_time,
                seg_duration,
                cfg.edge_time_headway_s,
            )
            launch_delay += reserved_seg_start - current_time
            seg_start = reserved_seg_start
            seg_end = seg_start + seg_duration
            current_time = seg_end
            prev_dir = d
            used_edge_counts[ekey] = used_edge_counts.get(ekey, 0) + 1
            seg_records.append(
                {
                    "u": a,
                    "v": b,
                    "direction": d,
                    "alt_ft": heading_altitude_ft(d, cfg),
                    "start_s": seg_start,
                    "end_s": seg_end,
                    "length_m": seg_len_m,
                }
            )

        landing_time = heading_altitude_ft(directions[-1], cfg) / cfg.descend_rate_ft_s
        finish_time = current_time + landing_time
        total_time = finish_time - request_time

        results.append(
            {
                "order_id": order.order_id,
                "request_time_s": request_time,
                "launch_time_s": launch_time,
                "finish_time_s": finish_time,
                "launch_delay_s": launch_delay,
                "total_time_s": total_time,
                "origin_node": order.origin_node,
                "dest_node": order.dest_node,
                "path": path,
                "path_xy": path_xy,
                "seg_records": seg_records,
                "manhattan_like_m": manhattan_like_m,
                "euclidean_m": euclid_m,
                "detour_ratio": manhattan_like_m / max(euclid_m, 1e-9),
                "n_turns": n_turns,
            }
        )

    # Pairwise near-conflict screening based on same edge/direction overlap and spatial threshold
    conflict_count = 0
    for bookings in edge_bookings.values():
        bookings_sorted = sorted(bookings, key=lambda x: x[0])
        for i in range(len(bookings_sorted)):
            for j in range(i + 1, len(bookings_sorted)):
                s1, e1 = bookings_sorted[i]
                s2, e2 = bookings_sorted[j]
                overlap = min(e1, e2) - max(s1, s2)
                if overlap > 0:
                    # on same corridor/direction, spatial gap may shrink below threshold during overlap
                    # approximate worst-case gap from time offset * speed
                    time_offset = abs(s2 - s1)
                    gap_ft = time_offset * cfg.cruise_speed_ft_s
                    if gap_ft < cfg.collision_radius_ft:
                        conflict_count += 1

    results_df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in {"path", "path_xy", "seg_records"}}
            for r in results
        ]
    )

    return results, results_df, used_edge_counts, conflict_count


def visualize_all(
    G: nx.Graph,
    orders: pd.DataFrame,
    results: List[dict],
    results_df: pd.DataFrame,
    used_edge_counts: Dict[Tuple[str, str], int],
    cfg: SimConfig,
):
    if results_df.empty:
        raise RuntimeError(
            "No feasible routes were simulated. Check altitude / clearance assumptions."
        )

    fig = plt.figure(figsize=(18, 10))

    # 1. top-down route map
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("Top-Down Route Map")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    # draw base street graph lightly
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        ax1.plot([x1, x2], [y1, y2], color="0.88", lw=0.35, zorder=1)

    cmap = cm.get_cmap("viridis", min(len(results), cfg.max_demo_routes_map))
    for i, r in enumerate(results[: cfg.max_demo_routes_map]):
        xy = r["path_xy"]
        ax1.plot(xy[:, 0], xy[:, 1], color=cmap(i), lw=1.0, alpha=0.65, zorder=2)
    first = results[0]["path_xy"]
    ax1.scatter(first[0, 0], first[0, 1], c="g", s=30, label="Origin", zorder=3)
    ax1.scatter(
        first[-1, 0],
        first[-1, 1],
        c="r",
        s=30,
        marker="s",
        label="Destination",
        zorder=3,
    )
    ax1.legend(loc="best")
    ax1.set_aspect("equal", adjustable="box")

    # 2. 3D trajectories
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.set_title("3D Trajectories")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_zlabel("Altitude (ft)")
    c3 = cm.get_cmap("tab10", cfg.max_demo_routes_3d)
    for i, r in enumerate(results[: cfg.max_demo_routes_3d]):
        xy = r["path_xy"]
        segs = r["seg_records"]
        wx = [xy[0, 0], xy[0, 0]]
        wy = [xy[0, 1], xy[0, 1]]
        wz = [0.0, segs[0]["alt_ft"]]
        prev_dir = segs[0]["direction"]
        for s_idx, seg in enumerate(segs):
            ax_u = xy[s_idx]
            ax_v = xy[s_idx + 1]
            cur_alt = seg["alt_ft"]
            if s_idx > 0 and seg["direction"] != prev_dir:
                wx.extend([ax_u[0], ax_u[0]])
                wy.extend([ax_u[1], ax_u[1]])
                wz.extend([cfg.alt_turn_ft, cur_alt])
            wx.append(ax_v[0])
            wy.append(ax_v[1])
            wz.append(cur_alt)
            prev_dir = seg["direction"]
        wx.append(xy[-1, 0])
        wy.append(xy[-1, 1])
        wz.append(0.0)
        ax2.plot(wx, wy, wz, color=c3(i), lw=1.4)
    ax2.view_init(elev=24, azim=32)

    # 3. Manhattan-like vs Euclidean scatter
    ax3 = fig.add_subplot(2, 3, 3)
    sc = ax3.scatter(
        results_df["euclidean_m"],
        results_df["manhattan_like_m"],
        c=results_df["n_turns"],
        s=28,
        cmap="hot",
        alpha=0.75,
    )
    md = (
        max(results_df["manhattan_like_m"].max(), results_df["euclidean_m"].max())
        * 1.05
    )
    ax3.plot([0, md], [0, md], "k--", lw=0.8)
    ax3.plot([0, md], [0, md * math.sqrt(2)], "r--", lw=0.8)
    ax3.set_title("Route Efficiency")
    ax3.set_xlabel("Euclidean distance (m)")
    ax3.set_ylabel("Grid-constrained path metric (m)")
    plt.colorbar(sc, ax=ax3, label="Turns")

    # 4. demand over time
    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.arange(0, cfg.sim_duration_s + cfg.dt_s, cfg.dt_s)
    counts, _ = np.histogram(orders["request_time_s"], bins=bins)
    tmins = bins[:-1] / 60.0
    ax4.plot(tmins, counts, lw=1.8)
    ax4.set_title("Demand Over Time")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel(f"Orders per {cfg.dt_s}s")

    # 5. flight time + delay distributions
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(
        results_df["total_time_s"] / 60.0,
        bins=20,
        alpha=0.7,
        label="Total flight time (min)",
    )
    ax5.hist(
        results_df["launch_delay_s"] / 60.0,
        bins=20,
        alpha=0.6,
        label="Launch delay (min)",
    )
    ax5.set_title("Time Distributions")
    ax5.set_xlabel("Minutes")
    ax5.set_ylabel("Count")
    ax5.legend(loc="best")

    # 6. edge utilization heatmap
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Edge Utilization Heatmap")
    ax6.set_xlabel("x (m)")
    ax6.set_ylabel("y (m)")
    if used_edge_counts:
        vals = np.array(list(used_edge_counts.values()), dtype=float)
        vmin, vmax = vals.min(), vals.max()
        for (u, v), c in used_edge_counts.items():
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            color = cm.plasma((c - vmin) / (max(vmax - vmin, 1e-9)))
            ax6.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                lw=1.4 + 2.5 * (c - vmin) / (max(vmax - vmin, 1e-9)),
            )
    ax6.set_aspect("equal", adjustable="box")

    plt.suptitle(
        "San Francisco Drone Delivery Simulation — Python Visualization",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    # Extra figure: capacity-style summaries
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
    axs[1].bar(
        [cfg.alt_north_ft, cfg.alt_south_ft, cfg.alt_east_ft, cfg.alt_west_ft],
        [dir_counts["N"], dir_counts["S"], dir_counts["E"], dir_counts["W"]],
        width=8,
    )
    axs[1].set_title("Altitude Band Utilization")
    axs[1].set_xlabel("Altitude (ft)")
    axs[1].set_ylabel("Segments used")

    served_by_min = np.cumsum(
        np.histogram(
            results_df["finish_time_s"], bins=np.arange(0, cfg.sim_duration_s + 60, 60)
        )[0]
    )
    axs[2].plot(np.arange(len(served_by_min)), served_by_min, lw=2)
    axs[2].set_title("Cumulative Deliveries Completed")
    axs[2].set_xlabel("Time (min)")
    axs[2].set_ylabel("Completed orders")

    plt.tight_layout()
    plt.show()


def print_summary(
    results_df: pd.DataFrame, conflict_count: int, orders: pd.DataFrame, G: nx.Graph
):
    served = len(results_df)
    total = len(orders)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Orders generated: {total}")
    print(f"Feasible / served: {served} ({100 * served / max(total, 1):.1f}%)")
    print(
        f"Average total time: {results_df['total_time_s'].mean():.1f} s ({results_df['total_time_s'].mean() / 60:.2f} min)"
    )
    print(f"Average launch delay: {results_df['launch_delay_s'].mean():.1f} s")
    print(f"Average detour ratio: {results_df['detour_ratio'].mean():.3f}")
    print(f"Average turns: {results_df['n_turns'].mean():.2f}")
    print(f"Conflict-risk count: {conflict_count}")
    print(f"Max total time: {results_df['total_time_s'].max():.1f} s")


def main():
    cfg = SimConfig()
    np.random.seed(cfg.random_seed)

    nodes, edges, census, restaurants = load_real_data(cfg)
    print(
        f"Loaded {len(nodes)} nodes, {len(edges)} edges, {len(census)} census zones, {len(restaurants)} restaurants"
    )

    G = build_graph(nodes, edges)
    orders = generate_time_series_orders(census, restaurants, G, cfg)
    results, results_df, used_edge_counts, conflict_count = simulate_orders(
        G, orders, cfg
    )
    if results_df.empty:
        raise RuntimeError(
            "Simulation produced no feasible routes. Reduce altitude constraints or inspect edge corridor heights."
        )
    print_summary(results_df, conflict_count, orders, G)
    visualize_all(G, orders, results, results_df, used_edge_counts, cfg)


if __name__ == "__main__":
    main()
