"""
simulator.py — Core simulation engine for Decentralized Drone Delivery
Handles: grid construction, drone routing, turning protocols,
         conflict detection, and capacity analysis.

IEOR 290 Transportation Analytics, UC Berkeley, Spring 2026
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from config import ExperimentConfig, GridConfig, AltitudeConfig, DroneConfig


# ═════════════════════════════════════════════════════════════════════
# 1. GRID TOPOLOGY
# ═════════════════════════════════════════════════════════════════════


class GridTopology:
    """
    Builds the airspace graph for drone routing.
    Supports: pure Manhattan grid, diagonal overlay.
    """

    def __init__(self, cfg: GridConfig, topology: str = "grid"):
        self.cfg = cfg
        self.topology = topology
        self.G = nx.Graph()
        self.node_positions = {}  # node_id -> (x, y)

        self._build_grid()
        if topology == "diagonal_overlay":
            self._add_diagonals()

    def _build_grid(self):
        """Build NxN Manhattan grid graph."""
        n = self.cfg.n_blocks + 1  # nodes per side
        bl = self.cfg.block_length

        for r in range(n):
            for c in range(n):
                nid = r * n + c
                self.G.add_node(nid)
                self.node_positions[nid] = (c * bl, r * bl)

                # East edge
                if c < n - 1:
                    self.G.add_edge(nid, nid + 1, length=bl, edge_type="grid")
                # North edge
                if r < n - 1:
                    self.G.add_edge(nid, nid + n, length=bl, edge_type="grid")

        self.n_nodes = n * n
        self.n_side = n

    def _add_diagonals(self):
        """
        Add 45° diagonal corridors that cut across blocks.
        Each block gets two diagonals (NE-SW and NW-SE).
        Diagonal length = block_length * sqrt(2).
        """
        n = self.n_side
        bl = self.cfg.block_length
        diag_len = bl * np.sqrt(2)

        for r in range(n - 1):
            for c in range(n - 1):
                bl_node = r * n + c  # bottom-left
                br_node = r * n + c + 1  # bottom-right
                tl_node = (r + 1) * n + c  # top-left
                tr_node = (r + 1) * n + c + 1  # top-right

                # NE diagonal: bottom-left to top-right
                self.G.add_edge(bl_node, tr_node, length=diag_len, edge_type="diagonal")
                # NW diagonal: bottom-right to top-left
                self.G.add_edge(br_node, tl_node, length=diag_len, edge_type="diagonal")

    def get_position(self, node_id: int) -> Tuple[float, float]:
        return self.node_positions[node_id]

    def compute_heading(self, from_node: int, to_node: int) -> float:
        """Compute compass heading from one node to another (degrees, 0=N)."""
        x1, y1 = self.get_position(from_node)
        x2, y2 = self.get_position(to_node)
        dx, dy = x2 - x1, y2 - y1
        heading = np.degrees(np.arctan2(dx, dy)) % 360
        return heading

    def shortest_path(self, origin: int, destination: int) -> List[int]:
        """Shortest path on the grid graph (unweighted = fewest hops)."""
        try:
            return nx.shortest_path(self.G, origin, destination, weight="length")
        except nx.NetworkXNoPath:
            return []


# ═════════════════════════════════════════════════════════════════════
# 2. SF NETWORK TOPOLOGY
# ═════════════════════════════════════════════════════════════════════


class SFTopology:
    """Wraps a real SF street network (from OSMnx) with the same interface."""

    def __init__(
        self, G_projected: nx.MultiDiGraph, building_clearances: Optional[dict] = None
    ):
        # Convert to undirected simple graph for routing
        self.G_full = G_projected
        self.G = nx.Graph()

        for u, v, data in G_projected.edges(data=True):
            length = data.get("length", 100)
            if self.G.has_edge(u, v):
                # Keep shorter edge
                if length < self.G[u][v]["length"]:
                    self.G[u][v]["length"] = length
            else:
                self.G.add_edge(u, v, length=length, edge_type="street")

        # Store node positions
        self.node_positions = {}
        for node, data in G_projected.nodes(data=True):
            self.node_positions[node] = (data["x"], data["y"])

        self.node_list = list(self.G.nodes())
        self.n_nodes = len(self.node_list)
        self.building_clearances = building_clearances or {}

    def get_position(self, node_id) -> Tuple[float, float]:
        return self.node_positions[node_id]

    def compute_heading(self, from_node, to_node) -> float:
        x1, y1 = self.get_position(from_node)
        x2, y2 = self.get_position(to_node)
        dx, dy = x2 - x1, y2 - y1
        heading = np.degrees(np.arctan2(dx, dy)) % 360
        return heading

    def shortest_path(self, origin, destination) -> list:
        try:
            return nx.shortest_path(self.G, origin, destination, weight="length")
        except nx.NetworkXNoPath:
            return []

    def get_min_altitude(self, from_node, to_node) -> float:
        """Minimum safe altitude for a corridor segment, based on building heights."""
        edge_key = (min(from_node, to_node), max(from_node, to_node))
        clearance = self.building_clearances.get(edge_key, 12.0)
        return clearance + 10.0  # safety buffer


# ═════════════════════════════════════════════════════════════════════
# 3. DRONE AGENT
# ═════════════════════════════════════════════════════════════════════


@dataclass
class DroneState:
    """State of a single drone at a given time."""

    drone_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0  # compass degrees
    speed: float = 0.0
    phase: str = "idle"  # idle, takeoff, cruise, turning, landing, done
    segment_idx: int = 0
    time_on_segment: float = 0.0


@dataclass
class DroneMission:
    """Complete mission plan for one drone."""

    drone_id: int
    origin: int
    destination: int
    path: List[int] = field(default_factory=list)
    launch_time: float = 0.0

    # Computed route info
    waypoints_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    segment_headings: List[float] = field(default_factory=list)
    segment_altitudes: List[float] = field(default_factory=list)
    segment_lengths: List[float] = field(default_factory=list)
    segment_directions: List[str] = field(default_factory=list)
    n_turns: int = 0
    total_distance: float = 0.0
    euclidean_distance: float = 0.0
    total_flight_time: float = 0.0


# ═════════════════════════════════════════════════════════════════════
# 4. TURNING PROTOCOLS
# ═════════════════════════════════════════════════════════════════════


class TurnProtocol:
    """Base class for intersection turning protocols."""

    def compute_turn_waypoints(
        self,
        intersection_pos: Tuple[float, float],
        entry_heading: float,
        exit_heading: float,
        entry_alt: float,
        exit_alt: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Returns list of 3D waypoints that the drone follows through
        the intersection volume when changing direction.
        """
        raise NotImplementedError

    def compute_turn_time(
        self,
        entry_heading: float,
        exit_heading: float,
        entry_alt: float,
        exit_alt: float,
        drone_cfg: DroneConfig,
    ) -> float:
        raise NotImplementedError


class TurnLayerProtocol(TurnProtocol):
    """
    TU Delft style: climb to transition altitude, execute turn,
    descend to new heading's altitude band.

    Simple and proven. Vertical separation prevents conflicts during turns.
    """

    def __init__(self, transition_alt: float = 135.0, turn_time: float = 4.0):
        self.transition_alt = transition_alt
        self.turn_time = turn_time

    def compute_turn_waypoints(
        self, intersection_pos, entry_heading, exit_heading, entry_alt, exit_alt
    ):
        ix, iy = intersection_pos
        return [
            (ix, iy, entry_alt),  # arrive at intersection
            (ix, iy, self.transition_alt),  # climb to transition
            (ix, iy, exit_alt),  # descend to new band
        ]

    def compute_turn_time(
        self, entry_heading, exit_heading, entry_alt, exit_alt, drone_cfg
    ):
        climb = (self.transition_alt - entry_alt) / drone_cfg.climb_rate
        turn = self.turn_time
        descend = (self.transition_alt - exit_alt) / drone_cfg.descend_rate
        return climb + turn + descend


class IntersectionCubeProtocol(TurnProtocol):
    """
    3D Intersection Cube: drones follow diagonal paths through a cube
    volume centered at the intersection. Each (entry_dir, exit_dir) pair
    maps to a unique diagonal path so no two paths cross.

    The cube has 4 entry faces (N, S, E, W) and the drone enters at its
    current altitude, traverses a diagonal through the cube interior, and
    exits at the new altitude on the correct face.

    Key insight: in 3D, 4 non-crossing paths can connect any pair of
    opposite or adjacent faces if they use different vertical levels.
    """

    def __init__(self, cube_side: float = 40.0, entry_speed: float = 10.0):
        self.cube_side = cube_side
        self.entry_speed = entry_speed
        self.half = cube_side / 2

    def compute_turn_waypoints(
        self, intersection_pos, entry_heading, exit_heading, entry_alt, exit_alt
    ):
        ix, iy = intersection_pos
        h = self.half

        # Entry point: offset from intersection center by half cube side
        # in the direction the drone is coming FROM
        entry_angle_rad = np.radians((entry_heading + 180) % 360)  # reverse
        ex = ix + h * np.sin(entry_angle_rad)
        ey = iy + h * np.cos(entry_angle_rad)

        # Exit point: offset in the direction the drone is going TO
        exit_angle_rad = np.radians(exit_heading)
        ox = ix + h * np.sin(exit_angle_rad)
        oy = iy + h * np.cos(exit_angle_rad)

        # Midpoint is inside the cube at average altitude
        # (the diagonal path through the interior)
        mid_alt = (entry_alt + exit_alt) / 2

        return [
            (ex, ey, entry_alt),  # enter cube face
            (ix, iy, mid_alt),  # cube center (diagonal midpoint)
            (ox, oy, exit_alt),  # exit cube face
        ]

    def compute_turn_time(
        self, entry_heading, exit_heading, entry_alt, exit_alt, drone_cfg
    ):
        # Diagonal through cube: sqrt(side² + side² + dalt²)
        dalt = abs(exit_alt - entry_alt)
        diag_len = np.sqrt(2 * self.cube_side**2 + dalt**2)
        return diag_len / self.entry_speed


class SpheraboutProtocol(TurnProtocol):
    """
    Sphereabout: drones fly along great-circle arcs on the surface of
    a sphere centered at the intersection.

    Based on Moosavi & Farooq (2025). The sphere has a fixed radius.
    Drones enter at one point on the sphere surface, follow a CCW
    great-circle arc, and exit at the point corresponding to their
    new heading. All drones circulate in the same direction (CCW),
    eliminating head-on conflicts.
    """

    def __init__(self, radius: float = 25.0, arc_speed: float = 8.0):
        self.radius = radius
        self.arc_speed = arc_speed

    def _heading_to_sphere_point(
        self, heading: float, center: Tuple[float, float], altitude: float
    ) -> Tuple[float, float, float]:
        """Map a heading to a point on the sphere surface."""
        angle = np.radians(heading)
        x = center[0] + self.radius * np.sin(angle)
        y = center[1] + self.radius * np.cos(angle)
        z = altitude  # entry/exit at the heading's altitude
        return (x, y, z)

    def compute_turn_waypoints(
        self, intersection_pos, entry_heading, exit_heading, entry_alt, exit_alt
    ):
        # Entry and exit points on the sphere
        self._heading_to_sphere_point(
            (entry_heading + 180) % 360, intersection_pos, entry_alt
        )
        self._heading_to_sphere_point(exit_heading, intersection_pos, exit_alt)

        # Generate intermediate arc points (CCW on the sphere surface)
        n_arc_pts = 8
        angle_entry = np.radians((entry_heading + 180) % 360)
        angle_exit = np.radians(exit_heading)

        # Always go CCW (increasing angle)
        if angle_exit <= angle_entry:
            angle_exit += 2 * np.pi

        angles = np.linspace(angle_entry, angle_exit, n_arc_pts)
        altitudes = np.linspace(entry_alt, exit_alt, n_arc_pts)

        waypoints = []
        cx, cy = intersection_pos
        for a, z in zip(angles, altitudes):
            x = cx + self.radius * np.sin(a)
            y = cy + self.radius * np.cos(a)
            waypoints.append((x, y, z))

        return waypoints

    def compute_turn_time(
        self, entry_heading, exit_heading, entry_alt, exit_alt, drone_cfg
    ):
        # Arc length on the sphere
        angle_diff = (exit_heading - (entry_heading + 180)) % 360
        if angle_diff == 0:
            angle_diff = 360  # U-turn = full circle
        arc_length = self.radius * np.radians(angle_diff)
        return arc_length / self.arc_speed


# ═════════════════════════════════════════════════════════════════════
# 5. MISSION PLANNER
# ═════════════════════════════════════════════════════════════════════


class MissionPlanner:
    """Plans 3D routes for drones on the grid with altitude assignment."""

    def __init__(
        self,
        topology,
        alt_cfg: AltitudeConfig,
        drone_cfg: DroneConfig,
        turn_protocol: TurnProtocol,
        n_bands: int = 4,
    ):
        self.topo = topology
        self.alt = alt_cfg
        self.drone = drone_cfg
        self.turn = turn_protocol
        self.n_bands = n_bands  # 4 for grid, 8 for diagonal overlay

    def plan_mission(
        self, drone_id: int, origin: int, destination: int, launch_time: float
    ) -> DroneMission:
        """Plan a complete 3D mission from origin to destination."""
        mission = DroneMission(
            drone_id=drone_id,
            origin=origin,
            destination=destination,
            launch_time=launch_time,
        )

        # Route on the 2D graph
        path = self.topo.shortest_path(origin, destination)
        if len(path) < 2:
            mission.path = [origin]
            return mission

        mission.path = path

        # Compute per-segment info
        for i in range(len(path) - 1):
            heading = self.topo.compute_heading(path[i], path[i + 1])
            alt = self.alt.get_altitude(heading, self.n_bands)
            direction = self.alt.get_direction_label(heading, self.n_bands)

            p1 = self.topo.get_position(path[i])
            p2 = self.topo.get_position(path[i + 1])
            length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            mission.segment_headings.append(heading)
            mission.segment_altitudes.append(alt)
            mission.segment_directions.append(direction)
            mission.segment_lengths.append(length)

        # Count turns
        for i in range(1, len(mission.segment_directions)):
            if mission.segment_directions[i] != mission.segment_directions[i - 1]:
                mission.n_turns += 1

        # Distances
        mission.total_distance = sum(mission.segment_lengths)
        p_orig = self.topo.get_position(origin)
        p_dest = self.topo.get_position(destination)
        mission.euclidean_distance = np.sqrt(
            (p_dest[0] - p_orig[0]) ** 2 + (p_dest[1] - p_orig[1]) ** 2
        )

        # Build 3D waypoints and compute flight time
        self._build_3d_trajectory(mission)

        return mission

    def _build_3d_trajectory(self, mission: DroneMission):
        """Build the full 3D waypoint sequence and compute total flight time."""
        path = mission.path
        if len(path) < 2:
            return

        waypoints = []
        total_time = 0.0

        # Starting position (ground)
        p0 = self.topo.get_position(path[0])
        waypoints.append((p0[0], p0[1], 0.0))

        # Takeoff to first segment altitude
        first_alt = mission.segment_altitudes[0]
        waypoints.append((p0[0], p0[1], first_alt))
        total_time += first_alt / self.drone.climb_rate

        prev_dir = mission.segment_directions[0]

        for seg_idx in range(len(path) - 1):
            cur_dir = mission.segment_directions[seg_idx]
            cur_alt = mission.segment_altitudes[seg_idx]

            # Turn at intersection if direction changed
            if seg_idx > 0 and cur_dir != prev_dir:
                prev_alt = mission.segment_altitudes[seg_idx - 1]
                prev_heading = mission.segment_headings[seg_idx - 1]
                cur_heading = mission.segment_headings[seg_idx]

                intersection_pos = self.topo.get_position(path[seg_idx])

                turn_wps = self.turn.compute_turn_waypoints(
                    intersection_pos, prev_heading, cur_heading, prev_alt, cur_alt
                )
                waypoints.extend(turn_wps)

                turn_time = self.turn.compute_turn_time(
                    prev_heading, cur_heading, prev_alt, cur_alt, self.drone
                )
                total_time += turn_time

            # Cruise along segment
            p_end = self.topo.get_position(path[seg_idx + 1])
            waypoints.append((p_end[0], p_end[1], cur_alt))
            total_time += mission.segment_lengths[seg_idx] / self.drone.cruise_speed

            prev_dir = cur_dir

        # Landing
        p_final = self.topo.get_position(path[-1])
        final_alt = mission.segment_altitudes[-1]
        waypoints.append((p_final[0], p_final[1], 0.0))
        total_time += final_alt / self.drone.descend_rate

        mission.waypoints_3d = waypoints
        mission.total_flight_time = total_time


# ═════════════════════════════════════════════════════════════════════
# 6. CONFLICT DETECTION
# ═════════════════════════════════════════════════════════════════════


@dataclass
class Conflict:
    """A detected conflict between two drones."""

    drone_a: int
    drone_b: int
    time: float
    edge_key: str
    altitude: float
    h_separation: float
    v_separation: float


class ConflictDetector:
    """
    Detects conflicts by discretizing time and checking edge co-occupancy.
    A conflict = two drones on the same edge, at the same altitude band,
    with temporal overlap.
    """

    def __init__(self, h_threshold: float = 50.0, v_threshold: float = 8.0):
        self.h_threshold = h_threshold
        self.v_threshold = v_threshold

    def detect_all_conflicts(
        self, missions: List[DroneMission], topology
    ) -> List[Conflict]:
        """
        Build edge occupancy timetable and check for temporal overlaps
        between drones on the same edge at the same altitude.
        """
        # Build edge occupancy records
        edge_occupancy = defaultdict(list)  # edge_key -> list of occupancy records

        for mission in missions:
            if len(mission.path) < 2:
                continue

            cum_time = mission.launch_time
            path = mission.path

            # Takeoff
            first_alt = mission.segment_altitudes[0]
            cum_time += first_alt / 3.0  # climb_rate

            prev_dir = mission.segment_directions[0]

            for seg_idx in range(len(path) - 1):
                cur_dir = mission.segment_directions[seg_idx]
                cur_alt = mission.segment_altitudes[seg_idx]
                seg_len = mission.segment_lengths[seg_idx]

                # Turn penalty
                if seg_idx > 0 and cur_dir != prev_dir:
                    # Approximate turn time
                    cum_time += 8.0  # average turn time

                # Time on this edge
                travel_time = seg_len / 15.0  # cruise_speed
                enter_time = cum_time
                exit_time = cum_time + travel_time

                # Edge key (undirected)
                n1, n2 = (
                    min(path[seg_idx], path[seg_idx + 1]),
                    max(path[seg_idx], path[seg_idx + 1]),
                )
                edge_key = f"{n1}_{n2}"

                edge_occupancy[edge_key].append(
                    {
                        "drone": mission.drone_id,
                        "enter": enter_time,
                        "exit": exit_time,
                        "alt": cur_alt,
                        "dir": cur_dir,
                    }
                )

                cum_time = exit_time
                prev_dir = cur_dir

        # Check for conflicts: two drones on same edge, same altitude,
        # overlapping time windows
        conflicts = []

        for edge_key, records in edge_occupancy.items():
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    ri, rj = records[i], records[j]

                    # Same altitude band?
                    v_sep = abs(ri["alt"] - rj["alt"])
                    if v_sep > self.v_threshold:
                        continue

                    # Temporal overlap?
                    overlap = min(ri["exit"], rj["exit"]) - max(
                        ri["enter"], rj["enter"]
                    )
                    if overlap > 0:
                        conflicts.append(
                            Conflict(
                                drone_a=ri["drone"],
                                drone_b=rj["drone"],
                                time=max(ri["enter"], rj["enter"]),
                                edge_key=edge_key,
                                altitude=ri["alt"],
                                h_separation=0.0,  # approximation
                                v_separation=v_sep,
                            )
                        )

        return conflicts


# ═════════════════════════════════════════════════════════════════════
# 7. ZONE-BASED ADMISSION CONTROL
# ═════════════════════════════════════════════════════════════════════


class ZoneAdmissionController:
    """
    Divides the airspace into zones and limits the number of active
    drones per zone. Drones requesting entry to a full zone are queued.
    """

    def __init__(
        self, topology, n_zones_per_side: int = 5, max_drones_per_zone: int = 50
    ):
        self.n_zones = n_zones_per_side
        self.max_per_zone = max_drones_per_zone

        # Compute zone boundaries
        all_positions = [
            topology.get_position(n)
            for n in (
                topology.node_positions.keys()
                if hasattr(topology, "node_positions")
                else range(topology.n_nodes)
            )
        ]
        xs = [p[0] for p in all_positions]
        ys = [p[1] for p in all_positions]

        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        self.zone_width = (self.x_max - self.x_min) / n_zones_per_side
        self.zone_height = (self.y_max - self.y_min) / n_zones_per_side

        # Zone occupancy counters
        self.occupancy = defaultdict(int)
        self.queue = defaultdict(list)  # zone_id -> list of waiting drone_ids

    def get_zone(self, x: float, y: float) -> Tuple[int, int]:
        """Map (x, y) to zone (row, col)."""
        col = min(int((x - self.x_min) / self.zone_width), self.n_zones - 1)
        row = min(int((y - self.y_min) / self.zone_height), self.n_zones - 1)
        return (row, col)

    def can_enter(self, zone: Tuple[int, int]) -> bool:
        return self.occupancy[zone] < self.max_per_zone

    def enter_zone(self, zone: Tuple[int, int], drone_id: int) -> bool:
        if self.can_enter(zone):
            self.occupancy[zone] += 1
            return True
        else:
            self.queue[zone].append(drone_id)
            return False

    def leave_zone(self, zone: Tuple[int, int]):
        self.occupancy[zone] = max(0, self.occupancy[zone] - 1)
        # Release queued drone
        if self.queue[zone]:
            self.queue[zone].pop(0)
            self.occupancy[zone] += 1


# ═════════════════════════════════════════════════════════════════════
# 8. MAIN SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════


class DroneDeliverySimulation:
    """
    Main simulation engine.

    Workflow:
    1. Build topology (abstract grid or SF network)
    2. Generate OD demand
    3. Plan missions (route + 3D trajectory + altitude assignment)
    4. Detect conflicts
    5. Compute metrics
    6. Sweep drone density for capacity analysis
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.sim.seed)

        # Build topology
        if config.use_sf_data:
            self._setup_sf_topology()
        else:
            n_bands = 8 if config.topology == "diagonal_overlay" else 4
            self.topology = GridTopology(config.grid, config.topology)
            self.n_bands = n_bands

        # Select turning protocol
        if config.turning_protocol == "turn_layer":
            self.turn_protocol = TurnLayerProtocol(
                config.altitude.transition, config.turn_layer.turn_time
            )
        elif config.turning_protocol == "intersection_cube":
            self.turn_protocol = IntersectionCubeProtocol(
                config.cube.cube_side, config.cube.entry_speed
            )
        elif config.turning_protocol == "sphereabout":
            self.turn_protocol = SpheraboutProtocol(
                config.sphereabout.radius, config.sphereabout.arc_speed
            )
        else:
            raise ValueError(f"Unknown turning protocol: {config.turning_protocol}")

        # Mission planner
        self.planner = MissionPlanner(
            self.topology,
            config.altitude,
            config.drone,
            self.turn_protocol,
            self.n_bands,
        )

        # Conflict detector
        self.detector = ConflictDetector(
            config.sim.conflict_h_threshold, config.sim.conflict_v_threshold
        )

        # Zone admission control (optional)
        self.admission = None
        if config.sim.enable_admission_control:
            n_zones = config.sf.zone_grid_size if config.use_sf_data else 5
            self.admission = ZoneAdmissionController(
                self.topology, n_zones, config.sim.zone_capacity
            )

    def _setup_sf_topology(self):
        """Set up SF street network topology."""
        from data_loader import load_sf_street_network, load_sf_buildings
        from data_loader import compute_corridor_clearances

        G = load_sf_street_network(self.cfg.sf)
        buildings = load_sf_buildings(self.cfg.sf)
        clearances = compute_corridor_clearances(buildings, G)
        self.topology = SFTopology(G, clearances)
        self.n_bands = 8 if self.cfg.topology == "diagonal_overlay" else 4

    def run_single(
        self, n_drones: int, od_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Run a single simulation with n_drones.
        Returns dict with metrics.
        """
        # Generate OD pairs if not provided
        if od_pairs is None:
            nodes = list(self.topology.G.nodes())
            origins = self.rng.choice(nodes, n_drones)
            destinations = self.rng.choice(nodes, n_drones)
            # Ensure origin != destination
            for i in range(n_drones):
                while destinations[i] == origins[i]:
                    destinations[i] = self.rng.choice(nodes)
            od_pairs = list(zip(origins, destinations))

        # Generate launch times (Poisson arrivals)
        launch_times = np.sort(
            self.rng.exponential(
                self.cfg.sim.launch_window / n_drones, n_drones
            ).cumsum()
        )

        # Plan all missions
        missions = []
        for i, (orig, dest) in enumerate(od_pairs):
            mission = self.planner.plan_mission(i, orig, dest, launch_times[i])
            missions.append(mission)

        # Detect conflicts
        conflicts = self.detector.detect_all_conflicts(missions, self.topology)

        # Compute metrics
        valid_missions = [m for m in missions if len(m.path) >= 2]

        if not valid_missions:
            return self._empty_metrics(n_drones)

        flight_times = [m.total_flight_time for m in valid_missions]
        distances = [m.total_distance for m in valid_missions]
        euclidean = [m.euclidean_distance for m in valid_missions]
        turns = [m.n_turns for m in valid_missions]
        detour_ratios = [
            m.total_distance / max(m.euclidean_distance, 1) for m in valid_missions
        ]

        # Altitude band utilization
        alt_usage = defaultdict(int)
        for m in valid_missions:
            for d in m.segment_directions:
                alt_usage[d] += 1

        return {
            "n_drones": n_drones,
            "n_conflicts": len(conflicts),
            "conflicts_per_drone": len(conflicts) / n_drones,
            "conflict_rate": len(conflicts) / max(sum(distances), 1) * 1000,
            "avg_flight_time": np.mean(flight_times),
            "std_flight_time": np.std(flight_times),
            "max_flight_time": np.max(flight_times),
            "avg_distance": np.mean(distances),
            "avg_euclidean": np.mean(euclidean),
            "avg_detour_ratio": np.mean(detour_ratios),
            "avg_turns": np.mean(turns),
            "total_distance": sum(distances),
            "alt_band_usage": dict(alt_usage),
            "missions": valid_missions,
            "conflicts": conflicts,
        }

    def _empty_metrics(self, n_drones):
        return {
            "n_drones": n_drones,
            "n_conflicts": 0,
            "conflicts_per_drone": 0,
            "conflict_rate": 0,
            "avg_flight_time": 0,
            "std_flight_time": 0,
            "max_flight_time": 0,
            "avg_distance": 0,
            "avg_euclidean": 0,
            "avg_detour_ratio": 0,
            "avg_turns": 0,
            "total_distance": 0,
            "alt_band_usage": {},
            "missions": [],
            "conflicts": [],
        }

    def capacity_sweep(
        self, drone_counts: Optional[List[int]] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run simulations across increasing drone densities.
        This is the core capacity breakdown analysis.
        """
        if drone_counts is None:
            drone_counts = self.cfg.sim.drone_counts

        results = []

        for n in tqdm(drone_counts, desc="Capacity sweep", disable=not verbose):
            metrics = self.run_single(n)
            results.append(metrics)

            if verbose:
                print(
                    f"  {n:5d} drones → {metrics['n_conflicts']:5d} conflicts | "
                    f"avg time {metrics['avg_flight_time']:.1f}s | "
                    f"detour ratio {metrics['avg_detour_ratio']:.3f}"
                )

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "n_drones": r["n_drones"],
                    "n_conflicts": r["n_conflicts"],
                    "conflicts_per_drone": r["conflicts_per_drone"],
                    "conflict_rate_per_1000m": r["conflict_rate"],
                    "avg_flight_time_s": r["avg_flight_time"],
                    "avg_distance_m": r["avg_distance"],
                    "avg_detour_ratio": r["avg_detour_ratio"],
                    "avg_turns": r["avg_turns"],
                }
                for r in results
            ]
        )

        return df

    def compare_turning_protocols(self, n_drones: int = 200) -> pd.DataFrame:
        """
        Run the same scenario with all three turning protocols and compare.
        """
        # Fix OD pairs across protocols
        nodes = list(self.topology.G.nodes())
        origins = self.rng.choice(nodes, n_drones)
        destinations = self.rng.choice(nodes, n_drones)
        for i in range(n_drones):
            while destinations[i] == origins[i]:
                destinations[i] = self.rng.choice(nodes)
        od_pairs = list(zip(origins, destinations))

        protocols = {
            "turn_layer": TurnLayerProtocol(
                self.cfg.altitude.transition, self.cfg.turn_layer.turn_time
            ),
            "intersection_cube": IntersectionCubeProtocol(
                self.cfg.cube.cube_side, self.cfg.cube.entry_speed
            ),
            "sphereabout": SpheraboutProtocol(
                self.cfg.sphereabout.radius, self.cfg.sphereabout.arc_speed
            ),
        }

        results = []
        for name, protocol in protocols.items():
            self.planner.turn = protocol
            metrics = self.run_single(n_drones, od_pairs)
            metrics["protocol"] = name
            results.append(metrics)

        # Restore original protocol
        self.planner.turn = self.turn_protocol

        return pd.DataFrame(
            [
                {
                    "protocol": r["protocol"],
                    "n_conflicts": r["n_conflicts"],
                    "avg_flight_time_s": r["avg_flight_time"],
                    "avg_detour_ratio": r["avg_detour_ratio"],
                    "avg_turns": r["avg_turns"],
                }
                for r in results
            ]
        )
