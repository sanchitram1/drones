"""
config.py — Simulation parameters for Decentralized Drone Delivery
IEOR 290 Transportation Analytics, UC Berkeley, Spring 2026
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────
# GRID PARAMETERS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class GridConfig:
    n_blocks: int = 10  # NxN blocks
    block_length: float = 200.0  # meters per block edge

    @property
    def n_intersections(self) -> int:
        return (self.n_blocks + 1) ** 2

    @property
    def grid_extent(self) -> float:
        return self.n_blocks * self.block_length


# ─────────────────────────────────────────────────────────────────────
# ALTITUDE SCHEME
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AltitudeConfig:
    """
    4-band scheme for cardinal directions (grid topology).
    8-band scheme adds diagonals (diagonal overlay topology).
    Each direction gets its own altitude to prevent crossing conflicts.
    """

    # Cardinal direction bands (meters AGL)
    north: float = 50.0
    south: float = 60.0
    east: float = 70.0
    west: float = 80.0

    # Diagonal overlay bands (meters AGL)
    northeast: float = 90.0
    southwest: float = 100.0
    northwest: float = 110.0
    southeast: float = 120.0

    # Transition / turn layer (above all cruise bands)
    transition: float = 135.0

    # Ground
    ground: float = 0.0

    # Vertical separation between adjacent bands
    min_separation: float = 10.0

    # Safety buffer above tallest building
    safety_buffer: float = 10.0

    def get_altitude(self, heading_deg: float, n_bands: int = 4) -> float:
        """Map compass heading to altitude band."""
        h = heading_deg % 360
        if n_bands == 4:
            if 315 <= h or h < 45:
                return self.north
            elif 45 <= h < 135:
                return self.east
            elif 135 <= h < 225:
                return self.south
            else:
                return self.west
        else:  # 8-band
            if 337.5 <= h or h < 22.5:
                return self.north
            elif 22.5 <= h < 67.5:
                return self.northeast
            elif 67.5 <= h < 112.5:
                return self.east
            elif 112.5 <= h < 157.5:
                return self.southeast
            elif 157.5 <= h < 202.5:
                return self.south
            elif 202.5 <= h < 247.5:
                return self.southwest
            elif 247.5 <= h < 292.5:
                return self.west
            else:
                return self.northwest

    def get_direction_label(self, heading_deg: float, n_bands: int = 4) -> str:
        h = heading_deg % 360
        if n_bands == 4:
            if 315 <= h or h < 45:
                return "N"
            elif 45 <= h < 135:
                return "E"
            elif 135 <= h < 225:
                return "S"
            else:
                return "W"
        else:
            dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            idx = int((h + 22.5) / 45) % 8
            return dirs[idx]


# ─────────────────────────────────────────────────────────────────────
# DRONE PARAMETERS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DroneConfig:
    cruise_speed: float = 15.0  # m/s horizontal (~54 km/h)
    climb_rate: float = 3.0  # m/s vertical climb
    descend_rate: float = 2.0  # m/s vertical descend
    turn_rate: float = 45.0  # degrees/second yaw rate
    min_separation_h: float = 50.0  # horizontal separation (meters)
    min_separation_v: float = 10.0  # vertical separation (meters)
    sensor_range: float = 100.0  # detect-and-avoid range (meters)
    max_range: float = 15000.0  # max delivery range (meters)
    payload_kg: float = 2.5  # typical package weight


# ─────────────────────────────────────────────────────────────────────
# TURNING PROTOCOL CONFIGS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TurnLayerConfig:
    """TU Delft-style: climb to turn layer, change heading, descend to new band."""

    turn_time: float = 4.0  # seconds at transition altitude
    name: str = "turn_layer"


@dataclass
class IntersectionCubeConfig:
    """
    3D cube intersection: drones follow diagonal paths through a cube volume.
    The cube is centered at each intersection, with side length = cube_side.
    Each (entry_direction, exit_direction) pair maps to a unique diagonal
    path through the cube, so no two paths cross.
    """

    cube_side: float = 40.0  # meters (side length of intersection cube)
    entry_speed: float = 10.0  # m/s (reduced speed in intersection)
    name: str = "intersection_cube"


@dataclass
class SpheraboutConfig:
    """
    Spherical roundabout: drones follow great-circle arcs on the surface
    of a sphere centered at the intersection.
    Based on Moosavi & Farooq (2025) Sphereabout concept.
    """

    radius: float = 25.0  # meters (sphere radius)
    rotation_direction: str = "CCW"  # counterclockwise circulation
    arc_speed: float = 8.0  # m/s on the sphere surface
    name: str = "sphereabout"


# ─────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    dt: float = 1.0  # time step (seconds)
    launch_window: float = 300.0  # seconds — all drones launch within this
    max_sim_time: float = 3600.0  # max simulation duration (seconds)
    seed: int = 42  # random seed

    # Capacity sweep
    drone_counts: List[int] = field(
        default_factory=lambda: [10, 25, 50, 100, 200, 400, 600, 800, 1000]
    )

    # Conflict detection
    conflict_h_threshold: float = 50.0  # horizontal (meters)
    conflict_v_threshold: float = 8.0  # vertical (meters)

    # Zone-based admission control (SF mode)
    enable_admission_control: bool = False
    zone_capacity: int = 50  # max drones per zone


# ─────────────────────────────────────────────────────────────────────
# SAN FRANCISCO PARAMETERS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SFConfig:
    # Bounding box (downtown / SoMa / Mission core)
    lat_range: Tuple[float, float] = (37.755, 37.795)
    lon_range: Tuple[float, float] = (-122.420, -122.390)

    # Reference point (center)
    @property
    def ref_lat(self) -> float:
        return sum(self.lat_range) / 2

    @property
    def ref_lon(self) -> float:
        return sum(self.lon_range) / 2

    # Data paths
    osm_cache: str = "data/sf_network.graphml"
    buildings_cache: str = "data/sf_buildings.gpkg"
    census_cache: str = "data/sf_census.csv"
    restaurants_cache: str = "data/sf_restaurants.csv"

    # Zone grid for admission control
    zone_grid_size: int = 5  # 5x5 zones over the study area

    # FAA constraints
    max_altitude_agl: float = 120.0  # 400 ft AGL per Part 107
    sfo_class_b_floor: float = 60.0  # meters — approximate


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Top-level config that bundles everything."""

    grid: GridConfig = field(default_factory=GridConfig)
    altitude: AltitudeConfig = field(default_factory=AltitudeConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    turn_layer: TurnLayerConfig = field(default_factory=TurnLayerConfig)
    cube: IntersectionCubeConfig = field(default_factory=IntersectionCubeConfig)
    sphereabout: SpheraboutConfig = field(default_factory=SpheraboutConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    sf: SFConfig = field(default_factory=SFConfig)

    # Topology flags
    topology: str = "grid"  # "grid" or "diagonal_overlay"
    turning_protocol: str = (
        "turn_layer"  # "turn_layer", "intersection_cube", "sphereabout"
    )
    use_sf_data: bool = False  # False = abstract grid, True = SF network
