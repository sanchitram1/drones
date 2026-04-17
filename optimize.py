"""
optimize.py — Analytical Throughput Optimizer for Decentralized Drone Delivery

Computes the maximum achievable drone throughput (λ*) for each
(topology, turning_protocol) configuration given a demand distribution,
without running the full simulation.

Formulation
-----------
max  λ                                        (drones / second)

s.t. λ² · Q          ≤ K_max                 [C1 corridor conflicts, quadratic]
     λ · ρ̄_v · τ̄_p  ≤ 1   ∀ v with turns   [C2 intersection capacity, linear]
     λ · μ̄_{e,a}     ≤ v_c / h_sep  ∀ (e,a) [C3 minimum headway, linear]
     λ · T_z          ≤ cap_z        ∀ z      [C4 zone capacity, optional, linear]

Closed-form solution:
  λ* = min(
      √(K_max / Q),
      min_v  1 / (ρ̄_v · τ̄_p(v)),
      min_{e,a} (v_c / h_sep) / μ̄_{e,a},
      min_z  cap_z / T_z          [if admission control enabled]
  )

IEOR 290 Transportation Analytics, UC Berkeley, Spring 2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config import (
    ExperimentConfig, GridConfig, AltitudeConfig,
    DroneConfig, SimConfig,
    TurnLayerConfig, IntersectionCubeConfig, SpheraboutConfig,
)
from simulator import (
    GridTopology, SFTopology,
    TurnProtocol, TurnLayerProtocol, IntersectionCubeProtocol, SpheraboutProtocol,
    MissionPlanner,
)


# ══════════════════════════════════════════════════════════════════════
# 1.  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """
    Full result of the throughput optimization for one configuration.
    All λ values are in drones/second; multiply by T to get drones/window.
    """
    topology: str
    turning_protocol: str

    # ── closed-form bounds from each constraint ──────────────────────
    lambda_conflict:     float = 0.0   # C1: √(K_max / Q)
    lambda_intersection: float = 0.0   # C2: min_v 1/(ρ̄_v · τ̄_p(v))
    lambda_headway:      float = 0.0   # C3: min_{e,a} (v_c/h_sep) / μ̄_{e,a}
    lambda_zone:         float = np.inf  # C4: min_z cap_z/T_z  (inf if disabled)

    lambda_star: float = 0.0           # optimal = min of all bounds
    binding_constraint: str = ""       # which constraint is tight at λ*

    # ── diagnostic coefficients ──────────────────────────────────────
    Q: float = 0.0                     # conflict load coefficient
    K_max: float = 0.0
    bottleneck_edge: Optional[Tuple]   = None   # (edge, altitude) tightest on C3
    bottleneck_node: Optional[int]     = None   # node tightest on C2
    bottleneck_zone: Optional[Tuple]   = None   # zone tightest on C4

    # ── distribution info ────────────────────────────────────────────
    n_od_sampled: int = 0
    mean_path_length_m: float = 0.0
    mean_turns_per_path: float = 0.0
    n_active_edge_alt_lanes: int = 0    # lanes with μ̄ > 0
    n_turning_nodes: int = 0

    # ── per-band utilisation (share of total flow, dict band→fraction) ─
    altitude_band_share: Dict[str, float] = field(default_factory=dict)

    def throughput_per_hour(self) -> float:
        """Deliveries per hour at optimal throughput."""
        return self.lambda_star * 3600.0

    def summary(self) -> str:
        lines = [
            f"── {self.topology} / {self.turning_protocol} ──",
            f"  λ* = {self.lambda_star:.4f} drones/s  "
            f"({self.throughput_per_hour():.1f} deliveries/hr)",
            f"  Binding constraint : {self.binding_constraint}",
            f"  C1 bound (conflict): {self.lambda_conflict:.4f}  [Q = {self.Q:.6f}]",
            f"  C2 bound (intersect): {self.lambda_intersection:.4f}",
            f"  C3 bound (headway) : {self.lambda_headway:.4f}",
            f"  C4 bound (zone)    : {self.lambda_zone:.4f}",
            f"  Paths sampled      : {self.n_od_sampled}",
            f"  Mean path length   : {self.mean_path_length_m:.1f} m",
            f"  Mean turns/path    : {self.mean_turns_per_path:.2f}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 2.  DEMAND MODEL
# ══════════════════════════════════════════════════════════════════════

class DemandModel:
    """
    Builds an OD demand distribution d_{ij} and samples weighted OD pairs.

    Two modes:
      • 'gravity'  — d_{ij} ~ origin_i · dest_j · exp(-β · dist_{ij})
                     (uses population for destinations, restaurant density for origins)
      • 'uniform'  — d_{ij} = 1 for all i ≠ j (abstract grid baseline)
    """

    def __init__(
        self,
        topology,
        mode: str = "gravity",
        beta: float = 1e-4,          # distance decay (1/m units)
        rng: Optional[np.random.Generator] = None,
        origin_weights: Optional[Dict] = None,   # node→weight (restaurants)
        dest_weights:   Optional[Dict] = None,   # node→weight (population)
    ):
        self.topology = topology
        self.mode = mode
        self.beta = beta
        self.rng = rng or np.random.default_rng(42)
        self.nodes = list(topology.G.nodes())
        n = len(self.nodes)

        # Node-level weights
        if mode == "uniform" or origin_weights is None:
            self.origin_w = np.ones(n)
        else:
            self.origin_w = np.array([origin_weights.get(v, 1.0) for v in self.nodes])

        if mode == "uniform" or dest_weights is None:
            self.dest_w = np.ones(n)
        else:
            self.dest_w = np.array([dest_weights.get(v, 1.0) for v in self.nodes])

        # Normalise weights to avoid numerical overflow
        self.origin_w = self.origin_w / (self.origin_w.sum() + 1e-12)
        self.dest_w   = self.dest_w   / (self.dest_w.sum()   + 1e-12)

    # ── precompute pairwise gravity weights ──────────────────────────
    def _gravity_pair_weights(self, origins_idx, dests_idx) -> np.ndarray:
        """Return unnormalised gravity weight for each (origin, dest) index pair."""
        weights = np.zeros(len(origins_idx))
        for k, (i, j) in enumerate(zip(origins_idx, dests_idx)):
            if i == j:
                continue
            xi, yi = self.topology.get_position(self.nodes[i])
            xj, yj = self.topology.get_position(self.nodes[j])
            dist = np.hypot(xj - xi, yj - yi)
            weights[k] = self.origin_w[i] * self.dest_w[j] * np.exp(-self.beta * dist)
        return weights

    # ── main sampling interface ───────────────────────────────────────
    def sample(self, n_samples: int) -> List[Tuple[int, int]]:
        """
        Return n_samples (origin_node, dest_node) pairs drawn from d_{ij}.
        Uses importance-sampling: draw origin by origin_w, then dest by
        dest_w * exp(-β·dist) given that origin.

        For 'uniform' mode this degenerates to pure random sampling.
        """
        nodes = self.nodes
        n = len(nodes)
        pairs = []

        # Draw origins proportional to origin_w
        origins_idx = self.rng.choice(n, size=n_samples, p=self.origin_w)

        for i in origins_idx:
            if self.mode == "uniform":
                # uniform destination, reject self-loops
                j = self.rng.integers(0, n)
                while j == i:
                    j = self.rng.integers(0, n)
            else:
                # gravity destination: proportional to dest_w * exp(-β·dist)
                xi, yi = self.topology.get_position(nodes[i])
                d_weights = np.zeros(n)
                for j in range(n):
                    if j == i:
                        continue
                    xj, yj = self.topology.get_position(nodes[j])
                    dist = np.hypot(xj - xi, yj - yi)
                    d_weights[j] = self.dest_w[j] * np.exp(-self.beta * dist)
                total = d_weights.sum()
                if total < 1e-12:
                    j = self.rng.integers(0, n)
                    while j == i:
                        j = self.rng.integers(0, n)
                else:
                    d_weights /= total
                    j = int(self.rng.choice(n, p=d_weights))
            pairs.append((nodes[i], nodes[j]))

        return pairs


# ══════════════════════════════════════════════════════════════════════
# 3.  PATH DISTRIBUTION  — μ̄, ρ̄, turn info
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PathDistributionStats:
    """
    Aggregate statistics computed from a weighted sample of OD paths.

    edge_alt_load[edge_key][alt] = μ̄_{e,a}  (fraction of demand on that lane)
    turn_rate[node]              = ρ̄_v       (fraction of demand turning at v)
    turn_events[node]            = list of (entry_heading, exit_heading,
                                             entry_alt, exit_alt)
                                   — used to compute average τ̄_p(v)
    """
    edge_alt_load:  Dict[str, Dict[float, float]]     = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    turn_rate:      Dict[int, float]                  = field(default_factory=lambda: defaultdict(float))
    turn_events:    Dict[int, List[Tuple]]            = field(default_factory=lambda: defaultdict(list))
    path_lengths_m: List[float]                       = field(default_factory=list)
    turns_per_path: List[int]                         = field(default_factory=list)
    zone_transit:   Dict[Tuple[int,int], float]       = field(default_factory=lambda: defaultdict(float))  # zone→fraction


class PathDistributionBuilder:
    """
    Walks a sample of demand-weighted paths and accumulates μ̄, ρ̄, and
    turn-event tables needed for the optimization.
    """

    def __init__(
        self,
        topology,
        alt_cfg: AltitudeConfig,
        n_bands: int = 4,
        zone_controller=None,           # optional ZoneAdmissionController
    ):
        self.topo     = topology
        self.alt      = alt_cfg
        self.n_bands  = n_bands
        self.zones    = zone_controller

    def build(
        self,
        od_pairs: List[Tuple[int, int]],
        verbose: bool = False,
    ) -> PathDistributionStats:
        """
        Walk each OD pair's shortest path and accumulate statistics.
        Each path has equal weight (1 / len(od_pairs)).
        """
        stats   = PathDistributionStats()
        w       = 1.0 / len(od_pairs)   # uniform weight per sample
        topo    = self.topo

        for (orig, dest) in tqdm(od_pairs, desc="Building path distribution",
                                  disable=not verbose):
            path = topo.shortest_path(orig, dest)
            if len(path) < 2:
                continue

            headings   = []
            altitudes  = []
            directions = []
            lengths    = []
            n_turns    = 0
            total_len  = 0.0

            for s in range(len(path) - 1):
                u, v   = path[s], path[s + 1]
                h      = topo.compute_heading(u, v)
                alt    = self.alt.get_altitude(h, self.n_bands)
                direc  = self.alt.get_direction_label(h, self.n_bands)
                pu, pv = topo.get_position(u), topo.get_position(v)
                le     = np.hypot(pv[0] - pu[0], pv[1] - pu[1])

                headings.append(h)
                altitudes.append(alt)
                directions.append(direc)
                lengths.append(le)
                total_len += le

                # μ̄_{e,a}: accumulate edge-altitude load
                ekey = f"{min(u,v)}_{max(u,v)}"
                stats.edge_alt_load[ekey][alt] += w

                # Zone transit
                if self.zones is not None:
                    pos  = topo.get_position(u)
                    zone = self.zones.get_zone(*pos)
                    stats.zone_transit[zone] += w

            # ρ̄_v and turn events: detect direction changes
            for s in range(1, len(path) - 1):
                if directions[s] != directions[s - 1]:
                    n_turns += 1
                    node = path[s]
                    stats.turn_rate[node] += w
                    stats.turn_events[node].append((
                        headings[s - 1], headings[s],
                        altitudes[s - 1], altitudes[s],
                    ))

            stats.path_lengths_m.append(total_len)
            stats.turns_per_path.append(n_turns)

        return stats


# ══════════════════════════════════════════════════════════════════════
# 4.  CONSTRAINT EVALUATORS
# ══════════════════════════════════════════════════════════════════════

class ConstraintEvaluator:
    """
    Evaluates the four throughput bounds given a PathDistributionStats
    and a TurnProtocol instance.
    """

    def __init__(
        self,
        dist:         PathDistributionStats,
        topology,
        turn_protocol: TurnProtocol,
        drone_cfg:    DroneConfig,
        sim_cfg:      SimConfig,
        alt_cfg:      AltitudeConfig,
        n_bands:      int = 4,
        K_max:        float = 50.0,
        zone_caps:    Optional[Dict[Tuple[int,int], int]] = None,
    ):
        self.dist     = dist
        self.topo     = topology
        self.proto    = turn_protocol
        self.drone    = drone_cfg
        self.sim      = sim_cfg
        self.alt      = alt_cfg
        self.n_bands  = n_bands
        self.K_max    = K_max
        self.zone_caps = zone_caps or {}

    # ── C1: corridor conflict  λ² · Q ≤ K_max ───────────────────────
    def compute_Q(self) -> Tuple[float, str, float]:
        """
        Returns (Q, bottleneck_edge_key, bottleneck_mu_bar).

        Q = (T/2) · Σ_{e,a} μ̄_{e,a}² · (l_e / v_c)

        The Poisson-overlap formula for expected pairwise conflicts
        on a single lane: E[conflicts] ≈ T·(λ·μ̄)²·τ_e/2,
        where τ_e = l_e/v_c is edge traversal time.
        """
        T   = self.sim.launch_window
        v_c = self.drone.cruise_speed

        Q              = 0.0
        worst_key      = None
        worst_contrib  = 0.0

        for ekey, alt_loads in self.dist.edge_alt_load.items():
            # Recover edge length from topology
            try:
                u_str, v_str = ekey.split("_")
                u, v = int(u_str), int(v_str)
                pu   = self.topo.get_position(u)
                pv   = self.topo.get_position(v)
                le   = np.hypot(pv[0] - pu[0], pv[1] - pu[1])
            except (ValueError, KeyError):
                # For OSMnx node IDs that are large ints; split on '_' may give
                # more than 2 parts if node IDs contain underscores — handle gracefully
                parts = ekey.split("_")
                le    = 200.0   # fallback: default block length

            tau_e = le / v_c

            for alt, mu_bar in alt_loads.items():
                contrib = (T / 2.0) * (mu_bar ** 2) * tau_e
                Q += contrib
                if contrib > worst_contrib:
                    worst_contrib = contrib
                    worst_key     = (ekey, alt)

        return Q, worst_key, worst_contrib

    def lambda_c1(self, Q: float) -> float:
        """C1 bound: √(K_max / Q)."""
        if Q < 1e-12:
            return np.inf
        return np.sqrt(self.K_max / Q)

    # ── C2: intersection throughput  λ · ρ̄_v · τ̄_p(v) ≤ 1 ──────────
    def compute_intersection_bounds(self) -> Tuple[float, int, Dict[int, float]]:
        """
        Returns (min_bound, bottleneck_node, {node: bound}).

        For each turning node v:
          τ̄_p(v) = mean turn time over all observed (θ_in, θ_out, alt_in, alt_out)
          bound_v = 1 / (ρ̄_v · τ̄_p(v))
        """
        bounds   = {}
        min_b    = np.inf
        worst_v  = None

        for node, rho_bar in self.dist.turn_rate.items():
            if rho_bar < 1e-12:
                continue

            events  = self.dist.turn_events[node]
            if not events:
                continue

            # Average turn time over all observed direction-change events at v
            turn_times = [
                self.proto.compute_turn_time(
                    theta_in, theta_out, alt_in, alt_out, self.drone
                )
                for (theta_in, theta_out, alt_in, alt_out) in events
            ]
            tau_bar = float(np.mean(turn_times))

            if tau_bar < 1e-12:
                continue

            b = 1.0 / (rho_bar * tau_bar)
            bounds[node] = b

            if b < min_b:
                min_b   = b
                worst_v = node

        return (min_b if min_b < np.inf else np.inf), worst_v, bounds

    # ── C3: minimum headway  λ · μ̄_{e,a} ≤ v_c / h_sep ────────────
    def compute_headway_bounds(self) -> Tuple[float, Tuple, Dict]:
        """
        Returns (min_bound, bottleneck_(ekey, alt), {(ekey,alt): bound}).

        Bound per lane: (v_c / h_sep) / μ̄_{e,a}
        Physical capacity of a single-direction lane: at most
        v_c / h_sep = 15/50 = 0.3 drone arrivals per second.
        """
        cap_per_lane = self.drone.cruise_speed / self.drone.min_separation_h
        bounds       = {}
        min_b        = np.inf
        worst_lane   = None

        for ekey, alt_loads in self.dist.edge_alt_load.items():
            for alt, mu_bar in alt_loads.items():
                if mu_bar < 1e-12:
                    continue
                b = cap_per_lane / mu_bar
                bounds[(ekey, alt)] = b
                if b < min_b:
                    min_b      = b
                    worst_lane = (ekey, alt)

        return (min_b if min_b < np.inf else np.inf), worst_lane, bounds

    # ── C4: zone admission control  λ · T_z ≤ cap_z ─────────────────
    def compute_zone_bounds(self) -> Tuple[float, Tuple, Dict]:
        """
        Returns (min_bound, bottleneck_zone, {zone: bound}).
        Only active when zone_caps is provided.
        """
        if not self.zone_caps:
            return np.inf, None, {}

        bounds   = {}
        min_b    = np.inf
        worst_z  = None

        for zone, T_z in self.dist.zone_transit.items():
            cap = self.zone_caps.get(zone)
            if cap is None or T_z < 1e-12:
                continue
            # T_z is a fraction; cap is absolute drones per zone.
            # At arrival rate λ, expected drones in zone ~ λ · T_z · mean_dwell
            # Here we use T_z directly as the fractional demand proxy.
            b = float(cap) / T_z
            bounds[zone] = b
            if b < min_b:
                min_b   = b
                worst_z = zone

        return (min_b if min_b < np.inf else np.inf), worst_z, bounds

    # ── full optimisation solve ──────────────────────────────────────
    def solve(self) -> OptimizationResult:
        """Compute all bounds and return the OptimizationResult."""
        Q, worst_edge, _ = self.compute_Q()

        lc1             = self.lambda_c1(Q)
        lc2, worst_node, _ = self.compute_intersection_bounds()
        lc3, worst_lane, edge_bounds = self.compute_headway_bounds()
        lc4, worst_zone, _ = self.compute_zone_bounds()

        candidates = {
            "C1 corridor conflicts":     lc1,
            "C2 intersection capacity":  lc2,
            "C3 minimum headway":        lc3,
            "C4 zone admission control": lc4,
        }
        binding    = min(candidates, key=candidates.get)
        lambda_star = candidates[binding]

        # Altitude band utilisation
        band_totals: Dict[str, float] = defaultdict(float)
        grand_total = 0.0
        for ekey, alt_loads in self.dist.edge_alt_load.items():
            for alt, mu in alt_loads.items():
                label = self._alt_to_label(alt)
                band_totals[label] += mu
                grand_total        += mu
        if grand_total > 0:
            alt_share = {k: v / grand_total for k, v in band_totals.items()}
        else:
            alt_share = {}

        res = OptimizationResult(
            topology          = "—",   # filled in by ThroughputOptimizer
            turning_protocol  = "—",
            lambda_conflict   = lc1,
            lambda_intersection = lc2,
            lambda_headway    = lc3,
            lambda_zone       = lc4,
            lambda_star       = lambda_star,
            binding_constraint = binding,
            Q                 = Q,
            K_max             = self.K_max,
            bottleneck_edge   = worst_edge,
            bottleneck_node   = worst_node,
            bottleneck_zone   = worst_zone,
            n_od_sampled      = len(self.dist.path_lengths_m),
            mean_path_length_m = float(np.mean(self.dist.path_lengths_m)) if self.dist.path_lengths_m else 0.0,
            mean_turns_per_path = float(np.mean(self.dist.turns_per_path)) if self.dist.turns_per_path else 0.0,
            n_active_edge_alt_lanes = sum(
                len(al) for al in self.dist.edge_alt_load.values()
            ),
            n_turning_nodes   = len(self.dist.turn_rate),
            altitude_band_share = alt_share,
        )
        return res

    def _alt_to_label(self, alt: float) -> str:
        """Reverse-map altitude value to direction label."""
        mapping = {
            self.alt.north:     "N",
            self.alt.south:     "S",
            self.alt.east:      "E",
            self.alt.west:      "W",
            self.alt.northeast: "NE",
            self.alt.southwest: "SW",
            self.alt.northwest: "NW",
            self.alt.southeast: "SE",
        }
        return mapping.get(alt, f"{alt:.0f}m")


# ══════════════════════════════════════════════════════════════════════
# 5.  TOP-LEVEL OPTIMIZER
# ══════════════════════════════════════════════════════════════════════

class ThroughputOptimizer:
    """
    Main entry point.  Given an ExperimentConfig (or the components),
    build the demand model, compute the path distribution, evaluate
    all constraints, and return λ* for this configuration.

    Usage
    -----
    optimizer = ThroughputOptimizer(config)
    result    = optimizer.optimize(
                    n_od_samples = 2000,
                    K_max        = 50.0,
                    verbose      = True,
                )
    print(result.summary())
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.sim.seed)

        # Build topology (mirrors DroneDeliverySimulation._setup)
        if config.use_sf_data:
            self._setup_sf_topology()
        else:
            n_bands       = 8 if config.topology == "diagonal_overlay" else 4
            self.topology = GridTopology(config.grid, config.topology)
            self.n_bands  = n_bands

        # Build turning protocol
        self.turn_protocol = self._make_protocol(config.turning_protocol)

    # ── topology helpers ──────────────────────────────────────────────
    def _setup_sf_topology(self):
        from data_loader import (
            load_sf_street_network, load_sf_buildings, compute_corridor_clearances
        )
        G          = load_sf_street_network(self.cfg.sf)
        buildings  = load_sf_buildings(self.cfg.sf)
        clearances = compute_corridor_clearances(buildings, G)
        self.topology = SFTopology(G, clearances)
        self.n_bands  = 8 if self.cfg.topology == "diagonal_overlay" else 4

    def _make_protocol(self, name: str) -> TurnProtocol:
        if name == "turn_layer":
            return TurnLayerProtocol(
                self.cfg.altitude.transition,
                self.cfg.turn_layer.turn_time,
            )
        elif name == "intersection_cube":
            return IntersectionCubeProtocol(
                self.cfg.cube.cube_side,
                self.cfg.cube.entry_speed,
            )
        elif name == "sphereabout":
            return SpheraboutProtocol(
                self.cfg.sphereabout.radius,
                self.cfg.sphereabout.arc_speed,
            )
        raise ValueError(f"Unknown turning protocol: {name}")

    # ── demand helpers ────────────────────────────────────────────────
    def build_demand_model(
        self,
        mode: str = "gravity",
        origin_weights: Optional[Dict] = None,
        dest_weights:   Optional[Dict] = None,
        beta: float = 1e-4,
    ) -> DemandModel:
        return DemandModel(
            self.topology,
            mode           = mode,
            beta           = beta,
            rng            = self.rng,
            origin_weights = origin_weights,
            dest_weights   = dest_weights,
        )

    # ── zone caps (pass-through from sim config) ──────────────────────
    def _get_zone_caps(self, zone_controller) -> Dict:
        if zone_controller is None or not self.cfg.sim.enable_admission_control:
            return {}
        cap  = self.cfg.sim.zone_capacity
        n    = zone_controller.n_zones
        caps = {(r, c): cap for r in range(n) for c in range(n)}
        return caps

    # ── main optimise call ────────────────────────────────────────────
    def optimize(
        self,
        n_od_samples:   int   = 2000,
        K_max:          float = 50.0,
        demand_mode:    str   = "gravity",
        origin_weights: Optional[Dict] = None,
        dest_weights:   Optional[Dict] = None,
        beta:           float = 1e-4,
        verbose:        bool  = True,
    ) -> OptimizationResult:
        """
        Full optimisation pipeline for the current (topology, protocol) config.

        Parameters
        ----------
        n_od_samples  : number of OD pairs sampled for path distribution
        K_max         : maximum tolerated number of conflicts per launch window
        demand_mode   : 'gravity' or 'uniform'
        origin_weights: node-keyed dict of origin attractiveness (restaurants)
        dest_weights  : node-keyed dict of destination attractiveness (population)
        beta          : distance decay coefficient for gravity model (1/m)
        verbose       : print progress bars and result summary
        """
        if verbose:
            print(f"\nOptimising  topology={self.cfg.topology}  "
                  f"protocol={self.cfg.turning_protocol}")

        # 1. Sample OD pairs from demand model
        demand = self.build_demand_model(demand_mode, origin_weights, dest_weights, beta)
        od_pairs = demand.sample(n_od_samples)

        # 2. Build path distribution statistics
        builder = PathDistributionBuilder(
            self.topology,
            self.cfg.altitude,
            self.n_bands,
        )
        dist = builder.build(od_pairs, verbose=verbose)

        # 3. Evaluate constraints and solve
        evaluator = ConstraintEvaluator(
            dist          = dist,
            topology      = self.topology,
            turn_protocol = self.turn_protocol,
            drone_cfg     = self.cfg.drone,
            sim_cfg       = self.cfg.sim,
            alt_cfg       = self.cfg.altitude,
            n_bands       = self.n_bands,
            K_max         = K_max,
            zone_caps     = {},     # zone caps unused unless enabled via config
        )
        result = evaluator.solve()

        # Tag result with config info
        result.topology         = self.cfg.topology
        result.turning_protocol = self.cfg.turning_protocol

        if verbose:
            print(result.summary())

        return result


# ══════════════════════════════════════════════════════════════════════
# 6.  MULTI-CONFIG COMPARISON
# ══════════════════════════════════════════════════════════════════════

def compare_configs(
    topologies:  List[str]   = ("grid", "diagonal_overlay"),
    protocols:   List[str]   = ("turn_layer", "intersection_cube", "sphereabout"),
    n_od_samples: int        = 2000,
    K_max:        float      = 50.0,
    demand_mode:  str        = "gravity",
    origin_weights: Optional[Dict] = None,
    dest_weights:   Optional[Dict] = None,
    base_config:  Optional[ExperimentConfig] = None,
    verbose:      bool       = True,
) -> pd.DataFrame:
    """
    Run the optimizer for every (topology, protocol) pair and return a
    summary DataFrame sorted by λ* descending.

    Parameters
    ----------
    topologies    : list of topology strings to compare
    protocols     : list of turning_protocol strings to compare
    n_od_samples  : OD pairs sampled per configuration
    K_max         : conflict budget
    demand_mode   : 'gravity' or 'uniform'
    origin_weights: node-keyed origin attractiveness (restaurants / None)
    dest_weights  : node-keyed destination attractiveness (population / None)
    base_config   : ExperimentConfig template (defaults constructed if None)
    verbose       : print progress

    Returns
    -------
    pd.DataFrame with one row per (topology, protocol) configuration,
    columns: topology, protocol, lambda_star, throughput_hr, binding_constraint,
             lambda_conflict, lambda_intersection, lambda_headway,
             Q, mean_path_m, mean_turns, n_turning_nodes, n_lanes
    """
    if base_config is None:
        base_config = ExperimentConfig()

    rows = []
    for topo in topologies:
        for proto in protocols:
            # Clone config with this (topo, proto) combination
            cfg                  = ExperimentConfig(
                grid             = base_config.grid,
                altitude         = base_config.altitude,
                drone            = base_config.drone,
                turn_layer       = base_config.turn_layer,
                cube             = base_config.cube,
                sphereabout      = base_config.sphereabout,
                sim              = base_config.sim,
                sf               = base_config.sf,
                topology         = topo,
                turning_protocol = proto,
                use_sf_data      = base_config.use_sf_data,
            )

            optimizer = ThroughputOptimizer(cfg)
            result    = optimizer.optimize(
                n_od_samples   = n_od_samples,
                K_max          = K_max,
                demand_mode    = demand_mode,
                origin_weights = origin_weights,
                dest_weights   = dest_weights,
                verbose        = verbose,
            )

            rows.append({
                "topology":            result.topology,
                "protocol":            result.turning_protocol,
                "lambda_star":         round(result.lambda_star,    4),
                "throughput_hr":       round(result.throughput_per_hour(), 1),
                "binding_constraint":  result.binding_constraint,
                "lambda_conflict":     round(result.lambda_conflict,     4),
                "lambda_intersection": round(result.lambda_intersection, 4),
                "lambda_headway":      round(result.lambda_headway,      4),
                "Q":                   round(result.Q,                   6),
                "mean_path_m":         round(result.mean_path_length_m,  1),
                "mean_turns":          round(result.mean_turns_per_path, 2),
                "n_turning_nodes":     result.n_turning_nodes,
                "n_lanes":             result.n_active_edge_alt_lanes,
            })

    df = pd.DataFrame(rows).sort_values("lambda_star", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════
# 7.  SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    config:        ExperimentConfig,
    K_max_values:  List[float] = (10, 25, 50, 100, 200),
    n_od_samples:  int         = 1500,
    demand_mode:   str         = "gravity",
    verbose:       bool        = False,
) -> pd.DataFrame:
    """
    How does λ* change as the conflict budget K_max varies?
    Only C1 depends on K_max; the intersection / headway / zone bounds
    are invariant, so the binding constraint may shift.

    Returns a DataFrame with (K_max, lambda_star, binding_constraint, lambda_conflict).
    """
    optimizer = ThroughputOptimizer(config)
    demand    = optimizer.build_demand_model(demand_mode)
    od_pairs  = demand.sample(n_od_samples)

    builder = PathDistributionBuilder(
        optimizer.topology, config.altitude, optimizer.n_bands
    )
    dist = builder.build(od_pairs, verbose=False)

    # Compute protocol-independent stats once
    dummy_eval = ConstraintEvaluator(
        dist          = dist,
        topology      = optimizer.topology,
        turn_protocol = optimizer.turn_protocol,
        drone_cfg     = config.drone,
        sim_cfg       = config.sim,
        alt_cfg       = config.altitude,
        n_bands       = optimizer.n_bands,
        K_max         = 1.0,   # placeholder
    )
    Q, _, _           = dummy_eval.compute_Q()
    lc2, _, _         = dummy_eval.compute_intersection_bounds()
    lc3, _, _         = dummy_eval.compute_headway_bounds()
    lc4, _, _         = dummy_eval.compute_zone_bounds()

    rows = []
    for K in K_max_values:
        lc1       = np.sqrt(K / Q) if Q > 1e-12 else np.inf
        lstar     = min(lc1, lc2, lc3, lc4)
        binding   = {lc1: "C1 corridor", lc2: "C2 intersection",
                     lc3: "C3 headway",  lc4: "C4 zone"}[lstar]
        rows.append({
            "K_max":               K,
            "lambda_star":         round(lstar, 4),
            "throughput_hr":       round(lstar * 3600, 1),
            "binding_constraint":  binding,
            "lambda_conflict":     round(lc1, 4),
            "lambda_intersection": round(lc2, 4),
            "lambda_headway":      round(lc3, 4),
        })
        if verbose:
            print(f"  K_max={K:6.1f}  λ*={lstar:.4f}  binding={binding}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 8.  QUICK-RUN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analytical throughput optimizer for drone airspace."
    )
    parser.add_argument("--topology",  default="grid",
                        choices=["grid", "diagonal_overlay"])
    parser.add_argument("--protocol",  default="turn_layer",
                        choices=["turn_layer", "intersection_cube", "sphereabout"])
    parser.add_argument("--K-max",     type=float, default=50.0,
                        help="Maximum tolerated conflicts per launch window")
    parser.add_argument("--samples",   type=int,   default=2000,
                        help="Number of OD pairs sampled for path distribution")
    parser.add_argument("--demand",    default="gravity",
                        choices=["gravity", "uniform"],
                        help="Demand model: gravity (SF pop-weighted) or uniform")
    parser.add_argument("--compare",   action="store_true",
                        help="Compare all topology × protocol combinations")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Sweep K_max for the chosen configuration")
    args = parser.parse_args()

    base_cfg = ExperimentConfig(
        topology         = args.topology,
        turning_protocol = args.protocol,
    )

    if args.compare:
        print("\n=== Comparing all configurations ===")
        df = compare_configs(
            K_max        = args.K_max,
            n_od_samples = args.samples,
            demand_mode  = args.demand,
            base_config  = base_cfg,
        )
        print("\n" + df.to_string(index=False))

    elif args.sensitivity:
        print(f"\n=== Sensitivity analysis: K_max sweep "
              f"({args.topology} / {args.protocol}) ===")
        df = sensitivity_analysis(
            config       = base_cfg,
            n_od_samples = args.samples,
            demand_mode  = args.demand,
            verbose      = True,
        )
        print("\n" + df.to_string(index=False))

    else:
        optimizer = ThroughputOptimizer(base_cfg)
        result    = optimizer.optimize(
            n_od_samples = args.samples,
            K_max        = args.K_max,
            demand_mode  = args.demand,
            verbose      = True,
        )
        print("\nAltitude band utilisation:")
        for band, share in sorted(result.altitude_band_share.items()):
            print(f"  {band:3s}  {share*100:.1f}%")
