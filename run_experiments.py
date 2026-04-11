"""
run_experiments.py — Main experiment runner for Decentralized Drone Delivery
IEOR 290 Transportation Analytics, UC Berkeley, Spring 2026

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --phase abstract   # Abstract grid only
    python run_experiments.py --phase sf         # SF network only
    python run_experiments.py --download-data    # Download SF data only
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from config import ExperimentConfig, GridConfig, AltitudeConfig, SimConfig, SFConfig
from simulator import DroneDeliverySimulation, GridTopology
from visualize import (plot_capacity_sweep, plot_topology_comparison,
                       plot_protocol_comparison, plot_3d_trajectories,
                       plot_grid_heatmap, plot_altitude_utilization,
                       plot_flight_time_distribution)
from bluesky_export import export_scenario, export_conflict_markers


RESULTS_DIR = "results"
FIGURES_DIR = "figures"
SCENARIOS_DIR = "bluesky_scenarios"

for d in [RESULTS_DIR, FIGURES_DIR, SCENARIOS_DIR]:
    os.makedirs(d, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# PHASE 1: ABSTRACT GRID EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════

def run_abstract_grid_experiments():
    """
    Experiment 1: Capacity breakdown on abstract NxN grid
    - Compare grid vs diagonal overlay topologies
    - Compare all three turning protocols
    - Generate fundamental-diagram-analog plots
    """
    print("\n" + "=" * 70)
    print("PHASE 1: ABSTRACT GRID EXPERIMENTS")
    print("=" * 70)
    
    drone_counts = [10, 25, 50, 100, 200, 300, 400, 600, 800, 1000]
    
    # ── Experiment 1a: Grid topology — capacity sweep per protocol ──
    topo_results = {}
    
    for topology in ["grid", "diagonal_overlay"]:
        for protocol in ["turn_layer", "intersection_cube", "sphereabout"]:
            
            label = f"{topology}__{protocol}"
            print(f"\n--- {label} ---")
            
            cfg = ExperimentConfig(
                topology=topology,
                turning_protocol=protocol,
                grid=GridConfig(n_blocks=10, block_length=200),
                sim=SimConfig(drone_counts=drone_counts, seed=42)
            )
            
            sim = DroneDeliverySimulation(cfg)
            df = sim.capacity_sweep(verbose=True)
            df['topology'] = topology
            df['protocol'] = protocol
            
            topo_results[label] = df
            df.to_csv(f"{RESULTS_DIR}/capacity_{label}.csv", index=False)
    
    # ── Combine and save all results ──
    all_df = pd.concat(topo_results.values(), ignore_index=True)
    all_df.to_csv(f"{RESULTS_DIR}/all_capacity_results.csv", index=False)
    
    # ── Plot: Grid vs Diagonal (with turn_layer protocol) ──
    comparison = {
        'grid': topo_results.get('grid__turn_layer'),
        'diagonal_overlay': topo_results.get('diagonal_overlay__turn_layer'),
    }
    if all(v is not None for v in comparison.values()):
        fig = plot_topology_comparison(comparison)
        fig.savefig(f"{FIGURES_DIR}/topology_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ── Plot: Protocol comparison (on grid topology, 200 drones) ──
    protocol_rows = []
    for protocol in ["turn_layer", "intersection_cube", "sphereabout"]:
        key = f"grid__{protocol}"
        if key in topo_results:
            row200 = topo_results[key][topo_results[key]['n_drones'] == 200]
            if not row200.empty:
                protocol_rows.append({
                    'protocol': protocol,
                    'n_conflicts': row200.iloc[0]['n_conflicts'],
                    'avg_flight_time_s': row200.iloc[0]['avg_flight_time_s'],
                    'avg_turns': row200.iloc[0]['avg_turns'],
                })
    
    if protocol_rows:
        prot_df = pd.DataFrame(protocol_rows)
        fig = plot_protocol_comparison(prot_df)
        fig.savefig(f"{FIGURES_DIR}/protocol_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ── Plot: Capacity sweep for each topology ──
    for label, df in topo_results.items():
        fig = plot_capacity_sweep(df, f"({label})")
        fig.savefig(f"{FIGURES_DIR}/capacity_{label}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ── Generate 3D trajectory visualization (grid, 20 drones) ──
    cfg_viz = ExperimentConfig(
        topology="grid",
        turning_protocol="intersection_cube",
        grid=GridConfig(n_blocks=10, block_length=200),
        sim=SimConfig(seed=42)
    )
    sim_viz = DroneDeliverySimulation(cfg_viz)
    result = sim_viz.run_single(20)
    
    fig = plot_3d_trajectories(result['missions'], sim_viz.topology, 
                                n_show=20, alt_config=cfg_viz.altitude,
                                title="3D Drone Trajectories — Grid + Cube Intersections")
    fig.savefig(f"{FIGURES_DIR}/3d_trajectories_grid.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = plot_grid_heatmap(result['missions'], sim_viz.topology,
                            "Edge Utilization — 20 Drones on 10×10 Grid")
    fig.savefig(f"{FIGURES_DIR}/heatmap_grid.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = plot_altitude_utilization(result['missions'], cfg_viz.altitude)
    fig.savefig(f"{FIGURES_DIR}/altitude_usage.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = plot_flight_time_distribution(result['missions'])
    fig.savefig(f"{FIGURES_DIR}/flight_times.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ── Export BlueSky scenario ──
    export_scenario(result['missions'], sim_viz.topology,
                    filename=f"{SCENARIOS_DIR}/grid_20drones.scn",
                    n_drones=20)
    
    if result['conflicts']:
        export_conflict_markers(result['conflicts'], sim_viz.topology,
                                filename=f"{SCENARIOS_DIR}/grid_conflicts.scn")
    
    print(f"\n✓ Phase 1 complete. Results in {RESULTS_DIR}/, figures in {FIGURES_DIR}/")
    return topo_results


# ═════════════════════════════════════════════════════════════════════
# PHASE 2: SAN FRANCISCO NETWORK EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════

def run_sf_experiments():
    """
    Experiment 2: Capacity analysis on SF street network
    - Real street geometry from OSMnx
    - Building height constraints
    - Census-weighted demand (restaurant origins, residential destinations)
    - Zone-based admission control
    """
    print("\n" + "=" * 70)
    print("PHASE 2: SAN FRANCISCO NETWORK EXPERIMENTS")
    print("=" * 70)
    
    # Download data
    print("\n[1] Loading SF data...")
    from data_loader import (load_sf_street_network, load_sf_buildings,
                             load_sf_census_population, load_sf_restaurants,
                             generate_demand, snap_to_graph)
    
    sf_cfg = SFConfig()
    
    try:
        G = load_sf_street_network(sf_cfg)
        buildings = load_sf_buildings(sf_cfg)
        census = load_sf_census_population(sf_cfg)
        restaurants = load_sf_restaurants(sf_cfg)
    except Exception as e:
        print(f"\n⚠ Could not load SF data: {e}")
        print("  Skipping SF experiments. Run with --download-data first.")
        return None
    
    # Generate demand from real data
    print("\n[2] Generating demand from restaurant + census data...")
    n_orders = 200
    orders = generate_demand(n_orders, restaurants, census, 
                             pd.DataFrame(), np.random.default_rng(42))
    
    # Snap origins and destinations to graph nodes
    print("[3] Snapping OD pairs to street network...")
    od_pairs = []
    for _, row in orders.iterrows():
        try:
            orig_node = snap_to_graph(row['orig_lat'], row['orig_lon'], G)
            dest_node = snap_to_graph(row['dest_lat'], row['dest_lon'], G)
            if orig_node != dest_node:
                od_pairs.append((orig_node, dest_node))
        except Exception:
            continue
    
    print(f"  Generated {len(od_pairs)} valid OD pairs")
    
    # Run simulation on SF network
    print("\n[4] Running capacity sweep on SF network...")
    
    from simulator import SFTopology, MissionPlanner, ConflictDetector
    from simulator import TurnLayerProtocol, DroneMission
    
    from data_loader import compute_corridor_clearances
    try:
        clearances = compute_corridor_clearances(buildings, G)
    except Exception:
        clearances = {}
    
    sf_topo = SFTopology(G, clearances)
    alt_cfg = AltitudeConfig()
    
    # Adjust altitudes based on max building height
    max_bldg = buildings['height_m'].max() if 'height_m' in buildings.columns else 50
    if alt_cfg.north < max_bldg + alt_cfg.safety_buffer:
        offset = max_bldg + alt_cfg.safety_buffer - alt_cfg.north + 10
        print(f"  Adjusting altitude bands up by {offset:.0f}m "
              f"(tallest building: {max_bldg:.0f}m)")
        alt_cfg.north += offset
        alt_cfg.south += offset
        alt_cfg.east += offset
        alt_cfg.west += offset
        alt_cfg.transition += offset
    
    # Run with different drone counts
    from config import DroneConfig
    drone_cfg = DroneConfig()
    turn_protocol = TurnLayerProtocol(alt_cfg.transition, 4.0)
    planner = MissionPlanner(sf_topo, alt_cfg, drone_cfg, turn_protocol, n_bands=4)
    detector = ConflictDetector()
    
    drone_counts_sf = [10, 25, 50, 100, 150, 200]
    sf_results = []
    
    for nd in drone_counts_sf:
        rng = np.random.default_rng(42)
        pairs = [od_pairs[i % len(od_pairs)] for i in range(nd)]
        launch_times = np.sort(rng.exponential(300 / nd, nd).cumsum())
        
        missions = []
        for i, (orig, dest) in enumerate(pairs):
            m = planner.plan_mission(i, orig, dest, launch_times[i])
            missions.append(m)
        
        conflicts = detector.detect_all_conflicts(missions, sf_topo)
        valid = [m for m in missions if len(m.path) >= 2]
        
        if valid:
            result = {
                'n_drones': nd,
                'n_conflicts': len(conflicts),
                'conflicts_per_drone': len(conflicts) / nd,
                'avg_flight_time_s': np.mean([m.total_flight_time for m in valid]),
                'avg_distance_m': np.mean([m.total_distance for m in valid]),
                'avg_detour_ratio': np.mean([
                    m.total_distance / max(m.euclidean_distance, 1) for m in valid]),
            }
        else:
            result = {'n_drones': nd, 'n_conflicts': 0, 'conflicts_per_drone': 0,
                      'avg_flight_time_s': 0, 'avg_distance_m': 0, 'avg_detour_ratio': 0}
        
        sf_results.append(result)
        print(f"  {nd:4d} drones → {result['n_conflicts']:4d} conflicts | "
              f"avg time {result['avg_flight_time_s']:.1f}s")
    
    sf_df = pd.DataFrame(sf_results)
    sf_df['conflict_rate_per_1000m'] = sf_df['n_conflicts'] / \
        sf_df['avg_distance_m'].clip(lower=1) * 1000
    sf_df.to_csv(f"{RESULTS_DIR}/sf_capacity.csv", index=False)
    
    # Plot SF results
    fig = plot_capacity_sweep(sf_df, "(San Francisco Network)")
    fig.savefig(f"{FIGURES_DIR}/sf_capacity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Generate BlueSky scenario for SF
    if valid:
        export_scenario(missions[:30], sf_topo,
                        filename=f"{SCENARIOS_DIR}/sf_delivery.scn",
                        ref_lat=sf_cfg.ref_lat, ref_lon=sf_cfg.ref_lon,
                        n_drones=30)
    
    # ── Zone-based admission control experiment ──
    print("\n[5] Testing zone-based admission control...")
    from simulator import ZoneAdmissionController
    
    zone_ctrl = ZoneAdmissionController(sf_topo, n_zones_per_side=5, 
                                         max_drones_per_zone=30)
    
    # Simulate admission: count how many drones get queued
    queued_count = 0
    for m in missions:
        if len(m.path) < 2:
            continue
        p = sf_topo.get_position(m.path[0])
        zone = zone_ctrl.get_zone(p[0], p[1])
        if not zone_ctrl.enter_zone(zone, m.drone_id):
            queued_count += 1
    
    print(f"  With zone capacity=30: {queued_count}/{len(missions)} drones queued")
    
    from visualize import plot_zone_occupancy
    fig = plot_zone_occupancy(zone_ctrl, "Zone Occupancy — SF Network")
    fig.savefig(f"{FIGURES_DIR}/sf_zones.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ── Save demand data visualization ──
    try:
        import folium
        
        m_map = folium.Map(location=[sf_cfg.ref_lat, sf_cfg.ref_lon], zoom_start=14)
        
        # Restaurant origins (blue)
        for _, row in restaurants.head(100).iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']], radius=3,
                color='blue', fill=True, popup=row.get('name', '')
            ).add_to(m_map)
        
        # Census density (red heatmap points)
        for _, row in census.iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']], 
                radius=max(1, row['pop_density'] / 5000),
                color='red', fill=True, opacity=0.3
            ).add_to(m_map)
        
        m_map.save(f"{FIGURES_DIR}/sf_demand_map.html")
        print(f"  Demand map saved: {FIGURES_DIR}/sf_demand_map.html")
    except ImportError:
        print("  (folium not installed — skipping interactive map)")
    
    print(f"\n✓ Phase 2 complete.")
    return sf_df


# ═════════════════════════════════════════════════════════════════════
# PHASE 3: COMPARATIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════

def generate_summary_report(abstract_results, sf_results):
    """Generate a summary of all experimental findings."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    report = []
    report.append("# Decentralized Drone Delivery — Experiment Summary")
    report.append("# IEOR 290, UC Berkeley, Spring 2026")
    report.append("")
    
    if abstract_results:
        report.append("## Phase 1: Abstract Grid")
        for label, df in abstract_results.items():
            topo, proto = label.split('__')
            # Find breakdown point
            rates = df['conflict_rate_per_1000m'].values if 'conflict_rate_per_1000m' in df.columns else df['conflicts_per_drone'].values
            if len(rates) > 2:
                diffs = np.diff(rates)
                if max(diffs) > 0:
                    bd_idx = np.argmax(diffs) + 1
                    bd_count = df.iloc[bd_idx]['n_drones']
                    report.append(f"  {label}: breakdown ≈ {bd_count} drones")
        report.append("")
    
    if sf_results is not None:
        report.append("## Phase 2: San Francisco Network")
        report.append(f"  Max tested: {sf_results['n_drones'].max()} drones")
        report.append(f"  Conflicts at max: {sf_results.iloc[-1]['n_conflicts']}")
        report.append("")
    
    report.append("## Key Findings")
    report.append("  1. [Fill in: At what density does each topology break down?]")
    report.append("  2. [Fill in: How much do diagonals reduce conflicts?]")
    report.append("  3. [Fill in: Which turning protocol performs best?]")
    report.append("  4. [Fill in: How does zone admission control help?]")
    
    report_text = '\n'.join(report)
    
    with open(f"{RESULTS_DIR}/summary.md", 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  Saved to {RESULTS_DIR}/summary.md")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Decentralized Drone Delivery Simulation")
    parser.add_argument('--phase', choices=['abstract', 'sf', 'all'], default='all',
                       help='Which experiments to run')
    parser.add_argument('--download-data', action='store_true',
                       help='Download SF data only (no experiments)')
    args = parser.parse_args()
    
    if args.download_data:
        from data_loader import download_all_data
        download_all_data()
        return
    
    abstract_results = None
    sf_results = None
    
    if args.phase in ['abstract', 'all']:
        abstract_results = run_abstract_grid_experiments()
    
    if args.phase in ['sf', 'all']:
        sf_results = run_sf_experiments()
    
    generate_summary_report(abstract_results, sf_results)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  Results:  {RESULTS_DIR}/")
    print(f"  Figures:  {FIGURES_DIR}/")
    print(f"  BlueSky:  {SCENARIOS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
