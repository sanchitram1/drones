"""
visualize.py — Plotting and visualization for drone delivery simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from collections import defaultdict
from typing import List, Dict, Optional
import pandas as pd


def plot_capacity_sweep(df: pd.DataFrame, title_suffix: str = ""):
    """Plot conflicts and flight time vs drone count (fundamental diagram analog)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Conflicts vs drone count
    ax = axes[0, 0]
    ax.plot(df['n_drones'], df['n_conflicts'], 'ro-', lw=2, markersize=6)
    ax.set_xlabel('Number of Drones')
    ax.set_ylabel('Total Conflicts')
    ax.set_title('Conflicts vs Drone Density')
    ax.grid(True, alpha=0.3)
    
    # 2. Conflict rate (per 1000m) vs drone count — analog of flow-density
    ax = axes[0, 1]
    ax.plot(df['n_drones'], df['conflict_rate_per_1000m'], 'bs-', lw=2, markersize=6)
    ax.set_xlabel('Number of Drones (Density)')
    ax.set_ylabel('Conflicts per 1000m')
    ax.set_title('Conflict Rate (Fundamental Diagram Analog)')
    ax.grid(True, alpha=0.3)
    
    # Annotate breakdown point (steepest increase)
    if len(df) > 2:
        rates = df['conflict_rate_per_1000m'].values
        diffs = np.diff(rates)
        if len(diffs) > 0:
            breakdown_idx = np.argmax(diffs) + 1
            ax.axvline(df.iloc[breakdown_idx]['n_drones'], color='r', ls='--', 
                      alpha=0.5, label=f"Breakdown ≈ {df.iloc[breakdown_idx]['n_drones']} drones")
            ax.legend()
    
    # 3. Avg flight time vs drone count
    ax = axes[1, 0]
    ax.plot(df['n_drones'], df['avg_flight_time_s'] / 60, 'g^-', lw=2, markersize=6)
    ax.set_xlabel('Number of Drones')
    ax.set_ylabel('Avg Flight Time (min)')
    ax.set_title('Average Flight Time vs Density')
    ax.grid(True, alpha=0.3)
    
    # 4. Conflicts per drone vs drone count
    ax = axes[1, 1]
    ax.plot(df['n_drones'], df['conflicts_per_drone'], 'mp-', lw=2, markersize=6)
    ax.set_xlabel('Number of Drones')
    ax.set_ylabel('Conflicts per Drone')
    ax.set_title('Per-Drone Conflict Risk')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Capacity Breakdown Analysis {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_topology_comparison(results: Dict[str, pd.DataFrame]):
    """Compare capacity curves for grid vs diagonal overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {'grid': '#e74c3c', 'diagonal_overlay': '#3498db'}
    labels = {'grid': 'Manhattan Grid', 'diagonal_overlay': 'Grid + Diagonals'}
    
    for topo_name, df in results.items():
        c = colors.get(topo_name, 'gray')
        l = labels.get(topo_name, topo_name)
        
        axes[0].plot(df['n_drones'], df['n_conflicts'], 'o-', 
                    color=c, lw=2, label=l)
        axes[1].plot(df['n_drones'], df['avg_flight_time_s'] / 60, 's-',
                    color=c, lw=2, label=l)
        axes[2].plot(df['n_drones'], df['avg_detour_ratio'], '^-',
                    color=c, lw=2, label=l)
    
    axes[0].set_xlabel('Drones'); axes[0].set_ylabel('Conflicts')
    axes[0].set_title('Conflicts'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Drones'); axes[1].set_ylabel('Avg Flight Time (min)')
    axes[1].set_title('Flight Time'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Drones'); axes[2].set_ylabel('Detour Ratio')
    axes[2].set_title('Detour Ratio (Manhattan/Euclidean)')
    axes[2].axhline(np.sqrt(2), color='gray', ls=':', label=f'Max grid penalty √2')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Grid vs Diagonal Overlay Topology', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_protocol_comparison(df: pd.DataFrame):
    """Bar chart comparing turning protocols."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    protocols = df['protocol'].values
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    axes[0].bar(protocols, df['n_conflicts'], color=colors)
    axes[0].set_ylabel('Conflicts'); axes[0].set_title('Total Conflicts')
    
    axes[1].bar(protocols, df['avg_flight_time_s'] / 60, color=colors)
    axes[1].set_ylabel('Avg Time (min)'); axes[1].set_title('Flight Time')
    
    axes[2].bar(protocols, df['avg_turns'], color=colors)
    axes[2].set_ylabel('Avg Turns'); axes[2].set_title('Turns per Route')
    
    for ax in axes:
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Turning Protocol Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_3d_trajectories(missions, topology, n_show: int = 10, 
                          alt_config=None, title: str = "3D Drone Trajectories"):
    """Plot 3D drone routes over the grid."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw grid edges on the ground
    for u, v in topology.G.edges():
        p1 = topology.get_position(u)
        p2 = topology.get_position(v)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0],
                color='lightgray', lw=0.5, alpha=0.5)
    
    # Draw altitude reference planes
    if alt_config and hasattr(topology, 'cfg'):
        extent = topology.cfg.grid_extent if hasattr(topology.cfg, 'grid_extent') else 2000
        grid_pts = np.linspace(0, extent, 3)
        Xp, Yp = np.meshgrid(grid_pts, grid_pts)
        for alt_val in [alt_config.north, alt_config.south, 
                        alt_config.east, alt_config.west]:
            ax.plot_surface(Xp, Yp, np.full_like(Xp, alt_val),
                           alpha=0.03, color='blue')
    
    # Draw drone trajectories
    cmap = plt.cm.tab10
    show_missions = missions[:n_show]
    
    for i, mission in enumerate(show_missions):
        if not mission.waypoints_3d:
            continue
        
        wps = np.array(mission.waypoints_3d)
        color = cmap(i % 10)
        
        ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], '-', 
                color=color, lw=1.5, alpha=0.8)
        ax.scatter(*wps[0], color='green', s=40, marker='o', zorder=5)
        ax.scatter(*wps[-1], color='red', s=40, marker='s', zorder=5)
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.view_init(elev=25, azim=-45)
    
    return fig


def plot_grid_heatmap(missions, topology, title="Edge Utilization Heatmap"):
    """Heatmap of how many drones use each edge."""
    edge_usage = defaultdict(int)
    
    for m in missions:
        for i in range(len(m.path) - 1):
            n1, n2 = min(m.path[i], m.path[i+1]), max(m.path[i], m.path[i+1])
            edge_usage[(n1, n2)] += 1
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    max_usage = max(edge_usage.values()) if edge_usage else 1
    cmap = plt.cm.YlOrRd
    
    for (n1, n2), count in edge_usage.items():
        p1 = topology.get_position(n1)
        p2 = topology.get_position(n2)
        color = cmap(count / max_usage)
        lw = 1 + 4 * (count / max_usage)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=0.7)
    
    # Draw all edges faintly
    for u, v in topology.G.edges():
        if (min(u,v), max(u,v)) not in edge_usage:
            p1 = topology.get_position(u)
            p2 = topology.get_position(v)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color='lightgray', lw=0.5, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_usage))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Drones per edge')
    
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    
    return fig


def plot_altitude_utilization(missions, alt_config):
    """Bar chart of altitude band usage."""
    dir_counts = defaultdict(int)
    for m in missions:
        for d in m.segment_directions:
            dir_counts[d] += 1
    
    fig, ax = plt.subplots(figsize=(8, 5))
    dirs = sorted(dir_counts.keys())
    counts = [dir_counts[d] for d in dirs]
    colors = plt.cm.Set2(np.linspace(0, 1, len(dirs)))
    
    ax.bar(dirs, counts, color=colors)
    ax.set_xlabel('Direction (Altitude Band)')
    ax.set_ylabel('Total Segments')
    ax.set_title('Altitude Band Utilization', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_flight_time_distribution(missions):
    """Histogram of flight times."""
    times = [m.total_flight_time / 60 for m in missions if m.total_flight_time > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(times, bins=25, color='#3498db', alpha=0.8, edgecolor='white')
    axes[0].axvline(np.mean(times), color='red', ls='--', 
                    label=f'Mean: {np.mean(times):.1f} min')
    axes[0].set_xlabel('Flight Time (min)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Flight Time Distribution')
    axes[0].legend()
    
    detours = [m.total_distance / max(m.euclidean_distance, 1) 
               for m in missions if m.euclidean_distance > 0]
    axes[1].hist(detours, bins=25, color='#2ecc71', alpha=0.8, edgecolor='white')
    axes[1].axvline(np.sqrt(2), color='red', ls='--', 
                    label=f'Max grid: √2 ≈ {np.sqrt(2):.2f}')
    axes[1].axvline(np.mean(detours), color='blue', ls='--',
                    label=f'Mean: {np.mean(detours):.2f}')
    axes[1].set_xlabel('Detour Ratio')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Detour Ratio Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_zone_occupancy(admission_controller, title="Zone Occupancy"):
    """Heatmap of zone-level drone occupancy."""
    n = admission_controller.n_zones
    grid = np.zeros((n, n))
    
    for (r, c), count in admission_controller.occupancy.items():
        if 0 <= r < n and 0 <= c < n:
            grid[r, c] = count
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap='YlOrRd', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label='Active Drones')
    
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f'{int(grid[r,c])}', ha='center', va='center', fontsize=9)
    
    ax.set_xlabel('Zone Column'); ax.set_ylabel('Zone Row')
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    return fig


def generate_all_plots(results: Dict, missions: list, topology, 
                        alt_config, save_dir: str = "figures"):
    """Generate and save all figures for the report."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    figs = {}
    
    if 'capacity_sweep' in results:
        figs['capacity'] = plot_capacity_sweep(results['capacity_sweep'])
    
    if 'topology_comparison' in results:
        figs['topology'] = plot_topology_comparison(results['topology_comparison'])
    
    if 'protocol_comparison' in results:
        figs['protocols'] = plot_protocol_comparison(results['protocol_comparison'])
    
    if missions:
        figs['3d_routes'] = plot_3d_trajectories(missions, topology, 10, alt_config)
        figs['heatmap'] = plot_grid_heatmap(missions, topology)
        figs['alt_usage'] = plot_altitude_utilization(missions, alt_config)
        figs['times'] = plot_flight_time_distribution(missions)
    
    for name, fig in figs.items():
        fig.savefig(f"{save_dir}/{name}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved {save_dir}/{name}.png")
    
    return figs
