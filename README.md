# Decentralized Drone Delivery Airspace Simulation

**IEOR 290 — Transportation Analytics, UC Berkeley, Spring 2026** **Team:** Sanchit
Arvind, Constantin Ertel, JP Schuchter, Shreya Krishnan

---

## Project Overview

This project investigates **decentralized drone delivery airspace design** — how to
structure urban airspace so that delivery drones can operate safely without a
centralized air traffic control tower. Each drone follows simple, local rules:

1. **Corridor routing**: Fly along street-grid corridors only
2. **Altitude-by-heading**: Your altitude is determined by your compass direction
3. **Intersection protocols**: Use geometric turning procedures at intersections
4. **No communication**: Drones don't talk to each other

We answer the question: **At what drone density does each airspace topology break down,
and how do different intersection designs affect capacity?**

## Architecture

```
drone_project/
├── config.py              # All simulation parameters (dataclasses)
├── data_loader.py         # Download SF street network, buildings, census, restaurants
├── simulator.py           # Core engine: grids, routing, conflicts, capacity sweep
├── visualize.py           # Matplotlib plots (capacity curves, 3D routes, heatmaps)
├── bluesky_export.py      # Export to BlueSky ATM Simulator (.scn files)
├── run_experiments.py      # Main experiment runner (Phase 1: grid, Phase 2: SF)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run abstract grid experiments only (no data download needed)
python run_experiments.py --phase abstract

# 3. Download SF data
python run_experiments.py --download-data

# 4. Run everything (abstract grid + SF network)
python run_experiments.py --phase all
```

## Experiments

### Phase 1: Abstract Grid (no external data required)
- **Topology comparison**: Manhattan grid vs. grid + diagonal overlay
- **Turning protocol comparison**: TU Delft turn-layer vs. 3D intersection cube vs.
  sphereabout
- **Capacity sweep**: 10 → 1000 drones, measuring conflicts, flight time, detour ratio
- Outputs: capacity curves, 3D trajectory plots, heatmaps, BlueSky scenarios

### Phase 2: San Francisco Network (requires internet for data download)
- **Real street geometry** from OpenStreetMap via OSMnx
- **Building height constraints** from LIDAR data (140K buildings)
- **Census-weighted demand**: restaurant origins (OSM POIs), residential destinations
- **Zone-based admission control**: divide SF into zones, cap drones per zone
- Outputs: SF capacity curves, demand maps, zone occupancy heatmaps

## Topologies

| Feature | Manhattan Grid | Grid + Diagonals |
|---------|---------------|-----------------| | Altitude bands | 4 (N/S/E/W) | 8 (+
NE/NW/SE/SW) | | Worst-case detour | √2 ≈ 1.41x | ~1.10x | | Rule complexity | Low |
Moderate | | Intersection conflicts | Low | Higher (more crossing) |

## Turning Protocols

1. **Turn Layer** (TU Delft): Climb to transition altitude → turn → descend to new band
2. **Intersection Cube**: Follow diagonal paths through a 3D cube volume at
   intersections
3. **Sphereabout**: Follow great-circle arcs on a sphere (Moosavi & Farooq, 2025)

## BlueSky Integration

The simulation exports `.scn` scenario files for BlueSky ATM Simulator:

```bash
pip install bluesky-simulator
python -m bluesky       # launches BlueSky GUI
# In BlueSky console:
PCALL bluesky_scenarios/grid_20drones.scn
```

## Data Sources

- **Street network**: OpenStreetMap via [OSMnx](https://github.com/gboeing/osmnx)
- **Building heights**: SF LIDAR import (CC0 license)
- **Population**: U.S. Census Bureau, 2020 Decennial Census
- **Restaurants**: OpenStreetMap amenity=restaurant POIs
- **Airspace**: FAA Part 107 (400ft AGL ceiling)

## Key References

- Doole, Ellerbroek, Knoop & Hoekstra (2021). "Constrained Urban Airspace Design for
  Large-Scale Drone-Based Delivery Traffic." *Aerospace*.
- Moosavi & Farooq (2025). "Sphereabout" — spherical roundabout intersections. *IEEE
  ITSC*.
- Bauranov & Rakas (2021). "Designing Airspace for Urban Air Mobility." *Progress in
  Aerospace Sciences*.
- Sunil et al. (2015). "Metropolis: Relating Airspace Structure and Capacity." *ICRAT*.
- Cummings & Mahmassani (2021). "Emergence of 4-D System Fundamental Diagram in UAM
  Traffic." *TRR*.
