# Drone Delivery Simulation — Visualization & Controls Guide

## Overview

This simulation models a decentralized drone delivery system operating over a real-world map (San Francisco). The goal is to understand how drones move, interact, and scale under different routing and airspace design rules.

The interface provides:
- Multiple **visualizations** of system behavior
- Two **dropdown controls** to modify how drones move:
  - **Turn Type**
  - **Lane Type**

This guide explains what each visualization shows and how to interpret the dropdown options.

---

# 📊 Visualizations

## 1. Top-Down Route Map

### What it shows
- A 2D map of the street network
- Each colored line represents a drone’s route
- Green dot = pickup location (restaurant)
- Red square = delivery destination

### What it tells you
- Where drones are traveling
- Which areas have the highest traffic
- Whether routes are direct or constrained

---

## 2. 3D Trajectories

### What it shows
- Drone paths in 3D space
- X/Y = map position
- Z = altitude

### What it tells you
- How drones move vertically and horizontally
- How turns are executed in space

---

## 3. Route Efficiency

### What it shows
- Straight-line vs actual path distance

### What it tells you
- How efficient routes are

---

## 4. Demand Over Time

### What it shows
- Delivery requests over time

---

## 5. Time Distributions

### What it shows
- Delivery time and delays

---

## 6. Edge Utilization Heatmap

### What it shows
- Most-used routes

---

# 🎛️ Controls

## Turn Type

### Normal
- Sharp 90° turn
- Fast but abrupt

### Sphereabout
- Smooth quarter-circle turn
- Safer, slightly longer

## Lane Type

### Normal
- Follow street grid

### Intersection
- Diagonal intersection-based routing
- More direct paths

