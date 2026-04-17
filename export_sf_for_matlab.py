import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox

from data_loader import (
    load_sf_street_network,
    load_sf_buildings,
    load_sf_census_population,
    load_sf_restaurants,
    compute_corridor_clearances,
)

OUTDIR = Path("data")
OUTDIR.mkdir(exist_ok=True)


def safe_str(x):
    return str(x).strip()


def export_for_matlab():
    print("Loading SF datasets...")
    G = load_sf_street_network()
    buildings = load_sf_buildings()
    census = load_sf_census_population()
    restaurants = load_sf_restaurants()

    print("Converting graph to GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    nodes_reset = nodes_gdf.reset_index()

    possible_node_cols = ["osmid", "node", "id", "index"]
    node_col = None
    for c in possible_node_cols:
        if c in nodes_reset.columns:
            node_col = c
            break
    if node_col is None:
        node_col = nodes_reset.columns[0]

    nodes_df = pd.DataFrame(
        {
            "node_id": nodes_reset[node_col].map(safe_str).values,
            "x": nodes_reset.geometry.x.values,
            "y": nodes_reset.geometry.y.values,
        }
    )

    nodes_df.to_csv(OUTDIR / "sf_nodes.csv", index=False)
    print(f"Saved {len(nodes_df)} nodes")

    # ---- Building Clearing Per Edge ----
    print("Computing corridor clearances...")
    clearances = compute_corridor_clearances(buildings, G)

    # ---- Edges ----
    edges_reset = edges_gdf.reset_index()

    for col in ["u", "v"]:
        if col not in edges_reset.columns:
            raise ValueError(f"Expected column '{col}' in edges GeoDataFrame")

    if "length" not in edges_reset.columns:
        edges_reset["length"] = edges_reset.geometry.length

    clearance_vals = []
    for idx in range(len(edges_reset)):
        clearance_vals.append(float(clearances.get(idx, 12.0)))

    edges_df = pd.DataFrame(
        {
            "u": edges_reset["u"].map(safe_str).values,
            "v": edges_reset["v"].map(safe_str).values,
            "length_m": pd.to_numeric(edges_reset["length"], errors="coerce")
            .fillna(0)
            .values,
            "corridor_height_m": np.array(clearance_vals, dtype=float),
        }
    )

    edges_df = edges_df.dropna(
        subset=["u", "v", "length_m", "corridor_height_m"]
    ).copy()
    edges_df = edges_df[edges_df["u"] != edges_df["v"]].copy()

    edges_df.to_csv(OUTDIR / "sf_edges.csv", index=False)
    print(f"Saved {len(edges_df)} edges")

    # ---- Census ----
    census_out = census.copy()
    census_out.to_csv(OUTDIR / "sf_census.csv", index=False)
    print(f"Saved {len(census_out)} census rows")

    # ---- Resteraunts ----
    restaurants_out = restaurants.copy()
    restaurants_out.to_csv(OUTDIR / "sf_restaurants.csv", index=False)
    print(f"Saved {len(restaurants_out)} restaurant rows")

    print("Done.")


if __name__ == "__main__":
    export_for_matlab()
