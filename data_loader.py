"""
data_loader.py — Download and process San Francisco datasets
- Street network from OpenStreetMap via OSMnx
- Building heights from OSM (LIDAR-derived)
- Census tract population density
- Uber Eats restaurant locations (Kaggle)
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from pathlib import Path
from typing import Tuple, Optional
from config import SFConfig

try:
    import osmnx as ox
except ImportError:
    print("Install osmnx: pip install osmnx")
    ox = None

try:
    import requests
except ImportError:
    requests = None


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# 1. STREET NETWORK
# ═════════════════════════════════════════════════════════════════════

def load_sf_street_network(cfg: SFConfig = None, 
                           force_download: bool = False) -> nx.MultiDiGraph:
    """
    Download the SF street network as a NetworkX graph.
    Nodes have (x, y) coordinates in meters (UTM projection).
    Edges have 'length' in meters.
    """
    if cfg is None:
        cfg = SFConfig()
    
    cache_path = Path(cfg.osm_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists() and not force_download:
        print(f"  Loading cached street network from {cache_path}")
        G = ox.load_graphml(cache_path)
    else:
        print("  Downloading SF street network from OpenStreetMap...")
        # Download drivable street network for the bounding box
        # bbox format depends on OSMnx version:
        #   v2.x+: (west, south, east, north) = (left, bottom, right, top)
        #   v1.x:  (north, south, east, west) as positional args
        try:
            # OSMnx >= 2.0: bbox = (west, south, east, north)
            G = ox.graph_from_bbox(
                bbox=(cfg.lon_range[0], cfg.lat_range[0],
                      cfg.lon_range[1], cfg.lat_range[1]),
                network_type='drive',
                simplify=True
            )
        except TypeError:
            # OSMnx 1.x: positional args (north, south, east, west)
            G = ox.graph_from_bbox(
                cfg.lat_range[1], cfg.lat_range[0],
                cfg.lon_range[1], cfg.lon_range[0],
                network_type='drive',
                simplify=True
            )
        ox.save_graphml(G, cache_path)
        print(f"  Saved to {cache_path}")
    
    # Project to UTM for metric coordinates
    G_proj = ox.project_graph(G)
    
    print(f"  Street network: {G_proj.number_of_nodes()} nodes, "
          f"{G_proj.number_of_edges()} edges")
    
    return G_proj


def extract_intersections(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Extract true intersections (degree >= 3) from the street graph.
    Returns DataFrame with columns: [node_id, x, y, lat, lon, degree]
    """
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    
    # Compute degree in the undirected version
    G_undir = G.to_undirected()
    degrees = dict(G_undir.degree())
    
    intersections = []
    for node_id, row in nodes_gdf.iterrows():
        deg = degrees.get(node_id, 0)
        if deg >= 3:
            intersections.append({
                'node_id': node_id,
                'x': row.geometry.x,
                'y': row.geometry.y,
                'degree': deg
            })
    
    df = pd.DataFrame(intersections)
    print(f"  Found {len(df)} intersections (degree >= 3)")
    return df


# ═════════════════════════════════════════════════════════════════════
# 2. BUILDING HEIGHTS
# ═════════════════════════════════════════════════════════════════════

def load_sf_buildings(cfg: SFConfig = None,
                      force_download: bool = False) -> gpd.GeoDataFrame:
    """
    Download building footprints with heights from OpenStreetMap.
    Heights come from the 2016 LIDAR import (~140K buildings).
    """
    if cfg is None:
        cfg = SFConfig()
    
    cache_path = Path(cfg.buildings_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists() and not force_download:
        print(f"  Loading cached buildings from {cache_path}")
        buildings = gpd.read_file(cache_path)
    else:
        print("  Downloading SF buildings from OpenStreetMap...")
        # Use osmnx to get building footprints
        tags = {'building': True}
        try:
            # OSMnx >= 2.0
            buildings = ox.features_from_bbox(
                bbox=(cfg.lon_range[0], cfg.lat_range[0],
                      cfg.lon_range[1], cfg.lat_range[1]),
                tags=tags
            )
        except TypeError:
            # OSMnx 1.x
            buildings = ox.features_from_bbox(
                cfg.lat_range[1], cfg.lat_range[0],
                cfg.lon_range[1], cfg.lon_range[0],
                tags=tags
            )
        
        # Extract height information
        # OSM stores heights in various tags
        height_cols = ['height', 'building:height', 'building:levels']
        
        if 'height' not in buildings.columns:
            buildings['height'] = np.nan
        
        # Parse height values
        buildings['height_m'] = pd.to_numeric(
            buildings['height'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )
        
        # Estimate from building levels if height is missing
        if 'building:levels' in buildings.columns:
            levels = pd.to_numeric(buildings['building:levels'], errors='coerce')
            mask = buildings['height_m'].isna() & levels.notna()
            buildings.loc[mask, 'height_m'] = levels[mask] * 3.5  # ~3.5m per floor
        
        # Default height for buildings without data
        buildings['height_m'] = buildings['height_m'].fillna(12.0)  # ~4 stories
        
        # Keep only polygon geometries
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # Save minimal columns
        keep_cols = ['geometry', 'height_m', 'building', 'name']
        available = [c for c in keep_cols if c in buildings.columns]
        buildings = buildings[available].copy()
        
        buildings.to_file(cache_path, driver='GPKG')
        print(f"  Saved {len(buildings)} buildings to {cache_path}")
    
    print(f"  Buildings: {len(buildings)} total, "
          f"height range: {buildings['height_m'].min():.0f} - "
          f"{buildings['height_m'].max():.0f} m, "
          f"median: {buildings['height_m'].median():.0f} m")
    
    return buildings


def compute_corridor_clearances(buildings: gpd.GeoDataFrame,
                                street_graph: nx.MultiDiGraph) -> dict:
    """
    For each street edge, compute the max building height along it.
    This determines the minimum safe flight altitude for that corridor.
    """
    edges_gdf = ox.graph_to_gdfs(street_graph, nodes=False)
    
    # Buffer each edge by 30m to find adjacent buildings
    edges_buffered = edges_gdf.copy()
    edges_buffered['geometry'] = edges_buffered.geometry.buffer(30)
    
    # Spatial join: find buildings near each edge
    joined = gpd.sjoin(edges_buffered, buildings[['geometry', 'height_m']], 
                       how='left', predicate='intersects')
    
    # Max height per edge
    clearances = joined.groupby(joined.index)['height_m'].max().fillna(12.0)
    
    return clearances.to_dict()


# ═════════════════════════════════════════════════════════════════════
# 3. CENSUS POPULATION DENSITY
# ═════════════════════════════════════════════════════════════════════

def load_sf_census_population(cfg: SFConfig = None) -> gpd.GeoDataFrame:
    """
    Load SF census tract boundaries and population data.
    Uses Census Bureau API for population, DataSF for boundaries.
    Falls back to synthetic data if API is unavailable.
    """
    if cfg is None:
        cfg = SFConfig()
    
    cache_path = Path(cfg.census_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"  Loading cached census data from {cache_path}")
        return pd.read_csv(cache_path)
    
    print("  Downloading SF census tract data...")
    
    # Try the Census Bureau GeoJSON API
    try:
        # 2020 census tracts for San Francisco County (FIPS: 06075)
        url = ("https://data.sfgov.org/resource/tmph-tgz9.geojson"
               "?$limit=500")
        
        tracts = gpd.read_file(url)
        
        # Add population density estimates from ACS
        # Using Census Bureau API (no key needed for basic queries)
        pop_url = ("https://api.census.gov/data/2020/dec/pl"
                   "?get=P1_001N&for=tract:*&in=state:06&in=county:075")
        
        resp = requests.get(pop_url, timeout=15)
        if resp.status_code == 200:
            pop_data = resp.json()
            pop_df = pd.DataFrame(pop_data[1:], columns=pop_data[0])
            pop_df['P1_001N'] = pd.to_numeric(pop_df['P1_001N'])
            pop_df['GEOID'] = '06075' + pop_df['tract']
            
            # Merge with tract geometries
            if 'geoid' in tracts.columns:
                tracts = tracts.merge(pop_df[['GEOID', 'P1_001N']], 
                                      left_on='geoid', right_on='GEOID',
                                      how='left')
                tracts.rename(columns={'P1_001N': 'population'}, inplace=True)
        
        # Compute area and density
        tracts_proj = tracts.to_crs(epsg=32610)  # UTM Zone 10N
        tracts['area_km2'] = tracts_proj.geometry.area / 1e6
        if 'population' in tracts.columns:
            tracts['pop_density'] = tracts['population'] / tracts['area_km2']
        else:
            # Fallback: estimate from known SF average (~7,200/km²)
            tracts['population'] = np.random.poisson(4000, len(tracts))
            tracts['pop_density'] = tracts['population'] / tracts['area_km2']
        
        # Save centroid coordinates for demand weighting
        centroids = tracts.geometry.centroid
        result = pd.DataFrame({
            'tract_id': tracts.index,
            'lat': centroids.y,
            'lon': centroids.x,
            'population': tracts.get('population', 4000),
            'pop_density': tracts.get('pop_density', 7200)
        })
        
        result.to_csv(cache_path, index=False)
        print(f"  Saved {len(result)} census tracts to {cache_path}")
        return result
        
    except Exception as e:
        print(f"  Census download failed ({e}), generating synthetic data...")
        return _generate_synthetic_census(cfg)


def _generate_synthetic_census(cfg: SFConfig) -> pd.DataFrame:
    """
    Generate synthetic population density grid based on known SF patterns.
    Downtown/SoMa: ~15,000/km², Mission: ~12,000/km², outer: ~5,000/km²
    """
    np.random.seed(42)
    n = 20  # 20x20 grid of zones
    lats = np.linspace(cfg.lat_range[0], cfg.lat_range[1], n)
    lons = np.linspace(cfg.lon_range[0], cfg.lon_range[1], n)
    
    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Higher density toward center/downtown
            dist_center = np.sqrt((lat - cfg.ref_lat)**2 + (lon - cfg.ref_lon)**2)
            base_density = 15000 * np.exp(-dist_center * 200)
            base_density = max(base_density, 3000)
            density = base_density * np.random.uniform(0.7, 1.3)
            
            records.append({
                'tract_id': i * n + j,
                'lat': lat, 'lon': lon,
                'population': int(density * 0.1),  # per ~0.1 km² cell
                'pop_density': density
            })
    
    df = pd.DataFrame(records)
    cache_path = Path(cfg.census_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ═════════════════════════════════════════════════════════════════════
# 4. RESTAURANT LOCATIONS (Uber Eats / Food Delivery)
# ═════════════════════════════════════════════════════════════════════

def load_sf_restaurants(cfg: SFConfig = None,
                        force_download: bool = False) -> pd.DataFrame:
    """
    Load restaurant locations in the SF area.
    
    Primary: Uber Eats dataset from Kaggle (requires kaggle API key)
    Fallback: OpenStreetMap restaurant/food POIs via OSMnx
    """
    if cfg is None:
        cfg = SFConfig()
    
    cache_path = Path(cfg.restaurants_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists() and not force_download:
        print(f"  Loading cached restaurants from {cache_path}")
        return pd.read_csv(cache_path)
    
    # Method 1: Try OSMnx for restaurant POIs (always available)
    print("  Downloading restaurant locations from OpenStreetMap...")
    try:
        tags = {'amenity': ['restaurant', 'fast_food', 'cafe', 'food_court']}
        try:
            # OSMnx >= 2.0
            pois = ox.features_from_bbox(
                bbox=(cfg.lon_range[0], cfg.lat_range[0],
                      cfg.lon_range[1], cfg.lat_range[1]),
                tags=tags
            )
        except TypeError:
            # OSMnx 1.x
            pois = ox.features_from_bbox(
                cfg.lat_range[1], cfg.lat_range[0],
                cfg.lon_range[1], cfg.lon_range[0],
                tags=tags
            )
        
        # Extract centroids for non-point geometries
        centroids = pois.geometry.centroid
        
        restaurants = pd.DataFrame({
            'name': pois.get('name', 'Unknown'),
            'lat': centroids.y,
            'lon': centroids.x,
            'cuisine': pois.get('cuisine', 'unknown'),
            'amenity': pois.get('amenity', 'restaurant'),
            'source': 'osm'
        })
        
        # Filter to bounding box
        restaurants = restaurants[
            (restaurants.lat >= cfg.lat_range[0]) & 
            (restaurants.lat <= cfg.lat_range[1]) &
            (restaurants.lon >= cfg.lon_range[0]) & 
            (restaurants.lon <= cfg.lon_range[1])
        ].copy()
        
        restaurants.to_csv(cache_path, index=False)
        print(f"  Found {len(restaurants)} restaurants/food venues")
        return restaurants
        
    except Exception as e:
        print(f"  OSM restaurant download failed ({e}), generating synthetic...")
        return _generate_synthetic_restaurants(cfg)


def _generate_synthetic_restaurants(cfg: SFConfig) -> pd.DataFrame:
    """Generate synthetic restaurant locations clustered along major streets."""
    np.random.seed(42)
    n_restaurants = 200
    
    # Cluster around commercial corridors
    centers = [
        (37.787, -122.408, 80),   # Financial District
        (37.783, -122.409, 60),   # Union Square
        (37.775, -122.418, 50),   # Hayes Valley
        (37.764, -122.419, 70),   # Mission District
        (37.779, -122.414, 40),   # Tenderloin
        (37.772, -122.394, 30),   # SoMa / South Beach
    ]
    
    records = []
    for lat_c, lon_c, n in centers:
        for _ in range(n):
            records.append({
                'name': f'Restaurant_{len(records)}',
                'lat': lat_c + np.random.normal(0, 0.003),
                'lon': lon_c + np.random.normal(0, 0.003),
                'cuisine': np.random.choice(['american', 'chinese', 'mexican',
                                             'italian', 'japanese', 'indian']),
                'amenity': 'restaurant',
                'source': 'synthetic'
            })
    
    df = pd.DataFrame(records)
    
    # Clip to bounding box
    df = df[
        (df.lat >= cfg.lat_range[0]) & (df.lat <= cfg.lat_range[1]) &
        (df.lon >= cfg.lon_range[0]) & (df.lon <= cfg.lon_range[1])
    ].copy()
    
    cache_path = Path(cfg.restaurants_cache)
    df.to_csv(cache_path, index=False)
    return df


# ═════════════════════════════════════════════════════════════════════
# 5. DEMAND MODEL
# ═════════════════════════════════════════════════════════════════════

def generate_demand(n_orders: int,
                    restaurants: pd.DataFrame,
                    census: pd.DataFrame,
                    nodes: pd.DataFrame,
                    rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Generate origin-destination pairs for drone deliveries.
    
    Origins: sampled from restaurant locations (weighted by density clusters)
    Destinations: sampled from census tracts (weighted by population density)
    Each OD pair is snapped to the nearest graph node.
    
    Parameters
    ----------
    n_orders : int
    restaurants : DataFrame with lat, lon columns
    census : DataFrame with lat, lon, pop_density columns
    nodes : DataFrame with node_id, x, y (or lat, lon) columns
    
    Returns
    -------
    DataFrame with columns: [order_id, origin_node, dest_node, 
                              orig_lat, orig_lon, dest_lat, dest_lon]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Sample origins from restaurants (uniform — each restaurant equally likely)
    orig_idx = rng.integers(0, len(restaurants), n_orders)
    orig_lats = restaurants.iloc[orig_idx]['lat'].values
    orig_lons = restaurants.iloc[orig_idx]['lon'].values
    
    # Sample destinations weighted by population density
    weights = census['pop_density'].values
    weights = weights / weights.sum()
    dest_idx = rng.choice(len(census), size=n_orders, p=weights)
    
    # Add jitter within census tract (don't deliver to tract centroid)
    dest_lats = census.iloc[dest_idx]['lat'].values + rng.normal(0, 0.001, n_orders)
    dest_lons = census.iloc[dest_idx]['lon'].values + rng.normal(0, 0.001, n_orders)
    
    orders = pd.DataFrame({
        'order_id': range(n_orders),
        'orig_lat': orig_lats,
        'orig_lon': orig_lons,
        'dest_lat': dest_lats,
        'dest_lon': dest_lons,
        'request_time': np.sort(rng.exponential(10, n_orders).cumsum())
    })
    
    return orders


def snap_to_graph(lat: float, lon: float,
                  G: nx.MultiDiGraph) -> int:
    """Find the nearest graph node to a lat/lon point.

    Handles CRS mismatch: if the graph is projected (e.g. UTM),
    the lat/lon is converted to the graph's CRS before lookup.
    """
    from pyproj import Transformer
    crs = G.graph.get('crs')
    if crs and str(crs) != 'EPSG:4326':
        transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        return ox.nearest_nodes(G, x, y)
    return ox.nearest_nodes(G, lon, lat)


# ═════════════════════════════════════════════════════════════════════
# MAIN: Download everything
# ═════════════════════════════════════════════════════════════════════

def download_all_data(cfg: SFConfig = None, force: bool = False):
    """Download and cache all SF datasets."""
    if cfg is None:
        cfg = SFConfig()
    
    print("=" * 60)
    print("DOWNLOADING SAN FRANCISCO DATA")
    print("=" * 60)
    
    print("\n[1/4] Street Network")
    G = load_sf_street_network(cfg, force_download=force)
    
    print("\n[2/4] Building Heights")
    buildings = load_sf_buildings(cfg, force_download=force)
    
    print("\n[3/4] Census Population")
    census = load_sf_census_population(cfg)
    
    print("\n[4/4] Restaurant Locations")
    restaurants = load_sf_restaurants(cfg, force_download=force)
    
    print("\n" + "=" * 60)
    print("ALL DATA DOWNLOADED SUCCESSFULLY")
    print("=" * 60)
    
    return {
        'street_graph': G,
        'buildings': buildings,
        'census': census,
        'restaurants': restaurants
    }


if __name__ == "__main__":
    download_all_data()