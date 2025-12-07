import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import random

def sample_point_in_polygon(poly, max_tries=20):
    """Returns a random Point within a Polygon."""
    if poly is None or poly.is_empty:
        return None
    
    # Handle MultiPolygon: take the largest part
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)

    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        pt = Point(x, y)
        if poly.contains(pt):
            return pt
    return poly.representative_point()

def clean_geodataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cleans list/dict columns for GeoJSON export."""
    gdf_clean = gdf.copy()
    for col in gdf_clean.columns:
        if col == 'geometry': continue
        
        # Check a sample for list/dict types
        sample = gdf_clean[col].dropna().head(5)
        if not sample.empty:
            first = sample.iloc[0]
            if isinstance(first, (list, dict, pd.Series)):
                gdf_clean[col] = gdf_clean[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(gdf_clean[col]):
                gdf_clean[col] = gdf_clean[col].astype(str)
    return gdf_clean

def ensure_crs(gdf: gpd.GeoDataFrame, epsg=4326) -> gpd.GeoDataFrame:
    """Ensures the GeoDataFrame is in the target CRS."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    elif gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf