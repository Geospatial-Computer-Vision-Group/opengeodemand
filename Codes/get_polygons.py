"""
Extract Building Polygons for a City using OSMnx

This script:
1. Geocodes a city name to get its boundary
2. Downloads all building polygons from OpenStreetMap
3. Saves them as GeoJSON (JSON format)
"""

import os
import sys
from typing import Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd


def get_city_boundary(city_name: str) -> gpd.GeoDataFrame:
    """
    Get the boundary polygon for a city using OSMnx geocoding.
    
    Args:
        city_name: City name (e.g., "Surat, Gujarat, India")
    
    Returns:
        GeoDataFrame with city boundary geometry
    """
    try:
        print(f"Geocoding city: {city_name}")
        boundary = ox.geocoder.geocode_to_gdf(city_name)
        boundary = boundary.to_crs(epsg=4326)
        print(f"Found city boundary with {len(boundary)} feature(s)")
        return boundary
    except Exception as e:
        raise SystemExit(f"Failed to geocode city '{city_name}': {e}")


def get_building_polygons(city_name: str, boundary: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """
    Get all building polygons for a city.
    
    Args:
        city_name: City name for querying
        boundary: Optional pre-computed city boundary (if None, will geocode)
    
    Returns:
        GeoDataFrame with building polygons
    """
    if boundary is None:
        boundary = get_city_boundary(city_name)
    
    # Get the city polygon (use first geometry if multiple)
    city_polygon = boundary.geometry.iloc[0]
    
    # Query building polygons
    print("Querying building polygons from OpenStreetMap...")
    tags_buildings = {"building": True}
    
    try:
        # Try polygon-based query first (more precise)
        # Use features_from_polygon (new API) with fallback to geometries_from_polygon (old API)
        try:
            buildings = ox.features_from_polygon(city_polygon, tags_buildings)
        except AttributeError:
            # Fallback for older OSMnx versions
            buildings = ox.geometries_from_polygon(city_polygon, tags_buildings)
        print(f"Found {len(buildings)} building features using polygon query")
    except Exception as e:
        print(f"Polygon query failed ({e}), trying place-based query...")
        try:
            # Fallback to place-based query
            try:
                buildings = ox.features_from_place(city_name, tags_buildings)
            except AttributeError:
                # Fallback for older OSMnx versions
                buildings = ox.geometries_from_place(city_name, tags_buildings)
            print(f"Found {len(buildings)} building features using place query")
        except Exception as e2:
            raise SystemExit(f"Failed to get buildings: {e2}")
    
    # Ensure CRS is WGS84
    if buildings.crs is None:
        buildings = buildings.set_crs(epsg=4326, allow_override=True)
    else:
        buildings = buildings.to_crs(epsg=4326)
    
    # Filter to only polygon geometries (exclude points, lines)
    print("Filtering to polygon geometries...")
    initial_count = len(buildings)
    buildings = buildings[
        buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"Filtered from {initial_count} to {len(buildings)} polygon features")
    
    # Clip to city boundary to ensure we only get buildings within the city
    print("Clipping buildings to city boundary...")
    try:
        buildings = gpd.clip(buildings, boundary)
        print(f"After clipping: {len(buildings)} buildings")
    except Exception as e:
        print(f"Warning: Clipping failed ({e}), using all buildings")
    
    return buildings


def clean_geodataframe_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean GeoDataFrame to remove columns with unsupported types (lists, dicts) for GeoJSON export.
    
    Args:
        gdf: GeoDataFrame to clean
    
    Returns:
        Cleaned GeoDataFrame
    """
    gdf_clean = gdf.copy()
    columns_to_drop = []
    columns_to_convert = []
    
    for col in gdf_clean.columns:
        if col == 'geometry':
            continue
        
        # Check if column contains lists or dicts
        sample_values = gdf_clean[col].dropna().head(100)
        if len(sample_values) > 0:
            first_val = sample_values.iloc[0]
            if isinstance(first_val, (list, dict)):
                # Convert to string representation
                columns_to_convert.append(col)
            elif isinstance(first_val, pd.Series):
                # Drop complex nested structures
                columns_to_drop.append(col)
    
    # Convert list/dict columns to strings
    for col in columns_to_convert:
        print(f"  Converting column '{col}' (list/dict type) to string...")
        gdf_clean[col] = gdf_clean[col].apply(
            lambda x: str(x) if isinstance(x, (list, dict)) else x
        )
    
    # Drop problematic columns
    for col in columns_to_drop:
        print(f"  Dropping column '{col}' (unsupported type)...")
        gdf_clean = gdf_clean.drop(columns=[col])
    
    return gdf_clean


def save_buildings_geojson(buildings: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Save building polygons as GeoJSON.
    
    Args:
        buildings: GeoDataFrame with building polygons
        output_path: Path to save GeoJSON file
    """
    if len(buildings) == 0:
        print("Warning: No buildings to save!")
        return
    
    # Ensure WGS84 for GeoJSON
    buildings = buildings.to_crs(epsg=4326)
    
    # Clean the GeoDataFrame to remove unsupported types
    print("Cleaning data for GeoJSON export...")
    buildings_clean = clean_geodataframe_for_geojson(buildings)
    
    # Save as GeoJSON
    print(f"Saving {len(buildings_clean)} buildings to: {output_path}")
    buildings_clean.to_file(output_path, driver="GeoJSON")
    print("Saved successfully!")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Total buildings: {len(buildings):,}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Show available attributes
    if len(buildings.columns) > 1:  # More than just geometry
        print(f"\nAvailable attributes: {', '.join([c for c in buildings.columns if c != 'geometry'])}")
        
        # Show building type distribution if available
        if 'building' in buildings.columns:
            building_types = buildings['building'].value_counts()
            print(f"\nBuilding types (top 10):")
            for btype, count in building_types.head(10).items():
                print(f"  {btype}: {count:,}")


def main():
    """Main function to extract and save building polygons."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python get_polygons.py <city_name> [output_path]")
        print("Example: python get_polygons.py 'Surat, Gujarat, India'")
        print("Example: python get_polygons.py 'Mumbai, Maharashtra, India' buildings_mumbai.geojson")
        sys.exit(1)
    
    city_name = sys.argv[1]
    
    # Generate output path if not provided
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        if not output_path.endswith('.geojson'):
            output_path += '.geojson'
    else:
        # Default: create filename from city name
        safe_name = city_name.replace(', ', '_').replace(' ', '_').replace('/', '_')
        output_path = os.path.join(base_dir, f"buildings_{safe_name}.geojson")
    
    # Ensure absolute path
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    
    print("="*80)
    print("Building Polygon Extraction")
    print("="*80)
    print(f"City: {city_name}")
    print(f"Output: {output_path}")
    print("="*80 + "\n")
    
    # Get city boundary
    boundary = get_city_boundary(city_name)
    
    # Get building polygons
    buildings = get_building_polygons(city_name, boundary)
    
    # Save as GeoJSON
    save_buildings_geojson(buildings, output_path)
    
    print(f"\n✓ Complete! Buildings saved to: {output_path}")


if __name__ == "__main__":
    main()

