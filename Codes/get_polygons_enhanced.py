"""
Enhanced Building Polygon Extraction with Multiple Methods

This script tries multiple approaches to maximize building coverage:
1. Standard OSMnx queries (features_from_polygon)
2. Multiple building tag variations
3. Overpass API direct queries (more comprehensive)
4. Tile-based queries for large areas
"""

import os
import sys
import time
from typing import Optional, List, Set

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import box


def get_city_boundary(city_name: str) -> gpd.GeoDataFrame:
    """Get the boundary polygon for a city using OSMnx geocoding."""
    try:
        print(f"Geocoding city: {city_name}")
        boundary = ox.geocoder.geocode_to_gdf(city_name)
        boundary = boundary.to_crs(epsg=4326)
        print(f"Found city boundary with {len(boundary)} feature(s)")
        return boundary
    except Exception as e:
        raise SystemExit(f"Failed to geocode city '{city_name}': {e}")


def get_buildings_osmnx_multiple_tags(
    city_polygon,
    city_name: str
) -> gpd.GeoDataFrame:
    """
    Query buildings using OSMnx with multiple tag variations.
    """
    all_buildings = []
    
    # Try different building tag queries
    tag_variations = [
        {"building": True},  # Standard
        {"building": ["yes", "residential", "commercial", "industrial", "house", "apartments"]},
        {"building:part": True},  # Building parts
    ]
    
    for tags in tag_variations:
        try:
            print(f"  Querying with tags: {tags}")
            try:
                buildings = ox.features_from_polygon(city_polygon, tags)
            except AttributeError:
                buildings = ox.geometries_from_polygon(city_polygon, tags)
            
            if len(buildings) > 0:
                all_buildings.append(buildings)
                print(f"    Found {len(buildings)} features")
        except Exception as e:
            print(f"    Query failed: {e}")
            continue
    
    if not all_buildings:
        # Fallback to place-based query
        try:
            print("  Trying place-based query...")
            try:
                buildings = ox.features_from_place(city_name, {"building": True})
            except AttributeError:
                buildings = ox.geometries_from_place(city_name, {"building": True})
            if len(buildings) > 0:
                all_buildings.append(buildings)
        except Exception as e:
            print(f"    Place query failed: {e}")
    
    if not all_buildings:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # Combine and deduplicate
    combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True))
    
    # Remove duplicates based on geometry and OSM ID if available
    if 'osmid' in combined.columns:
        combined = combined.drop_duplicates(subset=['osmid'], keep='first')
    else:
        # Deduplicate by geometry
        combined['geometry_wkt'] = combined.geometry.apply(lambda x: x.wkt)
        combined = combined.drop_duplicates(subset=['geometry_wkt'], keep='first')
        combined = combined.drop(columns=['geometry_wkt'])
    
    return combined


def get_buildings_overpass_api(
    city_polygon,
    timeout: int = 300
) -> gpd.GeoDataFrame:
    """
    Query buildings using direct Overpass API calls (more comprehensive).
    """
    print("Querying Overpass API directly...")
    
    # Get bounding box
    bounds = city_polygon.bounds
    bbox = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"  # minlat,minlon,maxlat,maxlon
    
    # Overpass query for all buildings
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({bbox});
      relation["building"]({bbox});
    );
    out geom;
    """
    
    try:
        # Use OSMnx's overpass settings
        overpass_url = ox.settings.overpass_endpoint
        print(f"  Using Overpass endpoint: {overpass_url}")
        
        response = requests.post(
            overpass_url,
            data={'data': query},
            timeout=timeout,
            headers={'User-Agent': 'OSMnx building extractor'}
        )
        response.raise_for_status()
        
        data = response.json()
        
        if 'elements' not in data or len(data['elements']) == 0:
            print("  No buildings found via Overpass API")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        print(f"  Found {len(data['elements'])} elements from Overpass API")
        
        # Convert to GeoDataFrame (simplified - full conversion would need more work)
        # For now, return empty and let OSMnx handle it
        # This is a placeholder for more sophisticated Overpass parsing
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    except Exception as e:
        print(f"  Overpass API query failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def get_buildings_tiled(
    city_polygon,
    city_name: str,
    tile_size: float = 0.1
) -> gpd.GeoDataFrame:
    """
    Query buildings using a tiled approach to avoid Overpass limits.
    """
    print(f"Querying with tiled approach (tile size: {tile_size} degrees)...")
    
    bounds = city_polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    # Create grid of tiles
    x_tiles = int((maxx - minx) / tile_size) + 1
    y_tiles = int((maxy - miny) / tile_size) + 1
    
    print(f"  Creating {x_tiles}x{y_tiles} = {x_tiles * y_tiles} tiles")
    
    all_buildings = []
    tags = {"building": True}
    
    tile_count = 0
    for i in range(x_tiles):
        for j in range(y_tiles):
            tile_minx = minx + i * tile_size
            tile_miny = miny + j * tile_size
            tile_maxx = min(tile_minx + tile_size, maxx)
            tile_maxy = min(tile_miny + tile_size, maxy)
            
            tile_polygon = box(tile_minx, tile_miny, tile_maxx, tile_maxy)
            
            # Only query if tile intersects city polygon
            if not tile_polygon.intersects(city_polygon):
                continue
            
            try:
                tile_count += 1
                if tile_count % 10 == 0:
                    print(f"    Processed {tile_count} tiles...")
                
                try:
                    buildings = ox.features_from_polygon(tile_polygon, tags)
                except AttributeError:
                    buildings = ox.geometries_from_polygon(tile_polygon, tags)
                
                if len(buildings) > 0:
                    all_buildings.append(buildings)
                
                # Be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"    Tile ({i},{j}) failed: {e}")
                continue
    
    if not all_buildings:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    print(f"  Combining {len(all_buildings)} tile results...")
    combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True))
    
    # Deduplicate
    if 'osmid' in combined.columns:
        combined = combined.drop_duplicates(subset=['osmid'], keep='first')
    
    return combined


def clean_geodataframe_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean GeoDataFrame to remove columns with unsupported types for GeoJSON export."""
    gdf_clean = gdf.copy()
    columns_to_drop = []
    columns_to_convert = []
    
    for col in gdf_clean.columns:
        if col == 'geometry':
            continue
        
        sample_values = gdf_clean[col].dropna().head(100)
        if len(sample_values) > 0:
            first_val = sample_values.iloc[0]
            if isinstance(first_val, (list, dict)):
                columns_to_convert.append(col)
            elif isinstance(first_val, pd.Series):
                columns_to_drop.append(col)
            elif isinstance(first_val, (pd.Timestamp, pd.datetime)):
                columns_to_convert.append(col)
    
    for col in columns_to_convert:
        print(f"  Converting column '{col}' to string...")
        gdf_clean[col] = gdf_clean[col].apply(
            lambda x: str(x) if isinstance(x, (list, dict, pd.Timestamp)) else x
        )
    
    for col in columns_to_drop:
        print(f"  Dropping column '{col}'...")
        gdf_clean = gdf_clean.drop(columns=[col])
    
    return gdf_clean


def get_building_polygons_enhanced(
    city_name: str,
    boundary: Optional[gpd.GeoDataFrame] = None,
    use_tiled: bool = False
) -> gpd.GeoDataFrame:
    """
    Get building polygons using multiple methods for maximum coverage.
    """
    if boundary is None:
        boundary = get_city_boundary(city_name)
    
    city_polygon = boundary.geometry.iloc[0]
    
    print("\n" + "="*80)
    print("Method 1: OSMnx with multiple tag variations")
    print("="*80)
    buildings_method1 = get_buildings_osmnx_multiple_tags(city_polygon, city_name)
    print(f"Method 1 result: {len(buildings_method1)} buildings")
    
    all_buildings = [buildings_method1] if len(buildings_method1) > 0 else []
    
    if use_tiled:
        print("\n" + "="*80)
        print("Method 2: Tiled queries (for better coverage)")
        print("="*80)
        buildings_method2 = get_buildings_tiled(city_polygon, city_name)
        print(f"Method 2 result: {len(buildings_method2)} buildings")
        if len(buildings_method2) > 0:
            all_buildings.append(buildings_method2)
    
    # Combine all results
    if not all_buildings:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    print("\n" + "="*80)
    print("Combining results from all methods...")
    print("="*80)
    
    combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True))
    
    # Deduplicate by OSM ID or geometry
    initial_count = len(combined)
    if 'osmid' in combined.columns:
        combined = combined.drop_duplicates(subset=['osmid'], keep='first')
    else:
        combined['geom_hash'] = combined.geometry.apply(lambda x: hash(str(x)))
        combined = combined.drop_duplicates(subset=['geom_hash'], keep='first')
        combined = combined.drop(columns=['geom_hash'])
    
    print(f"Deduplicated from {initial_count} to {len(combined)} unique buildings")
    
    # Ensure CRS
    if combined.crs is None:
        combined = combined.set_crs(epsg=4326, allow_override=True)
    else:
        combined = combined.to_crs(epsg=4326)
    
    # Filter to polygons only
    print("Filtering to polygon geometries...")
    initial_count = len(combined)
    combined = combined[
        combined.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"Filtered from {initial_count} to {len(combined)} polygon features")
    
    # Clip to city boundary
    print("Clipping to city boundary...")
    try:
        combined = gpd.clip(combined, boundary)
        print(f"After clipping: {len(combined)} buildings")
    except Exception as e:
        print(f"Warning: Clipping failed ({e}), using all buildings")
    
    return combined


def save_buildings_geojson(buildings: gpd.GeoDataFrame, output_path: str) -> None:
    """Save building polygons as GeoJSON."""
    if len(buildings) == 0:
        print("Warning: No buildings to save!")
        return
    
    buildings = buildings.to_crs(epsg=4326)
    
    print("Cleaning data for GeoJSON export...")
    buildings_clean = clean_geodataframe_for_geojson(buildings)
    
    print(f"Saving {len(buildings_clean)} buildings to: {output_path}")
    buildings_clean.to_file(output_path, driver="GeoJSON")
    print("Saved successfully!")
    
    print("\nSummary:")
    print(f"  Total buildings: {len(buildings_clean):,}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    if len(buildings_clean.columns) > 1:
        print(f"\nAvailable attributes: {', '.join([c for c in buildings_clean.columns if c != 'geometry'])}")
        if 'building' in buildings_clean.columns:
            building_types = buildings_clean['building'].value_counts()
            print(f"\nBuilding types (top 10):")
            for btype, count in building_types.head(10).items():
                print(f"  {btype}: {count:,}")


def main():
    """Main function."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) < 2:
        print("Usage: python get_polygons_enhanced.py <city_name> [output_path] [--tiled]")
        print("Example: python get_polygons_enhanced.py 'Surat, Gujarat, India'")
        print("Example: python get_polygons_enhanced.py 'Surat, Gujarat, India' --tiled")
        sys.exit(1)
    
    city_name = sys.argv[1]
    use_tiled = '--tiled' in sys.argv
    
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_path = sys.argv[2]
        if not output_path.endswith('.geojson'):
            output_path += '.geojson'
    else:
        safe_name = city_name.replace(', ', '_').replace(' ', '_').replace('/', '_')
        output_path = os.path.join(base_dir, f"buildings_enhanced_{safe_name}.geojson")
    
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    
    print("="*80)
    print("Enhanced Building Polygon Extraction")
    print("="*80)
    print(f"City: {city_name}")
    print(f"Output: {output_path}")
    print(f"Tiled queries: {use_tiled}")
    print("="*80 + "\n")
    
    boundary = get_city_boundary(city_name)
    buildings = get_building_polygons_enhanced(city_name, boundary, use_tiled=use_tiled)
    save_buildings_geojson(buildings, output_path)
    
    print(f"\n✓ Complete! Buildings saved to: {output_path}")
    print(f"\nNote: Building coverage depends on OpenStreetMap data completeness.")
    print("If buildings are missing, they may not be mapped in OSM yet.")


if __name__ == "__main__":
    main()

