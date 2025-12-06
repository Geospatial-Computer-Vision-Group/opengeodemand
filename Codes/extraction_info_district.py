import os
import sys
from typing import List, Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd


def read_vector(path: str) -> gpd.GeoDataFrame:
    """Read a vector file (GeoJSON, GPKG, etc.) with fallback engines."""
    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        try:
            return gpd.read_file(path)
        except AttributeError as e:
            raise SystemExit(
                "Reading file failed due to Fiona/GDAL mismatch.\n"
                "Conda: conda install -c conda-forge pyogrio geopandas fiona shapely pyproj gdal\n"
                "Pip:   pip install -U pyogrio geopandas fiona shapely pyproj gdal\n"
                f"Original error: {e}"
            )


def get_city_boundary(city_name: str) -> gpd.GeoDataFrame:
    """
    Get the boundary polygon for a city using OSMnx geocoding.
    
    Args:
        city_name: City name (e.g., "Surat, Gujarat, India")
    
    Returns:
        GeoDataFrame with city boundary geometry
    """
    try:
        boundary = ox.geocoder.geocode_to_gdf(city_name)
        boundary = boundary.to_crs(epsg=4326)
        return boundary
    except Exception as e:
        raise SystemExit(f"Failed to geocode city '{city_name}': {e}")


def find_intersecting_districts(
    city_boundary: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Find all districts that intersect with the city boundary.
    
    Args:
        city_boundary: GeoDataFrame with city boundary
        districts: GeoDataFrame with district polygons
    
    Returns:
        GeoDataFrame with intersecting districts and all their attributes
    """
    # Ensure both are in the same CRS
    if districts.crs != city_boundary.crs:
        districts = districts.to_crs(city_boundary.crs)
    
    # Get the city polygon (use first geometry if multiple)
    city_geom = city_boundary.geometry.iloc[0]
    
    # Find districts that intersect with the city boundary
    intersecting = districts[districts.geometry.intersects(city_geom)].copy()
    
    return intersecting


def extract_district_data(city_name: str, geojson_path: str) -> gpd.GeoDataFrame:
    """
    Main function: Extract district data for a given city.
    
    Args:
        city_name: City name to geocode (e.g., "Surat, Gujarat, India")
        geojson_path: Path to the district GeoJSON file
    
    Returns:
        GeoDataFrame with all district data that intersects the city
    """
    if not os.path.exists(geojson_path):
        raise SystemExit(f"GeoJSON file not found: {geojson_path}")
    
    print(f"Geocoding city: {city_name}")
    city_boundary = get_city_boundary(city_name)
    
    print(f"Loading districts from: {geojson_path}")
    districts = read_vector(geojson_path)
    
    print("Finding intersecting districts...")
    intersecting_districts = find_intersecting_districts(city_boundary, districts)
    
    print(f"Found {len(intersecting_districts)} district(s) intersecting with {city_name}")
    
    return intersecting_districts


def main():
    """Command-line interface for district data extraction."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_geojson = os.path.join(base_dir, "india_district_ec_hces.geojson")
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python extraction_info_district.py <city_name> [geojson_path]")
        print(f"Example: python extraction_info_district.py 'Surat, Gujarat, India'")
        print(f"Default GeoJSON: {default_geojson}")
        sys.exit(1)
    
    city_name = sys.argv[1]
    geojson_path = sys.argv[2] if len(sys.argv) > 2 else default_geojson
    
    # Extract district data
    result = extract_district_data(city_name, geojson_path)
    
    if len(result) == 0:
        print(f"No districts found intersecting with {city_name}")
        sys.exit(0)
    
    # Display summary
    print("\n" + "="*80)
    print(f"District data for: {city_name}")
    print("="*80)
    
    # Show district names if available
    name_cols = [c for c in result.columns if "name" in c.lower() or "district" in c.lower()]
    if name_cols:
        print("\nDistrict(s):")
        for idx, row in result.iterrows():
            names = [str(row[c]) for c in name_cols if pd.notna(row[c])]
            if names:
                print(f"  - {', '.join(names)}")
    
    # Show key metrics (EC and HCES)
    print("\nKey Metrics:")
    ec_cols = [c for c in result.columns if c.startswith("ec_")]
    hces_cols = [c for c in result.columns if c.startswith("hces_")]
    
    if ec_cols:
        print("\nEconomic Census metrics:")
        for col in ec_cols[:10]:  # Show first 10
            val = result[col].iloc[0] if len(result) == 1 else result[col].sum()
            print(f"  {col}: {val:,.2f}")
        if len(ec_cols) > 10:
            print(f"  ... and {len(ec_cols) - 10} more EC metrics")
    
    if hces_cols:
        print("\nHCES metrics:")
        for col in hces_cols[:10]:  # Show first 10
            val = result[col].iloc[0] if len(result) == 1 else result[col].sum()
            print(f"  {col}: {val:,.2f}")
        if len(hces_cols) > 10:
            print(f"  ... and {len(hces_cols) - 10} more HCES metrics")
    
    # Save to CSV (without geometry for easier viewing)
    result_csv = result.drop(columns=["geometry"]).copy() if "geometry" in result.columns else result.copy()
    output_csv = os.path.join(base_dir, f"district_data_{city_name.replace(', ', '_').replace(' ', '_')}.csv")
    result_csv.to_csv(output_csv, index=False)
    print(f"\nFull data saved to: {output_csv}")
    
    # Optionally save as GeoJSON
    output_geojson = os.path.join(base_dir, f"district_data_{city_name.replace(', ', '_').replace(' ', '_')}.geojson")
    result.to_file(output_geojson, driver="GeoJSON")
    print(f"Geospatial data saved to: {output_geojson}")
    
    return result


if __name__ == "__main__":
    main()



