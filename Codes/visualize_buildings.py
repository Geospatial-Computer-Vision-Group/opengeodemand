"""
Visualize Building Polygons from GeoJSON

This script:
1. Loads building polygons from GeoJSON
2. Creates two visualizations:
   - Sample of 1000 buildings
   - All buildings
3. Generates both static (matplotlib) and interactive (Folium) maps
"""

import os
import sys
from typing import Optional
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd


def read_vector(path: str) -> gpd.GeoDataFrame:
    """Read a vector file with fallback engines."""
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


def create_static_map(
    buildings: gpd.GeoDataFrame,
    title: str,
    output_path: str,
    max_buildings: Optional[int] = None
) -> None:
    """
    Create a static matplotlib map of buildings.
    
    Args:
        buildings: GeoDataFrame with building polygons
        title: Title for the map
        output_path: Path to save the image
        max_buildings: Maximum number of buildings to show (None = all)
    """
    # Sample if needed
    if max_buildings is not None and len(buildings) > max_buildings:
        print(f"Sampling {max_buildings} buildings from {len(buildings)} total...")
        buildings = buildings.sample(n=max_buildings, random_state=42)
    
    # Ensure WGS84
    buildings = buildings.to_crs(epsg=4326)
    
    # Get bounds
    bounds = buildings.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot buildings
    buildings.plot(
        ax=ax,
        color='#2c3e50',
        edgecolor='#34495e',
        linewidth=0.3,
        alpha=0.7
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add text with building count
    num_buildings = len(buildings)
    ax.text(
        0.02, 0.98,
        f'Buildings: {num_buildings:,}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static map saved to: {output_path}")
    plt.close()


def clean_for_folium(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean GeoDataFrame to make it JSON-serializable for Folium.
    Converts Timestamp, datetime, and other non-serializable types to strings.
    
    Args:
        gdf: GeoDataFrame to clean
    
    Returns:
        Cleaned GeoDataFrame
    """
    gdf_clean = gdf.copy()
    
    for col in gdf_clean.columns:
        if col == 'geometry':
            continue
        
        # Check if column contains datetime/timestamp objects
        if pd.api.types.is_datetime64_any_dtype(gdf_clean[col]):
            print(f"  Converting datetime column '{col}' to string...")
            gdf_clean[col] = gdf_clean[col].astype(str)
        else:
            # Check for Timestamp objects in object columns
            sample = gdf_clean[col].dropna()
            if len(sample) > 0:
                first_val = sample.iloc[0]
                if isinstance(first_val, (datetime, pd.Timestamp)):
                    print(f"  Converting timestamp column '{col}' to string...")
                    gdf_clean[col] = gdf_clean[col].astype(str)
                elif isinstance(first_val, (list, dict)):
                    print(f"  Converting complex column '{col}' to string...")
                    gdf_clean[col] = gdf_clean[col].apply(
                        lambda x: str(x) if isinstance(x, (list, dict)) else x
                    )
    
    return gdf_clean


def create_interactive_map(
    buildings: gpd.GeoDataFrame,
    title: str,
    output_path: str,
    max_buildings: Optional[int] = None
) -> None:
    """
    Create an interactive Folium map of buildings.
    
    Args:
        buildings: GeoDataFrame with building polygons
        title: Title for the map
        output_path: Path to save the HTML file
        max_buildings: Maximum number of buildings to show (None = all)
    """
    # Sample if needed
    if max_buildings is not None and len(buildings) > max_buildings:
        print(f"Sampling {max_buildings} buildings from {len(buildings)} total...")
        buildings = buildings.sample(n=max_buildings, random_state=42)
    
    # Ensure WGS84
    buildings = buildings.to_crs(epsg=4326)
    
    # Clean data for JSON serialization (convert Timestamps, etc.)
    print("Cleaning data for Folium visualization...")
    buildings = clean_for_folium(buildings)
    
    # Get bounds and center
    bounds = buildings.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Calculate zoom level based on bounds
    lat_range = bounds[3] - bounds[1]
    lon_range = bounds[2] - bounds[0]
    max_range = max(lat_range, lon_range)
    
    if max_range > 1:
        zoom_start = 8
    elif max_range > 0.5:
        zoom_start = 9
    elif max_range > 0.2:
        zoom_start = 10
    elif max_range > 0.1:
        zoom_start = 11
    elif max_range > 0.05:
        zoom_start = 12
    else:
        zoom_start = 13
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='cartodbpositron'
    )
    
    # Add buildings as GeoJSON layer
    # For large datasets, we'll add them directly
    num_buildings = len(buildings)
    
    # Create a style function for buildings
    def style_function(feature):
        return {
            'fillColor': '#2c3e50',
            'color': '#34495e',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }
    
    # Add buildings layer
    buildings_layer = folium.GeoJson(
        buildings,
        style_function=style_function,
        name=f'Buildings ({num_buildings:,})',
        tooltip=folium.features.GeoJsonTooltip(
            fields=['building'] if 'building' in buildings.columns else [],
            aliases=['Type:'] if 'building' in buildings.columns else [],
            localize=True
        )
    )
    buildings_layer.add_to(m)
    
    # Add title as HTML
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 60px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px">
    <b>{title}</b><br>
    Buildings: {num_buildings:,}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")


def main():
    """Main function to visualize buildings."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_buildings.py <geojson_path> [output_dir]")
        print("Example: python visualize_buildings.py buildings_Surat_Gujarat_India.geojson")
        sys.exit(1)
    
    geojson_path = sys.argv[1]
    if not os.path.isabs(geojson_path):
        geojson_path = os.path.join(base_dir, geojson_path)
    
    if not os.path.exists(geojson_path):
        raise SystemExit(f"GeoJSON file not found: {geojson_path}")
    
    # Output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        # Create output directory based on input filename
        base_name = os.path.splitext(os.path.basename(geojson_path))[0]
        output_dir = os.path.join(base_dir, f"visualization_{base_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Building Polygon Visualization")
    print("="*80)
    print(f"Input: {geojson_path}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    # Load buildings
    print("Loading buildings from GeoJSON...")
    buildings = read_vector(geojson_path)
    print(f"Loaded {len(buildings):,} buildings")
    
    if len(buildings) == 0:
        raise SystemExit("No buildings found in GeoJSON file!")
    
    # Ensure WGS84
    buildings = buildings.to_crs(epsg=4326)
    
    # Create visualizations
    
    # 1. Sample of 1000 buildings
    print("\n" + "-"*80)
    print("Creating visualization: 1000 buildings (sample)")
    print("-"*80)
    
    # Static map - 1000 buildings
    static_1000_path = os.path.join(output_dir, "buildings_1000_static.png")
    create_static_map(
        buildings,
        "Building Polygons (Sample: 1000 buildings)",
        static_1000_path,
        max_buildings=1000
    )
    
    # Interactive map - 1000 buildings
    interactive_1000_path = os.path.join(output_dir, "buildings_1000_interactive.html")
    create_interactive_map(
        buildings,
        "Building Polygons (Sample: 1000 buildings)",
        interactive_1000_path,
        max_buildings=1000
    )
    
    # 2. All buildings
    print("\n" + "-"*80)
    print("Creating visualization: All buildings")
    print("-"*80)
    
    # Static map - all buildings
    static_all_path = os.path.join(output_dir, "buildings_all_static.png")
    create_static_map(
        buildings,
        f"Building Polygons (All {len(buildings):,} buildings)",
        static_all_path,
        max_buildings=None
    )
    
    # Interactive map - all buildings
    interactive_all_path = os.path.join(output_dir, "buildings_all_interactive.html")
    create_interactive_map(
        buildings,
        f"Building Polygons (All {len(buildings):,} buildings)",
        interactive_all_path,
        max_buildings=None
    )
    
    # Summary
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. {os.path.basename(static_1000_path)} - Static map (1000 buildings)")
    print(f"  2. {os.path.basename(interactive_1000_path)} - Interactive map (1000 buildings)")
    print(f"  3. {os.path.basename(static_all_path)} - Static map (all buildings)")
    print(f"  4. {os.path.basename(interactive_all_path)} - Interactive map (all buildings)")
    print("="*80)


if __name__ == "__main__":
    main()

