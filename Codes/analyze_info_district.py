"""
Analysis and Visualization Script for District-Level Economic Census and HCES Data

This script:
1. Loads district data from GeoJSON
2. Gets city boundary using OSMnx
3. Creates interactive and static visualizations
4. Analyzes key economic and consumption indicators
5. Generates summary statistics and insights
"""

import os
import sys
from typing import List, Dict, Optional

import geopandas as gpd
import pandas as pd
import osmnx as ox
import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


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


def get_city_boundary(city_name: str) -> gpd.GeoDataFrame:
    """Get city boundary using OSMnx."""
    try:
        boundary = ox.geocoder.geocode_to_gdf(city_name)
        boundary = boundary.to_crs(epsg=4326)
        return boundary
    except Exception as e:
        raise SystemExit(f"Failed to geocode city '{city_name}': {e}")


def analyze_district_data(gdf: gpd.GeoDataFrame) -> Dict:
    """
    Analyze district data and return summary statistics.
    
    Feature meanings:
    - EC (Economic Census):
      * ec_n_enterprises: Number of establishments/enterprises
      * ec_hh_enterprises: Household-based enterprises
      * Total_worker: Total workers in enterprises
      * M_H: Male household workers
      * M_NH: Male non-household workers
      * F_H: Female household workers
      * F_NH: Female non-household workers
      * BACT: Business activity code (aggregated)
    
    - HCES (Household Consumption Expenditure Survey):
      * hces_lvl01_multiplier: Survey weights for Level 1
      * hces_lvl15_monthly_consumption_exp: Monthly consumption expenditure (Rs)
      * hces_lvl15_online_expenditure: Online spending (Rs)
      * hces_lvl15_household_size: Average household size
    """
    analysis = {}
    
    # District count
    analysis['num_districts'] = len(gdf)
    
    # District names
    if 'DISTRICT' in gdf.columns:
        analysis['districts'] = gdf['DISTRICT'].tolist()
    
    # Economic Census metrics
    ec_metrics = {
        'ec_n_enterprises': 'Total Enterprises',
        'ec_hh_enterprises': 'Household Enterprises',
        'Total_worker': 'Total Workers',
        'M_H': 'Male Household Workers',
        'M_NH': 'Male Non-Household Workers',
        'F_H': 'Female Household Workers',
        'F_NH': 'Female Non-Household Workers',
    }
    
    analysis['ec_summary'] = {}
    for col, label in ec_metrics.items():
        if col in gdf.columns:
            analysis['ec_summary'][label] = {
                'total': float(gdf[col].sum()),
                'mean': float(gdf[col].mean()),
                'max': float(gdf[col].max()),
                'max_district': gdf.loc[gdf[col].idxmax(), 'DISTRICT'] if 'DISTRICT' in gdf.columns else 'N/A'
            }
    
    # HCES metrics
    hces_key_metrics = {
        'hces_lvl01_multiplier': 'Survey Multiplier (Level 1)',
        'hces_lvl15_monthly_consumption_exp': 'Monthly Consumption Expenditure (Rs)',
        'hces_lvl15_online_expenditure': 'Online Expenditure (Rs)',
        'hces_lvl15_household_size': 'Average Household Size',
    }
    
    analysis['hces_summary'] = {}
    for col, label in hces_key_metrics.items():
        if col in gdf.columns:
            analysis['hces_summary'][label] = {
                'total': float(gdf[col].sum()),
                'mean': float(gdf[col].mean()),
                'max': float(gdf[col].max()),
                'max_district': gdf.loc[gdf[col].idxmax(), 'DISTRICT'] if 'DISTRICT' in gdf.columns else 'N/A'
            }
    
    # Calculate derived metrics
    if 'Total_worker' in gdf.columns and 'ec_n_enterprises' in gdf.columns:
        gdf['workers_per_enterprise'] = gdf['Total_worker'] / gdf['ec_n_enterprises'].replace(0, np.nan)
        analysis['workers_per_enterprise'] = {
            'mean': float(gdf['workers_per_enterprise'].mean()),
            'median': float(gdf['workers_per_enterprise'].median())
        }
    
    if 'F_H' in gdf.columns and 'F_NH' in gdf.columns and 'Total_worker' in gdf.columns:
        gdf['female_worker_share'] = (gdf['F_H'] + gdf['F_NH']) / gdf['Total_worker'].replace(0, np.nan) * 100
        analysis['female_worker_share'] = {
            'mean': float(gdf['female_worker_share'].mean()),
            'median': float(gdf['female_worker_share'].median())
        }
    
    if 'hces_lvl15_online_expenditure' in gdf.columns and 'hces_lvl15_monthly_consumption_exp' in gdf.columns:
        gdf['online_expenditure_share'] = (
            gdf['hces_lvl15_online_expenditure'] / 
            gdf['hces_lvl15_monthly_consumption_exp'].replace(0, np.nan) * 100
        )
        analysis['online_expenditure_share'] = {
            'mean': float(gdf['online_expenditure_share'].mean()),
            'median': float(gdf['online_expenditure_share'].median())
        }
    
    return analysis, gdf


def create_interactive_map(
    districts: gpd.GeoDataFrame,
    city_boundary: gpd.GeoDataFrame,
    city_name: str,
    output_path: str
) -> None:
    """Create an interactive Folium map with multiple metric layers."""
    # Ensure WGS84
    districts = districts.to_crs(epsg=4326)
    city_boundary = city_boundary.to_crs(epsg=4326)
    
    # Center map on city
    center = city_boundary.geometry.iloc[0].centroid
    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=10,
        tiles="cartodbpositron"
    )
    
    # Add city boundary
    folium.GeoJson(
        city_boundary,
        style_function=lambda x: {
            "color": "red",
            "weight": 3,
            "fillOpacity": 0.1,
            "fillColor": "red"
        },
        name="City Boundary",
        tooltip=folium.features.GeoJsonTooltip(fields=[], aliases=[])
    ).add_to(m)
    
    # Key metrics to visualize
    metrics = [
        ('ec_n_enterprises', 'Total Enterprises', 'YlOrRd'),
        ('Total_worker', 'Total Workers', 'Blues'),
        ('ec_hh_enterprises', 'Household Enterprises', 'Greens'),
    ]
    
    if 'hces_lvl15_monthly_consumption_exp' in districts.columns:
        metrics.append(('hces_lvl15_monthly_consumption_exp', 'Monthly Consumption (Rs)', 'Purples'))
    
    # Add choropleth layers for each metric
    for col, name, colormap in metrics:
        if col in districts.columns:
            # Create choropleth
            ch = folium.Choropleth(
                geo_data=districts.__geo_interface__,
                data=districts,
                columns=[districts.index, col],
                key_on="feature.id",
                fill_color=colormap,
                fill_opacity=0.7,
                line_opacity=0.3,
                nan_fill_opacity=0.1,
                legend_name=name,
                name=name,
                show=False
            )
            ch.add_to(m)
            
            # Add tooltips
            folium.GeoJson(
                districts,
                style_function=lambda x: {
                    "color": "#555",
                    "weight": 1,
                    "fillOpacity": 0
                },
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['DISTRICT', col] if 'DISTRICT' in districts.columns else [col],
                    aliases=['District', name],
                    localize=True
                )
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")


def create_static_analysis_plots(
    districts: gpd.GeoDataFrame,
    city_boundary: gpd.GeoDataFrame,
    analysis: Dict,
    output_dir: str,
    city_name: str
) -> None:
    """Create static matplotlib plots for analysis."""
    
    districts = districts.to_crs(epsg=4326)
    city_boundary = city_boundary.to_crs(epsg=4326)
    
    # Figure 1: Map with multiple metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('ec_n_enterprises', 'Total Enterprises', 0),
        ('Total_worker', 'Total Workers', 1),
        ('ec_hh_enterprises', 'Household Enterprises', 2),
    ]
    
    if 'hces_lvl15_monthly_consumption_exp' in districts.columns:
        metrics_to_plot.append(('hces_lvl15_monthly_consumption_exp', 'Monthly Consumption (Rs)', 3))
    
    for col, title, idx in metrics_to_plot:
        if col in districts.columns and idx < len(axes):
            ax = axes[idx]
            districts.plot(column=col, cmap='YlOrRd', legend=True, ax=ax, edgecolor='black', linewidth=0.5)
            city_boundary.boundary.plot(ax=ax, color='red', linewidth=2, label='City Boundary')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_axis_off()
            if idx == 0:
                ax.legend(loc='upper right')
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'District Analysis: {city_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{city_name.replace(" ", "_")}_district_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Bar charts for key metrics
    if 'DISTRICT' in districts.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        metrics_for_bars = [
            ('ec_n_enterprises', 'Total Enterprises', 0),
            ('Total_worker', 'Total Workers', 1),
            ('ec_hh_enterprises', 'Household Enterprises', 2),
        ]
        
        if 'hces_lvl15_monthly_consumption_exp' in districts.columns:
            metrics_for_bars.append(('hces_lvl15_monthly_consumption_exp', 'Monthly Consumption (Rs)', 3))
        
        for col, title, idx in metrics_for_bars:
            if col in districts.columns and idx < len(axes):
                ax = axes[idx]
                data = districts.sort_values(col, ascending=True)
                bars = ax.barh(range(len(data)), data[col], color='steelblue')
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data['DISTRICT'], fontsize=9)
                ax.set_xlabel(title, fontsize=11)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, data[col])):
                    ax.text(val, i, f' {val:,.0f}', va='center', fontsize=8)
        
        plt.suptitle(f'District Comparison: {city_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{city_name.replace(" ", "_")}_district_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 3: Summary statistics
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary_text = f"District Analysis Summary: {city_name}\n"
    summary_text += "=" * 80 + "\n\n"
    
    summary_text += f"Number of Districts: {analysis['num_districts']}\n"
    if 'districts' in analysis:
        summary_text += f"Districts: {', '.join(analysis['districts'])}\n\n"
    
    summary_text += "Economic Census Metrics:\n"
    summary_text += "-" * 40 + "\n"
    for metric, data in analysis.get('ec_summary', {}).items():
        summary_text += f"{metric}:\n"
        summary_text += f"  Total: {data['total']:,.0f}\n"
        summary_text += f"  Mean per District: {data['mean']:,.0f}\n"
        summary_text += f"  Maximum: {data['max']:,.0f} ({data['max_district']})\n\n"
    
    summary_text += "HCES Metrics:\n"
    summary_text += "-" * 40 + "\n"
    for metric, data in analysis.get('hces_summary', {}).items():
        summary_text += f"{metric}:\n"
        summary_text += f"  Total: {data['total']:,.0f}\n"
        summary_text += f"  Mean per District: {data['mean']:,.0f}\n"
        summary_text += f"  Maximum: {data['max']:,.0f} ({data['max_district']})\n\n"
    
    if 'workers_per_enterprise' in analysis:
        summary_text += f"Workers per Enterprise (mean): {analysis['workers_per_enterprise']['mean']:.2f}\n"
    
    if 'female_worker_share' in analysis:
        summary_text += f"Female Worker Share (mean): {analysis['female_worker_share']['mean']:.2f}%\n"
    
    if 'online_expenditure_share' in analysis:
        summary_text += f"Online Expenditure Share (mean): {analysis['online_expenditure_share']['mean']:.2f}%\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, f'{city_name.replace(" ", "_")}_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the analysis."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python analyze_info_district.py <city_name> <geojson_path>")
        print("Example: python analyze_info_district.py 'Surat, Gujarat, India' district_data_Surat_Gujarat_India.geojson")
        sys.exit(1)
    
    city_name = sys.argv[1]
    geojson_path = sys.argv[2]
    
    if not os.path.exists(geojson_path):
        # Try relative to base_dir
        geojson_path = os.path.join(base_dir, geojson_path)
        if not os.path.exists(geojson_path):
            raise SystemExit(f"GeoJSON file not found: {geojson_path}")
    
    print(f"Loading district data from: {geojson_path}")
    districts = read_vector(geojson_path)
    
    print(f"Getting city boundary for: {city_name}")
    city_boundary = get_city_boundary(city_name)
    
    print("Analyzing district data...")
    analysis, districts_enhanced = analyze_district_data(districts.copy())
    
    # Print summary to console
    print("\n" + "="*80)
    print(f"ANALYSIS SUMMARY: {city_name}")
    print("="*80)
    print(f"\nNumber of Districts: {analysis['num_districts']}")
    if 'districts' in analysis:
        print(f"Districts: {', '.join(analysis['districts'])}")
    
    print("\nEconomic Census Summary:")
    for metric, data in analysis.get('ec_summary', {}).items():
        print(f"  {metric}: Total={data['total']:,.0f}, Mean={data['mean']:,.0f}, Max={data['max']:,.0f} ({data['max_district']})")
    
    print("\nHCES Summary:")
    for metric, data in analysis.get('hces_summary', {}).items():
        print(f"  {metric}: Total={data['total']:,.0f}, Mean={data['mean']:,.0f}, Max={data['max']:,.0f} ({data['max_district']})")
    
    # Create output directory
    output_dir = os.path.join(base_dir, f"analysis_{city_name.replace(', ', '_').replace(' ', '_')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Interactive map
    interactive_map_path = os.path.join(output_dir, "interactive_map.html")
    create_interactive_map(districts_enhanced, city_boundary, city_name, interactive_map_path)
    
    # Static plots
    create_static_analysis_plots(districts_enhanced, city_boundary, analysis, output_dir, city_name)
    
    # Save enhanced data
    output_geojson = os.path.join(output_dir, "districts_with_analysis.geojson")
    districts_enhanced.to_file(output_geojson, driver="GeoJSON")
    print(f"Enhanced data saved to: {output_geojson}")
    
    # Save summary as CSV
    summary_df = districts_enhanced.drop(columns=['geometry']).copy() if 'geometry' in districts_enhanced.columns else districts_enhanced.copy()
    summary_csv = os.path.join(output_dir, "district_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()



