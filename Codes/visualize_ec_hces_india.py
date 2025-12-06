import os
import sys
from typing import List

import geopandas as gpd
import folium
import matplotlib.pyplot as plt


def read_vector(path: str) -> gpd.GeoDataFrame:
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


def pick_metrics(gdf: gpd.GeoDataFrame, prefix: str) -> List[str]:
    cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    cols = [c for c in cols if c not in ["ST", "DT"] and c.startswith(prefix)]
    return cols


def plot_interactive_layers(gdf: gpd.GeoDataFrame, out_html: str, metrics: List[str]) -> None:
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    m = folium.Map(location=[22.5, 79.0], zoom_start=4, tiles="cartodbpositron")
    # Folium.Choropleth must be added to a Map directly (not a FeatureGroup)
    for col in metrics:
        ch = folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=[gdf.index, col],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.2,
            nan_fill_opacity=0.15,
            legend_name=col,
            name=col,
            show=False,
        )
        ch.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"Saved: {out_html}")


def plot_static_grid(gdf: gpd.GeoDataFrame, metrics: List[str], title: str) -> None:
    import math
    n = len(metrics)
    if n == 0:
        print("No metrics to plot.")
        return
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, col in enumerate(metrics):
        ax = axes[i]
        gdf.plot(column=col, cmap="YlOrRd", legend=True, ax=ax)
        ax.set_title(col)
        ax.set_axis_off()
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "india_district_ec_hces.geojson")
    in_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    mode = sys.argv[2] if len(sys.argv) > 2 else "interactive"  # interactive|static

    if not os.path.exists(in_path):
        print(f"Not found: {in_path}")
        sys.exit(1)

    gdf = read_vector(in_path)

    # Select metric groups
    ec_metrics = pick_metrics(gdf, "ec_")
    # Example: pick some HCES common fields (prefixes based on level slugs)
    # You can pass 'all' to include all numeric columns starting with 'hces_'
    hces_metrics = [c for c in gdf.columns if c.startswith("hces_") and gdf[c].dtype.kind in "if"]

    # Limit number of layers for readability
    ec_metrics = ec_metrics[:8]
    hces_metrics = hces_metrics[:20]

    if mode == "static":
        plot_static_grid(gdf, ec_metrics, "Economic Census metrics")
        plot_static_grid(gdf, hces_metrics, "HCES metrics")
    else:
        out1 = os.path.splitext(in_path)[0] + "_ec_layers.html"
        out2 = os.path.splitext(in_path)[0] + "_hces_layers.html"
        plot_interactive_layers(gdf, out1, ec_metrics)
        plot_interactive_layers(gdf, out2, hces_metrics)


if __name__ == "__main__":
    main()




