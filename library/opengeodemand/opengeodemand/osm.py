import geopandas as gpd
import osmnx as ox
import pandas as pd
from .utils import ensure_crs

class OSMExtractor:
    @staticmethod
    def get_city_boundary(city_name: str) -> gpd.GeoDataFrame:
        """Geocodes a city name to get its boundary."""
        try:
            print(f"Geocoding city: {city_name}")
            boundary = ox.geocoder.geocode_to_gdf(city_name)
            return ensure_crs(boundary, 4326)
        except Exception as e:
            raise ValueError(f"Failed to geocode city '{city_name}': {e}")

    @staticmethod
    def get_buildings(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts buildings within the boundary using OSMnx.
        Implements the multi-tag strategy from your enhanced script.
        """
        city_polygon = boundary.geometry.iloc[0]
        all_buildings = []
        
        # Strategy 1: Polygon Query with multiple tags
        tag_variations = [
            {"building": True},
            {"building": ["yes", "residential", "commercial", "apartments"]}
        ]
        
        print("Querying OSM for buildings...")
        for tags in tag_variations:
            try:
                try:
                    b = ox.features_from_polygon(city_polygon, tags)
                except AttributeError:
                    b = ox.geometries_from_polygon(city_polygon, tags)
                
                if not b.empty:
                    all_buildings.append(b)
            except Exception:
                continue

        if not all_buildings:
            print("No buildings found via polygon query.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Combine
        combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True))
        
        # Deduplicate
        if 'osmid' in combined.columns:
            combined = combined.drop_duplicates(subset=['osmid'])
        else:
            combined = combined.drop_duplicates(subset=['geometry'])

        # Filter for Polygons only
        combined = combined[combined.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        
        # Clip to boundary
        combined = ensure_crs(combined, 4326)
        boundary = ensure_crs(boundary, 4326)
        combined = gpd.clip(combined, boundary)
        
        # Set a clean index
        combined = combined.reset_index(drop=True).reset_index().rename(columns={"index": "building_idx"})
        
        print(f"Extracted {len(combined)} buildings.")
        return combined