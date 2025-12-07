import geopandas as gpd
import os
from .utils import ensure_crs

class DistrictLoader:
    def __init__(self, master_file_path: str):
        if not os.path.exists(master_file_path):
            raise FileNotFoundError(f"Master District file not found: {master_file_path}")
        self.master_path = master_file_path

    def get_districts_for_city(self, city_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Loads the master file and filters for districts intersecting the city.
        """
        print(f"Loading master district data from {self.master_path}...")
        try:
            # pyogrio is faster for large files
            gdf = gpd.read_file(self.master_path, engine="pyogrio")
        except:
            gdf = gpd.read_file(self.master_path)
            
        gdf = ensure_crs(gdf, 4326)
        city_boundary = ensure_crs(city_boundary, 4326)
        
        # Spatial Filter
        city_geom = city_boundary.geometry.iloc[0]
        intersecting = gdf[gdf.geometry.intersects(city_geom)].copy()
        
        if intersecting.empty:
            print("Warning: No districts found intersecting this city.")
            
        return intersecting