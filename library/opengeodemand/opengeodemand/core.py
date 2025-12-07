import os
import geopandas as gpd
import pandas as pd
import numpy as np
import re
from .osm import OSMExtractor
from .districts import DistrictLoader
from .simulation import DemandSimulator
from .utils import clean_geodataframe
from .profiles import DemandProfile

class GeoDemandModel:
    def __init__(self, city_name: str, master_district_path: str = None):
        self.city_name = city_name
        
        # --- AUTO-LOCATE BUNDLED DATA ---
        if master_district_path is None:
            # This looks inside opengeodemand/data/ relative to this script
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            master_district_path = os.path.join(current_file_dir, "data", "india_district_ec_hces.geojson")
            
            if not os.path.exists(master_district_path):
                raise FileNotFoundError(f"Bundled data file not found at: {master_district_path}")
            print(f"Using bundled data source.")
            
        self.district_loader = DistrictLoader(master_district_path)
        
        self.boundary = None
        self.buildings = None
        self.districts = None
        self.enriched_data = None

    def load_data(self):
        self.boundary = OSMExtractor.get_city_boundary(self.city_name)
        self.buildings = OSMExtractor.get_buildings(self.boundary)
        self.districts = self.district_loader.get_districts_for_city(self.boundary)
        self.districts = self.districts.reset_index(drop=True).reset_index().rename(columns={"index": "district_id_join"})

    def sample_buildings(self, n):
        if self.buildings is not None and len(self.buildings) > n:
            print(f"Sampling: Reducing {len(self.buildings)} buildings to {n}...")
            self.buildings = self.buildings.sample(n=n, random_state=42).copy()

    def enrich_data(self):
        if self.buildings is None or self.districts is None:
            raise RuntimeError("Data not loaded.")

        print("Enriching building data...")
        joined = gpd.sjoin(self.buildings, self.districts, how="left", predicate="within", rsuffix="_dist")
        
        if joined.index.duplicated().any():
            joined = joined[~joined.index.duplicated(keep='first')]

        missing = joined[joined["district_id_join"].isna()]
        if not missing.empty:
            print(f"  Fixing {len(missing)} buildings using intersects...")
            fixed = gpd.sjoin(self.buildings.loc[missing.index], self.districts, how="left", predicate="intersects", rsuffix="_dist")
            if fixed.index.duplicated().any():
                fixed = fixed[~fixed.index.duplicated(keep='first')]
            joined.update(fixed)

        # 1. Estimate Households
        def est_hh(row):
            levels = 1.0
            for col in row.index:
                if "levels" in str(col) and pd.notnull(row[col]):
                    try:
                        levels = float(re.search(r'(\d+)', str(row[col])).group(0))
                        break
                    except: pass
            return max(1, int(levels))

        joined["est_households"] = joined.apply(est_hh, axis=1)
        
        # 2. Economic Proxy (FIXED LOGIC)
        priority_cols = [
            "hces_lvl15_multiplier", 
            "hces_lvl15_monthly_consumption_exp",
            "district_hces_lvl02_multiplier",
            "district_avg_daily_expenditure", 
            "district_consumption_per_day"
        ]
        
        # Calculate raw proxy
        raw_proxies = []
        for _, row in joined.iterrows():
            val = np.nan
            for c in priority_cols:
                if c in row and pd.notnull(row[c]): 
                    val = row[c]
                    break
                if f"district_{c}" in row and pd.notnull(row[f"district_{c}"]): 
                    val = row[f"district_{c}"]
                    break
            raw_proxies.append(val)
            
        joined["district_consumption_proxy"] = raw_proxies
        
        # --- FIX: Fill NaNs with the MEAN of the found proxies, NOT 1.0 ---
        # This ensures fallback buildings don't get 0 orders when others have 700M value
        mean_val = joined["district_consumption_proxy"].mean()
        if pd.isna(mean_val) or mean_val == 0: 
            mean_val = 1.0
            
        joined["district_consumption_proxy"] = joined["district_consumption_proxy"].fillna(mean_val)
        
        # Normalize around 1.0
        joined["district_consumption_proxy_norm"] = joined["district_consumption_proxy"] / mean_val
        
        self.enriched_data = joined
        print(f"Enrichment complete. Mean Proxy: {mean_val:.2f}")

    def simulate(self, days=7, category="food", custom_params=None):
        """
        Runs the demand simulation.
        
        Args:
            days (int): Number of days to simulate
            category (str): 'food', 'grocery', 'pharma', 'electronics', 'fashion'
            custom_params (dict): Override specific values (e.g. {'max_orders_per_bldg': 5})
        """
        if self.enriched_data is None:
            self.enrich_data()
            
        # 1. Get Profile Config
        config = DemandProfile.get_config(category, custom_params)
        
        # 2. Add Run-specifics
        config["num_days"] = days
        config["category_name"] = category
        
        sim = DemandSimulator(self.enriched_data, self.districts, config)
        return sim.run()

    def save_buildings(self, path):
        if self.enriched_data is not None:
            clean = clean_geodataframe(self.enriched_data)
            clean.to_file(path, driver="GeoJSON")