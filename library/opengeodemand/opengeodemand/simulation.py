import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import random
from .utils import sample_point_in_polygon
import warnings

class DemandSimulator:
    def __init__(self, enriched_buildings: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, config: dict):
        self.bgeo = enriched_buildings.copy()
        self.districts = districts.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.districts['centroid'] = self.districts.geometry.centroid
        
        # Base Defaults
        self.config = {
            "start_date": datetime.now(),
            "num_days": 7,
            "weights": {"poisson": 0.35, "gravity": 0.30, "agent": 0.25, "self_exciting": 0.10},
            "min_order_value": 50,
            "safe_lambda_cap": 100.0,
            "base_rate": 20,
            "order_value_mean": 250,
            "order_value_std": 100,
            "max_orders_per_bldg": 50,
            "tod_probs": None,
            "wealth_sensitivity": 1.0, 
            "price_sensitivity": 1.0,
            "weekend_multiplier": 1.0  # Default to no change
        }
        
        if config:
            self.config.update(config)

        # Time of Day
        self.hours = np.arange(24)
        if self.config["tod_probs"] is not None and len(self.config["tod_probs"]) == 24:
            self.tod_probs = np.array(self.config["tod_probs"])
        else:
            self.tod_probs = np.array([0.04]*24)
        if self.tod_probs.sum() > 0: self.tod_probs = self.tod_probs / self.tod_probs.sum()

    def _calculate_gravity_weights(self):
        centroid_map = self.districts.set_index("district_id_join")['centroid']
        b_dist_centroids = self.bgeo['district_id_join'].map(centroid_map)
        mask_nan = b_dist_centroids.isna()
        if mask_nan.any():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                b_dist_centroids.loc[mask_nan] = self.bgeo.loc[mask_nan].geometry.centroid
        b_dist_centroids_geo = gpd.GeoSeries(b_dist_centroids, index=self.bgeo.index, crs=self.bgeo.crs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            dists = self.bgeo.geometry.centroid.distance(b_dist_centroids_geo)
        dists = dists.replace(0, 1e-6)
        
        wealth_factor = np.power(self.bgeo["district_consumption_proxy_norm"], self.config["wealth_sensitivity"])
        mass = wealth_factor * self.bgeo["est_households"]
        raw_gravity = mass / dists
        mean_grav = raw_gravity.mean()
        return raw_gravity / mean_grav if mean_grav > 0 else raw_gravity

    def run(self):
        all_orders = []
        print(f"  Simulating: {self.config.get('category_name', 'Custom')}")
        print(f"  Params: Base={self.config['base_rate']}, Wkend={self.config['weekend_multiplier']}x")

        # 1. BASE LAMBDA CALCULATION (Per Building)
        mean_hh = max(1.0, self.bgeo["est_households"].mean())
        hh_factor = self.bgeo["est_households"] / mean_hh
        econ_factor = np.power(self.bgeo["district_consumption_proxy_norm"], self.config["wealth_sensitivity"])
        
        rate_poisson = self.config["base_rate"] * econ_factor * hh_factor
        rate_agent = (0.02 * self.bgeo["est_households"]) * econ_factor
        rate_gravity = self.config["base_rate"] * self._calculate_gravity_weights()
        rate_se = 0.5 * econ_factor * hh_factor

        w = self.config["weights"]
        base_lambda = (
            (rate_poisson * w["poisson"]) +
            (rate_gravity * w["gravity"]) +
            (rate_agent * w["agent"]) +
            (rate_se * w["self_exciting"])
        )
        
        # 2. SIMULATE DAYS
        total_generated = 0
        for i in range(self.config['num_days']):
            curr_date = self.config['start_date'] + timedelta(days=i)
            
            # --- APPLY TEMPORAL MULTIPLIER ---
            is_weekend = curr_date.weekday() >= 5 # 5=Sat, 6=Sun
            day_multiplier = self.config["weekend_multiplier"] if is_weekend else 1.0
            
            # Apply multiplier to the base lambda for today
            daily_lambda = base_lambda * day_multiplier
            daily_lambda = daily_lambda.clip(upper=self.config["safe_lambda_cap"])
            
            # Draw Counts
            daily_counts = np.random.poisson(daily_lambda.values)
            daily_counts = np.minimum(daily_counts, self.config["max_orders_per_bldg"])
            active_indices = np.where(daily_counts > 0)[0]
            
            if len(active_indices) == 0: continue

            day_orders = []
            for idx in active_indices:
                count = daily_counts[idx]
                row = self.bgeo.iloc[idx]
                poly = row.geometry
                
                # Calculate Price Mean with Sensitivity
                price_factor = np.power(row["district_consumption_proxy_norm"], self.config["price_sensitivity"])
                local_mean_price = self.config["order_value_mean"] * price_factor
                
                for _ in range(count):
                    h = np.random.choice(self.hours, p=self.tod_probs)
                    ts = datetime(curr_date.year, curr_date.month, curr_date.day, 
                                  int(h), random.randint(0,59), random.randint(0,59))
                    
                    loc = sample_point_in_polygon(poly, max_tries=5)
                    lat, lon = (loc.y, loc.x) if loc else (None, None)
                    
                    val = np.random.normal(local_mean_price, self.config["order_value_std"])
                    val = max(self.config["min_order_value"], val)
                    
                    day_orders.append({
                        "building_idx": self.bgeo.index[idx],
                        "timestamp": ts,
                        "order_value": round(val, 2),
                        "lat": lat,
                        "lon": lon,
                        "method": "combined"
                    })
            
            all_orders.extend(day_orders)
            total_generated += len(day_orders)
            if total_generated > 5_000_000: break

        print(f"  Generated {len(all_orders)} orders.")
        return pd.DataFrame(all_orders)