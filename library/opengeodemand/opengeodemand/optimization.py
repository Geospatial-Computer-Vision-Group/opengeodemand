import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2

class StoreOptimizer:
    def __init__(self, orders_df: pd.DataFrame):
        self.orders = orders_df.copy()
        # Ensure we have clean data
        self.orders = self.orders.dropna(subset=['lat', 'lon', 'order_value'])

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calc distance in KM between two points."""
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def optimize_locations(self, n_stores: int, min_spacing_km: float = 2.0):
        """
        Finds optimal store locations using Revenue-Weighted K-Means.
        Then iteratively merges stores that are too close.
        """
        print(f"  Optimizing for {n_stores} stores (Min Spacing: {min_spacing_km}km)...")
        
        # 1. Weighted K-Means
        # We weigh the clustering by 'order_value' so stores gravitate towards MONEY, not just density.
        X = self.orders[['lat', 'lon']].values
        weights = self.orders['order_value'].values
        
        kmeans = KMeans(n_clusters=n_stores, random_state=42, n_init=10)
        kmeans.fit(X, sample_weight=weights)
        
        centers = kmeans.cluster_centers_
        
        # 2. Check Constraints & Merge
        # Simple greedy merge: if two stores are too close, merge them
        final_stores = []
        
        # Convert to list for manipulation
        candidates = list(centers) # [[lat, lon], ...]
        
        while len(candidates) > 0:
            current = candidates.pop(0)
            merged = [current]
            
            # Find all neighbors too close to 'current'
            kept_candidates = []
            for other in candidates:
                dist = self._haversine(current[0], current[1], other[0], other[1])
                if dist < min_spacing_km:
                    merged.append(other) # Too close, mark for merge
                else:
                    kept_candidates.append(other)
            
            # Average the location of the merged group
            # (In reality you'd weight this by cluster revenue, but simple avg is fine for MVP)
            merged_arr = np.array(merged)
            avg_lat = merged_arr[:,0].mean()
            avg_lon = merged_arr[:,1].mean()
            
            final_stores.append([avg_lat, avg_lon])
            candidates = kept_candidates # Loop with remaining
            
        # 3. Calculate Potential Revenue per Store
        # Assign every order to nearest Final Store
        store_stats = []
        final_stores_arr = np.array(final_stores)
        
        # Simple nearest assignment
        # Note: This is computationally expensive O(N*M), but fine for <1M orders
        from scipy.spatial.distance import cdist
        dists = cdist(X, final_stores_arr)
        labels = np.argmin(dists, axis=1)
        
        self.orders['assigned_store'] = labels
        
        for i, (lat, lon) in enumerate(final_stores):
            cluster_orders = self.orders[self.orders['assigned_store'] == i]
            rev = cluster_orders['order_value'].sum()
            count = len(cluster_orders)
            store_stats.append({
                "store_id": i,
                "lat": lat,
                "lon": lon,
                "projected_revenue": rev,
                "projected_orders": count
            })
            
        print(f"  Optimization complete. Reduced {n_stores} -> {len(final_stores)} stores based on proximity.")
        return pd.DataFrame(store_stats)