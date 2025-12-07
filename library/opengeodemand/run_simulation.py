import os
import sys
import pandas as pd
import time
from opengeodemand import GeoDemandModel, StoreOptimizer, DemandAnalyzer
from opengeodemand.profiles import DemandProfile

# --- CONFIGURATION ---
CITY_NAME = "Surat, Gujarat, India"
SAMPLE_SIZE = 2000   
OUTPUT_DIR = "output_surat_comprehensive"

def main():
    start_global = time.time()
    
    # 1. SETUP FOLDERS
    paths = {k: os.path.join(OUTPUT_DIR, k) for k in ["data", "orders", "optimization", "analysis"]}
    for p in paths.values(): os.makedirs(p, exist_ok=True)
    
    # NOTE: We rely on auto-loading the bundled data now!
    print(f"\n=== 1. INITIALIZING & LOADING DATA FOR {CITY_NAME} ===")
    model = GeoDemandModel(CITY_NAME) # No file path needed, it's inside the library now!
    model.load_data()
    
    # SAVE RAW DISTRICTS
    model.districts.to_file(os.path.join(paths["data"], "districts.geojson"), driver="GeoJSON")
    
    # 2. SAMPLE & ENRICH
    print(f"\n=== 2. SAMPLING ({SAMPLE_SIZE}) & ENRICHING ===")
    model.sample_buildings(SAMPLE_SIZE)
    model.enrich_data()
    model.save_buildings(os.path.join(paths["data"], "buildings_enriched.geojson"))

    # ---------------------------------------------------------
    # SCENARIOS
    # ---------------------------------------------------------
    scenarios = [
        {
            "name": "01_Food_Standard",
            "category": "food",
            "desc": "Default food behavior (Lunch/Dinner peaks, slight weekend boost)",
            "params": {}
        },
        {
            "name": "02_Grocery_WeekendRush",
            "category": "grocery",
            "desc": "Grocery stores with massive weekend spikes (2.5x)",
            "params": {"weekend_multiplier": 2.5, "base_rate": 10}
        },
        {
            "name": "03_Electronics_Luxury",
            "category": "electronics",
            "desc": "High Wealth Sensitivity (3.0) - Only rich districts buy",
            "params": {"base_rate": 1.0, "wealth_sensitivity": 3.0, "max_orders_per_bldg": 5}
        },
        {
            "name": "04_Fashion_FlashSale",
            "category": "fashion",
            "desc": "High Price Sensitivity - Rich districts pay much more",
            "params": {"base_rate": 15, "price_sensitivity": 1.5}
        },
        {
            "name": "05_Office_Supplies_B2B",
            "category": "custom",
            "desc": "Daytime only (9-5), Closed on Weekends (0.05x)",
            "params": {"base_rate": 8, "weekend_multiplier": 0.05, "time_curve": "daytime", "order_value_mean": 2000}
        },
        {
            "name": "06_NightLife_Parties",
            "category": "food",
            "desc": "Late night orders only (8pm-2am), High Weekend Multiplier",
            "params": {"base_rate": 5, "time_curve": "night", "weekend_multiplier": 2.0, "order_value_mean": 600}
        }
    ]

    print(f"\n=== 3. RUNNING {len(scenarios)} SCENARIOS ===")
    
    for scen in scenarios:
        print(f"\n>> RUNNING: {scen['name']}")
        orders = model.simulate(days=14, category=scen['category'], custom_params=scen['params'])
        
        filename = f"{scen['name']}.csv"
        orders.to_csv(os.path.join(paths["orders"], filename), index=False)
        print(f"   Generated: {len(orders)} orders")

    # 4. DEMO OPTIMIZATION (Using Scenario 1 Data)
    print(f"\n=== 4. OPTIMIZING STORES (For Scenario 1) ===")
    # Reload scenario 1 data for optimization demo
    orders_food = pd.read_csv(os.path.join(paths["orders"], "01_Food_Standard.csv"))
    
    optimizer = StoreOptimizer(orders_food)
    stores = optimizer.optimize_locations(n_stores=10, min_spacing_km=2.0)
    stores.to_csv(os.path.join(paths["optimization"], "optimal_stores_food.csv"), index=False)
    
    # 5. DEMO ANALYSIS
    print(f"\n=== 5. GENERATING DASHBOARD ===")
    analyzer = DemandAnalyzer(orders_food, OUTPUT_DIR, city_name=CITY_NAME)
    analyzer.generate_report(stores_df=stores)

    print("\nDONE. Everything restored.")

if __name__ == "__main__":
    main()