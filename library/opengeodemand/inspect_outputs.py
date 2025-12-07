import os
import glob
import pandas as pd
import geopandas as gpd

OUTPUT_DIR = "output_surat_comprehensive"

def check_file(filename, subfolder, is_geo=False):
    full_path = os.path.join(OUTPUT_DIR, subfolder, filename)
    print(f"\n>> FILE: {filename}")
    if not os.path.exists(full_path):
        print(f"   [ERROR] Missing: {full_path}")
        return
    try:
        if is_geo:
            df = gpd.read_file(full_path)
            print(f"   GeoDataFrame | CRS: {df.crs} | Shape: {df.shape}")
        else:
            df = pd.read_csv(full_path)
            print(f"   DataFrame    | Shape: {df.shape}")
            if "order_value" in df.columns:
                print(f"   Total Rev: ₹{df['order_value'].sum():,.0f} | Avg: ₹{df['order_value'].mean():,.2f}")
    except Exception as e:
        print(f"   [ERROR] {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        print("Run simulation first.")
        return
    
    print("1. INFRASTRUCTURE")
    check_file("buildings_enriched.geojson", "data", is_geo=True)
    
    print("\n2. SCENARIOS")
    for csv in sorted(glob.glob(os.path.join(OUTPUT_DIR, "orders", "*.csv"))):
        check_file(os.path.basename(csv), "orders", is_geo=False)

if __name__ == "__main__":
    main()