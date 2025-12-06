import pandas as pd
import geopandas as gpd

# 1. Load EC dataset (Jammu & Kashmir example)
ec = pd.read_csv("../EC6A_ST01_JAMMU_AND_KASHMIR.csv", low_memory=False)

# 2. Load district shapefile (2011 Census districts)
districts = gpd.read_file("../maps-master/Districts/Census_2011/2011_Dist.shp")
print("Shapefile columns:", districts.columns)

# 3. Convert codes for join
ec["ST"] = ec["ST"].astype(int)
ec["DT"] = ec["DT"].astype(int)

districts["ST"] = districts["ST_CEN_CD"].astype(int)
districts["DT"] = districts["DT_CEN_CD"].astype(int)

# 4. Aggregate EC by district
ec_agg = (
    ec.groupby(["ST", "DT"])
      .agg(
          n_enterprises=("C_HOUSE", "count"),   # total enterprises
          hh_enterprises=("IN_HH", "sum"),      # household enterprises
          total_workers=("Total_worker", "sum") # total workers
      )
      .reset_index()
)

# 5. Merge EC with district shapefile
ec_geo = districts.merge(ec_agg, on=["ST", "DT"], how="left")

# Fill missing values
for col in ["n_enterprises", "hh_enterprises", "total_workers"]:
    ec_geo[col] = ec_geo[col].fillna(0)

# 6. Quick plot (number of enterprises per district)
ec_geo.plot(column="n_enterprises", cmap="OrRd", legend=True, figsize=(12,10))

# 7. Save output as GeoJSON (optional)
ec_geo.to_file("ec_jammu_kashmir.geojson", driver="GeoJSON")
print("Saved joined EC + districts to ec_jammu_kashmir.geojson")
