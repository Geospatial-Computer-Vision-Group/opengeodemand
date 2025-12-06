
"""
generate_orders.py  (fixed)

Fixes:
 - do not drop geometry before spatial join (gpd.sjoin requires GeoDataFrame on both sides)
 - fallback to 'intersects' for buildings missed by 'within'
 - prefix district columns to avoid collisions
 - preserve building index as building_idx
 - ensure simulation is for 7 days (NUM_DAYS = 7)

Usage:
  python generate_orders.py
"""

import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Config
# --------------------------

BUILDINGS_GEOJSON = Path("buildings_Surat_Gujarat_India.geojson")
DISTRICTS_GEOJSON = Path("district_data_Surat_Gujarat_India.geojson")

OUTPUT_ORDERS_CSV = Path("/orders/synthetic_orders.csv")
OUTPUT_ORDERS_GEOJSON = Path("/orders/synthetic_orders.geojson")


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Simulation period
START_DATE = datetime(2023, 11, 1)   # start date for simulation (changeable)
NUM_DAYS = 7                         # simulate for one week

# Method weights when combining counts (sum to 1)
WEIGHTS = {
    "poisson": 0.35,
    "gravity": 0.30,
    "agent": 0.25,
    "self_exciting": 0.10
}

# Basic parameters (tunable)
BASE_ORDER_RATE_PER_DISTRICT_PER_DAY = 200   # baseline demand per district per day (will scale by district features)
ORDER_VALUE_MEAN = 250                       # mean order value INR
ORDER_VALUE_STD = 100                        # std dev
MAX_AGENTS_PER_BUILDING = 200                # for agent-based sampling upper bound

# --------------------------
# Utility helpers
# --------------------------
def sample_point_in_polygon(poly, max_tries=200):
    if poly is None or poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        pt = Point(x, y)
        if poly.contains(pt):
            return pt
    return poly.representative_point()

# --------------------------
# Load and merge datasets
# --------------------------
print("Loading datasets...")
g_buildings = gpd.read_file(str(BUILDINGS_GEOJSON))
g_districts = gpd.read_file(str(DISTRICTS_GEOJSON))

# ensure both in EPSG:4326
g_buildings = g_buildings.to_crs(epsg=4326)
g_districts = g_districts.to_crs(epsg=4326)

# preserve original building index as building_idx
g_buildings = g_buildings.reset_index(drop=False).rename(columns={"index":"building_idx"}).set_index("building_idx", drop=False)

# ensure district id column for join
g_districts = g_districts.reset_index(drop=False).rename(columns={"index":"district_id_join"}).set_index("district_id_join", drop=False)

print("Performing spatial join (building -> district) using 'within'...")
# do sjoin with districts (do NOT drop geometry)
joined = gpd.sjoin(g_buildings, g_districts, how="left", predicate="within", lsuffix="_b", rsuffix="_d")

# fallback: buildings not assigned by 'within' -> try intersects
missing_mask = joined['district_id_join'].isna()
if missing_mask.any():
    missing_count = missing_mask.sum()
    print(f"  {missing_count} buildings not assigned by 'within' - attempting 'intersects' for those...")
    # sjoin on only missing buildings
    missing_buildings = g_buildings.loc[missing_mask.index[missing_mask]]
    sjoin2 = gpd.sjoin(missing_buildings, g_districts, how="left", predicate="intersects", lsuffix="_b", rsuffix="_d")
    # update joined rows where district_id_join was missing
    for idx, row in sjoin2.iterrows():
        joined.loc[idx, sjoin2.columns] = row

# At this point 'joined' contains building columns + district columns (some may be NaN)
# To avoid column name collisions, prefix district columns with 'district_'
district_cols = [c for c in g_districts.columns if c != "geometry" and c != "district_id_join"]
for c in district_cols:
    if c in joined.columns:
        joined = joined.rename(columns={c: f"district_{c}"})

# For district geometry keep only geometry from buildings (we don't need district geometries per-building)
# The joined GeoDataFrame currently uses geometry from the left (buildings) by default.

bgeo = joined.copy()
print("Buildings merged with district attributes. Building count:", len(bgeo))

# --------------------------
# Compute derived building properties
# --------------------------
def est_households_from_props(row):
    # try common tags; fallback = 1
    levels = None
    for key in ["building:levels", "building_levels", "building:levels:underground", "building:levels_total", "buildinglevels"]:
        if key in row and row[key] not in (None, ""):
            try:
                levels = float(row[key])
                break
            except:
                pass
    if levels is None:
        levels = 1.0
    try:
        # compute approximate planar area by projecting temporarily to metric CRS for area estimate
        geom = row.geometry
        if geom is None or geom.is_empty:
            area_m2 = 0.0
        else:
            # project to EPSG:3857 for area approximation
            area_m2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=3857).geometry.area.iloc[0]
    except Exception:
        area_m2 = 0.0
    # heuristic: one household per 35 m2 per floor
    hh = max(1, int(round(levels * max(1.0, area_m2) / 35.0)))
    return hh

print("Computing estimated households (this may take a bit)...")
bgeo["est_households"] = bgeo.apply(est_households_from_props, axis=1)
import numpy as np
import pandas as pd

# Build district consumption proxy - look for likely district_* columns and use first numeric
def safe_float(x):
    """Convert to float cleanly; return np.nan if fails."""
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return np.nan
        return v
    except:
        return np.nan

# 1. Find candidate district fields
district_candidates = [
    c for c in bgeo.columns
    if str(c).startswith("district_")
]

# 2. Try to construct district consumption proxy
def build_consumption_proxy(row):
    preferred = [
        "district_hces_lvl02_multiplier",
        "district_avg_daily_expenditure",
        "district_consumption_per_day",
        "district_avg_expenditure",
        "district_BACT",
        "district_DT"
    ]

    # try preferred fields first
    for key in preferred:
        if key in row:
            val = safe_float(row[key])
            if val is not np.nan:
                return val

    # fallback: scan every numeric-looking district column
    for key in district_candidates:
        val = safe_float(row[key])
        if val is not np.nan:
            return val

    return 1.0  # final fallback

print("Building district consumption proxy...")
bgeo["district_consumption_proxy"] = bgeo.apply(build_consumption_proxy, axis=1)

# 3. Normalize & clean values
vals = bgeo["district_consumption_proxy"].astype("float64")
meanv = np.nanmean(vals)

if np.isnan(meanv) or meanv <= 0:
    meanv = 1.0

bgeo["district_consumption_proxy_norm"] = vals / meanv

# Replace NaNs, negatives, inf
bgeo["district_consumption_proxy_norm"] = (
    bgeo["district_consumption_proxy_norm"]
    .fillna(1.0)
    .clip(lower=0.01)   # never 0
)

# Ensure households >= 1
bgeo["est_households"] = (
    bgeo["est_households"]
    .astype("float64")
    .fillna(1)
    .clip(lower=1)
)

# FINAL SANITY CHECK
print("=== Sanity Check ===")
print("NaN in district_consumption_proxy_norm:", bgeo["district_consumption_proxy_norm"].isna().sum())
print("Min district_consumption_proxy_norm:", bgeo["district_consumption_proxy_norm"].min())
print("Max district_consumption_proxy_norm:", bgeo["district_consumption_proxy_norm"].max())
print("Min households:", bgeo["est_households"].min())
print("====================")

# --------------------------
# Demand generation functions (same approach as previous script)
# --------------------------
def time_of_day_profile(dt):
    h = dt.hour
    if 7 <= h < 10:
        return 0.9
    if 11 <= h < 14:
        return 1.2
    if 14 <= h < 17:
        return 0.7
    if 18 <= h < 22:
        return 1.6
    if 6 <= h < 7:
        return 0.5
    return 0.3

def method_poisson_nhpp(buildings_df, date):
    df = buildings_df
    mean_hh = max(1.0, df["est_households"].mean())
    base = BASE_ORDER_RATE_PER_DISTRICT_PER_DAY
    results = []
    for idx, row in df.iterrows():
        lam = base * float(row["district_consumption_proxy_norm"]) * (float(row["est_households"]) / mean_hh)
        lam = float(lam)
        if np.isnan(lam) or lam < 0:
                lam = 0
        n_orders = np.random.poisson(lam)
        for _ in range(n_orders):
            hours = np.arange(24)
            probs = np.array([time_of_day_profile(datetime(date.year, date.month, date.day, int(h))) for h in hours])
            probs = probs / probs.sum()
            hour = np.random.choice(hours, p=probs)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            ts = datetime(date.year, date.month, date.day, hour, minute, second)
            loc = sample_point_in_polygon(row.geometry)
            value = max(20, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
            results.append({
                "method": "poisson",
                "building_idx": idx,
                "timestamp": ts,
                "order_value": float(abs(value)),
                "latitude": loc.y if loc is not None else None,
                "longitude": loc.x if loc is not None else None
            })
    return results

def method_gravity(buildings_df, date, distance_decay=1.0):
    results = []
    grouped = buildings_df.groupby("district_id_join")
    for dist_id, group in grouped:
        if pd.isna(dist_id):
            continue
        group = group.copy()
        group["building_mass"] = group["district_consumption_proxy_norm"] * group["est_households"]
        # get district centroid if available
        try:
            dist_geom = g_districts.loc[g_districts.index == dist_id].geometry.values[0]
            centroid = dist_geom.centroid
        except Exception:
            centroid = group.geometry.unary_union.centroid
        group["dist_to_centroid"] = group.geometry.centroid.distance(centroid)
        group["dist_to_centroid"] = group["dist_to_centroid"].replace(0.0, 1e-6)
        group["gravity_w"] = group["building_mass"] / (group["dist_to_centroid"] ** distance_decay)
        district_base = BASE_ORDER_RATE_PER_DISTRICT_PER_DAY * (group["district_consumption_proxy_norm"].iloc[0] if len(group)>0 else 1.0)
        total_w = group["gravity_w"].sum()
        if total_w <= 0:
            continue
        group["expected_orders"] = district_base * (group["gravity_w"] / total_w)
        for idx, row in group.iterrows():
            lam = float(lam)
            if np.isnan(lam) or lam < 0:
                lam = 0
            n = np.random.poisson(row["expected_orders"])
            for _ in range(n):
                hours = np.arange(24)
                probs = np.array([time_of_day_profile(datetime(date.year, date.month, date.day, int(h))) for h in hours])
                probs = probs / probs.sum()
                hour = np.random.choice(hours, p=probs)
                ts = datetime(date.year, date.month, date.day, hour, np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                value = max(20, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                results.append({
                    "method": "gravity",
                    "building_idx": idx,
                    "timestamp": ts,
                    "order_value": float(abs(value)),
                    "latitude": loc.y if loc is not None else None,
                    "longitude": loc.x if loc is not None else None
                })
    return results

def method_agent_based(buildings_df, date, avg_agent_order_prob=0.02):
    results = []
    for idx, row in buildings_df.iterrows():
        n_agents = min(MAX_AGENTS_PER_BUILDING, max(1, int(row["est_households"])))
        p = avg_agent_order_prob * float(row["district_consumption_proxy_norm"])
        for agent in range(n_agents):
            if random.random() < p:
                hours = np.arange(24)
                probs = np.array([time_of_day_profile(datetime(date.year, date.month, date.day, int(h))) for h in hours])
                probs = probs / probs.sum()
                hour = np.random.choice(hours, p=probs)
                ts = datetime(date.year, date.month, date.day, hour, np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                val = max(20, np.random.lognormal(mean=np.log(max(1.0, ORDER_VALUE_MEAN)), sigma=0.6))
                results.append({
                    "method": "agent",
                    "building_idx": idx,
                    "timestamp": ts,
                    "order_value": float(abs(val)),
                    "latitude": loc.y if loc is not None else None,
                    "longitude": loc.x if loc is not None else None
                })
    return results

def method_self_exciting(buildings_df, date, base_rate=0.5, excitation_prob=0.3, decay_minutes=30):
    results = []
    mean_hh = max(1.0, buildings_df["est_households"].mean())
    for idx, row in buildings_df.iterrows():
        lam = base_rate * float(row["district_consumption_proxy_norm"]) * (float(row["est_households"]) / mean_hh)
        lam = float(lam)
        if np.isnan(lam) or lam < 0:
            lam = 0
        n_base = np.random.poisson(lam)
        for _ in range(n_base):
            hours = np.arange(24)
            probs = np.array([time_of_day_profile(datetime(date.year, date.month, date.day, int(h))) for h in hours])
            probs = probs / probs.sum()
            hour = np.random.choice(hours, p=probs)
            t0 = datetime(date.year, date.month, date.day, hour, np.random.randint(0,60), np.random.randint(0,60))
            loc = sample_point_in_polygon(row.geometry)
            value = max(20, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
            results.append({
                "method": "self_exciting",
                "building_idx": idx,
                "timestamp": t0,
                "order_value": float(abs(value)),
                "latitude": loc.y if loc is not None else None,
                "longitude": loc.x if loc is not None else None
            })
            if random.random() < excitation_prob:
                n_off = np.random.poisson(1.5)
                for i in range(n_off):
                    delta = np.random.exponential(scale=decay_minutes)
                    ts_off = t0 + timedelta(minutes=float(delta))
                    if ts_off.date() != t0.date():
                        continue
                    loc_off = sample_point_in_polygon(row.geometry)
                    val_off = max(20, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                    results.append({
                        "method": "self_exciting",
                        "building_idx": idx,
                        "timestamp": ts_off,
                        "order_value": float(abs(val_off)),
                        "latitude": loc_off.y if loc_off is not None else None,
                        "longitude": loc_off.x if loc_off is not None else None
                    })
    return results

# --------------------------
# Main simulation loop (7 days)
# --------------------------
all_orders = []
print("Starting simulation for", NUM_DAYS, "days...")
for day_idx in range(NUM_DAYS):
    date = START_DATE + timedelta(days=day_idx)
    print("  Simulating date:", date.date())
    orders_poisson = method_poisson_nhpp(bgeo, date)
    orders_gravity = method_gravity(bgeo, date)
    orders_agent = method_agent_based(bgeo, date)
    orders_se = method_self_exciting(bgeo, date)

    # aggregate counts by building from each method
    from collections import defaultdict
    def build_count_map(lst):
        d = defaultdict(int)
        for o in lst:
            d[o["building_idx"]] += 1
        return d

    c_poisson = build_count_map(orders_poisson)
    c_gravity = build_count_map(orders_gravity)
    c_agent = build_count_map(orders_agent)
    c_se = build_count_map(orders_se)

    all_building_ids = set(list(c_poisson.keys()) + list(c_gravity.keys()) + list(c_agent.keys()) + list(c_se.keys()))
    for bidx in all_building_ids:
        wpo = WEIGHTS["poisson"] * c_poisson.get(bidx, 0)
        wgr = WEIGHTS["gravity"] * c_gravity.get(bidx, 0)
        wag = WEIGHTS["agent"] * c_agent.get(bidx, 0)
        wse = WEIGHTS["self_exciting"] * c_se.get(bidx, 0)
        expected = wpo + wgr + wag + wse
        lam = float(lam)
        if np.isnan(lam) or lam < 0:
            lam = 0
        final_n = np.random.poisson(expected)
        candidates = [o for o in (orders_poisson + orders_gravity + orders_agent + orders_se) if o["building_idx"] == bidx]
        for i in range(final_n):
            if i < len(candidates):
                entry = candidates[i]
                entry_out = dict(entry)
                entry_out["method"] = "combined"
                all_orders.append(entry_out)
            else:
                row = bgeo.loc[bidx]
                hours = np.arange(24)
                probs = np.array([time_of_day_profile(datetime(date.year, date.month, date.day, int(h))) for h in hours])
                probs = probs / probs.sum()
                ts_hour = np.random.choice(hours, p=probs)
                ts = datetime(date.year, date.month, date.day, int(ts_hour), np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                val = max(20, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                all_orders.append({
                    "method": "combined",
                    "building_idx": bidx,
                    "timestamp": ts,
                    "order_value": float(abs(val)),
                    "latitude": loc.y if loc is not None else None,
                    "longitude": loc.x if loc is not None else None
                })

print("Total orders generated:", len(all_orders))

# --------------------------
# Save results
# --------------------------
print("Saving orders to CSV and GeoJSON...")
orders_df = pd.DataFrame(all_orders)
orders_df["timestamp_iso"] = orders_df["timestamp"].apply(lambda t: t.isoformat() if pd.notna(t) else None)

# merge simple building meta
meta_cols = ["est_households", "district_consumption_proxy", "district_consumption_proxy_norm"]
bmeta = bgeo[meta_cols].copy()
orders_df = orders_df.merge(bmeta, left_on="building_idx", right_index=True, how="left")

orders_df.to_csv(OUTPUT_ORDERS_CSV, index=False)

geo_orders = gpd.GeoDataFrame(
    orders_df,
    geometry=[Point(xy) if (not pd.isna(xy[0]) and not pd.isna(xy[1])) else None for xy in zip(orders_df['longitude'], orders_df['latitude'])],
    crs="EPSG:4326"
)
geo_orders.to_file(OUTPUT_ORDERS_GEOJSON, driver="GeoJSON")

print("Saved:", OUTPUT_ORDERS_CSV, OUTPUT_ORDERS_GEOJSON)
print("Done.")
