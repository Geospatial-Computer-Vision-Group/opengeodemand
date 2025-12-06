"""
generate_orders_sample_tuned.py

Tuned sample run of synthetic quick-commerce orders.
- Samples SAMPLE_N buildings to validate pipeline quickly.
- Safety caps and parameter tuning applied.
- Max orders per building per day = 100 (user requested).
- Outputs saved to /mnt/data

Inputs (uploaded in current session):
 - /mnt/data/buildings_Surat_Gujarat_India.geojson
 - /mnt/data/district_data_Surat_Gujarat_India.geojson

Outputs:
 - /mnt/data/synthetic_orders_sample_tuned.csv
 - /mnt/data/synthetic_orders_sample_tuned.geojson
 - /mnt/data/synthetic_orders_sample_tuned_flags.csv (anomalies)
"""

import math, random, time, re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Config
# --------------------------
BUILDINGS_GEOJSON = Path("buildings_Surat_Gujarat_India.geojson")
DISTRICTS_GEOJSON = Path("district_data_Surat_Gujarat_India.geojson")

OUTPUT_ORDERS_CSV = Path("orders/synthetic_orders_sample.csv")
OUTPUT_ORDERS_GEOJSON = Path("orders/synthetic_orders_sample.geojson")
OUTPUT_FLAGS_CSV = Path("orders/synthetic_orders_sample_tuned_flags.csv")


RANDOM_SEED = 42
SAMPLE_N = 1000          # sample buildings to run quickly
START_DATE = datetime(2023, 11, 1)
NUM_DAYS = 7             # run for a week

# method weights
WEIGHTS = {"poisson": 0.35, "gravity": 0.30, "agent": 0.25, "self_exciting": 0.10}

# demand/value params (tuned)
BASE_ORDER_RATE_PER_DISTRICT_PER_DAY = 20    # reduced from 200 -> 20
AVG_AGENT_ORDER_PROB = 0.002                # 0.2% per household per day
ORDER_VALUE_MEAN = 250
ORDER_VALUE_STD = 100
MIN_ORDER_VALUE = 50.0                      # enforce minimum order value
MAX_AGENTS_PER_BUILDING = 200

# safety caps
SAFE_POISSON_LAMBDA_CAP = 200.0             # cap lambda for Poisson
MAX_ORDERS_PER_BUILDING_PER_DAY = 100       # user requested cap

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --------------------------
# Helpers
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
        p = Point(x, y)
        if poly.contains(p):
            return p
    return poly.representative_point()

def safe_float(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except:
        return None

def safe_poisson(lam, cap=SAFE_POISSON_LAMBDA_CAP):
    try:
        lam = float(lam)
    except:
        lam = 0.0
    if np.isnan(lam) or lam < 0:
        lam = 0.0
    lam = min(lam, cap)
    return np.random.poisson(lam)

def time_of_day_profile(hour):
    if 7 <= hour < 10: return 0.9
    if 11 <= hour < 14: return 1.2
    if 14 <= hour < 17: return 0.7
    if 18 <= hour < 22: return 1.6
    if 6 <= hour < 7: return 0.5
    return 0.3

hours = np.arange(24)
tod_weights = np.array([time_of_day_profile(h) for h in hours])
tod_weights = tod_weights / tod_weights.sum()

# --------------------------
# Load & sample
# --------------------------
print("Loading district GeoJSON...")
g_districts = gpd.read_file(str(DISTRICTS_GEOJSON)).to_crs(epsg=4326)
g_districts = g_districts.reset_index(drop=False).rename(columns={"index":"district_id_join"}).set_index("district_id_join", drop=False)
print("Districts:", len(g_districts))

print("Loading buildings (full file). This may be slow but we sample after loading.")
g_buildings_all = gpd.read_file(str(BUILDINGS_GEOJSON)).to_crs(epsg=4326)
print("Total buildings available:", len(g_buildings_all))

n_sample = min(int(SAMPLE_N), len(g_buildings_all))
print(f"Sampling {n_sample} buildings (seed={RANDOM_SEED}) ...")
g_buildings = g_buildings_all.sample(n=n_sample, random_state=RANDOM_SEED).copy()
# preserve building index used as stable identifier
g_buildings = g_buildings.reset_index(drop=False).rename(columns={"index":"building_idx"}).set_index("building_idx", drop=False)
print("Sampled buildings:", len(g_buildings))

# --------------------------
# Spatial join (sample -> districts)
# --------------------------
print("Spatial join (sampled buildings -> districts) using 'within' ...")
joined = gpd.sjoin(g_buildings, g_districts, how="left", predicate="within", lsuffix="_b", rsuffix="_d")

# fallback intersects for those not assigned
missing_mask = joined['district_id_join'].isna()
if missing_mask.any():
    missing_idxs = missing_mask.index[missing_mask]
    print(f"  {len(missing_idxs)} sampled buildings not assigned by 'within' - trying 'intersects' fallback...")
    sjoin2 = gpd.sjoin(g_buildings.loc[missing_idxs], g_districts, how="left", predicate="intersects", lsuffix="_b", rsuffix="_d")
    for idx, row in sjoin2.iterrows():
        joined.loc[idx, sjoin2.columns] = row

# prefix district columns
district_cols = [c for c in g_districts.columns if c not in ("geometry","district_id_join")]
for c in district_cols:
    if c in joined.columns:
        joined = joined.rename(columns={c: f"district_{c}"})

bgeo = joined.copy()
print("Merged sample buildings count:", len(bgeo))

# --------------------------
# Estimate households (area-based, projected to metric CRS)
# --------------------------
print("Estimating households using projected areas (EPSG:3857)...")
geom_series = gpd.GeoSeries(bgeo.geometry.values, crs="EPSG:4326").to_crs(epsg=3857)
areas_m2 = geom_series.geometry.area.values

def coerce_levels_to_num(v):
    try:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v)
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return np.nan
    except:
        return np.nan

level_keys = [k for k in bgeo.columns if k.lower().startswith("building:") or "level" in k.lower()]
levels = np.full(len(bgeo), np.nan)
for key in ["building:levels", "building_levels", "building:levels:underground", "building:levels_total", "levels"]:
    if key in bgeo.columns:
        vals = bgeo[key].values
        coerced = np.array([coerce_levels_to_num(v) for v in vals], dtype=float)
        levels = np.where(np.isnan(levels), coerced, levels)
levels = np.where(np.isnan(levels), 1.0, levels)

est_hh = (levels * np.maximum(areas_m2, 1.0) / 35.0).astype(int)
est_hh = np.where(est_hh < 1, 1, est_hh)
bgeo["est_households"] = est_hh

# --------------------------
# District consumption proxy (robust)
# --------------------------
print("Constructing district consumption proxy...")
preferred = [
    "district_hces_lvl02_multiplier", "district_avg_daily_expenditure",
    "district_consumption_per_day", "district_avg_expenditure",
    "district_BACT", "district_DT", "district_population"
]

def pick_proxy_from_row(row):
    for p in preferred:
        if p in row and row[p] not in (None, "", "null", np.nan):
            v = safe_float(row[p])
            if v is not None:
                return v
    # fallback: scan any district_* column
    for c in row.index:
        if str(c).startswith("district_"):
            v = safe_float(row[c])
            if v is not None:
                return v
    return 1.0

proxy_list = []
for idx, row in bgeo.iterrows():
    proxy_list.append(pick_proxy_from_row(row))
bgeo["district_consumption_proxy"] = np.array(proxy_list, dtype=float)

mean_proxy = np.nanmean(bgeo["district_consumption_proxy"].values)
if np.isnan(mean_proxy) or mean_proxy <= 0:
    mean_proxy = 1.0
bgeo["district_consumption_proxy_norm"] = bgeo["district_consumption_proxy"].values / mean_proxy
bgeo["district_consumption_proxy_norm"] = np.nan_to_num(bgeo["district_consumption_proxy_norm"], nan=1.0)
bgeo["district_consumption_proxy_norm"] = np.clip(bgeo["district_consumption_proxy_norm"], 0.01, None)

# ensure households integer >=1
bgeo["est_households"] = pd.to_numeric(bgeo["est_households"], errors="coerce").fillna(1).astype(int).clip(lower=1)

print("Sanity: proxy min/max:", bgeo["district_consumption_proxy_norm"].min(), bgeo["district_consumption_proxy_norm"].max())
print("Sanity: households min/max:", bgeo["est_households"].min(), bgeo["est_households"].max())

# build district centroid mapping for gravity method
district_geom_map = {idx: geom for idx, geom in g_districts.geometry.items()}

# --------------------------
# Demand generation methods
# --------------------------
def method_poisson_nhpp(buildings_df, date):
    results = []
    mean_hh = max(1.0, buildings_df["est_households"].mean())
    base = BASE_ORDER_RATE_PER_DISTRICT_PER_DAY
    for idx, row in buildings_df.iterrows():
        lam = base * float(row["district_consumption_proxy_norm"]) * (float(row["est_households"]) / mean_hh)
        n_orders = safe_poisson(lam)
        # cap per-building daily orders to avoid runaway
        n_orders = min(n_orders, MAX_ORDERS_PER_BUILDING_PER_DAY)
        for _ in range(n_orders):
            h = np.random.choice(hours, p=tod_weights)
            ts = datetime(date.year, date.month, date.day, int(h), np.random.randint(0,60), np.random.randint(0,60))
            loc = sample_point_in_polygon(row.geometry)
            val = max(MIN_ORDER_VALUE, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
            results.append({"method":"poisson","building_idx":idx,"timestamp":ts,"order_value":abs(float(val)),"latitude":loc.y if loc else None,"longitude":loc.x if loc else None})
    return results

def method_gravity(buildings_df, date, distance_decay=1.0):
    results = []
    grouped = buildings_df.groupby("district_id_join")
    for dist_id, group in grouped:
        if pd.isna(dist_id):
            continue
        group = group.copy()
        group["building_mass"] = group["district_consumption_proxy_norm"] * group["est_households"]
        centroid = district_geom_map.get(dist_id, None)
        if centroid is None:
            centroid = group.geometry.unary_union.centroid
        group["dist_to_centroid"] = group.geometry.centroid.distance(centroid)
        group["dist_to_centroid"].replace(0.0, 1e-6, inplace=True)
        group["gravity_w"] = group["building_mass"] / (group["dist_to_centroid"] ** distance_decay)
        district_base = BASE_ORDER_RATE_PER_DISTRICT_PER_DAY * (group["district_consumption_proxy_norm"].iloc[0] if len(group)>0 else 1.0)
        total_w = group["gravity_w"].sum()
        if total_w <= 0:
            continue
        group["expected_orders"] = district_base * (group["gravity_w"] / total_w)
        for idx, row in group.iterrows():
            n = safe_poisson(row["expected_orders"])
            n = min(n, MAX_ORDERS_PER_BUILDING_PER_DAY)
            for _ in range(n):
                h = np.random.choice(hours, p=tod_weights)
                ts = datetime(date.year, date.month, date.day, int(h), np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                val = max(MIN_ORDER_VALUE, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                results.append({"method":"gravity","building_idx":idx,"timestamp":ts,"order_value":abs(float(val)),"latitude":loc.y if loc else None,"longitude":loc.x if loc else None})
    return results

def method_agent_based(buildings_df, date, avg_agent_order_prob=AVG_AGENT_ORDER_PROB):
    results = []
    for idx, row in buildings_df.iterrows():
        n_agents = min(MAX_AGENTS_PER_BUILDING, max(1, int(row["est_households"])))
        p = avg_agent_order_prob * float(row["district_consumption_proxy_norm"])
        # expected agents ordering per building = n_agents * p
        # but cap the total produced orders per building per day
        cnt = 0
        for agent in range(n_agents):
            if random.random() < p:
                cnt += 1
                if cnt > MAX_ORDERS_PER_BUILDING_PER_DAY:
                    break
                h = np.random.choice(hours, p=tod_weights)
                ts = datetime(date.year, date.month, date.day, int(h), np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                val = max(MIN_ORDER_VALUE, np.random.lognormal(mean=np.log(max(1.0, ORDER_VALUE_MEAN)), sigma=0.6))
                results.append({"method":"agent","building_idx":idx,"timestamp":ts,"order_value":abs(float(val)),"latitude":loc.y if loc else None,"longitude":loc.x if loc else None})
    return results

def method_self_exciting(buildings_df, date, base_rate=0.5, excitation_prob=0.3, decay_minutes=30):
    results = []
    mean_hh = max(1.0, buildings_df["est_households"].mean())
    for idx, row in buildings_df.iterrows():
        lam = base_rate * float(row["district_consumption_proxy_norm"]) * (float(row["est_households"]) / mean_hh)
        n_base = safe_poisson(lam)
        n_base = min(n_base, MAX_ORDERS_PER_BUILDING_PER_DAY)
        for _ in range(n_base):
            h = np.random.choice(hours, p=tod_weights)
            t0 = datetime(date.year, date.month, date.day, int(h), np.random.randint(0,60), np.random.randint(0,60))
            loc = sample_point_in_polygon(row.geometry)
            val = max(MIN_ORDER_VALUE, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
            results.append({"method":"self_exciting","building_idx":idx,"timestamp":t0,"order_value":abs(float(val)),"latitude":loc.y if loc else None,"longitude":loc.x if loc else None})
            if random.random() < excitation_prob:
                n_off = np.random.poisson(1.5)
                for i in range(n_off):
                    delta = np.random.exponential(scale=decay_minutes)
                    ts_off = t0 + timedelta(minutes=float(delta))
                    if ts_off.date() != t0.date():
                        continue
                    loc_off = sample_point_in_polygon(row.geometry)
                    val_off = max(MIN_ORDER_VALUE, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                    results.append({"method":"self_exciting","building_idx":idx,"timestamp":ts_off,"order_value":abs(float(val_off)),"latitude":loc_off.y if loc_off else None,"longitude":loc_off.x if loc_off else None})
    return results

# --------------------------
# Simulation loop (sample)
# --------------------------
print("Simulating orders for", NUM_DAYS, "days on sample of", len(bgeo), "buildings...")
all_orders = []
start_sim = time.time()

for d in range(NUM_DAYS):
    date = START_DATE + timedelta(days=d)
    print(" Day", d+1, "->", date.date())
    po = method_poisson_nhpp(bgeo, date)
    gr = method_gravity(bgeo, date)
    ag = method_agent_based(bgeo, date)
    se = method_self_exciting(bgeo, date)

    # aggregate counts by building across methods
    from collections import defaultdict
    def count_map(lst):
        dd = defaultdict(int)
        for o in lst:
            dd[o["building_idx"]] += 1
        return dd

    cpo, cgr, cag, cse = count_map(po), count_map(gr), count_map(ag), count_map(se)
    all_bids = set(list(cpo.keys()) + list(cgr.keys()) + list(cag.keys()) + list(cse.keys()))

    # combine method counts using weights and sample final per-building counts
    for bidx in all_bids:
        wpo = WEIGHTS["poisson"] * cpo.get(bidx, 0)
        wgr = WEIGHTS["gravity"] * cgr.get(bidx, 0)
        wag = WEIGHTS["agent"] * cag.get(bidx, 0)
        wse = WEIGHTS["self_exciting"] * cse.get(bidx, 0)
        expected = wpo + wgr + wag + wse

        # sample final_n and apply per-building daily cap
        final_n = safe_poisson(expected)
        final_n = int(min(final_n, MAX_ORDERS_PER_BUILDING_PER_DAY))

        candidates = [o for o in (po + gr + ag + se) if o["building_idx"] == bidx]
        for i in range(final_n):
            if i < len(candidates):
                e = dict(candidates[i])
                e["method"] = "combined"
                all_orders.append(e)
            else:
                row = bgeo.loc[bidx]
                h = np.random.choice(hours, p=tod_weights)
                ts = datetime(date.year, date.month, date.day, int(h), np.random.randint(0,60), np.random.randint(0,60))
                loc = sample_point_in_polygon(row.geometry)
                val = max(MIN_ORDER_VALUE, np.random.normal(ORDER_VALUE_MEAN * float(row["district_consumption_proxy_norm"]), ORDER_VALUE_STD))
                all_orders.append({"method":"combined","building_idx":bidx,"timestamp":ts,"order_value":abs(float(val)),"latitude":loc.y if loc else None,"longitude":loc.x if loc else None})

    # optional safety break if too many orders overall
    if len(all_orders) > 5_000_000:
        print("Reached safety total-order cap; stopping early.")
        break

elapsed = time.time() - start_sim
print("Simulation complete. Time: {:.1f}s Orders generated: {}".format(elapsed, len(all_orders)))

# --------------------------
# Save outputs and produce a small anomalies flags file
# --------------------------
print("Saving outputs to:", OUTPUT_ORDERS_CSV, OUTPUT_ORDERS_GEOJSON)
orders_df = pd.DataFrame(all_orders)
orders_df["timestamp_iso"] = orders_df["timestamp"].apply(lambda t: t.isoformat() if pd.notna(t) else None)

# add small building metadata for convenience
meta_cols = ["est_households", "district_consumption_proxy", "district_consumption_proxy_norm"]
meta_present = [c for c in meta_cols if c in bgeo.columns]
bmeta = bgeo[meta_present].copy()
orders_df = orders_df.merge(bmeta, left_on="building_idx", right_index=True, how="left")

orders_df.to_csv(OUTPUT_ORDERS_CSV, index=False)

# build GeoDataFrame for geojson output
geo_orders = gpd.GeoDataFrame(orders_df, geometry=[Point(xy) if (not pd.isna(xy[0]) and not pd.isna(xy[1])) else None for xy in zip(orders_df['longitude'], orders_df['latitude'])], crs="EPSG:4326")
geo_orders.to_file(OUTPUT_ORDERS_GEOJSON, driver="GeoJSON")

# anomalies / flags
flags = []
for i,row in orders_df.iterrows():
    issues = []
    if (pd.isna(row['latitude']) or pd.isna(row['longitude'])):
        issues.append("missing_coords")
    if row["order_value"] < MIN_ORDER_VALUE:
        issues.append("low_value")
    if row["order_value"] > 50000:
        issues.append("very_high_value")
    if issues:
        flags.append({"index": i, "building_idx": row["building_idx"], "timestamp_iso": row.get("timestamp_iso", None), "order_value": row["order_value"], "issues": ";".join(issues)})

if len(flags) > 0:
    pd.DataFrame(flags).to_csv(OUTPUT_FLAGS_CSV, index=False)
    print("Saved flags to:", OUTPUT_FLAGS_CSV)
else:
    # create an empty flags file for consistency
    pd.DataFrame(columns=["index","building_idx","timestamp_iso","order_value","issues"]).to_csv(OUTPUT_FLAGS_CSV, index=False)
    print("No flags found; wrote empty flags file:", OUTPUT_FLAGS_CSV)

print("Done.")
