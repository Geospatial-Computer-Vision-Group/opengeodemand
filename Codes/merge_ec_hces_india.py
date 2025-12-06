import os
import re
import glob
from typing import Dict, List, Set

import pandas as pd
import geopandas as gpd


def ensure_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(".0", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype("float").fillna(0).astype(int)


def read_vector(path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        try:
            return gpd.read_file(path)
        except AttributeError as e:
            raise SystemExit(
                "Reading file failed due to Fiona/GDAL mismatch.\n"
                "Conda: conda install -c conda-forge pyogrio geopandas fiona shapely pyproj gdal\n"
                "Pip:   pip install -U pyogrio geopandas fiona shapely pyproj gdal\n"
                f"Original error: {e}"
            )


def discover_ec_csvs(ec_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(ec_dir, "EC6A_ST*.csv")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(ec_dir, "EC6A_ST*.CSV")))
    return paths


def aggregate_ec(csv_path: str, numeric_cols_global: Set[str]) -> pd.DataFrame:
    required_keys = ["ST", "DT"]
    chunks = pd.read_csv(csv_path, low_memory=False, chunksize=200_000)
    accum: pd.DataFrame | None = None

    for idx, chunk in enumerate(chunks):
        for k in required_keys:
            if k not in chunk.columns:
                raise SystemExit(f"EC file missing key {k}: {csv_path}")
        chunk["ST"] = ensure_int(chunk["ST"]) 
        chunk["DT"] = ensure_int(chunk["DT"]) 

        if idx == 0:
            # discover numeric columns, excluding identifier/code fields that shouldn't be summed
            CODE_BLOCKLIST = {
                "STATE", "ST", "DISTRICT", "DT", "TEH", "T_V", "WC", "EB", "EBX", "SG", "SEX",
                "RELIGION", "SOF", "NOP", "OWN_SHIP_C", "NIC3", "HLOOM_ACT", "SECTOR"
            }
            for col in chunk.columns:
                if col in required_keys:
                    continue
                if str(col).upper() in CODE_BLOCKLIST:
                    continue
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    numeric_cols_global.add(col)
                else:
                    coerced = pd.to_numeric(chunk[col], errors="coerce")
                    if coerced.notna().any():
                        numeric_cols_global.add(col)
            # ensure common metrics exist if present
            for c in ["C_HOUSE", "IN_HH", "TOTAL_WORKER"]:
                if c in chunk.columns:
                    numeric_cols_global.add(c)

        present = [c for c in numeric_cols_global if c in chunk.columns]
        work = chunk[required_keys + present].copy()
        for c in present:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
        work["n_records"] = 1
        gb = work.groupby(["ST", "DT"], as_index=False).sum(numeric_only=True)
        accum = gb if accum is None else (
            pd.concat([accum, gb], ignore_index=True)
              .groupby(["ST", "DT"], as_index=False)
              .sum(numeric_only=True)
        )

    if accum is None:
        return pd.DataFrame(columns=["ST", "DT"]) 

    # rename a few known EC metrics with ec_ prefix
    rename_map: Dict[str, str] = {}
    if "C_HOUSE" in accum.columns:
        rename_map["C_HOUSE"] = "ec_n_enterprises"
    if "IN_HH" in accum.columns:
        rename_map["IN_HH"] = "ec_hh_enterprises"
    if "TOTAL_WORKER" in accum.columns:
        rename_map["TOTAL_WORKER"] = "ec_total_workers"
    accum = accum.rename(columns=rename_map)
    return accum


def discover_hces_csvs(hces_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(hces_dir, "LEVEL - *.csv")))


def slug_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"LEVEL\s*-\s*([0-9A-Za-z]+)", name)
    if m:
        return f"lvl{m.group(1).lower()}"
    return os.path.splitext(name)[0].lower().replace(" ", "_")


def aggregate_hces(csv_path: str) -> pd.DataFrame:
    # Aggregate numeric columns by State, District; prefix metrics by level slug
    slug = slug_from_filename(csv_path)
    # Read all as strings first to avoid mixed-type issues, then coerce
    df = pd.read_csv(csv_path, low_memory=False, dtype=str)

    # Normalize column names (strip spaces, unify case)
    df.columns = [c.strip().replace("\u00a0", " ") for c in df.columns]

    # Locate key columns case-insensitively
    colmap = {c.lower(): c for c in df.columns}
    state_col = colmap.get("state")
    district_col = colmap.get("district")
    if state_col is None or district_col is None:
        raise SystemExit(f"HCES file missing State/District keys: {csv_path} (found: {list(df.columns)})")

    # Create numeric versions of all columns (except identifiers) by coercion
    df["ST"] = ensure_int(df[state_col])
    df["DT"] = ensure_int(df[district_col])

    numeric_cols: List[str] = []
    for c in df.columns:
        if c in [state_col, district_col, "ST", "DT"]:
            continue
        # Attempt numeric coercion; if any non-null after coercion, treat as numeric
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().any():
            df[c] = coerced.fillna(0)
            numeric_cols.append(c)

    if not numeric_cols:
        # No numeric columns in this file; return just counts
        out = df.groupby(["ST", "DT"], as_index=False).size()
        out = out.rename(columns={"size": f"hces_{slug}_n_records"})
        return out

    work = df[["ST", "DT"] + numeric_cols].copy()

    agg = work.groupby(["ST", "DT"], as_index=False).sum(numeric_only=True)
    # prefix metric names to avoid collisions between levels
    rename = {c: f"hces_{slug}_{c.lower()}" for c in agg.columns if c not in ["ST", "DT"]}
    agg = agg.rename(columns=rename)
    return agg


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ec_dir = os.path.join(base_dir, "..")
    hces_dir = os.path.join(base_dir, "..", "..", "HCES")
    shp_path = os.path.join(ec_dir, "maps-master", "Districts", "Census_2011", "2011_Dist.shp")

    # Aggregate EC across all state files
    ec_files = discover_ec_csvs(ec_dir)
    if not ec_files:
        print(f"No Economic Census EC6A files found under: {ec_dir}")
    numeric_cols_global: Set[str] = set()
    ec_parts: List[pd.DataFrame] = []
    for p in ec_files:
        print(f"EC: {os.path.basename(p)}")
        ec_parts.append(aggregate_ec(p, numeric_cols_global))
    ec_all = pd.concat(ec_parts, ignore_index=True) if ec_parts else pd.DataFrame(columns=["ST","DT"])

    # Aggregate HCES across all levels
    hces_files = discover_hces_csvs(hces_dir)
    if not hces_files:
        print(f"No HCES 'LEVEL - *.csv' files found under: {hces_dir}")
    hces_parts: List[pd.DataFrame] = []
    for p in hces_files:
        print(f"HCES: {os.path.basename(p)}")
        try:
            hces_parts.append(aggregate_hces(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    # Outer-merge all HCES level aggregates on ST, DT
    if hces_parts:
        hces_all = hces_parts[0]
        for part in hces_parts[1:]:
            hces_all = hces_all.merge(part, on=["ST", "DT"], how="outer")
    else:
        hces_all = pd.DataFrame(columns=["ST","DT"]) 

    # Combine EC + HCES per district
    combined = ec_all.merge(hces_all, on=["ST", "DT"], how="outer")
    combined["ST"] = ensure_int(combined["ST"]) 
    combined["DT"] = ensure_int(combined["DT"]) 

    # Read districts geometry and prepare keys
    districts = read_vector(shp_path)
    up = {c.upper(): c for c in districts.columns}
    stc = up.get("ST_CEN_CD") or up.get("STCODE") or up.get("STATE_CODE")
    dtc = up.get("DT_CEN_CD") or up.get("DTCODE") or up.get("DIST_CODE")
    if not stc or not dtc:
        raise SystemExit(f"Could not detect ST/DT code columns in {shp_path}. Columns: {list(districts.columns)}")

    districts["ST"] = ensure_int(districts[stc])
    districts["DT"] = ensure_int(districts[dtc])

    gdf = districts.merge(combined, on=["ST", "DT"], how="left")
    # Fill NaNs in numeric columns
    num_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if c not in ["ST", "DT"]:
            gdf[c] = gdf[c].fillna(0)

    # Drop EC/HCES accidental identifier duplicates that collide with shapefile names case-insensitively
    drop_cols = [c for c in gdf.columns if c.lower() == "district" and c != "DISTRICT"]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)

    # Sanitize and uniquify column names for safe writing to SQLite/GPKG (case-insensitive)
    def sanitize_names(cols: List[str]) -> List[str]:
        seen = set()
        out = []
        for name in cols:
            base = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
            if re.match(r"^[0-9]", base):
                base = f"c_{base}"
            key = base.lower()
            if key in seen:
                i = 1
                new = f"{base}_{i}"
                while new.lower() in seen:
                    i += 1
                    new = f"{base}_{i}"
                base = new
                key = base.lower()
            seen.add(key)
            out.append(base)
        return out

    gdf.columns = sanitize_names(list(gdf.columns))

    out_dir = base_dir
    out_geojson = os.path.join(out_dir, "india_district_ec_hces.geojson")
    out_gpkg = os.path.join(out_dir, "india_district_ec_hces.gpkg")
    gdf.to_file(out_geojson, driver="GeoJSON")
    gdf.to_file(out_gpkg, layer="india_district_ec_hces", driver="GPKG")
    print("Saved:")
    print(f" - {out_geojson}")
    print(f" - {out_gpkg}")


if __name__ == "__main__":
    main()


