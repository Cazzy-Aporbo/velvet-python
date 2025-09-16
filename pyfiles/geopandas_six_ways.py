# -*- coding: utf-8 -*-
"""
GeoPandas Masterclass - Six Ways to Think and Teach
Author: Cazandra Aporbo  |  December 2024
Updated: August 13, 2025

Six distinct approaches to common spatial patterns:

  0) Synthetic dataset: deterministic admin cells, cities, clinics (EPSG:4326)
  1) Lab-notebook pipeline: GeoDataFrame basics, CRS, quick plot
  2) Spatial join + dissolve: assign cities->admin, rollups
  3) Service areas: local equal-area projection -> buffers -> overlay coverage
  4) Nearest facility two ways: (a) sjoin_nearest (b) manual sindex.query_nearest
  5) Coverage cross-check: overlay vs centroid-weighted join
  6) Biggish pattern: windowed sindex nearest (avoid global N*M joins)

Dependencies: geopandas, shapely>=2, pyproj, pandas, numpy, matplotlib
Optional: rich (for pretty tables)

Works with geopandas 0.10+ (includes fallbacks for older versions)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import distance as shp_distance
import matplotlib.pyplot as plt
from pyproj import CRS

# Optional pretty console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None


# -------------------------- utilities -----------------------------------------

class Timer:
    """Simple timer for performance tracking"""
    def __init__(self, label: str):
        self.label = label
        self.t0 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        dt = (time.perf_counter() - self.t0) * 1000.0
        print(f"[timing] {self.label}: {dt:.2f} ms")

def ensure_crs(gdf: gpd.GeoDataFrame, expected: str | CRS) -> None:
    """Verify CRS matches expectations - catches many common errors"""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Use gdf.set_crs('EPSG:4326') or similar.")
    if CRS.from_user_input(gdf.crs) != CRS.from_user_input(expected):
        raise ValueError(f"CRS mismatch. Expected {expected}, got {gdf.crs}")

def local_equal_area_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Create custom equal-area projection centered on dataset for accurate area/distance calcs"""
    c = gdf.unary_union.centroid
    return CRS.from_proj4(f"+proj=laea +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m +no_defs")

def make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries using zero-width buffer trick"""
    geom = gdf.geometry.buffer(0)
    out = gdf.set_geometry(geom)
    out = out[~out.geometry.is_empty]
    out = out[~out.geometry.isna()]
    return out

def rng(seed: int = 11) -> np.random.Generator:
    """Seeded random generator for reproducibility"""
    return np.random.default_rng(seed)


# -------------------------- dataset builder -----------------------------------
@dataclass
class ToyData:
    admin: gpd.GeoDataFrame   # polygons with admin_id
    cities: gpd.GeoDataFrame  # points with pop, name
    clinics: gpd.GeoDataFrame # points with capacity, name

def build_dataset(seed: int = 11) -> ToyData:
    """
    Create synthetic 6x6 admin grid with random cities and clinics.
    All in EPSG:4326 (lat/lon) - do NOT measure distances/areas in this CRS.
    """
    rnd = rng(seed)
    xmin, ymin, xmax, ymax = -5.0, 45.0, 5.0, 55.0
    nx, ny = 6, 6

    # Build admin grid
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    polys, admin_id = [], []
    aid = 0
    for i in range(nx):
        for j in range(ny):
            polys.append(box(xs[i], ys[j], xs[i+1], ys[j+1]))
            admin_id.append(aid); aid += 1
    admin = gpd.GeoDataFrame({"admin_id": admin_id}, geometry=polys, crs="EPSG:4326")

    # Cities with lognormal population distribution
    n_cities = 150
    cx = rnd.uniform(xmin, xmax, n_cities)
    cy = rnd.uniform(ymin, ymax, n_cities)
    cpop = rnd.lognormal(mean=10.2, sigma=0.6, size=n_cities).astype(int)
    cnames = [f"City_{i}" for i in range(n_cities)]
    cities = gpd.GeoDataFrame(
        {"name": cnames, "pop": cpop},
        geometry=gpd.points_from_xy(cx, cy),
        crs="EPSG:4326"
    )

    # Clinics with random capacities
    n_clinics = 30
    kx = rnd.uniform(xmin, xmax, n_clinics)
    ky = rnd.uniform(ymin, ymax, n_clinics)
    kcap = rnd.integers(50, 600, size=n_clinics)
    knames = [f"Clinic_{i}" for i in range(n_clinics)]
    clinics = gpd.GeoDataFrame(
        {"name": knames, "capacity": kcap},
        geometry=gpd.points_from_xy(kx, ky),
        crs="EPSG:4326"
    )

    return ToyData(admin=admin, cities=cities, clinics=clinics)


# -------------------------- 1) basics + plot ----------------------------------
def demo_basics(data: ToyData, out_png: Optional[str] = None) -> None:
    """Basic visualization - always look at your data first"""
    g_admin = make_valid(data.admin)
    ensure_crs(g_admin, "EPSG:4326")
    
    ax = g_admin.boundary.plot(figsize=(7.5, 7), color="#888")
    data.cities.plot(ax=ax, color="#004DFF", markersize=6, alpha=0.7, label="cities")
    data.clinics.plot(ax=ax, color="#FF006E", markersize=18, marker="^", alpha=0.8, label="clinics")
    ax.set_title("Synthetic Dataset (CRS=EPSG:4326)\nDo not measure distances/areas in this CRS")
    ax.legend(loc="lower left")
    
    if out_png:
        plt.tight_layout(); plt.savefig(out_png, dpi=160)
    plt.close()


def cities_to_admin_rollup(data: ToyData) -> gpd.GeoDataFrame:
    """
    Classic spatial join pattern: assign cities to admin regions and aggregate.
    Uses 'left' join to keep cities that don't fall in any admin region.
    """
    joined = gpd.sjoin(
        data.cities,
        data.admin[["admin_id", "geometry"]],
        predicate="within",
        how="left"
    )
    
    joined["admin_id"] = joined["admin_id"].fillna(-1).astype(int)
    
    roll = joined.groupby("admin_id", as_index=False).agg(
        pop_total=("pop", "sum"),
        n_cities=("name", "count"),
    )
    
    out = data.admin.merge(roll, on="admin_id", how="left").fillna({"pop_total": 0, "n_cities": 0})
    return out


# -------------------------- 3) service areas (buffers + overlay) --------------
def service_areas(data: ToyData, radius_km: float = 25.0):
    """
    Calculate service coverage by buffering clinics and overlaying with admin regions.
    Must project to equal-area CRS for accurate area calculations.
    """
    laea = local_equal_area_crs(data.admin)
    admin_m = data.admin.to_crs(laea).copy()
    clinics_m = data.clinics.to_crs(laea).copy()

    # Buffer clinics to create service areas
    buf = clinics_m.copy()
    buf["geometry"] = buf.buffer(radius_km * 1000.0)
    buf["clinic_id"] = np.arange(len(buf))
    buf = make_valid(buf)

    # Union all buffers and calculate coverage
    service_union = gpd.GeoDataFrame(geometry=[unary_union(buf.geometry)], crs=laea)
    inter = gpd.overlay(admin_m, service_union, how="intersection")
    admin_m["area_m2"] = admin_m.area
    inter["area_m2"] = inter.area
    
    cov = inter.groupby("admin_id", as_index=False)["area_m2"].sum().rename(columns={"area_m2": "covered_m2"})
    admin_cov = admin_m.merge(cov, on="admin_id", how="left").fillna({"covered_m2": 0.0})
    admin_cov["coverage_pct"] = (admin_cov["covered_m2"] / admin_cov["area_m2"]).clip(0, 1) * 100.0

    return buf, admin_cov


# -------------------------- 4a) nearest via sjoin_nearest ---------------------
# -------------------------- 4a) nearest via sjoin_nearest ---------------------
def nearest_facility_sjoin(data: ToyData, max_km: float = 50.0) -> gpd.GeoDataFrame:
    """High-level approach using geopandas sjoin_nearest"""
    laea = local_equal_area_crs(data.cities)
    cities_m = data.cities.to_crs(laea)
    clinics_m = data.clinics.to_crs(laea)
    
    joined = gpd.sjoin_nearest(
        cities_m,
        clinics_m[["name", "capacity", "geometry"]],
        how="left",
        distance_col="dist_m"
    )
    
    joined["within_max_km"] = (joined["dist_m"] <= max_km * 1000.0)
    return joined


# -------------------------- 4b) nearest via sindex.query_nearest --------------
def nearest_facility_query_manual(data: ToyData) -> pd.DataFrame:
    """
    Manual approach using spatial index directly.
    Handles version differences between geopandas releases.
    """
    laea = local_equal_area_crs(data.cities)
    cities_m = data.cities.to_crs(laea).reset_index(drop=True)
    clinics_m = data.clinics.to_crs(laea).reset_index(drop=True)

    try:
        # Try modern query_nearest if available
        if hasattr(clinics_m.sindex, 'query_nearest'):
            res = clinics_m.sindex.query_nearest(
                cities_m.geometry,
                return_distance=True,
            )
            
            if isinstance(res, tuple) and len(res) == 3:
                left_ix, right_ix, dist = res
                left_ix = np.asarray(left_ix, dtype=int)
                right_ix = np.asarray(right_ix, dtype=int)
                dist = np.asarray(dist, dtype=float)
            else:
                left_ix, right_ix = res
                left_ix = np.asarray(left_ix, dtype=int)
                right_ix = np.asarray(right_ix, dtype=int)
                c_pts = cities_m.geometry.values[left_ix]
                k_pts = clinics_m.geometry.values[right_ix]
                dist = np.asarray(shp_distance(c_pts, k_pts), dtype=float)
        else:
            raise AttributeError("No query_nearest")
            
    except (AttributeError, TypeError):
        # Fallback for older geopandas versions
        print("  (using fallback method - consider upgrading geopandas)")
        
        city_names = []
        clinic_names = []
        distances = []
        
        for idx, city_geom in enumerate(cities_m.geometry):
            dists = clinics_m.geometry.distance(city_geom)
            nearest_idx = dists.idxmin()
            
            city_names.append(cities_m.loc[idx, "name"])
            clinic_names.append(clinics_m.loc[nearest_idx, "name"])
            distances.append(dists[nearest_idx])
        
        return pd.DataFrame({
            "city": city_names,
            "clinic": clinic_names,
            "dist_m": distances,
        })

    # Return formatted results
    out = pd.DataFrame({
        "city":   cities_m.loc[left_ix,  "name"].to_numpy(),
        "clinic": clinics_m.loc[right_ix, "name"].to_numpy(),
        "dist_m": dist,
    })
    return out


# -------------------------- 5) coverage cross-check ---------------------------
def coverage_by_overlay(admin_cov: gpd.GeoDataFrame) -> pd.Series:
    """The 'exact' method - we already calculated this via overlay"""
    return admin_cov.set_index("admin_id")["coverage_pct"]

def coverage_by_weighted_join(data: ToyData, buf: gpd.GeoDataFrame) -> pd.Series:
    """
    Alternative method using centroids. Not as accurate but MUCH faster
    for large datasets. Good enough for a sanity check.
    """
    laea = local_equal_area_crs(data.admin)
    admin_m = data.admin.to_crs(laea).copy()
    
    # Get centroids of admin regions
    admin_m["centroid"] = admin_m.geometry.centroid
    cent = admin_m[["admin_id", "centroid"]].copy()
    cent = cent.set_geometry("centroid")
    
    # Union all clinic buffers
    union = gpd.GeoDataFrame(geometry=[unary_union(buf.geometry)], crs=laea)
    
    # Which centroids are within coverage?
    hit = gpd.sjoin(cent, union, predicate="within", how="left")
    
    # Count hits vs total
    n_total = cent.groupby("admin_id").size().rename("n_total")
    n_hit = hit.dropna().groupby("admin_id").size().rename("n_hit")
    df = pd.concat([n_total, n_hit], axis=1).fillna(0)
    pct = (df["n_hit"] / df["n_total"]).clip(0, 1) * 100.0
    pct.name = "coverage_pct_centroid_weighted"
    return pct


# -------------------------- 6) windowed nearest (biggish pattern) -------------
def windowed_join(cities: gpd.GeoDataFrame, clinics: gpd.GeoDataFrame, window_km: float = 80.0) -> pd.DataFrame:
    """
    Windowed nearest neighbor for large datasets.
    Breaks area into tiles to avoid global N*M joins.
    Note: edge effects possible - choose window size carefully.
    """
    laea = local_equal_area_crs(cities)
    cities_m = cities.to_crs(laea)
    clinics_m = clinics.to_crs(laea)

    win = window_km * 1000.0
    cxmin, cymin, cxmax, cymax = cities_m.total_bounds
    xs = np.arange(cxmin, cxmax + win, win)
    ys = np.arange(cymin, cymax + win, win)

    tree = clinics_m.sindex
    out_rows = []
    has_query_nearest = hasattr(tree, 'query_nearest')
    
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = box(xs[i], ys[j], xs[i+1], ys[j+1])
            
            cities_cell = cities_m[cities_m.intersects(cell)]
            if cities_cell.empty:
                continue
                
            idx = list(tree.query(cell))
            if not idx:
                continue
                
            cand = clinics_m.iloc[idx]
            
            if has_query_nearest and hasattr(cand.sindex, 'query_nearest'):
                try:
                    left_ix, right_ix = cand.sindex.query_nearest(
                        cities_cell.geometry, 
                        return_distance=False
                    )
                    seen = set()
                    for ci, ki in zip(left_ix, right_ix):
                        city_idx = cities_cell.index[ci]
                        if city_idx not in seen:
                            seen.add(city_idx)
                            city_geom = cities_cell.geometry.iloc[ci]
                            clinic_geom = cand.geometry.iloc[ki]
                            dist = city_geom.distance(clinic_geom)
                            out_rows.append((city_idx, cand.index[ki], dist))
                except:
                    has_query_nearest = False
            
            if not has_query_nearest:
                # Fallback to brute force within window
                for ci, city_geom in enumerate(cities_cell.geometry):
                    if len(cand) > 0:
                        dists = cand.geometry.distance(city_geom)
                        nearest_idx = dists.idxmin()
                        out_rows.append((
                            cities_cell.index[ci],
                            nearest_idx,
                            dists[nearest_idx]
                        ))

    if not out_rows:
        return pd.DataFrame(columns=["city", "clinic", "dist_m"])

    df = pd.DataFrame(out_rows, columns=["city_idx", "clinic_idx", "dist_m"])
    df["city"] = cities_m.loc[df["city_idx"], "name"].to_numpy()
    df["clinic"] = clinics_m.loc[df["clinic_idx"], "name"].to_numpy()
    return df[["city", "clinic", "dist_m"]]


# -------------------------- mini game -----------------------------------------
# -------------------------- mini game -----------------------------------------
def game(data: ToyData) -> None:
    """Quick interactive test of coverage estimation skills"""
    laea = local_equal_area_crs(data.admin)
    _, admin_cov = service_areas(data, radius_km=25)
    admin_cov = admin_cov.to_crs(laea)

    cell = admin_cov.sample(1, random_state=42).iloc[0]
    cid = int(cell["admin_id"])
    pct = float(cell["coverage_pct"])
    print(f"\n-- Mini Game -- Admin cell #{cid}")
    guess = input("Is coverage >= 50%? (y/n): ").strip().lower()
    truth = pct >= 50.0
    print(f"Coverage is {pct:.1f}% -> correct answer: {'YES' if truth else 'NO'}")
    
    city = data.cities.sample(1, random_state=7).iloc[0]
    nearest = nearest_facility_sjoin(data, max_km=1e6)
    row = nearest.loc[nearest["name"] == city["name"]].iloc[0]
    clinic_col = "name_right" if "name_right" in nearest.columns else "name"
    print(f"Nearest clinic to {city['name']}: {row[clinic_col]} at {row['dist_m'] / 1000.0:.1f} km")


# -------------------------- plotting helper -----------------------------------
def quick_plot_service(data: ToyData, buf: gpd.GeoDataFrame, admin_cov: gpd.GeoDataFrame, out_png: str) -> None:
    """Create service coverage heatmap"""
    ax = admin_cov.plot(column="coverage_pct", cmap="viridis", legend=True, figsize=(8,7),
                        legend_kwds={"label": "Coverage (%)"})
    buf.boundary.plot(ax=ax, color="#FF006E", linewidth=0.8, alpha=0.7)
    data.clinics.to_crs(admin_cov.crs).plot(ax=ax, color="#FF006E", markersize=16, marker="^", label="clinics")
    data.cities.to_crs(admin_cov.crs).plot(ax=ax, color="#004DFF", markersize=4, alpha=0.6, label="cities")
    ax.set_title("Service Coverage by Admin Cell (equal-area projection)")
    ax.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_png, dpi=170); plt.close()


# -------------------------- main ----------------------------------------------
def main() -> int:
    """Run through all six spatial analysis patterns with timing"""
    print("\nGeoPandas Masterclass - by Cazandra Aporbo\n")
    
    # Version check
    try:
        import geopandas
        gpd_version = geopandas.__version__
        print(f"Running with geopandas {gpd_version}")
        if gpd_version < "0.12":
            print("  Note: Using fallback methods for some operations\n")
    except:
        pass

    data = build_dataset(seed=11)

    # Pattern 1: Visualization
    with Timer("build basics + plot"):
        demo_basics(data, out_png="00_basics.png")

    # Pattern 2: Spatial join and aggregation
    with Timer("cities->admin rollup"):
        admin_roll = cities_to_admin_rollup(data)

    if RICH:
        table = Table(title="Admin rollup (head)", box=rich_box.SIMPLE_HEAVY)
        table.add_column("admin_id"); table.add_column("pop_total"); table.add_column("n_cities")
        for _, r in admin_roll.head(6)[["admin_id","pop_total","n_cities"]].iterrows():
            table.add_row(str(int(r.admin_id)), str(int(r.pop_total)), str(int(r.n_cities)))
        console.print(table)
    else:
        print("\nAdmin rollup (first 6 regions):")
        print(admin_roll.head(6)[["admin_id", "pop_total", "n_cities"]])

    # Pattern 3: Buffer analysis
    with Timer("service areas (buffer + overlay)"):
        buf, admin_cov = service_areas(data, radius_km=25)
        quick_plot_service(data, buf, admin_cov, out_png="01_coverage.png")

    # Pattern 4: Nearest neighbor (two approaches)
    with Timer("nearest via sjoin_nearest"):
        nf1 = nearest_facility_sjoin(data, max_km=60)
    print(f"→ Found {len(nf1)} city-clinic pairs")

    with Timer("nearest via query_nearest (manual)"):
        nf2 = nearest_facility_query_manual(data)
    print(f"→ Manual method found {len(nf2)} pairs")

    # Pattern 5: Cross-validation
    with Timer("coverage cross-check"):
        exact = coverage_by_overlay(admin_cov)
        approx = coverage_by_weighted_join(data, buf).reindex(exact.index).fillna(0)
        corr = np.corrcoef(exact.fillna(0), approx.fillna(0))[0, 1]
        print(f"→ Coverage methods correlation: {corr:.3f}")

    # Pattern 6: Windowed approach
    with Timer("windowed nearest join"):
        wnd = windowed_join(data.cities, data.clinics, window_km=80.0)
        print(f"→ Windowed approach found {len(wnd)} pairs")

    play = input("\nPlay coverage guessing game? [y/N]: ").strip().lower()
    if play == "y":
        game(data)

    print("\nKey takeaways:")
    print("  • Always verify CRS before spatial operations")
    print("  • Project to equal-area for distance/area calculations")
    print("  • Use spatial joins for point-in-polygon operations")
    print("  • Buffer + overlay for coverage analysis")
    print("  • Consider windowed approaches for large datasets")
    
    print("\nGenerated files:")
    print("  • 00_basics.png (dataset overview)")
    print("  • 01_coverage.png (service coverage map)")
    
    print("\nDone - Cazandra\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())