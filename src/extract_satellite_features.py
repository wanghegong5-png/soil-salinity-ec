"""Stage 3: Self-extracted satellite & terrain features from public STAC APIs.

Pulls data directly from Microsoft Planetary Computer (no account/key required):
  1. Copernicus DEM 30m (cop-dem-glo-30):
     - dem_elev (m), dem_slope (deg), dem_tpi_300m (m), dem_tpi_1500m (m)
  2. Landsat-8 Collection 2 Level-2 (landsat-c2-l2) cloud-free autumn bare-soil scenes
     across the sampling years (2013, 2014, 2018, 2019):
     - Surface reflectance: SR_B2 (Blue), SR_B3 (Green), SR_B4 (Red),
       SR_B5 (NIR), SR_B6 (SWIR1), SR_B7 (SWIR2)
     - Relative surface temperature anomaly within each scene: ST_rel_K
       (removes between-date seasonal temperature confounding while keeping
       evaporative cooling signal)
     - Spectral salinity/moisture/vegetation indices:
       NDVI, SIT_l8, SI1, SI2, SI3, BI, NDWI, SWIR_ratio

Saves:
  data/processed/samples_with_satellite.csv
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "processed" / "samples_clean.csv"
OUT_CSV = ROOT / "data" / "processed" / "samples_with_satellite.csv"
BBOX = [123.0, 44.4, 124.4, 46.0]

# Cloud-free (<3% cloud) bare-soil harvest window scenes covering paths 119 & 120, rows 028 & 029
SCENE_IDS = [
    # 2013 (proxy for 2012/2013 bare soil)
    "LC08_L2SP_119028_20131007_02_T1",
    "LC08_L2SP_119029_20131007_02_T1",
    "LC08_L2SP_120028_20131030_02_T1",
    "LC08_L2SP_120029_20131030_02_T1",
    # 2014
    "LC08_L2SP_119028_20140924_02_T1",
    "LC08_L2SP_119029_20140924_02_T1",
    "LC08_L2SP_120028_20141102_02_T1",
    "LC08_L2SP_120029_20141102_02_T1",
    # 2018
    "LC08_L2SP_119028_20181005_02_T1",
    "LC08_L2SP_119029_20181005_02_T1",
    "LC08_L2SP_120028_20181012_02_T1",
    "LC08_L2SP_120029_20181012_02_T1",
    # 2019
    "LC08_L2SP_119028_20191008_02_T1",
    "LC08_L2SP_119029_20191008_02_T1",
    "LC08_L2SP_120028_20191031_02_T1",
    "LC08_L2SP_120029_20191031_02_T1",
]
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "lwir11", "qa_pixel"]


def extract_dem(cat: pystac_client.Client, lons: np.ndarray, lats: np.ndarray) -> pd.DataFrame:
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=BBOX).items())
    n = len(lons)
    elev = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    tpi300 = np.full(n, np.nan)
    tpi1500 = np.full(n, np.nan)

    for it in items:
        with rasterio.open(it.assets["data"].href) as src:
            b = src.bounds
            m = (lons >= b.left) & (lons <= b.right) & (lats >= b.bottom) & (lats <= b.top)
            if not m.any():
                continue
            idx = np.where(m)[0]
            rc = [src.index(lons[i], lats[i]) for i in idx]
            rows = np.array([r for r, _ in rc])
            cols = np.array([c for _, c in rc])
            pad = 60  # 60 px * 30 m = 1800 m margin
            r0 = max(0, int(rows.min() - pad))
            r1 = min(src.height, int(rows.max() + pad + 1))
            c0 = max(0, int(cols.min() - pad))
            c1 = min(src.width, int(cols.max() + pad + 1))
            arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
            gy, gx = np.gradient(arr, 30.0, 21.0)  # ~30m NS, ~21m EW at 45N
            slp = np.degrees(np.arctan(np.hypot( gx, gy)))
            mean300 = uniform_filter(arr, size=21, mode="nearest")    # ~600m box (+/-300m)
            mean1500 = uniform_filter(arr, size=101, mode="nearest")  # ~3km box (+/-1500m)
            rr = rows - r0
            cc = cols - c0
            elev[idx] = arr[rr, cc]
            slope[idx] = slp[rr, cc]
            tpi300[idx] = arr[rr, cc] - mean300[rr, cc]
            tpi1500[idx] = arr[rr, cc] - mean1500[rr, cc]

    return pd.DataFrame({
        "dem_elev": elev,
        "dem_slope": slope,
        "dem_tpi_300m": tpi300,
        "dem_tpi_1500m": tpi1500,
    })


def _sample_band(href: str, pts_sorted: list[tuple[float, float]]) -> np.ndarray:
    with rasterio.open(href) as src:
        return np.array([v[0] for v in src.sample(pts_sorted)], dtype=np.float64)


def extract_landsat(cat: pystac_client.Client, lons: np.ndarray, lats: np.ndarray) -> pd.DataFrame:
    items = list(cat.search(collections=["landsat-c2-l2"], ids=SCENE_IDS).items())
    print(f"Found {len(items)}/{len(SCENE_IDS)} target Landsat-8 scenes")
    n = len(lons)
    sr_names = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    accum = {k: [[] for _ in range(n)] for k in sr_names + ["st_rel"]}

    for it in items:
        t0 = time.time()
        with rasterio.open(it.assets["red"].href) as src:
            xs, ys = transform("EPSG:4326", src.crs, lons, lats)
            xs, ys = np.asarray(xs), np.asarray(ys)
            b = src.bounds
            m = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            rc = [src.index(xs[i], ys[i]) for i in idx]
            order = np.lexsort(([c for _, c in rc], [r // 256 for r, _ in rc]))
            idx_sorted = idx[order]
            pts_sorted = [(xs[i], ys[i]) for i in idx_sorted]

        hrefs = [it.assets[bname].href for bname in BANDS]
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = dict(zip(BANDS, ex.map(lambda h: _sample_band(h, pts_sorted), hrefs)))

        qa = res["qa_pixel"].astype(np.uint16)
        # Landsat C2 QA_PIXEL: bit 0 fill, bit 1 dilated cloud, bit 2 cirrus, bit 3 cloud, bit 4 shadow, bit 5 snow
        clear = (qa & 0b111111) == 0
        valid = clear & (res["red"] > 0) & (res["nir08"] > 0) & (res["lwir11"] > 0)
        if valid.sum() < 10:
            continue

        # Convert Landsat C2 L2 digital numbers to physical units
        # Surface reflectance: DN * 0.0000275 - 0.2
        # Surface temperature (K): DN * 0.00341802 + 149.0
        st_k = res["lwir11"] * 0.00341802 + 149.0
        st_rel = st_k - np.median(st_k[valid])  # scene-relative ΔT (K)

        for k in sr_names:
            sr = np.clip(res[k] * 0.0000275 - 0.2, 0.001, 1.0)
            for pos, orig_i in enumerate(idx_sorted):
                if valid[pos]:
                    accum[k][orig_i].append(float(sr[pos]))
        for pos, orig_i in enumerate(idx_sorted):
            if valid[pos]:
                accum["st_rel"][orig_i].append(float(st_rel[pos]))
        print(f"  {it.id}: {valid.sum()} clear pts ({time.time() - t0:.1f}s)")

    out = {}
    for k in sr_names + ["st_rel"]:
        out[f"l8_{k}"] = np.array([np.median(v) if len(v) else np.nan for v in accum[k]])
    out["l8_n_obs"] = np.array([len(v) for v in accum["red"]])

    d = pd.DataFrame(out)
    B, G, R, N, S1, S2 = d["l8_blue"], d["l8_green"], d["l8_red"], d["l8_nir08"], d["l8_swir16"], d["l8_swir22"]
    d["l8_NDVI"] = (N - R) / (N + R)
    d["l8_SIT"] = R / N
    d["l8_SI1"] = np.sqrt(B * R)
    d["l8_SI2"] = np.sqrt(G**2 + R**2 + N**2)
    d["l8_SI3"] = np.sqrt(G**2 + R**2)
    d["l8_BI"] = np.sqrt(R**2 + N**2)
    d["l8_NDWI"] = (G - N) / (G + N)
    d["l8_SWIR_ratio"] = (S1 - S2) / (S1 + S2)
    return d


def main() -> None:
    df = pd.read_csv(IN_CSV)
    lons = df["Lon"].to_numpy()
    lats = df["Lat"].to_numpy()
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print("1) Extracting Copernicus 30m DEM & terrain indices...")
    dem_df = extract_dem(cat, lons, lats)
    print(dem_df.describe().round(3))

    print("\n2) Extracting Landsat-8 bare-soil surface reflectance & indices...")
    l8_df = extract_landsat(cat, lons, lats)
    print(l8_df.describe().round(3))

    merged = pd.concat([df, dem_df, l8_df], axis=1)
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows x {merged.shape[1]} cols -> {OUT_CSV}")


if __name__ == "__main__":
    main()
