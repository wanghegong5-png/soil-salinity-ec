"""Stage 4: Generate 100m regional soil salinity & risk map (GeoTIFF + portfolio figure).

Target mapping region:
  Core Western Songnen Plain sampling zone (Baicheng / Zhenlai / Da'an)
  Lon: 123.35°E – 124.15°E, Lat: 45.25°N – 45.60°N (~63 km x 39 km, 667 field samples)
  Grid resolution: 0.001 deg (~100 m), 350 rows x 800 cols

Inputs (pulled on the fly via Planetary Computer STAC API):
  - Copernicus 30m DEM (N45E123, N45E124) -> dem_elev, dem_slope, dem_tpi_300m, dem_tpi_1500m
  - Cloud-free autumn bare-soil Landsat-8 Collection 2 Level-2 composite
    (LC08_L2SP_119029_20181005_02_T1 + LC08_L2SP_120028_20181012_02_T1)

Outputs:
  - reports/salinity_risk_map_100m.tif (2-band GeoTIFF: Band 1 = Predicted EC1:5 dS/m,
    Band 2 = High-Salinity Risk Probability P(ECe >= 8 dS/m))
  - reports/figures/regional_salinity_risk_map.png (3-panel portfolio hero figure)
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "processed" / "samples_with_satellite.csv"
OUT_TIF = ROOT / "reports" / "salinity_risk_map_100m.tif"
OUT_PNG = ROOT / "reports" / "figures" / "regional_salinity_risk_map.png"

WEST, SOUTH, EAST, NORTH = 123.35, 45.25, 124.15, 45.60
WIDTH, HEIGHT = 800, 350  # 0.001 deg (~100m) grid
DST_CRS = "EPSG:4326"
DST_TRANSFORM = from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)
SEED = 42

L8_FEATURES = [
    "l8_blue", "l8_green", "l8_red", "l8_nir08", "l8_swir16", "l8_swir22",
    "l8_st_rel", "l8_NDVI", "l8_SIT", "l8_SI1", "l8_SI2", "l8_SI3",
    "l8_BI", "l8_NDWI", "l8_SWIR_ratio",
]
DEM_FEATURES = ["dem_elev", "dem_slope", "dem_tpi_300m", "dem_tpi_1500m"]
FEATURES = L8_FEATURES + DEM_FEATURES
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "lwir11", "qa_pixel"]
MAP_SCENES = ["LC08_L2SP_119029_20181005_02_T1", "LC08_L2SP_120028_20181012_02_T1"]


def _warp_asset(href: str, resampling=Resampling.bilinear) -> np.ndarray:
    dst = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float64)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=DST_TRANSFORM,
            dst_crs=DST_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def build_dem_grid(cat: pystac_client.Client) -> dict[str, np.ndarray]:
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=[WEST, SOUTH, EAST, NORTH]).items())
    layers = [_warp_asset(it.assets["data"].href, Resampling.bilinear) for it in items]
    elev = np.nanmean(np.stack(layers, axis=0), axis=0)
    # 1 pixel is ~111m NS x ~78m EW
    gy, gx = np.gradient(elev, 111.0, 78.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    tpi300 = elev - uniform_filter(elev, size=7, mode="nearest")
    tpi1500 = elev - uniform_filter(elev, size=31, mode="nearest")
    return {
        "dem_elev": elev,
        "dem_slope": slope,
        "dem_tpi_300m": tpi300,
        "dem_tpi_1500m": tpi1500,
    }


def build_landsat_grid(cat: pystac_client.Client) -> dict[str, np.ndarray]:
    items = list(cat.search(collections=["landsat-c2-l2"], ids=MAP_SCENES).items())
    sr_names = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    stack = {k: [] for k in sr_names + ["st_rel", "water"]}

    for it in items:
        def _load(bname: str):
            resamp = Resampling.nearest if bname == "qa_pixel" else Resampling.bilinear
            return bname, _warp_asset(it.assets[bname].href, resamp)

        with ThreadPoolExecutor(max_workers=8) as ex:
            raw = dict(ex.map(_load, BANDS))

        qa = np.nan_to_num(raw["qa_pixel"], nan=1.0).astype(np.uint16)
        clear = (qa & 0b111111) == 0
        water = (qa & (1 << 7)) != 0
        valid = clear & (raw["red"] > 0) & (raw["nir08"] > 0) & (raw["lwir11"] > 0)

        st_k = raw["lwir11"] * 0.00341802 + 149.0
        med_st = np.nanmedian(st_k[valid])
        st_rel = np.where(valid, st_k - med_st, np.nan)
        stack["st_rel"].append(st_rel)
        stack["water"].append(water.astype(np.float64))

        for k in sr_names:
            sr = np.clip(raw[k] * 0.0000275 - 0.2, 0.001, 1.0)
            stack[k].append(np.where(valid, sr, np.nan))

    out = {f"l8_{k}": np.nanmedian(np.stack(stack[k], axis=0), axis=0) for k in sr_names + ["st_rel"]}
    out["water_mask"] = np.nanmean(np.stack(stack["water"], axis=0), axis=0) > 0.5
    B, G, R, N, S1, S2 = out["l8_blue"], out["l8_green"], out["l8_red"], out["l8_nir08"], out["l8_swir16"], out["l8_swir22"]
    out["l8_NDVI"] = (N - R) / (N + R)
    out["l8_SIT"] = R / N
    out["l8_SI1"] = np.sqrt(B * R)
    out["l8_SI2"] = np.sqrt(G**2 + R**2 + N**2)
    out["l8_SI3"] = np.sqrt(G**2 + R**2)
    out["l8_BI"] = np.sqrt(R**2 + N**2)
    out["l8_NDWI"] = (G - N) / (G + N)
    out["l8_SWIR_ratio"] = (S1 - S2) / (S1 + S2)
    return out


def main() -> None:
    t0 = time.time()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    # Train models on all 942 points using self-extracted Landsat-8 + Copernicus DEM features
    X_train = df[FEATURES].to_numpy()
    y_ln = np.log(df["EC"].to_numpy())
    y_high = (df["EC"].to_numpy() >= 0.753).astype(int)  # USDA ECe >= 8 dS/m

    reg = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1).fit(X_train, y_ln)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=8, random_state=SEED, n_jobs=-1).fit(X_train, y_high)

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    print("Warping Copernicus DEM & Landsat-8 grids onto 100m regional raster...")
    dem_grid = build_dem_grid(cat)
    l8_grid = build_landsat_grid(cat)
    grid_all = {**l8_grid, **dem_grid}

    # Build feature matrix for all 350 x 800 = 280,000 pixels
    X_grid = np.column_stack([grid_all[f].ravel() for f in FEATURES])
    valid_px = ~np.isnan(X_grid).any(axis=1) & (~grid_all["water_mask"].ravel())
    # Also mask open water where NDWI > 0.1
    valid_px &= (grid_all["l8_NDWI"].ravel() <= 0.1)

    pred_ec = np.full(X_grid.shape[0], np.nan, dtype=np.float32)
    prob_high = np.full(X_grid.shape[0], np.nan, dtype=np.float32)
    pred_ec[valid_px] = np.exp(reg.predict(X_grid[valid_px])).astype(np.float32)
    prob_high[valid_px] = clf.predict_proba(X_grid[valid_px])[:, 1].astype(np.float32)

    ec_map = pred_ec.reshape(HEIGHT, WIDTH)
    risk_map = prob_high.reshape(HEIGHT, WIDTH)

    # Save 2-band GeoTIFF
    with rasterio.open(
        OUT_TIF, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=2,
        dtype="float32", crs=DST_CRS, transform=DST_TRANSFORM, nodata=np.nan, compress="deflate"
    ) as dst:
        dst.write(ec_map, 1); dst.set_band_description(1, "Predicted_EC1_5_dS_m")
        dst.write(risk_map, 2); dst.set_band_description(2, "High_Salinity_Risk_Prob_ECe_ge_8")

    # Filter ground-truth points inside bbox for overlay
    sub = df[(df["Lon"] >= WEST) & (df["Lon"] <= EAST) & (df["Lat"] >= SOUTH) & (df["Lat"] <= NORTH)]
    extent = [WEST, EAST, SOUTH, NORTH]

    # Plot 3-panel hero figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)

    # Panel A: Copernicus 30m DEM elevation + sample locations
    im0 = axes[0].imshow(dem_grid["dem_elev"], extent=extent, origin="upper", cmap="terrain", aspect="auto")
    axes[0].scatter(sub["Lon"], sub["Lat"], c="k", s=8, alpha=0.5, label=f"Samples (n={len(sub)})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Elevation (m)")
    axes[0].set_title("(a) Copernicus DEM (30m) & Field Samples")
    axes[0].set_xlabel("Longitude (°E)"); axes[0].set_ylabel("Latitude (°N)")
    axes[0].legend(loc="lower left", fontsize=8)

    # Panel B: Predicted EC1:5 (dS/m) with ground-truth points colored on same scale
    vmin, vmax = 0.2, 2.5
    im1 = axes[1].imshow(ec_map, extent=extent, origin="upper", cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].scatter(
        sub["Lon"], sub["Lat"], c=sub["EC"], cmap="YlOrRd", vmin=vmin, vmax=vmax,
        s=18, edgecolors="k", linewidths=0.4, label="Observed EC"
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Soil EC1:5 (dS/m)")
    axes[1].set_title("(b) Predicted EC1:5 (Landsat-8 + DEM) & Ground Truth")
    axes[1].set_xlabel("Longitude (°E)")

    # Panel C: Moderate-to-High Salinity Risk Screening Map P(ECe >= 8 dS/m)
    im2 = axes[2].imshow(risk_map, extent=extent, origin="upper", cmap="RdYlBu_r", vmin=0.1, vmax=0.8, aspect="auto")
    high_pts = sub[sub["EC"] >= 0.753]
    axes[2].scatter(high_pts["Lon"], high_pts["Lat"], c="none", edgecolors="black", s=22, linewidths=0.7,
                    label=f"Observed ECe >= 8 (n={len(high_pts)})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="P(ECe >= 8 dS/m)")
    axes[2].set_title("(c) High-Salinity Risk Screening Map (100m)")
    axes[2].set_xlabel("Longitude (°E)")
    axes[2].legend(loc="lower left", fontsize=8)

    fig.suptitle("Western Songnen Plain Soil Salinity Risk Mapping (100m, Unconfounded Landsat-8 + Copernicus DEM)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Done in {time.time() - t0:.1f}s -> {OUT_TIF.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
