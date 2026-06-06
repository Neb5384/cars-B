"""
Choropleth GIF — Passenger cars per 1 000 inhabitants, Europe 2000-2024
With inset line plot of total passenger cars (sum across countries with full data).
Dependencies: pandas, matplotlib, Pillow  (no geopandas, no shapely)
"""

import json
import os
import glob
import io
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────────
CSV_PATH      = "road_eqs_carhab_linear_2_0.csv"
GEOJSON_URL   = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
    "/master/geojson/ne_110m_admin_0_countries.geojson"
)
GEOJSON_LOCAL = "ne_110m_admin_0_countries.geojson"
POP_URL       = "https://raw.githubusercontent.com/datasets/population/main/data/population.csv"
OUTPUT_GIF    = "cars_per_1000_europe.gif"
FRAMES_DIR    = "frames"
START_YEAR    = 2000
FRAME_MS      = 700   # ms per frame
HOLD_FRAMES   = 4     # extra copies of last frame
VMIN, VMAX    = 100, 800
CMAP          = plt.cm.YlOrRd
BBOX          = (-25, 45, 34, 72)   # lon_min, lon_max, lat_min, lat_max
BG            = "#1a1a2e"
NO_DATA_CLR   = "#3a3a5c"
BORDER_CLR    = "#555577"
DPI           = 130

# iso2 → iso3 for World Bank population data
ISO2_TO_ISO3 = {
    "AL":"ALB","AT":"AUT","BA":"BIH","BE":"BEL","BG":"BGR","CH":"CHE",
    "CY":"CYP","CZ":"CZE","DE":"DEU","DK":"DNK","EE":"EST","GR":"GRC",
    "ES":"ESP","FI":"FIN","FR":"FRA","GE":"GEO","HR":"HRV","HU":"HUN",
    "IE":"IRL","IS":"ISL","IT":"ITA","LT":"LTU","LU":"LUX","LV":"LVA",
    "MD":"MDA","ME":"MNE","MK":"MKD","MT":"MLT","NL":"NLD","NO":"NOR",
    "PL":"POL","PT":"PRT","RO":"ROU","RS":"SRB","SE":"SWE","SI":"SVN",
    "SK":"SVK","TR":"TUR","UA":"UKR","GB":"GBR",
}

# ── 1. Load & clean Eurostat data ───────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
df = df[~df["geo"].isin(["EU27_2020", "LI"])]   # drop EU aggregate & Liechtenstein
df = df[df["TIME_PERIOD"] >= START_YEAR]
df["iso2"] = df["geo"].replace({"EL": "GR", "UK": "GB"})  # align with Natural Earth codes

# ── DATA CLEANING ────────────────────────────────────────────────────────────────
# Build a full (country × year) grid and interpolate gaps linearly.
# Only interior gaps are filled; leading/trailing NaNs are left as-is
# (a country with no data at all stays invisible).
all_years     = list(range(df["TIME_PERIOD"].min(), df["TIME_PERIOD"].max() + 1))
all_countries = df["iso2"].unique()
full_index    = pd.MultiIndex.from_product([all_countries, all_years],
                                            names=["iso2", "TIME_PERIOD"])
df = (
    df.set_index(["iso2", "TIME_PERIOD"])["OBS_VALUE"]
      .reindex(full_index)
      .groupby(level="iso2")
      .transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
      .reset_index()
)
df = df.dropna(subset=["OBS_VALUE"])             # drop still-empty edges
# ── END DATA CLEANING ────────────────────────────────────────────────────────────

years = sorted(df["TIME_PERIOD"].unique())
print(f"  {years[0]}-{years[-1]}, {df['iso2'].nunique()} countries")

# ── 2. Population data → total cars time series ─────────────────────────────────
print("Fetching population data...")
req = urllib.request.Request(POP_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=20) as r:
    pop = pd.read_csv(io.StringIO(r.read().decode()))
pop = pop[pop["Year"].between(START_YEAR, years[-1])][["Country Code", "Year", "Value"]].copy()
pop.columns = ["iso3", "year", "population"]

df["iso3"] = df["iso2"].map(ISO2_TO_ISO3)
merged = df.merge(pop, left_on=["iso3", "TIME_PERIOD"], right_on=["iso3", "year"], how="inner")

# Keep only countries that have data for every year in the range
n_years   = len(years)
coverage  = merged.groupby("iso2")["TIME_PERIOD"].count()
full_iso2 = coverage[coverage == n_years].index.tolist()
print(f"  Countries with full {START_YEAR}-{years[-1]} coverage: {len(full_iso2)}")

merged_full = merged[merged["iso2"].isin(full_iso2)].copy()
merged_full["total_cars"] = merged_full["OBS_VALUE"] / 1000 * merged_full["population"]
total_cars_ts = (
    merged_full.groupby("TIME_PERIOD")["total_cars"].sum() / 1e6
)  # in millions

# ── 3. Download GeoJSON if needed ───────────────────────────────────────────────
if not os.path.exists(GEOJSON_LOCAL):
    print(f"Downloading {GEOJSON_URL} ...")
    urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_LOCAL)

# ── 4. Parse GeoJSON → per-country polygon rings ───────────────────────────────
def geom_to_rings(geometry):
    """Yield (exterior_coords, [hole_coords, ...]) for every polygon in a GeoJSON geometry."""
    t = geometry["type"]
    if t == "Polygon":
        yield geometry["coordinates"][0], geometry["coordinates"][1:]
    elif t == "MultiPolygon":
        for part in geometry["coordinates"]:
            yield part[0], part[1:]

print("Parsing GeoJSON...")
with open(GEOJSON_LOCAL) as f:
    gj = json.load(f)

country_polys: dict[str, list] = {}
for feat in gj["features"]:
    p = feat["properties"]
    if p.get("CONTINENT") != "Europe" and p.get("NAME") != "Turkey":
        continue
    iso = p.get("ISO_A2_EH", "")
    country_polys[iso] = list(geom_to_rings(feat["geometry"]))

print(f"  {len(country_polys)} countries/territories loaded")

# ── 5. Draw a single frame (map left, line plot right) ──────────────────────────
def draw_frame(year: int, val_map: dict, path: str) -> None:
    norm = Normalize(vmin=VMIN, vmax=VMAX)

    # Wide figure: map on left (~70%), line plot on right (~30%)
    fig, (ax, ax_line) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        facecolor=BG,
        gridspec_kw={"width_ratios": [75, 25], "wspace": 0.15},
    )

    # ── Map ────────────────────────────────────────────────────────────────────
    ax.set_facecolor(BG)
    ax.set_xlim(BBOX[0], BBOX[1])
    ax.set_ylim(BBOX[2], BBOX[3])
    ax.set_aspect("equal")
    ax.set_axis_off()

    for iso, rings in country_polys.items():
        val  = val_map.get(iso)
        face = CMAP(norm(val)) if val is not None else mcolors.to_rgba(NO_DATA_CLR)
        for ext, holes in rings:
            ax.add_patch(MplPolygon(
                ext, closed=True, facecolor=face,
                edgecolor=BORDER_CLR, linewidth=0.5, zorder=1,
            ))
            for hole in holes:
                ax.add_patch(MplPolygon(
                    hole, closed=True, facecolor=BG,
                    edgecolor=BORDER_CLR, linewidth=0.3, zorder=2,
                ))

    # Colorbar beneath map
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.01, shrink=0.7)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.set_label("Passenger cars per 1 000 inhabitants", color="white", fontsize=10)
    cbar.outline.set_edgecolor("white")

    # Year label
    ax.text(0.5, 0.97, str(year), transform=ax.transAxes,
            ha="center", va="top", fontsize=42, fontweight="bold", color="white", alpha=0.9)

    # ── Line plot ──────────────────────────────────────────────────────────────
    ax_line.set_facecolor("#12122a")
    for spine in ax_line.spines.values():
        spine.set_edgecolor("#444466")

    ts_years  = list(total_cars_ts.index)
    ts_values = list(total_cars_ts.values)
    idx_now   = ts_years.index(year) + 1

    y_pad = (max(ts_values) - min(ts_values)) * 0.08

    # Full faint trace
    ax_line.plot(ts_years, ts_values, color="#aaaacc", linewidth=1, alpha=0.25)
    # Progress line
    ax_line.plot(ts_years[:idx_now], ts_values[:idx_now],
                 color="#f4a261", linewidth=2.5, solid_capstyle="round")
    # Current dot
    ax_line.scatter([year], [ts_values[idx_now - 1]], color="#f4a261", s=60, zorder=5)
    # Value label
    ax_line.text(year, ts_values[idx_now - 1] + y_pad,
                 f"{ts_values[idx_now - 1]:.0f}M",
                 color="#f4a261", fontsize=10, ha="center", va="bottom", fontweight="bold")

    ax_line.set_xlim(ts_years[0] - 0.5, ts_years[-1] + 0.5)
    ax_line.set_ylim(min(ts_values) - y_pad, max(ts_values) + y_pad * 3)
    ax_line.tick_params(colors="#aaaacc", labelsize=9)
    ax_line.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax_line.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax_line.grid(axis="y", color="#2a2a4a", linewidth=0.8, linestyle="--")
    ax_line.set_title(
        f"Total passenger cars\n({len(full_iso2)} countries)",
        color="#ccccee", fontsize=10, pad=10, linespacing=1.5,
    )
    ax_line.set_xlabel("Year", color="#aaaacc", fontsize=9)
    ax_line.set_ylabel("Passenger cars", color="#aaaacc", fontsize=9)

    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()

# ── 6. Render all frames ─────────────────────────────────────────────────────────
os.makedirs(FRAMES_DIR, exist_ok=True)
print("Rendering frames...")
for year in years:
    ydf = df[df["TIME_PERIOD"] == year]
    draw_frame(year, dict(zip(ydf["iso2"], ydf["OBS_VALUE"])),
               f"{FRAMES_DIR}/{year:04d}.png")
    print(f"  {year}")

# ── 7. Assemble GIF ──────────────────────────────────────────────────────────────
print("\nAssembling GIF...")
pngs   = sorted(glob.glob(f"{FRAMES_DIR}/*.png"))
target = Counter(Image.open(p).size for p in pngs).most_common(1)[0][0]
frames = [Image.open(p).convert("RGBA").resize(target, Image.LANCZOS) for p in pngs]
for _ in range(HOLD_FRAMES):
    frames.append(frames[-1].copy())

frames[0].save(
    OUTPUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_MS,
    loop=0,
    optimize=True,
)
print(f"Saved → {OUTPUT_GIF}")