import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Load & clean ─────────────────────────────────────────────────────────────
df = pd.read_csv("Car_Dataset_1945-2020.csv")
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"], errors="coerce")
df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(
    df["mixed_fuel_consumption_per_100_km_l"], errors="coerce"
)
df = df[df["Year_from"] >= 1980]

# ── Series 1: mean weight per year ───────────────────────────────────────────
yr_wt = (
    df.dropna(subset=["Year_from", "full_weight_kg"])
    .groupby("Year_from")["full_weight_kg"]
    .mean()
    .reset_index()
)
yr_wt.columns = ["year", "mean_weight"]

# ── Series 2: mean consumption per weight bucket (50 kg bins) ────────────────
df_wf = df.dropna(subset=["full_weight_kg", "mixed_fuel_consumption_per_100_km_l"])
df_wf = df_wf[
    (df_wf["full_weight_kg"].between(
        df_wf["full_weight_kg"].quantile(0.02),
        df_wf["full_weight_kg"].quantile(0.98)
    ))
]
df_wf["weight_bin"] = (df_wf["full_weight_kg"] // 50) * 50
wt_fuel = (
    df_wf.groupby("weight_bin")["mixed_fuel_consumption_per_100_km_l"]
    .mean()
    .reset_index()
)
wt_fuel.columns = ["weight", "mean_fuel"]

# ── LSE fits ─────────────────────────────────────────────────────────────────
c_yw  = np.polyfit(yr_wt["year"],       yr_wt["mean_weight"],  1)
c_wf  = np.polyfit(wt_fuel["weight"],   wt_fuel["mean_fuel"],  1)

r_yw, p_yw = stats.pearsonr(yr_wt["year"],     yr_wt["mean_weight"])
r_wf, p_wf = stats.pearsonr(wt_fuel["weight"], wt_fuel["mean_fuel"])

# ── Figure: two subplots sharing the weight axis in the middle ───────────────
# Layout: [year | weight] [weight | fuel]
# We flip the left plot so weight is on the RIGHT and year on the LEFT.

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, wspace=0.0)   # zero gap between panels

ax_left  = fig.add_subplot(gs[0])   # year (x) vs weight (y)  — will be flipped
ax_right = fig.add_subplot(gs[1])   # weight (x) vs fuel (y)

# ── LEFT panel: Year → Weight (x=year, y=weight; we'll invert x later) ───────
ax_left.scatter(yr_wt["year"], yr_wt["mean_weight"],
                s=22, color="#1d4ed8", alpha=0.7, zorder=3, linewidths=0)
x_l = np.linspace(yr_wt["year"].min(), yr_wt["year"].max(), 300)
ax_left.plot(x_l, np.poly1d(c_yw)(x_l), color="crimson", lw=2, zorder=4)
ax_left.set_xlabel("Year", fontsize=11)
ax_left.set_ylabel("Mean Full Weight (kg)", fontsize=11)
ax_left.set_title(
    f"Year  →  Weight\nr={r_yw:.3f}, p={p_yw:.1e}",
    fontsize=11, pad=8
)
ax_left.grid(True, alpha=0.25)
ax_left.invert_xaxis()   # weight axis is on the RIGHT edge, mirrored into right panel

# ── RIGHT panel: Weight → Fuel consumption ───────────────────────────────────
ax_right.scatter(wt_fuel["weight"], wt_fuel["mean_fuel"],
                 s=22, color="#f59e0b", alpha=0.8, zorder=3, linewidths=0)
x_r = np.linspace(wt_fuel["weight"].min(), wt_fuel["weight"].max(), 300)
ax_right.plot(x_r, np.poly1d(c_wf)(x_r), color="crimson", lw=2, zorder=4)
ax_right.set_xlabel("Mean Full Weight (kg)", fontsize=11)
ax_right.set_ylabel("Mean Fuel Consumption (L/100 km)", fontsize=11)
ax_right.set_title(
    f"Weight  →  Fuel Consumption\nr={r_wf:.3f}, p={p_wf:.1e}",
    fontsize=11, pad=8
)
ax_right.grid(True, alpha=0.25)
ax_right.yaxis.set_label_position("right")
ax_right.yaxis.tick_right()

# ── Align the shared weight axis limits ──────────────────────────────────────
w_min = min(yr_wt["mean_weight"].min(), wt_fuel["weight"].min())
w_max = max(yr_wt["mean_weight"].max(), wt_fuel["weight"].max())
pad   = (w_max - w_min) * 0.05
ax_left.set_ylim(w_min - pad, w_max + pad)
ax_right.set_xlim(w_min - pad, w_max + pad)

# ── Shared weight axis label in the middle ────────────────────────────────────
fig.text(0.5, 0.01, "← shared weight axis (kg) →",
         ha="center", va="bottom", fontsize=10, color="#555555", style="italic")

fig.suptitle(
    "Cars got heavier over time  ·  heavier cars consume more fuel",
    fontsize=13, fontweight="bold", y=1.02
)

plt.savefig("chain_year_weight_fuel.png", dpi=150, bbox_inches="tight")
print("Saved: chain_year_weight_fuel.png")