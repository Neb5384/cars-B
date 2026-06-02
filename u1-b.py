import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import stats

# Load CSV
df = pd.read_csv("Car_Dataset_1945-2020.csv")

df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(df["mixed_fuel_consumption_per_100_km_l"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"], errors="coerce")

df = df.dropna(subset=["Year_from", "full_weight_kg", "mixed_fuel_consumption_per_100_km_l"])
df["Year_from"] = df["Year_from"].astype(int)

# Fixed axis limits
x_min, x_max = df["full_weight_kg"].quantile(0.01), df["full_weight_kg"].quantile(0.99)
y_min, y_max = df["mixed_fuel_consumption_per_100_km_l"].quantile(0.01), df["mixed_fuel_consumption_per_100_km_l"].quantile(0.99)

# Overall LSE fit (fixed reference line)
x_all  = df["full_weight_kg"].values
y_all  = df["mixed_fuel_consumption_per_100_km_l"].values
c_all  = np.polyfit(x_all, y_all, 1)
x_line = np.linspace(x_min, x_max, 300)

# Years & window
years       = sorted(df["Year_from"].unique())
start_years = years[:-4]
cmap        = plt.cm.plasma
year_min, year_max = years[0], years[-1]

def year_color(yr):
    return cmap((yr - year_min) / (year_max - year_min))

ALPHAS = [0.12, 0.35, 0.65, 0.35, 0.12]

# Pre-compute ALL window means upfront (indexed by frame)
all_means = []
for y0 in start_years:
    subset = df[df["Year_from"].isin([y0 + i for i in range(5)])]
    all_means.append((subset["full_weight_kg"].mean(),
                      subset["mixed_fuel_consumption_per_100_km_l"].mean()))

# ── Figure ──
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.82)
ax = fig.add_subplot(111)

ax.plot(x_line, np.poly1d(c_all)(x_line), color="crimson", linewidth=2,
        zorder=5, label=f"Overall LSE (slope={c_all[0]:+.5f})")

ax.set_xlim(x_min - 50, x_max + 50)
ax.set_ylim(y_min - 1,  y_max + 1)
ax.set_xlabel("Full Weight (kg)")
ax.set_ylabel("Mixed Fuel Consumption (L/100 km)")
ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=year_min, vmax=year_max))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Year", pad=0.02)

scatters   = [ax.scatter([], [], s=12, zorder=3, color=cmap(0)) for _ in range(5)]
trail_line, = ax.plot([], [], color="black", linewidth=1.2, zorder=6, alpha=0.6)
mean_dot    = ax.scatter([], [], s=80, color="limegreen", zorder=8,
                         edgecolors="darkgreen", linewidths=1.2, label="Window mean")

ax.legend(loc="upper left", fontsize=8)

year_label = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                     ha="right", va="top", fontsize=15, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
stats_text = ax.text(0.98, 0.05, "", transform=ax.transAxes,
                     ha="right", va="bottom", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

fig.suptitle("Fuel Consumption vs Vehicle Weight", fontsize=13, fontweight="bold")


def update(frame_idx):
    y0     = start_years[frame_idx]
    window = [y0 + i for i in range(5)]

    # Scatter clouds
    all_x, all_y = [], []
    for i, yr in enumerate(window):
        subset = df[df["Year_from"] == yr]
        x = subset["full_weight_kg"].values
        y = subset["mixed_fuel_consumption_per_100_km_l"].values
        scatters[i].set_offsets(np.c_[x, y] if len(x) else np.empty((0, 2)))
        scatters[i].set_color(year_color(yr))
        scatters[i].set_alpha(ALPHAS[i])
        all_x.extend(x); all_y.extend(y)

    # Stats
    if len(all_x) >= 2:
        r, p = stats.pearsonr(all_x, all_y)
        stats_text.set_text(f"r = {r:.3f},  p = {p:.2e}\nn = {len(all_x)}")
    else:
        stats_text.set_text(f"n = {len(all_x)}")

    # Trail: only means up to and including this frame
    trail_xs = [all_means[i][0] for i in range(frame_idx + 1)]
    trail_ys = [all_means[i][1] for i in range(frame_idx + 1)]
    trail_line.set_data(trail_xs, trail_ys)

    # Current mean dot
    mx, my = all_means[frame_idx]
    mean_dot.set_offsets([[mx, my]])

    year_label.set_text(f"{window[0]} – {window[4]}")
    return (*scatters, trail_line, mean_dot, year_label, stats_text)


ani = animation.FuncAnimation(
    fig, update,
    frames=len(start_years),
    interval=150,
    blit=True,
    repeat=False
)

ani.save("fuel_vs_weight_animation.gif", writer="pillow", fps=8, dpi=120)
print("Saved: fuel_vs_weight_animation.gif")

plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.show()