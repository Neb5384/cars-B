import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from scipy import stats

# ── Load & clean ────────────────────────────────────────────────────────────
df = pd.read_csv("Car_Dataset_1945-2020.csv")

numeric_cols = [
    "Year_from", "full_weight_kg", "curb_weight_kg",
    "mixed_fuel_consumption_per_100_km_l",
    "CO2_emissions_g/km", "engine_hp",
    "capacity_cm3", "number_of_cylinders",
    "length_mm", "width_mm", "height_mm",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Year_from"] = df["Year_from"].dropna()
df["Year_from"] = df["Year_from"].astype("Int64")

# Derived columns
df["volume_m3"] = (df["length_mm"] / 1000) * (df["width_mm"] / 1000) * (df["height_mm"] / 1000)

# Decade label
df["Decade"] = (df["Year_from"] // 10 * 10).astype("Int64").astype(str) + "s"

# ── Helper: save or show ─────────────────────────────────────────────────────
import os
OUTPUT_DIR = "."   # change to e.g. "frames/" if you prefer

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 – PIE CHART: Fuel type share across the whole dataset
# ════════════════════════════════════════════════════════════════════════════
if "engine_type" in df.columns:
    fuel_counts = df["engine_type"].value_counts()
    # Merge tiny slices into "Other"
    threshold = 0.02 * fuel_counts.sum()
    other = fuel_counts[fuel_counts < threshold].sum()
    fuel_counts = fuel_counts[fuel_counts >= threshold]
    if other:
        fuel_counts["Other"] = other

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        fuel_counts,
        labels=fuel_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    ax.set_title("Engine / Fuel Type Share (all years)", fontsize=14, pad=18)
    plt.tight_layout()
    savefig("pie_fuel_type_share.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 – PIE CHART: Drive-wheel type share
# ════════════════════════════════════════════════════════════════════════════
if "drive_wheels" in df.columns:
    dw_counts = df["drive_wheels"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        dw_counts,
        labels=dw_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    ax.set_title("Drive-Wheel Type Distribution", fontsize=14, pad=18)
    plt.tight_layout()
    savefig("pie_drive_wheels.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 – STACKED BAR: Fuel type count per decade
# ════════════════════════════════════════════════════════════════════════════
if "engine_type" in df.columns:
    df_dec = df.dropna(subset=["Decade", "engine_type"])
    pivot = (
        df_dec.groupby(["Decade", "engine_type"])
        .size()
        .unstack(fill_value=0)
    )
    # Keep only engine types that appear in at least 2 decades
    pivot = pivot.loc[:, (pivot > 0).sum(axis=0) >= 2]

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of Models")
    ax.set_title("Engine / Fuel Type Count per Decade (stacked)", fontsize=13)
    ax.legend(title="Engine type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("stacked_bar_fuel_type_per_decade.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 – GROUPED BAR: Mean fuel consumption & mean weight per decade
# ════════════════════════════════════════════════════════════════════════════
df_decade_agg = (
    df.dropna(subset=["Decade", "mixed_fuel_consumption_per_100_km_l", "full_weight_kg"])
    .groupby("Decade", as_index=False)
    .agg(
        mean_fuel=("mixed_fuel_consumption_per_100_km_l", "mean"),
        mean_weight=("full_weight_kg", "mean"),
    )
)

fig, ax1 = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_decade_agg))
w = 0.35

bars1 = ax1.bar(x - w / 2, df_decade_agg["mean_fuel"], w, label="Mean Fuel (L/100km)", color="darkorange", alpha=0.85)
ax1.set_ylabel("Mean Fuel Consumption (L/100km)", color="darkorange")
ax1.tick_params(axis="y", labelcolor="darkorange")

ax2 = ax1.twinx()
bars2 = ax2.bar(x + w / 2, df_decade_agg["mean_weight"], w, label="Mean Full Weight (kg)", color="steelblue", alpha=0.85)
ax2.set_ylabel("Mean Full Weight (kg)", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

ax1.set_xticks(x)
ax1.set_xticklabels(df_decade_agg["Decade"], rotation=45, ha="right")
ax1.set_title("Mean Fuel Consumption vs Mean Weight per Decade", fontsize=13)
lines = [bars1, bars2]
ax1.legend(lines, [b.get_label() for b in lines], loc="upper left")
plt.tight_layout()
savefig("grouped_bar_fuel_weight_per_decade.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 – HEATMAP: Pearson correlations among key numeric columns
# ════════════════════════════════════════════════════════════════════════════
corr_cols = [
    "Year_from", "full_weight_kg", "curb_weight_kg",
    "mixed_fuel_consumption_per_100_km_l",
    "CO2_emissions_g/km", "engine_hp",
    "capacity_cm3", "number_of_cylinders", "volume_m3",
]
corr_cols = [c for c in corr_cols if c in df.columns]

corr_matrix = df[corr_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Pearson Correlation Heatmap – Key Vehicle Variables", fontsize=13)
plt.tight_layout()
savefig("heatmap_correlations.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 6 – PIE / DONUT: Transmission type share
# ════════════════════════════════════════════════════════════════════════════
if "transmission" in df.columns:
    trans_counts = df["transmission"].value_counts()
    threshold = 0.015 * trans_counts.sum()
    other = trans_counts[trans_counts < threshold].sum()
    trans_counts = trans_counts[trans_counts >= threshold]
    if other:
        trans_counts["Other"] = other

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        trans_counts,
        labels=trans_counts.index,
        autopct="%1.1f%%",
        startangle=120,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, linewidth=0.8, edgecolor="white"),  # donut
    )
    ax.set_title("Transmission Type Distribution (Donut)", fontsize=14, pad=18)
    plt.tight_layout()
    savefig("donut_transmission_type.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 7 – SCATTER with marginal histograms: Weight vs CO2
# ════════════════════════════════════════════════════════════════════════════
df_scat = df.dropna(subset=["full_weight_kg", "CO2_emissions_g/km", "Year_from"])
df_scat = df_scat[(df_scat["Year_from"] >= 1990)]  # plenty of CO2 data post-1990

g = sns.JointGrid(
    data=df_scat,
    x="full_weight_kg",
    y="CO2_emissions_g/km",
    height=8,
    ratio=4,
)
g.plot_joint(
    sns.scatterplot,
    alpha=0.25,
    s=12,
    hue=df_scat["Year_from"],
    palette="viridis",
    legend=False,
)
g.plot_marginals(sns.histplot, bins=40, color="steelblue", alpha=0.6)

# LSE line
x_vals = df_scat["full_weight_kg"].values
y_vals = df_scat["CO2_emissions_g/km"].values
coeffs = np.polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
g.ax_joint.plot(x_line, np.poly1d(coeffs)(x_line), color="crimson", lw=2, label=f"LSE slope={coeffs[0]:+.4f}")
g.ax_joint.legend(fontsize=9)

r, p = stats.pearsonr(x_vals, y_vals)
g.ax_joint.set_title(f"Weight vs CO₂  (post-1990)  r={r:.3f}, p={p:.1e}", pad=12)
g.ax_joint.set_xlabel("Full Weight (kg)")
g.ax_joint.set_ylabel("CO₂ Emissions (g/km)")
g.figure.tight_layout()
savefig("joint_weight_co2.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 8 – AREA CHART: Normalised trends (weight, fuel, CO2) over years
# ════════════════════════════════════════════════════════════════════════════
df_agg = (
    df.dropna(subset=["Year_from", "full_weight_kg",
                       "mixed_fuel_consumption_per_100_km_l",
                       "CO2_emissions_g/km"])
    .groupby("Year_from", as_index=False)
    .agg(
        weight=("full_weight_kg", "median"),
        fuel=("mixed_fuel_consumption_per_100_km_l", "median"),
        co2=("CO2_emissions_g/km", "median"),
    )
)
df_agg = df_agg[df_agg["Year_from"] >= 1950]

# Normalise each series to its value nearest to 1980 for fair comparison
base_year = 1980
available_years = df_agg["Year_from"].values
closest_year = available_years[np.argmin(np.abs(available_years - base_year))]
base_year = int(closest_year)
base = df_agg[df_agg["Year_from"] == base_year].iloc[0]
for col in ["weight", "fuel", "co2"]:
    df_agg[f"{col}_norm"] = df_agg[col] / base[col]

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(df_agg["Year_from"], df_agg["weight_norm"], alpha=0.35, color="steelblue", label="Median Weight")
ax.fill_between(df_agg["Year_from"], df_agg["fuel_norm"],   alpha=0.35, color="darkorange", label="Median Fuel Consumption")
ax.fill_between(df_agg["Year_from"], df_agg["co2_norm"],    alpha=0.35, color="mediumpurple", label="Median CO₂")
ax.plot(df_agg["Year_from"], df_agg["weight_norm"], color="steelblue",    lw=2)
ax.plot(df_agg["Year_from"], df_agg["fuel_norm"],   color="darkorange",   lw=2)
ax.plot(df_agg["Year_from"], df_agg["co2_norm"],    color="mediumpurple", lw=2)
ax.axhline(1.0, color="black", lw=0.8, ls="--", label=f"Reference ({base_year})")
ax.set_xlabel("Year")
ax.set_ylabel(f"Normalised value  (1 = {base_year} median)")
ax.set_title("Relative Trends: Weight, Fuel Consumption & CO₂ (normalised to 1980)", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig("area_normalised_trends.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 9 – EVOLVING PIE CHARTS: Vehicle size-class share per decade
#
# WHY THIS SUPPORTS THE THESIS:
#   The share of "Compact" / "Small" cars shrinks decade after decade, while
#   "Midsize", "Large", and SUV/Crossover classes grow.  This structural shift
#   in *what consumers buy* directly explains why fleet-wide fuel consumption
#   stayed stubbornly flat even as individual engines became more efficient:
#   lighter, smaller cars were progressively replaced by heavier ones.
#   The pie chart makes the substitution effect impossible to miss — each
#   slice is a vote for a heavier vehicle class.
# ════════════════════════════════════════════════════════════════════════════

# Size classes ordered light → heavy
SIZE_LABELS = [
    "Mini (<1655 kg)",
    "Small (1655–1870 kg)",
    "Medium (1870–2050 kg)",
    "Large (2050–2310 kg)",
    "Extra-Large (>2310 kg)",
]

# Single sequential palette: white → blue (light → heavy)
SIZE_COLORS = ["#d3e4f9", "#93c5fd", "#3b82f6", "#1d4ed8", "#04288D"]
color_map = dict(zip(SIZE_LABELS, SIZE_COLORS))

# Derive size bucket from full weight
df["_size_class"] = pd.cut(
    df["full_weight_kg"],
    bins=[0, 1655, 1870, 2050, 2310, 99999],
    labels=SIZE_LABELS,
)
size_col = "_size_class"

DECADES = [1980, 1990, 2000, 2010]
DECADE_LABELS = {
    1980: "1980\u20131989",
    1990: "1990\u20131999",
    2000: "2000\u20132009",
    2010: "2010\u20132019",
}

df_pie = df.dropna(subset=["Year_from", size_col]).copy()
df_pie["Decade"] = (df_pie["Year_from"] // 10 * 10).astype(int)

decade_data = {}
for d in DECADES:
    sub = df_pie[df_pie["Decade"] == d][size_col].value_counts()
    if sub.sum() >= 10:
        sub = sub.reindex(SIZE_LABELS, fill_value=0)
        sub = sub[sub > 0]
        decade_data[d] = sub

n = len(decade_data)
if n == 0:
    print("Not enough data for evolving pie charts — skipping Plot 9.")
else:
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    axes = np.array(axes).flatten()

    for ax, (decade, counts) in zip(axes, decade_data.items()):
        colors = [color_map[lbl] for lbl in counts.index]
        ax.pie(
            counts,
            labels=None,
            autopct="%1.0f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
            textprops=dict(fontsize=11, fontweight="bold", color="#ff861c"),
        )
        ax.set_title(f"{DECADE_LABELS[decade]}\n(n={counts.sum()})", fontsize=12, fontweight="bold")

    for ax in axes[len(decade_data):]:
        ax.set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[lbl], label=lbl) for lbl in SIZE_LABELS
    ]
    fig.legend(
        handles=legend_handles,
        title="Vehicle Weight Class  (light \u2192 heavy)",
        loc="lower center",
        ncol=len(SIZE_LABELS),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Vehicle Size-Class Share per Decade\n"
        "(growing heavy-vehicle share explains flat fuel consumption despite engine progress)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    savefig("pie_evolving_size_class.png")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 10 – EVOLVING PIE CHARTS: Body/car-type share per decade
#
# WHY THIS SUPPORTS THE THESIS:
#   Sedans and hatchbacks — typically the lightest body styles — lose market
#   share decade by decade, while SUVs, crossovers, and pickups — among the
#   heaviest — grow steadily.  This shift in body-type preference is the
#   consumer-behaviour story behind the rising weight trend: it is not just
#   that individual models got heavier, but that buyers migrated en masse
#   toward fundamentally heavier vehicle categories.
# ════════════════════════════════════════════════════════════════════════════

BODY_COL = None
for candidate in ["body_type", "Vehicle_Style", "vehicle_style", "Body_Type"]:
    if candidate in df.columns:
        BODY_COL = candidate
        break

if BODY_COL is None:
    print("No body-type column found — skipping Plot 10.")
else:
    df_body = df.dropna(subset=["Year_from", BODY_COL]).copy()
    df_body["Decade"] = (df_body["Year_from"] // 10 * 10).astype(int)
    df_body = df_body[df_body["Decade"].isin([1980, 1990, 2000, 2010])]

    # Find the top N most frequent body types across the whole period
    TOP_N = 6
    top_types = (
        df_body[BODY_COL]
        .value_counts()
        .head(TOP_N)
        .index.tolist()
    )

    # Recode everything outside top N as "Other"
    df_body["_body"] = df_body[BODY_COL].where(
        df_body[BODY_COL].isin(top_types), other="Other"
    )

    # Colour palette: qualitative, distinct colours per type
    TYPE_PALETTE = [
        "#2563eb", "#16a34a", "#dc2626", "#d97706",
        "#7c3aed", "#0891b2", "#9ca3af",   # last = Other
    ]
    all_types = top_types + (["Other"] if "Other" in df_body["_body"].values else [])
    type_color_map = {t: TYPE_PALETTE[i] for i, t in enumerate(all_types)}

    DECADES = [1980, 1990, 2000, 2010]
    DECADE_LABELS = {
        1980: "1980\u20131989",
        1990: "1990\u20131999",
        2000: "2000\u20132009",
        2010: "2010\u20132019",
    }

    decade_body_data = {}
    for d in DECADES:
        sub = df_body[df_body["Decade"] == d]["_body"].value_counts()
        if sub.sum() >= 10:
            # Keep consistent order: top types first, then Other
            sub = sub.reindex(all_types, fill_value=0)
            sub = sub[sub > 0]
            decade_body_data[d] = sub

    n = len(decade_body_data)
    if n == 0:
        print("Not enough body-type data for evolving pie charts — skipping Plot 10.")
    else:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
        axes = np.array(axes).flatten()

        for ax, (decade, counts) in zip(axes, decade_body_data.items()):
            colors = [type_color_map[t] for t in counts.index]
            ax.pie(
                counts,
                labels=None,
                autopct="%1.0f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.75,
                wedgeprops=dict(linewidth=0.8, edgecolor="white"),
                textprops=dict(fontsize=11, fontweight="bold", color="green"),
            )
            ax.set_title(
                f"{DECADE_LABELS[decade]}\n(n={counts.sum()})",
                fontsize=12, fontweight="bold"
            )

        for ax in axes[len(decade_body_data):]:
            ax.set_visible(False)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=type_color_map[t], label=t) for t in all_types
        ]
        fig.legend(
            handles=legend_handles,
            title="Body Type",
            loc="lower center",
            ncol=min(len(all_types), 4),
            fontsize=9,
            bbox_to_anchor=(0.5, -0.06),
        )
        fig.suptitle(
            "Body-Type Share per Decade\n"
            "(shift from light body styles toward heavier SUVs / crossovers)",
            fontsize=13,
            y=1.02,
        )
        plt.tight_layout()
        savefig("pie_evolving_body_type.png")


print("\nAll plots generated successfully.")