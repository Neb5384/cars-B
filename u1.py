import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load CSV
df = pd.read_csv("Car_Dataset_1945-2020.csv")

# Convert relevant columns to numeric
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(df["mixed_fuel_consumption_per_100_km_l"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"], errors="coerce")

# fuel consumption per year ------------
df = df.dropna(subset=[
    "Year_from",
    "mixed_fuel_consumption_per_100_km_l"
])
df["Year_from"] = df["Year_from"].astype(int)
df = df.sort_values("Year_from")

r_fuel_year, p_fuel_year = stats.pearsonr(df["Year_from"], df["mixed_fuel_consumption_per_100_km_l"])

plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="mixed_fuel_consumption_per_100_km_l",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Mixed Fuel Consumption (L/100km)")
plt.title(f"Fuel Consumption Distribution per Year\n(Pearson r = {r_fuel_year:.3f}, p = {p_fuel_year:.2e})")
plt.tight_layout()
plt.show()

# full weight per year -----------------
df = df.dropna(subset=[
    "Year_from",
    "full_weight_kg"
])
df["Year_from"] = df["Year_from"].astype(int)
df = df.sort_values("Year_from")

r_weight_year, p_weight_year = stats.pearsonr(df["Year_from"], df["full_weight_kg"])

plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="full_weight_kg",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Full weight [kg]")
plt.title(f"Full weight per Year\n(Pearson r = {r_weight_year:.3f}, p = {p_weight_year:.2e})")
plt.tight_layout()
plt.show()

# CO2 emissions per year ------------------------
df = df.dropna(subset=[
    "Year_from",
    "CO2_emissions_g/km"
])
df["Year_from"] = df["Year_from"].astype(int)
df = df.sort_values("Year_from")

r_co2_year, p_co2_year = stats.pearsonr(df["Year_from"], df["CO2_emissions_g/km"])

plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="CO2_emissions_g/km",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("CO2 emissions (g/km)")
plt.title(f"CO2 Emissions per Year\n(Pearson r = {r_co2_year:.3f}, p = {p_co2_year:.2e})")
plt.tight_layout()
plt.show()

# means and medians with LSE fit -------------------------
df_agg = (
    df.groupby("Year_from", as_index=False)
    .agg(
        mean_weight   = ("full_weight_kg",                     "mean"),
        mean_fuel     = ("mixed_fuel_consumption_per_100_km_l", "mean"),
        mean_co2      = ("CO2_emissions_g/km",                  "mean"),
        median_weight = ("full_weight_kg",                     "median"),
        median_fuel   = ("mixed_fuel_consumption_per_100_km_l", "median"),
        median_co2    = ("CO2_emissions_g/km",                  "median"),
    )
    .dropna()
)

df_agg = df_agg[df_agg["Year_from"] >= 1950]

df_agg["mean_fuel_per_weight"]   = df_agg["mean_fuel"] / df_agg["mean_weight"]
df_agg["median_fuel_per_weight"] = df_agg["median_fuel"] / df_agg["median_weight"]

years = df_agg["Year_from"].values


def plot_with_lse(ax, x, y, color, ylabel, title):
    ax.scatter(x, y, s=18, alpha=0.7, color=color, zorder=3, label="Annual value")
    coeffs = np.polyfit(x, y, 1)
    trend  = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 300)
    r, p = stats.pearsonr(x, y)
    ax.plot(x_line, trend(x_line), color="crimson", linewidth=2,
            label=f"LSE fit  (slope={coeffs[0]:+.3f})")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)


# Plot means ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("Annual Means with Least-Squares Fit (1950–2020)", fontsize=14, y=1.02)

plot_with_lse(axes[0], years, df_agg["mean_weight"].values,
              "steelblue",     "Mean Full Weight (kg)",                "Mean Weight / Year")
plot_with_lse(axes[1], years, df_agg["mean_fuel"].values,
              "darkorange",    "Mean Fuel Consumption (L/100 km)",     "Mean Fuel Consumption / Year")
plot_with_lse(axes[2], years, df_agg["mean_co2"].values,
              "mediumpurple",  "Mean CO2 Emissions (g/km)",            "Mean CO2 Emissions / Year")
plot_with_lse(axes[3], years, df_agg["mean_fuel_per_weight"].values,
              "mediumseagreen","Mean Fuel per Weight (L/100km/kg)",    "Mean Fuel per Weight over Years")

plt.tight_layout()
plt.show()

# Plot medians ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("Annual Medians with Least-Squares Fit (1950–2020)", fontsize=14, y=1.02)

plot_with_lse(axes[0], years, df_agg["median_weight"].values,
              "steelblue",     "Median Full Weight (kg)",              "Median Weight / Year")
plot_with_lse(axes[1], years, df_agg["median_fuel"].values,
              "darkorange",    "Median Fuel Consumption (L/100 km)",   "Median Fuel Consumption / Year")
plot_with_lse(axes[2], years, df_agg["median_co2"].values,
              "mediumpurple",  "Median CO2 Emissions (g/km)",          "Median CO2 Emissions / Year")
plot_with_lse(axes[3], years, df_agg["median_fuel_per_weight"].values,
              "mediumseagreen","Median Fuel per Weight (L/100km/kg)",  "Median Fuel per Weight over Years")

plt.tight_layout()
plt.show()

# Plot all individual data points with LSE fit ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("All Data Points with Least-Squares Fit (1950–2020)", fontsize=14, y=1.02)

df_raw = df[df["Year_from"] >= 1950]


def plot_raw_with_lse(ax, data, ycol, color, ylabel, title):
    subset = data.dropna(subset=["Year_from", ycol])
    x = subset["Year_from"].values
    y = subset[ycol].values
    r, p = stats.pearsonr(x, y)
    ax.scatter(x, y, s=8, alpha=0.2, color=color, zorder=3, label="Data point")
    coeffs = np.polyfit(x, y, 1)
    trend  = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_line, trend(x_line), color="crimson", linewidth=2,
            label=f"LSE fit  (slope={coeffs[0]:+.3f})")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)


plot_raw_with_lse(axes[0], df_raw, "full_weight_kg",
                  "steelblue",     "Full Weight (kg)",             "Weight / Year")
plot_raw_with_lse(axes[1], df_raw, "mixed_fuel_consumption_per_100_km_l",
                  "darkorange",    "Fuel Consumption (L/100 km)",  "Fuel Consumption / Year")
plot_raw_with_lse(axes[2], df_raw, "CO2_emissions_g/km",
                  "mediumpurple",  "CO2 Emissions (g/km)",         "CO2 Emissions / Year")

df_ratio = df_raw.dropna(subset=["full_weight_kg", "mixed_fuel_consumption_per_100_km_l"])
df_ratio = df_ratio.copy()
df_ratio["fuel_per_weight"] = df_ratio["mixed_fuel_consumption_per_100_km_l"] / df_ratio["full_weight_kg"]
plot_raw_with_lse(axes[3], df_ratio, "fuel_per_weight",
                  "mediumseagreen","Fuel per Weight (L/100km/kg)", "Fuel per Weight over Years")

plt.tight_layout()
plt.show()

# Fuel consumption vs weight ------------------------
df_scatter = df.dropna(subset=["full_weight_kg", "mixed_fuel_consumption_per_100_km_l", "Year_from"])

x = df_scatter["full_weight_kg"].values
y = df_scatter["mixed_fuel_consumption_per_100_km_l"].values

r, p = stats.pearsonr(x, y)

coeffs = np.polyfit(x, y, 1)
trend  = np.poly1d(coeffs)
x_line = np.linspace(x.min(), x.max(), 300)

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(x, y, s=8, alpha=0.2, c=df_scatter["Year_from"], zorder=3, label="Data point")
plt.colorbar(sc, ax=ax, label="Year")
ax.plot(x_line, trend(x_line), color="crimson", linewidth=2,
        label=f"LSE fit  (slope={coeffs[0]:+.5f})")
ax.set_xlabel("Full Weight (kg)")
ax.set_ylabel("Mixed Fuel Consumption (L/100 km)")
ax.set_title(f"Fuel Consumption vs Vehicle Weight\nPearson r = {r:.3f}, p = {p:.2e}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()