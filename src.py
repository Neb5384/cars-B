import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load CSV
df = pd.read_csv("Car_Dataset_1945-2020.csv")

# Convert relevant columns to numeric
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(df["mixed_fuel_consumption_per_100_km_l"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"], errors="coerce")

# fuel consumption per year ------------
# Remove missing values
df = df.dropna(subset=[
    "Year_from",
    "mixed_fuel_consumption_per_100_km_l"
])
df["Year_from"] = df["Year_from"].astype(int)

# Sort by year
df = df.sort_values("Year_from")

# Plot
plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="mixed_fuel_consumption_per_100_km_l",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Mixed Fuel Consumption (L/100km)")
plt.title("Fuel Consumption Distribution per Year")
plt.tight_layout()
plt.show()

# full weight per year -----------------
# Remove missing values
df = df.dropna(subset=[
    "Year_from",
    "full_weight_kg"
])
df["Year_from"] = df["Year_from"].astype(int)

# Sort by year
df = df.sort_values("Year_from")

# Plot
plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="full_weight_kg",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Full weight [kg]")
plt.title("Full weight per Year")
plt.tight_layout()
plt.show()

# CO2 emissions per year ------------------------
# Remove missing values
df = df.dropna(subset=[
    "Year_from",
    "CO2_emissions_g/km"
])
df["Year_from"] = df["Year_from"].astype(int)

# Sort by year
df = df.sort_values("Year_from")

# Plot
plt.figure(figsize=(18, 8))
sns.boxplot(
    x="Year_from",
    y="CO2_emissions_g/km",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("CO2 emissions (g/km)")
plt.title("CO2 Emissions per Year")
plt.tight_layout()
plt.show()

# means and medians with LSE fit -------------------------
# Aggregate means and medians per year
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

# Filter from 2010 onwards
df_agg = df_agg[df_agg["Year_from"] >= 2000]

# Derived ratio columns
df_agg["mean_weight_per_fuel"]   = df_agg["mean_weight"]   / df_agg["mean_fuel"]
df_agg["median_weight_per_fuel"] = df_agg["median_weight"] / df_agg["median_fuel"]

years = df_agg["Year_from"].values

# Helper: scatter + LSE fit
def plot_with_lse(ax, x, y, color, ylabel, title):
    ax.scatter(x, y, s=18, alpha=0.7, color=color, zorder=3, label="Annual value")
    coeffs = np.polyfit(x, y, 1)
    trend  = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_line, trend(x_line), color="crimson", linewidth=2,
            label=f"LSE fit  (slope={coeffs[0]:+.3f})")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot means ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("Annual Means with Least-Squares Fit (2010–2020)", fontsize=14, y=1.02)

plot_with_lse(axes[0], years, df_agg["mean_weight"].values,
              "steelblue",     "Mean Full Weight (kg)",                "Mean Weight / Year")
plot_with_lse(axes[1], years, df_agg["mean_fuel"].values,
              "darkorange",    "Mean Fuel Consumption (L/100 km)",     "Mean Fuel Consumption / Year")
plot_with_lse(axes[2], years, df_agg["mean_co2"].values,
              "mediumpurple",  "Mean CO2 Emissions (g/km)",            "Mean CO2 Emissions / Year")
plot_with_lse(axes[3], years, df_agg["mean_weight_per_fuel"].values,
              "mediumseagreen","Mean Weight / Fuel (kg·100km/L)",      "Mean Weight per Fuel / Year")

plt.tight_layout()
plt.show()

# Plot medians ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("Annual Medians with Least-Squares Fit (2010–2020)", fontsize=14, y=1.02)

plot_with_lse(axes[0], years, df_agg["median_weight"].values,
              "steelblue",     "Median Full Weight (kg)",              "Median Weight / Year")
plot_with_lse(axes[1], years, df_agg["median_fuel"].values,
              "darkorange",    "Median Fuel Consumption (L/100 km)",   "Median Fuel Consumption / Year")
plot_with_lse(axes[2], years, df_agg["median_co2"].values,
              "mediumpurple",  "Median CO2 Emissions (g/km)",          "Median CO2 Emissions / Year")
plot_with_lse(axes[3], years, df_agg["median_weight_per_fuel"].values,
              "mediumseagreen","Median Weight / Fuel (kg·100km/L)",    "Median Weight per Fuel / Year")

plt.tight_layout()
plt.show()