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


# Benno part

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

# Filter 
df_agg = df_agg[df_agg["Year_from"] >= 1900]

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

plt.tight_layout()
plt.show()

# Plot all individual data points with LSE fit ------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle("All Data Points with Least-Squares Fit (2010–2020)", fontsize=14, y=1.02)

# Filter
df_raw = df[df["Year_from"] >= 1900]

# Helper: scatter all points + LSE fit
def plot_raw_with_lse(ax, data, ycol, color, ylabel, title):
    subset = data.dropna(subset=["Year_from", ycol])
    x = subset["Year_from"].values
    y = subset[ycol].values
    ax.scatter(x, y, s=8, alpha=0.2, color=color, zorder=3, label="Data point")
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

plot_raw_with_lse(axes[0], df_raw, "full_weight_kg",
                  "steelblue",     "Full Weight (kg)",             "Weight / Year")
plot_raw_with_lse(axes[1], df_raw, "mixed_fuel_consumption_per_100_km_l",
                  "darkorange",    "Fuel Consumption (L/100 km)",  "Fuel Consumption / Year")
plot_raw_with_lse(axes[2], df_raw, "CO2_emissions_g/km",
                  "mediumpurple",  "CO2 Emissions (g/km)",         "CO2 Emissions / Year")


plt.tight_layout()
plt.show()

# Fuel consumption vs weight ------------------------

df_scatter = df.dropna(subset=["full_weight_kg", "mixed_fuel_consumption_per_100_km_l"])

x = df_scatter["full_weight_kg"].values
y = df_scatter["mixed_fuel_consumption_per_100_km_l"].values

# LSE fit
coeffs = np.polyfit(x, y, 1)
trend  = np.poly1d(coeffs)
x_line = np.linspace(x.min(), x.max(), 300)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=8, alpha=0.2, color="steelblue", zorder=3, label="Data point")
ax.plot(x_line, trend(x_line), color="crimson", linewidth=2,
        label=f"LSE fit  (slope={coeffs[0]:+.5f})")
ax.set_xlabel("Full Weight (kg)")
ax.set_ylabel("Mixed Fuel Consumption (L/100 km)")
ax.set_title("Fuel Consumption vs Vehicle Weight")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()







# James Part

# --- full weight ---
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"],errors="coerce")
df["curb_weight_kg"] = pd.to_numeric(df["curb_weight_kg"],errors="coerce")
# Remove missing values
df = df.dropna(subset=[
    "Year_from",
    "full_weight_kg",
    "curb_weight_kg"
])
df["Year_from"] = df["Year_from"].astype(int)
# Sort by year
df = df.sort_values("Year_from")
# Plot
plt.figure(figsize=(18, 8))
sns.lineplot(
    x="Year_from",
    y="full_weight_kg",
    data=df,
    label="curb weight"
)
sns.lineplot(
    x="Year_from",
    y="curb_weight_kg",
    data=df,
    label="full weight"
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Weight [kg]")
plt.title("Car weigths per Year")
plt.tight_layout()
plt.show()

# --- car volume ---
# Remove missing values
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["length_mm"] = pd.to_numeric(df["length_mm"],errors="coerce")
df["width_mm"] = pd.to_numeric(df["width_mm"],errors="coerce")
df["height_mm"] = pd.to_numeric(df["height_mm"],errors="coerce")

df = df.dropna(subset=[
    "Year_from",
    "length_mm",
    "width_mm",
    "height_mm"
])
df["Year_from"] = df["Year_from"].astype(int)
# Sort by year
df = df.sort_values("Year_from")
# Plot
length = df["length_mm"]/1000
width = df["width_mm"]/1000
height = df["height_mm"]/1000
df["volume"] = length*width*height
plt.figure(figsize=(18, 8))
sns.lineplot(
    x="Year_from",
    y="volume",
    data=df
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Car volume [m³]")
plt.title("Car volume per Year")
plt.tight_layout()
plt.show()


# --- engine ---
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["capacity_cm3"] = pd.to_numeric(df["capacity_cm3"],errors="coerce")
df["number_of_cylinders"] = pd.to_numeric(df["number_of_cylinders"],errors="coerce")
# Remove missing values
df = df.dropna(subset=[
    "Year_from",
    "capacity_cm3",
    "number_of_cylinders"
])
df["Year_from"] = df["Year_from"].astype(int)
# Sort by year
df = df.sort_values("Year_from")
# Plot
capacity = df["capacity_cm3"]
nbr_cylinders = df["number_of_cylinders"]
df["capacity_x_nbr_cylinders"] = capacity*nbr_cylinders
plt.figure(figsize=(18, 8))
sns.barplot(
    x="Year_from",
    y="capacity_x_nbr_cylinders",
    # y="capacity_cm3",
    # y="number_of_cylinders",
    data=df,
)
plt.xticks(rotation=90)
plt.xlabel("Year From")
plt.ylabel("Engine capacity [cm³] x number of cylinders")
plt.title("Engine capacity per Year")
plt.tight_layout()
plt.show()
