import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("Car_Dataset_1945-2020.csv")
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(df["mixed_fuel_consumption_per_100_km_l"], errors="coerce")
df["full_weight_kg"] = pd.to_numeric(df["full_weight_kg"], errors="coerce")

df_scatter = df.dropna(subset=["full_weight_kg", "mixed_fuel_consumption_per_100_km_l"])
df_scatter = df_scatter[df_scatter["Year_from"] >= 1990]


# Filter outliers: keep only within 2 standard deviations on both axes
for col in ["full_weight_kg", "mixed_fuel_consumption_per_100_km_l"]:
    mean, std = df_scatter[col].mean(), df_scatter[col].std()
    df_scatter = df_scatter[(df_scatter[col] >= mean - 5 * std) & (df_scatter[col] <= mean + 5 * std)]

x = df_scatter["full_weight_kg"].values
y = df_scatter["mixed_fuel_consumption_per_100_km_l"].values

r, p = stats.pearsonr(x, y)
coeffs = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 300)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=8, alpha=0.15, color="steelblue", zorder=3, linewidths=0)
ax.plot(x_line, np.poly1d(coeffs)(x_line), color="crimson", linewidth=2,
        label=f"LSE fit  (slope={coeffs[0]:+.5f})", zorder=4)
ax.set_xlabel("Full Weight (kg)")
ax.set_ylabel("Mixed Fuel Consumption (L/100 km)")
ax.set_title(f"Fuel Consumption vs Vehicle Weight\nPearson r = {r:.3f}, p = {p:.2e}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_fuel_vs_weight_clean.png", dpi=150)
print("Saved: scatter_fuel_vs_weight_clean.png")