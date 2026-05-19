import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("Car_Dataset_1945-2020.csv")

# Convert relevant columns to numeric
df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")

df["mixed_fuel_consumption_per_100_km_l"] = pd.to_numeric(df["mixed_fuel_consumption_per_100_km_l"],errors="coerce")

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