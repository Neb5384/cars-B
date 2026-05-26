import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("Car_Dataset_1945-2020.csv")
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
