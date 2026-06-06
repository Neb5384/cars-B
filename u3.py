import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# Load CSV
df = pd.read_csv("Car_Dataset_1945-2020.csv", low_memory=False)

# Convert relevant columns to numeric
numeric_columns = [
    "Year_from",
    "mixed_fuel_consumption_per_100_km_l",
    "city_fuel_per_100km_l",
    "highway_fuel_per_100km_l",
    "full_weight_kg",
    "length_mm",
    "height_mm",
    "width_mm",
]

for column in numeric_columns:
    df[column] = to_numeric(df[column])

df["volume__m3"] = df["length_mm"] * df["height_mm"] * df["width_mm"] / 1_000_000_000.0

df["car_label"] = (
    df[["Make", "Modle", "Generation", "Trim"]]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

plt_vars = [
    "mixed_fuel_consumption_per_100_km_l",
    "highway_fuel_per_100km_l",
    "city_fuel_per_100km_l",
    "full_weight_kg",
    "volume__m3",
]
axis_labels = [
    "Mixed fuel",
    "Highway fuel",
    "City fuel",
    "Weight",
    "Volume",
]
df_plt = df[["car_label", "Year_from", *plt_vars]].dropna(subset=plt_vars).reset_index(drop=True)

mins = df_plt[plt_vars].min()
ranges = df_plt[plt_vars].max() - mins
df_plt_normalized = df_plt.copy()
df_plt_normalized[plt_vars] = (df_plt[plt_vars] - mins) / ranges.replace(0, np.nan)
df_plt_normalized[plt_vars] = df_plt_normalized[plt_vars].fillna(0)


def configure_star_axis(ax):
    angles = np.linspace(0, 2 * np.pi, len(plt_vars), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{value:.1f}" for value in np.linspace(0, 1, 6)])
    ax.tick_params(axis="x", pad=10, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.35)
    return angles


def aggregate_by_category(category_column):
    category_labels = [f"Category {i}" for i in range(1, 6)]
    df_categorized = df_plt.copy()
    df_categorized["category"] = pd.qcut(
        df_categorized[category_column],
        q=5,
        labels=category_labels,
    )
    df_normalized_categorized = df_plt_normalized.copy()
    df_normalized_categorized["category"] = df_categorized["category"]

    df_aggregated = (
        df_normalized_categorized
        .groupby("category", observed=True)
        .agg(
            mixed_fuel_consumption_per_100_km_l=("mixed_fuel_consumption_per_100_km_l", "mean"),
            highway_fuel_per_100km_l=("highway_fuel_per_100km_l", "mean"),
            city_fuel_per_100km_l=("city_fuel_per_100km_l", "mean"),
            full_weight_kg=("full_weight_kg", "mean"),
            volume__m3=("volume__m3", "mean"),
            cars_count=(category_column, "size"),
        )
        .reset_index()
    )
    maxes = df_aggregated[plt_vars].max()
    df_dilated = df_aggregated.copy()
    df_dilated[plt_vars] = df_aggregated[plt_vars] / maxes.replace(0, np.nan)
    df_dilated[plt_vars] = df_dilated[plt_vars].fillna(0)

    category_ranges = (
        df_categorized
        .groupby("category", observed=True)
        .agg(
            min_category_value=(category_column, "min"),
            max_category_value=(category_column, "max"),
        )
        .reset_index()
    )
    return df_dilated.merge(category_ranges, on="category")


def plot_categories(df_normalized, title, category_unit, value_format, output_path):
    columns = 3
    rows = int(np.ceil(len(df_normalized) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 8),
        constrained_layout=True,
        subplot_kw={"polar": True},
    )
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, df_normalized.iterrows()):
        angles = configure_star_axis(ax)
        values = row[plt_vars].tolist()
        values += values[:1]
        line, = ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25, color=line.get_color())
        ax.set_title(
            f"{row['category']}\n"
            f"{value_format.format(row['min_category_value'])}-"
            f"{value_format.format(row['max_category_value'])} {category_unit}\n"
            f"n={int(row['cars_count'])}",
            y=1.12,
        )

    for ax in axes[len(df_normalized):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_vehicle_ownership_growth(output_path):
    carhab = pd.read_csv("road_eqs_carhab_linear_2_0.csv")
    carhab["TIME_PERIOD"] = pd.to_numeric(carhab["TIME_PERIOD"], errors="coerce")
    carhab["OBS_VALUE"] = pd.to_numeric(carhab["OBS_VALUE"], errors="coerce")
    carhab = carhab.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    carhab["TIME_PERIOD"] = carhab["TIME_PERIOD"].astype(int)

    country_data = carhab[~carhab["geo"].str.startswith("EU", na=False)]
    median_by_year = (
        country_data
        .groupby("TIME_PERIOD", as_index=False)
        .agg(
            median_cars_per_1000=("OBS_VALUE", "median"),
            countries_count=("geo", "nunique"),
        )
    )
    eu27 = carhab[carhab["geo"].eq("EU27_2020")].sort_values("TIME_PERIOD")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.22)

    for _, group in country_data.groupby("geo"):
        group = group.sort_values("TIME_PERIOD")
        ax.plot(
            group["TIME_PERIOD"],
            group["OBS_VALUE"],
            color="0.75",
            linewidth=0.8,
            alpha=0.35,
        )

    ax.plot(
        median_by_year["TIME_PERIOD"],
        median_by_year["median_cars_per_1000"],
        color="#1f77b4",
        linewidth=3,
        label="Median of reporting countries",
    )

    if not eu27.empty:
        ax.plot(
            eu27["TIME_PERIOD"],
            eu27["OBS_VALUE"],
            color="#d62728",
            linewidth=3,
            marker="o",
            markersize=4,
            label="European Union - 27 countries",
        )

    first = median_by_year.iloc[0]
    last = median_by_year.iloc[-1]
    ax.annotate(
        f"{last['median_cars_per_1000']:.0f} cars / 1,000 inhabitants",
        xy=(last["TIME_PERIOD"], last["median_cars_per_1000"]),
        xytext=(-185, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
        fontsize=10,
    )
    ax.annotate(
        f"{first['median_cars_per_1000']:.0f}",
        xy=(first["TIME_PERIOD"], first["median_cars_per_1000"]),
        xytext=(10, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
        fontsize=10,
    )

    ax.set_title(
        "Vehicle ownership keeps rising over time",
        fontsize=16,
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Passenger cars per 1,000 inhabitants")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0,
        -0.16,
        "Thin grey lines are individual reporting countries. This indicator captures ownership intensity, "
        "not absolute vehicle counts or kilometers driven.",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def load_passenger_car_vehicle_km(start_year=2013, end_year=2020):
    rows = []
    with open("series.jsonl") as file:
        for line in file:
            series = json.loads(line)
            freq, regisveh, unit, vehicle, geo = series["dimensions"]
            if (freq, regisveh, unit, vehicle) != ("A", "TERNAT_REG", "MIO_VKM", "CAR"):
                continue

            for period, value, _, _ in series["observations"][1:]:
                rows.append(
                    {
                        "geo": geo,
                        "year": int(period),
                        "vehicle_km_million": float(value),
                    }
                )

    vehicle_km = pd.DataFrame(rows)
    selected_years = set(range(start_year, end_year + 1))
    complete_geos = (
        vehicle_km[vehicle_km["year"].isin(selected_years)]
        .groupby("geo")["year"]
        .nunique()
    )
    complete_geos = complete_geos[complete_geos.eq(len(selected_years))].index

    return (
        vehicle_km[
            vehicle_km["geo"].isin(complete_geos)
            & vehicle_km["year"].isin(selected_years)
        ]
        .groupby("year", as_index=False)
        .agg(
            vehicle_km_million=("vehicle_km_million", "sum"),
            vehicle_km_countries=("geo", "nunique"),
        )
    )


def plot_ownership_vs_efficiency(output_path):
    start_year = 2013
    end_year = 2019
    carhab = pd.read_csv("road_eqs_carhab_linear_2_0.csv")
    carhab["TIME_PERIOD"] = pd.to_numeric(carhab["TIME_PERIOD"], errors="coerce")
    carhab["OBS_VALUE"] = pd.to_numeric(carhab["OBS_VALUE"], errors="coerce")
    carhab = carhab.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    carhab["TIME_PERIOD"] = carhab["TIME_PERIOD"].astype(int)

    country_data = carhab[~carhab["geo"].str.startswith("EU", na=False)]
    ownership = (
        country_data
        .groupby("TIME_PERIOD", as_index=False)
        .agg(cars_per_1000=("OBS_VALUE", "median"))
        .rename(columns={"TIME_PERIOD": "year"})
    )

    efficiency = (
        df.dropna(subset=["Year_from", "mixed_fuel_consumption_per_100_km_l"])
        .groupby("Year_from", as_index=False)
        .agg(
            median_fuel_l_100km=("mixed_fuel_consumption_per_100_km_l", "median"),
            models_count=("mixed_fuel_consumption_per_100_km_l", "size"),
        )
        .rename(columns={"Year_from": "year"})
    )
    efficiency["year"] = efficiency["year"].astype(int)
    vehicle_km = load_passenger_car_vehicle_km(start_year, end_year)

    comparison = ownership.merge(efficiency, on="year", how="inner").merge(vehicle_km, on="year", how="inner")
    comparison = comparison[(comparison["year"] >= start_year) & (comparison["year"] <= end_year)]
    comparison = comparison[comparison["models_count"] >= 100].sort_values("year")

    baseline = comparison.iloc[0]
    comparison["vehicle_growth_index"] = comparison["cars_per_1000"] / baseline["cars_per_1000"]
    comparison["vehicle_km_index"] = (
        comparison["vehicle_km_million"] / baseline["vehicle_km_million"]
    )
    comparison["efficiency_gain_index"] = (
        baseline["median_fuel_l_100km"] / comparison["median_fuel_l_100km"]
    )
    comparison["vkm_adjusted_consumption_proxy"] = (
        comparison["vehicle_km_index"] / comparison["efficiency_gain_index"]
    )

    first = comparison.iloc[0]
    last = comparison.iloc[-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.86, bottom=0.14)

    ax.axhline(1, color="0.25", linewidth=1, linestyle=":", label=f"{int(first['year'])} baseline")
    ax.plot(
        comparison["year"],
        comparison["vehicle_growth_index"],
        color="#d62728",
        linewidth=2,
        alpha=0.8,
        label="Vehicle ownership: cars / 1,000 inhabitants",
    )
    ax.plot(
        comparison["year"],
        comparison["vehicle_km_index"],
        color="#ff7f0e",
        linewidth=3,
        label="Passenger-car traffic: vehicle-km",
    )
    ax.plot(
        comparison["year"],
        comparison["efficiency_gain_index"],
        color="#2ca02c",
        linewidth=3,
        label="Efficiency gain: lower L/100 km",
    )
    ax.plot(
        comparison["year"],
        comparison["vkm_adjusted_consumption_proxy"],
        color="#1f77b4",
        linewidth=3,
        linestyle="--",
        label="Combined proxy: vehicle-km x fuel consumption",
    )
    ax.fill_between(
        comparison["year"],
        1,
        comparison["vkm_adjusted_consumption_proxy"],
        where=comparison["vkm_adjusted_consumption_proxy"] >= 1,
        color="#1f77b4",
        alpha=0.12,
        label=f"Remaining pressure above {int(first['year'])}",
    )

    ax.annotate(
        f"x{last['vehicle_growth_index']:.2f}\nmore vehicles",
        xy=(last["year"], last["vehicle_growth_index"]),
        xytext=(1.04, last["vehicle_growth_index"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        color="#d62728",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        f"x{last['vehicle_km_index']:.2f}\nvehicle-km",
        xy=(last["year"], last["vehicle_km_index"]),
        xytext=(1.04, last["vehicle_km_index"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#ff7f0e"},
        color="#ff7f0e",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        f"x{last['efficiency_gain_index']:.2f}\nefficiency gain",
        xy=(last["year"], last["efficiency_gain_index"]),
        xytext=(1.04, last["efficiency_gain_index"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
        color="#2ca02c",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        f"x{last['vkm_adjusted_consumption_proxy']:.2f}\nnet proxy",
        xy=(last["year"], last["vkm_adjusted_consumption_proxy"]),
        xytext=(1.04, last["vkm_adjusted_consumption_proxy"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )

    ax.set_title(
        "Vehicle-km growth offsets part of efficiency gains",
        fontsize=16,
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index ({int(first['year'])} = 1)")
    plotted_columns = [
        "vehicle_growth_index",
        "vehicle_km_index",
        "efficiency_gain_index",
        "vkm_adjusted_consumption_proxy",
    ]
    y_min = max(0.9, comparison[plotted_columns].min().min() - 0.05)
    y_max = comparison[plotted_columns].max().max() + 0.08
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


weight_categories = aggregate_by_category("full_weight_kg")
plot_categories(
    weight_categories,
    title="Star charts by vehicle weight category",
    category_unit="kg",
    value_format="{:.0f}",
    output_path="star_charts_by_weight.png",
)

volume_categories = aggregate_by_category("volume__m3")
plot_categories(
    volume_categories,
    title="Star charts by vehicle volume category",
    category_unit="m3",
    value_format="{:.2f}",
    output_path="star_charts_by_volume.png",
)

plot_vehicle_ownership_growth("vehicle_ownership_growth.png")
plot_ownership_vs_efficiency("ownership_vs_efficiency.png")
