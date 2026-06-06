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
    "Mixed\nfuel",
    "Highway\nfuel",
    "City\nfuel",
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

    ax.set_facecolor("#fbfbfb")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="0.35")
    ax.set_rlabel_position(180)
    ax.tick_params(axis="x", pad=12, labelsize=10, colors="0.20")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(color="0.80", linewidth=0.8, alpha=0.65)
    ax.spines["polar"].set_color("0.75")
    ax.spines["polar"].set_linewidth(0.9)
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
    colors = ["#2f6fbb", "#d9822b", "#31936a", "#9b59b6", "#c84f4f"]
    columns = 3 if len(df_normalized) >= 5 else 2
    rows = int(np.ceil(len(df_normalized) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 8.8) if columns == 3 else (11, 9),
        subplot_kw={"polar": True},
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.80,
        bottom=0.07,
        wspace=0.48,
        hspace=0.58,
    )
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)

    for ax, (plot_index, row) in zip(axes, df_normalized.iterrows()):
        angles = configure_star_axis(ax)
        values = row[plt_vars].tolist()
        values += values[:1]
        color = colors[plot_index % len(colors)]
        ax.plot(angles, values, linewidth=2.4, color=color)
        ax.fill(angles, values, alpha=0.22, color=color)
        ax.set_title(
            f"{row['category']}\n"
            f"{value_format.format(row['min_category_value'])}-"
            f"{value_format.format(row['max_category_value'])} {category_unit}",
            y=1.22,
            fontsize=11,
            fontweight="semibold",
            color="0.15",
        )

    for ax in axes[len(df_normalized):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.98)
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


def plot_ownership_vs_efficiency(output_path, show_efficiency_gain=False):
    start_year = 1990
    end_year = 2020
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

    comparison = ownership.merge(efficiency, on="year", how="inner")
    comparison = comparison[(comparison["year"] >= start_year) & (comparison["year"] <= end_year)]
    comparison = comparison[comparison["models_count"] >= 100].sort_values("year")

    baseline = comparison.iloc[0]
    comparison["vehicle_growth_index"] = comparison["cars_per_1000"] / baseline["cars_per_1000"]
    comparison["efficiency_gain_index"] = (
        baseline["median_fuel_l_100km"] / comparison["median_fuel_l_100km"]
    )
    comparison["relative_fuel_consumption_index"] = (
        comparison["median_fuel_l_100km"] / baseline["median_fuel_l_100km"]
    )
    comparison["ownership_adjusted_consumption_proxy"] = (
        comparison["vehicle_growth_index"]
        * comparison["relative_fuel_consumption_index"]
    )

    first = comparison.iloc[0]
    last = comparison.iloc[-1]

    fig, (ax, ax_summary) = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        gridspec_kw={"width_ratios": [4.5, 1.55]},
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.14, wspace=0.56)

    ax.axhline(1, color="0.25", linewidth=1, linestyle=":", label=f"{int(first['year'])} baseline")
    ax.plot(
        comparison["year"],
        comparison["vehicle_growth_index"],
        color="#d62728",
        linewidth=3,
        label="Vehicle ownership index",
    )
    secondary_column = "efficiency_gain_index" if show_efficiency_gain else "relative_fuel_consumption_index"
    secondary_label = "Efficiency index: 1990 fuel use / yearly fuel use" if show_efficiency_gain else "Fuel use index: yearly fuel use / 1990 fuel use"
    secondary_annotation = "efficiency index" if show_efficiency_gain else "fuel use index"
    ax.plot(
        comparison["year"],
        comparison[secondary_column],
        color="#2ca02c",
        linewidth=3,
        label=secondary_label,
    )
    ax.plot(
        comparison["year"],
        comparison["ownership_adjusted_consumption_proxy"],
        color="#1f77b4",
        linewidth=3,
        linestyle="--",
        label="Net effect: ownership index x fuel use index",
    )
    ax.fill_between(
        comparison["year"],
        1,
        comparison["ownership_adjusted_consumption_proxy"],
        where=comparison["ownership_adjusted_consumption_proxy"] >= 1,
        color="#1f77b4",
        alpha=0.12,
    )
    ax.annotate(
        f"x{last['vehicle_growth_index']:.2f}\nmore vehicles",
        xy=(last["year"], last["vehicle_growth_index"]),
        xytext=(1.035, last["vehicle_growth_index"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        color="#d62728",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        f"x{last[secondary_column]:.2f}\n{secondary_annotation}",
        xy=(last["year"], last[secondary_column]),
        xytext=(1.035, last[secondary_column]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
        color="#2ca02c",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        f"x{last['ownership_adjusted_consumption_proxy']:.2f}\nnet proxy",
        xy=(last["year"], last["ownership_adjusted_consumption_proxy"]),
        xytext=(1.035, last["ownership_adjusted_consumption_proxy"]),
        textcoords=ax.get_yaxis_transform(),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
        fontsize=10,
        ha="left",
        va="center",
        annotation_clip=False,
    )

    ax.set_title(
        "Efficiency improves, but vehicle ownership grows faster",
        fontsize=16,
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index ({int(first['year'])} = 1)")
    plotted_columns = [
        "vehicle_growth_index",
        secondary_column,
        "ownership_adjusted_consumption_proxy",
    ]
    y_min = max(0.65, comparison[plotted_columns].min().min() - 0.05)
    y_max = comparison[plotted_columns].max().max() + 0.08
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    summary_secondary_value = (
        last["efficiency_gain_index"] - 1
        if show_efficiency_gain
        else last["relative_fuel_consumption_index"] - 1
    )
    summary_secondary_label = "Efficiency" if show_efficiency_gain else "Fuel/veh."
    summary_values = [
        last["vehicle_growth_index"] - 1,
        summary_secondary_value,
        last["ownership_adjusted_consumption_proxy"] - 1,
    ]
    summary_labels = ["Vehicles", summary_secondary_label, "Net"]
    summary_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    y_positions = np.arange(len(summary_values))

    ax_summary.axvline(0, color="0.25", linewidth=1)
    ax_summary.barh(y_positions, summary_values, color=summary_colors, alpha=0.88)
    ax_summary.set_yticks(y_positions)
    ax_summary.set_yticklabels(summary_labels)
    ax_summary.tick_params(axis="y", pad=8)
    ax_summary.invert_yaxis()
    ax_summary.set_title(f"{int(first['year'])} to {int(last['year'])}", fontsize=12)
    ax_summary.set_xlabel("Change vs baseline")
    ax_summary.grid(axis="x", alpha=0.25)
    ax_summary.spines[["top", "right", "left"]].set_visible(False)

    for y, value in zip(y_positions, summary_values):
        x_offset = 0.035 if value >= 0 else -0.035
        ha = "left" if value >= 0 else "right"
        ax_summary.text(
            value + x_offset,
            y,
            f"{value:+.0%}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="semibold",
            color="0.15",
        )

    x_min = min(summary_values + [0]) - 0.22
    x_max = max(summary_values + [0]) + 0.24
    ax_summary.set_xlim(x_min, x_max)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_fuel_consumption_debug(output_path):
    yearly_fuel = (
        df.dropna(subset=["Year_from", "mixed_fuel_consumption_per_100_km_l"])
        .groupby("Year_from", as_index=False)
        .agg(
            median_fuel_l_100km=("mixed_fuel_consumption_per_100_km_l", "median"),
            mean_fuel_l_100km=("mixed_fuel_consumption_per_100_km_l", "mean"),
            models_count=("mixed_fuel_consumption_per_100_km_l", "size"),
        )
        .sort_values("Year_from")
    )
    yearly_fuel = yearly_fuel[
        (yearly_fuel["Year_from"] >= 1990)
        & (yearly_fuel["Year_from"] <= 2020)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        yearly_fuel["Year_from"],
        yearly_fuel["median_fuel_l_100km"],
        linewidth=2.5,
        marker="o",
        label="Median fuel consumption",
    )
    ax.plot(
        yearly_fuel["Year_from"],
        yearly_fuel["mean_fuel_l_100km"],
        linewidth=2,
        linestyle="--",
        label="Mean fuel consumption",
    )
    ax.set_title("Debug: fuel consumption by model launch year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mixed fuel consumption (L/100 km)")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.bar(
        yearly_fuel["Year_from"],
        yearly_fuel["models_count"],
        color="0.8",
        alpha=0.35,
        label="Models count",
    )
    ax2.set_ylabel("Number of models")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")

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
plot_ownership_vs_efficiency("ownership_vs_efficiency_gain.png", show_efficiency_gain=True)
plot_fuel_consumption_debug("fuel_consumption_debug.png")
