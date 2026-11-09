"""
Create plots from clearing result files for storage and non-storage scenarios.

Required inputs per scenario:
    - market_meta.csv
    - flows.csv
    - CNECs.csv

Created figures per scenario:
    1. zonal_prices.png
    2. zonal_supply_demand.png
    3. zonal_net_exports.png
    4. cnec_utilization_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent

RESULTS_DIR = PROJECT_DIR / "output" / "three_zone_run" / "results"

SCENARIO_DIRS = [
    RESULTS_DIR / "without_storage",
    RESULTS_DIR / "with_storage",
]

FIGURE_DPI = 200
SHOW_FIGURES = True

# Use absolute CNEC utilization for the heatmap.
# The output contains separate positive and negative directional CNECs.
USE_ABSOLUTE_CNEC_UTILIZATION = True

# Optional threshold for showing CNECs in the heatmap.
# 0.0 shows every CNEC.
MINIMUM_MAX_UTILIZATION = 0.0

# =============================================================================


def configure_plot_style() -> None:
    """Set consistent plotting defaults."""
    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )


def validate_input_files(
    market_meta_file: Path,
    flows_file: Path,
    cnecs_file: Path,
) -> None:
    """Ensure that all required scenario result files exist."""
    required_files = [
        market_meta_file,
        flows_file,
        cnecs_file,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(f" - {path}" for path in missing_files)

        raise FileNotFoundError(
            f"Required result files were not found:\n{missing_text}"
        )


def load_market_meta(path: Path) -> pd.DataFrame:
    """Load market metadata and create a timestamp column."""
    market_meta = pd.read_csv(path)

    required_columns = {
        "zone",
        "product_start",
        "price",
        "supply_volume",
        "demand_volume",
    }

    missing_columns = required_columns.difference(market_meta.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    market_meta["time"] = pd.to_datetime(
        market_meta["product_start"],
        errors="raise",
    )

    market_meta["zone"] = market_meta["zone"].astype(str)

    for column in (
        "price",
        "supply_volume",
        "demand_volume",
    ):
        market_meta[column] = pd.to_numeric(
            market_meta[column],
            errors="raise",
        )

    return market_meta.sort_values(["zone", "time"])


def load_flows(path: Path) -> pd.DataFrame:
    """Load NTC and FBMC net-export flow results."""
    flows = pd.read_csv(path)

    required_columns = {
        "time",
        "line",
        "flow",
        "network_type",
    }

    missing_columns = required_columns.difference(flows.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    flows["time"] = pd.to_datetime(flows["time"], errors="raise")
    flows["line"] = flows["line"].astype(str)
    flows["flow"] = pd.to_numeric(flows["flow"], errors="raise")

    return flows.sort_values(["time", "line"])


def load_cnecs(path: Path) -> pd.DataFrame:
    """Load CNEC diagnostics from the dedicated CNECs output file."""
    cnecs = pd.read_csv(path)

    required_columns = {
        "time",
        "line",
        "network_type",
        "utilization",
    }

    missing_columns = required_columns.difference(cnecs.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not cnecs.empty:
        cnecs["time"] = pd.to_datetime(cnecs["time"], errors="raise")
        cnecs["line"] = cnecs["line"].astype(str)
        cnecs["utilization"] = pd.to_numeric(
            cnecs["utilization"],
            errors="coerce",
        )

    return cnecs.sort_values(["time", "line"])


def zone_order(market_meta: pd.DataFrame) -> list[str]:
    """Return zones in natural lexicographic order."""
    return sorted(market_meta["zone"].unique(), key=str)


def format_time_axis(axis: plt.Axes) -> None:
    """Format quarter-hourly date-time axes consistently."""
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def save_figure(
    figure: plt.Figure,
    figures_dir: Path,
    filename: str,
) -> None:
    """Save a figure in the scenario figure directory."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / filename

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)


def create_price_plots(
    market_meta: pd.DataFrame,
    zones: list[str],
    figures_dir: Path,
) -> None:
    """Create side-by-side zonal price line plots."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(zones),
        figsize=(6 * len(zones), 4.5),
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for axis, zone in zip(axes, zones):
        data = market_meta.loc[
            market_meta["zone"] == zone
        ].sort_values("time")

        axis.plot(
            data["time"],
            data["price"],
            color="tab:blue",
            linewidth=1.8,
        )

        axis.axhline(
            0,
            color="black",
            linewidth=0.8,
            alpha=0.7,
        )

        axis.set_title(f"{zone}: Clearing price")
        axis.set_xlabel("Time")
        axis.set_ylabel("Price [EUR/MWh]")
        format_time_axis(axis)

    figure.suptitle(
        "Zonal day-ahead clearing prices",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    save_figure(
        figure,
        figures_dir,
        "zonal_prices.png",
    )


def create_volume_plots(
    market_meta: pd.DataFrame,
    zones: list[str],
    figures_dir: Path,
) -> None:
    """Create side-by-side zonal supply and demand volume plots."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(zones),
        figsize=(6 * len(zones), 4.5),
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for axis, zone in zip(axes, zones):
        data = market_meta.loc[
            market_meta["zone"] == zone
        ].sort_values("time")

        axis.plot(
            data["time"],
            data["supply_volume"],
            label="Supply",
            color="tab:green",
            linewidth=1.8,
        )

        axis.plot(
            data["time"],
            data["demand_volume"],
            label="Demand",
            color="tab:red",
            linewidth=1.8,
        )

        axis.set_title(f"{zone}: Supply and demand")
        axis.set_xlabel("Time")
        axis.set_ylabel("Volume [MW]")
        axis.legend(loc="best")
        format_time_axis(axis)

    figure.suptitle(
        "Zonal cleared supply and demand volumes",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    save_figure(
        figure,
        figures_dir,
        "zonal_supply_demand.png",
    )


def create_net_export_plot(
    flows: pd.DataFrame,
    zones: list[str],
    figures_dir: Path,
) -> None:
    """
    Plot FBMC net export positions for all zones.

    Positive value: net export.
    Negative value: net import.
    """
    net_exports = flows.loc[
        flows["network_type"].eq("FBMC_NET_EXPORT")
    ].copy()

    if net_exports.empty:
        print(
            "Skipping zonal_net_exports.png: "
            "flows.csv contains no FBMC_NET_EXPORT rows."
        )
        return

    figure, axis = plt.subplots(figsize=(12, 5))

    for zone in zones:
        data = net_exports.loc[
            net_exports["line"] == zone
        ].sort_values("time")

        if data.empty:
            print(f"Warning: No net-export data found for {zone}.")
            continue

        axis.plot(
            data["time"],
            data["flow"],
            label=zone,
            linewidth=1.8,
        )

    axis.axhline(
        0,
        color="black",
        linewidth=0.9,
    )

    axis.set_title("Zonal net export positions")
    axis.set_xlabel("Time")
    axis.set_ylabel("Net export [MW]")
    axis.legend(title="Zone", ncol=len(zones), loc="best")
    format_time_axis(axis)

    save_figure(
        figure,
        figures_dir,
        "zonal_net_exports.png",
    )


def create_cnec_utilization_heatmap(
    cnecs: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Create a CNEC utilization heatmap.

    CNEC diagnostics are read from CNECs.csv. Positive and negative
    directional CNECs are kept separate.
    """
    cnecs = cnecs.loc[
        cnecs["network_type"].eq("FBMC_CNEC")
    ].copy()

    if cnecs.empty:
        print(
            "Skipping cnec_utilization_heatmap.png: "
            "CNECs.csv contains no FBMC_CNEC rows."
        )
        return

    cnecs = cnecs.dropna(subset=["utilization"])

    if cnecs.empty:
        print(
            "Skipping cnec_utilization_heatmap.png: "
            "No valid CNEC utilization values are available."
        )
        return

    if USE_ABSOLUTE_CNEC_UTILIZATION:
        cnecs["plot_utilization"] = cnecs["utilization"].abs()
        color_map = "YlOrRd"
        color_min = 0.0
        color_max = max(1.0, cnecs["plot_utilization"].max())
        colorbar_label = "Absolute RAM utilization [-]"
    else:
        cnecs["plot_utilization"] = cnecs["utilization"]
        color_map = "coolwarm"
        maximum = max(1.0, cnecs["plot_utilization"].abs().max())
        color_min = -maximum
        color_max = maximum
        colorbar_label = "Signed RAM utilization [-]"

    for column, default_value in (
        ("cnec", ""),
        ("cne", ""),
        ("contingency", "basecase"),
        ("direction", ""),
    ):
        if column not in cnecs.columns:
            cnecs[column] = default_value

    cnecs["cnec_label"] = (
        cnecs["cnec"].fillna(cnecs["line"]).astype(str)
        + " | "
        + cnecs["cne"].fillna("").astype(str)
        + " | "
        + cnecs["contingency"].fillna("basecase").astype(str)
        + " | "
        + cnecs["direction"].fillna("").astype(str)
    )

    heatmap_data = cnecs.pivot_table(
        index="cnec_label",
        columns="time",
        values="plot_utilization",
        aggfunc="first",
    ).sort_index()

    max_utilization = heatmap_data.abs().max(axis=1)

    heatmap_data = heatmap_data.loc[
        max_utilization.ge(MINIMUM_MAX_UTILIZATION)
    ]

    if heatmap_data.empty:
        print(
            "Skipping cnec_utilization_heatmap.png: "
            "No CNECs remain after applying MINIMUM_MAX_UTILIZATION."
        )
        return

    figure_height = max(
        8,
        min(28, 0.24 * len(heatmap_data)),
    )

    figure, axis = plt.subplots(
        figsize=(18, figure_height),
    )

    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        cbar_kws={
            "label": colorbar_label,
            "shrink": 0.8,
        },
        xticklabels=8,
        yticklabels=True,
        linewidths=0.0,
    )

    axis.set_title("CNEC utilization over time")
    axis.set_xlabel("Time")
    axis.set_ylabel("CNEC | CNE | contingency | direction")

    timestamp_columns = heatmap_data.columns

    tick_positions = list(
        range(0, len(timestamp_columns), 8)
    )
    tick_labels = [
        timestamp_columns[position].strftime("%H:%M")
        for position in tick_positions
    ]

    axis.set_xticks([
        position + 0.5
        for position in tick_positions
    ])
    axis.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
    )

    save_figure(
        figure,
        figures_dir,
        "cnec_utilization_heatmap.png",
    )


def plot_scenario(scenario_dir: Path) -> None:
    """Load one scenario folder and create all applicable figures."""
    market_meta_file = scenario_dir / "market_meta.csv"
    flows_file = scenario_dir / "flows.csv"
    cnecs_file = scenario_dir / "CNECs.csv"
    figures_dir = scenario_dir / "figures"

    validate_input_files(
        market_meta_file=market_meta_file,
        flows_file=flows_file,
        cnecs_file=cnecs_file,
    )

    market_meta = load_market_meta(market_meta_file)
    flows = load_flows(flows_file)
    cnecs = load_cnecs(cnecs_file)
    zones = zone_order(market_meta)

    print(
        f"\nCreating plots for scenario '{scenario_dir.name}' "
        f"and zones: {', '.join(zones)}"
    )

    create_price_plots(
        market_meta=market_meta,
        zones=zones,
        figures_dir=figures_dir,
    )
    create_volume_plots(
        market_meta=market_meta,
        zones=zones,
        figures_dir=figures_dir,
    )
    create_net_export_plot(
        flows=flows,
        zones=zones,
        figures_dir=figures_dir,
    )
    create_cnec_utilization_heatmap(
        cnecs=cnecs,
        figures_dir=figures_dir,
    )

    print(f"Figures written to: {figures_dir}")


def main() -> None:
    """Create result plots for the no-storage and storage scenarios."""
    configure_plot_style()

    for scenario_dir in SCENARIO_DIRS:
        plot_scenario(scenario_dir)


if __name__ == "__main__":
    main()