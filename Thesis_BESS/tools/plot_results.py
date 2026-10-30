"""
Create plots from clearing result files.

Required inputs:
    - market_meta.csv
    - flows.csv

Created figures:
    1. zonal_prices.png
    2. zonal_supply_demand.png
    3. zonal_net_exports.png
    4. cnec_utilization_heatmap.png
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "output" / "three_zone_run"
MARKET_META_FILE = RESULTS_DIR / "results" / "market_meta.csv"
FLOWS_FILE = RESULTS_DIR / "results" / "flows.csv"

FIGURES_DIR = RESULTS_DIR / "figures"

FIGURE_DPI = 200
SHOW_FIGURES = True

# Use absolute CNEC utilization for the heatmap.
# The flow file contains separate positive and negative directional CNECs, and
# utilization can therefore be negative for the negative direction.
USE_ABSOLUTE_CNEC_UTILIZATION = True

# Optional threshold for showing CNECs in the heatmap. 0.0 shows all CNECs.
# Example: 0.10 only shows CNECs that reach at least 10% utilization.
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


def validate_input_files() -> None:
    """Ensure that both required result files exist."""
    missing_files = [
        path
        for path in [MARKET_META_FILE, FLOWS_FILE]
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(f" - {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Required result files were not found:\n{missing_text}"
        )


def load_market_meta() -> pd.DataFrame:
    """Load market metadata and create a timestamp column."""
    market_meta = pd.read_csv(MARKET_META_FILE)

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
            "market_meta.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    market_meta["time"] = pd.to_datetime(
        market_meta["product_start"],
        errors="raise",
    )

    market_meta["zone"] = market_meta["zone"].astype(str)

    return market_meta.sort_values(["zone", "time"])


def load_flows() -> pd.DataFrame:
    """Load flow results and parse timestamps."""
    flows = pd.read_csv(FLOWS_FILE)

    required_columns = {
        "time",
        "line",
        "flow",
        "network_type",
    }

    missing_columns = required_columns.difference(flows.columns)

    if missing_columns:
        raise ValueError(
            "flows.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    flows["time"] = pd.to_datetime(flows["time"], errors="raise")
    flows["line"] = flows["line"].astype(str)

    return flows.sort_values(["time", "line"])


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


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a figure and optionally display it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)


def create_price_plots(
    market_meta: pd.DataFrame,
    zones: list[str],
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
        data = market_meta.loc[market_meta["zone"] == zone]

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

    save_figure(figure, "zonal_prices.png")


def create_volume_plots(
    market_meta: pd.DataFrame,
    zones: list[str],
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
        data = market_meta.loc[market_meta["zone"] == zone]

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

    save_figure(figure, "zonal_supply_demand.png")


def create_net_export_plot(
    flows: pd.DataFrame,
    zones: list[str],
) -> None:
    """
    Plot net export positions for all zones.

    Positive value: net export.
    Negative value: net import.
    """
    net_exports = flows.loc[
        flows["network_type"].eq("FBMC_NET_EXPORT")
    ].copy()

    if net_exports.empty:
        raise ValueError(
            "flows.csv contains no rows with "
            "network_type == 'FBMC_NET_EXPORT'."
        )

    figure, axis = plt.subplots(figsize=(12, 5))

    for zone in zones:
        data = net_exports.loc[net_exports["line"] == zone]

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

    save_figure(figure, "zonal_net_exports.png")


def create_cnec_utilization_heatmap(flows: pd.DataFrame) -> None:
    """
    Create a CNEC utilization heatmap.

    The rows are individual directional CNECs. The values represent the
    percentage of RAM utilization. Positive and negative directions are kept
    separate because the FBMC constraints are directional.
    """
    cnecs = flows.loc[
        flows["network_type"].eq("FBMC_CNEC")
    ].copy()

    if cnecs.empty:
        raise ValueError(
            "flows.csv contains no rows with network_type == 'FBMC_CNEC'."
        )

    if "utilization" not in cnecs.columns:
        raise ValueError(
            "flows.csv contains no 'utilization' column."
        )

    cnecs["utilization"] = pd.to_numeric(
        cnecs["utilization"],
        errors="coerce",
    )

    cnecs = cnecs.dropna(subset=["utilization"])

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
    )

    heatmap_data = heatmap_data.sort_index()

    max_utilization = heatmap_data.abs().max(axis=1)
    heatmap_data = heatmap_data.loc[
        max_utilization.ge(MINIMUM_MAX_UTILIZATION)
    ]

    if heatmap_data.empty:
        raise ValueError(
            "No CNECs remain after applying "
            "MINIMUM_MAX_UTILIZATION."
        )

    figure_height = max(8, min(28, 0.24 * len(heatmap_data)))
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

    tick_positions = list(range(0, len(timestamp_columns), 8))
    tick_labels = [
        timestamp_columns[position].strftime("%H:%M")
        for position in tick_positions
    ]

    axis.set_xticks([position + 0.5 for position in tick_positions])
    axis.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
    )

    save_figure(figure, "cnec_utilization_heatmap.png")


def main() -> None:
    """Load result data and create all requested plots."""
    configure_plot_style()
    validate_input_files()

    market_meta = load_market_meta()
    flows = load_flows()
    zones = zone_order(market_meta)

    print(f"Creating plots for zones: {', '.join(zones)}")

    create_price_plots(market_meta, zones)
    create_volume_plots(market_meta, zones)
    create_net_export_plot(flows, zones)
    create_cnec_utilization_heatmap(flows)

    print(f"All figures were written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()