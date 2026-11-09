"""
Create supply and demand curve plots for every MTU and bidding zone.

Input per scenario:
    aggregated_curves_<zone>.csv

Scenarios:
    - without_storage
    - with_storage

Output per scenario:
    figures/curve_plots/<zone>/<date>_<hour>_<quarter>.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
SHOW_PLOTS = False

# Optional plot limits. Use None for automatic scaling.
X_LIMIT = None
Y_LIMIT = None

# =============================================================================


def safe_filename(value: str) -> str:
    """Return a filesystem-safe version of a zone or timestamp label."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def load_aggregated_curves(path: Path) -> pd.DataFrame:
    """
    Load one EPEX-style aggregated curve file.

    Expected columns:
        Date, Week, Week Day, Hour, Quarter hour, Price, Volume,
        Sale/Purchase
    """
    if not path.exists():
        raise FileNotFoundError(f"Aggregated curve file not found: {path}")

    data = pd.read_csv(path, comment="#")

    required_columns = {
        "Date",
        "Hour",
        "Quarter hour",
        "Price",
        "Volume",
        "Sale/Purchase",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing_columns)}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%d/%m/%Y",
        errors="raise",
    )

    for column in ("Hour", "Quarter hour", "Price", "Volume"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["Hour"] = data["Hour"].astype(int)
    data["Quarter hour"] = data["Quarter hour"].astype(int)
    data["Sale/Purchase"] = data["Sale/Purchase"].astype(str).str.strip()

    valid_sides = {"Sell", "Purchase"}
    invalid_sides = set(data["Sale/Purchase"]).difference(valid_sides)

    if invalid_sides:
        raise ValueError(
            f"{path} contains invalid Sale/Purchase values: "
            f"{sorted(invalid_sides)}"
        )

    if not data["Hour"].between(1, 24).all():
        raise ValueError(f"{path} contains invalid Hour values.")

    if not data["Quarter hour"].between(1, 4).all():
        raise ValueError(f"{path} contains invalid Quarter hour values.")

    if (data["Volume"] < 0).any():
        raise ValueError(
            f"{path} contains negative cumulative curve volumes."
        )

    data["time"] = (
        data["Date"]
        + pd.to_timedelta(data["Hour"] - 1, unit="h")
        + pd.to_timedelta(
            (data["Quarter hour"] - 1) * 15,
            unit="min",
        )
    )

    return data.sort_values(
        ["time", "Sale/Purchase", "Price", "Volume"]
    )


def plot_curve_side(
    axis: plt.Axes,
    curve: pd.DataFrame,
    side: str,
) -> None:
    """
    Plot one cumulative curve side.

    The exported curve contains repeated points at individual bid prices,
    which produce the horizontal and vertical EPEX-style steps directly.
    """
    side_data = curve.loc[
        curve["Sale/Purchase"] == side
    ].copy()

    if side_data.empty:
        return

    side_data = side_data.sort_values(
        ["Price", "Volume"],
        ascending=[True, side == "Sell"],
        kind="stable",
    )

    color = "tab:green" if side == "Sell" else "tab:red"
    label = "Supply (Sell)" if side == "Sell" else "Demand (Purchase)"

    axis.plot(
        side_data["Volume"],
        side_data["Price"],
        color=color,
        linewidth=1.5,
        label=label,
    )


def plot_mtu_curve(
    mtu_data: pd.DataFrame,
    zone: str,
    scenario_name: str,
    output_path: Path,
) -> None:
    """Create and save one supply-demand plot for one zone and MTU."""
    timestamp = mtu_data["time"].iloc[0]

    figure, axis = plt.subplots(figsize=(11, 7))

    plot_curve_side(axis, mtu_data, "Sell")
    plot_curve_side(axis, mtu_data, "Purchase")

    axis.set_title(
        "Aggregated Demand and Supply Curves\n"
        f"Scenario: {scenario_name} | Zone: {zone} | "
        f"{timestamp:%d/%m/%Y %H:%M}",
        fontweight="bold",
    )

    axis.set_xlabel("Cumulative volume [MW]")
    axis.set_ylabel("Price [EUR/MWh]")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="best")

    if X_LIMIT is not None:
        axis.set_xlim(X_LIMIT)

    if Y_LIMIT is not None:
        axis.set_ylim(Y_LIMIT)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(figure)


def zone_name_from_file(path: Path) -> str:
    """Extract the zone identifier from aggregated_curves_<zone>.csv."""
    prefix = "aggregated_curves_"

    if not path.stem.startswith(prefix):
        raise ValueError(
            f"Unexpected aggregated curve filename: {path.name}"
        )

    return path.stem.removeprefix(prefix)


def plot_scenario(scenario_dir: Path) -> int:
    """
    Generate every zone/MTU curve plot in one scenario directory.

    Returns:
        Number of created plots.
    """
    curve_files = sorted(
        scenario_dir.glob("aggregated_curves_*.csv")
    )

    if not curve_files:
        raise FileNotFoundError(
            f"No aggregated_curves_<zone>.csv files found in: "
            f"{scenario_dir}"
        )

    output_dir = scenario_dir / "figures" / "curve_plots"
    scenario_name = scenario_dir.name
    plot_count = 0

    print(f"\nProcessing scenario: {scenario_name}")

    for curve_file in curve_files:
        zone = zone_name_from_file(curve_file)
        curves = load_aggregated_curves(curve_file)

        timestamps = sorted(curves["time"].unique())

        print(
            f"  Zone '{zone}': creating {len(timestamps)} MTU curve plots."
        )

        for timestamp in timestamps:
            mtu_data = curves.loc[
                curves["time"] == timestamp
            ].copy()

            output_path = (
                output_dir
                / safe_filename(zone)
                / (
                    f"{pd.Timestamp(timestamp):%Y-%m-%d_}"
                    f"H{pd.Timestamp(timestamp):%H}_"
                    f"Q{pd.Timestamp(timestamp).minute // 15 + 1}.png"
                )
            )

            plot_mtu_curve(
                mtu_data=mtu_data,
                zone=zone,
                scenario_name=scenario_name,
                output_path=output_path,
            )

            plot_count += 1

    print(
        f"Created {plot_count} curve plots in: {output_dir}"
    )

    return plot_count


def main() -> None:
    """Create all aggregated curve plots for both result scenarios."""
    total_plots = 0

    for scenario_dir in SCENARIO_DIRS:
        total_plots += plot_scenario(scenario_dir)

    print(f"\nFinished. Created {total_plots} curve plots in total.")


if __name__ == "__main__":
    main()