"""Plot baseline and storage-inclusive clearing prices for each zone."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================
TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent

RESULTS_DIR = PROJECT_DIR / "output" / "three_zone_run" / "results"

BASELINE_CSV = RESULTS_DIR / "without_storage" / "market_meta.csv"
STORAGE_CSV = RESULTS_DIR / "with_storage" / "market_meta.csv"
PLOT_DIR = RESULTS_DIR / "price_comparison_plots"

SHOW_PLOTS = True
# =============================================================================


def load_prices(path: Path) -> pd.DataFrame:
    """Read exported zonal prices and validate zone/timestamp uniqueness."""
    data = pd.read_csv(path, dtype={"zone": str})

    required = {"zone", "product_start", "price"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    data = data[["zone", "product_start", "price"]].copy()
    data["product_start"] = pd.to_datetime(
        data["product_start"], errors="raise"
    )
    data["price"] = pd.to_numeric(data["price"], errors="raise")

    if data.isna().any().any():
        raise ValueError(f"{path} contains missing zones, timestamps, or prices.")

    if data.duplicated(["zone", "product_start"]).any():
        raise ValueError(f"{path} contains duplicate zone/timestamp rows.")

    return data.sort_values(["zone", "product_start"])


def plot_price_comparison(
    baseline_csv: Path,
    storage_csv: Path,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Create a separate two-curve price comparison for each bidding zone."""
    baseline = load_prices(baseline_csv)
    storage = load_prices(storage_csv)

    comparison = baseline.merge(
        storage,
        on=["zone", "product_start"],
        how="outer",
        suffixes=("_baseline", "_storage"),
        indicator=True,
        validate="one_to_one",
    )

    if (comparison["_merge"] != "both").any():
        raise ValueError(
            "The scenarios must contain the same zones and delivery timestamps."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for index, zone in enumerate(sorted(comparison["zone"].unique()), start=1):
        prices = comparison.loc[
            comparison["zone"] == zone
        ].sort_values("product_start")

        # Include the final MTU end so the last quarter-hour is fully drawn.
        timestamps = prices["product_start"].tolist()
        timestamps.append(timestamps[-1] + pd.Timedelta(minutes=15))

        fig, ax = plt.subplots(figsize=(12, 5))

        for column, label, color, style in (
            ("price_baseline", "Without storage", "tab:blue", "-"),
            ("price_storage", "With storage", "tab:orange", "--"),
        ):
            values = prices[column].tolist()
            ax.step(
                timestamps,
                values + [values[-1]],
                where="post",
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.8,
            )

        ax.set_title(f"Clearing price comparison — {zone}")
        ax.set_xlabel("Delivery time")
        ax.set_ylabel("Price [EUR/MWh]")
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.autofmt_xdate()
        fig.tight_layout()

        safe_zone = re.sub(r"[^A-Za-z0-9_-]+", "_", zone)
        output_path = output_dir / f"{index:02d}_prices_{safe_zone}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path}")

        if not show:
            plt.close(fig)

    if show:
        plt.show()


if __name__ == "__main__":
    plot_price_comparison(
        baseline_csv=BASELINE_CSV,
        storage_csv=STORAGE_CSV,
        output_dir=PLOT_DIR,
        show=SHOW_PLOTS,
    )