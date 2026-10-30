"""Run EUPHEMIA-like market clearing for the three-zone FBMC-only test system.

This script converts the zonal simple- and block-order CSVs produced by
`create_zonal_order_books.py` into the flat layout expected by
`clearing_test_tool.py`, aligns the test network's snapshots to the real
15-minute delivery timestamps, and clears the three-zone FBMC-only system.
"""

from datetime import datetime
from pathlib import Path
import csv
import logging

import pandas as pd
import pypsa

from Thesis_BESS.tools.clearing_test_tool import ClearingTestTool

logger = logging.getLogger(__name__)

# Configuration
TEST_DATA_DIR = Path("assume/testing_tools/test_data")
NETWORK_PATH = TEST_DATA_DIR / "filtered_network.nc"
ZONAL_SIMPLE_CSV = TEST_DATA_DIR / "zonal_simple_orders.csv"
ZONAL_BLOCK_CSV = TEST_DATA_DIR / "zonal_block_orders.csv"

OUTPUT_DIR = Path("assume/testing_tools/ieee_clearing_results")
LAYOUT_SIMPLE_CSV = OUTPUT_DIR / "layout_simple_orders.csv"
LAYOUT_BLOCK_CSV = OUTPUT_DIR / "layout_block_orders.csv"

DELIVERY_DATE = datetime(2025, 10, 1)
MARKET_ID = "three_zone_fbmc"
ZONES_IDENTIFIER = "zone"
DEFAULT_NODE = "1"

PROFILE_COLUMNS = [f"Q_{index:03d}" for index in range(1, 97)]
REQUIRED_SNAPSHOT_COUNT = 96
MTU_FREQUENCY = "15min"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def normalize_zone(value) -> str:
    """Return a plain integer string for numeric-looking zone labels."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()

DEFAULT_NODE = "Z1"

def align_network_snapshots(network: pypsa.Network, delivery_date: datetime) -> None:
    """
    Relabel the network's 96 snapshots to real 15-minute MTU timestamps,
    without losing any time-varying data.

    pypsa.Network.set_snapshots() reindexes every component's time-varying
    dataframe against the new snapshot index. Because the new labels share
    no values with the old ones, that reindex would silently replace every
    cell with its component default (e.g. 0.0 for loads_t.p_set) before we
    get a chance to relabel it. To avoid this, all time-varying dataframes
    are saved first and reassigned by position after set_snapshots().
    """
    old_snapshots = network.snapshots

    if len(old_snapshots) != REQUIRED_SNAPSHOT_COUNT:
        raise ValueError(
            f"Expected {REQUIRED_SNAPSHOT_COUNT} snapshots, "
            f"found {len(old_snapshots)}."
        )

    new_snapshots = pd.date_range(
        start=delivery_date, periods=REQUIRED_SNAPSHOT_COUNT, freq=MTU_FREQUENCY
    )

    # 1. Save every non-empty time-varying dataframe BEFORE touching snapshots.
    saved_time_series = {}
    for component in network.iterate_components():
        pnl = network.pnl(component.name)
        for attribute_name, time_series in pnl.items():
            if not time_series.empty:
                saved_time_series[(component.name, attribute_name)] = (
                    time_series.copy()
                )

    # 2. Now it is safe to change the snapshot index.
    network.set_snapshots(new_snapshots)

    # 3. Restore the saved data with the new index, by position.
    for (component_name, attribute_name), original_series in saved_time_series.items():
        if len(original_series.index) != REQUIRED_SNAPSHOT_COUNT:
            raise ValueError(
                f"Component '{component_name}' attribute '{attribute_name}' "
                f"has {len(original_series.index)} rows, expected "
                f"{REQUIRED_SNAPSHOT_COUNT}."
            )

        restored_series = original_series.copy()
        restored_series.index = new_snapshots

        pnl = network.pnl(component_name)
        pnl[attribute_name] = restored_series

    network.snapshot_weightings.index = new_snapshots

# Target: must-run floor must never exceed this fraction of minimum load.
MUST_RUN_SAFETY_MARGIN = 0.85


def rescale_must_run_floor(
    network: pypsa.Network,
    safety_margin: float = MUST_RUN_SAFETY_MARGIN,
) -> None:
    """
    Proportionally scale down generator p_min_pu to guarantee D-2 feasibility.

    Must be called AFTER snapshots are aligned and load is correctly injected
    (network.loads_t.p_set populated), and BEFORE run_clearing() /
    precompute_fbmc_parameters().
    """
    if not 0.0 < safety_margin <= 1.0:
        raise ValueError("safety_margin must be in (0, 1].")

    static_load = network.loads["p_set"].fillna(0.0)
    load = pd.DataFrame(
        [static_load.to_numpy()] * len(network.snapshots),
        index=network.snapshots,
        columns=network.loads.index,
    )
    if not network.loads_t.p_set.empty:
        dynamic_load = network.loads_t.p_set.reindex(
            index=network.snapshots, columns=network.loads.index
        )
        load = dynamic_load.fillna(load)
    total_load_per_snapshot = load.sum(axis=1)

    min_load = float(total_load_per_snapshot.min())
    target_floor = safety_margin * min_load

    must_run_mask = network.generators["p_min_pu"].fillna(0.0) > 0
    current_floor = (
        network.generators.loc[must_run_mask, "p_min_pu"]
        * network.generators.loc[must_run_mask, "p_nom"]
    ).sum()

    if current_floor <= target_floor:
        logger.info(
            "Must-run floor (%.1f MW) already fits within %.0f%% of "
            "minimum load (%.1f MW); no rescaling needed.",
            current_floor, safety_margin * 100, min_load,
        )
        return

    scale_factor = target_floor / current_floor
    network.generators.loc[must_run_mask, "p_min_pu"] *= scale_factor

    new_floor = (
        network.generators.loc[must_run_mask, "p_min_pu"]
        * network.generators.loc[must_run_mask, "p_nom"]
    ).sum()

    logger.info(
        "Rescaled must-run p_min_pu by factor %.4f: floor %.1f MW -> "
        "%.1f MW (target <= %.1f MW, min load = %.1f MW).",
        scale_factor, current_floor, new_floor, target_floor, min_load,
    )    

def load_fbmc_network(path: Path, delivery_date: datetime) -> pypsa.Network:
    """Load the network, align its snapshots, and normalise zone labels."""
    network = pypsa.Network(str(path))

    if ZONES_IDENTIFIER not in network.buses.columns:
        raise ValueError(
            f"Network buses do not contain zone column '{ZONES_IDENTIFIER}'."
        )

    network.buses[ZONES_IDENTIFIER] = network.buses[ZONES_IDENTIFIER].map(
        normalize_zone
    )

    align_network_snapshots(network, delivery_date)
    rescale_must_run_floor(network)

    log.info(
        "Loaded network with zones: %s",
        sorted(network.buses[ZONES_IDENTIFIER].unique()),
    )
    return network


def convert_simple_orders(input_csv: Path, output_csv: Path) -> None:
    """Convert a zonal simple-order CSV to the clearing_test_tool layout."""
    orders = pd.read_csv(input_csv)

    required_columns = {
        "Zone", "Date", "Hour", "Quarter hour",
        "Price", "Volume", "Sale/Purchase",
    }
    missing_columns = required_columns.difference(orders.columns)
    if missing_columns:
        raise ValueError(
            f"Zonal simple-order CSV is missing columns: {sorted(missing_columns)}"
        )

    orders["bidding_zone"] = orders["Zone"].map(normalize_zone)
    orders["Week"] = ""
    orders["Week Day"] = ""

    layout_columns = [
        "Date", "Week", "Week Day", "Hour", "Quarter hour",
        "Price", "Volume", "Sale/Purchase", "bidding_zone",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    orders[layout_columns].to_csv(output_csv, index=False)
    log.info("Wrote %d simple orders to %s", len(orders), output_csv)


def convert_block_orders(input_csv: Path, output_csv: Path) -> None:
    """
    Convert a zonal block-order CSV to the clearing_test_tool BB layout.

    The reader negates the stored interval values, so sell (supply) volumes
    are written as negative and purchase (demand) volumes as positive.
    """
    blocks = pd.read_csv(input_csv)

    required_columns = {
        "Zone", "BlockID", "Block Type", "BlockCodePRM",
        "Limit Price", "Side", "MAR", *PROFILE_COLUMNS,
    }
    missing_columns = required_columns.difference(blocks.columns)
    if missing_columns:
        raise ValueError(
            f"Zonal block-order CSV is missing columns: {sorted(missing_columns)}"
        )

    rows = []

    for _, block in blocks.iterrows():
        sign = -1.0 if block["Side"] == "Sell" else 1.0
        intervals = [sign * float(block[column]) for column in PROFILE_COLUMNS]

        special_id = block["BlockCodePRM"]
        special_id = "" if pd.isna(special_id) else str(special_id)

        acc_ratio = block["MAR"]
        acc_ratio_field = "" if pd.isna(acc_ratio) else str(float(acc_ratio))

        row = [
            "BB",
            str(block["BlockID"]),
            str(block["Block Type"]),
            special_id,
            "",
            str(float(block["Limit Price"])),
            *[str(value) for value in intervals],
            acc_ratio_field,
            "",
            normalize_zone(block["Zone"]),
        ]

        if len(row) != 105:
            raise AssertionError(
                f"Block row for '{block['BlockID']}' has {len(row)} fields, "
                "expected 105."
            )

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        csv.writer(file, delimiter=";").writerows(rows)

    log.info("Wrote %d block orders to %s", len(rows), output_csv)


def main() -> None:
    network = load_fbmc_network(NETWORK_PATH, DELIVERY_DATE)
    flowbased_region = sorted(network.buses[ZONES_IDENTIFIER].unique())

    convert_simple_orders(ZONAL_SIMPLE_CSV, LAYOUT_SIMPLE_CSV)
    convert_block_orders(ZONAL_BLOCK_CSV, LAYOUT_BLOCK_CSV)

    tool = ClearingTestTool(
        market_id=MARKET_ID,
        network=network,
        flowbased_region=flowbased_region,
        zones_identifier=ZONES_IDENTIFIER,
    )

    tool.load_orders(
        delivery_date=DELIVERY_DATE,
        quarter_hourly_csv=LAYOUT_SIMPLE_CSV,
        block_csv=LAYOUT_BLOCK_CSV,
        default_node=DEFAULT_NODE,
    )

    tool.run_clearing()
    tool.print_summary()
    tool.export_results(OUTPUT_DIR)


if __name__ == "__main__":
    main()