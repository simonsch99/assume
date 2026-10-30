"""
Run preprocessing, zonal order-book creation, and FBMC market clearing.

Edit the CONFIGURATION block below and run this file directly
(for example with the "Execute file" button). No command-line arguments are used.
`run_pipeline()` can still be imported from other scripts such as run_ieee118.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pypsa

from tools.clearing_test_tool import ClearingTestTool
from tools.create_zonal_orderbooks import (
    create_zonal_orderbooks,
    normalize_zone_label,
)
from tools.preprocess_orders import preprocess_orders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - edit here
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

SIMPLE_ORDERS = BASE_DIR / "data" / "auction_15_aggregated_curves_germany_luxembourg_20251001.csv"
BLOCK_ORDERS = BASE_DIR / "data" / "all_exchanges_bbof_15_germany_luxembourg_20251001.csv"
NETWORK_FILE = BASE_DIR / "data" / "filtered_network.nc"
OUTPUT_DIR = BASE_DIR / "output" / "three_zone_run"

ZONE_COLUMN = "zone"        # bus column holding the bidding-zone identifier
SOURCE_ZONE = "DE-LU"       # zone label assigned to the published order book
DELIVERY_DATE = datetime(2025, 10, 1)        # e.g. datetime(2025, 10, 1); None = infer from simple orders
PRICES_CSV = BASE_DIR / "data" / "auction_spot_15_prices_germany_luxembourg_2025.csv"           # optional: BASE_DIR / "data" / "clearing_prices.csv"
SOLVER_NAME = "gurobi"      # e.g. "gurobi" or "highs"
ALIGN_SNAPSHOTS = True # reassign network timestamps to DELIVERY_DATE
# =============================================================================


def infer_delivery_date(simple_orders: str | Path) -> datetime:
    """Infer the unique delivery date from the raw published simple orders."""
    simple = pd.read_csv(
        simple_orders,
        comment="#",
        encoding="utf-8-sig",
    )
    simple.columns = [str(column).strip() for column in simple.columns]

    if "Date" not in simple.columns:
        raise ValueError(
            "The published simple-order input must contain a 'Date' column."
        )

    dates = pd.to_datetime(
        simple["Date"],
        dayfirst=True,
        errors="raise",
    ).dt.normalize().unique()

    if len(dates) != 1:
        raise ValueError(
            "The simple-order input must contain exactly one delivery date. "
            f"Found: {list(dates)}"
        )

    return pd.Timestamp(dates[0]).to_pydatetime()

def align_network_snapshots(
    network: pypsa.Network,
    delivery_date: datetime,
    periods: int = 96,
    freq: str = "15min",
) -> None:
    """
    Assign the delivery-day timestamps to the network, in place.

    Snapshot i receives the timestamp delivery_date + i * freq. All
    time-dependent data (loads_t, generators_t, ...) keep their values; only
    the time index is replaced.
    """
    old_snapshots = network.snapshots

    if isinstance(old_snapshots, pd.MultiIndex):
        raise ValueError(
            "Multi-period networks (investment periods) are not supported."
        )

    if len(old_snapshots) != periods:
        raise ValueError(
            f"The network has {len(old_snapshots)} snapshots; exactly "
            f"{periods} are required to map them onto the delivery day."
        )

    if not old_snapshots.is_monotonic_increasing:
        raise ValueError(
            "Network snapshots are not in chronological order. Positional "
            "reassignment would shuffle the time series."
        )

    first = pd.Timestamp(old_snapshots[0])
    if first != first.normalize():
        logger.warning(
            "First network snapshot is %s (not 00:00). Positional mapping "
            "assumes snapshot 1 = first quarter hour of the day.",
            first,
        )

    new_snapshots = pd.date_range(
        start=pd.Timestamp(delivery_date).normalize(),
        periods=periods,
        freq=freq,
        name=old_snapshots.name,
    )

    if old_snapshots.equals(new_snapshots):
        return

    load_before = (
        network.loads_t.p_set.to_numpy(copy=True)
        if not network.loads_t.p_set.empty
        else None
    )

    # 1) Replace the index of every time-dependent table (values untouched).
    for component in network.components:
        tables = getattr(component, "dynamic", None)
        if tables is None:
            tables = component.pnl  # older PyPSA versions

        for table in tables.values():
            if len(table.index) == periods:
                table.index = new_snapshots

    network.snapshot_weightings.index = new_snapshots

    # 2) Synchronise PyPSA's internal snapshot state. All tables already carry
    #    the new index, so the internal reindex does not alter any values.
    network.set_snapshots(new_snapshots)

    # 3) Safety check: load time series must be unchanged.
    if load_before is not None:
        load_after = network.loads_t.p_set.to_numpy()
        if load_after.shape != load_before.shape or not (
            (load_after == load_before) | (pd.isna(load_after) & pd.isna(load_before))
        ).all():
            raise RuntimeError(
                "Load time series changed during snapshot reassignment."
            )

    logger.info(
        "Network snapshots reassigned: %s ... %s -> %s ... %s",
        old_snapshots[0],
        old_snapshots[-1],
        new_snapshots[0],
        new_snapshots[-1],
    )

def normalize_network_zone_labels(
    network: pypsa.Network,
    zone_column: str,
) -> None:
    """Normalize numeric zone values to unambiguous labels such as Z1."""
    if zone_column not in network.buses.columns:
        raise ValueError(
            f"Network buses do not contain zone column '{zone_column}'."
        )

    network.buses[zone_column] = network.buses[zone_column].map(
        normalize_zone_label
    )


def validate_network_snapshots(
    network: pypsa.Network,
    delivery_date: datetime,
) -> None:
    """Require all 96 quarter-hourly snapshots for the delivery day."""
    required = pd.date_range(
        start=pd.Timestamp(delivery_date).normalize(),
        periods=96,
        freq="15min",
    )

    missing = required.difference(pd.Index(network.snapshots))

    if not missing.empty:
        raise ValueError(
            "The PyPSA network must contain all 96 15-minute snapshots for "
            f"{pd.Timestamp(delivery_date).date()}. Missing: {list(missing)}"
        )


def run_pipeline(
    simple_orders: str | Path,
    block_orders: str | Path,
    network: pypsa.Network | str | Path,
    output_dir: str | Path,
    delivery_date: datetime | None = None,
    zone_column: str = "zone",
    source_zone: str = "DE-LU",
    prices_csv: str | Path | None = None,
    solver_name: str = "gurobi",
    align_snapshots: bool = True,    
) -> dict:
    """
    Run the complete published-data-to-clearing-results pipeline.

    `prices_csv` is optional. When omitted, simple orders are aggregated only
    at exact native price levels instead of clearing-price-centred bins.
    """
    output_dir = Path(output_dir)
    intermediate_dir = output_dir / "intermediate"
    result_dir = output_dir / "results"

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(network, pypsa.Network):
        pypsa_network = network
    else:
        pypsa_network = pypsa.Network(str(network))

    if delivery_date is None:
        delivery_date = infer_delivery_date(simple_orders)

    delivery_date = pd.Timestamp(delivery_date).normalize().to_pydatetime()

    normalize_network_zone_labels(pypsa_network, zone_column)
    if align_snapshots:
        align_network_snapshots(pypsa_network, delivery_date)
    validate_network_snapshots(pypsa_network, delivery_date)

    preprocessed_simple = intermediate_dir / "preprocessed_simple_orders.csv"
    preprocessed_blocks = intermediate_dir / "preprocessed_block_orders.csv"
    zonal_simple = intermediate_dir / "zonal_simple_orders.csv"
    zonal_blocks = intermediate_dir / "zonal_block_orders.csv"

    logger.info("Stage 1/4: preprocessing published orders.")
    preprocess_orders(
        simple_input=simple_orders,
        block_input=block_orders,
        simple_output=preprocessed_simple,
        block_output=preprocessed_blocks,
        source_zone=source_zone,
        prices_csv=prices_csv,
    )

    logger.info("Stage 2/4: creating zonal order books.")
    create_zonal_orderbooks(
        simple_input=preprocessed_simple,
        block_input=preprocessed_blocks,
        network=pypsa_network,
        zone_column=zone_column,
        simple_output=zonal_simple,
        block_output=zonal_blocks,
    )

    flowbased_region = sorted(
        pypsa_network.buses[zone_column].dropna().unique(),
        key=str,
    )

    logger.info("Stage 3/4: loading zonal orders.")
    tool = ClearingTestTool(
        market_id="fbmc_market",
        network=pypsa_network,
        flowbased_region=flowbased_region,
        zones_identifier=zone_column,
        param_dict={
            "solver_name": solver_name,
            "log_flows": True,
        },
    )

    tool.load_orders(
        delivery_date=delivery_date,
        quarter_hourly_csv=zonal_simple,
        block_csv=zonal_blocks,
        default_node=flowbased_region[0],
    )

    logger.info("Stage 4/4: clearing market.")
    results = tool.run_clearing()

    tool.print_summary()
    tool.export_results(result_dir)

    logger.info("Pipeline completed. Results written to %s", result_dir)
    return results


def main() -> None:
    run_pipeline(
        simple_orders=SIMPLE_ORDERS,
        block_orders=BLOCK_ORDERS,
        network=NETWORK_FILE,
        output_dir=OUTPUT_DIR,
        delivery_date=DELIVERY_DATE,
        zone_column=ZONE_COLUMN,
        source_zone=SOURCE_ZONE,
        prices_csv=PRICES_CSV,
        solver_name=SOLVER_NAME,
        align_snapshots=ALIGN_SNAPSHOTS,
    )


if __name__ == "__main__":
    main()