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
from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

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

SIMPLE_ORDERS = BASE_DIR / "data" / "auction_15_aggregated_curves_germany_luxembourg_20260120.csv"
BLOCK_ORDERS = BASE_DIR / "data" / "all_exchanges_bbof_15_germany_luxembourg_20260120.csv"
NETWORK_FILE = BASE_DIR / "data" / "filtered_network.nc"
OUTPUT_DIR = BASE_DIR / "output" / "three_zone_run"
PRICES_CSV = BASE_DIR / "data" / "auction_spot_15_prices_germany_luxembourg_2026.csv"           # optional: BASE_DIR / "data" / "clearing_prices.csv"


ZONE_COLUMN = "zone"        # bus column holding the bidding-zone identifier
SOURCE_ZONE = "DE-LU"       # zone label assigned to the published order book
DELIVERY_DATE = datetime(2026, 1, 20)        # e.g. datetime(2025, 10, 1); None = infer from simple orders
SOLVER_NAME = "gurobi"      # e.g. "gurobi" or "highs"
ALIGN_SNAPSHOTS = True # reassign network timestamps to DELIVERY_DATE


# Storage Order Config
STORAGE_ORDER_COUNT = 5       # 0 disables storage additions
STORAGE_ZONE = "Z1"           # must exist in the normalized network zones

# Energy: MWh; power: MW; costs and terminal value: EUR/MWh.
# Each operating limit accepts either a scalar or a complete mapping
# {market_product_start_timestamp: value}.
STORAGE_CONFIG = {
    "max_energy": 2000.0,
    "min_energy": 0.0,
    "max_input_power": 250.0,
    "max_output_power": 250.0,
    "starting_energy": 0.0,
    "efficiency": 0.86,
    "cycling_cost": 10.0,
    "end_of_day_value": 0.0,
    "bid_id_prefix": "SO",
    "agent_addr": None,       # None -> (f"agent_{zone}", "test")
}

WITHOUT_STORAGE_LABEL = "without_storage"
WITH_STORAGE_LABEL = "with_storage"
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

def add_storage_orders(
    tool: ClearingTestTool,
    zone: str,
    number_of_orders: int,
    *,
    max_energy: float | Mapping,
    max_input_power: float | Mapping,
    max_output_power: float | Mapping,
    cycling_cost: float,
    min_energy: float | Mapping = 0.0,
    starting_energy: float = 0.0,
    efficiency: float = 1.0,
    end_of_day_value: float = 0.0,
    bid_id_prefix: str = "SO",
    agent_addr: tuple[str, str] | None = None,
) -> list[dict]:
    """
    Append identically configured storage orders to a loaded zonal orderbook.

    Operating limits accept a scalar, expanded across all market MTUs, or a
    timestamp-to-value mapping covering exactly the loaded market horizon.

    Each order has independent operating-limit dictionaries and a unique ID.
    Returns the newly added orders.
    """
    if (
        isinstance(number_of_orders, bool)
        or not isinstance(number_of_orders, int)
        or number_of_orders < 0
    ):
        raise ValueError("number_of_orders must be a non-negative integer.")

    if number_of_orders == 0:
        return []

    if not tool.orderbook or not tool.market_products:
        raise ValueError("Call tool.load_orders() before adding storage orders.")

    if zone is None or not str(zone).strip():
        raise ValueError("zone must be a non-empty zone label.")

    zone = normalize_zone_label(zone)

    if tool.network is not None:
        known_zones = {
            str(value)
            for value in tool.network.buses[
                tool.zones_identifier
            ].dropna().unique()
        }
    else:
        known_zones = {
            str(order["node"]) for order in tool.orderbook
        } | {str(value) for value in tool.flowbased_region}

    if zone not in known_zones:
        raise ValueError(
            f"Unknown storage zone '{zone}'. "
            f"Known zones: {sorted(known_zones)}"
        )

    products = sorted(tool.market_products, key=lambda product: product[0])
    timestamps = [product[0] for product in products]

    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Market products contain duplicate start timestamps.")

    def numeric(value, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric.") from exc

        if not isfinite(result):
            raise ValueError(f"{field} must be finite.")

        return result

    def operating_limit(value, field: str) -> dict:
        if isinstance(value, Mapping):
            missing = [time for time in timestamps if time not in value]
            extra = [time for time in value if time not in set(timestamps)]

            if missing or extra:
                raise ValueError(
                    f"{field} must cover exactly all market MTUs. "
                    f"Missing: {missing[:5]}; extra: {extra[:5]}."
                )

            result = {
                time: numeric(value[time], f"{field}[{time}]")
                for time in timestamps
            }
        else:
            scalar = numeric(value, field)
            result = {time: scalar for time in timestamps}

        if any(limit < 0.0 for limit in result.values()):
            raise ValueError(f"{field} must be non-negative.")

        return result

    limits = {
        "max_energy": operating_limit(max_energy, "max_energy"),
        "min_energy": operating_limit(min_energy, "min_energy"),
        "max_input_power": operating_limit(
            max_input_power, "max_input_power"
        ),
        "max_output_power": operating_limit(
            max_output_power, "max_output_power"
        ),
    }

    for time in timestamps:
        if limits["min_energy"][time] > limits["max_energy"][time]:
            raise ValueError(
                f"min_energy exceeds max_energy at '{time}'."
            )

    static_values = {
        "starting_energy": numeric(starting_energy, "starting_energy"),
        "efficiency": numeric(efficiency, "efficiency"),
        "cycling_cost": numeric(cycling_cost, "cycling_cost"),
        "end_of_day_value": numeric(end_of_day_value, "end_of_day_value"),
    }

    if static_values["starting_energy"] < 0.0:
        raise ValueError("starting_energy must be non-negative.")

    if not 0.0 <= static_values["efficiency"] <= 1.0:
        raise ValueError("efficiency must be in [0, 1].")

    if static_values["cycling_cost"] < 0.0:
        raise ValueError("cycling_cost must be non-negative.")

    bid_id_prefix = str(bid_id_prefix).strip()
    if not bid_id_prefix:
        raise ValueError("bid_id_prefix must not be empty.")

    if agent_addr is None:
        agent_addr = (f"agent_{zone}", "test")
    elif (
        not isinstance(agent_addr, tuple)
        or len(agent_addr) != 2
        or not all(isinstance(value, str) for value in agent_addr)
    ):
        raise ValueError("agent_addr must be a tuple of two strings.")

    existing_ids = {order["bid_id"] for order in tool.orderbook}
    added_orders = []
    candidate_index = 1

    for _ in range(number_of_orders):
        while True:
            bid_id = f"{bid_id_prefix}_{zone}_{candidate_index:04d}"
            candidate_index += 1
            if bid_id not in existing_ids:
                break

        existing_ids.add(bid_id)

        added_orders.append(
            {
                "bid_id": bid_id,
                "agent_addr": agent_addr,
                "bid_type": "SO",
                "node": zone,
                "price": 0.0,
                "volume": {},
                "start_time": products[0][0],
                "end_time": products[-1][1],
                "only_hours": None,
                "min_acceptance_ratio": None,
                "parent_bid_id": None,
                "exclusive_group_id": None,
                **static_values,
                **{
                    field: values.copy()
                    for field, values in limits.items()
                },
            }
        )

    # Only modify the orderbook after the full configuration has been validated.
    tool.orderbook.extend(added_orders)
    tool.submitted_order_count = len(tool.orderbook)

    logger.info(
        "Added %s identically configured storage orders to zone '%s'.",
        len(added_orders),
        zone,
    )

    return added_orders

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
    storage_order_count: int = STORAGE_ORDER_COUNT,
    storage_zone: str = STORAGE_ZONE,
    storage_config: Mapping | None = None,
) -> dict:
    """
    Run preprocessing and clearing without storage, then with storage.

    `prices_csv` is required for historical MCP demand calibration.
    Returns results keyed by the two scenario labels.
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
        prices_csv=prices_csv,
        delivery_date=delivery_date,

        # Diagnostics and legacy report outputs.
        diagnostics_dir=intermediate_dir / "zonal_diagnostics",
        supply_report_output=intermediate_dir / "supply_validation.csv",
        demand_report_output=intermediate_dir / "demand_validation.csv",

        # Optional provenance: use the actual pipeline input, not a global.
        network_input=(
            None if isinstance(network, pypsa.Network) else network
        ),
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
    # Preserve an independent storage-free baseline before adding storage.
    baseline_orderbook = deepcopy(
        [
            order
            for order in tool.orderbook
            if order.get("bid_type") != "SO"
        ]
    )

    add_storage_orders(
        tool=tool,
        zone=storage_zone,
        number_of_orders=storage_order_count,
        **dict(STORAGE_CONFIG if storage_config is None else storage_config),
    )

    storage_orderbook = tool.orderbook

    scenario_results = {}

    for label, orderbook in (
        (WITHOUT_STORAGE_LABEL, baseline_orderbook),
        (WITH_STORAGE_LABEL, storage_orderbook),
    ):
        tool.orderbook = orderbook
        tool.submitted_order_count = len(orderbook)

        logger.info("Stage 4/4: clearing market [%s].", label)

        # run_clearing() creates a fresh clearing role and works on a deep
        # copy of the submitted orderbook, isolating the two scenarios.
        scenario_results[label] = tool.run_clearing()

        tool.print_summary()
        tool.export_results(result_dir / label)

    logger.info(
        "Pipeline completed. Scenario results written to %s.",
        result_dir,
    )

    return scenario_results


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
        storage_order_count=STORAGE_ORDER_COUNT,
        storage_zone=STORAGE_ZONE,
        storage_config=STORAGE_CONFIG,
    )


if __name__ == "__main__":
    main()