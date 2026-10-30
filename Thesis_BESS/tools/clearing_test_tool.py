# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test tool for the EUPHEMIA-like market-clearing implementation.

Only fixed 15-minute day-ahead market time units (MTUs) are supported.

Supported input:
- Existing PyPSA network or PyPSA NetCDF path;
- EUPHEMIA-like 15-minute simple-order CSV;
- EUPHEMIA-like block-order CSV;
- NTC connection CSV/DataFrame;
- an explicit FBMC bidding-zone region.

The physical PyPSA network is used only by the FBMC preprocessing stage.
It is not used as a D-1 transport network.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from math import isfinite
from typing import Iterator

import pandas as pd
import pypsa
from dateutil import rrule as rr

from assume.common.market_objects import (
    MarketConfig,
    MarketProduct,
    Orderbook,
)
from clearing_algorithm.euphemia_like import (
    ComplexClearingRole as EuphemiaLikeClearingRole,
)

from tools.order_layout import native_profile_columns, read_block_layout

logger = logging.getLogger(__name__)

MTU_DURATION = timedelta(minutes=15)
MTU_DURATION_HOURS = 0.25


# ---------------------------------------------------------------------------
# EUPHEMIA-like CSV readers
# ---------------------------------------------------------------------------


def _non_comment_csv_lines(path: str | Path) -> Iterator[str]:
    """Yield non-empty, non-comment CSV lines."""
    with Path(path).open(newline="") as file:
        for line in file:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line


class QuarterHourlyOrderReader:
    """
    Read simple orders from layout_simple_orders.csv.

    Required header columns:
    Date, Week, Week Day, Hour, Quarter hour, Price, Volume, Sale/Purchase,
    bidding_zone

    ``bidding_zone`` is optional per row, but must be the trailing layout
    column. An empty zone is resolved by ClearingTestTool.default_node.
    """

    REQUIRED_COLUMNS = {
        "Date",
        "Week",
        "Week Day",
        "Hour",
        "Quarter hour",
        "Price",
        "Volume",
        "Sale/Purchase",
        "bidding_zone",
    }

    def read(self, path: str | Path) -> list[dict]:
        reader = csv.DictReader(_non_comment_csv_lines(path))
        reader.fieldnames = [
            (field or "").strip()
            for field in (reader.fieldnames or [])
        ]

        missing = self.REQUIRED_COLUMNS.difference(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Simple-order CSV is missing required columns: "
                f"{sorted(missing)}."
            )

        orders = []

        for line_number, row in enumerate(reader, start=2):
            try:
                hour = int(row["Hour"]) - 1
                quarter = int(row["Quarter hour"]) - 1
                interval = hour * 4 + quarter
                side = (row["Sale/Purchase"] or "").strip()

                if not 0 <= interval < 96:
                    raise ValueError(
                        "Hour and Quarter hour must identify an MTU in 1..96."
                    )

                if side not in {"Purchase", "Sell"}:
                    raise ValueError(
                        "Sale/Purchase must be either 'Purchase' or 'Sell'."
                    )

                orders.append(
                    {
                        "raw_id": f"simple_{line_number}",
                        "side": "BUY" if side == "Purchase" else "SELL",
                        "price": float(row["Price"]),
                        "quantity": float(row["Volume"]),
                        "interval": interval,
                        "bidding_zone": (
                            row.get("bidding_zone") or ""
                        ).strip(),
                    }
                )
            except (TypeError, ValueError) as error:
                logger.warning(
                    "Skipping simple-order row %s: %s",
                    line_number,
                    error,
                )

        logger.info("Read %s simple orders from %s.", len(orders), path)
        return orders

class BlockOrderReader:
    """Read native-layout EPEX-like block bids."""

    def read(self, path: str | Path) -> list[dict]:
        layout = read_block_layout(path)
        blocks = layout.frame.copy()
        profile_columns = native_profile_columns(blocks.columns)

        required = {
            "Data type(BB)",
            "BlockID",
            "Block Type",
            "BlockCodePRM",
            "Limit Price",
            "MAR",
            "bidding_zone",
        }
        missing = required.difference(blocks.columns)
        if missing:
            raise ValueError(
                f"Block-order CSV is missing columns: {sorted(missing)}"
            )

        result = []

        for row_number, (_, row) in enumerate(blocks.iterrows(), start=1):
            if str(row["Data type(BB)"]).strip() != "BB":
                continue

            try:
                result.append(
                    {
                        "raw_id": str(row["BlockID"]).strip(),
                        "type": str(row["Block Type"]).strip(),
                        "special_id": (
                            str(row["BlockCodePRM"]).strip() or None
                        ),
                        "price": float(row["Limit Price"]),
                        # Native convention:
                        # positive raw profile = sell;
                        # negative raw profile = purchase.
                        # ASSUME convention is the reverse sign.
                        "intervals": [
                            -float(row[column])
                            for column in profile_columns
                        ],
                        "acc_ratio": (
                            float(row["MAR"])
                            if str(row["MAR"]).strip()
                            else None
                        ),
                        "bidding_zone": (
                            str(row["bidding_zone"]).strip()
                        ),
                    }
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping block row %s: %s",
                    row_number,
                    exc,
                )

        return result

class StorageOrderReader:
    """
    Read wide-format storage orders.

    One row represents one storage order. Mandatory time-series columns are
    max_energy_01..96, max_input_power_01..96, and max_output_power_01..96.
    The min_energy_01..96 group is optional as a whole.
    """

    STATIC_COLUMNS = {
        "bid_id",
        "bidding_zone",
        "starting_energy",
        "efficiency",
        "cycling_cost",
        "end_of_day_value",
    }
    MANDATORY_SERIES = (
        "max_energy",
        "max_input_power",
        "max_output_power",
    )

    @staticmethod
    def _series_columns(prefix: str) -> list[str]:
        return [f"{prefix}_{interval:02d}" for interval in range(1, 97)]

    @staticmethod
    def _numeric(value, field: str, bid_id: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Storage Order '{bid_id}' field '{field}' must be numeric."
            ) from exc

        if not isfinite(number):
            raise ValueError(
                f"Storage Order '{bid_id}' field '{field}' must be finite."
            )

        return number

    def read(self, path: str | Path) -> list[dict]:
        reader = csv.DictReader(_non_comment_csv_lines(path))
        reader.fieldnames = [
            (field or "").strip()
            for field in (reader.fieldnames or [])
        ]
        columns = set(reader.fieldnames)

        if len(columns) != len(reader.fieldnames):
            raise ValueError("Storage-order CSV contains duplicate column names.")

        required = set(self.STATIC_COLUMNS)

        for prefix in self.MANDATORY_SERIES:
            required.update(self._series_columns(prefix))

        missing = required.difference(columns)

        if missing:
            raise ValueError(
                "Storage-order CSV is missing required columns: "
                f"{sorted(missing)}."
            )

        min_columns = self._series_columns("min_energy")
        present_min_columns = [column for column in min_columns if column in columns]

        if present_min_columns and len(present_min_columns) != 96:
            missing_min = sorted(set(min_columns).difference(columns))
            raise ValueError(
                "Storage-order CSV has an incomplete optional min_energy "
                f"series. Missing columns: {missing_min}."
            )

        orders = []
        bid_ids = set()

        for line_number, row in enumerate(reader, start=2):
            bid_id = (row.get("bid_id") or "").strip()

            if not bid_id:
                raise ValueError(
                    f"Storage-order CSV row {line_number} has an empty bid_id."
                )

            if bid_id in bid_ids:
                raise ValueError(
                    f"Storage-order CSV contains duplicate bid_id '{bid_id}'."
                )

            bid_ids.add(bid_id)

            static_values = {
                field: self._numeric(row.get(field), field, bid_id)
                for field in (
                    "starting_energy",
                    "efficiency",
                    "cycling_cost",
                    "end_of_day_value",
                )
            }

            parsed_series = {}

            for prefix in self.MANDATORY_SERIES:
                values = [
                    self._numeric(row.get(column), column, bid_id)
                    for column in self._series_columns(prefix)
                ]

                if any(value < 0 for value in values):
                    raise ValueError(
                        f"Storage Order '{bid_id}' series '{prefix}' must "
                        "contain only non-negative values."
                    )

                parsed_series[prefix] = values

            if present_min_columns:
                min_energy = [
                    self._numeric(row.get(column), column, bid_id)
                    for column in min_columns
                ]

                if any(value < 0 for value in min_energy):
                    raise ValueError(
                        f"Storage Order '{bid_id}' series 'min_energy' must "
                        "contain only non-negative values."
                    )
            else:
                min_energy = [0.0] * 96

            orders.append(
                {
                    "bid_id": bid_id,
                    "bidding_zone": (row.get("bidding_zone") or "").strip(),
                    **static_values,
                    **parsed_series,
                    "min_energy": min_energy,
                }
            )

        logger.info("Read %s storage orders from %s.", len(orders), path)
        return orders


# ---------------------------------------------------------------------------
# Clearing test tool
# ---------------------------------------------------------------------------


class ClearingTestTool:
    """
    Test tool for NTC-only, FBMC-only, and hybrid FBMC/NTC clearing.

    Only fixed 15-minute market products are supported.

    Example:

    ```python
    tool = ClearingTestTool(
        market_id="hybrid_market",
        network="hybrid_network.nc",
        flowbased_region=["Zone A", "Zone B", "Zone C"],
        ntc_connections="ntc_connections.csv",
    )

    tool.load_orders(
        delivery_date=datetime(2026, 1, 1),
        quarter_hourly_csv="quarter_hourly_orders.csv",
        block_csv="block_orders.csv",
    )

    tool.run_clearing()
    tool.print_summary()
    tool.export_results("results")
    ```
    """

    def __init__(
        self,
        market_id: str = "test_market",
        clearing_type: Literal["euphemia_like"] = "euphemia_like",
        product_duration: timedelta = MTU_DURATION,
        network: pypsa.Network | str | Path | None = None,
        flowbased_region: list[str] | None = None,
        ntc_connections: str | Path | pd.DataFrame | None = None,
        zones_identifier: str = "zone_id",
        **market_config_kwargs,
    ):
        if product_duration != MTU_DURATION:
            raise ValueError(
                "ClearingTestTool supports only 15-minute products "
                f"({MTU_DURATION})."
            )

        self.market_id = market_id
        self.clearing_type = clearing_type
        self.product_duration = MTU_DURATION
        self.market_config_kwargs = market_config_kwargs

        self.network: pypsa.Network | None = None
        self.flowbased_region = list(flowbased_region or [])
        self.ntc_connections = ntc_connections
        self.zones_identifier = zones_identifier

        self.delivery_date: datetime | None = None
        self.orderbook: Orderbook = []
        self.market_products: list[MarketProduct] = []
        self.clearing_algorithm = None
        self.submitted_order_count = 0

        self.results = {
            "accepted_orders": [],
            "rejected_orders": [],
            "market_meta": [],
            "flows": {},
            "line_results": [],
        }

        if network is not None:
            self.load_network(network)

    # ------------------------------------------------------------------
    # Network and coupling setup
    # ------------------------------------------------------------------

    def load_network(
        self,
        network: pypsa.Network | str | Path,
    ) -> pypsa.Network:
        """
        Load or assign an existing PyPSA network.

        The network must contain the selected zone column on its buses.
        Required 15-minute timestamps are validated after orders are loaded.
        """
        if isinstance(network, pypsa.Network):
            self.network = network

        elif isinstance(network, (str, Path)):
            path = Path(network)

            if not path.exists():
                raise FileNotFoundError(
                    f"PyPSA network file not found: {path}"
                )

            self.network = pypsa.Network(path)

        else:
            raise TypeError(
                "network must be a pypsa.Network or a PyPSA network path."
            )

        if self.zones_identifier not in self.network.buses.columns:
            raise ValueError(
                f"Network buses do not contain zone column "
                f"'{self.zones_identifier}'."
            )

        logger.info(
            "Loaded PyPSA network with %s buses, %s lines, and %s snapshots.",
            len(self.network.buses),
            len(self.network.lines),
            len(self.network.snapshots),
        )

        return self.network

    def set_coupling(
        self,
        flowbased_region: list[str] | None = None,
        ntc_connections: str | Path | pd.DataFrame | None = None,
    ) -> None:
        """Configure the FB region and optional virtual NTC connections."""
        if flowbased_region is not None:
            self.flowbased_region = list(flowbased_region)

        if ntc_connections is not None:
            self.ntc_connections = ntc_connections

    # ------------------------------------------------------------------
    # Order loading
    # ------------------------------------------------------------------

    def load_orders(
        self,
        delivery_date: datetime,
        quarter_hourly_csv: str | Path | None = None,
        block_csv: str | Path | None = None,
        storage_csv: str | Path | None = None,
        default_node: str = "node0",
    ) -> Orderbook:
        """Read simple, block, and/or wide-format storage-order files."""
        if (
            quarter_hourly_csv is None
            and block_csv is None
            and storage_csv is None
        ):
            raise ValueError(
                "At least one of quarter_hourly_csv, block_csv, or storage_csv "
                "must be supplied."
            )

        self.delivery_date = (
            pd.Timestamp(delivery_date).normalize().to_pydatetime()
        )
        orderbook: Orderbook = []

        if quarter_hourly_csv is not None:
            orderbook.extend(
                self._quarter_hourly_to_orderbook(
                    QuarterHourlyOrderReader().read(quarter_hourly_csv),
                    default_node,
                )
            )

        if block_csv is not None:
            orderbook.extend(
                self._blocks_to_orderbook(
                    BlockOrderReader().read(block_csv),
                    default_node,
                )
            )

        if storage_csv is not None:
            orderbook.extend(
                self._storage_to_orderbook(
                    StorageOrderReader().read(storage_csv),
                    default_node,
                )
            )

        if not orderbook:
            raise ValueError("No valid orders could be read from the input files.")

        self._validate_unique_bid_ids(orderbook)
        self._validate_linked_block_parents(orderbook)

        self.orderbook = orderbook
        self.submitted_order_count = len(orderbook)
        self.market_products = self._build_market_products()
        self._validate_network_market_product_snapshots()

        # Check that orders exist for each zone
        if self.network is not None:
            order_zones = {str(order["node"]) for order in orderbook}
            network_zones = {
                str(zone)
                for zone in self.network.buses[
                    self.zones_identifier
                ].dropna().unique()
            }

            unknown_zones = order_zones.difference(network_zones)

            if unknown_zones:
                raise ValueError(
                    "Order zones do not exist in the network: "
                    f"{sorted(unknown_zones)}"
                )

            if self.flowbased_region:
                missing_order_zones = set(self.flowbased_region).difference(order_zones)

                if missing_order_zones:
                    raise ValueError(
                        "The FBMC region contains zones without submitted orders: "
                        f"{sorted(missing_order_zones)}"
                    )
                
        logger.info(
            "Built orderbook with %s orders: %s simple, %s block, %s storage.",
            len(orderbook),
            sum(order["bid_type"] == "SB" for order in orderbook),
            sum(order["bid_type"] in {"BB", "LB"} for order in orderbook),
            sum(order["bid_type"] == "SO" for order in orderbook),
        )

        return self.orderbook

    def _interval_start(
        self,
        interval: int,
    ) -> datetime:
        """
        Return the start timestamp of a zero-based delivery interval.

        Interval zero corresponds to delivery interval 1 at 00:00.
        Interval 95 corresponds to delivery interval 96 at 23:45.
        """
        if self.delivery_date is None:
            raise RuntimeError("delivery_date was not set.")

        if not 0 <= interval < 96:
            raise ValueError(
                "Delivery interval index must be in 0..95, "
                f"got {interval}."
            )

        return self.delivery_date + interval * MTU_DURATION

    def _quarter_hourly_to_orderbook(
        self,
        quarter_hourly_orders: list[dict],
        default_node: str,
    ) -> Orderbook:
        """Convert parsed 15-minute rows to ASSUME simple orders."""
        orderbook: Orderbook = []

        for index, order in enumerate(quarter_hourly_orders):
            start_time = self._interval_start(order["interval"])

            # ASSUME convention:
            # positive volume = supply
            # negative volume = demand
            volume = (
                -order["quantity"]
                if order["side"] == "BUY"
                else order["quantity"]
            )

            orderbook.append(
                {
                    "bid_id": (
                        f"SB_{order['raw_id']}_"
                        f"{order['interval']:02d}_{index}"
                    ),
                    "agent_addr": (
                        f"agent_{order['bidding_zone'] or default_node}",
                        "test",
                    ),
                    "bid_type": "SB",
                    "price": order["price"],
                    "volume": volume,
                    "node": order["bidding_zone"] or default_node,
                    "start_time": start_time,
                    "end_time": start_time + MTU_DURATION,
                    "only_hours": None,
                    "min_acceptance_ratio": None,
                    "parent_bid_id": None,
                    "exclusive_group_id": None,
                }
            )

        return orderbook

    def _blocks_to_orderbook(
        self,
        blocks: list[dict],
        default_node: str,
    ) -> Orderbook:
        """
        Convert parsed block rows to ASSUME block orders.

        The volume mapping must be contiguous between start_time and end_time,
        including zero-volume MTUs. This is required by separate_orders(), which
        otherwise derives invalid timestamps for sparse block profiles.
        """
        orderbook: Orderbook = []

        for block in blocks:
            all_volumes = block["intervals"]
            active_intervals = [
                interval
                for interval, quantity in enumerate(all_volumes)
                if abs(quantity) > 0.0
            ]

            if not active_intervals:
                logger.warning(
                    "Skipping block '%s' because it has no non-zero volume.",
                    block["raw_id"],
                )
                continue

            first_interval = min(active_intervals)
            last_interval = max(active_intervals)

            # Preserve every MTU between the first and final active MTU,
            # including any zero-volume gaps within a block profile.
            volume = {
                self._interval_start(interval): all_volumes[interval]
                for interval in range(first_interval, last_interval + 1)
            }

            is_linked = (
                block["type"] == "C02"
                and block["special_id"] is not None
            )
            is_exclusive = (
                block["type"] == "C04"
                and block["special_id"] is not None
            )

            start_time = self._interval_start(first_interval)
            end_time = self._interval_start(last_interval) + MTU_DURATION

            orderbook.append(
                {
                    "bid_id": block["raw_id"],
                    "agent_addr": (
                        f"agent_{block['bidding_zone'] or default_node}",
                        "test",
                    ),
                    "bid_type": "LB" if is_linked else "BB",
                    "price": block["price"],
                    "volume": volume,
                    "node": block["bidding_zone"] or default_node,
                    "start_time": start_time,
                    "end_time": end_time,
                    "only_hours": None,
                    "min_acceptance_ratio": block["acc_ratio"],
                    "parent_bid_id": (
                        block["special_id"] if is_linked else None
                    ),
                    "exclusive_group_id": (
                        block["special_id"] if is_exclusive else None
                    ),
                }
            )

        return orderbook

    def _storage_to_orderbook(
        self,
        storage_orders: list[dict],
        default_node: str,
    ) -> Orderbook:
        """Convert wide storage rows to SO orders with timestamp mappings."""
        timestamps = [self._interval_start(interval) for interval in range(96)]
        start_time = timestamps[0]
        end_time = timestamps[-1] + MTU_DURATION
        orderbook: Orderbook = []

        for storage in storage_orders:
            node = storage["bidding_zone"] or default_node

            orderbook.append(
                {
                    "bid_id": storage["bid_id"],
                    "agent_addr": (f"agent_{node}", "test"),
                    "bid_type": "SO",
                    "node": node,
                    "price": 0.0,
                    "volume": {},
                    "start_time": start_time,
                    "end_time": end_time,
                    "only_hours": None,
                    "parent_bid_id": None,
                    "exclusive_group_id": None,
                    "starting_energy": storage["starting_energy"],
                    "efficiency": storage["efficiency"],
                    "cycling_cost": storage["cycling_cost"],
                    "end_of_day_value": storage["end_of_day_value"],
                    "max_energy": dict(zip(timestamps, storage["max_energy"])),
                    "min_energy": dict(zip(timestamps, storage["min_energy"])),
                    "max_input_power": dict(
                        zip(timestamps, storage["max_input_power"])
                    ),
                    "max_output_power": dict(
                        zip(timestamps, storage["max_output_power"])
                    ),
                }
            )

        return orderbook


    @staticmethod
    def _validate_unique_bid_ids(orderbook: Orderbook) -> None:
        """Reject duplicate IDs across simple, block, and storage input files."""
        bid_ids = [order["bid_id"] for order in orderbook]
        duplicates = sorted(
            bid_id
            for bid_id in set(bid_ids)
            if bid_ids.count(bid_id) > 1
        )

        if duplicates:
            raise ValueError(
                f"Duplicate bid_id values across input files: {duplicates}."
            )
        
    @staticmethod
    def _validate_linked_block_parents(
        orderbook: Orderbook,
    ) -> None:
        """Remove invalid parent references from linked blocks."""
        bid_ids = {
            order["bid_id"]
            for order in orderbook
        }

        for order in orderbook:
            parent_bid_id = order.get("parent_bid_id")

            if parent_bid_id is not None and parent_bid_id not in bid_ids:
                logger.warning(
                    "Parent bid '%s' of linked bid '%s' was not found. "
                    "Treating the linked block as a normal block.",
                    parent_bid_id,
                    order["bid_id"],
                )

                order["parent_bid_id"] = None
                order["bid_type"] = "BB"

    def _build_market_products(
        self,
    ) -> list[MarketProduct]:
        """
        Build exactly 96 fixed 15-minute market products for the day.

        Products are created even if no bid exists in a specific MTU.
        """
        if self.delivery_date is None:
            raise RuntimeError("delivery_date was not set.")

        return [
            (
                self.delivery_date + interval * MTU_DURATION,
                self.delivery_date + (interval + 1) * MTU_DURATION,
                None,
            )
            for interval in range(96)
        ]

    def _validate_network_market_product_snapshots(self) -> None:
        """
        Require a supplied PyPSA network to contain every daily 15-minute MTU.

        An hourly network is rejected because it does not contain the 15-minute
        timestamps required for all 96 delivery intervals.
        """
        if self.network is None:
            return

        required_snapshots = pd.Index(
            [product[0] for product in self.market_products]
        )
        network_snapshots = pd.Index(self.network.snapshots)

        if network_snapshots.has_duplicates:
            raise ValueError(
                "The supplied PyPSA network contains duplicate snapshots."
            )

        missing_snapshots = required_snapshots.difference(network_snapshots)

        if not missing_snapshots.empty:
            raise ValueError(
                "The supplied PyPSA network must contain every required "
                "15-minute delivery timestamp. Missing timestamps: "
                f"{list(missing_snapshots)}"
            )

    # ------------------------------------------------------------------
    # Market configuration and clearing
    # ------------------------------------------------------------------

    def _create_market_config(
        self,
    ) -> MarketConfig:
        """Create a market configuration for fixed 15-minute products."""
        if not self.market_products:
            raise ValueError(
                "No market products exist. Call load_orders() first."
            )

        first_start = self.market_products[0][0]
        last_end = self.market_products[-1][1]
        market_open = first_start - timedelta(hours=12)

        param_dict = {
            "solver_name": "gurobi",
            "log_flows": True,
            "pricing_mechanism": "pay_as_clear",
            "network": self.network,
            "flowbased_region": self.flowbased_region,
            "zones_identifier": self.zones_identifier,
            "ntc_connections": self.ntc_connections,
        }

        user_param_dict = self.market_config_kwargs.get("param_dict", {})
        param_dict.update(user_param_dict)

        config_dict = {
            "market_id": self.market_id,
            "opening_hours": rr.rrule(
                rr.DAILY,
                dtstart=market_open,
                until=last_end,
                cache=True,
            ),
            "opening_duration": timedelta(hours=1),
            "market_mechanism": self.clearing_type,
            "market_products": [
                MarketProduct(
                    duration=MTU_DURATION,
                    count=96,
                    first_delivery=first_start - market_open,
                )
            ],
            "maximum_bid_price": 4000.0,
            "minimum_bid_price": -500.0,
            "maximum_bid_volume": 1e6,
            "additional_fields": [
                "bid_type",
                "min_acceptance_ratio",
                "parent_bid_id",
                "exclusive_group_id",
                "node",
                "starting_energy",
                "efficiency",
                "cycling_cost",
                "end_of_day_value",
                "max_energy",
                "min_energy",
                "max_input_power",
                "max_output_power",
            ],
            "param_dict": param_dict,
        }

        config_dict.update(
            {
                key: value
                for key, value in self.market_config_kwargs.items()
                if key != "param_dict"
            }
        )

        return MarketConfig(**config_dict)

    def run_clearing(
        self,
    ) -> dict:
        """Run EUPHEMIA-like market clearing for the 96 daily MTUs."""
        if not self.orderbook:
            raise ValueError("No orders loaded. Call load_orders() first.")

        if self.flowbased_region and self.network is None:
            raise ValueError(
                "A PyPSA network is required when flowbased_region is set."
            )

        if self.clearing_type != "euphemia_like":
            raise ValueError(
                f"Unknown clearing type '{self.clearing_type}'."
            )

        self._validate_network_market_product_snapshots()
        market_config = self._create_market_config()

        self.clearing_algorithm = EuphemiaLikeClearingRole(
            market_config
        )

        # The clearing role may modify the supplied orderbook. Preserve the
        # original self.orderbook for inspection after the clearing.
        working_orderbook = copy.deepcopy(self.orderbook)

        # Mango-free validation setup.
        self.clearing_algorithm.open_auctions = set(
            self.market_products
        )

        # MarketRole.validate_orderbook requires every field listed in
        # MarketConfig.additional_fields to exist on every order, even when a field
        # is not applicable to that particular bid type.
        optional_fields = (
            "bid_type",
            "min_acceptance_ratio",
            "parent_bid_id",
            "exclusive_group_id",
            "node",
            "starting_energy",
            "efficiency",
            "cycling_cost",
            "end_of_day_value",
            "max_energy",
            "min_energy",
            "max_input_power",
            "max_output_power",
        )

        for order in working_orderbook:
            for field in optional_fields:
                order.setdefault(field, None)

            self.clearing_algorithm.validate_orderbook(
                [order],
                order["agent_addr"],
            )
        logger.info(
            "Running '%s' for market '%s' over %s 15-minute MTUs.",
            self.clearing_type,
            self.market_id,
            len(self.market_products),
        )

        accepted, rejected, meta, flows = self.clearing_algorithm.clear(
            working_orderbook,
            self.market_products,
        )

        self.results = {
            "accepted_orders": accepted,
            "rejected_orders": rejected,
            "market_meta": meta,
            "flows": flows or {},
            "line_results": getattr(
                self.clearing_algorithm,
                "line_results",
                [],
            ),
        }

        logger.info(
            "Clearing completed: %s accepted, %s rejected, %s storage orders.",
            len(accepted),
            len(rejected),
            sum(order["bid_type"] == "SO" for order in working_orderbook),
        )

        return self.results

    # ------------------------------------------------------------------
    # Result output
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(
        value,
    ):
        """Convert dictionary-valued order fields to CSV-safe strings."""
        if isinstance(value, dict):
            return json.dumps(
                {
                    str(key): item
                    for key, item in value.items()
                }
            )

        return value

    def _orders_to_dataframe(
        self,
        orders: Orderbook,
    ) -> pd.DataFrame:
        """Convert accepted/rejected orders, including storage fields, to CSV."""
        rows = []

        for order in orders:
            accepted_volume = order.get("accepted_volume", 0.0)

            rows.append(
                {
                    "bid_id": order.get("bid_id"),
                    "agent_id": order.get("agent_addr", ("", ""))[0],
                    "bid_type": order.get("bid_type"),
                    "price": order.get("price"),
                    "volume": self._serialize(order.get("volume")),
                    "node": order.get("node"),
                    "start_time": order.get("start_time"),
                    "end_time": order.get("end_time"),
                    "min_acceptance_ratio": order.get("min_acceptance_ratio"),
                    "parent_bid_id": order.get("parent_bid_id"),
                    "exclusive_group_id": order.get("exclusive_group_id"),
                    "starting_energy": order.get("starting_energy"),
                    "efficiency": order.get("efficiency"),
                    "cycling_cost": order.get("cycling_cost"),
                    "end_of_day_value": order.get("end_of_day_value"),
                    "accepted_volume": self._serialize(accepted_volume),
                    "accepted_net_volume": self._serialize(accepted_volume),
                    "accepted_price": self._serialize(
                        order.get("accepted_price", 0.0)
                    ),
                    "charge": self._serialize(order.get("charge")),
                    "discharge": self._serialize(order.get("discharge")),
                    "soc": self._serialize(order.get("soc")),
                    "storage_surplus_eur": self._serialize(
                        order.get("storage_surplus_eur")
                    ),
                    "total_storage_surplus_eur": order.get(
                        "total_storage_surplus_eur"
                    ),
                }
            )

        return pd.DataFrame(rows)

    def export_results(
        self,
        output_dir: str | Path,
        include_flows: bool = True,
    ) -> None:
        """Export accepted orders, rejected orders, prices, and flows."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._orders_to_dataframe(
            self.results["accepted_orders"]
        ).to_csv(
            output_dir / "accepted_orders.csv",
            index=False,
        )

        self._orders_to_dataframe(
            self.results["rejected_orders"]
        ).to_csv(
            output_dir / "rejected_orders.csv",
            index=False,
        )

        pd.DataFrame(
            self.results["market_meta"]
        ).to_csv(
            output_dir / "market_meta.csv",
            index=False,
        )

        if include_flows:
            line_results = self.results.get("line_results", [])

            if line_results:
                pd.DataFrame(line_results).to_csv(
                    output_dir / "flows.csv",
                    index=False,
                )

        logger.info("Results exported to %s.", output_dir)

    def print_summary(
        self,
    ) -> None:
        """Print a compact presentation-friendly result summary."""
        print("\n" + "=" * 78)
        print(f"CLEARING RESULTS: {self.market_id}")
        print("=" * 78)

        print("\nOrders:")
        print(f"  Submitted: {self.submitted_order_count}")
        print(f"  Accepted:  {len(self.results['accepted_orders'])}")
        print(f"  Rejected:  {len(self.results['rejected_orders'])}")

        submitted_storage = sum(
            order["bid_type"] == "SO"
            for order in self.orderbook
        )
        accepted_storage = sum(
            order["bid_type"] == "SO"
            for order in self.results["accepted_orders"]
        )

        print(f"  Storage submitted: {submitted_storage}")
        print(f"  Storage accepted:  {accepted_storage}")

        market_meta = self.results.get("market_meta", [])

        if market_meta:
            meta = pd.DataFrame(market_meta)
            first_timestamp = meta["product_start"].min()

            summary_columns = [
                "zone",
                "product_start",
                "price",
                "supply_volume",
                "demand_volume",
                "supply_volume_energy",
                "demand_volume_energy",
            ]

            print("\nMarket outcome in the first 15-minute interval:")
            print(
                meta.loc[
                    meta["product_start"] == first_timestamp,
                    summary_columns,
                ].to_string(index=False)
            )

        line_results = self.results.get("line_results", [])

        if line_results:
            flows = pd.DataFrame(line_results)
            first_timestamp = flows["time"].min()

            print(
                "\nCommercial flow variables in the first "
                "15-minute interval:"
            )
            print(
                flows.loc[
                    flows["time"] == first_timestamp
                ].to_string(index=False)
            )

        print("=" * 78 + "\n")