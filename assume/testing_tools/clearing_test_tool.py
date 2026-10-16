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

import pandas as pd
import pypsa
from dateutil import rrule as rr

from assume.common.market_objects import (
    MarketConfig,
    MarketProduct,
    Orderbook,
)
from assume.markets.clearing_algorithms.euphemia_like import (
    ComplexClearingRole as EuphemiaLikeClearingRole,
)

logger = logging.getLogger(__name__)

MTU_DURATION = timedelta(minutes=15)
MTU_DURATION_HOURS = 0.25


# ---------------------------------------------------------------------------
# EUPHEMIA-like CSV readers
# ---------------------------------------------------------------------------


class QuarterHourlyOrderReader:
    """
    Read simple 15-minute orders from a comma-separated EUPHEMIA-like CSV.

    Expected row layout, without header:

        0: bid ID
        1: order type
        2: optional portfolio ID
        3: delivery interval, 1..96
        4: price
        5: quantity, always positive
        6: Purchase or Sell
        7: bidding zone

    Delivery intervals are mapped as follows:

        interval 1  -> 00:00
        interval 2  -> 00:15
        ...
        interval 96 -> 23:45

    CSV convention:
        Purchase -> positive quantity in CSV -> negative ASSUME volume
        Sell     -> positive quantity in CSV -> positive ASSUME volume
    """

    def read(
        self,
        path: str | Path,
    ) -> list[dict]:
        """Read simple 15-minute orders from a CSV file."""
        orders = []

        with Path(path).open(newline="") as file:
            reader = csv.reader(file)

            for line_number, row in enumerate(reader, start=1):
                if len(row) < 8:
                    logger.warning(
                        "Skipping 15-minute order row %s: expected at "
                        "least 8 columns.",
                        line_number,
                    )
                    continue

                side = row[6].strip()

                if side not in {"Purchase", "Sell"}:
                    continue

                try:
                    interval = int(row[3]) - 1

                    if not 0 <= interval < 96:
                        raise ValueError(
                            "delivery interval must be in 1..96, "
                            f"got {interval + 1}"
                        )

                    orders.append(
                        {
                            "raw_id": (
                                row[0].strip()
                                or f"quarter_hourly_{line_number}"
                            ),
                            "side": (
                                "BUY"
                                if side == "Purchase"
                                else "SELL"
                            ),
                            "price": float(row[4]),
                            "quantity": float(row[5]),
                            "interval": interval,
                            "bidding_zone": row[7].strip() or "node0",
                        }
                    )

                except ValueError as error:
                    logger.warning(
                        "Skipping 15-minute order row %s: %s",
                        line_number,
                        error,
                    )

        logger.info(
            "Read %s 15-minute simple orders from %s.",
            len(orders),
            path,
        )
        return orders


class BlockOrderReader:
    """
    Read block orders from a semicolon-separated EUPHEMIA-like CSV.

    Expected row layout, without header:

        0: BB marker
        1: block ID
        2: block type:
             C01 = ordinary block
             C02 = linked block
             C04 = exclusive block
        3: parent bid ID for C02 or exclusive-group ID for C04
        5: price
        6..101: quantities for delivery intervals 1..96
        102: minimum acceptance ratio
        104: bidding zone

    CSV convention:
        The input uses the inverse ASSUME volume sign convention.

        negative CSV quantity -> positive ASSUME supply volume
        positive CSV quantity -> negative ASSUME demand volume
    """

    def read(
        self,
        path: str | Path,
    ) -> list[dict]:
        """Read 15-minute block orders from a CSV file."""
        blocks = []

        with Path(path).open(newline="") as file:
            reader = csv.reader(file, delimiter=";")

            for line_number, row in enumerate(reader, start=1):
                if not row or row[0].strip() != "BB":
                    continue

                if len(row) < 105:
                    logger.warning(
                        "Skipping block row %s: expected at least "
                        "105 columns, got %s.",
                        line_number,
                        len(row),
                    )
                    continue

                try:
                    blocks.append(self._read_line(row))

                except (ValueError, IndexError) as error:
                    logger.warning(
                        "Skipping block row %s: %s",
                        line_number,
                        error,
                    )

        logger.info(
            "Read %s 15-minute block orders from %s.",
            len(blocks),
            path,
        )
        return blocks

    @staticmethod
    def _read_line(
        row: list[str],
    ) -> dict:
        """Parse one 96-interval block-order CSV row."""
        if len(row) < 105:
            raise ValueError(
                "Block rows must contain at least 105 columns."
            )

        # Convert EUPHEMIA-style sign convention to ASSUME convention.
        quantities = [
            -float(row[6 + interval])
            for interval in range(96)
        ]

        minimum_acceptance_ratio = (
            float(row[102])
            if row[102].strip()
            else None
        )

        return {
            "raw_id": row[1].strip(),
            "type": row[2].strip(),
            "special_id": row[3].strip() or None,
            "price": float(row[5]),
            "intervals": quantities,
            "acc_ratio": minimum_acceptance_ratio,
            "bidding_zone": row[104].strip() or "node0",
        }


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
        default_node: str = "node0",
    ) -> Orderbook:
        """
        Read 15-minute simple and/or block-order files.

        The generated delivery day always contains exactly 96 fixed
        15-minute market products, independently of whether bids exist
        in every interval.
        """
        if quarter_hourly_csv is None and block_csv is None:
            raise ValueError(
                "At least one of quarter_hourly_csv or block_csv must "
                "be supplied."
            )

        self.delivery_date = (
            pd.Timestamp(delivery_date)
            .normalize()
            .to_pydatetime()
        )

        orderbook: Orderbook = []

        if quarter_hourly_csv is not None:
            quarter_hourly_orders = QuarterHourlyOrderReader().read(
                quarter_hourly_csv
            )

            orderbook.extend(
                self._quarter_hourly_to_orderbook(
                    quarter_hourly_orders,
                    default_node,
                )
            )

        if block_csv is not None:
            block_orders = BlockOrderReader().read(block_csv)

            orderbook.extend(
                self._blocks_to_orderbook(
                    block_orders,
                    default_node,
                )
            )

        if not orderbook:
            raise ValueError(
                "No valid orders could be read from the input files."
            )

        self._validate_linked_block_parents(orderbook)

        self.orderbook = orderbook
        self.submitted_order_count = len(orderbook)
        self.market_products = self._build_market_products()

        self._validate_network_market_product_snapshots()

        logger.info(
            "Built orderbook with %s orders: %s simple and %s block orders.",
            len(orderbook),
            sum(
                order["bid_type"] == "SB"
                for order in orderbook
            ),
            sum(
                order["bid_type"] in {"BB", "LB"}
                for order in orderbook
            ),
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
        """Convert parsed 96-interval block rows to ASSUME block orders."""
        orderbook: Orderbook = []

        for block in blocks:
            volume = {
                self._interval_start(interval): quantity
                for interval, quantity in enumerate(block["intervals"])
                if abs(quantity) > 0.0
            }

            if not volume:
                logger.warning(
                    "Skipping block '%s' because it has no non-zero volume.",
                    block["raw_id"],
                )
                continue

            timestamps = sorted(volume)
            is_linked = (
                block["type"] == "C02"
                and block["special_id"] is not None
            )
            is_exclusive = (
                block["type"] == "C04"
                and block["special_id"] is not None
            )

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
                    "start_time": timestamps[0],
                    "end_time": timestamps[-1] + MTU_DURATION,
                    "only_hours": None,
                    "min_acceptance_ratio": block["acc_ratio"],
                    "parent_bid_id": (
                        block["special_id"]
                        if is_linked
                        else None
                    ),
                    "exclusive_group_id": (
                        block["special_id"]
                        if is_exclusive
                        else None
                    ),
                }
            )

        return orderbook

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

        for order in working_orderbook:
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
            "Clearing completed: %s accepted and %s rejected orders.",
            len(accepted),
            len(rejected),
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
        """Convert accepted/rejected orders to a DataFrame."""
        rows = []

        for order in orders:
            rows.append(
                {
                    "bid_id": order["bid_id"],
                    "agent_id": order["agent_addr"][0],
                    "bid_type": order["bid_type"],
                    "price": order["price"],
                    "volume": self._serialize(order["volume"]),
                    "node": order.get("node"),
                    "start_time": order["start_time"],
                    "end_time": order["end_time"],
                    "min_acceptance_ratio": order.get(
                        "min_acceptance_ratio"
                    ),
                    "parent_bid_id": order.get("parent_bid_id"),
                    "exclusive_group_id": order.get(
                        "exclusive_group_id"
                    ),
                    "accepted_volume": self._serialize(
                        order.get("accepted_volume", 0.0)
                    ),
                    "accepted_price": self._serialize(
                        order.get("accepted_price", 0.0)
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