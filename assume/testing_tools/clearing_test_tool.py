# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dateutil import rrule as rr

from assume.common.market_objects import (
    MarketConfig,
    MarketProduct,
    Orderbook,
)
from assume.common.utils import create_incidence_matrix

# clearing algorithm under test
from assume.markets.clearing_algorithms.euphemia_like import ComplexClearingRole as EuphemiaLikeClearingRole

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
class HourlyOrderReader:
    """
    Reads hourly (simple) orders from an EPEX-like csv export.

    Expected row layout (comma separated, 8 columns):
        0: id / ignored
        1: ignored
        2: ignored
        3: hour (1..24)
        4: price
        5: quantity (always positive)
        6: side  -> 'Purchase' | 'Sell'
        7: bidding zone
    """

    def read(self, path: str | Path) -> list[dict]:
        orders: list[dict] = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader):
                if row[6] not in ("Purchase", "Sell"):
                    continue
                try:
                    orders.append(
                        {
                            "raw_id": row[0].strip() or f"h{line_no}",
                            "side": "BUY" if row[6] == "Purchase" else "SELL",
                            "price": float(row[4]),
                            "quantity": float(row[5]),
                            "hour": int(row[3]) - 1,  # 0-based
                            "bidding_zone": row[7].strip(),
                        }
                    )
                except ValueError as e:
                    logger.warning("Skipping hourly row %s: %s", line_no, e)
        logger.info("Read %s hourly orders from %s", len(orders), path)
        return orders


class BlockOrderReader:
    """
    Reads block orders from an EPEX-like csv export.

    Expected row layout (semicolon separated):
        0 : 'BB' marker
        1 : block id
        2 : block type  ('C01' regular, 'C02' linked, 'C04' exclusive)
        3 : special id  (parent id for C02, group id for C04)
        5 : price
        6..29 : hourly quantities (positive = sell, negative = purchase)
        30: minimum acceptance ratio
        32: bidding zone
    """

    def read(self, path: str | Path) -> list[dict]:
        blocks: list[dict] = []
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            for line_no, row in enumerate(reader):
                if not row or row[0].strip() != "BB":
                    continue
                try:
                    blocks.append(self._read_line(row))
                except (ValueError, IndexError) as e:
                    logger.warning("Skipping block row %s: %s", line_no, e)
        logger.info("Read %s block orders from %s", len(blocks), path)
        return blocks

    def _read_line(self, row: list[str]) -> dict:
        quantities = [float(row[6 + i]) for i in range(24)]
        return {
            "raw_id": row[1].strip(),
            "type": row[2].strip(),
            "special_id": row[3].strip() or None,
            "price": float(row[5]),
            "hours": -quantities, # EPEX SPOT data has demand as positive and supply as negative volumes, ASSUME works the other way around
            "acc_ratio": float(row[30]),
            "bidding_zone": row[32].strip() if len(row) > 32 else "node0",
        }


# --------------------------------------------------------------------------- #
# Test tool
# --------------------------------------------------------------------------- #
class ClearingTestTool:
    """
    Standalone tool to test market clearing algorithms of the ASSUME framework.

    It reads
    * hourly orders  (EPEX-like curve export, comma separated)
    * block orders   (EPEX-like block export, semicolon separated)
    * optional grid data (buses.csv / lines.csv)

    builds a single ASSUME Orderbook and runs the clearing algorithm.

    Example:
        >>> tool = ClearingTestTool(market_id="test_market")
        >>> tool.load_grid("grid/buses.csv", "grid/lines.csv", zones_identifier="zone_id")
        >>> tool.load_orders(
        ...     delivery_date=datetime(2026, 8, 7),
        ...     hourly_csv="orders/hourly.csv",
        ...     block_csv="orders/blocks.csv",
        ... )
        >>> results = tool.run_clearing()
        >>> tool.print_summary()
        >>> tool.export_results("results/")
    """

    def __init__(
        self,
        market_id: str = "test_market",
        clearing_type: Literal["euphemia_like"] = "euphemia_like",
        product_duration: timedelta = timedelta(hours=1),
        **market_config_kwargs,
    ):
        self.market_id = market_id
        self.clearing_type = clearing_type
        self.product_duration = product_duration
        self.market_config_kwargs = market_config_kwargs

        self.delivery_date: Optional[datetime] = None
        self.orderbook: Orderbook = []
        self.market_products: list[MarketProduct] = []
        self.grid_data: Optional[dict] = None
        self.clearing_algorithm = None
        self.results = {
            "accepted_orders": [],
            "rejected_orders": [],
            "market_meta": [],
            "flows": {},
        }

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def load_orders(
        self,
        delivery_date: datetime,
        hourly_csv: Optional[str] = None,
        block_csv: Optional[str] = None,
        block_acc_ratio_col: Optional[int] = None,
        default_node: str = "node0",
    ) -> Orderbook:
        """
        Reads hourly and/or block orders and builds one ASSUME orderbook.

        Args:
            delivery_date: date of delivery (hour 0 == delivery_date 00:00)
            hourly_csv: path to hourly order csv (comma separated)
            block_csv: path to block order csv (semicolon separated)
            block_acc_ratio_col: optional column index of the min acceptance ratio
            default_node: node used when the file has no bidding zone
        """
        if hourly_csv is None and block_csv is None:
            raise ValueError("At least one of hourly_csv / block_csv must be given.")

        self.delivery_date = delivery_date.replace(
            minute=0, second=0, microsecond=0
        )
        orderbook: Orderbook = []

        if hourly_csv:
            hourly = HourlyOrderReader().read(hourly_csv)
            orderbook += self._hourly_to_orderbook(hourly, default_node)

        if block_csv:
            blocks = BlockOrderReader().read(block_csv, block_acc_ratio_col)
            orderbook += self._blocks_to_orderbook(blocks, default_node)

        # sanity check: parent bids of linked bids must exist
        ids = {o["bid_id"] for o in orderbook}
        for order in orderbook:
            parent = order.get("parent_bid_id")
            if parent is not None and parent not in ids:
                logger.warning(
                    "Parent bid '%s' of linked bid '%s' not found -> treated as BB",
                    parent,
                    order["bid_id"],
                )
                order["parent_bid_id"] = None
                order["bid_type"] = "BB"

        self.orderbook = orderbook
        self.market_products = self._build_market_products()
        logger.info(
            "Built orderbook with %s orders (%s SB, %s BB/LB) and %s products",
            len(orderbook),
            sum(1 for o in orderbook if o["bid_type"] == "SB"),
            sum(1 for o in orderbook if o["bid_type"] in ("BB", "LB")),
            len(self.market_products),
        )
        return orderbook

    def _hour_start(self, hour: int) -> datetime:
        return self.delivery_date + hour * self.product_duration

    def _hourly_to_orderbook(self, hourly: list[dict], default_node: str) -> Orderbook:
        orderbook: Orderbook = []
        for i, o in enumerate(hourly):
            start = self._hour_start(o["hour"])
            # demand is negative in ASSUME, supply positive
            volume = -o["quantity"] if o["side"] == "BUY" else o["quantity"]
            orderbook.append(
                {
                    "bid_id": f"SB_{o['raw_id']}_{o['hour']}_{i}",
                    "agent_addr": (f"agent_{o['bidding_zone'] or default_node}", "test"),
                    "bid_type": "SB",
                    "price": o["price"],
                    "volume": volume,
                    "node": o["bidding_zone"] or default_node,
                    "start_time": start,
                    "end_time": start + self.product_duration,
                    "only_hours": None,
                    "min_acceptance_ratio": None,
                    "parent_bid_id": None,
                    "exclusive_group_id": None,
                }
            )
        return orderbook

    def _blocks_to_orderbook(self, blocks: list[dict], default_node: str) -> Orderbook:
        orderbook: Orderbook = []
        for b in blocks:
            volume = {
                self._hour_start(h): q
                for h, q in enumerate(b["hours"])
                if abs(q) > 0.0
            }
            if not volume:
                logger.warning("Block '%s' has no non-zero volume -> skipped", b["raw_id"])
                continue

            starts = sorted(volume.keys())
            is_linked = b["type"] == "C02" and b["special_id"] is not None
            is_exclusive = b["type"] == "C04" and b["special_id"] is not None

            orderbook.append(
                {
                    "bid_id": b["raw_id"],
                    "agent_addr": (f"agent_{b['bidding_zone'] or default_node}", "test"),
                    "bid_type": "LB" if is_linked else "BB",
                    "price": b["price"],
                    "volume": volume, # TODO check if negative is demand and vice versa, might have to inverse
                    "node": b["bidding_zone"] or default_node,
                    "start_time": starts[0],
                    "end_time": starts[-1] + self.product_duration,
                    "only_hours": None,
                    "min_acceptance_ratio": b["acc_ratio"],
                    "parent_bid_id": b["special_id"] if is_linked else None,
                    "exclusive_group_id": b["special_id"] if is_exclusive else None,
                }
            )
        return orderbook

    def _build_market_products(self) -> list[MarketProduct]:
        """One product per delivery hour covered by the orderbook."""
        starts = set()
        for order in self.orderbook:
            if order["bid_type"] == "SB":
                starts.add(order["start_time"])
            else:
                starts.update(order["volume"].keys())
        return [
            (start, start + self.product_duration, None) for start in sorted(starts)
        ]

    # ------------------------------------------------------------------ #
    # Grid
    # ------------------------------------------------------------------ #
    def load_grid(
        self,
        buses_csv: str,
        lines_csv: str,
        zones_identifier: Optional[str] = None,
    ) -> dict:
        buses_path, lines_path = Path(buses_csv), Path(lines_csv)
        if not buses_path.exists():
            raise FileNotFoundError(f"Buses file not found: {buses_path}")
        if not lines_path.exists():
            raise FileNotFoundError(f"Lines file not found: {lines_path}")

        buses = pd.read_csv(buses_path, index_col=0)
        lines = pd.read_csv(lines_path, index_col=0)
        return self.load_grid_from_dataframes(buses, lines, zones_identifier)

    def load_grid_from_dataframes(
        self,
        buses: pd.DataFrame,
        lines: pd.DataFrame,
        zones_identifier: Optional[str] = None,
    ) -> dict:
        if "s_nom" not in lines.columns:
            raise ValueError("lines data must contain a 's_nom' column")
        if "s_max_pu" not in lines.columns:
            lines["s_max_pu"] = 1.0

        incidence_matrix = create_incidence_matrix(
            lines, buses, zones_id=zones_identifier
        )
        self.grid_data = {
            "buses": buses,
            "lines": lines,
            "incidence_matrix": incidence_matrix,
            "zones_identifier": zones_identifier,
        }
        logger.info(
            "Loaded grid: %s buses, %s lines, incidence matrix %s (%s)",
            len(buses),
            len(lines),
            incidence_matrix.shape,
            "zonal" if zones_identifier else "nodal",
        )
        return self.grid_data

    # ------------------------------------------------------------------ #
    # Market config / clearing
    # ------------------------------------------------------------------ #
    def _create_market_config(self) -> MarketConfig:
        first_start = self.market_products[0][0]
        last_end = self.market_products[-1][1]
        market_open = first_start - timedelta(hours=12)

        param_dict = {
            "solver_name": "appsi_highs",
            "log_flows": True,
            "pricing_mechanism": "pay_as_clear",
        }
        if self.grid_data:
            param_dict["grid_data"] = {
                "buses": self.grid_data["buses"],
                "lines": self.grid_data["lines"],
            }
            if self.grid_data.get("zones_identifier"):
                param_dict["zones_identifier"] = self.grid_data["zones_identifier"]

        config_dict = {
            "market_id": self.market_id,
            "opening_hours": rr.rrule(
                rr.DAILY, dtstart=market_open, until=last_end, cache=True
            ),
            "opening_duration": timedelta(hours=1),
            "market_mechanism": self.clearing_type,
            "market_products": [
                MarketProduct(
                    duration=self.product_duration,
                    count=len(self.market_products),
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
        config_dict.update(self.market_config_kwargs)
        return MarketConfig(**config_dict)

    def run_clearing(self) -> dict:
        if not self.orderbook:
            raise ValueError("No orders loaded. Call load_orders() first.")

        market_config = self._create_market_config()

        if self.clearing_type == "euphemia_like":
            self.clearing_algorithm = EuphemiaLikeClearingRole(market_config)
        else:
            raise ValueError(f"Unknown clearing type: {self.clearing_type}")

        # mango-free validation: declare all products as open auctions
        self.clearing_algorithm.open_auctions = set(self.market_products)
        for order in self.orderbook:
            self.clearing_algorithm.validate_orderbook(
                [order], order["agent_addr"]
            )

        logger.info(
            "Running '%s' for market '%s' over %s products (%s - %s)",
            self.clearing_type,
            self.market_id,
            len(self.market_products),
            self.market_products[0][0],
            self.market_products[-1][1],
        )

        accepted, rejected, meta, flows = self.clearing_algorithm.clear(
            self.orderbook, self.market_products
        )

        self.results = {
            "accepted_orders": accepted,
            "rejected_orders": rejected,
            "market_meta": meta,
            "flows": flows or {},
        }
        logger.info(
            "Clearing done: %s accepted, %s rejected", len(accepted), len(rejected)
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def export_results(self, output_dir: str, include_flows: bool = True) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._orders_to_dataframe(self.results["accepted_orders"]).to_csv(
            out / "accepted_orders.csv", index=False
        )
        self._orders_to_dataframe(self.results["rejected_orders"]).to_csv(
            out / "rejected_orders.csv", index=False
        )
        pd.DataFrame(self.results["market_meta"]).to_csv(
            out / "market_meta.csv", index=False
        )

        if include_flows and self.results["flows"]:
            pd.DataFrame(
                [
                    {"time": k[0], "line": k[1], "flow": v}
                    for k, v in self.results["flows"].items()
                ]
            ).to_csv(out / "flows.csv", index=False)

        logger.info("Results exported to %s", out)

    @staticmethod
    def _serialize(value):
        if isinstance(value, dict):
            return json.dumps({str(k): v for k, v in value.items()})
        return value

    def _orders_to_dataframe(self, orders: Orderbook) -> pd.DataFrame:
        data = []
        for order in orders:
            data.append(
                {
                    "bid_id": order["bid_id"],
                    "agent_id": order["agent_addr"][0],
                    "bid_type": order["bid_type"],
                    "price": order["price"],
                    "volume": self._serialize(order["volume"]),
                    "node": order.get("node"),
                    "start_time": order["start_time"],
                    "end_time": order["end_time"],
                    "min_acceptance_ratio": order.get("min_acceptance_ratio"),
                    "parent_bid_id": order.get("parent_bid_id"),
                    "exclusive_group_id": order.get("exclusive_group_id"),
                    "accepted_volume": self._serialize(order.get("accepted_volume", 0)),
                    "accepted_price": self._serialize(order.get("accepted_price", 0)),
                }
            )
        return pd.DataFrame(data)

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print(f"CLEARING RESULTS FOR MARKET: {self.market_id}")
        print("=" * 70)
        print("\nOrderbook:")
        print(f"  Total orders : {len(self.orderbook)}")
        print(f"  Accepted     : {len(self.results['accepted_orders'])}")
        print(f"  Rejected     : {len(self.results['rejected_orders'])}")

        meta = self.results["market_meta"]
        if meta:
            df = pd.DataFrame(meta)[
                ["node", "product_start", "price", "supply_volume", "demand_volume"]
            ]
            print("\nMarket clearing prices:")
            print(df.to_string(index=False))

        if self.results["flows"]:
            print("\nFlows (first 5):")
            for (t, line), flow in list(self.results["flows"].items())[:5]:
                print(f"  {line} at {t}: {flow:.2f} MW")
        print("=" * 70 + "\n")


# # --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     tool = ClearingTestTool(market_id="test_market")
#     # tool.load_grid("grid/buses.csv", "grid/lines.csv", zones_identifier="zone_id")
#     tool.load_orders(
#         delivery_date=datetime(2026, 8, 7),
#         hourly_csv="orders/hourly_orders.csv",
#         block_csv="orders/block_orders.csv",
#     )
#     tool.run_clearing()
#     tool.print_summary()
#     tool.export_results("results/")