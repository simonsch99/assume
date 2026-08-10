# SPDX-FileCopyrightText: ASSUME Developers

# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal
import json

import pandas as pd
import numpy as np

from assume.common.market_objects import (
    MarketConfig,
    MarketProduct,
    Orderbook,
)
from assume.common.utils import create_incidence_matrix
from assume.markets.clearing_algorithms.complex_clearing import ComplexClearingRole

logger = logging.getLogger(__name__)

class ClearingTestTool:
    """
    A standalone tool for testing market clearing algorithms without running a full ASSUME simulation.
    
    This tool loads orderbooks from CSV files and grid data (if available) and executes clearing algorithms
    to validate and analyze their behavior.
    
    Attributes:
        clearing_algorithm: The clearing algorithm instance (e.g., ComplexClearingRole)
        orderbook: The loaded orderbook from CSV
        grid_data: Optional grid data (zones, lines, buses)
        results: Results from the last clearing execution
    
    Example:
        >>> tool = ClearingTestTool(
        ...     market_id="test_market",
        ...     clearing_type="complex_clearing"
        ... )
        >>> tool.load_orderbook("orderbook.csv")
        >>> tool.load_grid("grid_buses.csv", "grid_lines.csv", zones_identifier="zone_id")
        >>> results = tool.run_clearing(
        ...     market_open=datetime(2026, 8, 7, 12, 0),
        ...     market_close=datetime(2026, 8, 7, 13, 0)
        ... )
        >>> tool.export_results("results/")
    """
    
    def __init__(
        self,
        market_id: str = "test_market",
        clearing_type: Literal["complex_clearing", "custom"] = "complex_clearing",
        **market_config_kwargs
    ):
        """
        Initialize the clearing test tool.
        
        Args:
            market_id (str): Identifier for the market
            clearing_type (str): Type of clearing algorithm to use
            **market_config_kwargs: Additional arguments for MarketConfig
        """
        self.market_id = market_id
        self.clearing_type = clearing_type
        self.market_config_kwargs = market_config_kwargs
        
        self.orderbook: Optional[Orderbook] = None
        self.grid_data: Optional[dict] = None
        self.clearing_algorithm = None
        self.results = {
            "accepted_orders": [],
            "rejected_orders": [],
            "market_meta": [],
            "flows": {},
        }
        
    def load_orderbook(self, csv_path: str) -> Orderbook: #TODO rework so it works with EPEX SPOT data format
        """
        Load orderbook from CSV file.
        
        Expected CSV columns:
        - bid_id (str): Unique bid identifier
        - agent_id (str): Agent identifier
        - bid_type (str): 'SB' (Simple Bid), 'BB' (Block Bid), 'LB' (Linked Bid)
        - price (float): Price in €/MWh
        - volume (float or JSON): Volume in MW
        - node (str): Node/zone identifier
        - start_time (str): Start time (ISO format)
        - end_time (str): End time (ISO format)
        - min_acceptance_ratio (float, optional): Minimum acceptance ratio
        - parent_bid_id (str, optional): Parent bid ID for linked bids
        - only_hours (str, optional): JSON list of specific hours
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            Orderbook: Loaded and validated orderbook
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Orderbook file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded orderbook from {csv_path} with {len(df)} orders")
        
        orderbook: Orderbook = []
        
        for idx, row in df.iterrows():
            try:
                order = self._parse_order_row(row, idx)
                orderbook.append(order)
            except Exception as e:
                logger.warning(f"Error parsing row {idx}: {e}. Skipping.")
                continue
        
        self.orderbook = orderbook
        logger.info(f"Successfully parsed {len(orderbook)} orders")
        return orderbook
    
    def _parse_order_row(self, row: pd.Series, idx: int) -> dict:
        """Parse a single CSV row into an order dictionary."""
        order = {
            "bid_id": str(row['bid_id']),
            "agent_addr": (str(row.get('agent_id', f'agent_{idx}')), "agent"),
            "bid_type": row.get('bid_type', 'SB').upper(),
            "price": float(row['price']),
            "node": row.get('node', 'node0'),
            "start_time": pd.to_datetime(row['start_time']),
            "end_time": pd.to_datetime(row['end_time']),
            "only_hours": None,
        }
        
        # Parse volume (can be JSON for block bids)
        volume_raw = row['volume']
        try:
            order["volume"] = json.loads(volume_raw)
        except (json.JSONDecodeError, TypeError):
            order["volume"] = float(volume_raw)
        
        # Optional fields
        if "min_acceptance_ratio" in row and pd.notna(row["min_acceptance_ratio"]):
            order["min_acceptance_ratio"] = float(row["min_acceptance_ratio"])
        
        if "parent_bid_id" in row and pd.notna(row["parent_bid_id"]):
            order["parent_bid_id"] = str(row["parent_bid_id"])
        
        if "only_hours" in row and pd.notna(row["only_hours"]):
            try:
                order["only_hours"] = json.loads(row["only_hours"])
            except json.JSONDecodeError:
                order["only_hours"] = None
        
        return order
    
    def load_grid(
        self,
        buses_csv: str,
        lines_csv: str,
        zones_identifier: Optional[str] = None,
    ) -> dict:
        """
        Load grid data from CSV files.
        
        Args:
            buses_csv (str): Path to buses CSV (index: bus_id, columns: x, y, zone_id, etc.)
            lines_csv (str): Path to lines CSV (index: line_id, columns: bus0, bus1, s_nom, s_max_pu, etc.)
            zones_identifier (str, optional): Column name in buses CSV that identifies zones
            
        Returns:
            dict: Grid data with buses, lines, and incidence matrix
        """
        buses_path = Path(buses_csv)
        lines_path = Path(lines_csv)
        
        if not buses_path.exists():
            raise FileNotFoundError(f"Buses file not found: {buses_path}")
        if not lines_path.exists():
            raise FileNotFoundError(f"Lines file not found: {lines_path}")
        
        buses = pd.read_csv(buses_path, index_col=0)
        lines = pd.read_csv(lines_path, index_col=0)
        
        logger.info(f"Loaded grid with {len(buses)} buses and {len(lines)} lines")
        
        # Create incidence matrix
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
            f"Created incidence matrix with shape {incidence_matrix.shape} "
            f"({'zonal' if zones_identifier else 'nodal'} representation)"
        )
        
        return self.grid_data
    
    def load_grid_from_pypsa(
        self,
        pypsa_network,
        zones_identifier: Optional[str] = None,
    ) -> dict:
        """
        Load grid data from a PyPSA network.
        
        Args:
            pypsa_network: PyPSA Network object
            zones_identifier (str, optional): Column name for zone identification
            
        Returns:
            dict: Grid data compatible with ASSUME
        """
        try:
            import pypsa
        except ImportError:
            raise ImportError("PyPSA is required to use this method. Install with: pip install pypsa")
        
        # Extract buses
        buses = pypsa_network.buses.copy()
        if zones_identifier and zones_identifier not in buses.columns:
            logger.warning(
                f"Zone identifier '{zones_identifier}' not found in PyPSA buses. "
                f"Available columns: {list(buses.columns)}"
            )
            zones_identifier = None
        
        # Extract lines
        lines = pypsa_network.lines.copy()
        if "s_nom" not in lines.columns:
            lines["s_nom"] = 1000.0  # Default capacity
        if "s_max_pu" not in lines.columns:
            lines["s_max_pu"] = 1.0
        
        logger.info(
            f"Loaded PyPSA network with {len(buses)} buses and {len(lines)} lines"
        )
        
        return self.load_grid_from_dataframes(
            buses, lines, zones_identifier
        )
    
    def load_grid_from_dataframes(
        self,
        buses: pd.DataFrame,
        lines: pd.DataFrame,
        zones_identifier: Optional[str] = None,
    ) -> dict:
        """
        Load grid data from pandas DataFrames.
        
        Args:
            buses (pd.DataFrame): Buses data
            lines (pd.DataFrame): Lines data
            zones_identifier (str, optional): Column name for zone identification
            
        Returns:
            dict: Grid data
        """
        incidence_matrix = create_incidence_matrix(
            lines, buses, zones_id=zones_identifier
        )
        
        self.grid_data = {
            "buses": buses,
            "lines": lines,
            "incidence_matrix": incidence_matrix,
            "zones_identifier": zones_identifier,
        }
        
        return self.grid_data
    
    def _create_market_config(
        self,
        market_open: datetime,
        market_close: datetime,
    ) -> MarketConfig:
        """Create a MarketConfig for the clearing algorithm."""
        from assume.common.market_objects import MarketTimeFrame
        
        # Default configuration
        config_dict = {
            "market_id": self.market_id,
            "market_products": [MarketProduct(
                start=market_open,
                end=market_close,
                only_hours=None,
            )],
            "opening_hours": MarketTimeFrame(
                opening=market_open,
                closing=market_close,
                duration=market_close - market_open,
            ),
            "maximum_bid_price": 3000.0,
            "minimum_bid_price": -500.0,
            "maximum_bid_volume": 10000.0,
            "additional_fields": ["bid_type"],
        }
        
        # Merge with provided kwargs
        config_dict.update(self.market_config_kwargs)
        
        # Add grid data if available
        if self.grid_data:
            config_dict["param_dict"] = {
                "grid_data": {
                    "buses": self.grid_data["buses"],
                    "lines": self.grid_data["lines"],
                },
                "zones_identifier": self.grid_data.get("zones_identifier"),
            }
        
        return MarketConfig(**config_dict)
    
    def run_clearing(
        self,
        market_open: datetime,
        market_close: datetime,
        clearing_type: Optional[str] = None,
    ) -> dict:
        """
        Run the market clearing algorithm.
        
        Args:
            market_open (datetime): Market opening time
            market_close (datetime): Market closing time
            clearing_type (str, optional): Override the clearing type
            
        Returns:
            dict: Clearing results
            
        Raises:
            ValueError: If no orderbook is loaded
        """
        if self.orderbook is None:
            raise ValueError("No orderbook loaded. Call load_orderbook() first.")
        
        clearing_type = clearing_type or self.clearing_type
        
        # Create market config
        market_config = self._create_market_config(market_open, market_close)
        
        # Initialize clearing algorithm
        if clearing_type == "complex_clearing":
            self.clearing_algorithm = ComplexClearingRole(market_config)
        else:
            raise ValueError(f"Unknown clearing type: {clearing_type}")
        
        # Define market products
        market_products = [MarketProduct(
            start=market_open,
            end=market_close,
            only_hours=None,
        )]
        
        logger.info(
            f"Running {clearing_type} for market {self.market_id} "
            f"from {market_open} to {market_close}"
        )
        
        try:
            # Run clearing
            accepted, rejected, meta, flows = self.clearing_algorithm.clear(
                self.orderbook, market_products
            )
            
            self.results = {
                "accepted_orders": accepted,
                "rejected_orders": rejected,
                "market_meta": meta,
                "flows": flows,
            }
            
            logger.info(
                f"Clearing completed: {len(accepted)} accepted, "
                f"{len(rejected)} rejected orders"
            )
            
            return self.results
            
        except Exception as e:
            logger.error(f"Clearing failed: {e}")
            raise
    
    def export_results(
        self,
        output_dir: str,
        include_flows: bool = True,
    ) -> None:
        """
        Export clearing results to CSV files.
        
        Args:
            output_dir (str): Output directory
            include_flows (bool): Whether to export flow data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export accepted orders
        accepted_df = self._orders_to_dataframe(self.results["accepted_orders"])
        accepted_df.to_csv(output_path / "accepted_orders.csv", index=False)
        logger.info(f"Exported accepted orders to {output_path / 'accepted_orders.csv'}")
        
        # Export rejected orders
        rejected_df = self._orders_to_dataframe(self.results["rejected_orders"])
        rejected_df.to_csv(output_path / "rejected_orders.csv", index=False)
        logger.info(f"Exported rejected orders to {output_path / 'rejected_orders.csv'}")
        
        # Export market metadata
        meta_df = pd.DataFrame(self.results["market_meta"])
        meta_df.to_csv(output_path / "market_meta.csv", index=False)
        logger.info(f"Exported market metadata to {output_path / 'market_meta.csv'}")
        
        # Export flows if available
        if include_flows and self.results["flows"]:
            flows_df = pd.DataFrame([
                {
                    "time": k[0],
                    "line": k[1],
                    "flow": v,
                }
                for k, v in self.results["flows"].items()
            ])
            flows_df.to_csv(output_path / "flows.csv", index=False)
            logger.info(f"Exported flows to {output_path / 'flows.csv'}")
    
    def _orders_to_dataframe(self, orders: Orderbook) -> pd.DataFrame:
        """Convert orders to DataFrame for export."""
        data = []
        for order in orders:
            data.append({
                "bid_id": order["bid_id"],
                "agent_id": order["agent_addr"][0],
                "bid_type": order["bid_type"],
                "price": order["price"],
                "volume": json.dumps(order["volume"]) if isinstance(order["volume"], dict) else order["volume"],
                "node": order["node"],
                "start_time": order["start_time"],
                "end_time": order["end_time"],
                "accepted_volume": json.dumps(order.get("accepted_volume", {})) if isinstance(order.get("accepted_volume", {}), dict) else order.get("accepted_volume", 0),
                "accepted_price": json.dumps(order.get("accepted_price", {})) if isinstance(order.get("accepted_price", {}), dict) else order.get("accepted_price", 0),
            })
        return pd.DataFrame(data)
    
    def print_summary(self) -> None:
        """Print a summary of the clearing results."""
        print("\n" + "="*60)
        print(f"CLEARING RESULTS FOR MARKET: {self.market_id}")
        print("="*60)
        
        print(f"\nOrderbook Statistics:")
        print(f"  Total Orders: {len(self.orderbook)}")
        print(f"  Accepted: {len(self.results['accepted_orders'])}")
        print(f"  Rejected: {len(self.results['rejected_orders'])}")
        
        print(f"\nMarket Clearing Prices:")
        for meta in self.results["market_meta"]:
            print(f"  Node {meta['node']}, Time {meta['product_start']}: "
                  f"{meta['price']:.2f} €/MWh")
            print(f"    Supply: {meta['supply_volume']:.2f} MW, "
                  f"Demand: {meta['demand_volume']:.2f} MW")
        
        if self.results["flows"]:
            print(f"\nNetwork Flows:")
            for (t, line), flow in list(self.results["flows"].items())[:5]:
                print(f"  {line} at {t}: {flow:.2f} MW")
            if len(self.results["flows"]) > 5:
                print(f"  ... and {len(self.results['flows']) - 5} more flows")
        
        print("\n" + "="*60 + "\n")