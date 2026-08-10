### This test calls the clearing_test_tool and a specified clearing algorithm and uses user-specified input data to test the market clearing.

from assume.common.clearing_test_tool import ClearingTestTool
from datetime import datetime

# Beispiel 3: Mit CSV-Netzwerkdaten
tool = ClearingTestTool(
    market_id="test_market_csv",
    clearing_type="complex_clearing",
)

tool.load_orderbook("orderbook.csv")
tool.load_grid(
    "grid_buses.csv",
    "grid_lines.csv",
    zones_identifier="zone_id"
)

results = tool.run_clearing(
    market_open=datetime(2026, 8, 7, 12, 0),
    market_close=datetime(2026, 8, 7, 13, 0),
)

tool.export_results("results/")