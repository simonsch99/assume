import logging
from datetime import datetime, timedelta

from assume.testing_tools.clearing_test_tool import ClearingTestTool

logging.basicConfig(level=logging.INFO)

tool = ClearingTestTool(
    market_id="test_2zone",
    product_duration=timedelta(hours=1),
    maximum_bid_price=3000.0,
    minimum_bid_price=-500.0,
    maximum_bid_volume=20000.0,
)

# 1) Netz zonal laden
tool.load_grid(
    buses_csv="assume/testing_tools/test_data/buses.csv",
    lines_csv="assume/testing_tools/test_data/lines.csv",
    zones_identifier="zone_id",
)

# 2) Gebote laden
tool.load_orders(
    delivery_date=datetime(2026, 8, 12, 0, 0),
    hourly_csv="assume/testing_tools/test_data/twozone_old_hourly.csv",
    block_csv=None,
)

# 3) Clearing
tool.run_clearing()
tool.print_summary()
tool.export_results("assume/testing_tools/results/2zone_old")