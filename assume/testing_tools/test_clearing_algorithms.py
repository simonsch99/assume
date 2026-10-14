from datetime import datetime

from create_hybrid_input import create_hybrid_input
from clearing_test_tool import ClearingTestTool

input_data = create_hybrid_input(
    output_dir="assume/testing_tools/test_data/hybrid_example",
    delivery_date=datetime(2026, 1, 1),
)

tool = ClearingTestTool(
    market_id="hybrid_fbmc_ntc",
    network=input_data["network_path"],
    flowbased_region=input_data["flowbased_region"],
    ntc_connections=input_data["ntc_path"],
    zones_identifier="zone_id",
)

tool.load_orders(
    delivery_date=input_data["delivery_date"],
    hourly_csv=input_data["hourly_orders_path"],
    block_csv=input_data["block_orders_path"],
)

tool.run_clearing()
tool.print_summary()
tool.export_results("assume/testing_tools/test_data/hybrid_example/results")