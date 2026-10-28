from pathlib import Path
import csv
import logging

import numpy as np
import pandas as pd

from clearing_test_tool import ClearingTestTool

ZERO_TOLERANCE = 1e-9

# Configuration
DATA_DIR = Path("assume/testing_tools/test_data")
BINNED_ORDERS = DATA_DIR / "epex_binned_orders_20251001.csv"
BLOCK_ORDERS = DATA_DIR / "all_exchanges_bbof_15_germany_luxembourg_20251001.csv"

PREPARED_ORDERS = DATA_DIR / "epex_binned_orders_one_zone.csv"
PREPARED_BLOCK_ORDERS = DATA_DIR / "epex_block_orders_one_zone.csv"
RESULTS_DIR = Path("assume/testing_tools/epex_clearing_results")

DELIVERY_DATE = "2025-10-01"
ZONE = "DE_LU"
SOLVER = "gurobi"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def prepare_block_orders(input_path: Path, output_path: Path, zone: str) -> None:
    """
    Convert EPEX DST block orders to the 96-interval format expected by
    BlockOrderReader:

    - remove the four H03B quarter-hour values;
    - retain H03A as the normal Hour 3 values;
    - append the required trailing bidding_zone field.
    """
    block_count = 0

    with input_path.open(newline="", encoding="utf-8-sig") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.reader(source, delimiter=";")
        writer = csv.writer(target, delimiter=";")

        writer.writerow(
            [
                "# Preprocessed block orders: H03B removed; "
                "H03A mapped to H03; trailing bidding_zone appended."
            ]
        )

        for row in reader:
            if not row or row[0].strip() != "BB":
                continue

            # Original DST layout:
            # 6 fixed fields + 100 quarter-hour volumes + MAR + AAR = 108 fields.
            if len(row) != 108:
                raise ValueError(
                    f"Unexpected block-order row length ({len(row)}) for "
                    f"block '{row[1] if len(row) > 1 else '<unknown>'}'. "
                    "Expected 108 fields in the DST source file."
                )

            # Fields 6..9: H01; 10..13: H02; 14..17: H03A;
            # 18..21: H03B; 22..101: H04..H24.
            del row[18:22]

            # Result: 6 fixed + 96 volumes + MAR + AAR = 104 fields.
            # BlockOrderReader requires the bidding zone at index 104.
            row.append(zone)

            if len(row) != 105:
                raise RuntimeError(
                    f"Prepared block '{row[1]}' has {len(row)} fields; expected 105."
                )

            writer.writerow(row)
            block_count += 1

    logging.info(
        "Prepared %d block orders: removed H03B and mapped H03A to H03.",
        block_count,
    )


columns = [
    "Date",
    "Week",
    "Week Day",
    "Hour",
    "Quarter hour",
    "Price",
    "Volume",
    "Sale/Purchase",
]

if not BINNED_ORDERS.exists():
    raise FileNotFoundError(f"Simple-bid file not found: {BINNED_ORDERS}")

if not BLOCK_ORDERS.exists():
    raise FileNotFoundError(f"Block-order file not found: {BLOCK_ORDERS}")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Prepare simple orders.
orders = pd.read_csv(BINNED_ORDERS, comment="#")
missing = set(columns).difference(orders.columns)

if missing:
    raise ValueError(f"Missing binned-order columns: {sorted(missing)}")

delivery_day = pd.Timestamp(DELIVERY_DATE).normalize()
dates = pd.to_datetime(orders["Date"], dayfirst=True, errors="raise").dt.normalize()
orders = orders.loc[dates.eq(delivery_day), columns].copy()

if orders.empty:
    raise ValueError(f"No binned orders found for {delivery_day.date()}.")

hour = orders["Hour"].astype(str).str.strip().str.upper()
orders = orders.loc[hour.ne("3B")].copy()
orders["Hour"] = pd.to_numeric(
    hour.loc[orders.index].replace("3A", "3"),
    errors="raise",
).astype(int)
orders["Quarter hour"] = pd.to_numeric(
    orders["Quarter hour"],
    errors="raise",
).astype(int)

orders["Date"] = delivery_day.strftime("%d/%m/%Y")
orders["bidding_zone"] = ZONE
orders["Price"] = pd.to_numeric(orders["Price"], errors="raise")
orders["Volume"] = pd.to_numeric(orders["Volume"], errors="raise")

finite = np.isfinite(orders["Price"]) & np.isfinite(orders["Volume"])
if not finite.all():
    raise ValueError(f"Found {(~finite).sum()} rows with non-finite price or volume.")

negative = orders["Volume"] < -ZERO_TOLERANCE
if negative.any():
    raise ValueError(
        "Found materially negative individual order volumes:\n"
        + orders.loc[
            negative,
            ["Date", "Hour", "Quarter hour", "Sale/Purchase", "Price", "Volume"],
        ]
        .head()
        .to_string(index=False)
    )

zero = orders["Volume"].abs() <= ZERO_TOLERANCE
logging.info("Dropping %d zero-volume simple orders.", int(zero.sum()))
orders = orders.loc[~zero].copy()
orders.to_csv(PREPARED_ORDERS, index=False)

# Prepare DST block orders before loading them into ClearingTestTool.
prepare_block_orders(BLOCK_ORDERS, PREPARED_BLOCK_ORDERS, ZONE)

tool = ClearingTestTool(
    market_id=f"clearing_{ZONE}_{delivery_day:%Y%m%d}",
    param_dict={"solver_name": SOLVER},
)

tool.load_orders(
    delivery_date=delivery_day.to_pydatetime(),
    quarter_hourly_csv=PREPARED_ORDERS,
    block_csv=PREPARED_BLOCK_ORDERS,
    default_node=ZONE,
)

tool.run_clearing()
tool.print_summary()
tool.export_results(RESULTS_DIR)

logging.info("Prepared simple orders: %s", PREPARED_ORDERS)
logging.info("Prepared block orders: %s", PREPARED_BLOCK_ORDERS)
logging.info("Clearing results: %s", RESULTS_DIR)