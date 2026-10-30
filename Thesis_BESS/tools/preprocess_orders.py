"""Preprocess published EPEX-like simple and block orders.

Published source files do not contain ``bidding_zone``. This module creates it
once and otherwise retains native layouts. Shared layout I/O is in
``tools.order_layout``.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import pandas as pd

from tools.order_layout import (
    BlockLayout,
    SimpleLayout,
    native_profile_columns,
    read_block_layout,
    read_simple_layout,
    write_block_layout,
    write_simple_layout,
)

BANDWIDTH = 50.0
FINE_WIDTH = 0.01
COARSE_WIDTH = 5.0

SEGMENT_NEGATIVE_MAX = 0.0
SEGMENT_LOW_COST_MAX = 15.0
SEGMENT_LIGNITE_MAX = 100.0
SEGMENT_MID_MERIT_MAX = 180.0
SEGMENT_HIGH_COST_MAX = 270.0

DEMAND_SEGMENT_LOW_FLEXIBILITY_MAX = 0.0
DEMAND_SEGMENT_FLEXIBLE_MAX = 100.0
DEMAND_SEGMENT_PARTIALLY_FLEXIBLE_MAX = 500.0

SIDE = "Sale/Purchase"
MTU_COLUMNS = ["Date", "Hour", "Quarter hour"]
GROUP_COLUMNS = MTU_COLUMNS + [SIDE]

RAW_SIMPLE_REQUIRED = {
    "Date", "Week", "Week Day", "Hour", "Quarter hour", "Price",
    "Volume", SIDE,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger(__name__)


def normalise_hour(values: pd.Series) -> pd.Series:
    """Normalise EPEX hour labels; 3A becomes ordinary hour 3."""
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"3A": "3"})
    )


def price_bin(price: float, clearing: float) -> float:
    """Return the lower price-bin boundary around a clearing price."""
    lower = clearing - BANDWIDTH
    upper = clearing + BANDWIDTH

    if price < lower:
        return lower + math.floor((price - lower) / COARSE_WIDTH) * COARSE_WIDTH
    if price <= upper:
        return lower + math.floor((price - lower) / FINE_WIDTH) * FINE_WIDTH
    return upper + math.floor((price - upper) / COARSE_WIDTH) * COARSE_WIDTH


def price_segment(price: float, side: str) -> str:
    """Assign a supply or demand segment represented by a bid price."""
    if pd.isna(price):
        raise ValueError("Cannot assign a segment to a missing price")

    if side == "Sell":
        if price < SEGMENT_NEGATIVE_MAX:
            return "S0_NEGATIVE"
        if price < SEGMENT_LOW_COST_MAX:
            return "S1_LOW_COST"
        if price < SEGMENT_LIGNITE_MAX:
            return "S2_LIGNITE"
        if price < SEGMENT_MID_MERIT_MAX:
            return "S3_MID_MERIT"
        if price < SEGMENT_HIGH_COST_MAX:
            return "S4_HIGH_COST"
        return "S5_SCARCITY"

    if side == "Purchase":
        if price < DEMAND_SEGMENT_LOW_FLEXIBILITY_MAX:
            return "D0_LOW_FLEXIBILITY"
        if price < DEMAND_SEGMENT_FLEXIBLE_MAX:
            return "D1_FLEXIBLE"
        if price < DEMAND_SEGMENT_PARTIALLY_FLEXIBLE_MAX:
            return "D2_PARTIALLY_FLEXIBLE"
        return "D3_INELASTIC"

    raise ValueError(f"Unsupported order side: {side!r}")


def add_clearing_price_bins(orders: pd.DataFrame, prices_csv: str | Path) -> pd.DataFrame:
    """Add ``_bin`` using published quarter-hourly clearing prices."""
    prices = pd.read_csv(prices_csv, comment="#")
    prices.columns = [str(column).strip() for column in prices.columns]

    if "Delivery day" not in prices.columns:
        raise ValueError("The price file must contain a 'Delivery day' column")

    prices["_date_key"] = pd.to_datetime(
        prices["Delivery day"], dayfirst=True, errors="raise"
    ).dt.normalize()

    hour_columns = [
        column for column in prices.columns
        if re.fullmatch(r"Hour \d+(?:A|B)? Q[1-4]", str(column))
    ]
    if not hour_columns:
        raise ValueError("No 'Hour <n> Q<q>' price columns found")

    long_prices = prices.melt(
        id_vars="_date_key",
        value_vars=hour_columns,
        var_name="_mtu",
        value_name="_clearing",
    )
    parsed = long_prices["_mtu"].str.extract(r"Hour (\d+(?:A|B)?) Q([1-4])")
    long_prices["_raw_hour"] = parsed[0]
    long_prices["Quarter hour"] = parsed[1]
    long_prices = long_prices.loc[long_prices["_raw_hour"].ne("3B")].copy()
    long_prices["Hour"] = normalise_hour(long_prices["_raw_hour"])
    long_prices["_clearing"] = pd.to_numeric(long_prices["_clearing"], errors="coerce")
    long_prices = long_prices[["_date_key", "Hour", "Quarter hour", "_clearing"]]

    retained_mtus = orders[["_date_key", "Hour", "Quarter hour"]].drop_duplicates()
    price_counts = long_prices.groupby(
        ["_date_key", "Hour", "Quarter hour"]
    )["_clearing"].count()
    retained_mtus["_price_count"] = [
        price_counts.get((row["_date_key"], row["Hour"], row["Quarter hour"]), 0)
        for _, row in retained_mtus.iterrows()
    ]
    invalid = retained_mtus.loc[retained_mtus["_price_count"].ne(1)]
    if not invalid.empty:
        raise ValueError(
            "Each retained MTU requires exactly one clearing price.\n"
            f"{invalid.to_string(index=False)}"
        )

    orders = orders.merge(
        long_prices.dropna(subset=["_clearing"]).drop_duplicates(
            ["_date_key", "Hour", "Quarter hour"]
        ),
        on=["_date_key", "Hour", "Quarter hour"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    orders["_bin"] = [
        price_bin(price, clearing)
        for price, clearing in zip(orders["Price"], orders["_clearing"])
    ]
    return orders


def process_simple_orders(
    input_csv: str | Path,
    output_csv: str | Path,
    source_zone: str,
    prices_csv: str | Path | None = None,
) -> None:
    """Convert cumulative simple curves to incremental bids and add a zone."""
    layout: SimpleLayout = read_simple_layout(
        input_csv,
        required_columns=RAW_SIMPLE_REQUIRED,
    )
    orders = layout.frame.copy()

    orders["Hour"] = normalise_hour(orders["Hour"])
    orders["Quarter hour"] = orders["Quarter hour"].astype(str).str.strip()
    orders = orders.loc[orders["Hour"].ne("3B")].copy()
    orders["Price"] = pd.to_numeric(orders["Price"], errors="raise")
    orders["Volume"] = pd.to_numeric(orders["Volume"], errors="raise")
    orders["_date_key"] = pd.to_datetime(
        orders["Date"], dayfirst=True, errors="raise"
    ).dt.normalize()
    orders["bidding_zone"] = source_zone

    previous = orders.groupby(GROUP_COLUMNS, sort=False)["Volume"].shift()
    delta = orders["Volume"] - previous
    orders["Volume"] = delta.where(orders[SIDE].eq("Sell"), -delta)

    orders["_position"] = orders.groupby(GROUP_COLUMNS, sort=False).cumcount()
    orders["_count"] = orders.groupby(GROUP_COLUMNS, sort=False)["Volume"].transform("size")
    orders = orders.loc[
        (orders["_position"] >= 2)
        & (orders["_position"] < orders["_count"] - 2)
        & orders["Volume"].ne(0)
    ].copy()

    if orders.empty:
        raise ValueError("No simple orders remain after preprocessing")

    if prices_csv is None:
        orders["_bin"] = orders["Price"]
    else:
        orders = add_clearing_price_bins(orders, prices_csv)

    orders["_price_volume"] = orders["Price"] * orders["Volume"]
    grouping = GROUP_COLUMNS + ["Week", "Week Day", "bidding_zone", "_bin"]
    result = orders.groupby(grouping, as_index=False, sort=False).agg(
        Volume=("Volume", "sum"),
        _price_volume=("_price_volume", "sum"),
    )
    result = result.loc[result["Volume"].ne(0)].copy()
    result["Price"] = result["_price_volume"] / result["Volume"]
    result["Segment"] = [
        price_segment(price, side)
        for price, side in zip(result["Price"], result[SIDE])
    ]

    columns = [
        "Date", "Week", "Week Day", "Hour", "Quarter hour", "Price", "Volume",
        SIDE, "bidding_zone", "Segment",
    ]
    layout.frame = result[columns]
    write_simple_layout(layout, output_csv)
    LOG.info("Wrote %d preprocessed simple orders to %s", len(result), output_csv)


def process_block_bids(
    input_csv: str | Path,
    output_csv: str | Path,
    source_zone: str,
) -> None:
    """Add zone and segment while preserving signed native block profiles."""
    layout: BlockLayout = read_block_layout(input_csv)
    blocks = layout.frame.copy()
    profiles = native_profile_columns(blocks.columns)

    required = {
        "Data type(BB)", "BlockID", "Block Type", "BlockCodePRM", "Execution",
        "Limit Price", "MAR", "AAR",
    }
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f"Block-order file is missing columns: {sorted(missing)}")

    blocks = blocks.loc[blocks["Data type(BB)"].eq("BB")].copy()
    blocks["Limit Price"] = pd.to_numeric(blocks["Limit Price"], errors="raise")
    blocks[profiles] = blocks[profiles].apply(pd.to_numeric, errors="raise")

    positive = blocks[profiles].gt(0).any(axis=1)
    negative = blocks[profiles].lt(0).any(axis=1)
    if (positive & negative).any():
        raise ValueError("Mixed-sign block profiles are not supported")

    blocks = blocks.loc[positive | negative].copy()
    side = pd.Series("Sell", index=blocks.index)
    side.loc[negative.loc[blocks.index]] = "Purchase"

    blocks["bidding_zone"] = source_zone
    blocks["Segment"] = [
        price_segment(price, bid_side)
        for price, bid_side in zip(blocks["Limit Price"], side)
    ]

    for column in ["bidding_zone", "Segment"]:
        if column not in layout.header:
            layout.header.append(column)

    layout.frame = blocks.reindex(columns=layout.header)
    write_block_layout(layout, output_csv)
    LOG.info("Wrote %d preprocessed block orders to %s", len(blocks), output_csv)


def preprocess_orders(
    simple_input: str | Path,
    block_input: str | Path,
    simple_output: str | Path,
    block_output: str | Path,
    source_zone: str = "DE-LU",
    prices_csv: str | Path | None = None,
) -> None:
    """Run preprocessing for both published input files."""
    source_zone = str(source_zone).strip()
    if not source_zone:
        raise ValueError("source_zone must be a non-empty string")

    process_simple_orders(simple_input, simple_output, source_zone, prices_csv)
    process_block_bids(block_input, block_output, source_zone)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simple_input", type=Path)
    parser.add_argument("block_input", type=Path)
    parser.add_argument("simple_output", type=Path)
    parser.add_argument("block_output", type=Path)
    parser.add_argument("--source-zone", default="DE-LU")
    parser.add_argument("--prices-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_orders(
        simple_input=args.simple_input,
        block_input=args.block_input,
        simple_output=args.simple_output,
        block_output=args.block_output,
        source_zone=args.source_zone,
        prices_csv=args.prices_csv,
    )


if __name__ == "__main__":
    main()
