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
import csv

import numpy as np
import pandas as pd

from tools.units import mwh_to_mw
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


def bin_simple_steps(
    orders: pd.DataFrame,
    prices_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin MW step bids without changing historical MCP acceptance."""
    orders = add_clearing_price_bins(orders, prices_csv)
    orders["Segment"] = [
        price_segment(price, side)
        for price, side in zip(orders["Price"], orders[SIDE])
    ]
    orders["_mcp_relation"] = np.select(
        [
            orders["Price"].lt(orders["_clearing"]),
            orders["Price"].gt(orders["_clearing"]),
        ],
        ["below", "above"],
        default="at",
    )
    orders["_price_volume"] = orders["Price"] * orders["Volume"]

    grouping = GROUP_COLUMNS + [
        "Week", "Week Day", "bidding_zone", "Segment",
        "_bin", "_mcp_relation",
    ]
    result = orders.groupby(
        grouping, as_index=False, sort=False, dropna=False
    ).agg(
        Volume=("Volume", "sum"),
        _price_volume=("_price_volume", "sum"),
        _first_price=("Price", "first"),
        _clearing=("_clearing", "first"),
    )

    result["Price"] = result["_first_price"]
    nonzero = result["Volume"].ne(0.0)
    result.loc[nonzero, "Price"] = (
        result.loc[nonzero, "_price_volume"]
        / result.loc[nonzero, "Volume"]
    )

    # Prevent floating-point averaging from crossing onto the MCP boundary.
    below = result["_mcp_relation"].eq("below")
    above = result["_mcp_relation"].eq("above")
    at = result["_mcp_relation"].eq("at")

    result.loc[below, "Price"] = np.minimum(
        result.loc[below, "Price"].to_numpy(),
        np.nextafter(result.loc[below, "_clearing"].to_numpy(), -np.inf),
    )
    result.loc[above, "Price"] = np.maximum(
        result.loc[above, "Price"].to_numpy(),
        np.nextafter(result.loc[above, "_clearing"].to_numpy(), np.inf),
    )
    result.loc[at, "Price"] = result.loc[at, "_clearing"]

    if not np.isfinite(result[["Price", "Volume"]].to_numpy(dtype=float)).all():
        raise ValueError("Binning produced non-finite prices or volumes")
    if result["Volume"].lt(0.0).any():
        raise ValueError("Binning produced negative volumes")

    groups = orders.groupby(
        GROUP_COLUMNS, sort=False, dropna=False
    ).size().index

    def totals(frame: pd.DataFrame, matched: bool) -> pd.Series:
        selected = frame
        if matched:
            accepted = (
                (frame[SIDE].eq("Sell") & frame["Price"].le(frame["_clearing"]))
                | (
                    frame[SIDE].eq("Purchase")
                    & frame["Price"].ge(frame["_clearing"])
                )
            )
            selected = frame.loc[accepted]
        return (
            selected.groupby(GROUP_COLUMNS, sort=False, dropna=False)["Volume"]
            .sum().reindex(groups, fill_value=0.0)
        )

    gross_before, gross_after = totals(orders, False), totals(result, False)
    matched_before, matched_after = totals(orders, True), totals(result, True)

    for before, after, label in [
        (gross_before, gross_after, "submitted volume"),
        (matched_before, matched_after, "MCP-accepted volume"),
    ]:
        if not np.allclose(before, after, atol=1e-6, rtol=1e-8):
            raise ValueError(f"Binning changed {label}")

    diagnostics = groups.to_frame(index=False)
    diagnostics["submitted_before_binning_mw"] = gross_before.to_numpy()
    diagnostics["submitted_after_binning_mw"] = gross_after.to_numpy()
    diagnostics["matched_before_binning_mw"] = matched_before.to_numpy()
    diagnostics["matched_after_binning_mw"] = matched_after.to_numpy()
    diagnostics["matched_binning_error_mw"] = (
        matched_after - matched_before
    ).to_numpy()

    columns = [
        "Date", "Week", "Week Day", "Hour", "Quarter hour",
        "Price", "Volume", SIDE, "bidding_zone", "Segment",
    ]
    return result[columns], diagnostics


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
    prices = pd.read_csv(prices_csv, comment="#").copy()
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


def read_mtu_prices(prices_csv: str | Path, day: pd.Timestamp) -> dict:
    """Return the 96 quarter-hour clearing prices for an ordinary day."""
    prices = pd.read_csv(prices_csv, comment="#", encoding="utf-8-sig")
    prices.columns = prices.columns.str.strip()

    if "Delivery day" not in prices:
        raise ValueError("Prices CSV lacks 'Delivery day'")

    dates = pd.to_datetime(
        prices["Delivery day"], dayfirst=True, errors="raise"
    ).dt.normalize()
    selected = prices.loc[dates.eq(day)]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one price row for {day.date()}, found {len(selected)}"
        )

    result = {}
    row = selected.iloc[0]

    for column in prices.columns:
        match = re.fullmatch(r"Hour (\d+)([AB]?) Q([1-4])", column)
        if match is None or pd.isna(row[column]) or not str(row[column]).strip():
            continue

        hour, suffix, quarter = int(match[1]), match[2], int(match[3])
        if suffix == "B" or (suffix == "A" and hour != 3):
            raise ValueError("DST price coordinates are unsupported")

        key = (str(hour), str(quarter))
        if key in result:
            raise ValueError(f"Duplicate price for hour/quarter {key}")

        value = float(pd.to_numeric(row[column], errors="raise"))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite price in {column}")

        result[key] = value

    expected = {
        (str(hour), str(quarter))
        for hour in range(1, 25)
        for quarter in range(1, 5)
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        raise ValueError(f"Missing quarter-hour prices: {missing}")

    return result


def read_net_positions(
    path: str | Path,
    day: pd.Timestamp,
    source_zone: str,
) -> dict:
    """Read daily import/export positions and convert MW to quarter-hour MWh."""
    # Select by position because the source has duplicate column names.
    data = pd.read_csv(
        path,
        usecols=[0, 1, 2, 3],
        dtype=str,
        encoding="utf-8-sig",
    )
    data.columns = ["mtu", "area", "side", "mw"]

    area = source_zone
    if not area.startswith("BZN|"):
        area = f"BZN|{area}"

    data = data.loc[data["area"].str.strip().eq(area)].copy()
    intervals = data["mtu"].str.split(" - ", n=1, expand=True)
    if intervals.shape[1] != 2:
        raise ValueError("Invalid net-position MTU interval format")

    # The source timestamps are local CET/CEST. Remove the labels before
    # parsing; the pipeline does not support DST transition days.
    start_text = intervals[0].str.strip().str.replace(
        r"\s*\((?:CET|CEST)\)\s*$", "", regex=True
    )
    end_text = intervals[1].str.strip().str.replace(
        r"\s*\((?:CET|CEST)\)\s*$", "", regex=True
    )

    start = pd.to_datetime(
        start_text, format="%d/%m/%Y %H:%M", errors="raise"
    )
    end = pd.to_datetime(
        end_text, format="%d/%m/%Y %H:%M", errors="raise"
    )
    data["_start"] = start
    data["_end"] = end
    data = data.loc[data["_start"].dt.normalize().eq(day)].copy()

    if not (data["_end"] - data["_start"]).eq(
        pd.Timedelta(minutes=15)
    ).all():
        raise ValueError("Net-position intervals must be 15 minutes")

    data["side"] = data["side"].str.strip().str.lower()
    data["mw"] = pd.to_numeric(data["mw"], errors="raise")

    result = {}
    for start, side, mw in data[["_start", "side", "mw"]].itertuples(
        index=False, name=None
    ):
        if side not in {"import", "export"}:
            raise ValueError(f"Unexpected net-position side: {side!r}")
        if not math.isfinite(mw) or mw < 0:
            raise ValueError("Net-position MW must be finite and nonnegative")
        if start.minute not in {0, 15, 30, 45}:
            raise ValueError("Net-position interval is not quarter-hour aligned")

        key = (str(start.hour + 1), str(start.minute // 15 + 1))
        if key in result:
            raise ValueError(f"Duplicate net position for {key}")

        mwh = mw * 0.25
        result[key] = {
            "Sell": mwh if side == "import" else 0.0,
            "Purchase": mwh if side == "export" else 0.0,
        }

    expected = {
        (str(hour), str(quarter))
        for hour in range(1, 25)
        for quarter in range(1, 5)
    }
    if set(result) != expected:
        raise ValueError(
            f"Missing net positions for {day.date()}: "
            f"{sorted(expected - set(result))}"
        )

    return result


def cumulative_to_steps(
    curve: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Convert ascending-price cumulative curves to nonnegative step volumes.

    The first sell volume and last purchase volume are retained as boundary
    bids. Differences are assigned to the current row's price, matching the
    existing preprocessing convention. Zero-volume curve points are retained.
    """
    curve = curve.copy()
    grouped = curve.groupby(group_columns, sort=False, dropna=False)

    price_change = grouped["Price"].diff()
    if price_change.lt(0).any():
        raise ValueError("Curve prices must be in ascending source order")

    previous = grouped["Volume"].shift()
    delta = curve["Volume"] - previous

    sell = curve[SIDE].eq("Sell")
    purchase = curve[SIDE].eq("Purchase")
    if not (sell | purchase).all():
        raise ValueError("Curve side must be Sell or Purchase")

    # A sell curve begins at the price floor.
    steps = delta.where(sell, -delta)
    first = previous.isna()
    steps.loc[first & sell] = curve.loc[first & sell, "Volume"]
    steps.loc[first & purchase] = 0.0

    # A purchase curve can retain demand at the highest price.
    last = grouped.cumcount(ascending=False).eq(0)
    steps.loc[last & purchase] += curve.loc[last & purchase, "Volume"]

    tolerance = 1e-7
    if (steps < -tolerance).any():
        raise ValueError("Curve produced negative step volumes")

    # Remove only numerical noise; retain the rows themselves.
    curve["Volume"] = steps.clip(lower=0.0)
    return curve


def remove_loop_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    """Remove reciprocal two-block C02 loops without modifying other bids."""

    def normalise_ids(values: pd.Series) -> pd.Series:
        # Normalize only for comparison; preserve original output fields.
        return (
            values.astype("string")
            .fillna("")
            .str.strip()
            .str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        )

    ids = normalise_ids(blocks["BlockID"])
    references = normalise_ids(blocks["BlockCodePRM"])
    linked = blocks["Block Type"].astype("string").str.strip().eq("C02")

    if ids.eq("").any():
        raise ValueError("Block orders contain an empty BlockID")

    if ids.duplicated().any():
        duplicates = ids.loc[ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "Loop filtering requires unique BlockIDs within the input "
            f"delivery day. Duplicate IDs: {duplicates}"
        )

    links = dict(zip(ids.loc[linked], references.loc[linked]))
    pairs = set()

    for block_id, reference in links.items():
        if reference != block_id and links.get(reference) == block_id:
            pairs.add(tuple(sorted((block_id, reference))))

    if not pairs:
        return blocks

    removed_ids = {
        block_id
        for pair in pairs
        for block_id in pair
    }

    # Do not leave retained linked blocks pointing to deleted blocks.
    external_links = (
        linked
        & ~ids.isin(removed_ids)
        & references.isin(removed_ids)
    )
    if external_links.any():
        relationships = [
            f"{block_id} -> {reference}"
            for block_id, reference in zip(
                ids.loc[external_links],
                references.loc[external_links],
            )
        ]
        raise ValueError(
            "Cannot remove loop blocks: other linked blocks reference them: "
            + ", ".join(relationships)
        )

    result = blocks.loc[~ids.isin(removed_ids)].copy()

    for first, second in sorted(pairs):
        LOG.warning(
            "Removed EPEX loop block %s (reciprocal partner %s).",
            first,
            second,
        )
        LOG.warning(
            "Removed EPEX loop block %s (reciprocal partner %s).",
            second,
            first,
        )

    return result


def read_validated_source_blocks(
    path: str | Path,
    day: pd.Timestamp | None = None,
) -> tuple[BlockLayout, list[str], pd.Series]:
    """Read raw MWh blocks without removing or changing relationships."""
    layout = read_block_layout(path)
    blocks = layout.frame.copy().reset_index(drop=True)

    if len(layout.header) != len(set(layout.header)):
        raise ValueError("Duplicate block-column names")

    required = {
        "Data type(BB)", "BlockID", "Block Type", "BlockCodePRM",
        "Execution", "Limit Price", "MAR", "AAR",
    }
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Missing block fields: {sorted(missing)}")

    if {"bidding_zone", "Segment"} & set(blocks.columns):
        raise ValueError(
            "Block input appears already preprocessed. "
            "Do not convert an MW order book from MWh again."
        )

    if not blocks["Data type(BB)"].eq("BB").all():
        raise ValueError("Unexpected non-BB block records")

    # Reciprocal C02 loops are valid source-market constructs but unsupported
    # by our clearing model. Remove them before validating export relationships.
    blocks = remove_loop_blocks(blocks).reset_index(drop=True)

    dates = []
    for line in layout.st_rows:
        fields = next(csv.reader([line], delimiter=";"))
        if len(fields) < 2:
            raise ValueError("Malformed ST metadata")
        dates.append(
            pd.to_datetime(
                fields[1], dayfirst=True, errors="raise"
            ).normalize()
        )

    if not dates or len(set(dates)) != 1:
        raise ValueError("Block input must identify one delivery day")

    block_day = dates[0]
    if day is not None and block_day != pd.Timestamp(day).normalize():
        raise ValueError(
            f"Block delivery date {block_day.date()} does not match "
            f"{pd.Timestamp(day).date()}"
        )

    start = block_day.tz_localize("Europe/Berlin")
    end = (block_day + pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
    if end - start != pd.Timedelta(hours=24):
        raise ValueError("DST block delivery days are unsupported")

    def numeric_profile(column: str) -> pd.Series:
        values = pd.to_numeric(
            blocks[column].replace(r"^\s*$", np.nan, regex=True),
            errors="raise",
        ).fillna(0.0)
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite block profile: {column}")
        return values

    # A zero H03B placeholder is harmless; an active second occurrence is not.
    for column in blocks.columns:
        if not str(column).startswith("Volume H"):
            continue

        match = re.fullmatch(r"Volume H(\d{2})([AB]?) Q([1-4])", column)
        if match is None:
            raise ValueError(f"Malformed block-profile column: {column}")

        hour, suffix = int(match[1]), match[2]
        if not 1 <= hour <= 24:
            raise ValueError(f"Unexpected block-profile hour: {column}")
        if suffix and hour != 3:
            raise ValueError(f"Unexpected block-profile suffix: {column}")
        if suffix == "B" and numeric_profile(column).ne(0.0).any():
            raise ValueError("Active H03B profiles require an unsupported DST day")

    profiles = native_profile_columns(blocks.columns)
    coordinates = []
    for column in profiles:
        match = re.fullmatch(r"Volume H(\d{2})(?:A)? Q([1-4])", column)
        coordinates.append((int(match[1]), int(match[2])))

    expected = {
        (hour, quarter)
        for hour in range(1, 25)
        for quarter in range(1, 5)
    }
    if len(set(coordinates)) != 96 or set(coordinates) != expected:
        raise ValueError("Expected exactly 96 unique ordinary block coordinates")

    blocks[profiles] = pd.DataFrame(
        {column: numeric_profile(column) for column in profiles},
        index=blocks.index,
    )

    positive = blocks[profiles].gt(0.0).any(axis=1)
    negative = blocks[profiles].lt(0.0).any(axis=1)
    invalid_profiles = (positive & negative) | ~(positive | negative)
    if invalid_profiles.any():
        raise ValueError(
            "Block profiles must be nonempty and have one sign: "
            + ", ".join(
                blocks.loc[invalid_profiles, "BlockID"].astype(str)
            )
        )

    ids = blocks["BlockID"].fillna("").astype(str).str.strip()
    parameters = blocks["BlockCodePRM"].fillna("").astype(str).str.strip()
    types = blocks["Block Type"]

    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("BlockID must be nonempty and unique")

    blocks["BlockID"] = ids
    blocks["BlockCodePRM"] = parameters

    if not types.isin(["C01", "C02", "C04"]).all():
        raise ValueError("Unsupported block type")
    if parameters.loc[types.eq("C01")].ne("").any():
        raise ValueError("C01 blocks must have empty BlockCodePRM")
    if parameters.loc[types.eq("C04")].eq("").any():
        raise ValueError("C04 blocks require a nonempty exclusive-group code")

    prices = pd.to_numeric(blocks["Limit Price"], errors="raise")
    mar = pd.to_numeric(blocks["MAR"], errors="raise")
    aar = pd.to_numeric(blocks["AAR"], errors="raise")

    for values, label in [(prices, "Limit Price"), (mar, "MAR"), (aar, "AAR")]:
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Block {label} must be finite")
    if not mar.between(0.0, 1.0).all():
        raise ValueError("Block MAR must be in [0, 1]")
    if not aar.between(0.0, 1.0).all():
        raise ValueError("Block AAR must be in [0, 1]")

    execution = (
        blocks["Execution"].astype("string")
        .fillna("").str.strip().str.upper()
    )
    if not execution.isin(["Y", "N"]).all():
        raise ValueError("Block Execution must be Y or N")

    inconsistent = (
        (execution.eq("N") & aar.ne(0.0))
        | (execution.eq("Y") & aar.eq(0.0))
    )
    if inconsistent.any():
        raise ValueError(
            "Inconsistent historical Execution/AAR: "
            + ", ".join(ids.loc[inconsistent])
        )

    linked = types.eq("C02")
    parents = dict(zip(ids.loc[linked], parameters.loc[linked]))
    known_ids = set(ids)
    missing_parents = [
        f"{child}->{parent}"
        for child, parent in parents.items()
        if parent not in known_ids
    ]
    if missing_parents:
        raise ValueError(
            "Unknown or empty C02 parents: " + ", ".join(missing_parents)
        )

    for block_id in ids:
        seen = set()
        current = block_id
        while current in parents:
            if current in seen:
                raise ValueError(f"C02 cycle containing {block_id}")
            seen.add(current)
            current = parents[current]

    acceptance = dict(zip(ids, aar))
    invalid_acceptance = [
        f"{child}->{parent}"
        for child, parent in parents.items()
        if acceptance[child] > 0.0 and acceptance[parent] == 0.0
    ]
    if invalid_acceptance:
        raise ValueError(
            "Historically accepted children have rejected parents: "
            + ", ".join(invalid_acceptance)
        )

    accepted_exclusive = types.eq("C04") & aar.gt(0.0)
    accepted_counts = ids.loc[accepted_exclusive].groupby(
        parameters.loc[accepted_exclusive], sort=False
    ).size()
    if accepted_counts.gt(1).any():
        raise ValueError(
            "Multiple historically accepted C04 alternatives: "
            + ", ".join(accepted_counts.index[accepted_counts.gt(1)])
        )

    blocks["Limit Price"] = prices
    layout.frame = blocks
    return layout, profiles, aar.astype(float)


def read_accepted_blocks(path: str | Path, day: pd.Timestamp) -> dict:
    """Return historical accepted raw block MWh per MTU and side.

    Include reciprocal C02 loops here: their historical accepted volumes
    contributed to the published curves, even though they are omitted from
    the clearing-model input.
    """
    layout = read_block_layout(path)
    blocks = layout.frame.copy()
    profiles = native_profile_columns(blocks.columns)

    if "AAR" not in blocks or "Execution" not in blocks:
        raise ValueError("Block file requires AAR and Execution columns")

    dates = []
    for line in layout.st_rows:
        fields = next(csv.reader([line], delimiter=";"))
        if len(fields) < 2:
            raise ValueError("Malformed ST metadata")
        dates.append(
            pd.to_datetime(
                fields[1], dayfirst=True, errors="raise"
            ).normalize()
        )

    if not dates or any(date != day for date in dates):
        raise ValueError(
            f"Block-file delivery dates do not match {day.date()}: {dates}"
        )

    execution = blocks["Execution"].str.strip().str.upper()
    if not execution.isin(["Y", "N"]).all():
        raise ValueError("Block Execution must be Y or N")

    aar = pd.to_numeric(blocks["AAR"], errors="raise")
    if (
        not np.isfinite(aar.to_numpy(dtype=float)).all()
        or not aar.between(0.0, 1.0).all()
    ):
        raise ValueError("Block AAR must be a finite fraction in [0, 1]")

    if (execution.eq("N") & aar.ne(0.0)).any():
        raise ValueError("A rejected block has a nonzero AAR")

    values = blocks[profiles].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Block profiles contain non-finite values")

    accepted = values.mul(aar, axis=0)
    result = {}

    for column in profiles:
        match = re.fullmatch(r"Volume H(\d{2})(?:A)? Q([1-4])", column)
        if match is None:
            raise ValueError(f"Unexpected block profile column: {column}")

        key = (str(int(match[1])), match[2])
        if key in result:
            raise ValueError(f"Duplicate block profile coordinate: {key}")

        result[key] = {
            "Sell": float(-accepted[column].clip(upper=0.0).sum()),
            "Purchase": float(accepted[column].clip(lower=0.0).sum()),
        }

    return result


def process_block_bids(
    input_csv: str | Path,
    output_csv: str | Path,
    source_zone: str,
) -> None:
    """Validate raw blocks, convert MWh to MW once, and add metadata."""
    source_zone = str(source_zone).strip()
    if not source_zone:
        raise ValueError("source_zone must be nonempty")

    layout, profiles, _ = read_validated_source_blocks(input_csv)
    blocks = layout.frame.copy()

    side = np.where(
        blocks[profiles].lt(0.0).any(axis=1), "Sell", "Purchase"
    )
    blocks[profiles] = mwh_to_mw(blocks[profiles])
    if not np.isfinite(blocks[profiles].to_numpy(dtype=float)).all():
        raise ValueError("MWh-to-MW conversion produced non-finite profiles")

    blocks["bidding_zone"] = source_zone
    blocks["Segment"] = [
        price_segment(price, bid_side)
        for price, bid_side in zip(blocks["Limit Price"], side)
    ]

    layout.header.extend(["bidding_zone", "Segment"])
    layout.frame = blocks.reindex(columns=layout.header)
    write_block_layout(layout, output_csv)
    LOG.info("Wrote %d preprocessed block orders to %s", len(blocks), output_csv)


def process_simple_orders(
    input_csv: str | Path,
    output_csv: str | Path,
    source_zone: str,
    prices_csv: str | Path,
    block_input: str | Path,
    net_positions_csv: str | Path,
) -> None:
    """Deshift 15-minute curves, then convert and bin them as step bids."""
    layout = read_simple_layout(
        input_csv,
        required_columns=RAW_SIMPLE_REQUIRED,
    )
    orders = layout.frame.copy()

    if {"bidding_zone", "Segment"} & set(orders.columns):
        raise ValueError(
            "Simple input appears already preprocessed; refusing a second "
            "cumulative conversion or MWh-to-MW conversion."
        )
    
    orders["_date_key"] = pd.to_datetime(
        orders["Date"], dayfirst=True, errors="raise"
    ).dt.normalize()

    days = orders["_date_key"].unique()
    if len(days) != 1:
        raise ValueError("The 15-minute curve file must contain one delivery day")

    day = pd.Timestamp(days[0])
    start = day.tz_localize("Europe/Berlin")
    end = (day + pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
    if end - start != pd.Timedelta(hours=24):
        raise ValueError("DST delivery days are unsupported")

    raw_hour = orders["Hour"].astype(str).str.strip()
    if raw_hour.eq("3B").any():
        raise ValueError("DST curve coordinates are unsupported")

    orders["Hour"] = normalise_hour(orders["Hour"])
    orders["Quarter hour"] = (
        orders["Quarter hour"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    orders[SIDE] = orders[SIDE].astype(str).str.strip()
    orders["Price"] = pd.to_numeric(orders["Price"], errors="raise")
    orders["Volume"] = pd.to_numeric(orders["Volume"], errors="raise")

    if not orders["Price"].map(math.isfinite).all():
        raise ValueError("15-minute curve contains non-finite prices")
    if not orders["Volume"].map(math.isfinite).all():
        raise ValueError("15-minute curve contains non-finite volumes")

    expected = {
        (str(hour), str(quarter), side)
        for hour in range(1, 25)
        for quarter in range(1, 5)
        for side in ("Sell", "Purchase")
    }
    present = set(
        zip(orders["Hour"], orders["Quarter hour"], orders[SIDE])
    )
    if present != expected:
        raise ValueError(
            "15-minute curve has missing or unexpected MTU/side groups. "
            f"Missing: {sorted(expected - present)}; "
            f"unexpected: {sorted(present - expected)}"
        )

    mtu_prices = read_mtu_prices(prices_csv, day)
    net = read_net_positions(net_positions_csv, day, source_zone)
    blocks = read_accepted_blocks(block_input, day)

    shifts = []
    for hour, quarter, side in zip(
        orders["Hour"], orders["Quarter hour"], orders[SIDE]
    ):
        mtu = (hour, quarter)
        key = (hour, quarter, side)
        shifts.append(
            net[mtu][side]
            + blocks[mtu][side]
        )

    # Inspect the source endpoints and each shift component before subtraction.
    diagnostic_rows = []

    for (hour, quarter, side), curve in orders.groupby(
        ["Hour", "Quarter hour", SIDE], sort=False
    ):
        mtu = (hour, quarter)
        key = (hour, quarter, side)

        net_mwh = net[mtu][side]
        block_mwh = blocks[mtu][side]
        total_shift = net_mwh + block_mwh 

        first = curve.iloc[0]
        last = curve.iloc[-1]
        penultimate = curve.iloc[-2] if len(curve) >= 2 else last

        diagnostic_rows.append({
            "Date": first["Date"],
            "Hour": hour,
            "Quarter hour": quarter,
            SIDE: side,
            "net_position_mwh": net_mwh,
            "accepted_blocks_mwh": block_mwh,
            "total_shift_mwh": total_shift,
            "first_price": first["Price"],
            "first_volume_mwh": first["Volume"],
            "penultimate_price": penultimate["Price"],
            "penultimate_volume_mwh": penultimate["Volume"],
            "last_price": last["Price"],
            "last_volume_mwh": last["Volume"],
            "min_deshifted_mwh": curve["Volume"].min() - total_shift,
        })

    diagnostic_path = Path(output_csv).with_name("deshift_diagnostics.csv")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
    LOG.info("Wrote deshifting diagnostics to %s", diagnostic_path)
    
    # Both the source curve and the calculated shift are quarter-hour MWh.
    orders["Volume"] -= shifts

    tolerance = 1e-7
    negative = orders["Volume"].lt(-tolerance)
    if negative.any():
        examples = orders.loc[
            negative, ["Date", "Hour", "Quarter hour", "Price", SIDE, "Volume"]
        ].head(10)
        raise ValueError(
            "Deshifting produced negative cumulative volumes. Check dates, "
            "units, net-position convention and estimated coarser executions. "
            "JAO rounding and ignored complex orders may also matter.\n"
            f"{examples.to_string(index=False)}"
        )

    # Clip only floating-point noise, without dropping any source rows.
    orders["Volume"] = orders["Volume"].clip(lower=0.0)
    orders["bidding_zone"] = source_zone

    # Deshifting happens before cumulative-to-incremental conversion.
    orders = cumulative_to_steps(orders, GROUP_COLUMNS)

    # The downstream pipeline expects quarter-hour bid quantities in MW.
    orders["Volume"] = mwh_to_mw(orders["Volume"])

    result, binning_report = bin_simple_steps(orders, prices_csv)

    binning_path = Path(output_csv).with_name("simple_binning_diagnostics.csv")
    binning_path.parent.mkdir(parents=True, exist_ok=True)
    binning_report.to_csv(binning_path, index=False)

    layout.frame = result
    write_simple_layout(layout, output_csv)
    LOG.info("Wrote %d preprocessed simple orders to %s", len(result), output_csv)


def preprocess_orders(
    simple_input: str | Path,
    block_input: str | Path,
    simple_output: str | Path,
    block_output: str | Path,
    prices_csv: str | Path,
    net_positions_csv: str | Path,

    source_zone: str = "DE-LU",
) -> None:
    """Deshift curves and preprocess both simple and block orders."""
    source_zone = str(source_zone).strip()
    if not source_zone:
        raise ValueError("source_zone must be non-empty")

    process_simple_orders(
        input_csv=simple_input,
        output_csv=simple_output,
        source_zone=source_zone,
        prices_csv=prices_csv,
        block_input=block_input,
        net_positions_csv=net_positions_csv,
    )
    process_block_bids(block_input, block_output, source_zone)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simple_input", type=Path)
    parser.add_argument("block_input", type=Path)
    parser.add_argument("simple_output", type=Path)
    parser.add_argument("block_output", type=Path)
    parser.add_argument("--source-zone", default="DE-LU")
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument("--net-positions-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_orders(
        simple_input=args.simple_input,
        block_input=args.block_input,
        simple_output=args.simple_output,
        block_output=args.block_output,
        prices_csv=args.prices_csv,
        net_positions_csv=args.net_positions_csv,
        source_zone=args.source_zone,
    )


if __name__ == "__main__":
    main()