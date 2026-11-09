"""Create network-led zonal order books from preprocessed bids.

Demand calibration
------------------
Historical matched demand consists exclusively of:
* simple purchases with Price >= the historical MCP; and
* purchase-block profiles weighted by Execution: Y=1, N=0.

Every purchase block receives one common zone/day multiplier. Simple purchases
receive one multiplier per zone/MTU so benchmark matched demand equals PyPSA
load. This is calibration at historical prices and Execution, not a prediction
of acceptance in the new clearing. Gross submitted demand is not constrained.

Supply calibration
------------------
Installed generator p_nom and snapshot-matched p_max_pu determine available MW.
Static p_max_pu is used only when a generator has no dynamic column.

Each zone/supply segment receives one daily multiplier:
available-generation MWh / original offered-template MWh.

Supply-template and physical-budget calculations sum non-exclusive blocks and
per-MTU maxima within each exclusive group. Groups spanning segments are
budgeted conservatively using a separate maximum in each segment. All supply
alternatives are checked; physical reductions use one common reduction across
the sell alternatives of each exclusive group. Simple supply is subsequently
capped at remaining segment availability, never increased to fill headroom.

No historical supply/demand ratios, matched-to-total-demand shares, or local
balance constraints are imposed. Network-constrained clearing handles deficits,
surpluses, transmission constraints, and actual acceptance.

Preservation
------------
Prices, metadata, native signs, C02 parent references, C04 groups, and zonal
block-ID prefixes are preserved. Every block transformation is a constant
whole-profile multiplier. Only exact zeros are filtered, and filtering that
would leave a retained C02 child without its parent raises an informative error.

Unbacked scarcity supply is omitted. To back scarcity explicitly, assign real
generator carriers to S5_SCARCITY in SEGMENT_TYPES and remove those carriers
from their other segments. A carrier cannot be counted twice.

All daily energies use dt=0.25 hours. Ordinary 96-MTU delivery days are supported;
DST delivery days and multi-investment-period snapshots are not.

Run with --self-test for synthetic regression checks. Production transformations
also validate demand calibration, availability limits, nonnegative magnitudes,
metadata, relationships, and constant block-profile multipliers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa

from tools.order_layout import (
    BlockLayout,
    SimpleLayout,
    native_profile_columns,
    read_block_layout,
    read_simple_layout,
    write_block_layout,
    write_simple_layout,
)

SIDE = "Sale/Purchase"
MTUS = list(range(1, 97))
TOL = 1e-6
RTOL = 1e-8
QUARTER_HOURS = 0.25
MARKET_TIMEZONE = "Europe/Berlin"
SCARCITY_SEGMENT = "S5_SCARCITY"

# Compatibility constants. Neither raw curves nor historical scarcity shares
# are used by this transformation.
ORIGINAL_CURVES_CSV: Path | None = None
PRICES_CSV: Path | None = None
SCARCITY_SHARE: float | None = None

# Each carrier must belong to exactly one segment. Explicitly backing scarcity
# requires reassignment here, not an additional copy of existing capacity.
SEGMENT_TYPES = {
    "S0_NEGATIVE": ["Wind", "Wind Offshore", "PV"],
    "S1_LOW_COST": ["Nuclear"],
    "S2_LIGNITE": ["Lignite"],
    "S3_MID_MERIT": ["Hard Coal"],
    "S4_HIGH_COST": ["Gas/CCGT"],
    SCARCITY_SEGMENT: [],
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger(__name__)


def finite(values, label: str, nonnegative: bool = False) -> None:
    """Reject non-finite values and, optionally, negative magnitudes."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{label} contains negative values")


def require_close(actual, expected, label: str) -> None:
    """Check numerical equality without relying on Python assert statements."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    finite(actual, label)
    finite(expected, f"{label} reference")
    if not np.allclose(actual, expected, atol=TOL, rtol=RTOL):
        error = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"{label}: maximum absolute error is {error:g}")


def require_within(actual, limit, label: str) -> None:
    """Validate a nonnegative physical budget against its available MW."""
    actual = np.asarray(actual, dtype=float)
    limit = np.asarray(limit, dtype=float)
    finite(actual, label, nonnegative=True)
    finite(limit, f"{label} availability", nonnegative=True)
    if (actual > limit + TOL + RTOL * np.abs(limit)).any():
        excess = float(np.max(actual - limit))
        raise ValueError(f"{label} exceeds availability by up to {excess:g} MW")


def normalize_zone_label(value) -> str:
    """Return a nonempty label distinguishable from numerical bus IDs."""
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Empty bidding-zone label")
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid numerical zone label: {value!r}")
    return f"Z{int(number)}"


def normalize_network_zones(network, zone_column: str) -> list[str]:
    """Normalize bus zones in place and return deterministic zone ordering."""
    if zone_column not in network.buses:
        raise ValueError(f"Network buses do not contain {zone_column!r}")
    network.buses[zone_column] = network.buses[zone_column].map(
        normalize_zone_label
    )
    zones = sorted(network.buses[zone_column].unique(), key=str)
    if not zones:
        raise ValueError("Network contains no zoned buses")
    return zones


def date_keys(values: pd.Series) -> pd.Series:
    """Parse published day-first delivery dates."""
    return pd.to_datetime(
        values, dayfirst=True, errors="raise"
    ).dt.normalize()


def delivery_day(value) -> pd.Timestamp:
    """Require an ordinary 24-hour German-market delivery day."""
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert(MARKET_TIMEZONE).tz_localize(None)
    day = day.normalize()
    start = day.tz_localize(MARKET_TIMEZONE)
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TIMEZONE)
    if end - start != pd.Timedelta(hours=24):
        raise ValueError(
            f"DST delivery day {day.date()} is unsupported; 96 MTUs required"
        )
    return day


def add_mtu(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate hour/quarter coordinates and append internal MTU."""
    frame = frame.copy()
    frame["Hour"] = (
        frame["Hour"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"3A": "3"})
    )
    for column, upper in [("Hour", 24), ("Quarter hour", 4)]:
        values = pd.to_numeric(frame[column], errors="raise")
        finite(values, column)
        if (
            (values < 1)
            | (values > upper)
            | (values != np.floor(values))
        ).any():
            raise ValueError(f"Invalid {column}; DST labels are unsupported")
        frame[column] = values.astype(int)
    frame["MTU"] = (frame["Hour"] - 1) * 4 + frame["Quarter hour"]
    return frame


def load_simple_orders(path, day=None) -> tuple[SimpleLayout, pd.Timestamp]:
    """Select one delivery day without repeating preprocessing."""
    layout = read_simple_layout(
        path, required_columns={"bidding_zone", "Segment"}
    )
    dates = date_keys(layout.frame["Date"])
    if day is None:
        if dates.nunique() != 1:
            raise ValueError(
                "Specify delivery_date when simple input contains multiple days"
            )
        day = dates.iloc[0]
    day = delivery_day(day)
    frame = layout.frame.loc[dates.eq(day)].copy()
    if frame.empty:
        raise ValueError(f"No simple orders match {day.date()}")
    if "MTU" in frame:
        raise ValueError("Input column 'MTU' is reserved")
    frame = add_mtu(frame)
    for column in ["Price", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        finite(
            frame[column], f"Simple {column}", nonnegative=column == "Volume"
        )
    if frame[["Segment", SIDE]].isna().any().any():
        raise ValueError("Missing simple-bid side or segment")
    if not frame[SIDE].isin(["Sell", "Purchase"]).all():
        raise ValueError("Unsupported simple-bid side")
    # Sparse templates are permitted. Zero matched-template denominators are
    # handled explicitly during demand calibration.
    layout.frame = frame.reset_index(drop=True)
    return layout, day


def read_mcp(path, day: pd.Timestamp) -> pd.Series:
    """Map exactly 96 published quarter-hour prices chronologically."""
    prices = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    prices.columns = prices.columns.str.strip()
    if "Delivery day" not in prices:
        raise ValueError("Prices CSV lacks 'Delivery day'")
    rows = prices.loc[date_keys(prices["Delivery day"]).eq(day)]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one MCP row for {day.date()}, found {len(rows)}"
        )
    coordinates = {}
    for column in prices.columns:
        match = re.fullmatch(r"Hour (\d+)([AB]?) Q([1-4])", column)
        if match is None:
            continue
        raw = rows.iloc[0][column]
        if pd.isna(raw) or not str(raw).strip():
            continue
        value = pd.to_numeric(raw, errors="raise")
        finite([value], f"MCP {column}")
        hour, suffix, quarter = int(match[1]), match[2], int(match[3])
        if (
            suffix == "B"
            or not 1 <= hour <= 24
            or (suffix == "A" and hour != 3)
        ):
            raise ValueError(f"Unexpected/DST MCP coordinate {column}")
        mtu = (hour - 1) * 4 + quarter
        if mtu in coordinates:
            raise ValueError(f"Duplicate MCP coordinate at MTU {mtu}")
        coordinates[mtu] = float(value)
    if set(coordinates) != set(MTUS):
        missing = sorted(set(MTUS) - set(coordinates))
        raise ValueError(
            f"{day.date()} requires 96 valid MCPs; missing MTUs: {missing}"
        )
    return pd.Series(coordinates, name="mcp_eur_mwh").reindex(MTUS)


def normalize_block_profile_columns(layout: BlockLayout) -> list[str]:
    """Normalize ordinary-day profiles without discarding active volumes."""
    frame = layout.frame.copy()
    header = list(layout.header)

    def numeric_profile(column):
        values = frame[column].replace(r"^\s*$", np.nan, regex=True)
        values = pd.to_numeric(values, errors="raise").fillna(0.0)
        finite(values, f"Block profile {column}")
        return values

    for quarter in range(1, 5):
        ordinary = f"Volume H03 Q{quarter}"
        first = f"Volume H03A Q{quarter}"
        second = f"Volume H03B Q{quarter}"
        if second in frame:
            if numeric_profile(second).ne(0.0).any():
                raise ValueError(
                    f"{second} contains active volumes; DST is unsupported"
                )
            frame = frame.drop(columns=second)
            header.remove(second)
        if first in frame:
            first_values = numeric_profile(first)
            if ordinary in frame:
                ordinary_values = numeric_profile(ordinary)
                if (
                    first_values.ne(0.0).any()
                    and ordinary_values.ne(0.0).any()
                ):
                    raise ValueError(
                        f"Both {ordinary} and {first} have active volumes"
                    )
                frame[ordinary] = (
                    first_values
                    if first_values.ne(0.0).any()
                    else ordinary_values
                )
                frame = frame.drop(columns=first)
                header.remove(first)
            else:
                frame = frame.rename(columns={first: ordinary})
                header = [
                    ordinary if column == first else column
                    for column in header
                ]

    profiles = [
        f"Volume H{hour:02d} Q{quarter}"
        for hour in range(1, 25)
        for quarter in range(1, 5)
    ]
    detected = native_profile_columns(frame.columns)
    missing = sorted(set(profiles) - set(detected))
    unexpected = sorted(set(detected) - set(profiles))
    if missing or unexpected or len(detected) != 96:
        raise ValueError(
            f"Expected 96 ordinary profiles; missing={missing}, "
            f"unexpected={unexpected}"
        )
    frame[profiles] = pd.DataFrame(
        {column: numeric_profile(column) for column in profiles},
        index=frame.index,
    )
    profile_set = set(profiles)
    first_position = next(
        i for i, column in enumerate(header) if column in profile_set
    )
    layout.header = (
        [c for c in header[:first_position] if c not in profile_set]
        + profiles
        + [c for c in header[first_position:] if c not in profile_set]
    )
    layout.frame = frame.reindex(columns=layout.header)
    return profiles


def prepare_blocks(frame: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    """Validate native BB records and attach internal relationship metadata."""
    frame = frame.copy().reset_index(drop=True)
    required = {
        "BlockID", "Block Type", "BlockCodePRM", "Execution",
        "Limit Price", "MAR", "AAR", "Segment", "bidding_zone",
    }
    if required.difference(frame):
        raise ValueError(f"Missing block fields: {sorted(required - set(frame))}")
    if any(str(column).startswith("_") for column in frame):
        raise ValueError("Block columns beginning '_' are reserved")

    signed = frame[profiles].apply(pd.to_numeric, errors="raise")
    finite(signed, "Native block profiles")
    positive = signed.gt(0.0).any(axis=1)
    negative = signed.lt(0.0).any(axis=1)
    if (positive & negative).any() or (~(positive | negative)).any():
        raise ValueError("Block profiles must be nonempty and have one sign")

    ids = frame["BlockID"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("BlockID must be nonempty and unique")
    frame["BlockID"] = ids
    frame["BlockCodePRM"] = (
        frame["BlockCodePRM"].fillna("").astype(str).str.strip()
    )
    frame["_side"] = np.where(negative, "Purchase", "Sell")
    frame["_sign"] = np.where(negative, -1.0, 1.0)
    frame[profiles] = signed.abs()
    if frame[["Segment", "bidding_zone"]].isna().any().any():
        raise ValueError("Missing block segment or bidding_zone")
    finite(
        pd.to_numeric(frame["Limit Price"], errors="raise"), "Block prices"
    )

    execution = (
        frame["Execution"].astype("string").fillna("").str.strip().str.upper()
    )
    if not execution.isin(["Y", "N"]).all():
        invalid = sorted(execution.loc[~execution.isin(["Y", "N"])].unique())
        raise ValueError(f"Block Execution must be Y or N; found {invalid}")
    frame["_historical_acceptance_ratio"] = execution.eq("Y").astype(float)

    types = frame["Block Type"]
    if not types.isin(["C01", "C02", "C04"]).all():
        raise ValueError("Unsupported block type; define its semantics first")
    if frame.loc[types.eq("C01"), "BlockCodePRM"].ne("").any():
        raise ValueError("Unexpected BlockCodePRM on C01 block")
    if frame.loc[types.eq("C04"), "BlockCodePRM"].eq("").any():
        raise ValueError("Exclusive block has an empty group identifier")

    parents = dict(zip(
        ids.loc[types.eq("C02")],
        frame.loc[types.eq("C02"), "BlockCodePRM"],
    ))
    if any(parent not in set(ids) for parent in parents.values()):
        raise ValueError("C02 references an unknown/empty parent BlockID")
    acceptance = dict(zip(ids, frame["_historical_acceptance_ratio"]))
    inconsistent = [
        f"{child}->{parent}"
        for child, parent in parents.items()
        if acceptance[child] == 1.0 and acceptance[parent] == 0.0
    ]
    if inconsistent:
        raise ValueError(
            "Accepted linked children have rejected parents: "
            + ", ".join(inconsistent)
        )

    exclusive = types.eq("C04")
    # Each source book is copied in full to every output zone. Reused group
    # codes across distinct source zones would merge under zone_code prefixes.
    reused = (
        frame.loc[exclusive]
        .groupby("BlockCodePRM")["bidding_zone"]
        .nunique()
    )
    if reused.gt(1).any():
        raise ValueError(
            "C04 group codes reused across source zones would merge after "
            f"zonal prefixing: {reused.index[reused.gt(1)].tolist()}. "
            "Make source group identifiers globally unique."
        )
    executed = frame.loc[
        exclusive & frame["_historical_acceptance_ratio"].eq(1.0)
    ].groupby("BlockCodePRM").size()
    if executed.gt(1).any():
        raise ValueError(
            "Multiple historically accepted alternatives in C04 groups: "
            f"{executed.index[executed.gt(1)].tolist()}"
        )

    roots = []
    for block_id in ids:
        seen, root = set(), block_id
        while root in parents:
            if root in seen:
                raise ValueError(f"C02 cycle containing {block_id}")
            seen.add(root)
            root = parents[root]
        roots.append(root)
    frame["_exclusive_group"] = np.where(
        exclusive, frame["BlockCodePRM"], ""
    )
    frame["_relationship_family"] = [
        f"exclusive:{code}" if kind == "C04" else f"linked:{root}"
        for kind, code, root in zip(types, frame["BlockCodePRM"], roots)
    ]
    frame["_family"] = frame["_relationship_family"] + ":bid:" + ids
    frame["_start"] = [
        int(np.flatnonzero(row > 0)[0]) + 1
        for row in frame[profiles].to_numpy(dtype=float)
    ]
    return frame


def load_block_orders(path, day) -> tuple[BlockLayout, list[str]]:
    """Read signed blocks and validate their ST delivery dates."""
    layout = read_block_layout(path)
    dates = []
    for line in layout.st_rows:
        fields = next(csv.reader([line], delimiter=";"))
        if len(fields) < 2:
            raise ValueError("Malformed ST metadata")
        dates.append(pd.to_datetime(fields[1], dayfirst=True).normalize())
    if not dates or any(value != day for value in dates):
        raise ValueError(
            f"Block ST dates must match {day.date()}; found {dates}"
        )
    profiles = normalize_block_profile_columns(layout)
    layout.frame = prepare_blocks(layout.frame, profiles)
    return layout, profiles


def historical_accepted_block_totals(
    blocks: pd.DataFrame,
    profiles: list[str],
    side: str,
) -> pd.Series:
    """Sum block MW weighted by original historical Y/N Execution."""
    if side not in ["Sell", "Purchase"]:
        raise ValueError(f"Unsupported block side: {side}")
    selected = blocks.loc[blocks["_side"].eq(side)]
    values = selected[profiles].mul(
        selected["_historical_acceptance_ratio"], axis=0
    ).sum(axis=0)
    return pd.Series(values.to_numpy(dtype=float), index=MTUS)


def matched_demand(simple, blocks, profiles, mcp):
    """Return historical matched demand and its original simple/block parts."""
    if not mcp.index.equals(pd.Index(MTUS)):
        raise ValueError("MCP series must contain exactly MTUs 1 to 96")
    mask = simple[SIDE].eq("Purchase") & simple["Price"].ge(
        simple["MTU"].map(mcp)
    )
    matched_simple = (
        simple.loc[mask].groupby("MTU")["Volume"].sum()
        .reindex(MTUS, fill_value=0.0)
    )
    historical_blocks = historical_accepted_block_totals(
        blocks, profiles, "Purchase"
    )
    matched = matched_simple + historical_blocks
    finite(matched, "Historical matched demand", nonnegative=True)
    diagnostics = pd.DataFrame({
        "MTU": MTUS,
        "mcp_eur_mwh": mcp.to_numpy(),
        "M_simple_mw": matched_simple.to_numpy(),
        "B_historical_mw": historical_blocks.to_numpy(),
        "historical_matched_demand_mw": matched.to_numpy(),
        "matched_simple_bid_count": (
            simple.loc[mask].groupby("MTU").size()
            .reindex(MTUS, fill_value=0).to_numpy()
        ),
    })
    return matched, diagnostics


def snapshot_component_data(component, dynamic, attribute, selected, default):
    """Combine dynamic columns with static fallback; dynamic gaps are errors."""
    static = pd.to_numeric(
        component.get(
            attribute, pd.Series(default, index=component.index, dtype=float)
        ),
        errors="raise",
    )
    data = pd.DataFrame(index=selected, columns=component.index, dtype=float)
    for name in component.index:
        data[name] = (
            pd.to_numeric(dynamic[name], errors="raise").reindex(selected)
            if name in dynamic.columns
            else static.at[name]
        )
    finite(data, f"PyPSA {attribute}", nonnegative=True)
    return data


def network_zonal_data(network, zone_column, zones, day):
    """Return zonal load, installed capacity, and segment/MTU available MW.

    Timezone-aware snapshots are converted to German local delivery time.
    Naive network snapshots are interpreted as local delivery timestamps.
    Installed capacity means p_nom, not optimized p_nom_opt.
    """
    if isinstance(network.snapshots, pd.MultiIndex):
        raise ValueError("Multi-period network snapshots are unsupported")
    snapshots = pd.DatetimeIndex(network.snapshots)
    local = (
        snapshots.tz_convert(MARKET_TIMEZONE).tz_localize(None)
        if snapshots.tz is not None else snapshots
    )
    if local.has_duplicates or not local.is_monotonic_increasing:
        raise ValueError("Network snapshots must be unique and chronological")
    required = pd.date_range(day, periods=96, freq="15min")
    missing = required.difference(local)
    if not missing.empty:
        raise ValueError(f"Missing network snapshots: {list(missing)}")
    selected = snapshots[local.get_indexer(required)]

    carrier_segment = {}
    for segment, carriers in SEGMENT_TYPES.items():
        for carrier in carriers:
            if carrier in carrier_segment:
                raise ValueError(
                    f"Carrier {carrier!r} maps to multiple segments; "
                    "capacity must not be double counted"
                )
            carrier_segment[carrier] = segment

    bus_zone = network.buses[zone_column]
    generators = network.generators
    generator_zone = generators["bus"].map(bus_zone)
    generator_segment = generators["carrier"].map(carrier_segment)
    if generator_zone.isna().any():
        raise ValueError("Generator connected to unknown/unzoned bus")
    if generator_segment.isna().any():
        unknown = sorted(
            generators.loc[generator_segment.isna(), "carrier"]
            .astype(str).unique()
        )
        raise ValueError(
            f"Unmapped generator carriers: {unknown}; extend SEGMENT_TYPES"
        )
    p_nom = pd.to_numeric(generators["p_nom"], errors="raise")
    finite(p_nom, "Generator p_nom", nonnegative=True)
    p_max_pu = snapshot_component_data(
        generators, network.generators_t.p_max_pu,
        "p_max_pu", selected, 1.0,
    )
    generator_available = p_max_pu.mul(p_nom, axis=1)
    finite(generator_available, "Available generation", nonnegative=True)

    capacity, available = {}, {}
    for zone in zones:
        for segment in SEGMENT_TYPES:
            mask = generator_zone.eq(zone) & generator_segment.eq(segment)
            capacity[(zone, segment)] = float(p_nom.loc[mask].sum())
            available[(zone, segment)] = pd.Series(
                generator_available.loc[:, mask].sum(axis=1).to_numpy(),
                index=MTUS,
                dtype=float,
            )

    loads = network.loads
    load_zone = loads["bus"].map(bus_zone)
    if load_zone.isna().any():
        raise ValueError("Load connected to unknown/unzoned bus")
    data = snapshot_component_data(
        loads, network.loads_t.p_set, "p_set", selected, 0.0
    )
    zonal_load = pd.DataFrame(0.0, index=MTUS, columns=zones)
    for zone in zones:
        zonal_load[zone] = (
            data.loc[:, load_zone.eq(zone)].sum(axis=1).to_numpy()
        )
    return zonal_load, capacity, available


def simple_totals(simple, side, segment=None) -> pd.Series:
    """Return submitted simple MW by MTU."""
    mask = simple[SIDE].eq(side)
    if segment is not None:
        mask &= simple["Segment"].eq(segment)
    return (
        simple.loc[mask].groupby("MTU")["Volume"].sum()
        .reindex(MTUS, fill_value=0.0)
    )

def block_totals(blocks, profiles, side, *, effective_only=True):
    """Return gross block MW, or a conservative exclusivity envelope."""
    selected = blocks.loc[blocks["_side"].eq(side)]

    if not effective_only:
        values = selected[profiles].sum(axis=0).to_numpy(dtype=float)
    else:
        exclusive = selected["Block Type"].eq("C04")
        values = (
            selected.loc[~exclusive, profiles]
            .sum(axis=0)
            .to_numpy(dtype=float)
        )

        for _, group in selected.loc[exclusive].groupby("_exclusive_group"):
            group_maximum = group[profiles].max(axis=0).to_numpy(dtype=float)
            values = values + group_maximum

    return pd.Series(values, index=MTUS)


def supply_block_budget(blocks, profiles) -> pd.DataFrame:
    """Budget every sell alternative using conservative segment/MTU maxima.

    C02 blocks are included as non-exclusive contributions. Counting dependent
    children separately is conservative even when their parents are exclusive.
    """
    budget = pd.DataFrame(0.0, index=MTUS, columns=list(SEGMENT_TYPES))
    sell = blocks.loc[blocks["_side"].eq("Sell")]
    exclusive = sell["Block Type"].eq("C04")
    for segment in SEGMENT_TYPES:
        budget[segment] = (
            sell.loc[~exclusive & sell["Segment"].eq(segment), profiles]
            .sum(axis=0).to_numpy(dtype=float)
        )
    for _, group in sell.loc[exclusive].groupby("_exclusive_group"):
        for segment, alternatives in group.groupby("Segment"):
            budget[segment] += (
                alternatives[profiles].max(axis=0).to_numpy(dtype=float)
            )
    return budget


def demand_scaling(load, matched_simple, historical_blocks, zone):
    """Calculate one block multiplier and one simple multiplier per MTU."""
    for values, label in [
        (load, "load"),
        (matched_simple, "matched simple demand"),
        (historical_blocks, "historical block demand"),
    ]:
        finite(values, f"{zone}: {label}", nonnegative=True)

    denominator = float(
        (matched_simple + historical_blocks).sum() * QUARTER_HOURS
    )
    load_energy = float(load.sum() * QUARTER_HOURS)
    if denominator == 0.0:
        if load_energy > 0.0:
            raise ValueError(
                f"Zone {zone}: positive load energy {load_energy:g} MWh "
                "has no historical matched-demand template"
            )
        initial = 0.0
    else:
        initial = load_energy / denominator

    active_blocks = historical_blocks.gt(0.0)
    block_limit = (
        float((load.loc[active_blocks] / historical_blocks.loc[active_blocks]).min())
        if active_blocks.any() else np.inf
    )
    k_block = min(initial, block_limit)
    accepted_blocks = k_block * historical_blocks
    require_within(
        accepted_blocks, load, f"{zone}: accepted scaled purchase blocks"
    )
    residual = load - accepted_blocks
    if residual.lt(-TOL - RTOL * load.abs()).any():
        raise ValueError(f"Zone {zone}: negative residual demand")
    residual = residual.clip(lower=0.0)

    no_template = matched_simple.eq(0.0)
    numerical_zero = residual.le(TOL + RTOL * load.abs())
    impossible = no_template & ~numerical_zero
    if impossible.any():
        details = ", ".join(
            f"MTU {mtu}: {residual.at[mtu]:g} MW"
            for mtu in residual.index[impossible]
        )
        raise ValueError(
            f"Zone {zone}: positive residual demand has no matched simple "
            f"purchase template at historical MCPs ({details})"
        )
    # A zero denominator with zero residual uses c=0. Tolerance is used only
    # for floating-point equality at a block-anchored MTU.
    residual.loc[no_template] = 0.0
    c_simple = pd.Series(0.0, index=MTUS)
    usable = ~no_template
    c_simple.loc[usable] = (
        residual.loc[usable] / matched_simple.loc[usable]
    )
    finite(c_simple, f"{zone}: simple demand factors", nonnegative=True)
    require_close(
        c_simple * matched_simple + accepted_blocks,
        load,
        f"{zone}: analytical demand calibration",
    )
    daily = {
        "zone": zone,
        "pypsa_load_energy_mwh": load_energy,
        "historical_matched_demand_energy_mwh": denominator,
        "initial_purchase_block_factor": initial,
        "purchase_block_factor": k_block,
        "purchase_block_factor_reduced": k_block < initial,
        "zero_matched_simple_template_mtu_count": int(no_template.sum()),
    }
    return k_block, c_simple, daily


def supply_transformation(simple, blocks, profiles, available, capacity, zones):
    """Compute daily available-generation/template-energy supply factors.

    A segment without an offered template receives factor zero; its capacity
    remains unused headroom. No synthetic supply is invented.
    """
    supplied = (
        set(simple.loc[simple[SIDE].eq("Sell"), "Segment"])
        | set(blocks.loc[blocks["_side"].eq("Sell"), "Segment"])
    )
    unknown = supplied - set(SEGMENT_TYPES)
    if unknown:
        raise ValueError(f"Unmapped supply segments: {sorted(unknown)}")

    block_budget = supply_block_budget(blocks, profiles)
    factors, rows = {}, []
    for segment in SEGMENT_TYPES:
        simple_energy = float(
            simple_totals(simple, "Sell", segment).sum() * QUARTER_HOURS
        )
        block_energy = float(block_budget[segment].sum() * QUARTER_HOURS)
        template_energy = simple_energy + block_energy
        for zone in zones:
            available_energy = float(
                available[(zone, segment)].sum() * QUARTER_HOURS
            )
            factor = (
                available_energy / template_energy
                if template_energy > 0.0 else 0.0
            )
            if template_energy == 0.0 and available_energy > 0.0:
                LOG.warning(
                    "No supply template: zone=%s segment=%s "
                    "available_energy=%g MWh; capacity remains headroom",
                    zone, segment, available_energy,
                )
            finite([factor], "Daily supply factor", nonnegative=True)
            factors[(zone, segment)] = factor
            rows.append({
                "zone": zone,
                "segment": segment,
                "installed_capacity_mw": capacity[(zone, segment)],
                "available_generation_energy_mwh": available_energy,
                "original_simple_energy_mwh": simple_energy,
                "original_block_budget_energy_mwh": block_energy,
                "original_offered_template_energy_mwh": template_energy,
                "daily_supply_factor": factor,
                "available_capacity_without_template": (
                    template_energy == 0.0 and available_energy > 0.0
                ),
                "unbacked_scarcity_omitted": (
                    segment == SCARCITY_SEGMENT and available_energy == 0.0
                ),
                "block_budget_basis": (
                    "nonexclusive_sum_plus_segment_per_mtu_exclusive_maxima"
                ),
                "dt_hours": QUARTER_HOURS,
            })
    return factors, pd.DataFrame(rows)


def prefixed_value(zone, value) -> str:
    """Preserve the existing zone_identifier prefix convention."""
    return (
        "" if pd.isna(value) or not str(value).strip()
        else f"{zone}_{str(value).strip()}"
    )


def scale_blocks(blocks, profiles, zone, demand_multiplier, factors):
    """Apply daily supply factors or the common daily purchase-block factor."""
    if not np.isscalar(demand_multiplier):
        raise ValueError("Purchase blocks require one scalar zone/day factor")
    zonal = blocks.copy()
    zonal["_initial"] = [
        factors[(zone, segment)] if side == "Sell" else demand_multiplier
        for side, segment in zip(zonal["_side"], zonal["Segment"])
    ]
    zonal["_downscale"] = 1.0
    zonal[profiles] = zonal[profiles].mul(zonal["_initial"], axis=0)
    zonal["bidding_zone"] = zone
    for column in ["BlockID", "BlockCodePRM"]:
        zonal[column] = zonal[column].map(
            lambda value: prefixed_value(zone, value)
        )
    finite(zonal[profiles], "Initial zonal block magnitudes", nonnegative=True)
    return zonal


def limit_supply_blocks(blocks, profiles, available, zone, event_rows):
    """Downscale complete sell profiles only for physical availability.

    Limits are computed from every retained sell alternative. A single common
    reduction is applied to all sell alternatives of each C04 group, including
    alternatives in different segments. Non-exclusive bids remain individual
    whole-profile scaling units.
    """
    blocks = blocks.copy()
    initial_budget = supply_block_budget(blocks, profiles)
    limits = pd.DataFrame(1.0, index=MTUS, columns=list(SEGMENT_TYPES))
    for segment in SEGMENT_TYPES:
        active = initial_budget[segment].gt(0.0)
        limits.loc[active, segment] = np.minimum(
            1.0,
            available[(zone, segment)].loc[active]
            / initial_budget.loc[active, segment],
        )

    sell = blocks.loc[blocks["_side"].eq("Sell")]
    units = [
        ([index], f"bid:{row['BlockID']}")
        for index, row in sell.loc[
            ~sell["Block Type"].eq("C04")
        ].iterrows()
    ]
    units += [
        (group.index.tolist(), f"exclusive:{code}")
        for code, group in sell.loc[
            sell["Block Type"].eq("C04")
        ].groupby("_exclusive_group")
    ]

    for indices, unit in units:
        reduction = 1.0
        for index in indices:
            segment = blocks.at[index, "Segment"]
            active = blocks.loc[index, profiles].to_numpy(dtype=float) > 0.0
            if active.any():
                reduction = min(
                    reduction,
                    float(limits[segment].to_numpy()[active].min()),
                )
        blocks.loc[indices, "_downscale"] = reduction
        blocks.loc[indices, profiles] = (
            blocks.loc[indices, profiles] * reduction
        )
        if reduction < 1.0:
            event_rows.append({
                "zone": zone,
                "scaling_unit": unit,
                "block_ids": "|".join(blocks.loc[indices, "BlockID"]),
                "segments": "|".join(
                    sorted(blocks.loc[indices, "Segment"].unique())
                ),
                "physical_reduction_factor": reduction,
                "reason": "available_generation_limit",
            })

    final_budget = supply_block_budget(blocks, profiles)
    for segment in SEGMENT_TYPES:
        require_within(
            final_budget[segment], available[(zone, segment)],
            f"{zone}/{segment}: sell-block budget",
        )
    return blocks


def cap_simple_supply(simple, blocks, profiles, available, zone):
    """Only reduce simple supply when it exceeds remaining segment MW."""
    simple = simple.copy()
    budget = supply_block_budget(blocks, profiles)
    rows = []
    for segment in SEGMENT_TYPES:
        mask = simple[SIDE].eq("Sell") & simple["Segment"].eq(segment)
        initial = simple_totals(simple, "Sell", segment)
        remaining = (
            available[(zone, segment)] - budget[segment]
        ).clip(lower=0.0)
        cap = pd.Series(1.0, index=MTUS)
        active = initial.gt(0.0)
        cap.loc[active] = np.minimum(
            1.0, remaining.loc[active] / initial.loc[active]
        )
        simple.loc[mask, "Volume"] *= simple.loc[mask, "MTU"].map(cap)
        final = simple_totals(simple, "Sell", segment)
        total = final + budget[segment]
        require_within(
            total, available[(zone, segment)],
            f"{zone}/{segment}: submitted supply budget",
        )
        gross_blocks = pd.Series(
            blocks.loc[
                blocks["_side"].eq("Sell")
                & blocks["Segment"].eq(segment), profiles
            ].sum(axis=0).to_numpy(dtype=float),
            index=MTUS,
        )
        for mtu in MTUS:
            rows.append({
                "zone": zone,
                "segment": segment,
                "MTU": mtu,
                "available_generation_mw": available[(zone, segment)].at[mtu],
                "sell_block_budget_mw": budget.at[mtu, segment],
                "remaining_availability_before_simple_mw": remaining.at[mtu],
                "simple_supply_before_cap_mw": initial.at[mtu],
                "simple_supply_cap_factor": cap.at[mtu],
                "final_simple_supply_mw": final.at[mtu],
                "final_supply_budget_mw": total.at[mtu],
                "physical_headroom_mw": max(
                    0.0, available[(zone, segment)].at[mtu] - total.at[mtu]
                ),
                "gross_submitted_sell_block_mw": gross_blocks.at[mtu],
                "gross_submitted_supply_mw": (
                    final.at[mtu] + gross_blocks.at[mtu]
                ),
            })
    finite(simple["Volume"], "Final simple magnitudes", nonnegative=True)
    return simple, rows


def same_metadata_value(left, right) -> bool:
    """Treat two missing scalar metadata values as unchanged."""
    return bool(
        (pd.isna(left) and pd.isna(right))
        or (
            not pd.isna(left)
            and not pd.isna(right)
            and left == right
        )
    )


def validate_blocks(source, zonal, profiles, zone) -> list[dict]:
    """Validate preserved metadata, relationships, signs, and whole profiles."""
    if len(source) != len(zonal):
        raise ValueError("Blocks were removed before preservation validation")
    metadata = [
        column for column in source
        if column not in profiles
        and not column.startswith("_")
        and column != "bidding_zone"
    ]
    rows = []
    for position in range(len(source)):
        before, after = source.iloc[position], zonal.iloc[position]
        for column in metadata:
            expected = (
                prefixed_value(zone, before[column])
                if column in ["BlockID", "BlockCodePRM"]
                else before[column]
            )
            if not same_metadata_value(after[column], expected):
                raise ValueError(
                    f"Changed block metadata: {before['BlockID']}/{column}"
                )
        if after["bidding_zone"] != zone or after["_sign"] != before["_sign"]:
            raise ValueError("Changed block zone or native sign")
        original = before[profiles].to_numpy(dtype=float)
        transformed = after[profiles].to_numpy(dtype=float)
        factor = float(after["_initial"] * after["_downscale"])
        finite([factor], "Block multiplier", nonnegative=True)
        finite(transformed, "Final block magnitudes", nonnegative=True)
        require_close(
            transformed, original * factor,
            f"{after['BlockID']}: constant whole-profile multiplier",
        )
        if (transformed[original == 0.0] != 0.0).any():
            raise ValueError("Block scaling introduced an active profile period")
        rows.append({
            "zone": zone,
            "original_block_id": before["BlockID"],
            "block_id": after["BlockID"],
            "block_family_id": f"{zone}:{before['_family']}",
            "relationship_family_id": before["_relationship_family"],
            "block_type": before["Block Type"],
            "block_code_prm": after["BlockCodePRM"],
            "side": before["_side"],
            "segment": before["Segment"],
            "block_starting_mtu": before["_start"],
            "initial_multiplier": after["_initial"],
            "physical_reduction_factor": after["_downscale"],
            "final_total_multiplier": factor,
            "original_energy_mwh": original.sum() * QUARTER_HOURS,
            "final_energy_mwh": transformed.sum() * QUARTER_HOURS,
            "historical_execution": before["Execution"],
            "constant_profile_multiplier_valid": True,
            "metadata_and_relationships_valid": True,
            "native_sign_preserved": True,
        })
    validate_relationships(zonal)
    return rows


def validate_relationships(blocks):
    """Check exported ID uniqueness, C02 parent presence, and C04 group codes."""
    if blocks["BlockID"].duplicated().any():
        raise ValueError("Duplicate zonal BlockID")
    linked = blocks["Block Type"].eq("C02")
    missing = linked & ~blocks["BlockCodePRM"].isin(set(blocks["BlockID"]))
    if missing.any():
        raise ValueError(
            "Retained C02 blocks reference missing parents: "
            + ", ".join(blocks.loc[missing, "BlockID"])
        )
    if blocks.loc[
        blocks["Block Type"].eq("C04"), "BlockCodePRM"
    ].eq("").any():
        raise ValueError("Retained C04 block has an empty group code")


def filter_exact_zeros(simple, blocks, profiles):
    """Remove exact zeros only; reject filtering that breaks a retained link."""
    zero_simple = simple["Volume"].eq(0.0)
    zero_blocks = blocks[profiles].eq(0.0).all(axis=1)
    removed_ids = set(blocks.loc[zero_blocks, "BlockID"])
    dangling = (
        ~zero_blocks
        & blocks["Block Type"].eq("C02")
        & blocks["BlockCodePRM"].isin(removed_ids)
    )
    if dangling.any():
        links = [
            f"{row['BlockID']}->{row['BlockCodePRM']}"
            for _, row in blocks.loc[dangling].iterrows()
        ]
        raise ValueError(
            "Cannot filter zero-volume parents of retained C02 children: "
            + ", ".join(links)
        )
    retained_simple = simple.loc[~zero_simple].copy().reset_index(drop=True)
    retained_blocks = blocks.loc[~zero_blocks].copy().reset_index(drop=True)
    validate_relationships(retained_blocks)
    LOG.info(
        "Removed %d exact-zero simple orders and %d all-zero blocks",
        int(zero_simple.sum()), int(zero_blocks.sum()),
    )
    return retained_simple, retained_blocks


def save_table(frame, path) -> None:
    """Write a diagnostic table and create its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def create_zonal_orderbooks(
    simple_input: str | Path,
    block_input: str | Path,
    network: pypsa.Network,
    zone_column: str,
    simple_output: str | Path,
    block_output: str | Path,
    supply_report_output: str | Path | None = None,
    demand_report_output: str | Path | None = None,
    *,
    original_curves_csv: str | Path | None = None,
    prices_csv: str | Path | None = None,
    delivery_date=None,
    diagnostics_dir: str | Path | None = None,
    scarcity_share: float | None = SCARCITY_SHARE,
    report_prefix: str | Path | None = None,
    network_input: str | Path | None = None,
) -> None:
    """Create network-led books, physical/demand diagnostics, and a manifest.

    The original callable and positional report outputs remain supported.
    original_curves_csv is deprecated and ignored. scarcity_share is obsolete:
    zero is accepted and ignored for compatibility; positive shares are
    rejected because a share does not allocate real generator capacity.
    """
    if original_curves_csv is not None:
        warnings.warn(
            "original_curves_csv is deprecated and ignored; demand uses "
            "preprocessed orders and historical MCPs only.",
            FutureWarning, stacklevel=2,
        )
    if scarcity_share is not None:
        warnings.warn(
            "scarcity_share is obsolete; explicitly reassign real generator "
            "carriers in SEGMENT_TYPES to back scarcity.",
            FutureWarning, stacklevel=2,
        )
        if not np.isfinite(scarcity_share) or scarcity_share != 0.0:
            raise ValueError(
                "Nonzero scarcity_share is unsupported: allocate real "
                "capacity through a nonduplicated SEGMENT_TYPES mapping"
            )
    price_path = prices_csv if prices_csv is not None else PRICES_CSV
    if price_path is None:
        raise ValueError("Configure prices_csv for historical demand calibration")

    simple_layout, day = load_simple_orders(simple_input, delivery_date)
    simple = simple_layout.frame
    simple_columns = [column for column in simple if column != "MTU"]
    block_layout, profiles = load_block_orders(block_input, day)
    blocks = block_layout.frame
    zones = normalize_network_zones(network, zone_column)
    load, capacity, available = network_zonal_data(
        network, zone_column, zones, day
    )
    mcp = read_mcp(price_path, day)
    matched, anchor = matched_demand(simple, blocks, profiles, mcp)
    matched_simple = anchor.set_index("MTU")["M_simple_mw"]
    historical_blocks = anchor.set_index("MTU")["B_historical_mw"]
    factors, supply_daily = supply_transformation(
        simple, blocks, profiles, available, capacity, zones
    )

    diagnostic_path = (
        Path(diagnostics_dir)
        if diagnostics_dir is not None else Path(simple_output).parent
    )
    simple_frames, block_frames = [], []
    demand_rows, demand_daily, supply_rows = [], [], []
    scaling_rows, event_rows = [], []
    regression_rows = []

    for zone in zones:
        k_block, c_simple, daily = demand_scaling(
            load[zone], matched_simple, historical_blocks, zone
        )
        demand_daily.append(daily)
        zonal_blocks = scale_blocks(
            blocks, profiles, zone, k_block, factors
        )
        zonal_blocks = limit_supply_blocks(
            zonal_blocks, profiles, available, zone, event_rows
        )
        zonal_simple = simple.copy()
        zonal_simple["bidding_zone"] = zone
        buy = zonal_simple[SIDE].eq("Purchase")
        sell = zonal_simple[SIDE].eq("Sell")
        zonal_simple.loc[buy, "Volume"] *= (
            zonal_simple.loc[buy, "MTU"].map(c_simple)
        )
        zonal_simple.loc[sell, "Volume"] *= (
            zonal_simple.loc[sell, "Segment"]
            .map(lambda segment: factors[(zone, segment)])
        )
        finite(
            zonal_simple["Volume"],
            f"{zone}: initially scaled simple magnitudes",
            nonnegative=True,
        )
        zonal_simple, zone_supply = cap_simple_supply(
            zonal_simple, zonal_blocks, profiles, available, zone
        )
        supply_rows.extend(zone_supply)

        calibrated, _ = matched_demand(
            zonal_simple, zonal_blocks, profiles, mcp
        )
        require_close(
            calibrated, load[zone], f"{zone}: benchmark demand equals PyPSA load"
        )
        unchanged = [
            column for column in simple_columns
            if column not in ["Volume", "bidding_zone"]
        ]
        if not zonal_simple[unchanged].equals(simple[unchanged]):
            raise ValueError(f"{zone}: simple metadata changed")
        scaling_rows.extend(
            validate_blocks(blocks, zonal_blocks, profiles, zone)
        )

        simple_demand = simple_totals(zonal_simple, "Purchase")
        gross_block_demand = block_totals(
            zonal_blocks, profiles, "Purchase"
        )
        accepted_block_demand = historical_accepted_block_totals(
            zonal_blocks, profiles, "Purchase"
        )
        matched_simple_scaled = calibrated - accepted_block_demand
        for mtu in MTUS:
            demand_rows.append({
                "zone": zone,
                "MTU": mtu,
                "pypsa_load_mw": load.at[mtu, zone],
                "original_M_simple_mw": matched_simple.at[mtu],
                "original_B_historical_mw": historical_blocks.at[mtu],
                "original_matched_demand_mw": matched.at[mtu],
                "purchase_block_factor": k_block,
                "simple_purchase_factor": c_simple.at[mtu],
                "benchmark_matched_simple_mw": matched_simple_scaled.at[mtu],
                "benchmark_accepted_purchase_block_mw": (
                    accepted_block_demand.at[mtu]
                ),
                "benchmark_matched_demand_mw": calibrated.at[mtu],
                "demand_anchor_error_mw": (
                    calibrated.at[mtu] - load.at[mtu, zone]
                ),
                "gross_submitted_simple_purchase_mw": simple_demand.at[mtu],
                "gross_submitted_purchase_block_mw": (
                    gross_block_demand.at[mtu]
                ),
                "gross_submitted_demand_mw": (
                    simple_demand.at[mtu] + gross_block_demand.at[mtu]
                ),
                "new_clearing_acceptance_guaranteed": False,
            })
        regression_rows.append({
            "zone": zone,
            "demand_calibration_valid": True,
            "segment_availability_limits_valid": True,
            "nonnegative_magnitudes_valid": True,
            "simple_metadata_valid": True,
            "block_metadata_relationships_profiles_valid": True,
            "max_absolute_demand_anchor_error_mw": float(
                (calibrated - load[zone]).abs().max()
            ),
        })
        simple_frames.append(zonal_simple)
        block_frames.append(zonal_blocks)

    zonal_simple = pd.concat(simple_frames, ignore_index=True)
    zonal_blocks = pd.concat(block_frames, ignore_index=True)
    zonal_simple, zonal_blocks = filter_exact_zeros(
        zonal_simple, zonal_blocks, profiles
    )

    demand_report = pd.DataFrame(demand_rows)
    supply_report = pd.DataFrame(supply_rows)
    diagnostics = {
        "matched_demand_diagnostics.csv": anchor,
        "demand_daily_factors.csv": pd.DataFrame(demand_daily),
        "demand_scaling_diagnostics.csv": demand_report,
        "supply_daily_factors.csv": supply_daily,
        "supply_availability_diagnostics.csv": supply_report,
        "block_scaling_diagnostics.csv": pd.DataFrame(scaling_rows),
        "block_downscaling_events.csv": pd.DataFrame(
            event_rows,
            columns=[
                "zone", "scaling_unit", "block_ids", "segments",
                "physical_reduction_factor", "reason",
            ],
        ),
        "regression_checks.csv": pd.DataFrame(regression_rows),
    }
    for filename, table in diagnostics.items():
        save_table(table, diagnostic_path / filename)
    if supply_report_output is not None:
        save_table(supply_report, supply_report_output)
    if demand_report_output is not None:
        save_table(demand_report, demand_report_output)

    manifest = pd.DataFrame([{
        "delivery_date": str(day.date()),
        "simple_input": str(simple_input),
        "block_input": str(block_input),
        "prices_csv": str(price_path),
        "network_input": str(network_input or "in-memory"),
        "zone_column": zone_column,
        "simple_output": str(simple_output),
        "block_output": str(block_output),
        "diagnostics_dir": str(diagnostic_path),
        "transformation": "network_led_demand_and_independent_available_supply",
        "demand_anchor": "simple_price_ge_mcp_plus_purchase_execution_y",
        "purchase_block_scaling": "one_zone_day_factor_with_load_ceiling",
        "simple_purchase_scaling": "per_mtu_matched_template_residual",
        "supply_scaling": "available_generation_energy_over_template_energy",
        "installed_capacity_attribute": "p_nom",
        "availability_attribute": "snapshot_matched_p_max_pu_static_fallback",
        "supply_block_budget": (
            "nonexclusive_sum_plus_segment_per_mtu_exclusive_maxima"
        ),
        "exclusive_physical_reduction": "common_across_sell_alternatives",
        "unbacked_scarcity": "omitted",
        "historical_ratio_reconciliation": False,
        "local_balance_enforced": False,
        "new_clearing_acceptance_guaranteed": False,
        "native_block_signs_preserved": True,
        "safe_exact_zero_filtering_valid": True,
        "dt_hours": QUARTER_HOURS,
        "daily_energy_unit": "MWh",
    }])
    save_table(manifest, diagnostic_path / "manifest.csv")
    if report_prefix is not None:
        save_table(manifest, f"{report_prefix}_manifest.csv")

    simple_layout.frame = zonal_simple[simple_columns].sort_values(
        ["bidding_zone", "Date", "Hour", "Quarter hour", SIDE, "Price"],
        kind="stable",
    )
    Path(simple_output).parent.mkdir(parents=True, exist_ok=True)
    Path(block_output).parent.mkdir(parents=True, exist_ok=True)
    write_simple_layout(simple_layout, simple_output)

    # Restore native signed profiles only at export.
    zonal_blocks[profiles] = zonal_blocks[profiles].mul(
        zonal_blocks["_sign"], axis=0
    )
    finite(zonal_blocks[profiles], "Exported signed block profiles")
    block_layout.frame = zonal_blocks.reindex(columns=block_layout.header)
    write_block_layout(block_layout, block_output)
    LOG.info(
        "Wrote %d simple orders to %s", len(simple_layout.frame), simple_output
    )
    LOG.info(
        "Wrote %d block orders to %s", len(block_layout.frame), block_output
    )
    LOG.info("Diagnostics written to %s", diagnostic_path)


def run_regression_checks() -> None:
    """Run synthetic demand, network, exclusivity, and preservation checks."""
    profiles = [
        f"Volume H{hour:02d} Q{quarter}"
        for hour in range(1, 25)
        for quarter in range(1, 5)
    ]

    def block(block_id, kind, code, segment, values, execution="N"):
        row = {
            "BlockID": block_id,
            "Block Type": kind,
            "BlockCodePRM": code,
            "Execution": execution,
            "Limit Price": 123.0,
            "MAR": 1.0,
            "AAR": 1.0,
            "Segment": segment,
            "bidding_zone": "SOURCE",
            "Comment": "preserved",
        }
        row.update(dict(zip(profiles, values)))
        return row

    def expect_error(function, text):
        try:
            function()
        except ValueError as error:
            if text not in str(error):
                raise AssertionError(f"Unexpected error: {error}") from error
        else:
            raise AssertionError(f"Expected ValueError containing {text!r}")

    # Original historical demand includes Y purchases only and Price >= MCP.
    purchase_profile = np.full(96, 2.0)
    purchase_profile[48:] = 4.0
    purchases = prepare_blocks(pd.DataFrame([
        block("P1", "C01", "", "DEMAND", -purchase_profile, "Y"),
        block("P2", "C01", "", "DEMAND", -np.full(96, 7.0), "N"),
    ]), profiles)
    simple = pd.DataFrame({
        "MTU": MTUS + MTUS,
        SIDE: ["Purchase"] * 192,
        "Price": [50.0] * 96 + [49.0] * 96,
        "Volume": [10.0] * 96 + [3.0] * 96,
        "Segment": ["DEMAND"] * 192,
        "bidding_zone": ["SOURCE"] * 192,
    })
    mcp = pd.Series(50.0, index=MTUS)
    _, anchor = matched_demand(simple, purchases, profiles, mcp)
    require_close(anchor["M_simple_mw"], 10.0, "Historical simple matching")
    require_close(
        anchor["B_historical_mw"], purchase_profile, "Historical Y/N matching"
    )
    load = pd.Series(12.0, index=MTUS)
    load.at[1] = 1.0
    k, c, daily = demand_scaling(
        load, pd.Series(10.0, index=MTUS),
        pd.Series(purchase_profile, index=MTUS), "ZTEST",
    )
    require_close(k, 0.5, "Common purchase-block load ceiling")
    if not daily["purchase_block_factor_reduced"]:
        raise AssertionError("Expected daily purchase-block factor reduction")
    scaled_purchases = scale_blocks(purchases, profiles, "ZTEST", k, {})
    scaled_simple = simple.copy()
    scaled_simple["Volume"] *= scaled_simple["MTU"].map(c)
    calibrated, _ = matched_demand(
        scaled_simple, scaled_purchases, profiles, mcp
    )
    require_close(calibrated, load, "Demand regression calibration")
    validate_blocks(purchases, scaled_purchases, profiles, "ZTEST")

    zeros = pd.Series(0.0, index=MTUS)
    ones = pd.Series(1.0, index=MTUS)
    expect_error(
        lambda: demand_scaling(ones, zeros, zeros, "ZTEST"),
        "no historical matched-demand template",
    )
    sparse_simple = ones.copy()
    sparse_simple.at[1] = 0.0
    expect_error(
        lambda: demand_scaling(ones, sparse_simple, zeros, "ZTEST"),
        "no matched simple",
    )
    zero_k, zero_c, _ = demand_scaling(zeros, zeros, zeros, "ZTEST")
    require_close(zero_k, 0.0, "Zero-energy demand factor")
    require_close(zero_c, zeros, "Zero-template demand factors")
    block_only_k, block_only_c, _ = demand_scaling(
        2.0 * ones, zeros, ones, "ZTEST"
    )
    require_close(block_only_k, 2.0, "Block-only demand factor")
    require_close(block_only_c, zeros, "Block-only simple factor")

    # Snapshot-matched dynamic availability and independent static fallback.
    day = pd.Timestamp("2025-10-01")
    snapshots = pd.date_range(day, periods=96, freq="15min")
    wind_pu = np.full(96, 0.4)
    wind_pu[0] = 0.5
    network = SimpleNamespace(
        snapshots=snapshots,
        buses=pd.DataFrame({"zone": ["ZTEST"]}, index=["BUS"]),
        generators=pd.DataFrame({
            "bus": ["BUS", "BUS"],
            "carrier": ["Wind", "Gas/CCGT"],
            "p_nom": [10.0, 20.0],
            "p_max_pu": [1.0, 0.25],
        }, index=["WIND", "GAS"]),
        generators_t=SimpleNamespace(
            p_max_pu=pd.DataFrame({"WIND": wind_pu}, index=snapshots)
        ),
        loads=pd.DataFrame({
            "bus": ["BUS", "BUS"], "p_set": [99.0, 4.0],
        }, index=["DYNAMIC_LOAD", "STATIC_LOAD"]),
        loads_t=SimpleNamespace(
            p_set=pd.DataFrame(
                {"DYNAMIC_LOAD": np.full(96, 7.0)}, index=snapshots
            )
        ),
    )
    zonal_load, capacity, available = network_zonal_data(
        network, "zone", ["ZTEST"], day
    )
    require_close(zonal_load["ZTEST"], 11.0, "Dynamic/static load fallback")
    require_close(
        available[("ZTEST", "S0_NEGATIVE")], wind_pu * 10.0,
        "Dynamic generator availability",
    )
    require_close(
        available[("ZTEST", "S4_HIGH_COST")], 5.0,
        "Static generator availability",
    )
    require_close(
        available[("ZTEST", SCARCITY_SEGMENT)], 0.0,
        "No unbacked scarcity capacity",
    )
    network.generators_t.p_max_pu.loc[snapshots[0], "WIND"] = np.nan
    expect_error(
        lambda: network_zonal_data(network, "zone", ["ZTEST"], day),
        "non-finite",
    )

    # A representative would miss the other alternative's MTU and segment.
    a, b = np.zeros(96), np.zeros(96)
    a[0], b[1] = 10.0, 6.0
    supply_blocks = prepare_blocks(pd.DataFrame([
        block("A", "C04", "G", "S0_NEGATIVE", a),
        block("B", "C04", "G", "S4_HIGH_COST", b),
        block("ROOT", "C01", "", "S0_NEGATIVE", np.ones(96)),
        block("CHILD", "C02", "ROOT", "S4_HIGH_COST", np.ones(96)),
    ]), profiles)
    original_budget = supply_block_budget(supply_blocks, profiles)
    require_close(
        original_budget.at[1, "S0_NEGATIVE"], 11.0,
        "First exclusive alternative budget",
    )
    require_close(
        original_budget.at[2, "S4_HIGH_COST"], 7.0,
        "Other exclusive alternative budget",
    )

    physical = {
        ("ZTEST", segment): pd.Series(10.0, index=MTUS)
        for segment in SEGMENT_TYPES
    }
    physical[("ZTEST", "S0_NEGATIVE")].at[1] = 5.0
    physical[("ZTEST", "S4_HIGH_COST")].at[2] = 3.0
    physical[("ZTEST", SCARCITY_SEGMENT)] = zeros.copy()
    initial_factors = {
        ("ZTEST", segment): 1.0 for segment in SEGMENT_TYPES
    }
    scaled_supply = scale_blocks(
        supply_blocks, profiles, "ZTEST", 1.0, initial_factors
    )
    events = []
    scaled_supply = limit_supply_blocks(
        scaled_supply, profiles, physical, "ZTEST", events
    )
    alternatives = scaled_supply.loc[
        scaled_supply["Block Type"].eq("C04"), "_downscale"
    ]
    require_close(alternatives, 3.0 / 7.0, "Common alternative reduction")
    validate_blocks(supply_blocks, scaled_supply, profiles, "ZTEST")

    simple_supply = pd.DataFrame({
        "MTU": MTUS + MTUS,
        SIDE: ["Sell"] * 192,
        "Price": [25.0] * 192,
        "Volume": [20.0] * 192,
        "Segment": ["S0_NEGATIVE"] * 96 + ["S4_HIGH_COST"] * 96,
        "bidding_zone": ["ZTEST"] * 192,
    })
    capped, _ = cap_simple_supply(
        simple_supply, scaled_supply, profiles, physical, "ZTEST"
    )
    if (capped["Volume"] > simple_supply["Volume"]).any():
        raise AssertionError("Simple supply was increased to fill headroom")
    budget = supply_block_budget(scaled_supply, profiles)
    for segment in SEGMENT_TYPES:
        require_within(
            simple_totals(capped, "Sell", segment) + budget[segment],
            physical[("ZTEST", segment)],
            f"Regression physical limit/{segment}",
        )

    # Verify dt and the conservative template denominator in daily factors.
    daily_factors, daily_report = supply_transformation(
        simple_supply, supply_blocks, profiles, physical,
        {("ZTEST", segment): 10.0 for segment in SEGMENT_TYPES},
        ["ZTEST"],
    )
    expected_template = (
        simple_totals(simple_supply, "Sell", "S0_NEGATIVE").sum()
        + original_budget["S0_NEGATIVE"].sum()
    ) * QUARTER_HOURS
    expected_available = (
        physical[("ZTEST", "S0_NEGATIVE")].sum() * QUARTER_HOURS
    )
    require_close(
        daily_factors[("ZTEST", "S0_NEGATIVE")],
        expected_available / expected_template,
        "Daily available-generation/template-energy factor",
    )
    scarcity = daily_report.loc[
        daily_report["segment"].eq(SCARCITY_SEGMENT)
    ].iloc[0]
    require_close(scarcity["daily_supply_factor"], 0.0, "Scarcity omitted")

    # Exact-zero filtering must retain arbitrarily small positive bids and
    # every positive exclusive alternative, and reject dangling linked bids.
    tiny_simple = capped.iloc[:1].copy()
    tiny_simple["Volume"] = 1e-12
    retained_simple, retained_blocks = filter_exact_zeros(
        tiny_simple, scaled_supply, profiles
    )
    if len(retained_simple) != 1 or len(retained_blocks) != 4:
        raise AssertionError("Exact-zero filtering removed a positive bid")
    broken = scaled_supply.copy()
    broken.loc[broken["BlockID"].eq("ZTEST_ROOT"), profiles] = 0.0
    expect_error(
        lambda: filter_exact_zeros(tiny_simple, broken, profiles),
        "zero-volume parents",
    )
    LOG.info("All synthetic regression checks passed")


def parse_args() -> argparse.Namespace:
    """Expose network-led inputs and explicitly deprecated legacy options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create zonal bids calibrated to PyPSA load and independently "
            "bounded by available generation; preserve native bid structures."
        )
    )
    positional = [
        "simple_input", "block_input", "network",
        "simple_output", "block_output",
    ]
    for name in positional:
        parser.add_argument(name, type=Path, nargs="?")
    parser.add_argument("--zone-column", default="zone")
    parser.add_argument(
        "--prices-csv", type=Path, default=PRICES_CSV,
        help="Historical MCP CSV required for demand calibration",
    )
    parser.add_argument(
        "--delivery-date", default=None,
        help="ISO delivery date, e.g. 2025-10-01; DST days unsupported",
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--supply-report", type=Path, default=None)
    parser.add_argument("--demand-report", type=Path, default=None)
    parser.add_argument("--report-prefix", type=Path, default=None)
    parser.add_argument(
        "--original-curves-csv", type=Path, default=ORIGINAL_CURVES_CSV,
        help="Deprecated and ignored; raw curves are not an anchor",
    )
    parser.add_argument(
        "--scarcity-share", type=float, default=SCARCITY_SHARE,
        help=(
            "Obsolete; only zero is accepted for compatibility. Back scarcity "
            "by reassigning real carriers in SEGMENT_TYPES without duplication."
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run synthetic regression checks; input paths may be omitted",
    )
    args = parser.parse_args()
    provided = [getattr(args, name) is not None for name in positional]
    if not all(provided) and not (args.self_test and not any(provided)):
        parser.error(
            "Supply all five positional paths, or use --self-test alone"
        )
    return args


def main() -> None:
    """Load the supplied PyPSA network and create the zonal books."""
    args = parse_args()
    if args.self_test:
        run_regression_checks()
        if args.simple_input is None:
            return
    create_zonal_orderbooks(
        simple_input=args.simple_input,
        block_input=args.block_input,
        network=pypsa.Network(str(args.network)),
        zone_column=args.zone_column,
        simple_output=args.simple_output,
        block_output=args.block_output,
        supply_report_output=args.supply_report,
        demand_report_output=args.demand_report,
        original_curves_csv=args.original_curves_csv,
        prices_csv=args.prices_csv,
        delivery_date=args.delivery_date,
        diagnostics_dir=args.diagnostics_dir,
        scarcity_share=args.scarcity_share,
        report_prefix=args.report_prefix,
        network_input=args.network,
    )


if __name__ == "__main__":
    main()