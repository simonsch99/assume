"""Create capacity-feasible zonal MW order books for ordinary 96-MTU days.

The original block scaling, historical MCP/AAR calibration and aggregate
availability constraints are the BASELINE. Afterwards, only simple Sell
volumes are redistributed using soft carrier/price-segment proxies. Prices,
metadata, purchases, blocks and gross simple supply are not recalibrated.

Segments come from preprocess_orders.py, which is not changed. They are price
proxies, never technology-specific capacity budgets. Scarcity has neutral
weight and does not require a carrier. All physical limits remain aggregate.

Internal quantities are MW; energy diagnostics use dt=0.25 h. Historical MCP
eligibility is Price <= MCP for sells and Price >= MCP for purchases. AAR is
used only for historical calibration, not for physical supply constraints.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa

from tools.order_layout import (
    BlockLayout, SimpleLayout, native_profile_columns, read_block_layout,
    read_simple_layout, write_block_layout, write_simple_layout,
)

SIDE = "Sale/Purchase"
MTUS = list(range(1, 97))
TOL = 1e-6
RTOL = 1e-8
QUARTER_HOURS = 0.25
MARKET_TIMEZONE = "Europe/Berlin"
SCARCITY_SEGMENT = "S5_SCARCITY"
ORIGINAL_CURVES_CSV: Path | None = None
PRICES_CSV: Path | None = None
SCARCITY_SHARE: float | None = None
CARRIER_ALPHA = 0.35
CARRIER_RATIO_CAP = 5.0

# Coefficients are soft associations, not MW budgets. A carrier can contribute
# to several segments; its coefficients are normalized across those segments.
# Multiple carriers can contribute to the same segment. Missing/unmapped
# carriers are spread uniformly across non-scarcity segments, rather than
# discarded. Names are matched case-insensitively with punctuation removed.
DEFAULT_CARRIER_SEGMENT_MAPPING = {
    "S0_NEGATIVE": {"Wind": 0.6, "Wind Offshore": 0.6, "PV": 0.6},
    "S1_LOW_COST": {
        "Wind": 0.4, "Wind Offshore": 0.4, "PV": 0.4,
        "Nuclear": 1.0, "Biomass": 1.0, "Hydro": 1.0,
    },
    "S2_LIGNITE": {"Lignite": 1.0, "Hard Coal": 0.4},
    "S3_MID_MERIT": {"Hard Coal": 0.6, "Gas/CCGT": 0.65},
    "S4_HIGH_COST": {"Gas/CCGT": 0.35, "OCGT": 1.0, "Oil": 1.0},
    "S5_SCARCITY": {},
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger(__name__)


def finite(values, label: str, nonnegative: bool = False) -> None:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{label} contains negative values")


def require_close(actual, expected, label: str) -> None:
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    finite(actual, label)
    finite(expected, f"{label} reference")
    if not np.allclose(actual, expected, atol=TOL, rtol=RTOL):
        error = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"{label}: maximum absolute error is {error:g}")


def require_within(actual, limit, label: str) -> None:
    actual, limit = np.asarray(actual, float), np.asarray(limit, float)
    finite(actual, label, nonnegative=True)
    finite(limit, f"{label} availability", nonnegative=True)
    if (actual > limit + TOL + RTOL * np.abs(limit)).any():
        excess = float(np.max(actual - limit))
        raise ValueError(f"{label} exceeds availability by up to {excess:g} MW")


def normalize_zone_label(value) -> str:
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
    if zone_column not in network.buses:
        raise ValueError(f"Network buses do not contain {zone_column!r}")
    network.buses[zone_column] = network.buses[zone_column].map(normalize_zone_label)
    zones = sorted(network.buses[zone_column].unique(), key=str)
    if not zones:
        raise ValueError("Network contains no zoned buses")
    return zones


def date_keys(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, dayfirst=True, errors="raise").dt.normalize()


def delivery_day(value) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert(MARKET_TIMEZONE).tz_localize(None)
    day = day.normalize()
    start = day.tz_localize(MARKET_TIMEZONE)
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TIMEZONE)
    if end - start != pd.Timedelta(hours=24):
        raise ValueError(f"DST delivery day {day.date()} is unsupported; 96 MTUs required")
    return day


def add_mtu(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Hour"] = (
        frame["Hour"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True).replace({"3A": "3"})
    )
    for column, upper in [("Hour", 24), ("Quarter hour", 4)]:
        values = pd.to_numeric(frame[column], errors="raise")
        finite(values, column)
        if ((values < 1) | (values > upper) | (values != np.floor(values))).any():
            raise ValueError(f"Invalid {column}; DST labels are unsupported")
        frame[column] = values.astype(int)
    frame["MTU"] = (frame["Hour"] - 1) * 4 + frame["Quarter hour"]
    return frame


def load_simple_orders(path, day=None) -> tuple[SimpleLayout, pd.Timestamp]:
    layout = read_simple_layout(path, required_columns={"bidding_zone", "Segment"})
    dates = date_keys(layout.frame["Date"])
    if day is None:
        if dates.nunique() != 1:
            raise ValueError("Specify delivery_date when simple input contains multiple days")
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
        finite(frame[column], f"Simple {column}", nonnegative=column == "Volume")
    if frame[["Segment", SIDE]].isna().any().any():
        raise ValueError("Missing simple-bid side or segment")
    if not frame[SIDE].isin(["Sell", "Purchase"]).all():
        raise ValueError("Unsupported simple-bid side")
    layout.frame = frame.reset_index(drop=True)
    return layout, day


def read_mcp(path, day: pd.Timestamp) -> pd.Series:
    prices = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    prices.columns = prices.columns.str.strip()
    if "Delivery day" not in prices:
        raise ValueError("Prices CSV lacks 'Delivery day'")
    rows = prices.loc[date_keys(prices["Delivery day"]).eq(day)]
    if len(rows) != 1:
        raise ValueError(f"Expected one MCP row for {day.date()}, found {len(rows)}")
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
        if suffix == "B" or not 1 <= hour <= 24 or (suffix == "A" and hour != 3):
            raise ValueError(f"Unexpected/DST MCP coordinate {column}")
        mtu = (hour - 1) * 4 + quarter
        if mtu in coordinates:
            raise ValueError(f"Duplicate MCP coordinate at MTU {mtu}")
        coordinates[mtu] = float(value)
    if set(coordinates) != set(MTUS):
        missing = sorted(set(MTUS) - set(coordinates))
        raise ValueError(f"{day.date()} requires 96 valid MCPs; missing MTUs: {missing}")
    return pd.Series(coordinates, name="mcp_eur_mwh").reindex(MTUS)


def normalize_block_profile_columns(layout: BlockLayout) -> list[str]:
    frame, header = layout.frame.copy(), list(layout.header)

    def numeric_profile(column):
        values = pd.to_numeric(
            frame[column].replace(r"^\s*$", np.nan, regex=True), errors="raise"
        ).fillna(0.0)
        finite(values, f"Block profile {column}")
        return values

    for quarter in range(1, 5):
        ordinary = f"Volume H03 Q{quarter}"
        first, second = f"Volume H03A Q{quarter}", f"Volume H03B Q{quarter}"
        if second in frame:
            if numeric_profile(second).ne(0.0).any():
                raise ValueError(f"{second} contains active volumes; DST is unsupported")
            frame = frame.drop(columns=second)
            header.remove(second)
        if first in frame:
            first_values = numeric_profile(first)
            if ordinary in frame:
                ordinary_values = numeric_profile(ordinary)
                if first_values.ne(0.0).any() and ordinary_values.ne(0.0).any():
                    raise ValueError(f"Both {ordinary} and {first} have active volumes")
                frame[ordinary] = first_values if first_values.ne(0.0).any() else ordinary_values
                frame = frame.drop(columns=first)
                header.remove(first)
            else:
                frame = frame.rename(columns={first: ordinary})
                header = [ordinary if column == first else column for column in header]
    profiles = [f"Volume H{hour:02d} Q{quarter}" for hour in range(1, 25) for quarter in range(1, 5)]
    detected = native_profile_columns(frame.columns)
    missing, unexpected = sorted(set(profiles) - set(detected)), sorted(set(detected) - set(profiles))
    if missing or unexpected or len(detected) != 96:
        raise ValueError(f"Expected 96 ordinary profiles; missing={missing}, unexpected={unexpected}")
    frame[profiles] = pd.DataFrame({c: numeric_profile(c) for c in profiles}, index=frame.index)
    profile_set = set(profiles)
    first_position = next(i for i, c in enumerate(header) if c in profile_set)
    layout.header = (
        [c for c in header[:first_position] if c not in profile_set] + profiles
        + [c for c in header[first_position:] if c not in profile_set]
    )
    layout.frame = frame.reindex(columns=layout.header)
    return profiles


def prepare_blocks(frame: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    required = {
        "BlockID", "Block Type", "BlockCodePRM", "Execution", "Limit Price",
        "MAR", "AAR", "Segment", "bidding_zone",
    }
    if required.difference(frame):
        raise ValueError(f"Missing block fields: {sorted(required - set(frame))}")
    if any(str(c).startswith("_") for c in frame):
        raise ValueError("Block columns beginning '_' are reserved")
    signed = frame[profiles].apply(pd.to_numeric, errors="raise")
    finite(signed, "Native block profiles")
    positive, negative = signed.gt(0.0).any(axis=1), signed.lt(0.0).any(axis=1)
    if (positive & negative).any() or (~(positive | negative)).any():
        raise ValueError("Block profiles must be nonempty and have one sign")
    ids = frame["BlockID"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("BlockID must be nonempty and unique")
    frame["BlockID"] = ids
    frame["BlockCodePRM"] = frame["BlockCodePRM"].fillna("").astype(str).str.strip()
    frame["_side"], frame["_sign"] = np.where(negative, "Sell", "Purchase"), np.where(negative, -1.0, 1.0)
    frame[profiles] = signed.abs()
    if frame[["Segment", "bidding_zone"]].isna().any().any():
        raise ValueError("Missing block segment or bidding_zone")
    finite(pd.to_numeric(frame["Limit Price"], errors="raise"), "Block prices")
    mar = pd.to_numeric(frame["MAR"], errors="raise")
    finite(mar, "Block MAR", nonnegative=True)
    if mar.gt(1.0).any():
        raise ValueError("Block MAR must be in [0, 1]")
    execution = frame["Execution"].astype("string").fillna("").str.strip().str.upper()
    if not execution.isin(["Y", "N"]).all():
        invalid = sorted(execution.loc[~execution.isin(["Y", "N"])].unique())
        raise ValueError(f"Block Execution must be Y or N; found {invalid}")
    aar = pd.to_numeric(frame["AAR"], errors="coerce")
    finite(aar, "Block AAR", nonnegative=True)
    if aar.gt(1.0).any():
        raise ValueError("Block AAR must be a fraction between 0 and 1")
    inconsistent = (execution.eq("N") & aar.ne(0.0)) | (execution.eq("Y") & aar.eq(0.0))
    if inconsistent.any():
        raise ValueError("Block Execution and AAR are inconsistent:\n" + frame.loc[inconsistent, ["BlockID", "Execution", "AAR"]].to_string(index=False))
    frame["_historical_acceptance_ratio"] = aar.astype(float)
    types = frame["Block Type"]
    if not types.isin(["C01", "C02", "C04"]).all():
        raise ValueError("Unsupported block type; define its semantics first")
    if frame.loc[types.eq("C01"), "BlockCodePRM"].ne("").any():
        raise ValueError("Unexpected BlockCodePRM on C01 block")
    if frame.loc[types.eq("C04"), "BlockCodePRM"].eq("").any():
        raise ValueError("Exclusive block has an empty group identifier")
    parents = dict(zip(ids.loc[types.eq("C02")], frame.loc[types.eq("C02"), "BlockCodePRM"]))
    if any(parent not in set(ids) for parent in parents.values()):
        raise ValueError("C02 references an unknown/empty parent BlockID")
    acceptance = dict(zip(ids, frame["_historical_acceptance_ratio"]))
    inconsistent_links = [f"{child}->{parent}" for child, parent in parents.items() if acceptance[child] > 0.0 and acceptance[parent] == 0.0]
    if inconsistent_links:
        raise ValueError("Accepted linked children have rejected parents: " + ", ".join(inconsistent_links))
    exclusive = types.eq("C04")
    reused = frame.loc[exclusive].groupby("BlockCodePRM")["bidding_zone"].nunique()
    if reused.gt(1).any():
        raise ValueError("C04 group codes reused across source zones would merge after zonal prefixing: " + str(reused.index[reused.gt(1)].tolist()))
    executed = frame.loc[exclusive & frame["_historical_acceptance_ratio"].gt(0.0)].groupby("BlockCodePRM").size()
    if executed.gt(1).any():
        raise ValueError("Multiple historically accepted alternatives in C04 groups: " + str(executed.index[executed.gt(1)].tolist()))
    roots = []
    for block_id in ids:
        seen, root = set(), block_id
        while root in parents:
            if root in seen:
                raise ValueError(f"C02 cycle containing {block_id}")
            seen.add(root)
            root = parents[root]
        roots.append(root)
    frame["_root"] = roots
    exclusive_root_groups = dict(zip(ids.loc[exclusive], frame.loc[exclusive, "BlockCodePRM"]))
    frame["_exclusive_family_group"] = frame["_root"].map(exclusive_root_groups).fillna("")
    frame["_exclusive_group"] = np.where(exclusive, frame["BlockCodePRM"], "")
    frame["_relationship_family"] = np.where(
        frame["_exclusive_family_group"].ne(""),
        "exclusive:" + frame["_exclusive_family_group"], "linked:" + frame["_root"],
    )
    frame["_family"] = frame["_relationship_family"] + ":bid:" + ids
    frame["_start"] = [int(np.flatnonzero(row > 0)[0]) + 1 for row in frame[profiles].to_numpy(float)]
    return frame


def load_block_orders(path, day) -> tuple[BlockLayout, list[str]]:
    layout = read_block_layout(path)
    if len(layout.header) != len(set(layout.header)):
        raise ValueError("Duplicate block-column names")
    dates = []
    for line in layout.st_rows:
        fields = next(csv.reader([line], delimiter=";"))
        if len(fields) < 2:
            raise ValueError("Malformed ST metadata")
        dates.append(pd.to_datetime(fields[1], dayfirst=True).normalize())
    if not dates or any(value != day for value in dates):
        raise ValueError(f"Block ST dates must match {day.date()}; found {dates}")
    profiles = normalize_block_profile_columns(layout)
    layout.frame = prepare_blocks(layout.frame, profiles)
    return layout, profiles


def historical_accepted_block_totals(blocks, profiles, side) -> pd.Series:
    if side not in ["Sell", "Purchase"]:
        raise ValueError(f"Unsupported block side: {side}")
    selected = blocks.loc[blocks["_side"].eq(side)]
    values = selected[profiles].mul(selected["_historical_acceptance_ratio"], axis=0).sum(axis=0)
    return pd.Series(values.to_numpy(float), index=MTUS)


def matched_demand(simple, blocks, profiles, mcp):
    if not mcp.index.equals(pd.Index(MTUS)):
        raise ValueError("MCP series must contain exactly MTUs 1 to 96")
    mask = simple[SIDE].eq("Purchase") & simple["Price"].ge(simple["MTU"].map(mcp))
    matched_simple = simple.loc[mask].groupby("MTU")["Volume"].sum().reindex(MTUS, fill_value=0.0)
    historical_blocks = historical_accepted_block_totals(blocks, profiles, "Purchase")
    matched = matched_simple + historical_blocks
    finite(matched, "Historical matched demand", nonnegative=True)
    diagnostics = pd.DataFrame({
        "MTU": MTUS, "mcp_eur_mwh": mcp.to_numpy(),
        "M_simple_mw": matched_simple.to_numpy(), "B_historical_mw": historical_blocks.to_numpy(),
        "historical_matched_demand_mw": matched.to_numpy(),
        "matched_simple_bid_count": simple.loc[mask].groupby("MTU").size().reindex(MTUS, fill_value=0).to_numpy(),
    })
    return matched, diagnostics


def snapshot_component_data(component, dynamic, attribute, selected, default):
    """Keep the baseline policy: dynamic gaps are physical-input errors."""
    static = pd.to_numeric(component.get(attribute, pd.Series(default, index=component.index, dtype=float)), errors="raise")
    data = pd.DataFrame(index=selected, columns=component.index, dtype=float)
    for name in component.index:
        data[name] = pd.to_numeric(dynamic[name], errors="raise").reindex(selected) if name in dynamic.columns else static.at[name]
    finite(data, f"PyPSA {attribute}", nonnegative=True)
    return data


def network_zonal_data(network, zone_column, zones, day):
    """Return load, installed p_nom, aggregate availability and carrier MW.

    The fourth result has MTU rows and (zone, carrier) MultiIndex columns.
    All values use p_nom * snapshot p_max_pu, with the original static fallback.
    Absent zone/carrier combinations are zero. Blank carrier labels optionally
    use Type (as in df_gen_filtered.csv); otherwise they are unspecified.
    """
    if isinstance(network.snapshots, pd.MultiIndex):
        raise ValueError("Multi-period network snapshots are unsupported")
    snapshots = pd.DatetimeIndex(network.snapshots)
    local = snapshots.tz_convert(MARKET_TIMEZONE).tz_localize(None) if snapshots.tz is not None else snapshots
    if local.has_duplicates or not local.is_monotonic_increasing:
        raise ValueError("Network snapshots must be unique and chronological")
    required = pd.date_range(day, periods=96, freq="15min")
    missing = required.difference(local)
    if not missing.empty:
        raise ValueError(f"Missing network snapshots: {list(missing)}")
    if int((local.normalize() == day).sum()) != 96:
        raise ValueError("The network delivery day must contain exactly 96 MTUs")
    selected = snapshots[local.get_indexer(required)]
    bus_zone = network.buses[zone_column]
    generators = network.generators
    generator_zone = generators["bus"].map(bus_zone)
    if generator_zone.isna().any():
        raise ValueError("Generator connected to unknown/unzoned bus")
    p_nom = pd.to_numeric(generators["p_nom"], errors="raise")
    finite(p_nom, "Generator p_nom", nonnegative=True)
    p_max_pu = snapshot_component_data(generators, network.generators_t.p_max_pu, "p_max_pu", selected, 1.0)
    generator_available = p_max_pu.mul(p_nom, axis=1)
    finite(generator_available, "Available generation", nonnegative=True)
    carrier = generators.get("carrier", pd.Series("", index=generators.index)).astype("string").fillna("").str.strip()
    if "Type" in generators:
        fallback = generators["Type"].astype("string").fillna("").str.strip()
        carrier = carrier.mask(carrier.eq(""), fallback)
    carrier = carrier.mask(carrier.eq(""), "__unspecified__")
    carriers = sorted(carrier.unique(), key=str)
    carrier_columns = pd.MultiIndex.from_product([zones, carriers], names=["zone", "carrier"])
    available_by_carrier = pd.DataFrame(0.0, index=MTUS, columns=carrier_columns)
    capacity = {}
    available = pd.DataFrame(0.0, index=MTUS, columns=zones)
    for zone in zones:
        mask = generator_zone.eq(zone)
        capacity[zone] = float(p_nom.loc[mask].sum())
        available[zone] = generator_available.loc[:, mask].sum(axis=1).to_numpy()
        for name in carriers:
            available_by_carrier[(zone, name)] = generator_available.loc[:, mask & carrier.eq(name)].sum(axis=1).to_numpy()
        require_close(available_by_carrier[zone].sum(axis=1) if carriers else np.zeros(96), available[zone], f"{zone}: carrier availability aggregation")
    loads = network.loads
    load_zone = loads["bus"].map(bus_zone)
    if load_zone.isna().any():
        raise ValueError("Load connected to unknown/unzoned bus")
    load_data = snapshot_component_data(loads, network.loads_t.p_set, "p_set", selected, 0.0)
    zonal_load = pd.DataFrame(0.0, index=MTUS, columns=zones)
    for zone in zones:
        zonal_load[zone] = load_data.loc[:, load_zone.eq(zone)].sum(axis=1).to_numpy()
    finite(available, "Aggregate zonal availability", nonnegative=True)
    finite(zonal_load, "Zonal load", nonnegative=True)
    return zonal_load, capacity, available, available_by_carrier


def reference_generation_targets(load, available):
    expected_index = pd.Index(MTUS)
    if not load.index.equals(expected_index) or not available.index.equals(expected_index):
        raise ValueError("Load and availability must contain MTUs 1 to 96")
    if not load.columns.equals(available.columns):
        raise ValueError("Zonal load and availability must contain the same zones")
    finite(load, "System zonal load", nonnegative=True)
    finite(available, "System zonal availability", nonnegative=True)
    system_load, system_available = load.sum(axis=1), available.sum(axis=1)
    reference = pd.DataFrame(0.0, index=MTUS, columns=load.columns, dtype=float)
    positive_available = system_available.gt(0.0)
    for mtu in MTUS:
        if positive_available.at[mtu]:
            reference.loc[mtu] = available.loc[mtu] * system_load.at[mtu] / system_available.at[mtu]
    finite(reference, "Reference generation", nonnegative=True)
    if positive_available.any():
        require_close(reference.sum(axis=1).loc[positive_available], system_load.loc[positive_available], "Reference-generation allocation")
    unavailable_with_load = system_available.eq(0.0) & system_load.gt(TOL)
    capacity_shortfall = system_load > system_available + TOL + RTOL * system_available.abs()
    diagnostics = pd.DataFrame({
        "MTU": MTUS, "system_load_mw": system_load.to_numpy(),
        "system_available_generation_mw": system_available.to_numpy(),
        "reference_generation_total_mw": reference.sum(axis=1).to_numpy(),
        "unallocated_reference_generation_mw": (system_load - reference.sum(axis=1)).clip(lower=0.0).to_numpy(),
        "system_capacity_shortfall_mw": (system_load - system_available).clip(lower=0.0).to_numpy(),
        "reference_generation_defined": positive_available.to_numpy(),
        "reference_generation_input_feasible": (~(unavailable_with_load | capacity_shortfall)).to_numpy(),
        "reference_generation_status": np.select(
            [unavailable_with_load, capacity_shortfall],
            ["positive_system_load_without_available_generation", "system_load_exceeds_available_generation"], default="feasible",
        ),
    })
    return reference, diagnostics


def block_totals(blocks, profiles, side, *, effective_only=True):
    """Original conservative envelope, including descendants of C04 roots."""
    if side not in ["Sell", "Purchase"]:
        raise ValueError(f"Unsupported block side: {side}")
    selected = blocks.loc[blocks["_side"].eq(side)]
    if not effective_only:
        return pd.Series(selected[profiles].sum(axis=0).to_numpy(float), index=MTUS)
    independent = selected["_exclusive_family_group"].eq("")
    values = selected.loc[independent, profiles].sum(axis=0).to_numpy(dtype=float, copy=True)
    alternatives = selected.loc[~independent].groupby(["_exclusive_family_group", "_root"], sort=False)[profiles].sum()
    for _, group in alternatives.groupby(level=0, sort=False):
        values += group.max(axis=0).to_numpy(dtype=float, copy=True)
    return pd.Series(values, index=MTUS, dtype=float)


def template_scaling(target_remaining, template, label):
    finite(target_remaining, f"{label}: remaining target")
    finite(template, f"{label}: template", nonnegative=True)
    factors = pd.Series(0.0, index=MTUS, dtype=float)
    usable = template.gt(0.0)
    factors.loc[usable] = target_remaining.loc[usable].clip(lower=0.0) / template.loc[usable]
    finite(factors, f"{label}: factors", nonnegative=True)
    return factors


def zone_scaling(
    load, reference_generation, available, matched_simple_purchase,
    historical_buy_blocks, matched_simple_sell, historical_sell_blocks,
    gross_simple_sell, sell_block_envelope, zone,
):
    """Unchanged baseline: calibrate once, capped by gross physical supply."""
    vectors = [
        (load, "load"), (reference_generation, "reference generation"),
        (available, "available generation"), (matched_simple_purchase, "matched simple purchases"),
        (historical_buy_blocks, "historical purchase blocks"), (matched_simple_sell, "MCP-eligible simple sells"),
        (historical_sell_blocks, "historical sell blocks"), (gross_simple_sell, "gross simple sells"),
        (sell_block_envelope, "sell-block envelope"),
    ]
    for values, label in vectors:
        if not values.index.equals(pd.Index(MTUS)):
            raise ValueError(f"{zone}: {label} requires MTUs 1 to 96")
        finite(values, f"{zone}: {label}", nonnegative=True)
    historical_demand_energy = float((matched_simple_purchase + historical_buy_blocks).sum() * QUARTER_HOURS)
    load_energy = float(load.sum() * QUARTER_HOURS)
    initial_block_factor = load_energy / historical_demand_energy if historical_demand_energy > 0.0 else 0.0
    finite([initial_block_factor], f"{zone}: initial common block factor", nonnegative=True)
    active_envelope = sell_block_envelope.gt(0.0)
    block_capacity_ratios = available.loc[active_envelope] / sell_block_envelope.loc[active_envelope]
    block_factor_cap = float(block_capacity_ratios.min()) if active_envelope.any() else np.inf
    block_factor = min(initial_block_factor, block_factor_cap)
    scaled_sell_envelope = block_factor * sell_block_envelope
    require_within(scaled_sell_envelope, available, f"{zone}: aggregate sell-block envelope")
    purchase_factors = template_scaling(load - block_factor * historical_buy_blocks, matched_simple_purchase, f"{zone}: simple purchases")
    remaining_reference_generation = (reference_generation - block_factor * historical_sell_blocks).clip(lower=0.0)
    requested_sell_factors = template_scaling(remaining_reference_generation, matched_simple_sell, f"{zone}: MCP-eligible simple sells")
    remaining_availability = (available - scaled_sell_envelope).clip(lower=0.0)
    capacity_sell_factors = pd.Series(0.0, index=MTUS, dtype=float)
    gross_active = gross_simple_sell.gt(0.0)
    capacity_sell_factors.loc[gross_active] = remaining_availability.loc[gross_active] / gross_simple_sell.loc[gross_active]
    finite(capacity_sell_factors, f"{zone}: simple sell capacity factors", nonnegative=True)
    sell_factors = pd.concat([requested_sell_factors, capacity_sell_factors], axis=1).min(axis=1)
    sell_factors.name = "simple_sell_factor"
    finite(sell_factors, f"{zone}: final baseline simple sell factors", nonnegative=True)
    no_eligible_sell_template = matched_simple_sell.eq(0.0)
    capacity_limited = requested_sell_factors > capacity_sell_factors + TOL + RTOL * capacity_sell_factors.abs()
    target_exceeds_available = reference_generation > available + TOL + RTOL * available.abs()
    capped = block_factor < initial_block_factor
    daily = {
        "zone": zone, "pypsa_load_energy_mwh": load_energy,
        "reference_generation_energy_mwh": float(reference_generation.sum() * QUARTER_HOURS),
        "historical_matched_demand_energy_mwh": historical_demand_energy,
        "historical_matched_demand_template_missing": historical_demand_energy == 0.0,
        "initial_common_block_factor": initial_block_factor,
        "aggregate_sell_block_factor_cap": block_factor_cap if np.isfinite(block_factor_cap) else None,
        "common_block_factor": block_factor, "common_block_factor_capped": capped,
        "common_physical_reduction_factor": block_factor / initial_block_factor if initial_block_factor > 0.0 else 1.0,
        "block_cap_binding_mtus": "|".join(str(mtu) for mtu in block_capacity_ratios.index[block_capacity_ratios.eq(block_factor_cap)]) if capped else "",
        "zero_matched_simple_purchase_template_mtu_count": int(matched_simple_purchase.eq(0.0).sum()),
        "zero_matched_simple_sell_template_mtu_count": int(no_eligible_sell_template.sum()),
        "baseline_simple_sell_capacity_limited_mtu_count": int(capacity_limited.sum()),
        "reference_generation_exceeds_availability_mtu_count": int(target_exceeds_available.sum()),
    }
    return (block_factor, purchase_factors, sell_factors, requested_sell_factors, capacity_sell_factors, no_eligible_sell_template, capacity_limited, target_exceeds_available, daily)


def prefixed_value(zone, value) -> str:
    return "" if pd.isna(value) or not str(value).strip() else f"{zone}_{str(value).strip()}"


def scale_blocks(blocks, profiles, zone, multiplier, initial_multiplier):
    if not np.isscalar(multiplier) or not np.isscalar(initial_multiplier):
        raise ValueError("Block multipliers must be zone/day scalars")
    finite([multiplier, initial_multiplier], "Common block multipliers", nonnegative=True)
    if multiplier > initial_multiplier:
        raise ValueError("Final block multiplier exceeds its initial value")
    zonal = blocks.copy()
    zonal["_initial"] = initial_multiplier
    zonal["_downscale"] = multiplier / initial_multiplier if initial_multiplier > 0.0 else 1.0
    zonal[profiles] = zonal[profiles] * multiplier
    zonal["bidding_zone"] = zone
    for column in ["BlockID", "BlockCodePRM", "_root", "_exclusive_group", "_exclusive_family_group"]:
        zonal[column] = zonal[column].map(lambda value: prefixed_value(zone, value))
    finite(zonal[profiles], "Scaled block magnitudes", nonnegative=True)
    require_close(zonal["_initial"] * zonal["_downscale"], np.full(len(zonal), multiplier), f"{zone}: common multiplier across all blocks")
    return zonal


def calibration_status(residual, target, missing_template, side, *, target_exceeds_available=False):
    """Classify BASELINE residuals only; reweighting residuals are separate."""
    if abs(residual) <= TOL + RTOL * abs(target):
        return "calibrated"
    if residual < 0.0:
        return "purchase_blocks_exceed_load" if side == "Purchase" else "sell_blocks_exceed_reference_generation"
    if side == "Sell" and target_exceeds_available:
        return "reference_generation_exceeds_available_generation"
    if missing_template:
        return "missing_matched_simple_purchase_template" if side == "Purchase" else "missing_simple_sell_template"
    return "unfilled_target"


def simple_totals(simple, side, segment=None) -> pd.Series:
    # Preserve the original pandas aggregation used by baseline calibration.
    mask = simple[SIDE].eq(side)
    if segment is not None:
        mask &= simple["Segment"].eq(segment)
    return simple.loc[mask].groupby("MTU")["Volume"].sum().reindex(MTUS, fill_value=0.0)


def matched_simple_supply(simple, mcp) -> pd.Series:
    if not mcp.index.equals(pd.Index(MTUS)):
        raise ValueError("MCP series must contain exactly MTUs 1 to 96")
    eligible = simple[SIDE].eq("Sell") & simple["Price"].le(simple["MTU"].map(mcp))
    result = simple.loc[eligible].groupby("MTU")["Volume"].sum().reindex(MTUS, fill_value=0.0).astype(float)
    finite(result, "MCP-eligible simple supply", nonnegative=True)
    return result


def same_metadata_value(left, right) -> bool:
    return bool((pd.isna(left) and pd.isna(right)) or (not pd.isna(left) and not pd.isna(right) and left == right))


def validate_relationships(blocks):
    if blocks["BlockID"].duplicated().any():
        raise ValueError("Duplicate zonal BlockID")
    linked = blocks["Block Type"].eq("C02")
    missing = linked & ~blocks["BlockCodePRM"].isin(set(blocks["BlockID"]))
    if missing.any():
        raise ValueError("Retained C02 blocks reference missing parents: " + ", ".join(blocks.loc[missing, "BlockID"]))
    exclusive = blocks["Block Type"].eq("C04")
    if blocks.loc[exclusive, "BlockCodePRM"].eq("").any():
        raise ValueError("Retained C04 block has an empty group code")
    merged = blocks.loc[exclusive].groupby("BlockCodePRM")["bidding_zone"].nunique()
    if merged.gt(1).any():
        raise ValueError("Exclusive-group prefixes collide across output zones: " + ", ".join(merged.index[merged.gt(1)]))


def validate_blocks(source, zonal, profiles, zone) -> list[dict]:
    if len(source) != len(zonal):
        raise ValueError("Blocks were removed before preservation validation")
    metadata = [c for c in source if c not in profiles and not c.startswith("_") and c != "bidding_zone"]
    rows = []
    for position in range(len(source)):
        before, after = source.iloc[position], zonal.iloc[position]
        for column in metadata:
            expected = prefixed_value(zone, before[column]) if column in ["BlockID", "BlockCodePRM"] else before[column]
            if not same_metadata_value(after[column], expected):
                raise ValueError(f"Changed block metadata: {before['BlockID']}/{column}")
        if after["bidding_zone"] != zone or after["_sign"] != before["_sign"]:
            raise ValueError("Changed block zone or native sign")
        original, transformed = before[profiles].to_numpy(float), after[profiles].to_numpy(float)
        factor = float(after["_initial"] * after["_downscale"])
        finite([factor], "Block multiplier", nonnegative=True)
        finite(transformed, "Final block magnitudes", nonnegative=True)
        require_close(transformed, original * factor, f"{after['BlockID']}: constant whole-profile multiplier")
        if (transformed[original == 0.0] != 0.0).any():
            raise ValueError("Block scaling introduced an active profile period")
        rows.append({
            "zone": zone, "original_block_id": before["BlockID"], "block_id": after["BlockID"],
            "block_family_id": f"{zone}:{before['_family']}", "relationship_family_id": before["_relationship_family"],
            "block_type": before["Block Type"], "block_code_prm": after["BlockCodePRM"],
            "side": before["_side"], "segment": before["Segment"], "block_starting_mtu": before["_start"],
            "initial_multiplier": after["_initial"], "physical_reduction_factor": after["_downscale"],
            "final_total_multiplier": factor, "original_energy_mwh": original.sum() * QUARTER_HOURS,
            "final_energy_mwh": transformed.sum() * QUARTER_HOURS,
            "historical_execution": before["Execution"], "historical_aar": before["_historical_acceptance_ratio"],
            "constant_profile_multiplier_valid": True, "metadata_and_relationships_valid": True,
            "native_sign_preserved": True, "carrier_reweighting_applied": False,
        })
    validate_relationships(zonal)
    return rows


def filter_exact_zeros(simple, blocks, profiles):
    zero_simple, zero_blocks = simple["Volume"].eq(0.0), blocks[profiles].eq(0.0).all(axis=1)
    removed_ids = set(blocks.loc[zero_blocks, "BlockID"])
    dangling = ~zero_blocks & blocks["Block Type"].eq("C02") & blocks["BlockCodePRM"].isin(removed_ids)
    if dangling.any():
        links = [f"{row['BlockID']}->{row['BlockCodePRM']}" for _, row in blocks.loc[dangling].iterrows()]
        raise ValueError("Cannot filter zero-volume parents of retained C02 children: " + ", ".join(links))
    retained_simple = simple.loc[~zero_simple].copy().reset_index(drop=True)
    retained_blocks = blocks.loc[~zero_blocks].copy().reset_index(drop=True)
    validate_relationships(retained_blocks)
    LOG.info("Removed %d exact-zero simple orders and %d all-zero blocks", int(zero_simple.sum()), int(zero_blocks.sum()))
    return retained_simple, retained_blocks


def carrier_key(value) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return {
        "solar": "pv", "onwind": "wind", "offwind": "windoffshore",
        "offwindac": "windoffshore", "offwinddc": "windoffshore",
        "ccgt": "gasccgt", "gas": "gasccgt", "coal": "hardcoal",
    }.get(key, key)


def normalize_carrier_mapping(mapping=None) -> dict[str, dict[str, float]]:
    """Accept segment -> {carrier: coefficient} or segment -> [carriers]."""
    source = DEFAULT_CARRIER_SEGMENT_MAPPING if mapping is None else mapping
    if not isinstance(source, dict):
        raise ValueError("Carrier mapping must be a segment-keyed JSON object")
    result = {}
    for segment, associations in source.items():
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError("Carrier mapping has an invalid segment label")
        if isinstance(associations, list):
            associations = dict.fromkeys(associations, 1.0)
        if associations is None:
            associations = {}
        if not isinstance(associations, dict):
            raise ValueError(f"{segment}: use an object of coefficients or a list of carriers")
        normalized = {}
        for carrier, coefficient in associations.items():
            if not isinstance(carrier, str) or not carrier.strip():
                raise ValueError("Carrier mapping has an invalid carrier label")
            coefficient = float(coefficient)
            finite([coefficient], f"{segment}/{carrier} mapping coefficient", nonnegative=True)
            key = carrier_key(carrier)
            normalized[key] = normalized.get(key, 0.0) + coefficient
            finite([normalized[key]], f"{segment}/{carrier} combined coefficient", nonnegative=True)
        if segment == SCARCITY_SEGMENT and any(v > 0.0 for v in normalized.values()):
            raise ValueError("Scarcity is a neutral price proxy; its mapping must be empty")
        result[segment] = normalized
    return result


def validate_carrier_config(alpha, ratio_cap, ratio_caps):
    if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError("carrier_alpha must satisfy 0 <= alpha < 1")
    caps = {} if ratio_caps is None else ratio_caps
    if not isinstance(caps, dict):
        raise ValueError("carrier_ratio_caps must be a segment-keyed object")
    for name, cap in [("global", ratio_cap), *caps.items()]:
        if cap is not None and (not np.isfinite(float(cap)) or float(cap) < 1.0):
            raise ValueError(f"{name}: a carrier ratio cap must be finite and >= 1, or null")


def carrier_segment_weights(
    available_by_carrier, zones, segments, *, alpha=CARRIER_ALPHA,
    mapping=None, ratio_cap=CARRIER_RATIO_CAP, ratio_caps=None,
):
    """Compute c[z,s,t], c_system[s,t] and strictly positive soft weights.

    Each carrier's available MW is split using its normalized associations.
    Unmapped carriers contribute uniformly to all non-scarcity proxy segments.
    Proxy capacity shares sum to one whenever availability is defined and
    positive. System shares are calculated from pooled MW, NOT an unweighted
    average of zonal shares. Scarcity share is zero and its ratio is one.

    Missing zonal carriers are zero. Zero zonal totals or undefined availability
    use ratio=1. A zero system segment share also uses ratio=1. Undefined data
    in any system zone makes system shares undefined for that MTU, hence neutral
    ratios; physical network data still rejects gaps before reaching here.
    """
    validate_carrier_config(alpha, ratio_cap, ratio_caps)
    mapping = normalize_carrier_mapping(mapping)
    zones, segments = list(zones), list(dict.fromkeys(segments))
    if not available_by_carrier.index.equals(pd.Index(MTUS)):
        raise ValueError("Carrier availability must contain MTUs 1 to 96")
    columns = available_by_carrier.columns
    if not isinstance(columns, pd.MultiIndex) or columns.nlevels != 2:
        raise ValueError("Carrier availability columns must be (zone, carrier)")
    if columns.has_duplicates:
        raise ValueError("Duplicate zone/carrier availability columns")
    if set(columns.get_level_values(0)) - set(zones):
        raise ValueError("Carrier availability contains an unknown zone")
    values = available_by_carrier.to_numpy(float)
    if (values[np.isfinite(values)] < 0.0).any():
        raise ValueError("Carrier availability contains negative values")
    output_columns = pd.MultiIndex.from_product([zones, segments], names=["zone", "segment"])
    weights = pd.DataFrame(1.0, index=MTUS, columns=output_columns)
    shares = pd.DataFrame(0.0, index=MTUS, columns=output_columns)
    proxy_mw = pd.DataFrame(0.0, index=MTUS, columns=output_columns)
    totals = pd.DataFrame(0.0, index=MTUS, columns=zones)
    defined = pd.DataFrame(False, index=MTUS, columns=zones)
    system_shares = pd.DataFrame(0.0, index=MTUS, columns=segments)
    if not segments:
        return weights, shares, system_shares, pd.DataFrame()
    ordinary = [s for s in segments if s != SCARCITY_SEGMENT]
    carriers = list(dict.fromkeys(columns.get_level_values(1)))
    allocation = pd.DataFrame(0.0, index=carriers, columns=segments)
    for carrier in carriers:
        coefficients = np.array([mapping.get(s, {}).get(carrier_key(carrier), 0.0) for s in ordinary], float)
        if len(coefficients):
            # Rescale before summing to avoid overflow in user-supplied scores.
            largest = float(coefficients.max())
            if largest > 0.0:
                coefficients /= largest
                coefficients /= coefficients.sum()
            else:
                coefficients[:] = 1.0 / len(coefficients)
            allocation.loc[carrier, ordinary] = coefficients
    for zone in zones:
        zone_columns = [c for c in columns if c[0] == zone]
        if not zone_columns:
            continue
        data = available_by_carrier.loc[:, zone_columns].to_numpy(float)
        valid = np.isfinite(data).all(axis=1)
        safe = np.where(np.isfinite(data), data, 0.0)
        zone_total = safe.sum(axis=1)
        finite(zone_total, f"{zone}: total carrier availability", nonnegative=True)
        totals[zone] = zone_total
        defined[zone] = valid & (zone_total > 0.0)
        matrix = allocation.loc[[c[1] for c in zone_columns], segments].to_numpy(float)
        zone_proxy = safe @ matrix
        proxy_mw.loc[:, pd.IndexSlice[zone, :]] = zone_proxy
        zone_shares = np.zeros_like(zone_proxy)
        np.divide(zone_proxy, zone_total[:, None], out=zone_shares, where=defined[zone].to_numpy()[:, None])
        shares.loc[:, pd.IndexSlice[zone, :]] = zone_shares
        if ordinary and defined[zone].any():
            require_close(zone_shares.sum(axis=1)[defined[zone]], np.ones(int(defined[zone].sum())), f"{zone}: normalized segment capacity shares")
    system_total = totals.sum(axis=1)
    # Zero-availability zones do not invalidate an otherwise defined system.
    system_defined = pd.Series(True, index=MTUS)
    for zone in zones:
        zone_columns = [c for c in columns if c[0] == zone]
        if zone_columns:
            system_defined &= np.isfinite(available_by_carrier.loc[:, zone_columns].to_numpy(float)).all(axis=1)
    system_defined &= system_total.gt(0.0)
    for segment in segments:
        pooled = proxy_mw.loc[:, pd.IndexSlice[:, segment]].sum(axis=1)
        system_shares.loc[system_defined, segment] = pooled.loc[system_defined] / system_total.loc[system_defined]
    rows = []
    for zone in zones:
        for segment in segments:
            local_share, system_share = shares[(zone, segment)], system_shares[segment]
            ratio_defined = defined[zone] & system_defined & system_share.gt(0.0)
            raw_ratio = pd.Series(1.0, index=MTUS)
            raw_ratio.loc[ratio_defined] = local_share.loc[ratio_defined] / system_share.loc[ratio_defined]
            finite(raw_ratio, f"{zone}/{segment}: carrier share ratio", nonnegative=True)
            cap = (ratio_caps or {}).get(segment, ratio_cap)
            ratio = raw_ratio if cap is None else raw_ratio.clip(upper=float(cap))
            weight = (1.0 - alpha) + alpha * ratio
            finite(weight, f"{zone}/{segment}: carrier weights", nonnegative=True)
            if weight.le(0.0).any():
                raise ValueError("Carrier weights must be strictly positive")
            weights[(zone, segment)] = weight
            for mtu in MTUS:
                rows.append({
                    "zone": zone, "MTU": mtu, "segment": segment,
                    "proxy_available_generation_mw": proxy_mw.at[mtu, (zone, segment)],
                    "zonal_available_generation_mw": totals.at[mtu, zone],
                    "zonal_segment_capacity_share": local_share.at[mtu],
                    "system_segment_capacity_share": system_share.at[mtu],
                    "zonal_share_defined": bool(defined.at[mtu, zone]),
                    "system_share_defined": bool(system_defined.at[mtu]),
                    "carrier_ratio_defined": bool(ratio_defined.at[mtu]),
                    "carrier_ratio_fallback_one": bool(not ratio_defined.at[mtu]),
                    "carrier_ratio_raw": raw_ratio.at[mtu], "carrier_ratio": ratio.at[mtu],
                    "carrier_ratio_cap": cap, "carrier_ratio_capped": bool(ratio.at[mtu] < raw_ratio.at[mtu]),
                    "carrier_alpha": alpha, "carrier_weight": weight.at[mtu],
                    "scarcity_neutral": segment == SCARCITY_SEGMENT,
                })
    return weights, shares, system_shares, pd.DataFrame(rows)


def _pandas_group_total(values):
    """Match the original groupby sum as well as checking accurate fsum."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    return float(pd.Series(values).groupby(np.zeros(len(values), dtype=int)).sum().iloc[0])


def _conservative_segment_scaling(q0, segment_labels, order_weights):
    """Apply q0*w*sum(q0)/sum(q0*w), with floating-point safeguards.

    Numerical closure adjusts a WHOLE segment's multiplier by a few ulps,
    never a single bid. Thus within-segment proportions remain unchanged to
    floating-point precision. Both math.fsum and the baseline pandas groupby
    total must equal their baseline values exactly. If such closure cannot be
    represented safely (including underflow), retain this MTU's baseline and
    report the numerical fallback. No historical-MCP recalibration is involved.
    """
    q0, order_weights = np.asarray(q0, float), np.asarray(order_weights, float)
    target, target_pandas = math.fsum(q0), _pandas_group_total(q0)
    if target == 0.0 or np.all(order_weights == order_weights[0]):
        return q0.copy(), False
    unique = list(dict.fromkeys(segment_labels))
    masks = [np.asarray(segment_labels) == s for s in unique]
    # A common rescaling of all weights cancels in the requested formula.
    safe_weights = order_weights / float(order_weights.max())
    weighted_total = math.fsum(q0 * safe_weights)
    if weighted_total <= 0.0 or not math.isfinite(weighted_total):
        return q0.copy(), True
    multipliers = np.array([safe_weights[np.flatnonzero(mask)[0]] for mask in masks])
    multipliers *= target / weighted_total
    initial = multipliers.copy()
    segment_totals = np.array([math.fsum(q0[mask]) for mask in masks])

    def build(factors):
        result = np.zeros_like(q0)
        for mask, factor in zip(masks, factors):
            result[mask] = q0[mask] * factor
        return result

    def acceptable(result):
        return (
            np.isfinite(result).all() and (result >= 0.0).all()
            and not ((q0 > 0.0) & (result <= 0.0)).any()
            and math.fsum(result) == target and _pandas_group_total(result) == target_pandas
        )

    candidate = build(multipliers)
    if acceptable(candidate):
        return candidate, False
    # Try each active segment, largest first; corrections remain at roundoff
    # scale and do not impose a segment-specific physical capacity budget.
    for j in np.argsort(-segment_totals):
        if segment_totals[j] == 0.0:
            continue
        trial = initial.copy()
        for _ in range(12):
            candidate = build(trial)
            if acceptable(candidate):
                return candidate, False
            delta = target - math.fsum(candidate)
            if delta == 0.0:
                delta = target_pandas - _pandas_group_total(candidate)
            proposal = trial[j] + delta / segment_totals[j]
            if proposal == trial[j]:
                proposal = np.nextafter(trial[j], np.inf if delta > 0.0 else -np.inf)
            if not np.isfinite(proposal) or proposal <= 0.0:
                break
            if abs(proposal - initial[j]) > 128 * np.finfo(float).eps * max(abs(initial[j]), np.finfo(float).tiny):
                break
            trial[j] = proposal
        # Search nearby representable whole-segment factors if iteration
        # oscillated between the two sides of a rounded aggregate total.
        for direction in [-np.inf, np.inf]:
            trial = initial.copy()
            for _ in range(16):
                trial[j] = np.nextafter(trial[j], direction)
                if trial[j] <= 0.0 or not np.isfinite(trial[j]):
                    break
                candidate = build(trial)
                if acceptable(candidate):
                    return candidate, False
    return q0.copy(), True


def reweight_simple_sell_orders(baseline, weights, *, alpha=CARRIER_ALPHA):
    """Redistribute simple Sell orders by zone/MTU; never mutate the input."""
    if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError("carrier_alpha must satisfy 0 <= alpha < 1")
    required = {SIDE, "bidding_zone", "MTU", "Segment", "Volume"}
    if required - set(baseline):
        raise ValueError(f"Missing reweighting columns: {sorted(required - set(baseline))}")
    if not baseline.index.is_unique:
        raise ValueError("Simple order index must be unique")
    finite(baseline["Volume"], "Baseline simple volume", nonnegative=True)
    result = baseline.copy(deep=True)
    sell = baseline[SIDE].eq("Sell")
    rows = []
    for (zone, mtu), group in baseline.loc[sell].groupby(["bidding_zone", "MTU"], sort=False):
        q0 = group["Volume"].to_numpy(float)
        labels = group["Segment"].to_numpy()
        total = math.fsum(q0)
        order_weights = np.array([weights.at[mtu, (zone, segment)] for segment in labels], float)
        finite(order_weights, f"{zone}/{mtu}: order weights")
        if (order_weights <= 0.0).any():
            raise ValueError("Simple Sell weights must be strictly positive")
        if alpha == 0.0 or total == 0.0:
            q_new, fallback = q0.copy(), False
        else:
            q_new, fallback = _conservative_segment_scaling(q0, labels, order_weights)
        result.loc[group.index, "Volume"] = q_new
        if math.fsum(q_new) != total or _pandas_group_total(q_new) != _pandas_group_total(q0):
            raise ValueError(f"{zone}/{mtu}: simple supply was not conserved exactly")
        if ((q0 > 0.0) & (q_new <= 0.0)).any():
            raise ValueError(f"{zone}/{mtu}: carrier reweighting removed a positive offer")
        # Validate one multiplier per segment, not the obsolete one per MTU.
        for segment in dict.fromkeys(labels):
            mask = labels == segment
            positive = mask & (q0 > 0.0)
            if positive.any():
                factors = q_new[positive] / q0[positive]
                if not np.allclose(factors, factors[0], rtol=64 * np.finfo(float).eps, atol=0.0):
                    raise ValueError(f"{zone}/{mtu}/{segment}: within-segment ratios changed")
        rows.append({
            "zone": zone, "MTU": mtu, "carrier_alpha": alpha,
            "baseline_simple_supply_mw": total, "final_simple_supply_mw": math.fsum(q_new),
            "simple_supply_conservation_error_mw": math.fsum(q_new) - total,
            "simple_supply_exactly_conserved": True,
            "simple_sell_order_count": len(group),
            "positive_sell_order_count": int((q0 > 0.0).sum()),
            "redistribution_skipped_zero_total": total == 0.0,
            "redistribution_changed_volumes": bool(not np.array_equal(q0, q_new)),
            "numerical_baseline_fallback": fallback,
            "within_segment_relative_volumes_preserved": True,
        })
    if len(result) != len(baseline) or not result.drop(columns="Volume").equals(baseline.drop(columns="Volume")):
        raise ValueError("Reweighting changed order counts, prices or metadata")
    if not result.loc[~sell].equals(baseline.loc[~sell]):
        raise ValueError("Reweighting changed Purchase orders")
    if not result["Volume"].eq(0.0).equals(baseline["Volume"].eq(0.0)):
        raise ValueError("Reweighting changed the exact-zero filtering set")
    return result, pd.DataFrame(rows)


def remove_small_simple_sells(
    simple: pd.DataFrame,
    threshold_mw: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove small sells while conserving each zone/MTU's simple supply.

    Orders are marked with zero volume and later removed by filter_exact_zeros.
    If every positive order is below the threshold, the largest order is
    retained and receives the total. If that total is itself below the
    threshold, fail rather than silently lose supply or export a tiny offer.
    """
    if not np.isfinite(threshold_mw) or threshold_mw <= 0:
        raise ValueError("threshold_mw must be finite and positive")

    if not simple.index.is_unique:
        raise ValueError("Simple order index must be unique")

    finite(simple["Volume"], "Simple order volumes", nonnegative=True)
    result = simple.copy(deep=True)
    records = []

    sell = simple[SIDE].eq("Sell")
    groups = simple.loc[sell].groupby(
        ["bidding_zone", "MTU"], sort=False
    )

    for (zone, mtu), group in groups:
        q = group["Volume"].to_numpy(dtype=float)
        positive = q > 0.0
        small = positive & (q < threshold_mw)
        retained = positive & ~small

        target = _pandas_group_total(q)
        removed_count = int(small.sum())
        removed_mw = math.fsum(q[small])
        retained_largest = False

        if removed_count:
            if not retained.any():
                if target < threshold_mw:
                    raise ValueError(
                        f"{zone}, MTU {mtu}: total simple Sell supply "
                        f"{target:g} MW is below the minimum order size "
                        f"{threshold_mw:g} MW; cannot preserve both."
                    )
                largest = int(np.argmax(q))
                retained[largest] = True
                small[largest] = False
                removed_count = int(small.sum())
                removed_mw = math.fsum(q[small])
                retained_largest = True

            output = np.zeros_like(q)
            retained_total = math.fsum(q[retained])
            output[retained] = q[retained] * (target / retained_total)

            # Close floating-point rounding on the largest retained bid.
            # Fail explicitly if an exactly matching pandas aggregate cannot
            # be represented; never silently alter the physical budget.
            largest = int(np.argmax(output))
            for _ in range(128):
                actual = _pandas_group_total(output)
                if actual == target:
                    break
                updated = output[largest] + (target - actual)
                if updated == output[largest]:
                    updated = np.nextafter(
                        output[largest],
                        np.inf if actual < target else -np.inf,
                    )
                output[largest] = updated
            else:
                raise ValueError(
                    f"{zone}, MTU {mtu}: unable to conserve the exact "
                    "pandas simple-supply total after small-order removal"
                )

            if not np.isfinite(output).all():
                raise ValueError(f"{zone}, MTU {mtu}: non-finite volume")

            if (output[retained] < threshold_mw).any():
                raise ValueError(
                    f"{zone}, MTU {mtu}: retained Sell below threshold"
                )

            result.loc[group.index, "Volume"] = output

        records.append({
            "zone": zone,
            "MTU": mtu,
            "minimum_sell_volume_mw": threshold_mw,
            "removed_small_sell_count": removed_count,
            "removed_small_sell_volume_mw": removed_mw,
            "retained_largest_when_all_small": retained_largest,
            "simple_supply_before_mw": target,
            "simple_supply_after_mw": _pandas_group_total(
                result.loc[group.index, "Volume"]
            ),
        })

    if not result.loc[~sell].equals(simple.loc[~sell]):
        raise ValueError("Small-order filtering changed purchases")

    return result, pd.DataFrame(records)


def save_table(frame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def create_zonal_orderbooks(
    simple_input: str | Path, block_input: str | Path,
    network: pypsa.Network, zone_column: str,
    simple_output: str | Path, block_output: str | Path,
    supply_report_output: str | Path | None = None,
    demand_report_output: str | Path | None = None, *,
    original_curves_csv: str | Path | None = None,
    prices_csv: str | Path | None = None, delivery_date=None,
    diagnostics_dir: str | Path | None = None,
    scarcity_share: float | None = SCARCITY_SHARE,
    report_prefix: str | Path | None = None,
    network_input: str | Path | None = None,
    carrier_alpha: float = CARRIER_ALPHA,
    carrier_segment_mapping: dict | None = None,
    carrier_ratio_cap: float | None = CARRIER_RATIO_CAP,
    carrier_ratio_caps: dict[str, float | None] | None = None,
) -> None:
    """Keep baseline physical feasibility, then change only supply-curve shape."""
    validate_carrier_config(carrier_alpha, carrier_ratio_cap, carrier_ratio_caps)
    resolved_mapping = normalize_carrier_mapping(carrier_segment_mapping)
    if original_curves_csv is not None:
        warnings.warn("original_curves_csv is deprecated and ignored.", FutureWarning, stacklevel=2)
    if scarcity_share is not None:
        warnings.warn("scarcity_share is obsolete; segments do not allocate physical capacity.", FutureWarning, stacklevel=2)
        if not np.isfinite(scarcity_share) or scarcity_share != 0.0:
            raise ValueError("Nonzero scarcity_share is unsupported under aggregate capacity.")
    price_path = prices_csv if prices_csv is not None else PRICES_CSV
    if price_path is None:
        raise ValueError("Configure prices_csv for historical calibration")
    simple_layout, day = load_simple_orders(simple_input, delivery_date)
    simple = simple_layout.frame
    simple_columns = [c for c in simple if c != "MTU"]
    block_layout, profiles = load_block_orders(block_input, day)
    blocks = block_layout.frame
    zones = normalize_network_zones(network, zone_column)
    load, capacity, available, available_by_carrier = network_zonal_data(network, zone_column, zones, day)
    reference_generation, reference_diagnostics = reference_generation_targets(load, available)
    infeasible = reference_diagnostics.loc[~reference_diagnostics["reference_generation_input_feasible"]]
    if not infeasible.empty:
        LOG.warning("Reference-generation inputs are infeasible at MTUs: %s", "|".join(infeasible["MTU"].astype(str)))
    # Include the existing supply proxy universe, even when some price bands
    # are absent from this day's template. Do not recalculate Segment labels.
    segments = list(dict.fromkeys([
        *resolved_mapping.keys(),
        *simple.loc[simple[SIDE].eq("Sell"), "Segment"].tolist(), SCARCITY_SEGMENT,
    ]))
    weights, shares, system_shares, carrier_diagnostics = carrier_segment_weights(
        available_by_carrier, zones, segments, alpha=carrier_alpha,
        mapping=resolved_mapping, ratio_cap=carrier_ratio_cap, ratio_caps=carrier_ratio_caps,
    )
    mcp = read_mcp(price_path, day)
    historical_demand, anchor = matched_demand(simple, blocks, profiles, mcp)
    matched_simple = anchor.set_index("MTU")["M_simple_mw"]
    historical_buy_blocks = anchor.set_index("MTU")["B_historical_mw"]
    original_sell_envelope = block_totals(blocks, profiles, "Sell")
    original_simple_sell = simple_totals(simple, "Sell")
    historical_matched_simple_sell = matched_simple_supply(simple, mcp)
    historical_sell_blocks = historical_accepted_block_totals(blocks, profiles, "Sell")
    simple_frames, block_frames, anchors = [], [], []
    demand_rows, supply_rows, daily_rows = [], [], []
    scaling_rows, event_rows, regression_rows, redistribution_frames = [], [], [], []
    unchanged_simple_columns = [c for c in simple_columns if c not in ["Volume", "bidding_zone"]]
    small_sell_frames = []

    for zone in zones:
        (k, buy_factors, sell_factors, requested_sell_factors, capacity_sell_factors,
         no_eligible_sell_template, capacity_limited_simple_supply, target_exceeds_available, daily) = zone_scaling(
            load=load[zone], reference_generation=reference_generation[zone], available=available[zone],
            matched_simple_purchase=matched_simple, historical_buy_blocks=historical_buy_blocks,
            matched_simple_sell=historical_matched_simple_sell, historical_sell_blocks=historical_sell_blocks,
            gross_simple_sell=original_simple_sell, sell_block_envelope=original_sell_envelope, zone=zone,
        )
        zonal_blocks = scale_blocks(blocks, profiles, zone, k, daily["initial_common_block_factor"])
        baseline_simple = simple.copy()
        baseline_simple["bidding_zone"] = zone
        buy, sell = baseline_simple[SIDE].eq("Purchase"), baseline_simple[SIDE].eq("Sell")
        baseline_simple.loc[buy, "Volume"] *= baseline_simple.loc[buy, "MTU"].map(buy_factors)
        baseline_simple.loc[sell, "Volume"] *= baseline_simple.loc[sell, "MTU"].map(sell_factors)
        finite(baseline_simple["Volume"], f"{zone}: baseline simple magnitudes", nonnegative=True)
        baseline_blocks = zonal_blocks.copy(deep=True)
        baseline_simple_sell = simple_totals(baseline_simple, "Sell")
        baseline_matched_simple_sell = matched_simple_supply(baseline_simple, mcp)
        require_close(baseline_simple_sell, sell_factors * original_simple_sell, f"{zone}: baseline gross simple scaling")
        require_close(baseline_matched_simple_sell, sell_factors * historical_matched_simple_sell, f"{zone}: baseline MCP-eligible simple scaling")
        zonal_simple, redistribution = reweight_simple_sell_orders(baseline_simple, weights, alpha=carrier_alpha)
        redistribution_frames.append(redistribution)
        finite(zonal_simple["Volume"], f"{zone}: final simple magnitudes", nonnegative=True)
        if not zonal_blocks.equals(baseline_blocks):
            raise ValueError(f"{zone}: carrier reweighting changed blocks")
        if not zonal_simple[unchanged_simple_columns].equals(simple[unchanged_simple_columns]):
            raise ValueError(f"{zone}: simple metadata changed")
        zonal_simple, small_sell_report = remove_small_simple_sells(
            zonal_simple,
            threshold_mw=0.001,
        )
        small_sell_frames.append(small_sell_report)
        block_checks = validate_blocks(blocks, zonal_blocks, profiles, zone)
        zero_blocks = zonal_blocks[profiles].eq(0.0).all(axis=1)
        for record, zero in zip(block_checks, zero_blocks):
            record.update({
                "common_zone_day_factor": k, "common_factor_capped": daily["common_block_factor_capped"],
                "exported": not bool(zero), "omission_reason": "exact_zero_after_common_scaling" if zero else "",
            })
        scaling_rows.extend(block_checks)
        sell_envelope = block_totals(zonal_blocks, profiles, "Sell")
        require_close(sell_envelope, k * original_sell_envelope, f"{zone}: uniformly scaled sell-block envelope")
        final_simple_sell = simple_totals(zonal_simple, "Sell")
        # Replace obsolete final uniform-simple-scaling validation with exact
        # baseline conservation. Only baseline scaling remains uniform per MTU.
        if not np.array_equal(final_simple_sell.to_numpy(), baseline_simple_sell.to_numpy()):
            raise ValueError(f"{zone}: final simple supply totals differ from baseline")
        final_supply_budget = final_simple_sell + sell_envelope
        baseline_supply_budget = baseline_simple_sell + sell_envelope
        if not np.array_equal(final_supply_budget.to_numpy(), baseline_supply_budget.to_numpy()):
            raise ValueError(f"{zone}: physical budget changed during reweighting")
        require_within(final_supply_budget, available[zone], f"{zone}: aggregate submitted supply budget")
        benchmark_demand, _ = matched_demand(zonal_simple, zonal_blocks, profiles, mcp)
        require_close(benchmark_demand, buy_factors * matched_simple + k * historical_buy_blocks, f"{zone}: analytical benchmark demand")
        accepted_buy_blocks = historical_accepted_block_totals(zonal_blocks, profiles, "Purchase")
        accepted_sell_blocks = historical_accepted_block_totals(zonal_blocks, profiles, "Sell")
        matched_simple_buy = benchmark_demand - accepted_buy_blocks
        final_matched_simple_sell = matched_simple_supply(zonal_simple, mcp)
        baseline_benchmark_supply = baseline_matched_simple_sell + accepted_sell_blocks
        benchmark_supply = final_matched_simple_sell + accepted_sell_blocks
        mcp_supply_change = final_matched_simple_sell - baseline_matched_simple_sell
        gross_simple_buy = simple_totals(zonal_simple, "Purchase")
        gross_buy_blocks = block_totals(zonal_blocks, profiles, "Purchase", effective_only=False)
        gross_sell_blocks = block_totals(zonal_blocks, profiles, "Sell", effective_only=False)
        gross_supply = final_simple_sell + gross_sell_blocks
        baseline_gross_supply = baseline_simple_sell + gross_sell_blocks
        if not np.array_equal(gross_supply.to_numpy(), baseline_gross_supply.to_numpy()):
            raise ValueError(f"{zone}: gross offered supply changed")
        demand_residual = load[zone] - benchmark_demand
        baseline_supply_residual = reference_generation[zone] - baseline_benchmark_supply
        supply_residual = reference_generation[zone] - benchmark_supply
        require_close(supply_residual, baseline_supply_residual - mcp_supply_change, f"{zone}: separate MCP supply residual accounting")
        physical_headroom = available[zone] - final_supply_budget
        demand_exact = bool(np.allclose(benchmark_demand, load[zone], atol=TOL, rtol=RTOL))
        baseline_supply_exact = bool(np.allclose(baseline_benchmark_supply, reference_generation[zone], atol=TOL, rtol=RTOL))
        post_supply_exact = bool(np.allclose(benchmark_supply, reference_generation[zone], atol=TOL, rtol=RTOL))
        redistribution_by_mtu = redistribution.set_index("MTU") if not redistribution.empty else pd.DataFrame(index=MTUS)
        for mtu in MTUS:
            demand_rows.append({
                "zone": zone, "MTU": mtu, "mcp_eur_mwh": mcp.at[mtu], "pypsa_load_mw": load.at[mtu, zone],
                "original_M_simple_mw": matched_simple.at[mtu], "original_B_historical_mw": historical_buy_blocks.at[mtu],
                "original_matched_demand_mw": historical_demand.at[mtu],
                "initial_common_block_factor": daily["initial_common_block_factor"], "common_block_factor": k,
                "common_block_factor_capped": daily["common_block_factor_capped"],
                "simple_purchase_factor": buy_factors.at[mtu], "benchmark_matched_simple_mw": matched_simple_buy.at[mtu],
                "benchmark_accepted_purchase_block_mw": accepted_buy_blocks.at[mtu],
                "benchmark_matched_demand_mw": benchmark_demand.at[mtu], "demand_residual_mw": demand_residual.at[mtu],
                "demand_residual_mwh": demand_residual.at[mtu] * QUARTER_HOURS,
                "demand_anchor_error_mw": -demand_residual.at[mtu],
                "calibration_status": calibration_status(demand_residual.at[mtu], load.at[mtu, zone], matched_simple.at[mtu] == 0.0, "Purchase"),
                "gross_submitted_simple_purchase_mw": gross_simple_buy.at[mtu],
                "gross_submitted_purchase_block_mw": gross_buy_blocks.at[mtu],
                "gross_submitted_demand_mw": gross_simple_buy.at[mtu] + gross_buy_blocks.at[mtu],
                "carrier_reweighting_applied": False, "new_clearing_acceptance_guaranteed": False,
            })
            baseline_status = calibration_status(
                baseline_supply_residual.at[mtu], reference_generation.at[mtu, zone],
                bool(no_eligible_sell_template.at[mtu]), "Sell", target_exceeds_available=bool(target_exceeds_available.at[mtu]),
            )
            post_exact = abs(supply_residual.at[mtu]) <= TOL + RTOL * abs(reference_generation.at[mtu, zone])
            changed_mcp = abs(mcp_supply_change.at[mtu]) > TOL + RTOL * abs(baseline_benchmark_supply.at[mtu])
            post_status = "calibrated" if post_exact else ("post_carrier_reweighting_residual" if changed_mcp else baseline_status)
            supply_rows.append({
                "zone": zone, "MTU": mtu, "mcp_eur_mwh": mcp.at[mtu],
                "installed_capacity_mw": capacity[zone], "available_generation_mw": available.at[mtu, zone],
                "system_load_mw": reference_diagnostics.at[mtu - 1, "system_load_mw"],
                "system_available_generation_mw": reference_diagnostics.at[mtu - 1, "system_available_generation_mw"],
                "target_supply_mw": reference_generation.at[mtu, zone], "reference_generation_mw": reference_generation.at[mtu, zone],
                "reference_generation_exceeds_available_mw": max(0.0, reference_generation.at[mtu, zone] - available.at[mtu, zone]),
                "reference_generation_within_available": bool(not target_exceeds_available.at[mtu]),
                "original_simple_sell_template_mw": original_simple_sell.at[mtu],
                "original_mcp_eligible_simple_sell_template_mw": historical_matched_simple_sell.at[mtu],
                "original_sell_block_envelope_mw": original_sell_envelope.at[mtu],
                "original_aar_weighted_sell_block_mw": historical_sell_blocks.at[mtu],
                "common_block_factor": k, "common_block_factor_capped": daily["common_block_factor_capped"],
                "sell_block_budget_mw": sell_envelope.at[mtu],
                "benchmark_accepted_sell_block_mw": accepted_sell_blocks.at[mtu],
                "remaining_reference_generation_before_simple_mw": max(0.0, reference_generation.at[mtu, zone] - accepted_sell_blocks.at[mtu]),
                "baseline_requested_simple_sell_factor": requested_sell_factors.at[mtu],
                "baseline_capacity_limited_simple_sell_factor": capacity_sell_factors.at[mtu],
                "baseline_simple_sell_factor": sell_factors.at[mtu],
                "baseline_simple_sell_factor_capacity_limited": bool(capacity_limited_simple_supply.at[mtu]),
                "mcp_eligible_simple_sell_template_missing": bool(no_eligible_sell_template.at[mtu]),
                "carrier_alpha": carrier_alpha,
                "baseline_simple_supply_mw": baseline_simple_sell.at[mtu],
                "baseline_mcp_eligible_simple_supply_mw": baseline_matched_simple_sell.at[mtu],
                "post_reweighting_mcp_eligible_simple_supply_mw": final_matched_simple_sell.at[mtu],
                "mcp_eligible_supply_change_mw": mcp_supply_change.at[mtu],
                "baseline_mcp_eligible_supply_mw": baseline_benchmark_supply.at[mtu],
                "post_reweighting_mcp_eligible_supply_mw": benchmark_supply.at[mtu],
                "benchmark_matched_simple_sell_mw": final_matched_simple_sell.at[mtu],
                "benchmark_matched_supply_mw": benchmark_supply.at[mtu],
                "final_simple_supply_mw": final_simple_sell.at[mtu],
                "simple_supply_conservation_error_mw": final_simple_sell.at[mtu] - baseline_simple_sell.at[mtu],
                "simple_supply_exactly_conserved": True,
                "baseline_supply_budget_mw": baseline_supply_budget.at[mtu],
                "final_supply_budget_mw": final_supply_budget.at[mtu],
                "baseline_calibration_residual_mw": baseline_supply_residual.at[mtu],
                "baseline_calibration_residual_mwh": baseline_supply_residual.at[mtu] * QUARTER_HOURS,
                "post_reweighting_calibration_residual_mw": supply_residual.at[mtu],
                "post_reweighting_calibration_residual_mwh": supply_residual.at[mtu] * QUARTER_HOURS,
                "supply_residual_mw": supply_residual.at[mtu], "supply_residual_mwh": supply_residual.at[mtu] * QUARTER_HOURS,
                "physical_headroom_mw": max(0.0, physical_headroom.at[mtu]),
                "baseline_calibration_status": baseline_status,
                "post_reweighting_supply_status": post_status, "calibration_status": post_status,
                "recalibrated_after_carrier_reweighting": False,
                "numerical_baseline_fallback": bool(redistribution_by_mtu.at[mtu, "numerical_baseline_fallback"]) if mtu in redistribution_by_mtu.index and "numerical_baseline_fallback" in redistribution_by_mtu else False,
                "gross_submitted_sell_block_mw": gross_sell_blocks.at[mtu],
                "baseline_gross_submitted_supply_mw": baseline_gross_supply.at[mtu],
                "gross_submitted_supply_mw": gross_supply.at[mtu], "gross_supply_exactly_conserved": True,
            })
        reference_excess = (reference_generation[zone] - available[zone]).clip(lower=0.0)
        fallback_count = int(redistribution["numerical_baseline_fallback"].sum()) if not redistribution.empty else 0
        daily.update({
            "installed_capacity_mw": capacity[zone],
            "available_generation_energy_mwh": float(available[zone].sum() * QUARTER_HOURS),
            "original_simple_sell_template_energy_mwh": float(original_simple_sell.sum() * QUARTER_HOURS),
            "original_sell_block_envelope_energy_mwh": float(original_sell_envelope.sum() * QUARTER_HOURS),
            "demand_calibration_exact": demand_exact,
            "baseline_supply_calibration_exact": baseline_supply_exact,
            "post_reweighting_supply_calibration_exact": post_supply_exact,
            "carrier_alpha": carrier_alpha, "simple_supply_exactly_conserved": True,
            "gross_supply_exactly_conserved": True, "physical_supply_budget_exactly_conserved": True,
            "numerical_baseline_fallback_mtu_count": fallback_count,
            "baseline_mcp_eligible_supply_energy_mwh": float(baseline_benchmark_supply.sum() * QUARTER_HOURS),
            "post_reweighting_mcp_eligible_supply_energy_mwh": float(benchmark_supply.sum() * QUARTER_HOURS),
            "mcp_eligible_supply_change_energy_mwh": float(mcp_supply_change.sum() * QUARTER_HOURS),
            "baseline_max_absolute_supply_residual_mw": float(baseline_supply_residual.abs().max()),
            "max_absolute_supply_residual_mw": float(supply_residual.abs().max()),
            "baseline_supply_unfilled_energy_mwh": float(baseline_supply_residual.clip(lower=0.0).sum() * QUARTER_HOURS),
            "baseline_supply_overshoot_energy_mwh": float((-baseline_supply_residual).clip(lower=0.0).sum() * QUARTER_HOURS),
            "supply_unfilled_energy_mwh": float(supply_residual.clip(lower=0.0).sum() * QUARTER_HOURS),
            "supply_overshoot_energy_mwh": float((-supply_residual).clip(lower=0.0).sum() * QUARTER_HOURS),
            "demand_shortfall_energy_mwh": float(demand_residual.clip(lower=0.0).sum() * QUARTER_HOURS),
            "demand_overshoot_energy_mwh": float((-demand_residual).clip(lower=0.0).sum() * QUARTER_HOURS),
            "physical_unused_headroom_energy_mwh": float(physical_headroom.clip(lower=0.0).sum() * QUARTER_HOURS),
            "mcp_eligible_simple_sell_template_missing_mtu_count": int(no_eligible_sell_template.sum()),
            "max_absolute_demand_residual_mw": float(demand_residual.abs().max()),
            "exact_zero_block_count": int(zero_blocks.sum()),
            "block_budget_basis": "independent_sum_plus_per_mtu_exclusive_family_maxima",
            "reference_generation_exceeds_availability_energy_mwh": float(reference_excess.sum() * QUARTER_HOURS),
            "reference_generation_exceeds_availability_mtu_count": int(reference_excess.gt(TOL).sum()),
            "dt_hours": QUARTER_HOURS,
        })
        daily_rows.append(daily)
        if daily["common_block_factor_capped"]:
            event_rows.append({
                "zone": zone, "scaling_unit": "all_zone_day_blocks", "block_ids": "|".join(zonal_blocks["BlockID"]),
                "segments": "|".join(sorted(zonal_blocks["Segment"].astype(str).unique())),
                "initial_common_block_factor": daily["initial_common_block_factor"], "final_common_block_factor": k,
                "physical_reduction_factor": daily["common_physical_reduction_factor"],
                "binding_mtus": daily["block_cap_binding_mtus"], "reason": "aggregate_sell_block_availability",
            })
            LOG.warning("%s: common block factor capped from %g to %g; binding MTUs=%s", zone, daily["initial_common_block_factor"], k, daily["block_cap_binding_mtus"])
        if not demand_exact or not baseline_supply_exact:
            LOG.warning("%s: baseline best-effort calibration; max demand residual=%g MW, max supply residual=%g MW", zone, demand_residual.abs().max(), baseline_supply_residual.abs().max())
        LOG.info("%s: carrier reweighting MCP-eligible supply change=%g MWh; post residual max=%g MW; numerical baseline fallbacks=%d MTUs", zone, mcp_supply_change.sum() * QUARTER_HOURS, supply_residual.abs().max(), fallback_count)
        zone_anchor = anchor.copy()
        zone_anchor.insert(0, "zone", zone)
        anchors.append(zone_anchor)
        regression_rows.append({
            "zone": zone, "demand_calibration_valid": demand_exact,
            "baseline_mcp_eligible_supply_calibration_exact": baseline_supply_exact,
            "post_reweighting_mcp_eligible_supply_calibration_exact": post_supply_exact,
            "reference_generation_capacity_feasible": bool((~target_exceeds_available).all()),
            "baseline_simple_sell_capacity_limited_mtu_count": int(capacity_limited_simple_supply.sum()),
            "mcp_eligible_simple_sell_template_missing_mtu_count": int(no_eligible_sell_template.sum()),
            "simple_sell_totals_exactly_conserved": True, "gross_supply_exactly_conserved": True,
            "physical_supply_budget_exactly_conserved": True,
            "within_segment_relative_volumes_valid": True, "positive_simple_offers_preserved_after_carrier_reweighting": True,
            "small_positive_sells_intentionally_removed": True,
            "purchases_unchanged_from_baseline": True, "blocks_unchanged_from_baseline": True,
            "small_sell_orders_removed": int(small_sell_report["removed_small_sell_count"].sum()),
            "small_sell_volume_redistributed_mwh": float(small_sell_report["removed_small_sell_volume_mw"].sum() * QUARTER_HOURS),
            "order_counts_unchanged_from_baseline": False,
            "exact_zero_filtering_set_unchanged": False,
            "aggregate_availability_limits_valid": True, "nonnegative_magnitudes_valid": True,
            "simple_metadata_valid": True, "block_metadata_relationships_profiles_valid": True,
            "common_block_multiplier_valid": True,
            "baseline_best_effort_used": not (demand_exact and baseline_supply_exact),
            "historical_supply_recalibration_after_reweighting": False,
            "max_absolute_demand_anchor_error_mw": float(demand_residual.abs().max()),
            "baseline_max_absolute_supply_residual_mw": float(baseline_supply_residual.abs().max()),
            "post_reweighting_max_absolute_supply_residual_mw": float(supply_residual.abs().max()),
            "numerical_baseline_fallback_mtu_count": fallback_count,
        })
        simple_frames.append(zonal_simple)
        block_frames.append(zonal_blocks)

    zonal_simple = pd.concat(simple_frames, ignore_index=True)
    zonal_blocks = pd.concat(block_frames, ignore_index=True)
    zonal_simple, zonal_blocks = filter_exact_zeros(zonal_simple, zonal_blocks, profiles)
    diagnostic_path = Path(diagnostics_dir) if diagnostics_dir is not None else Path(simple_output).parent
    demand_report, supply_report, daily_report = pd.DataFrame(demand_rows), pd.DataFrame(supply_rows), pd.DataFrame(daily_rows)
    event_columns = ["zone", "scaling_unit", "block_ids", "segments", "initial_common_block_factor", "final_common_block_factor", "physical_reduction_factor", "binding_mtus", "reason"]
    carrier_availability_rows = [
        {"zone": zone, "carrier": carrier, "MTU": mtu, "available_generation_mw": available_by_carrier.at[mtu, (zone, carrier)]}
        for zone, carrier in available_by_carrier.columns for mtu in MTUS
    ]
    diagnostics = {
        "matched_demand_diagnostics.csv": pd.concat(anchors, ignore_index=True),
        "reference_generation_diagnostics.csv": reference_diagnostics,
        "demand_daily_factors.csv": daily_report, "demand_scaling_diagnostics.csv": demand_report,
        "supply_daily_factors.csv": daily_report, "supply_availability_diagnostics.csv": supply_report,
        "block_scaling_diagnostics.csv": pd.DataFrame(scaling_rows),
        "block_downscaling_events.csv": pd.DataFrame(event_rows, columns=event_columns),
        "regression_checks.csv": pd.DataFrame(regression_rows),
        "carrier_availability_diagnostics.csv": pd.DataFrame(carrier_availability_rows, columns=["zone", "carrier", "MTU", "available_generation_mw"]),
        "carrier_segment_reweighting_diagnostics.csv": carrier_diagnostics,
        "simple_supply_redistribution_diagnostics.csv": pd.concat(redistribution_frames, ignore_index=True),
        "small_simple_sell_diagnostics.csv": pd.concat(small_sell_frames, ignore_index=True),
    }
    for filename, table in diagnostics.items():
        save_table(table, diagnostic_path / filename)
    if supply_report_output is not None:
        save_table(supply_report, supply_report_output)
    if demand_report_output is not None:
        save_table(demand_report, demand_report_output)
    manifest = pd.DataFrame([{
        "delivery_date": str(day.date()), "simple_input": str(simple_input), "block_input": str(block_input),
        "prices_csv": str(price_path), "network_input": str(network_input or "in-memory"), "zone_column": zone_column,
        "simple_output": str(simple_output), "block_output": str(block_output), "diagnostics_dir": str(diagnostic_path),
        "transformation": "baseline_calibration_then_conservative_carrier_simple_sell_redistribution",
        "demand_anchor": "simple_price_ge_mcp_plus_purchase_block_aar",
        "block_scaling": "one_common_zone_day_factor_for_both_sides",
        "block_factor_cap": "aggregate_sell_block_availability_only",
        "simple_purchase_scaling": "per_mtu_nonnegative_matched_demand_residual",
        "baseline_simple_sell_scaling": "per_mtu_mcp_eligible_supply_residual_with_gross_supply_availability_cap",
        "final_simple_sell_scaling": "baseline_times_positive_segment_weight_then_zone_mtu_total_normalization",
        "final_uniform_simple_sell_scaling_required": False,
        "supply_target": "system_load_distributed_by_zonal_available_generation",
        "baseline_supply_anchor": "simple_sell_price_le_mcp_plus_sell_block_aar",
        "supply_capacity_check": "gross_simple_sell_plus_conservative_sell_block_envelope",
        "installed_capacity_attribute": "p_nom", "availability_attribute": "snapshot_matched_p_max_pu_static_fallback",
        "carrier_availability_formula": "p_nom_times_p_max_pu_grouped_by_zone_carrier_mtu",
        "carrier_attribute": "carrier_with_Type_fallback_for_blank_labels",
        "supply_block_budget": "independent_sum_plus_per_mtu_exclusive_family_maxima",
        "segment_capacity_budgets_used": False,
        "segment_labels": "preprocessed_price_proxies_for_soft_carrier_reweighting",
        "carrier_alpha": carrier_alpha, "carrier_ratio_cap": carrier_ratio_cap,
        "carrier_ratio_caps_json": json.dumps(carrier_ratio_caps or {}, sort_keys=True),
        "carrier_segment_mapping_json": json.dumps(resolved_mapping, sort_keys=True),
        "carrier_mapping_normalization": "normalize_coefficients_across_segments_for_each_carrier",
        "unmapped_carriers": "uniform_across_non_scarcity_price_proxies",
        "segment_share_normalization": "proxy_available_mw_over_total_zonal_available_mw",
        "system_segment_shares": "pooled_proxy_available_mw_over_total_system_available_mw",
        "carrier_weight_formula": "(1-alpha)+alpha*c_zonal/c_system",
        "undefined_availability_or_zero_system_share_ratio": 1.0,
        "scarcity_carrier_required": False, "scarcity_weight": 1.0,
        "simple_sell_redistribution_formula": "q0_i*w_segment*sum(q0)/sum(q0*w_segment)",
        "numerical_closure": "whole_segment_multiplier_ulps_or_reported_baseline_fallback",
        "volume_conservation_checks": "exact_math_fsum_and_exact_baseline_pandas_groupby_sum",
        "total_simple_supply_conserved_per_zone_mtu": True, "gross_supply_conserved_per_zone_mtu": True,
        "physical_supply_budget_conserved_per_zone_mtu": True,
        "purchase_orders_unchanged_from_baseline": True, "all_blocks_unchanged_from_baseline": True,
        "historical_supply_recalibration_after_reweighting": False,
        "mcp_supply_report": "separate_baseline_post_reweighting_change_and_residual_columns",
        "linked_family_removal_used": False, "historical_anchor_recalculated": False,
        "unattainable_targets": "write_capacity_feasible_baseline_and_report_baseline_and_post_residuals",
        "residual_convention": "target_minus_achieved",
        "simple_bids_exactly_at_mcp": "fully_accepted_approximation", "historical_aar": "calibration_only",
        "local_balance_enforced": False, "new_clearing_acceptance_guaranteed": False,
        "native_block_signs_preserved": True, "safe_exact_zero_filtering_valid": True,
        "dt_hours": QUARTER_HOURS, "raw_epex_volume_unit": "MWh",
        "intermediate_orderbook_quantity_unit": "MW", "network_power_unit": "MW", "daily_energy_unit": "MWh",
        "reference_generation_formula": "system_load_times_zonal_available_generation_over_system_available_generation",
        "reference_generation_infeasibility": "reported_in_reference_generation_diagnostics",
        "physical_supply_constraint": "aggregate_zonal_available_generation",
        "simple_order_counts_unchanged_from_baseline": False,
        "minimum_simple_sell_volume_mw": 0.001,
        "small_simple_sells": "removed_and_redistributed_per_zone_mtu",
    }])
    save_table(manifest, diagnostic_path / "manifest.csv")
    if report_prefix is not None:
        save_table(manifest, f"{report_prefix}_manifest.csv")
    simple_layout.frame = zonal_simple[simple_columns].sort_values(
        ["bidding_zone", "Date", "Hour", "Quarter hour", SIDE, "Price"], kind="stable",
    )
    write_simple_layout(simple_layout, simple_output)
    exported_blocks = zonal_blocks.copy()
    exported_blocks[profiles] = exported_blocks[profiles].mul(exported_blocks["_sign"], axis=0)
    finite(exported_blocks[profiles], "Exported signed block profiles")
    block_layout.frame = exported_blocks.reindex(columns=block_layout.header)
    write_block_layout(block_layout, block_output)
    LOG.info("Wrote %d simple orders and %d block orders", len(simple_layout.frame), len(block_layout.frame))
    LOG.info("Diagnostics written to %s", diagnostic_path)


def run_self_tests():
    """Small in-script smoke test; the bundled unittest file is comprehensive."""
    columns = pd.MultiIndex.from_product([["Z1", "Z2"], ["Wind", "Gas/CCGT"]], names=["zone", "carrier"])
    availability = pd.DataFrame([80.0, 20.0, 20.0, 80.0], index=columns).T
    availability = pd.DataFrame(np.repeat(availability.to_numpy(), 96, axis=0), index=MTUS, columns=columns)
    segments = ["S0_NEGATIVE", "S3_MID_MERIT", SCARCITY_SEGMENT]
    mapping = {"S0_NEGATIVE": ["Wind"], "S3_MID_MERIT": ["Gas/CCGT"], SCARCITY_SEGMENT: {}}
    weights, _, _, _ = carrier_segment_weights(availability, ["Z1", "Z2"], segments, alpha=0.5, mapping=mapping)
    source = pd.DataFrame({SIDE: ["Sell", "Sell", "Sell", "Purchase"], "bidding_zone": ["Z1"] * 4,
                           "MTU": [1] * 4, "Segment": [*segments, "D3_INELASTIC"],
                           "Volume": [10.0, 20.0, 5.0, 30.0], "Price": [-1.0, 130.0, 300.0, 1000.0]})
    final, diagnostics = reweight_simple_sell_orders(source, weights, alpha=0.5)
    if math.fsum(final.loc[final[SIDE].eq("Sell"), "Volume"]) != 35.0:
        raise ValueError("Self-test conservation failed")
    if not final.loc[final[SIDE].eq("Purchase")].equals(source.loc[source[SIDE].eq("Purchase")]):
        raise ValueError("Self-test purchase preservation failed")
    neutral, _ = reweight_simple_sell_orders(source, weights, alpha=0.0)
    if not neutral.equals(source):
        raise ValueError("Self-test alpha=0 failed")
    if not (final["Volume"] > 0.0).all() or not diagnostics["simple_supply_exactly_conserved"].all():
        raise ValueError("Self-test positive supply failed")
    LOG.info("Carrier reweighting smoke tests passed")


def read_json_object(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    positional = ["simple_input", "block_input", "network", "simple_output", "block_output"]
    for name in positional:
        parser.add_argument(name, type=Path, nargs="?")
    parser.add_argument("--zone-column", default="zone")
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV, help="Historical MCP CSV required for baseline calibration")
    parser.add_argument("--delivery-date", default=None, help="ISO delivery date; DST days unsupported")
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--supply-report", type=Path, default=None)
    parser.add_argument("--demand-report", type=Path, default=None)
    parser.add_argument("--report-prefix", type=Path, default=None)
    parser.add_argument("--original-curves-csv", type=Path, default=ORIGINAL_CURVES_CSV, help="Deprecated and ignored")
    parser.add_argument("--scarcity-share", type=float, default=SCARCITY_SHARE, help="Obsolete; only zero is accepted; scarcity has no dedicated carrier")
    parser.add_argument("--carrier-alpha", type=float, default=CARRIER_ALPHA, help="Soft reweighting strength, 0 <= alpha < 1; zero reproduces baseline")
    parser.add_argument("--carrier-segment-mapping", type=Path, help="JSON segment -> {carrier: score} or segment -> [carriers]; replaces default mapping")
    parser.add_argument("--carrier-ratio-cap", type=float, default=CARRIER_RATIO_CAP, help="Global upper share-ratio cap >= 1 (default 5)")
    parser.add_argument("--no-carrier-ratio-cap", action="store_true", help="Disable the global ratio cap")
    parser.add_argument("--carrier-ratio-caps", type=Path, help="JSON per-segment cap overrides; null disables a segment's cap")
    parser.add_argument("--self-test", action="store_true", help="Run smoke tests; positional inputs may be omitted")
    args = parser.parse_args()
    provided = [getattr(args, name) is not None for name in positional]
    if not all(provided) and not (args.self_test and not any(provided)):
        parser.error("Supply all five positional paths, or use --self-test alone")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        if args.simple_input is None:
            return
    create_zonal_orderbooks(
        simple_input=args.simple_input, block_input=args.block_input,
        network=pypsa.Network(str(args.network)), zone_column=args.zone_column,
        simple_output=args.simple_output, block_output=args.block_output,
        supply_report_output=args.supply_report, demand_report_output=args.demand_report,
        original_curves_csv=args.original_curves_csv, prices_csv=args.prices_csv,
        delivery_date=args.delivery_date, diagnostics_dir=args.diagnostics_dir,
        scarcity_share=args.scarcity_share, report_prefix=args.report_prefix, network_input=args.network,
        carrier_alpha=args.carrier_alpha,
        carrier_segment_mapping=read_json_object(args.carrier_segment_mapping) if args.carrier_segment_mapping else None,
        carrier_ratio_cap=None if args.no_carrier_ratio_cap else args.carrier_ratio_cap,
        carrier_ratio_caps=read_json_object(args.carrier_ratio_caps) if args.carrier_ratio_caps else None,
    )


if __name__ == "__main__":
    main()
