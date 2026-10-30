"""Create zonal native-layout order books from preprocessed published bids.

Shared layout I/O is imported from ``tools.order_layout``. This module only
contains zoning, scaling and validation logic.
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

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
DEMAND_OFFER_FACTOR = 1.1

SEGMENT_TYPES = {
    "S0_NEGATIVE": ["Wind", "Wind Offshore", "PV"],
    "S1_LOW_COST": ["Nuclear"],
    "S2_LIGNITE": ["Lignite"],
    "S3_MID_MERIT": ["Hard Coal"],
    "S4_HIGH_COST": ["Gas/CCGT"],
    "S5_SCARCITY": ["Nuclear", "Lignite", "Hard Coal", "Gas/CCGT"],
}

DEMAND_FLEXIBILITY_FACTORS: dict[str, dict[str, float]] = {}

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger(__name__)


def normalize_zone_label(value) -> str:
    """Return a zone label that cannot be mistaken for a numerical bus ID."""
    value = str(value).strip()
    if not value:
        raise ValueError("Empty bidding-zone label")
    try:
        return f"Z{int(float(value))}"
    except ValueError:
        return value


def normalize_network_zones(network: pypsa.Network, zone_column: str) -> list[str]:
    """Normalise network zone values in place and return its zone labels."""
    if zone_column not in network.buses.columns:
        raise ValueError(f"Network buses do not contain '{zone_column}'")
    if network.buses[zone_column].isna().any():
        raise ValueError("Every network bus must have a bidding-zone value")

    network.buses[zone_column] = network.buses[zone_column].map(normalize_zone_label)
    return sorted(network.buses[zone_column].unique(), key=str)


def demand_flexibility(zone: str, segment: str) -> float:
    """Return configurable demand flexibility; 1.0 is the generic default."""
    return DEMAND_FLEXIBILITY_FACTORS.get(zone, {}).get(segment, 1.0)


def network_zonal_data(
    network: pypsa.Network,
    zone_column: str,
    zones: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    """Derive 96-MTU zonal load and segment capacity from a PyPSA network."""
    if len(network.snapshots) != 96:
        raise ValueError(
            "The supplied network must contain 96 delivery-day snapshots; "
            f"found {len(network.snapshots)}."
        )
    if "carrier" not in network.generators.columns:
        raise ValueError("Network generators require a 'carrier' column")

    bus_zone = network.buses[zone_column]
    generator_zone = network.generators["bus"].map(bus_zone)
    if generator_zone.isna().any():
        raise ValueError("At least one generator is connected to an unknown bus")

    p_nom = pd.to_numeric(network.generators["p_nom"], errors="coerce").fillna(0.0)
    capacity: dict[tuple[str, str], float] = {}
    for zone in zones:
        for segment, carriers in SEGMENT_TYPES.items():
            mask = generator_zone.eq(zone) & network.generators["carrier"].isin(carriers)
            capacity[(zone, segment)] = float(p_nom.loc[mask].sum())

    load_names = network.loads.index
    if len(load_names) == 0:
        return pd.DataFrame(0.0, index=MTUS, columns=zones), capacity

    load_zone = network.loads["bus"].map(bus_zone)
    if load_zone.isna().any():
        raise ValueError("At least one load is connected to an unknown bus")

    static_values = pd.to_numeric(network.loads["p_set"], errors="coerce").fillna(0.0)
    static_load = pd.DataFrame(
        [static_values.to_numpy()] * len(network.snapshots),
        index=network.snapshots,
        columns=load_names,
    )
    dynamic_load = network.loads_t.p_set.reindex(
        index=network.snapshots,
        columns=load_names,
    )
    load = dynamic_load.combine_first(static_load)

    zonal_load = pd.DataFrame(index=MTUS, columns=zones, dtype=float)
    for mtu, (_, values) in enumerate(load.iterrows(), start=1):
        zonal_load.loc[mtu] = values.groupby(load_zone).sum().reindex(zones, fill_value=0.0)
    return zonal_load, capacity


def load_simple_orders(path: str | Path) -> SimpleLayout:
    """Read preprocessed simple orders and append an internal MTU field."""
    required = {"bidding_zone", "Segment"}
    layout = read_simple_layout(path, required_columns=required)
    orders = layout.frame.copy()

    orders["Hour"] = pd.to_numeric(orders["Hour"], errors="raise").astype(int)
    orders["Quarter hour"] = pd.to_numeric(orders["Quarter hour"], errors="raise").astype(int)
    orders["Price"] = pd.to_numeric(orders["Price"], errors="raise")
    orders["Volume"] = pd.to_numeric(orders["Volume"], errors="raise")
    orders["MTU"] = (orders["Hour"] - 1) * 4 + orders["Quarter hour"]

    if set(orders["MTU"]) != set(MTUS):
        raise ValueError("Simple orders must cover every MTU from 1 to 96")
    if orders[["Segment", SIDE]].isna().any().any():
        raise ValueError("Simple orders contain a missing segment or side")
    if orders["Volume"].lt(-TOL).any():
        raise ValueError("Simple-order volumes must be non-negative")

    layout.frame = orders
    return layout


def load_block_orders(path: str | Path) -> tuple[BlockLayout, list[str]]:
    """Read preprocessed block orders and determine their native bid side."""
    layout = read_block_layout(path)
    blocks = layout.frame.copy()
    profiles = native_profile_columns(blocks.columns)

    required = {
        "Data type(BB)", "BlockID", "Block Type", "BlockCodePRM", "Execution",
        "Limit Price", "MAR", "AAR", "bidding_zone", "Segment",
    }
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f"Preprocessed block orders lack: {sorted(missing)}")

    blocks[profiles] = blocks[profiles].apply(pd.to_numeric, errors="raise")
    blocks["Limit Price"] = pd.to_numeric(blocks["Limit Price"], errors="raise")
    positive = blocks[profiles].gt(TOL).any(axis=1)
    negative = blocks[profiles].lt(-TOL).any(axis=1)
    if (positive & negative).any():
        raise ValueError("Mixed-sign block profiles are not supported")

    blocks = blocks.loc[positive | negative].copy()
    blocks["_side"] = "Sell"
    blocks.loc[negative.loc[blocks.index], "_side"] = "Purchase"

    if blocks["Segment"].isna().any():
        raise ValueError("Block orders contain a missing segment")

    layout.frame = blocks
    return layout, profiles


def block_profiles_long(
    blocks: pd.DataFrame,
    profiles: list[str],
    extra_keys: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Melt signed native profiles to positive magnitude by MTU."""
    long = blocks.melt(
        id_vars=["_side", "Segment", *extra_keys],
        value_vars=profiles,
        var_name="_profile",
        value_name="Volume",
    )
    positions = {profile: mtu for mtu, profile in enumerate(profiles, start=1)}
    long["MTU"] = long["_profile"].map(positions)
    long["Volume"] = long["Volume"].abs()
    return long


def supply_multipliers(
    simple: pd.DataFrame,
    blocks: pd.DataFrame,
    profiles: list[str],
    capacity: dict[tuple[str, str], float],
    zones: list[str],
) -> dict[tuple[str, str], float]:
    """Calculate installed-capacity / source-template peak multipliers."""
    simple_supply = simple.loc[simple[SIDE].eq("Sell")]
    block_supply = block_profiles_long(blocks, profiles)
    block_supply = block_supply.loc[block_supply["_side"].eq("Sell")]

    template = simple_supply.groupby(["Segment", "MTU"])["Volume"].sum().add(
        block_supply.groupby(["Segment", "MTU"])["Volume"].sum(),
        fill_value=0.0,
    ).unstack(fill_value=0.0)

    multipliers: dict[tuple[str, str], float] = {}
    for segment, reference in template.max(axis=1).items():
        if segment not in SEGMENT_TYPES:
            raise ValueError(f"No capacity mapping for supply segment {segment!r}")
        for zone in zones:
            installed = capacity[(zone, segment)]
            factor = installed / reference if reference > TOL else 0.0
            multipliers[(zone, segment)] = factor
            LOG.info(
                "m_supply[zone=%s, segment=%s] = %.6f (capacity=%.2f MW, template_peak=%.2f MW)",
                zone, segment, factor, installed, reference,
            )
    return multipliers


def demand_block_multipliers(
    simple: pd.DataFrame,
    blocks: pd.DataFrame,
    profiles: list[str],
    targets: pd.DataFrame,
    zones: list[str],
) -> dict[str, float]:
    """Calculate one demand scaling factor per zone, including block demand."""
    simple_demand = simple.loc[simple[SIDE].eq("Purchase")].copy()
    block_demand = block_profiles_long(blocks, profiles)
    block_demand = block_demand.loc[block_demand["_side"].eq("Purchase")].copy()

    factors: dict[str, float] = {}
    for zone in zones:
        simple_weight = simple_demand["Segment"].map(lambda segment: demand_flexibility(zone, segment))
        block_weight = block_demand["Segment"].map(lambda segment: demand_flexibility(zone, segment))
        template = (simple_demand["Volume"] * simple_weight).groupby(
            simple_demand["MTU"]
        ).sum().reindex(MTUS, fill_value=0.0).add(
            (block_demand["Volume"] * block_weight).groupby(block_demand["MTU"])
            .sum().reindex(MTUS, fill_value=0.0),
            fill_value=0.0,
        )
        if template.le(TOL).any():
            missing_mtus = template.index[template.le(TOL)].tolist()
            raise ValueError(f"The source purchase template is empty for MTUs {missing_mtus}")
        factors[zone] = float((targets[zone] / template).min())
        LOG.info("k_demand[zone=%s] = %.6f", zone, factors[zone])
    return factors


def prefixed_value(zone: str, value) -> str:
    """Prefix a non-empty block identifier with the zonal label."""
    if pd.isna(value) or not str(value).strip():
        return ""
    return f"{zone}_{str(value).strip()}"


def build_zonal_blocks(
    blocks: pd.DataFrame,
    profiles: list[str],
    multipliers: dict[tuple[str, str], float],
    demand_factors: dict[str, float],
    zones: list[str],
) -> pd.DataFrame:
    """Copy each native block to every zone and scale signed profiles."""
    frames: list[pd.DataFrame] = []
    for zone in zones:
        zonal = blocks.copy()
        zonal["_factor"] = [
            multipliers.get((zone, segment), 0.0)
            if side == "Sell"
            else demand_flexibility(zone, segment) * demand_factors[zone]
            for side, segment in zip(zonal["_side"], zonal["Segment"])
        ]
        zonal[profiles] = zonal[profiles].mul(zonal["_factor"], axis=0)
        zonal["BlockID"] = zonal["BlockID"].map(lambda value: prefixed_value(zone, value))
        zonal["BlockCodePRM"] = zonal["BlockCodePRM"].map(lambda value: prefixed_value(zone, value))
        zonal["bidding_zone"] = zone

        non_empty = zonal[profiles].abs().gt(TOL).any(axis=1)
        linked = zonal["BlockCodePRM"].astype(str).str.strip().ne("")
        groups_with_volume = set(zonal.loc[non_empty & linked, "BlockCodePRM"])
        keep = non_empty | (linked & zonal["BlockCodePRM"].isin(groups_with_volume))
        frames.append(zonal.loc[keep].drop(columns=["_factor", "_side"]))

    return pd.concat(frames, ignore_index=True)


def build_zonal_simple(
    simple: pd.DataFrame,
    zonal_blocks: pd.DataFrame,
    profiles: list[str],
    targets: pd.DataFrame,
    multipliers: dict[tuple[str, str], float],
    zones: list[str],
) -> pd.DataFrame:
    """Scale supply by capacity and normalise demand to each zonal target."""
    blocks_for_demand = zonal_blocks.copy()
    negative = blocks_for_demand[profiles].lt(-TOL).any(axis=1)
    blocks_for_demand["_side"] = "Sell"
    blocks_for_demand.loc[negative, "_side"] = "Purchase"

    block_demand = block_profiles_long(
        blocks_for_demand.loc[blocks_for_demand["_side"].eq("Purchase")],
        profiles,
        extra_keys=("bidding_zone",),
    ).groupby(["bidding_zone", "MTU"])["Volume"].sum().unstack("bidding_zone")
    block_demand = block_demand.reindex(index=MTUS, columns=zones, fill_value=0.0).fillna(0.0)

    frames: list[pd.DataFrame] = []
    for zone in zones:
        sell = simple.loc[simple[SIDE].eq("Sell")].copy()
        sell["Volume"] = sell["Volume"] * [
            multipliers.get((zone, segment), 0.0) for segment in sell["Segment"]
        ]

        buy = simple.loc[simple[SIDE].eq("Purchase")].copy()
        buy["Volume"] = buy["Volume"] * buy["Segment"].map(
            lambda segment: demand_flexibility(zone, segment)
        )
        residual = targets[zone] - block_demand[zone]
        if residual.lt(-TOL).any():
            raise ValueError(f"Block demand exceeds target demand in {zone}")

        flexible_sum = buy.groupby("MTU")["Volume"].sum().reindex(MTUS, fill_value=0.0)
        if flexible_sum.le(TOL).any():
            raise ValueError(f"Simple demand template is empty in zone {zone}")
        normalisation = residual / flexible_sum
        buy["Volume"] = buy["Volume"] * buy["MTU"].map(normalisation)

        zonal = pd.concat([sell, buy], ignore_index=True)
        zonal["bidding_zone"] = zone
        frames.append(zonal.loc[zonal["Volume"].gt(TOL)])

    return pd.concat(frames, ignore_index=True)


def supply_report(
    zonal_simple: pd.DataFrame,
    zonal_blocks: pd.DataFrame,
    profiles: list[str],
    capacity: dict[tuple[str, str], float],
    zones: list[str],
) -> pd.DataFrame:
    """Validate submitted zonal supply against installed capacity."""
    block_copy = zonal_blocks.copy()
    block_copy["_side"] = "Sell"
    block_copy.loc[block_copy[profiles].lt(-TOL).any(axis=1), "_side"] = "Purchase"

    segments = sorted(set(zonal_simple.loc[zonal_simple[SIDE].eq("Sell"), "Segment"]))
    index = pd.MultiIndex.from_tuples(
        list(product(zones, MTUS, segments)),
        names=["bidding_zone", "MTU", "Segment"],
    )
    simple_volume = zonal_simple.loc[zonal_simple[SIDE].eq("Sell")].groupby(
        ["bidding_zone", "MTU", "Segment"]
    )["Volume"].sum()
    block_volume = block_profiles_long(
        block_copy.loc[block_copy["_side"].eq("Sell")], profiles,
        extra_keys=("bidding_zone",),
    ).groupby(["bidding_zone", "MTU", "Segment"])["Volume"].sum()

    report = pd.DataFrame(index=index).reset_index()
    report["InstalledCapacityMW"] = [capacity[(zone, segment)] for zone, segment in zip(report["bidding_zone"], report["Segment"])]
    report["SimpleVolumeMW"] = simple_volume.reindex(index, fill_value=0.0).to_numpy()
    report["BlockVolumeMW"] = block_volume.reindex(index, fill_value=0.0).to_numpy()
    report["CombinedVolumeMW"] = report["SimpleVolumeMW"] + report["BlockVolumeMW"]
    report["CapacityUtilisation"] = (
        report["CombinedVolumeMW"] / report["InstalledCapacityMW"].where(report["InstalledCapacityMW"].gt(TOL))
    ).fillna(0.0)

    violations = report.loc[report["CombinedVolumeMW"] > report["InstalledCapacityMW"] + 1e-3]
    if not violations.empty:
        raise ValueError("Submitted supply exceeds installed capacity:\n" + violations.head().to_string(index=False))
    return report


def demand_report(
    zonal_simple: pd.DataFrame,
    zonal_blocks: pd.DataFrame,
    profiles: list[str],
    zonal_load: pd.DataFrame,
    targets: pd.DataFrame,
    zones: list[str],
) -> pd.DataFrame:
    """Report submitted demand against each zonal physical-load target."""
    block_copy = zonal_blocks.copy()
    block_copy["_side"] = "Sell"
    block_copy.loc[block_copy[profiles].lt(-TOL).any(axis=1), "_side"] = "Purchase"

    simple_volume = zonal_simple.loc[zonal_simple[SIDE].eq("Purchase")].groupby(["bidding_zone", "MTU"])["Volume"].sum()
    block_volume = block_profiles_long(
        block_copy.loc[block_copy["_side"].eq("Purchase")], profiles,
        extra_keys=("bidding_zone",),
    ).groupby(["bidding_zone", "MTU"])["Volume"].sum()

    index = pd.MultiIndex.from_tuples(list(product(zones, MTUS)), names=["bidding_zone", "MTU"])
    report = pd.DataFrame(index=index).reset_index()
    report["PhysicalLoadMW"] = [zonal_load.at[mtu, zone] for zone, mtu in zip(report["bidding_zone"], report["MTU"])]
    report["DemandOfferFactor"] = DEMAND_OFFER_FACTOR
    report["TargetSubmittedDemandMW"] = [targets.at[mtu, zone] for zone, mtu in zip(report["bidding_zone"], report["MTU"])]
    report["SubmittedSimpleDemandMW"] = simple_volume.reindex(index, fill_value=0.0).to_numpy()
    report["SubmittedBlockDemandMW"] = block_volume.reindex(index, fill_value=0.0).to_numpy()
    report["TotalSubmittedDemandMW"] = report["SubmittedSimpleDemandMW"] + report["SubmittedBlockDemandMW"]
    return report


def create_zonal_orderbooks(
    simple_input: str | Path,
    block_input: str | Path,
    network: pypsa.Network,
    zone_column: str,
    simple_output: str | Path,
    block_output: str | Path,
    supply_report_output: str | Path | None = None,
    demand_report_output: str | Path | None = None,
) -> None:
    """Create scaled native-layout orders for every supplied network zone."""
    zones = normalize_network_zones(network, zone_column)
    zonal_load, capacity = network_zonal_data(network, zone_column, zones)
    targets = DEMAND_OFFER_FACTOR * zonal_load

    simple_layout = load_simple_orders(simple_input)
    block_layout, profiles = load_block_orders(block_input)
    simple = simple_layout.frame
    blocks = block_layout.frame

    multipliers = supply_multipliers(simple, blocks, profiles, capacity, zones)
    demand_factors = demand_block_multipliers(simple, blocks, profiles, targets, zones)
    zonal_blocks = build_zonal_blocks(blocks, profiles, multipliers, demand_factors, zones)
    zonal_simple = build_zonal_simple(simple, zonal_blocks, profiles, targets, multipliers, zones)

    supply = supply_report(zonal_simple, zonal_blocks, profiles, capacity, zones)
    demand = demand_report(zonal_simple, zonal_blocks, profiles, zonal_load, targets, zones)

    simple_columns = [
        "Date", "Week", "Week Day", "Hour", "Quarter hour", "Price", "Volume",
        SIDE, "bidding_zone", "Segment",
    ]
    simple_layout.frame = zonal_simple[simple_columns].sort_values(
        ["bidding_zone", "Date", "Hour", "Quarter hour", SIDE, "Price"],
        kind="stable",
    )
    write_simple_layout(simple_layout, simple_output)

    block_layout.frame = zonal_blocks.reindex(columns=block_layout.header)
    write_block_layout(block_layout, block_output)

    if supply_report_output is not None:
        Path(supply_report_output).parent.mkdir(parents=True, exist_ok=True)
        supply.to_csv(supply_report_output, index=False)
    if demand_report_output is not None:
        Path(demand_report_output).parent.mkdir(parents=True, exist_ok=True)
        demand.to_csv(demand_report_output, index=False)

    LOG.info("Wrote %d zonal simple orders to %s", len(simple_layout.frame), simple_output)
    LOG.info("Wrote %d zonal block orders to %s", len(block_layout.frame), block_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simple_input", type=Path)
    parser.add_argument("block_input", type=Path)
    parser.add_argument("network", type=Path)
    parser.add_argument("simple_output", type=Path)
    parser.add_argument("block_output", type=Path)
    parser.add_argument("--zone-column", default="zone")
    parser.add_argument("--supply-report", type=Path, default=None)
    parser.add_argument("--demand-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_zonal_orderbooks(
        simple_input=args.simple_input,
        block_input=args.block_input,
        network=pypsa.Network(args.network),
        zone_column=args.zone_column,
        simple_output=args.simple_output,
        block_output=args.block_output,
        supply_report_output=args.supply_report,
        demand_report_output=args.demand_report,
    )


if __name__ == "__main__":
    main()
