"""
Input preparation.

1. Network and day preparation (preprocessing, zonal order books).
2. NTC capacities derived from MaxBEX.
3. Price-simulator inputs (curves, block orders with benchmark AAR, coupling
   input, case specification), written to config.SIMULATOR_INPUT_ROOT.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from collections.abc import Mapping
from datetime import datetime
from itertools import combinations
from math import isfinite
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pypsa

from tools.create_zonal_orderbooks import create_zonal_orderbooks, normalize_zone_label
from tools.preprocess_orders import preprocess_orders

from . import config as cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Delivery date
# =============================================================================
def infer_delivery_date(simple_orders: str | Path) -> datetime:
    """Infer the unique delivery date from the raw published simple orders."""
    simple = pd.read_csv(simple_orders, comment="#", encoding="utf-8-sig")
    simple.columns = [str(column).strip() for column in simple.columns]

    if "Date" not in simple.columns:
        raise ValueError("The published simple-order input must contain a 'Date' column.")

    dates = pd.to_datetime(
        simple["Date"], dayfirst=True, errors="raise"
    ).dt.normalize().unique()

    if len(dates) != 1:
        raise ValueError(
            "The simple-order input must contain exactly one delivery date. "
            f"Found: {list(dates)}"
        )
    return pd.Timestamp(dates[0]).to_pydatetime()


def resolve_delivery_date(day: cfg.DayConfig) -> datetime:
    date = day.delivery_date or infer_delivery_date(day.simple_orders)
    return pd.Timestamp(date).normalize().to_pydatetime()


# =============================================================================
# Network preparation
# =============================================================================
def align_network_snapshots(
    network: pypsa.Network,
    delivery_date: datetime,
    periods: int = 96,
    freq: str = "15min",
) -> None:
    """
    Assign the delivery-day timestamps to the network, in place.

    Snapshot i receives delivery_date + i * freq. Time-dependent data keep
    their values; only the time index is replaced.
    """
    old_snapshots = network.snapshots

    if isinstance(old_snapshots, pd.MultiIndex):
        raise ValueError("Multi-period networks (investment periods) are not supported.")

    if len(old_snapshots) != periods:
        raise ValueError(
            f"The network has {len(old_snapshots)} snapshots; exactly "
            f"{periods} are required to map them onto the delivery day."
        )

    if not old_snapshots.is_monotonic_increasing:
        raise ValueError(
            "Network snapshots are not in chronological order. Positional "
            "reassignment would shuffle the time series."
        )

    first = pd.Timestamp(old_snapshots[0])
    if first != first.normalize():
        logger.warning(
            "First network snapshot is %s (not 00:00). Positional mapping "
            "assumes snapshot 1 = first quarter hour of the day.",
            first,
        )

    new_snapshots = pd.date_range(
        start=pd.Timestamp(delivery_date).normalize(),
        periods=periods,
        freq=freq,
        name=old_snapshots.name,
    )

    if old_snapshots.equals(new_snapshots):
        return

    load_before = (
        network.loads_t.p_set.to_numpy(copy=True)
        if not network.loads_t.p_set.empty
        else None
    )

    # 1) Replace the index of every time-dependent table (values untouched).
    for component in network.components:
        tables = getattr(component, "dynamic", None)
        if tables is None:
            tables = component.pnl  # older PyPSA versions

        for table in tables.values():
            if len(table.index) == periods:
                table.index = new_snapshots

    network.snapshot_weightings.index = new_snapshots

    # 2) Synchronise PyPSA's internal snapshot state.
    network.set_snapshots(new_snapshots)

    # 3) Safety check: load time series must be unchanged.
    if load_before is not None:
        load_after = network.loads_t.p_set.to_numpy()
        if load_after.shape != load_before.shape or not (
            (load_after == load_before) | (pd.isna(load_after) & pd.isna(load_before))
        ).all():
            raise RuntimeError("Load time series changed during snapshot reassignment.")

    logger.info(
        "Network snapshots reassigned: %s ... %s -> %s ... %s",
        old_snapshots[0], old_snapshots[-1], new_snapshots[0], new_snapshots[-1],
    )


def normalize_network_zone_labels(network: pypsa.Network, zone_column: str) -> None:
    """Normalize numeric zone values to unambiguous labels such as Z1."""
    if zone_column not in network.buses.columns:
        raise ValueError(f"Network buses do not contain zone column '{zone_column}'.")

    network.buses[zone_column] = network.buses[zone_column].map(normalize_zone_label)


def validate_network_snapshots(network: pypsa.Network, delivery_date: datetime) -> None:
    """Require all 96 quarter-hourly snapshots for the delivery day."""
    required = pd.date_range(
        start=pd.Timestamp(delivery_date).normalize(), periods=96, freq="15min"
    )
    missing = required.difference(pd.Index(network.snapshots))

    if not missing.empty:
        raise ValueError(
            "The PyPSA network must contain all 96 15-minute snapshots for "
            f"{pd.Timestamp(delivery_date).date()}. Missing: {list(missing)}"
        )


def scale_line_capacities(
    network: pypsa.Network,
    scaling: Mapping[str, float] | None,
    verbose: bool = True,
) -> None:
    """Scale the thermal rating (s_nom) of selected lines, in place."""
    if not scaling:
        return

    missing = [name for name in scaling if name not in network.lines.index]
    if missing:
        raise ValueError(f"Lines not found in network: {missing}")

    for name, factor in scaling.items():
        factor = float(factor)
        if not isfinite(factor) or factor < 0.0:
            raise ValueError(f"Invalid scaling factor for '{name}': {factor}")

        old = network.lines.at[name, "s_nom"]
        network.lines.at[name, "s_nom"] = old * factor

        # Keep extendable bounds consistent, if used.
        if network.lines.at[name, "s_nom_extendable"]:
            network.lines.at[name, "s_nom_max"] *= factor
            network.lines.at[name, "s_nom_min"] *= factor

        if verbose:
            logger.info(
                "Line '%s': s_nom %.2f -> %.2f MVA (factor %.2f).",
                name, old, network.lines.at[name, "s_nom"], factor,
            )


def load_network(
    delivery_date: datetime,
    scaling: Mapping[str, float] | None,
    verbose: bool = True,
) -> pypsa.Network:
    """Load the original network and prepare it (labels, snapshots, scaling)."""
    network = pypsa.Network(str(cfg.NETWORK_FILE))

    normalize_network_zone_labels(network, cfg.ZONE_COLUMN)
    if cfg.ALIGN_SNAPSHOTS:
        align_network_snapshots(network, delivery_date)
    validate_network_snapshots(network, delivery_date)
    scale_line_capacities(network, scaling, verbose=verbose)
    return network


def load_baseline_network(delivery_date: datetime) -> pypsa.Network:
    """Network with the baseline line deratings (LINE_CAPACITY_SCALING)."""
    return load_network(delivery_date, cfg.LINE_CAPACITY_SCALING)


def build_line_scaling(network: pypsa.Network, factor: float) -> dict[str, float]:
    """Combined per-line factor (baseline scaling times scenario factor)."""
    lines = network.lines

    if cfg.LINE_SCALING_SCOPE == "all":
        selected = lines.index
    elif cfg.LINE_SCALING_SCOPE == "cross_border":
        zones = network.buses[cfg.ZONE_COLUMN]
        selected = lines.index[lines["bus0"].map(zones) != lines["bus1"].map(zones)]
    else:
        raise ValueError(f"Unknown LINE_SCALING_SCOPE '{cfg.LINE_SCALING_SCOPE}'.")

    scaling = (
        dict(cfg.LINE_CAPACITY_SCALING) if cfg.APPLY_BASELINE_LINE_SCALING else {}
    )
    for name in selected:
        scaling[name] = scaling.get(name, 1.0) * factor
    return scaling


def load_scaled_network(factor: float, delivery_date: datetime) -> pypsa.Network:
    """Original network with baseline scaling times the scenario factor."""
    network = load_network(delivery_date, None)
    before = float(network.lines["s_nom"].sum())

    scale_line_capacities(network, build_line_scaling(network, factor), verbose=False)

    after = float(network.lines["s_nom"].sum())
    logger.info(
        "Factor %.2f (%s): total line rating %.0f -> %.0f MVA (%.2f x), %s lines.",
        factor, cfg.LINE_SCALING_SCOPE, before, after, after / before, len(network.lines),
    )
    return network


# =============================================================================
# Day preparation (preprocessing + zonal order books)
# =============================================================================
def prepare_day_inputs(day: cfg.DayConfig) -> None:
    """Preprocess the published orders and create the zonal order books."""
    for path in (day.simple_orders, day.block_orders):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    delivery_date = resolve_delivery_date(day)
    day.intermediate_dir.mkdir(parents=True, exist_ok=True)

    network = load_baseline_network(delivery_date)

    logger.info("[%s] preprocessing published orders.", day.name)
    preprocess_orders(
        simple_input=day.simple_orders,
        block_input=day.block_orders,
        simple_output=day.preprocessed_simple,
        block_output=day.preprocessed_blocks,
        source_zone=cfg.SOURCE_ZONE,
        prices_csv=cfg.PRICES_CSV,
        net_positions_csv=cfg.NET_POSITIONS_CSV,
    )

    logger.info("[%s] creating zonal order books.", day.name)
    create_zonal_orderbooks(
        simple_input=day.preprocessed_simple,
        block_input=day.preprocessed_blocks,
        network=network,
        zone_column=cfg.ZONE_COLUMN,
        simple_output=day.zonal_simple,
        block_output=day.zonal_blocks,
        prices_csv=cfg.PRICES_CSV,
        delivery_date=delivery_date,
        carrier_alpha=cfg.CARRIER_ALPHA,
        diagnostics_dir=day.intermediate_dir / "zonal_diagnostics",
        supply_report_output=day.intermediate_dir / "supply_validation.csv",
        demand_report_output=day.intermediate_dir / "demand_validation.csv",
        network_input=cfg.NETWORK_FILE,
    )


# =============================================================================
# NTC capacities from MaxBEX
# =============================================================================
def maxbex_to_ntc(maxbex_csv: Path) -> pd.DataFrame:
    """
    Convert directed MaxBEX values into one virtual NTC line per zone pair.

    For the pair (a, b) with a < b:
        capacity_up   = MaxBEX(a -> b)   (positive flow a -> b)
        capacity_down = MaxBEX(b -> a)   (negative flow)
    """
    maxbex = pd.read_csv(maxbex_csv, parse_dates=["product_start"])
    required = {"product_start", "from_zone", "to_zone", "maxbex_mw"}

    if required.difference(maxbex.columns):
        raise ValueError(
            f"{maxbex_csv} misses columns: {sorted(required.difference(maxbex.columns))}"
        )

    maxbex["from_zone"] = maxbex["from_zone"].astype(str)
    maxbex["to_zone"] = maxbex["to_zone"].astype(str)

    key = ["product_start", "from_zone", "to_zone"]
    if maxbex.duplicated(key).any():
        raise ValueError("MaxBEX file contains duplicate directed pairs.")

    lookup = maxbex.set_index(key)["maxbex_mw"]
    zones = sorted(set(maxbex["from_zone"]) | set(maxbex["to_zone"]))
    timestamps = sorted(maxbex["product_start"].unique())

    rows = []
    for zone_a, zone_b in combinations(zones, 2):
        for timestamp in timestamps:
            rows.append({
                "line_id": f"NTC_{zone_a}_{zone_b}",
                "from_zone": zone_a,
                "to_zone": zone_b,
                "timestamp": pd.Timestamp(timestamp),
                "capacity_up": max(0.0, float(lookup[(timestamp, zone_a, zone_b)])),
                "capacity_down": max(0.0, float(lookup[(timestamp, zone_b, zone_a)])),
            })

    ntc = pd.DataFrame(rows)
    logger.info("Built %s NTC lines from MaxBEX (%s rows).", ntc["line_id"].nunique(), len(ntc))
    return ntc


# =============================================================================
# Simulator input: coupling
# =============================================================================
def _allocate_reference_flows(
    net_positions: pd.Series,
    reference_zone: str,
    tolerance: float = 1e-6,
) -> pd.Series:
    """
    Return f0[z] for all non-reference zones.

    f0[z] > 0: zone z exports to the reference zone.
    f0[z] < 0: reference zone exports to zone z.
    The allocation is proportional to counterparty net positions.
    """
    reference_np = float(net_positions[reference_zone])
    other_np = net_positions.drop(reference_zone)
    result = pd.Series(0.0, index=other_np.index, dtype=float)

    if reference_np < -tolerance:
        exporters = other_np.clip(lower=0.0)
        total_exports = float(exporters.sum())
        if total_exports > tolerance:
            result = (-reference_np) * exporters / total_exports

    elif reference_np > tolerance:
        importers = (-other_np).clip(lower=0.0)
        total_imports = float(importers.sum())
        if total_imports > tolerance:
            result = -reference_np * importers / total_imports

    return result


def prepare_coupling_csv(
    market_meta_csv: str | Path,
    maxbex_csv: str | Path,
    output_csv: str | Path,
    reference_zone: str,
    kappa_by_zone: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Create the simulator coupling input from one storage-free benchmark run.

    `reference_zone` is the zone represented by the simulator's base curve;
    it is excluded from the generated neighbour rows.
    """
    kappa_by_zone = kappa_by_zone or {}

    meta = pd.read_csv(market_meta_csv)
    maxbex = pd.read_csv(maxbex_csv)

    meta["product_start"] = pd.to_datetime(meta["product_start"])
    maxbex["product_start"] = pd.to_datetime(maxbex["product_start"])
    meta["net_position_mw"] = meta["supply_volume"] - meta["demand_volume"]

    required_maxbex = {"product_start", "from_zone", "to_zone", "maxbex_mw"}
    missing = required_maxbex.difference(maxbex.columns)
    if missing:
        raise ValueError(f"maxbex.csv lacks required columns: {sorted(missing)}")

    timestamps = sorted(meta["product_start"].unique())
    zones = sorted(meta["zone"].unique())

    if reference_zone not in zones:
        raise ValueError(
            f"Reference zone '{reference_zone}' is not in market_meta.csv. "
            f"Available zones: {zones}"
        )

    capacity = maxbex.set_index(["product_start", "from_zone", "to_zone"])["maxbex_mw"]

    rows = []
    for period_index, timestamp in enumerate(timestamps, start=1):
        mtu_meta = meta.loc[meta["product_start"] == timestamp].set_index("zone")

        if set(mtu_meta.index) != set(zones):
            raise ValueError(f"Incomplete market_meta data at {timestamp}.")

        net_positions = mtu_meta["net_position_mw"].astype(float)
        baseline_flows = _allocate_reference_flows(net_positions, reference_zone)

        reference_demand = float(mtu_meta.at[reference_zone, "demand_volume"])
        if reference_demand <= 0.0:
            raise ValueError(
                f"Reference demand is non-positive at {timestamp}; "
                "provide external zonal load data for size calculation."
            )

        delivery_date = pd.Timestamp(timestamp).strftime("%Y%m%d")

        for zone in zones:
            if zone == reference_zone:
                continue

            try:
                capacity_zone_to_ref = float(capacity.loc[(timestamp, zone, reference_zone)])
                capacity_ref_to_zone = float(capacity.loc[(timestamp, reference_zone, zone)])
            except KeyError as exc:
                raise ValueError(
                    f"Missing MaxBex for {zone}<->{reference_zone} at {timestamp}."
                ) from exc

            f0 = float(baseline_flows.at[zone])

            # Initial proxy for L_z / L_ref.
            load_ratio = float(mtu_meta.at[zone, "demand_volume"]) / reference_demand
            size = kappa_by_zone.get(zone, 1.0) * load_ratio

            rows.append({
                "date": delivery_date,
                "period": f"Q{period_index:03d}",
                "zone": zone,
                "price_zone_eur_mwh": float(mtu_meta.at[zone, "price"]),
                "size": max(0.0, size),
                "headroom_import_mw": max(0.0, capacity_zone_to_ref - f0),
                "headroom_export_mw": max(0.0, capacity_ref_to_zone + f0),
                "baseline_flow_mw": f0,
                "explicit": False,
            })

    result = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


# =============================================================================
# Simulator input: EPEX-style block orders with benchmark acceptance ratios
# =============================================================================
EPEX_BLOCK_PREFIX = "Data type(BB)"
EPEX_BLOCK_FIXED_COLUMNS = [
    "Data type(BB)", "BlockID", "Block Type", "BlockCodePRM", "Execution", "Limit Price",
]
EPEX_PROFILE_COLUMNS = [
    f"Volume H{hour:02d} Q{quarter}"
    for hour in range(1, 25)
    for quarter in range(1, 5)
]
EPEX_BLOCK_SUFFIX_COLUMNS = ["MAR", "AAR"]
EPEX_BLOCK_OUTPUT_COLUMNS = [
    *EPEX_BLOCK_FIXED_COLUMNS, *EPEX_PROFILE_COLUMNS, *EPEX_BLOCK_SUFFIX_COLUMNS,
]


def _normalise_column_name(value: object) -> str:
    return " ".join(str(value).strip().split())


def _read_epex_block_orders(path: str | Path) -> pd.DataFrame:
    """Read an EPEX-style semicolon-separated block-order file (commented BB header)."""
    path = Path(path)
    header: list[str] | None = None
    data_lines: list[str] = []

    with path.open(mode="r", encoding="utf-8-sig", newline="") as file:
        for raw_line in file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip()
                if candidate.startswith(EPEX_BLOCK_PREFIX):
                    header = next(csv.reader([candidate], delimiter=";"))
                continue
            data_lines.append(raw_line)

    if header is None:
        raise ValueError(
            f"Could not find commented EPEX block header '{EPEX_BLOCK_PREFIX}' in '{path}'."
        )

    header = [_normalise_column_name(column) for column in header]
    rows = list(csv.reader(data_lines, delimiter=";"))
    if not rows:
        raise ValueError(f"EPEX block-order file '{path}' contains no data.")

    # Metadata rows (ST, AL) are shorter than BB rows; pad to header width.
    padded_rows = [row + [""] * max(0, len(header) - len(row)) for row in rows]
    frame = pd.DataFrame(padded_rows, columns=header)

    required_columns = {
        "Data type(BB)", "BlockID", "Block Type", "BlockCodePRM",
        "Execution", "Limit Price", "MAR", *EPEX_PROFILE_COLUMNS,
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"EPEX block-order file '{path}' is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame.loc[frame["Data type(BB)"].astype(str).str.strip() == "BB"].copy()
    if frame.empty:
        raise ValueError(f"No BB rows found in '{path}'.")

    frame["BlockID"] = frame["BlockID"].astype(str).str.strip()
    if frame["BlockID"].eq("").any():
        raise ValueError(f"An empty BlockID exists in '{path}'.")

    duplicated = frame["BlockID"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"Duplicate BlockID values in '{path}': "
            f"{sorted(frame.loc[duplicated, 'BlockID'].unique())}"
        )
    return frame


def _parse_json_mapping(value: object, field_name: str, bid_id: str) -> dict[str, float]:
    if pd.isna(value):
        raise ValueError(f"Block '{bid_id}' has an empty '{field_name}' field.")

    if isinstance(value, dict):
        mapping = value
    else:
        try:
            mapping = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Block '{bid_id}' has invalid JSON in '{field_name}'.") from exc

    if not isinstance(mapping, dict):
        raise ValueError(f"Block '{bid_id}' field '{field_name}' must be a mapping.")

    result = {}
    for timestamp, volume in mapping.items():
        try:
            result[str(timestamp)] = float(volume)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Block '{bid_id}' contains a non-numeric volume in '{field_name}'."
            ) from exc
    return result


def _calculate_actual_acceptance_ratio(
    submitted_volume: dict[str, float],
    accepted_volume: dict[str, float],
    bid_id: str,
    tolerance: float = 1e-7,
) -> float:
    """Actual acceptance ratio of one block bid, from absolute MW volumes."""
    submitted_total = 0.0
    accepted_total = 0.0
    pointwise_ratios = []

    for timestamp, submitted in submitted_volume.items():
        accepted = accepted_volume.get(timestamp, 0.0)

        if abs(submitted) <= tolerance:
            if abs(accepted) > tolerance:
                raise ValueError(
                    f"Block '{bid_id}' has accepted volume at a submitted "
                    f"zero-volume timestamp '{timestamp}'."
                )
            continue

        submitted_total += abs(submitted)
        accepted_total += abs(accepted)
        pointwise_ratios.append(accepted / submitted)

    if submitted_total <= tolerance:
        raise ValueError(f"Block '{bid_id}' has no non-zero submitted profile volume.")

    aar = accepted_total / submitted_total

    # One acceptance variable applies to the full profile.
    if pointwise_ratios:
        max_deviation = max(abs(ratio - aar) for ratio in pointwise_ratios)
        if max_deviation > 1e-5:
            raise ValueError(
                f"Block '{bid_id}' has inconsistent per-MTU acceptance ratios. "
                f"Maximum deviation: {max_deviation:.8f}"
            )

    if aar < -tolerance or aar > 1.0 + tolerance:
        raise ValueError(f"Block '{bid_id}' has invalid AAR '{aar}'.")
    if abs(aar) <= tolerance:
        return 0.0
    if abs(aar - 1.0) <= tolerance:
        return 1.0
    return float(min(1.0, max(0.0, aar)))


def _load_block_acceptance_results(
    accepted_orders_csv: str | Path,
    rejected_orders_csv: str | Path,
) -> pd.DataFrame:
    """Block AARs and zone assignment from the benchmark accepted/rejected files."""
    dtypes = {"parent_bid_id": "string", "exclusive_group_id": "string"}
    accepted = pd.read_csv(accepted_orders_csv, low_memory=False, dtype=dtypes)
    rejected = pd.read_csv(rejected_orders_csv, low_memory=False, dtype=dtypes)
    orders = pd.concat([accepted, rejected], ignore_index=True)

    required_columns = {"bid_id", "bid_type", "node", "volume", "accepted_volume"}
    missing_columns = required_columns.difference(orders.columns)
    if missing_columns:
        raise ValueError(
            f"Benchmark result CSVs are missing required columns: {sorted(missing_columns)}"
        )

    orders = orders.loc[orders["bid_type"].isin(["BB", "LB"])].copy()
    if orders.empty:
        raise ValueError("No BB or LB rows were found in the benchmark result CSVs.")

    orders["bid_id"] = orders["bid_id"].astype(str).str.strip()
    orders["node"] = orders["node"].astype(str).str.strip()

    duplicated = orders["bid_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "A block bid appears multiple times across accepted/rejected results: "
            f"{sorted(orders.loc[duplicated, 'bid_id'].unique())}"
        )

    rows = []
    for row in orders.itertuples(index=False):
        bid_id = str(row.bid_id).strip()
        aar = _calculate_actual_acceptance_ratio(
            _parse_json_mapping(row.volume, "volume", bid_id),
            _parse_json_mapping(row.accepted_volume, "accepted_volume", bid_id),
            bid_id,
        )
        rows.append({
            "BlockID": bid_id,
            "zone": str(row.node).strip(),
            "AAR": aar,
            "Execution": "Y" if aar > 0.0 else "N",
        })

    result = pd.DataFrame(rows)
    if result["zone"].eq("").any():
        raise ValueError(
            f"Block result rows without a zone: {result.loc[result['zone'].eq(''), 'BlockID'].tolist()}"
        )
    return result


def _format_epex_number(value: object, decimals: int = 8) -> str:
    number = float(value)
    if abs(number) < 1e-12:
        number = 0.0
    text = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _safe_zone_filename(zone: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(zone).strip())
    if not safe:
        raise ValueError(f"Zone '{zone}' cannot be converted to a file name.")
    return safe


def export_zone_block_orders_with_aar(
    source_epex_block_orders_csv: str | Path,
    accepted_orders_csv: str | Path,
    rejected_orders_csv: str | Path,
    output_dir: str | Path,
    delivery_date: str | pd.Timestamp,
    file_prefix: str,
    strict: bool = True,
) -> dict[str, Path]:
    """
    One EPEX-style block-order CSV per zone, with benchmark AAR values.

    The profile comes from the zonal block-order input; acceptance status and
    AAR come from the storage-free benchmark result. Files are named
    {file_prefix}_{zone}_{YYYYMMDD}.csv.
    """
    source = _read_epex_block_orders(source_epex_block_orders_csv)
    acceptance = _load_block_acceptance_results(accepted_orders_csv, rejected_orders_csv)

    # Replace the historical Execution/AAR by the benchmark values.
    source = source.drop(columns=["Execution", "AAR"], errors="ignore")
    source = source.merge(acceptance, on="BlockID", how="left", validate="one_to_one")

    missing_results = source["zone"].isna()
    if missing_results.any():
        message = (
            "The following source EPEX block IDs were not found in the "
            f"storage-free benchmark results: {source.loc[missing_results, 'BlockID'].tolist()}"
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s. They are exported as rejected blocks.", message)
        source.loc[missing_results, "AAR"] = 0.0
        source.loc[missing_results, "Execution"] = "N"
        source.loc[missing_results, "zone"] = "UNASSIGNED"

    delivery_timestamp = pd.Timestamp(delivery_date).normalize()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    for zone, zone_blocks in source.groupby("zone", sort=True):
        output = pd.DataFrame(index=zone_blocks.index, columns=EPEX_BLOCK_OUTPUT_COLUMNS)

        for column in EPEX_BLOCK_FIXED_COLUMNS:
            output[column] = zone_blocks[column].fillna("").astype(str)
        for column in EPEX_PROFILE_COLUMNS:
            output[column] = zone_blocks[column].fillna("0").astype(str)

        output["MAR"] = zone_blocks["MAR"].fillna("").astype(str)
        output["AAR"] = zone_blocks["AAR"].map(_format_epex_number)
        output["Execution"] = zone_blocks["Execution"].astype(str)

        output_path = (
            output_dir
            / f"{file_prefix}_{_safe_zone_filename(str(zone))}_{delivery_timestamp:%Y%m%d}.csv"
        )

        with output_path.open(mode="w", encoding="utf-8", newline="") as file:
            file.write("# " + ";".join(EPEX_BLOCK_OUTPUT_COLUMNS) + "\n")
            writer = csv.writer(file, delimiter=";", lineterminator="\n")
            for row in output.itertuples(index=False, name=None):
                writer.writerow(row)
            writer.writerow(["AL", len(output)])

        output_paths[str(zone)] = output_path

    return output_paths


# =============================================================================
# Simulator input: staging per case
# =============================================================================
def curve_filename(zone: str) -> str:
    """Curve file exported by the benchmark for one zone."""
    return f"aggregated_curves_{quote(zone, safe='-_.')}.csv"


def _empty_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    return path


def validate_simulator_inputs_available(case: cfg.Case) -> None:
    """Fail early if the benchmark files needed for staging are missing."""
    run_dir = case.no_storage_dir
    required = [run_dir / curve_filename(case.curve_zone)]

    if case.sim_needs_blocks:
        required += [
            run_dir / "accepted_orders.csv",
            run_dir / "rejected_orders.csv",
            case.day_config.zonal_blocks,
        ]
    if case.sim_needs_coupling:
        if case.maxbex_csv is None:
            raise ValueError(f"[{case.name}] coupling methods need maxbex_csv.")
        required += [run_dir / "market_meta.csv", case.maxbex_csv]

    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"[{case.name}] missing files: {missing}")


def stage_simulator_inputs(case: cfg.Case) -> Path:
    """
    Write the simulator input of one case below SIMULATOR_INPUT_ROOT/<case>:

        curves/    aggregated curve of the simulated zone
        blocks/    block orders of the simulated zone with benchmark AAR
        coupling/  neighbour prices, sizes and MaxBEX headroom
        simulator_case.json   specification for the manual simulator run
    """
    if not case.sim_methods:
        logger.info("[%s] no simulator methods configured; skipped.", case.name)
        return case.simulator_input_dir

    validate_simulator_inputs_available(case)

    run_dir = case.no_storage_dir
    input_dir = case.simulator_input_dir
    date = resolve_delivery_date(case.day_config).strftime("%Y%m%d")
    logger.info("[%s] staging simulator input in %s", case.name, input_dir)

    files: dict[str, str] = {}

    # 1. Curve of the simulated zone, named as the simulator expects.
    curves_dir = _empty_dir(input_dir / "curves")
    curve_target = curves_dir / cfg.SIM_CURVE_FILE_TEMPLATE.format(date=date)
    shutil.copy2(run_dir / curve_filename(case.curve_zone), curve_target)
    files["curves"] = str(curves_dir)

    # 2. Block orders with benchmark AAR, only for the simulated zone.
    if case.sim_needs_blocks:
        all_zones_dir = _empty_dir(input_dir / "blocks_all_zones")
        paths = export_zone_block_orders_with_aar(
            source_epex_block_orders_csv=case.day_config.zonal_blocks,
            accepted_orders_csv=run_dir / "accepted_orders.csv",
            rejected_orders_csv=run_dir / "rejected_orders.csv",
            output_dir=all_zones_dir,
            delivery_date=date,
            file_prefix=cfg.SIM_BLOCK_FILE_PREFIX,
            strict=True,
        )

        if case.curve_zone not in paths:
            raise ValueError(
                f"[{case.name}] no block orders for zone '{case.curve_zone}'. "
                f"Zones with blocks: {sorted(paths)}. Remove block methods."
            )

        blocks_dir = _empty_dir(input_dir / "blocks")
        shutil.copy2(paths[case.curve_zone], blocks_dir / paths[case.curve_zone].name)
        files["blocks"] = str(blocks_dir)

    # 3. Coupling input.
    if case.sim_needs_coupling:
        coupling_dir = _empty_dir(input_dir / "coupling")
        frame = prepare_coupling_csv(
            market_meta_csv=run_dir / "market_meta.csv",
            maxbex_csv=case.maxbex_csv,
            output_csv=input_dir / "coupling_all_days.csv",
            reference_zone=case.curve_zone,
            kappa_by_zone=cfg.SIM_KAPPA_BY_ZONE,
        )

        frame["date"] = frame["date"].astype(str)
        if sorted(frame["date"].unique()) != [date]:
            raise ValueError(
                f"[{case.name}] coupling dates {sorted(frame['date'].unique())} "
                f"do not match delivery date {date}."
            )

        for day, part in frame.groupby("date"):
            part.to_csv(coupling_dir / cfg.SIM_COUPLING_FILE_TEMPLATE.format(date=day), index=False)
        files["coupling"] = str(coupling_dir)

    # 4. Case specification for the manual simulator run.
    spec = {
        "case": case.name,
        "regime": case.name,
        "delivery_date": date,
        "curve_zone": case.curve_zone,
        "report_zone": case.report_zone,
        "methods": {m: cfg.SIMULATOR_METHODS[m] for m in case.sim_methods},
        "storage_orders": list(case.sim_storage_counts),
        "order_power_mw": cfg.ORDER_POWER_MW,
        "order_energy_mwh": cfg.ORDER_ENERGY_MWH,
        "efficiency": cfg.EFFICIENCY,
        "cycling_cost_eur_per_mwh": cfg.CYCLING_COST_EUR_PER_MWH,
        "end_of_day_value": cfg.END_OF_DAY_VALUE,
        "starting_energy": cfg.STARTING_ENERGY,
        "benchmark_results_dir": str(case.results_dir),
        "output_root": str(cfg.SIMULATOR_OUTPUT_ROOT),
        "files": files,
    }
    (input_dir / "simulator_case.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return input_dir