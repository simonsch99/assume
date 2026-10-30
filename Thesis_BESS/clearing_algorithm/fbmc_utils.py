# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Utilities for pre-computing Flow-Based Market Coupling (FBMC) parameters.

Implemented:
- nodal D-2 base case with PyPSA;
- nodal PTDFs;
- GSKs;
- zonal and zone-to-zone PTDFs;
- CNE selection;
- N-1 CNECs using LODFs;
- direct POMATO-style FB domain representation:
      zPTDF * NET_EXPORT_FB <= RAM;
- FRM and minRAM;
- an explicit flow-based region.

Only fixed 15-minute market time units are supported.

Physical quantities remain in MW:

- PTDFs;
- RAM;
- line capacities;
- line flows;
- net positions;
- generator dispatch.

The PyPSA D-2 objective uses a 0.25-hour snapshot weighting because
generator marginal costs are EUR/MWh and generator dispatch is MW.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
import pypsa

from assume.common.grid_utils import get_supported_solver_linopy

logger = logging.getLogger(__name__)

BASECASE_CONTINGENCY = "__basecase__"

MTU_DURATION = timedelta(minutes=15)
MTU_DURATION_HOURS = 0.25


# ---------------------------------------------------------------------------
# Network loading and validation
# ---------------------------------------------------------------------------


def load_pypsa_network(path: str | Path) -> pypsa.Network:
    """Load a PyPSA network, for example from a NetCDF file."""
    return pypsa.Network(path)


def _validate_network(
    network: pypsa.Network,
    zone_column: str,
) -> None:
    """Validate the minimum physical data needed for this implementation."""
    if network.buses.empty:
        raise ValueError("The PyPSA network does not contain buses.")

    if network.lines.empty:
        raise ValueError("The PyPSA network does not contain AC lines.")

    if network.generators.empty:
        raise ValueError("The PyPSA network does not contain generators.")

    if zone_column not in network.buses.columns:
        raise ValueError(
            f"network.buses must contain the bidding-zone column "
            f"'{zone_column}'."
        )

    required_line_columns = {"bus0", "bus1", "x", "s_nom"}
    missing_line_columns = required_line_columns.difference(network.lines.columns)

    if missing_line_columns:
        raise ValueError(
            "network.lines is missing required columns: "
            f"{sorted(missing_line_columns)}"
        )

    if (network.lines["s_nom"] <= 0).any():
        raise ValueError("All lines must have s_nom > 0.")

    if network.lines["x"].isna().any() or (network.lines["x"] == 0).any():
        raise ValueError(
            "All lines must have a non-zero reactance x for PTDF calculation."
        )

    unknown_bus0 = set(network.lines["bus0"]).difference(network.buses.index)
    unknown_bus1 = set(network.lines["bus1"]).difference(network.buses.index)

    if unknown_bus0 or unknown_bus1:
        raise ValueError(
            "Some lines refer to buses not present in network.buses. "
            f"Unknown bus0 values: {sorted(unknown_bus0)}; "
            f"unknown bus1 values: {sorted(unknown_bus1)}."
        )

    required_generator_columns = {"bus", "p_nom", "marginal_cost"}
    missing_generator_columns = required_generator_columns.difference(
        network.generators.columns
    )

    if missing_generator_columns:
        raise ValueError(
            "network.generators is missing required columns: "
            f"{sorted(missing_generator_columns)}"
        )

    unknown_generator_buses = set(network.generators["bus"]).difference(
        network.buses.index
    )

    if unknown_generator_buses:
        raise ValueError(
            "Some generators refer to unknown buses: "
            f"{sorted(unknown_generator_buses)}"
        )

    if not network.loads.empty:
        unknown_load_buses = set(network.loads["bus"]).difference(network.buses.index)

        if unknown_load_buses:
            raise ValueError(
                "Some loads refer to unknown buses: "
                f"{sorted(unknown_load_buses)}"
            )


def _validate_15_minute_snapshots(
    snapshots: pd.Index,
) -> pd.DatetimeIndex:
    """
    Validate fixed 15-minute FBMC snapshots.

    Snapshots must be unique datetime timestamps on 15-minute boundaries.
    Multiple snapshots must be ordered and separated by exactly 15 minutes.
    """
    try:
        timestamps = pd.DatetimeIndex(snapshots)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "FBMC snapshots must be valid datetime timestamps."
        ) from exc

    if timestamps.empty:
        raise ValueError(
            "No snapshots are available. Provide 15-minute snapshots "
            "explicitly or store them in the PyPSA network."
        )

    if timestamps.hasnans:
        raise ValueError("FBMC snapshots must not contain null timestamps.")

    if timestamps.has_duplicates:
        raise ValueError("FBMC snapshots must be unique.")

    for timestamp in timestamps:
        if (
            timestamp.minute % 15 != 0
            or timestamp.second != 0
            or timestamp.microsecond != 0
            or timestamp.nanosecond != 0
        ):
            raise ValueError(
                "FBMC snapshots must be aligned to 15-minute boundaries. "
                f"Invalid timestamp: '{timestamp}'."
            )

    if len(timestamps) > 1:
        differences = timestamps[1:] - timestamps[:-1]

        if not (differences == pd.Timedelta(MTU_DURATION)).all():
            raise ValueError(
                "FBMC snapshots must be consecutive 15-minute timestamps. "
                "Hourly, mixed-resolution, unordered, and gapped snapshot "
                "inputs are not supported."
            )

    return timestamps


def _resolve_snapshots(
    network: pypsa.Network,
    snapshots: pd.Index | None,
) -> pd.DatetimeIndex:
    """
    Determine and validate snapshots used for the FBMC calculation.

    If snapshots is omitted, the existing network snapshots are used.
    Only fixed consecutive 15-minute snapshots are supported.
    """
    if snapshots is None:
        snapshots = pd.Index(network.snapshots)
    else:
        snapshots = pd.Index(snapshots)

    return _validate_15_minute_snapshots(snapshots)


def _set_objective_snapshot_weighting(
    network: pypsa.Network,
    snapshots: pd.DatetimeIndex,
) -> None:
    """
    Set the objective weighting to 0.25 hours for every 15-minute snapshot.

    Only the objective weighting is changed. Physical quantities remain in MW,
    and no other PyPSA snapshot-weighting columns are overwritten.
    """
    snapshot_weightings = getattr(network, "snapshot_weightings", None)

    if isinstance(snapshot_weightings, pd.DataFrame):
        if "objective" not in snapshot_weightings.columns:
            snapshot_weightings["objective"] = 1.0

        snapshot_weightings.loc[
            snapshots,
            "objective",
        ] = MTU_DURATION_HOURS
        return

    if isinstance(snapshot_weightings, pd.Series):
        snapshot_weightings.loc[snapshots] = MTU_DURATION_HOURS
        return

    raise RuntimeError(
        "The installed PyPSA version does not expose snapshot_weightings "
        "as a pandas DataFrame or Series."
    )


# ---------------------------------------------------------------------------
# Time-series handling and D-2 base case
# ---------------------------------------------------------------------------


def _line_capacity(
    network: pypsa.Network,
    snapshots: pd.Index,
) -> pd.DataFrame:
    """
    Return physical line capacities in MW for every snapshot.

    Time-dependent ``lines_t.s_max_pu`` values are used when available;
    otherwise static ``lines.s_max_pu`` values are used. Missing values fall
    back to one.
    """
    static_s_max_pu = network.lines["s_max_pu"].fillna(1.0)

    factors = pd.DataFrame(
        np.tile(static_s_max_pu.to_numpy(), (len(snapshots), 1)),
        index=snapshots,
        columns=network.lines.index,
        dtype=float,
    )

    if (
        hasattr(network, "lines_t")
        and hasattr(network.lines_t, "s_max_pu")
        and not network.lines_t.s_max_pu.empty
    ):
        dynamic_factors = network.lines_t.s_max_pu.reindex(
            index=snapshots,
            columns=network.lines.index,
        )
        factors = dynamic_factors.fillna(factors)

    return factors.mul(network.lines["s_nom"], axis=1)


def _load_profile(
    network: pypsa.Network,
    snapshots: pd.Index,
) -> pd.DataFrame:
    """
    Return a snapshot x load-component demand profile in MW.

    ``loads_t.p_set`` is used where available. Missing values fall back to the
    static ``loads.p_set`` value repeated over all snapshots.
    """
    if network.loads.empty:
        return pd.DataFrame(index=snapshots, dtype=float)

    static_load = network.loads["p_set"].fillna(0.0)

    result = pd.DataFrame(
        np.tile(static_load.to_numpy(), (len(snapshots), 1)),
        index=snapshots,
        columns=network.loads.index,
        dtype=float,
    )

    if (
        hasattr(network, "loads_t")
        and hasattr(network.loads_t, "p_set")
        and not network.loads_t.p_set.empty
    ):
        dynamic_load = network.loads_t.p_set.reindex(
            index=snapshots,
            columns=network.loads.index,
        )
        result = dynamic_load.fillna(result)

    return result


def _generator_availability_profile(
    network: pypsa.Network,
    snapshots: pd.Index,
) -> pd.DataFrame:
    """
    Return generator availability as snapshot x generator p_max_pu values.

    ``generators_t.p_max_pu`` is used where available. Missing values fall
    back to 1.0.
    """
    result = pd.DataFrame(
        1.0,
        index=snapshots,
        columns=network.generators.index,
        dtype=float,
    )

    if (
        hasattr(network, "generators_t")
        and hasattr(network.generators_t, "p_max_pu")
        and not network.generators_t.p_max_pu.empty
    ):
        dynamic_availability = network.generators_t.p_max_pu.reindex(
            index=snapshots,
            columns=network.generators.index,
        )
        result = dynamic_availability.fillna(result)

    if (result < 0.0).any().any() or (result > 1.0).any().any():
        raise ValueError(
            "Generator p_max_pu values must be between zero and one."
        )

    return result


def _calculate_nodal_injections(
    network: pypsa.Network,
    dispatch: pd.DataFrame,
    load_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate nodal D-2 injections: generation minus load."""
    nodal_injections = pd.DataFrame(
        0.0,
        index=dispatch.index,
        columns=network.buses.index,
        dtype=float,
    )

    for bus in network.buses.index:
        generator_ids = network.generators.index[
            network.generators["bus"] == bus
        ]
        load_ids = network.loads.index[network.loads["bus"] == bus]

        generation = dispatch.reindex(
            columns=generator_ids,
            fill_value=0.0,
        ).sum(axis=1)

        demand = load_profile.reindex(
            columns=load_ids,
            fill_value=0.0,
        ).sum(axis=1)

        nodal_injections[bus] = generation - demand

    return nodal_injections


def _aggregate_zonal_net_positions(
    nodal_injections: pd.DataFrame,
    buses: pd.DataFrame,
    zone_column: str,
) -> pd.DataFrame:
    """Aggregate nodal injections to ordinary zonal D-2 net positions."""
    result = pd.DataFrame(index=nodal_injections.index, dtype=float)

    for zone in pd.Index(buses[zone_column].unique()):
        zone_buses = buses.index[buses[zone_column] == zone]

        result[zone] = nodal_injections.reindex(
            columns=zone_buses,
            fill_value=0.0,
        ).sum(axis=1)

    return result


def solve_d2_base_case(
    network: pypsa.Network,
    snapshots: pd.Index,
    zone_column: str,
    frm: float = 0.0,
    cnes: pd.Index | None = None,
    solver_name: str | None = None,
) -> dict:
    """
    Solve a nodal D-2 economic dispatch.

    The caller's network is never modified.

    Generator marginal costs are EUR/MWh and dispatch remains in MW. The
    PyPSA objective snapshot weighting is set to 0.25 hours for every fixed
    15-minute MTU so that the optimisation objective is measured in EUR.

    If CNEs are supplied, FRM is reserved on those normal-state CNEs by
    reducing their D-2 operational capacity to:

        capacity * (1 - FRM)

    This does not create preventive N-1 security constraints.
    """
    if not 0.0 <= frm < 1.0:
        raise ValueError("frm must be a fraction in [0, 1).")

    snapshots = _validate_15_minute_snapshots(pd.Index(snapshots))

    n = network.copy()
    n.set_snapshots(snapshots)
    _set_objective_snapshot_weighting(n, snapshots)

    load_profile = _load_profile(network, snapshots)
    generator_availability = _generator_availability_profile(network, snapshots)

    n.loads_t.p_set = load_profile
    n.generators_t.p_max_pu = generator_availability

    physical_line_capacity = _line_capacity(network, snapshots)

    if cnes is not None and len(cnes) > 0:
        cnes = pd.Index(cnes)
        unknown_cnes = cnes.difference(n.lines.index)

        if not unknown_cnes.empty:
            raise ValueError(f"Unknown CNEs: {list(unknown_cnes)}")

        reduced_capacity = physical_line_capacity.loc[:, cnes] * (1.0 - frm)

        line_s_max_pu = physical_line_capacity.div(
            n.lines["s_nom"],
            axis=1,
        )

        line_s_max_pu.loc[:, cnes] = reduced_capacity.div(
            n.lines.loc[cnes, "s_nom"],
            axis=1,
        )

        n.lines_t.s_max_pu = line_s_max_pu

    selected_solver = get_supported_solver_linopy(solver_name)

    status, termination_condition = n.optimize(
        solver_name=selected_solver,
        log_to_console=False,
        progress=False,
        include_objective_constant=True,
    )

    if status != "ok":
        raise RuntimeError(
            "D-2 base case did not solve successfully. "
            f"Status: {status}, termination: {termination_condition}"
        )

    dispatch = n.generators_t.p.reindex(
        index=snapshots,
        columns=n.generators.index,
        fill_value=0.0,
    )

    flows = n.lines_t.p0.reindex(
        index=snapshots,
        columns=n.lines.index,
        fill_value=0.0,
    )

    nodal_injections = _calculate_nodal_injections(
        network=n,
        dispatch=dispatch,
        load_profile=load_profile,
    )

    zonal_net_positions = _aggregate_zonal_net_positions(
        nodal_injections=nodal_injections,
        buses=n.buses,
        zone_column=zone_column,
    )

    balance_error = nodal_injections.sum(axis=1).abs().max()

    if balance_error > 1e-5:
        logger.warning(
            "D-2 energy-balance error is %.6f MW.",
            balance_error,
        )

    return {
        "network": n,
        "dispatch": dispatch,
        "flows": flows,
        "nodal_injections": nodal_injections,
        "zonal_net_positions": zonal_net_positions,
        "load_profile": load_profile,
        "generator_availability": generator_availability,
        "line_capacity": physical_line_capacity,
    }


# ---------------------------------------------------------------------------
# PTDF and LODF calculation
# ---------------------------------------------------------------------------


def calculate_nodal_ptdf(network: pypsa.Network) -> pd.DataFrame:
    """
    Calculate nodal PTDFs using PyPSA.

    Rows are lines and columns are buses. Flow directions follow PyPSA p0,
    i.e. bus0 -> bus1.
    """
    n = network.copy()
    n.determine_network_topology()

    if len(n.sub_networks.index) != 1:
        raise ValueError(
            "This FBMC implementation currently requires one connected "
            "AC network."
        )

    subnetwork = n.sub_networks["obj"].iloc[0]
    subnetwork.calculate_PTDF()

    nodal_ptdf = pd.DataFrame(
        0.0,
        index=n.lines.index,
        columns=n.buses.index,
        dtype=float,
    )

    branch_index = list(subnetwork.branches_i())
    bus_index = pd.Index(subnetwork.buses_i())

    for row, branch in enumerate(branch_index):
        component, branch_id = branch

        if component == "Line":
            nodal_ptdf.loc[branch_id, bus_index] = subnetwork.PTDF[row, :]

    return nodal_ptdf


def _outage_keeps_network_connected(
    network: pypsa.Network,
    outage_line: Hashable,
) -> bool:
    """Return whether the AC network remains connected after a line outage."""
    buses = list(network.buses.index)

    if len(buses) <= 1:
        return True

    neighbours = {bus: set() for bus in buses}

    for line_id, line in network.lines.iterrows():
        if line_id == outage_line:
            continue

        neighbours[line["bus0"]].add(line["bus1"])
        neighbours[line["bus1"]].add(line["bus0"])

    visited = set()
    stack = [buses[0]]

    while stack:
        bus = stack.pop()

        if bus in visited:
            continue

        visited.add(bus)
        stack.extend(neighbours[bus] - visited)

    return len(visited) == len(buses)


def calculate_lodf(
    network: pypsa.Network,
    nodal_ptdf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate N-1 LODFs from nodal PTDFs.

    ``lodf.loc[monitored_line, outage_line]`` gives the relative flow
    change on the monitored line caused by the outage of outage_line.

    Islanding outages receive NaN and are excluded from CNEC generation.
    """
    incidence = pd.DataFrame(
        0.0,
        index=network.lines.index,
        columns=network.buses.index,
        dtype=float,
    )

    for line_id, line in network.lines.iterrows():
        incidence.at[line_id, line["bus0"]] = 1.0
        incidence.at[line_id, line["bus1"]] = -1.0

    lodf = pd.DataFrame(
        np.nan,
        index=network.lines.index,
        columns=network.lines.index,
        dtype=float,
    )

    for outage_line in network.lines.index:
        if not _outage_keeps_network_connected(network, outage_line):
            continue

        transaction_ptdf = nodal_ptdf.dot(incidence.loc[outage_line])
        denominator = 1.0 - transaction_ptdf.at[outage_line]

        if abs(denominator) < 1e-8:
            continue

        lodf.loc[:, outage_line] = transaction_ptdf / denominator
        lodf.at[outage_line, outage_line] = -1.0

    return lodf


# ---------------------------------------------------------------------------
# GSKs and zonal PTDFs
# ---------------------------------------------------------------------------


def _renewable_mask(generators: pd.DataFrame) -> pd.Series:
    """Identify renewable units from optional generator metadata."""
    if "is_renewable" in generators.columns:
        values = generators["is_renewable"]

        def parse_value(value) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y"}
            return bool(value)

        return values.fillna(False).map(parse_value)

    metadata_column = None

    if "technology" in generators.columns:
        metadata_column = "technology"
    elif "carrier" in generators.columns:
        metadata_column = "carrier"

    if metadata_column is None:
        return pd.Series(False, index=generators.index)

    technology = (
        generators[metadata_column]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return technology.str.contains(
        "wind|solar|photovoltaic|pv|renewable|run.of.river",
        regex=True,
    )


def _flat_gen_buses(
    network: pypsa.Network,
    zone_buses: pd.Index,
) -> pd.Index:
    """
    Return buses with conventional generators.

    If no conventional generator exists in a zone, all zone buses are used.
    """
    renewable = _renewable_mask(network.generators)

    conventional_buses = pd.Index(
        network.generators.loc[~renewable, "bus"].unique()
    )

    selected_buses = zone_buses.intersection(conventional_buses)

    if selected_buses.empty:
        selected_buses = zone_buses

    return selected_buses


def create_gsk(
    network: pypsa.Network,
    zone_column: str,
    strategy: str = "flat_gen",
    base_case: dict | None = None,
    snapshot: Hashable | None = None,
) -> pd.DataFrame:
    """
    Create a GSK matrix with buses as rows and zones as columns.

    Supported strategies:
    - flat_node: equal weights over all nodes in a zone;
    - flat_gen: equal weights over conventional-generator nodes;
    - pnom: proportional to installed generator capacity;
    - pmax_remaining: proportional to unused D-2 generation capacity;
    - dispatch_prop: proportional to D-2 dispatch.

    The GSK is created for all zones because it is also used for physical
    CNE selection. Only FB-region columns are retained for the D-1 FB domain.
    """
    valid_strategies = {
        "flat_node",
        "flat_gen",
        "pnom",
        "pmax_remaining",
        "dispatch_prop",
    }

    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown GSK strategy '{strategy}'. "
            f"Use one of: {sorted(valid_strategies)}"
        )

    if strategy in {"pmax_remaining", "dispatch_prop"}:
        if base_case is None or snapshot is None:
            raise ValueError(
                f"GSK strategy '{strategy}' requires a base case and snapshot."
            )

    zones = pd.Index(network.buses[zone_column].unique())

    gsk = pd.DataFrame(
        0.0,
        index=network.buses.index,
        columns=zones,
        dtype=float,
    )

    renewable = _renewable_mask(network.generators)

    for zone in zones:
        zone_buses = network.buses.index[
            network.buses[zone_column] == zone
        ]

        if strategy == "flat_node":
            weights = pd.Series(
                1.0 / len(zone_buses),
                index=zone_buses,
                dtype=float,
            )

        elif strategy == "flat_gen":
            selected_buses = _flat_gen_buses(network, zone_buses)

            weights = pd.Series(
                1.0 / len(selected_buses),
                index=selected_buses,
                dtype=float,
            )

        else:
            zone_generators = network.generators.loc[
                network.generators["bus"].isin(zone_buses)
            ]

            # For capacity- and dispatch-dependent GSKs, renewable units are
            # excluded because GSKs represent expected market-driven changes.
            zone_generators = zone_generators.loc[
                ~renewable.reindex(zone_generators.index).fillna(False)
            ]

            if zone_generators.empty:
                selected_buses = _flat_gen_buses(network, zone_buses)

                weights = pd.Series(
                    1.0 / len(selected_buses),
                    index=selected_buses,
                    dtype=float,
                )
            else:
                if strategy == "pnom":
                    generator_weights = zone_generators["p_nom"].clip(lower=0.0)

                elif strategy == "pmax_remaining":
                    dispatch = base_case["dispatch"].loc[
                        snapshot,
                        zone_generators.index,
                    ]

                    availability = base_case["generator_availability"].loc[
                        snapshot,
                        zone_generators.index,
                    ]

                    generator_weights = (
                        zone_generators["p_nom"] * availability - dispatch
                    ).clip(lower=0.0)

                else:
                    generator_weights = base_case["dispatch"].loc[
                        snapshot,
                        zone_generators.index,
                    ].clip(lower=0.0)

                bus_weights = generator_weights.groupby(
                    zone_generators["bus"]
                ).sum()

                bus_weights = bus_weights.reindex(
                    zone_buses,
                    fill_value=0.0,
                )

                if bus_weights.sum() <= 1e-8:
                    selected_buses = _flat_gen_buses(network, zone_buses)

                    weights = pd.Series(
                        1.0 / len(selected_buses),
                        index=selected_buses,
                        dtype=float,
                    )
                else:
                    weights = bus_weights / bus_weights.sum()

        gsk.loc[weights.index, zone] = weights

    if not np.allclose(gsk.sum(axis=0).to_numpy(), 1.0):
        raise RuntimeError("Each GSK zone column must sum to one.")

    return gsk


def create_gsk_series(
    network: pypsa.Network,
    snapshots: pd.Index,
    zone_column: str,
    strategy: str,
    base_case: dict,
) -> dict:
    """Return one GSK matrix per snapshot."""
    static_strategies = {"flat_node", "flat_gen", "pnom"}

    if strategy in static_strategies:
        gsk = create_gsk(
            network=network,
            zone_column=zone_column,
            strategy=strategy,
        )

        return {snapshot: gsk.copy() for snapshot in snapshots}

    return {
        snapshot: create_gsk(
            network=network,
            zone_column=zone_column,
            strategy=strategy,
            base_case=base_case,
            snapshot=snapshot,
        )
        for snapshot in snapshots
    }


def create_zonal_ptdf(
    nodal_ptdf: pd.DataFrame,
    gsk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate zonal PTDFs.

    PTDF_zonal = PTDF_nodal @ GSK
    """
    missing_buses = nodal_ptdf.columns.difference(gsk.index)

    if not missing_buses.empty:
        raise ValueError(
            "The GSK does not contain all PTDF buses: "
            f"{list(missing_buses)}"
        )

    return nodal_ptdf.dot(gsk.reindex(nodal_ptdf.columns))


def create_zone_to_zone_ptdf(
    zonal_ptdf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate zone-to-zone PTDFs.

    Column ``(importing_zone, exporting_zone)`` gives the flow effect of a
    1 MW transfer from exporting_zone to importing_zone.
    """
    result = {}

    for importing_zone in zonal_ptdf.columns:
        for exporting_zone in zonal_ptdf.columns:
            if importing_zone == exporting_zone:
                continue

            result[(importing_zone, exporting_zone)] = (
                zonal_ptdf[importing_zone]
                - zonal_ptdf[exporting_zone]
            )

    return pd.DataFrame(result, index=zonal_ptdf.index)


# ---------------------------------------------------------------------------
# CNE selection and CNEC creation
# ---------------------------------------------------------------------------


def select_cnes(
    network: pypsa.Network,
    zone_to_zone_ptdf: dict,
    zone_column: str,
    flowbased_region: list,
    alpha: float = 0.05,
    include_cross_border_lines: bool = True,
) -> tuple[pd.Index, pd.Series]:
    """
    Select Critical Network Elements.

    A line is selected if at least one absolute zone-to-zone PTDF within the
    FB region reaches alpha. The CNE list is the union over all snapshots.

    Cross-border lines are automatically included only where both endpoint
    zones belong to the FB region.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a fraction in [0, 1].")

    sensitivity = pd.Series(
        0.0,
        index=network.lines.index,
        dtype=float,
    )

    for zone_to_zone in zone_to_zone_ptdf.values():
        max_sensitivity = zone_to_zone.abs().max(axis=1)

        sensitivity = pd.concat(
            [
                sensitivity,
                max_sensitivity.reindex(
                    network.lines.index,
                    fill_value=0.0,
                ),
            ],
            axis=1,
        ).max(axis=1)

    selected = sensitivity.index[sensitivity >= alpha]

    if include_cross_border_lines:
        bus0_zone = network.buses.loc[
            network.lines["bus0"],
            zone_column,
        ].to_numpy()

        bus1_zone = network.buses.loc[
            network.lines["bus1"],
            zone_column,
        ].to_numpy()

        both_in_fb_region = (
            pd.Series(bus0_zone).isin(flowbased_region).to_numpy()
            & pd.Series(bus1_zone).isin(flowbased_region).to_numpy()
        )

        cross_border = bus0_zone != bus1_zone

        selected_cross_border = network.lines.index[
            cross_border & both_in_fb_region
        ]

        selected = selected.union(selected_cross_border)

    cnes = network.lines.index[network.lines.index.isin(selected)]

    return cnes, sensitivity


def create_cnecs(
    network: pypsa.Network,
    nodal_ptdf: pd.DataFrame,
    cnes: pd.Index,
    lodf_sensitivity: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create directional N-0 and N-1 CNECs.

    Every selected CNE creates:
    - one positive-flow CNEC;
    - one negative-flow CNEC.

    Additional N-1 CNECs are created for outages satisfying:

        abs(LODF[cne, outage]) * capacity[outage]
        >= lodf_sensitivity * capacity[cne]

    The returned CNEC PTDF rows are already directional, so every later
    FBMC constraint has one positive RAM upper bound.
    """
    if not 0.0 <= lodf_sensitivity <= 1.0:
        raise ValueError("lodf_sensitivity must be a fraction in [0, 1].")

    lodf = calculate_lodf(network, nodal_ptdf)
    capacity = _line_capacity(network, pd.Index([0])).loc[0]

    if "contingency" in network.lines.columns:
        contingency_allowed = network.lines["contingency"].fillna(False).astype(bool)
    else:
        contingency_allowed = pd.Series(True, index=network.lines.index)

    info_rows = []
    ptdf_rows = []
    constraint_ids = []

    def add_directional_rows(
        cne,
        contingency,
        lodf_factor,
        base_ptdf,
    ) -> None:
        for sign, direction in [(1.0, "positive"), (-1.0, "negative")]:
            cnec_id = f"CNEC_{len(constraint_ids):05d}"

            constraint_ids.append(cnec_id)

            ptdf_rows.append(
                sign * base_ptdf.reindex(network.buses.index).to_numpy()
            )

            info_rows.append(
                {
                    "cne": cne,
                    "contingency": contingency,
                    "direction": direction,
                    "sign": sign,
                    "lodf": float(lodf_factor),
                    "capacity": float(capacity[cne]),
                }
            )

    for cne in cnes:
        add_directional_rows(
            cne=cne,
            contingency=BASECASE_CONTINGENCY,
            lodf_factor=0.0,
            base_ptdf=nodal_ptdf.loc[cne],
        )

        for outage in network.lines.index:
            if outage == cne:
                continue

            if not contingency_allowed.at[outage]:
                continue

            lodf_factor = lodf.at[cne, outage]

            if pd.isna(lodf_factor):
                continue

            impact = abs(lodf_factor) * capacity[outage]

            if impact < lodf_sensitivity * capacity[cne]:
                continue

            contingency_ptdf = (
                nodal_ptdf.loc[cne]
                + lodf_factor * nodal_ptdf.loc[outage]
            )

            add_directional_rows(
                cne=cne,
                contingency=outage,
                lodf_factor=lodf_factor,
                base_ptdf=contingency_ptdf,
            )

    nodal_cnec_ptdf = pd.DataFrame(
        ptdf_rows,
        index=constraint_ids,
        columns=network.buses.index,
        dtype=float,
    )

    cnec_info = pd.DataFrame(info_rows, index=constraint_ids)
    cnec_info.index.name = "cnec"

    return nodal_cnec_ptdf, cnec_info, lodf


# ---------------------------------------------------------------------------
# Direct RAM calculation
# ---------------------------------------------------------------------------


def calculate_cnec_reference_flows(
    base_case_flows: pd.DataFrame,
    cnec_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate signed D-2 flows for every directional CNEC.

    For an N-1 CNEC:

        F_cne,outage = F_cne + LODF_cne,outage * F_outage
    """
    reference_flows = pd.DataFrame(
        0.0,
        index=base_case_flows.index,
        columns=cnec_info.index,
        dtype=float,
    )

    for cnec, row in cnec_info.iterrows():
        flow = base_case_flows[row["cne"]].copy()

        if row["contingency"] != BASECASE_CONTINGENCY:
            flow += row["lodf"] * base_case_flows[row["contingency"]]

        reference_flows[cnec] = row["sign"] * flow

    return reference_flows


def _fb_reference_net_positions(
    zonal_net_positions: pd.DataFrame,
    market_zones: pd.Index,
    flowbased_region: pd.Index,
) -> tuple[pd.DataFrame, bool]:
    """
    Return D-2 FB reference net positions where they are unambiguous.

    If every market zone is in the FB region, ordinary zonal D-2 net positions
    are equal to the FB-region reference net positions.

    If external zones exist, physical flows crossing the region boundary are
    not interpreted as commercial ATC exchanges. Therefore, no FB reference
    net position is inferred. A zero vector is used; external base-case
    effects remain in f_ref.
    """
    if set(flowbased_region) == set(market_zones):
        return (
            zonal_net_positions.reindex(columns=flowbased_region, fill_value=0.0),
            True,
        )

    return (
        pd.DataFrame(
            0.0,
            index=zonal_net_positions.index,
            columns=flowbased_region,
            dtype=float,
        ),
        False,
    )


def calculate_direct_ram(
    reference_flows: pd.DataFrame,
    fb_reference_net_positions: pd.DataFrame,
    zonal_ptdf: dict,
    cnec_capacity: pd.Series,
    frm: float,
    min_ram: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate direct RAM values in MW.

    For every directional CNEC:

        f_ref = f_D2 - zPTDF * NP_FB,D2

        RAM = max(
            minRAM * capacity,
            capacity - FRM - f_ref
        )

    The later market-clearing constraint is:

        sum(
            zPTDF[t, cnec, fb_zone] * NET_EXPORT_FB[t, fb_zone]
            for fb_zone in flowbased_region
        ) <= RAM[t, cnec]

    No quantity is scaled by 0.25. PTDFs, RAM, capacities, and net positions
    remain MW-based.
    """
    if not 0.0 <= frm < 1.0:
        raise ValueError("frm must be a fraction in [0, 1).")

    if not 0.0 <= min_ram < 1.0:
        raise ValueError("min_ram must be a fraction in [0, 1).")

    snapshots = reference_flows.index
    cnecs = reference_flows.columns
    fb_zones = fb_reference_net_positions.columns

    capacity = cnec_capacity.reindex(cnecs)

    if capacity.isna().any():
        raise ValueError("A CNEC capacity is missing.")

    frm_mw = pd.DataFrame(
        np.tile((capacity * frm).to_numpy(), (len(snapshots), 1)),
        index=snapshots,
        columns=cnecs,
        dtype=float,
    )

    reference_flow_without_market = pd.DataFrame(
        0.0,
        index=snapshots,
        columns=cnecs,
        dtype=float,
    )

    ram_before_minram = pd.DataFrame(
        0.0,
        index=snapshots,
        columns=cnecs,
        dtype=float,
    )

    ram = pd.DataFrame(
        0.0,
        index=snapshots,
        columns=cnecs,
        dtype=float,
    )

    for snapshot in snapshots:
        ptdf = zonal_ptdf[snapshot].loc[cnecs, fb_zones]
        np_ref = fb_reference_net_positions.loc[snapshot, fb_zones]

        explained_fb_flow = ptdf.dot(np_ref)
        f_ref = reference_flows.loc[snapshot] - explained_fb_flow
        raw_ram = capacity - frm_mw.loc[snapshot] - f_ref

        reference_flow_without_market.loc[snapshot] = f_ref
        ram_before_minram.loc[snapshot] = raw_ram

        ram.loc[snapshot] = pd.concat(
            [
                raw_ram,
                min_ram * capacity,
            ],
            axis=1,
        ).max(axis=1)

    return (
        ram,
        frm_mw,
        reference_flow_without_market,
        ram_before_minram,
    )


# ---------------------------------------------------------------------------
# Full FBMC pre-computation pipeline
# ---------------------------------------------------------------------------


def precompute_fbmc_parameters(
    network: pypsa.Network,
    alpha: float = 0.25, # default 0.05
    frm: float = 0.10,  # default 0.10
    min_ram: float = 0.0, # default 0.10
    lodf_sensitivity: float = 0.50, # default 0.10
    gsk_cne_strategy: str = "flat_gen",
    gsk_market_strategy: str = "flat_gen",
    include_cross_border_lines: bool = True,
    flowbased_region: list | None = None,
    zone_column: str = "zone_id",
    snapshots: pd.Index | None = None,
    solver_name: str | None = None,
) -> dict:
    """
    Run the complete FBMC parameter pre-computation workflow.

    Only unique, consecutive, fixed 15-minute snapshots are accepted.

    The returned FB domain uses the direct MW-based representation:

        sum(
            zPTDF[t, cnec, fb_zone] * NET_EXPORT_FB[t, fb_zone]
            for fb_zone in flowbased_region
        ) <= RAM[t, cnec]

    Args:
        network:
            Existing PyPSA network. It is never mutated.
        alpha:
            CNE selection threshold in [0, 1].
        frm:
            Flow Reliability Margin as a fraction of CNEC capacity.
        min_ram:
            Minimum RAM as a fraction of CNEC capacity.
        lodf_sensitivity:
            An outage is included for a CNE where:

                abs(LODF[cne, outage]) * capacity[outage]
                >= lodf_sensitivity * capacity[cne]

        gsk_cne_strategy:
            GSK strategy used for CNE selection.
        gsk_market_strategy:
            GSK strategy used for D-1 FB PTDFs.
        include_cross_border_lines:
            Include cross-border lines where both endpoint zones are inside the
            FB region.
        flowbased_region:
            Zones represented by FB net-export variables. Defaults to all
            market zones.
        zone_column:
            Column in ``network.buses`` containing bidding-zone IDs.
        snapshots:
            Optional fixed 15-minute snapshots. Defaults to
            ``network.snapshots``.
        solver_name:
            Optional PyPSA/linopy solver name.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a fraction in [0, 1].")

    if not 0.0 <= frm < 1.0:
        raise ValueError("frm must be a fraction in [0, 1).")

    if not 0.0 <= min_ram < 1.0:
        raise ValueError("min_ram must be a fraction in [0, 1).")

    if not 0.0 <= lodf_sensitivity <= 1.0:
        raise ValueError("lodf_sensitivity must be a fraction in [0, 1].")

    _validate_network(network, zone_column)
    snapshots = _resolve_snapshots(network, snapshots)

    market_zones = pd.Index(network.buses[zone_column].unique())

    if flowbased_region is None:
        flowbased_region = list(market_zones)
    else:
        flowbased_region = list(flowbased_region)

    fb_zones = pd.Index(flowbased_region)

    if fb_zones.has_duplicates:
        raise ValueError("flowbased_region must not contain duplicate zones.")

    unknown_fb_zones = fb_zones.difference(market_zones)

    if not unknown_fb_zones.empty:
        raise ValueError(
            "flowbased_region contains unknown zones: "
            f"{list(unknown_fb_zones)}"
        )

    if len(fb_zones) < 2:
        raise ValueError(
            "flowbased_region must contain at least two bidding zones."
        )

    # 1. Initial D-2 base case. This is needed for dynamic GSK strategies.
    initial_base_case = solve_d2_base_case(
        network=network,
        snapshots=snapshots,
        zone_column=zone_column,
        solver_name=solver_name,
    )

    # 2. Nodal PTDFs.
    nodal_ptdf = calculate_nodal_ptdf(network)

    # 3. CNE-selection GSKs and zone-to-zone PTDFs.
    gsk_cne = create_gsk_series(
        network=initial_base_case["network"],
        snapshots=snapshots,
        zone_column=zone_column,
        strategy=gsk_cne_strategy,
        base_case=initial_base_case,
    )

    zonal_ptdf_cne = {
        snapshot: create_zonal_ptdf(
            nodal_ptdf=nodal_ptdf,
            gsk=gsk_cne[snapshot],
        ).loc[:, fb_zones]
        for snapshot in snapshots
    }

    zone_to_zone_ptdf = {
        snapshot: create_zone_to_zone_ptdf(zonal_ptdf_cne[snapshot])
        for snapshot in snapshots
    }

    # 4. CNE selection.
    cnes, cne_sensitivity = select_cnes(
        network=network,
        zone_to_zone_ptdf=zone_to_zone_ptdf,
        zone_column=zone_column,
        flowbased_region=list(fb_zones),
        alpha=alpha,
        include_cross_border_lines=include_cross_border_lines,
    )

    if cnes.empty:
        raise RuntimeError(
            "No CNEs were selected. Lower alpha or enable "
            "include_cross_border_lines."
        )

    logger.info(
        "FBMC selected %s CNEs out of %s lines.",
        len(cnes),
        len(network.lines.index),
    )

    # 5. Directional N-0 and N-1 CNECs.
    nodal_cnec_ptdf, cnec_info, lodf = create_cnecs(
        network=network,
        nodal_ptdf=nodal_ptdf,
        cnes=cnes,
        lodf_sensitivity=lodf_sensitivity,
    )

    cnecs = cnec_info.index

    logger.info(
        "FBMC created %s directional CNECs from %s CNEs.",
        len(cnecs),
        len(cnes),
    )

    # 6. Final D-2 base case with FRM reserved on normal-state CNEs.
    final_base_case = solve_d2_base_case(
        network=network,
        snapshots=snapshots,
        zone_column=zone_column,
        frm=frm,
        cnes=cnes,
        solver_name=solver_name,
    )

    # 7. Market GSKs and CNEC PTDFs. Only FB-zone columns are retained.
    gsk_market = create_gsk_series(
        network=final_base_case["network"],
        snapshots=snapshots,
        zone_column=zone_column,
        strategy=gsk_market_strategy,
        base_case=final_base_case,
    )

    zonal_ptdf = {
        snapshot: nodal_cnec_ptdf.dot(
            gsk_market[snapshot].reindex(nodal_cnec_ptdf.columns)
        ).loc[:, fb_zones]
        for snapshot in snapshots
    }

    # 8. D-2 diagnostics and FB reference values.
    d2_zonal_net_positions = final_base_case["zonal_net_positions"].reindex(
        columns=market_zones,
        fill_value=0.0,
    )

    fb_reference_net_positions, fb_reference_np_available = (
        _fb_reference_net_positions(
            zonal_net_positions=d2_zonal_net_positions,
            market_zones=market_zones,
            flowbased_region=fb_zones,
        )
    )

    if not fb_reference_np_available:
        logger.info(
            "The FB region does not include all market zones. Zero FB "
            "reference net positions are used. External base-case effects "
            "remain embedded in f_ref."
        )

    reference_flows = calculate_cnec_reference_flows(
        base_case_flows=final_base_case["flows"],
        cnec_info=cnec_info,
    )

    # 9. Direct RAM calculation.
    (
        ram,
        frm_mw,
        reference_flow_without_market,
        ram_before_minram,
    ) = calculate_direct_ram(
        reference_flows=reference_flows,
        fb_reference_net_positions=fb_reference_net_positions,
        zonal_ptdf=zonal_ptdf,
        cnec_capacity=cnec_info["capacity"],
        frm=frm,
        min_ram=min_ram,
    )

    # Diagnostic: check the D-2 FB reference point against the FB domain.
    max_reference_violation = 0.0

    for snapshot in snapshots:
        predicted_fb_flow = zonal_ptdf[snapshot].dot(
            fb_reference_net_positions.loc[snapshot]
        )

        violation = (predicted_fb_flow - ram.loc[snapshot]).max()

        max_reference_violation = max(
            max_reference_violation,
            float(violation),
        )

    if max_reference_violation > 1e-6:
        logger.warning(
            "The D-2 FB reference point violates at least one CNEC "
            "constraint by up to %.3f MW.",
            max_reference_violation,
        )

    return {
        "market_zones": market_zones,
        "flowbased_region": fb_zones,
        "snapshots": snapshots,
        "cnes": cnes,
        "cnecs": cnecs,
        "alpha": alpha,
        "frm": frm,
        "min_ram": min_ram,
        "lodf_sensitivity": lodf_sensitivity,
        "nodal_ptdf": nodal_ptdf,
        "nodal_cnec_ptdf": nodal_cnec_ptdf,
        "lodf": lodf,
        "zonal_ptdf": zonal_ptdf,
        "gsk_cne": gsk_cne,
        "gsk_market": gsk_market,
        "cne_sensitivity": cne_sensitivity,
        "cnec_info": cnec_info,
        "reference_flows": reference_flows,
        "reference_flow_without_market": reference_flow_without_market,
        "d2_zonal_net_positions": d2_zonal_net_positions,
        "fb_reference_net_positions": fb_reference_net_positions,
        "fb_reference_np_available": fb_reference_np_available,
        "ram": ram,
        "ram_before_minram": ram_before_minram,
        "frm_mw": frm_mw,
        "line_capacity": cnec_info["capacity"],
        "base_case": final_base_case,
    }


# ---------------------------------------------------------------------------
# Conversion for the EUPHEMIA-like clearing model
# ---------------------------------------------------------------------------


def to_clearing_input(fbmc_parameters: dict) -> dict:
    """
    Convert FBMC results to tuple-indexed dictionaries for Pyomo.

    The later clearing algorithm must implement the MW-based constraint:

        sum(
            ptdf_zonal[t, cnec, fb_zone]
            * net_export_fb[t, fb_zone]
            for fb_zone in fbmc_zone_ids
        ) <= ram[t, cnec]

    ``net_export_fb`` is intentionally distinct from total zonal net position.
    """
    ptdf_zonal = {}
    ram = {}

    for snapshot in fbmc_parameters["snapshots"]:
        ptdf = fbmc_parameters["zonal_ptdf"][snapshot]

        for cnec in fbmc_parameters["cnecs"]:
            ram[(snapshot, cnec)] = float(
                fbmc_parameters["ram"].at[snapshot, cnec]
            )

            for fb_zone in fbmc_parameters["flowbased_region"]:
                ptdf_zonal[(snapshot, cnec, fb_zone)] = float(
                    ptdf.at[cnec, fb_zone]
                )

    return {
        "market_zone_ids": list(fbmc_parameters["market_zones"]),
        "fbmc_zone_ids": list(fbmc_parameters["flowbased_region"]),
        "cne_list": list(fbmc_parameters["cnes"]),
        "cnec_list": list(fbmc_parameters["cnecs"]),
        "snapshots": list(fbmc_parameters["snapshots"]),
        "ptdf_zonal": ptdf_zonal,
        "ram": ram,
    }