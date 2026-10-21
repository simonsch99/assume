# test_euphemia_like_tools.py
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared tooling for the EUPHEMIA-like validation suite.

This module replaces both a ``conftest.py`` and a separate utility module.
It provides:

- fixed 15-minute market-product factories;
- a valid ``MarketConfig`` factory for ``ComplexClearingRole``;
- order factories for simple bids (SB), block bids (BB), linked blocks (LB)
  and storage orders (SO);
- a role builder and clearing wrapper;
- in-memory NTC input and minimal in-memory PyPSA networks for FBMC;
- result-extraction helpers keyed by ``bid_id``, zone and timestamp;
- independent feasibility checks (zonal balance, NTC bounds, storage SOC
  recursion, FBMC RAM feasibility, simple-order price consistency and
  block-surplus conditions).

It deliberately contains no test cases.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr
from pyomo.opt import SolverFactory

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets.clearing_algorithms.euphemia_like import ComplexClearingRole

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Shared numerical tolerance used by all assertions of this suite.
EPS = 1e-5

#: Slightly looser tolerance for quantities obtained from LP/QP price stages.
PRICE_TOL = 1e-4

MTU = timedelta(minutes=15)
MTU_HOURS = 0.25

DEFAULT_START = datetime(2025, 1, 1, 0, 0)

CANDIDATE_SOLVERS = ("gurobi", "appsi_highs", "highs", "cbc", "glpk")


# ---------------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------------


def _solver_is_available(name: str) -> bool:
    """Return whether a Pyomo solver interface is usable."""
    try:
        solver = SolverFactory(name)
    except Exception:
        return False

    if solver is None:
        return False

    try:
        return bool(solver.available(exception_flag=False))
    except TypeError:
        # APPSI interfaces do not accept ``exception_flag``.
        try:
            return bool(solver.available())
        except Exception:
            return False
    except Exception:
        return False


def available_solver_name() -> str | None:
    """Return the first locally available supported Pyomo solver."""
    for name in CANDIDATE_SOLVERS:
        if _solver_is_available(name):
            return name
    return None


SOLVER_NAME = available_solver_name()

#: Decorator used by all test modules to skip cleanly without a solver.
requires_solver = pytest.mark.skipif(
    SOLVER_NAME is None,
    reason=(
        "No supported Pyomo solver installed "
        f"(tried: {', '.join(CANDIDATE_SOLVERS)})."
    ),
)


@pytest.fixture(scope="session")
def solver_name() -> str:
    """Session fixture exposing the detected solver name."""
    if SOLVER_NAME is None:
        pytest.skip("No supported Pyomo solver installed.")
    return SOLVER_NAME


@pytest.fixture
def base_start() -> datetime:
    """Deterministic market start timestamp aligned to a 15-minute boundary."""
    return DEFAULT_START


# ---------------------------------------------------------------------------
# Market products and market configuration
# ---------------------------------------------------------------------------


def make_products(count: int, start: datetime = DEFAULT_START) -> list[tuple]:
    """
    Create ``count`` consecutive fixed 15-minute market products.

    Products follow the ``(start, end, only_hours)`` convention used by
    ``MarketRole.clear``.
    """
    return [
        (start + index * MTU, start + (index + 1) * MTU, None)
        for index in range(count)
    ]


def product_timestamps(products: list[tuple]) -> list[datetime]:
    """Return the MTU start timestamps of market products."""
    return [product[0] for product in products]


def make_market_config(
    products: list[tuple],
    param_dict: dict | None = None,
    market_id: str = "euphemia_like_test",
) -> MarketConfig:
    """Create a valid MarketConfig for the EUPHEMIA-like clearing role."""
    start = products[0][0]

    config = MarketConfig(
        market_id=market_id,
        market_products=[MarketProduct(MTU, len(products), MTU)],
        additional_fields=[
            "bid_type",
            "node",
            "min_acceptance_ratio",
            "parent_bid_id",
            "exclusive_group_id",
        ],
        opening_hours=rr.rrule(
            rr.MINUTELY,
            interval=15,
            dtstart=start - timedelta(hours=1),
            until=start + timedelta(days=1),
            cache=True,
        ),
        opening_duration=MTU,
        volume_unit="MW",
        price_unit="EUR/MWh",
        maximum_bid_volume=1e6,
        market_mechanism="complex_clearing",
    )

    defaults = {
        "solver_name": SOLVER_NAME or "gurobi",
        "log_flows": True,
        "log_performance": False,
        "price_floor": -500.0,
        "price_cap": 4000.0,
    }
    defaults.update(param_dict or {})
    config.param_dict = defaults

    return config


# ---------------------------------------------------------------------------
# Order factories
# ---------------------------------------------------------------------------


def make_simple_bid(
    bid_id: str,
    timestamp: datetime,
    volume: float,
    price: float,
    node: str = "Z1",
    agent_addr: str | None = None,
) -> dict:
    """
    Create a simple period order (SB).

    Positive volumes are supply, negative volumes are demand.
    """
    return {
        "bid_id": bid_id,
        "agent_addr": agent_addr or bid_id,
        "bid_type": "SB",
        "node": node,
        "start_time": timestamp,
        "end_time": timestamp + MTU,
        "only_hours": None,
        "volume": float(volume),
        "price": float(price),
        "min_acceptance_ratio": None,
        "parent_bid_id": None,
        "exclusive_group_id": None,
    }


def make_block_bid(
    bid_id: str,
    volumes: dict,
    price: float,
    node: str = "Z1",
    min_acceptance_ratio: float = 1.0,
    exclusive_group_id=None,
    parent_bid_id: str | None = None,
    bid_type: str = "BB",
    agent_addr: str | None = None,
) -> dict:
    """Create a block bid (BB) or a linked block bid (LB)."""
    timestamps = sorted(volumes)

    return {
        "bid_id": bid_id,
        "agent_addr": agent_addr or bid_id,
        "bid_type": bid_type,
        "node": node,
        "start_time": timestamps[0],
        "end_time": timestamps[-1] + MTU,
        "only_hours": None,
        "volume": {
            timestamp: float(volumes[timestamp]) for timestamp in timestamps
        },
        "price": float(price),
        "min_acceptance_ratio": float(min_acceptance_ratio),
        "parent_bid_id": parent_bid_id,
        "exclusive_group_id": exclusive_group_id,
    }


def make_linked_bid(
    bid_id: str,
    volumes: dict,
    price: float,
    parent_bid_id: str,
    node: str = "Z1",
    min_acceptance_ratio: float = 1.0,
    exclusive_group_id=None,
) -> dict:
    """Create a linked block bid (LB) with an explicit parent."""
    return make_block_bid(
        bid_id=bid_id,
        volumes=volumes,
        price=price,
        node=node,
        min_acceptance_ratio=min_acceptance_ratio,
        exclusive_group_id=exclusive_group_id,
        parent_bid_id=parent_bid_id,
        bid_type="LB",
    )


def _expand_series(value, timestamps: list[datetime]) -> dict:
    """Expand a scalar or mapping into a complete timestamp mapping."""
    if isinstance(value, dict):
        return {timestamp: float(value[timestamp]) for timestamp in timestamps}
    return {timestamp: float(value) for timestamp in timestamps}


def make_storage_order(
    bid_id: str,
    timestamps: list[datetime],
    max_energy,
    max_input_power,
    max_output_power,
    node: str = "Z1",
    min_energy=0.0,
    starting_energy: float = 0.0,
    efficiency: float = 1.0,
    cycling_cost: float = 0.0,
    end_of_day_value: float = 0.0,
    agent_addr: str | None = None,
) -> dict:
    """
    Create a Storage Order (SO).

    Energy fields are MWh, power fields are MW, cycling cost and terminal
    value are EUR/MWh.
    """
    return {
        "bid_id": bid_id,
        "agent_addr": agent_addr or bid_id,
        "bid_type": "SO",
        "node": node,
        "start_time": timestamps[0],
        "end_time": timestamps[-1] + MTU,
        "only_hours": None,
        "volume": 0.0,
        "price": 0.0,
        "max_energy": _expand_series(max_energy, timestamps),
        "min_energy": _expand_series(min_energy, timestamps),
        "max_input_power": _expand_series(max_input_power, timestamps),
        "max_output_power": _expand_series(max_output_power, timestamps),
        "starting_energy": float(starting_energy),
        "efficiency": float(efficiency),
        "cycling_cost": float(cycling_cost),
        "end_of_day_value": float(end_of_day_value),
        "min_acceptance_ratio": None,
        "parent_bid_id": None,
        "exclusive_group_id": None,
    }


# ---------------------------------------------------------------------------
# Role building and clearing
# ---------------------------------------------------------------------------


def make_role(config: MarketConfig) -> ComplexClearingRole:
    """Build the clearing role, skipping cleanly without a usable solver."""
    try:
        role = ComplexClearingRole(config)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ComplexClearingRole is not usable: {exc}")

    if not _solver_is_available(role.solver_name):  # pragma: no cover
        pytest.skip(f"Solver '{role.solver_name}' is not available.")

    return role


def clear_market(
    config: MarketConfig,
    orders: list[dict],
    products: list[tuple],
) -> dict:
    """
    Run one clearing and return a result bundle.

    The returned dictionary contains the role, accepted/rejected orders,
    meta records, flow results and the (mutated) input orderbook.
    """
    role = make_role(config)
    accepted, rejected, meta, flows = role.clear(orders, products)

    return {
        "role": role,
        "orders": orders,
        "accepted": accepted,
        "rejected": rejected,
        "meta": meta,
        "flows": flows,
    }


# ---------------------------------------------------------------------------
# NTC inputs
# ---------------------------------------------------------------------------


def make_ntc_dataframe(
    timestamps: list[datetime],
    lines: list[dict],
) -> pd.DataFrame:
    """
    Build a time-dependent in-memory NTC DataFrame.

    Each line dictionary requires ``line_id``, ``from_zone``, ``to_zone``,
    ``capacity_up`` and ``capacity_down``. Capacities may be scalars or
    timestamp mappings.
    """
    rows = []

    for line in lines:
        capacity_up = _expand_series(line["capacity_up"], timestamps)
        capacity_down = _expand_series(line["capacity_down"], timestamps)

        for timestamp in timestamps:
            rows.append(
                {
                    "line_id": line["line_id"],
                    "from_zone": line["from_zone"],
                    "to_zone": line["to_zone"],
                    "timestamp": timestamp,
                    "capacity_up": capacity_up[timestamp],
                    "capacity_down": capacity_down[timestamp],
                }
            )

    return pd.DataFrame(rows)


def ntc_line_definition(ntc_data: pd.DataFrame, line_id) -> dict:
    """Return ``from_zone``/``to_zone`` for one NTC line."""
    line_rows = ntc_data[ntc_data["line_id"] == line_id]

    return {
        "from_zone": line_rows["from_zone"].iloc[0],
        "to_zone": line_rows["to_zone"].iloc[0],
    }


def ntc_capacity(ntc_data: pd.DataFrame, line_id, timestamp) -> tuple:
    """Return ``(capacity_up, capacity_down)`` for one line and MTU."""
    row = ntc_data[
        (ntc_data["line_id"] == line_id)
        & (pd.DatetimeIndex(ntc_data["timestamp"]) == pd.Timestamp(timestamp))
    ].iloc[0]

    return float(row["capacity_up"]), float(row["capacity_down"])


# ---------------------------------------------------------------------------
# Minimal in-memory PyPSA networks for FBMC
# ---------------------------------------------------------------------------


def require_linopy_solver() -> None:
    """Skip cleanly if no linopy/PyPSA solver is installed."""
    try:
        from assume.common.grid_utils import get_supported_solver_linopy

        get_supported_solver_linopy(None)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"No supported PyPSA/linopy solver available: {exc}")


def make_two_zone_network(
    timestamps: list[datetime],
    s_nom: float,
    zone_a: str = "ZA",
    zone_b: str = "ZB",
    load_a: float = 100.0,
    load_b: float = 300.0,
):
    """
    Build a minimal two-bus/two-zone PyPSA network for FBMC tests.

    One bus per bidding zone makes the GSK an identity mapping, so the
    zonal PTDF equals the nodal PTDF and the D-2 reference flow explained
    by FB net positions is exact. Consequently:

        RAM = (1 - FRM) * s_nom = 0.9 * s_nom

    for both directional CNECs of the single cross-border line. Local
    generation at both buses guarantees a feasible D-2 base case.
    """
    pypsa = pytest.importorskip("pypsa")

    network = pypsa.Network()
    network.set_snapshots(pd.DatetimeIndex(timestamps))

    network.add("Bus", "bus_a", v_nom=380.0)
    network.add("Bus", "bus_b", v_nom=380.0)
    network.buses["zone_id"] = pd.Series(
        {"bus_a": zone_a, "bus_b": zone_b}
    )

    network.add(
        "Line",
        "line_ab",
        bus0="bus_a",
        bus1="bus_b",
        x=0.1,
        r=0.0,
        s_nom=float(s_nom),
    )

    network.add(
        "Generator",
        "gen_a",
        bus="bus_a",
        p_nom=1000.0,
        marginal_cost=10.0,
    )
    network.add(
        "Generator",
        "gen_b",
        bus="bus_b",
        p_nom=1000.0,
        marginal_cost=100.0,
    )

    network.add("Load", "load_a", bus="bus_a", p_set=float(load_a))
    network.add("Load", "load_b", bus="bus_b", p_set=float(load_b))

    return network


def precompute_fbmc_clearing_input(
    network,
    flowbased_region: list[str],
    timestamps: list[datetime],
) -> dict:
    """
    Independently recompute the FB domain used by the clearing role.

    The result is the tuple-indexed clearing input containing ``ptdf_zonal``
    and ``ram``.
    """
    require_linopy_solver()

    from assume.common.fbmc_utils import (
        precompute_fbmc_parameters,
        to_clearing_input,
    )

    parameters = precompute_fbmc_parameters(
        network=network,
        flowbased_region=list(flowbased_region),
        zone_column="zone_id",
        snapshots=pd.DatetimeIndex(timestamps),
    )

    return to_clearing_input(parameters)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def orders_by_id(orders: list[dict]) -> dict:
    """Index an order list by ``bid_id``."""
    return {order["bid_id"]: order for order in orders}


def find_order(result: dict, bid_id: str) -> dict:
    """Return an order from the cleared orderbook by ``bid_id``."""
    for order in result["orders"]:
        if order["bid_id"] == bid_id:
            return order

    raise AssertionError(f"Order '{bid_id}' is not part of the orderbook.")


def is_accepted(result: dict, bid_id: str) -> bool:
    """Return whether ``bid_id`` appears in the accepted orders."""
    return bid_id in orders_by_id(result["accepted"])


def is_rejected(result: dict, bid_id: str) -> bool:
    """Return whether ``bid_id`` appears in the rejected orders."""
    return bid_id in orders_by_id(result["rejected"])


def get_accepted_order(result: dict, bid_id: str) -> dict:
    """Return an accepted order, asserting its presence."""
    accepted = orders_by_id(result["accepted"])
    assert bid_id in accepted, f"Order '{bid_id}' was expected to be accepted."
    return accepted[bid_id]


def get_rejected_order(result: dict, bid_id: str) -> dict:
    """Return a rejected order, asserting its presence."""
    rejected = orders_by_id(result["rejected"])
    assert bid_id in rejected, f"Order '{bid_id}' was expected to be rejected."
    return rejected[bid_id]


def accepted_volume(result: dict, bid_id: str, timestamp=None) -> float:
    """Return the accepted MW volume of any order type."""
    order = find_order(result, bid_id)
    volume = order["accepted_volume"]

    if isinstance(volume, dict):
        if timestamp is None:
            raise ValueError("A timestamp is required for profile orders.")
        return float(volume.get(timestamp, 0.0))

    return float(volume)


def acceptance_ratio(result: dict, bid_id: str) -> float:
    """Return the acceptance ratio of a block or linked block bid."""
    order = find_order(result, bid_id)
    ratios = []

    for timestamp, volume in order["volume"].items():
        if abs(volume) <= EPS:
            continue
        ratios.append(
            order["accepted_volume"].get(timestamp, 0.0) / volume
        )

    if not ratios:
        return 0.0

    assert max(ratios) - min(ratios) <= 1e-4, (
        f"Block '{bid_id}' has inconsistent acceptance ratios: {ratios}"
    )

    return float(ratios[0])


def get_price(meta: list[dict], zone: str, timestamp) -> float:
    """Extract the market clearing price for one zone and MTU from meta."""
    for record in meta:
        if record["zone"] == zone and record["product_start"] == timestamp:
            return float(record["price"])

    raise AssertionError(f"No meta price for zone '{zone}' at '{timestamp}'.")


def get_meta_record(meta: list[dict], zone: str, timestamp) -> dict:
    """Extract the full meta record for one zone and MTU."""
    for record in meta:
        if record["zone"] == zone and record["product_start"] == timestamp:
            return record

    raise AssertionError(f"No meta record for zone '{zone}' at '{timestamp}'.")


def get_ntc_flow(flows: dict, timestamp, line_id) -> float:
    """Extract one NTC flow in MW from ``flows['ntc_flows']``."""
    ntc_flows = flows.get("ntc_flows", {})
    key = (timestamp, line_id)

    assert key in ntc_flows, f"No NTC flow logged for {key}."

    return float(ntc_flows[key])


def get_fb_net_export(flows: dict, timestamp, zone: str) -> float:
    """Extract one FB net export in MW from ``flows['fb_net_exports']``."""
    fb_net_exports = flows.get("fb_net_exports", {})
    key = (timestamp, zone)

    assert key in fb_net_exports, f"No FB net export logged for {key}."

    return float(fb_net_exports[key])


# ---------------------------------------------------------------------------
# Independent feasibility checks
# ---------------------------------------------------------------------------


def zone_net_position(result: dict, zone: str, timestamp) -> float:
    """
    Compute accepted supply minus demand in one zone and MTU.

    Only accepted orders are used. Storage orders contribute
    ``discharge - charge``.
    """
    total = 0.0

    for order in result["accepted"]:
        if order.get("node") != zone:
            continue

        volume = order["accepted_volume"]

        if isinstance(volume, dict):
            total += float(volume.get(timestamp, 0.0))
        elif order["start_time"] == timestamp:
            total += float(volume)

    return total


def assert_zone_balance(
    result: dict,
    zone: str,
    timestamp,
    expected_export: float = 0.0,
    tol: float = 1e-4,
) -> None:
    """Independently assert accepted supply/demand balance in one zone."""
    net_position = zone_net_position(result, zone, timestamp)

    assert net_position == pytest.approx(expected_export, abs=tol), (
        f"Zone '{zone}' at '{timestamp}' has net position {net_position} MW, "
        f"expected {expected_export} MW."
    )


def assert_isolated_zone_balances(
    result: dict,
    zones: list[str],
    timestamps: list[datetime],
    tol: float = 1e-4,
) -> None:
    """Assert zero net position for all zones without any coupling."""
    for zone in zones:
        for timestamp in timestamps:
            assert_zone_balance(result, zone, timestamp, 0.0, tol)


def assert_ntc_balance(
    result: dict,
    ntc_data: pd.DataFrame,
    zones: list[str],
    timestamps: list[datetime],
    tol: float = 1e-4,
) -> None:
    """
    Assert zonal balance including commercial NTC exchange.

    For every zone and MTU:

        net_position = sum(outgoing flows) - sum(incoming flows)
    """
    line_ids = list(pd.Index(ntc_data["line_id"]).unique())

    for timestamp in timestamps:
        for zone in zones:
            export = 0.0

            for line_id in line_ids:
                definition = ntc_line_definition(ntc_data, line_id)
                flow = get_ntc_flow(result["flows"], timestamp, line_id)

                if definition["from_zone"] == zone:
                    export += flow
                if definition["to_zone"] == zone:
                    export -= flow

            assert_zone_balance(result, zone, timestamp, export, tol)


def assert_ntc_capacity_bounds(
    result: dict,
    ntc_data: pd.DataFrame,
    timestamps: list[datetime],
    tol: float = 1e-4,
) -> None:
    """Assert ``-capacity_down <= flow <= capacity_up`` for every line/MTU."""
    line_ids = list(pd.Index(ntc_data["line_id"]).unique())

    for timestamp in timestamps:
        for line_id in line_ids:
            capacity_up, capacity_down = ntc_capacity(
                ntc_data, line_id, timestamp
            )
            flow = get_ntc_flow(result["flows"], timestamp, line_id)

            assert flow <= capacity_up + tol, (
                f"Line '{line_id}' at '{timestamp}' exceeds capacity_up."
            )
            assert flow >= -capacity_down - tol, (
                f"Line '{line_id}' at '{timestamp}' exceeds capacity_down."
            )


def assert_storage_schedule_feasible(
    result: dict,
    bid_id: str,
    timestamps: list[datetime],
    tol: float = 1e-4,
) -> dict:
    """
    Independently verify storage feasibility.

    Checks:

    - SOC recursion ``soc_t = soc_{t-1} + 0.25*eff*charge_t - 0.25*discharge_t``
      with ``soc_{-1} = starting_energy``;
    - minimum and maximum energy bounds;
    - input and output power bounds;
    - non-negative charge/discharge;
    - ``accepted_volume = discharge - charge``.

    The checked storage order is returned for further assertions.
    """
    order = find_order(result, bid_id)
    assert order["bid_type"] == "SO"

    previous_soc = float(order["starting_energy"])
    efficiency = float(order["efficiency"])

    for timestamp in timestamps:
        charge = float(order["charge"][timestamp])
        discharge = float(order["discharge"][timestamp])
        soc = float(order["soc"][timestamp])

        assert charge >= -tol, "Charging power must be non-negative."
        assert discharge >= -tol, "Discharging power must be non-negative."

        expected_soc = (
            previous_soc
            + MTU_HOURS * efficiency * charge
            - MTU_HOURS * discharge
        )
        assert soc == pytest.approx(expected_soc, abs=tol), (
            f"SOC recursion violated for '{bid_id}' at '{timestamp}'."
        )

        assert soc >= order["min_energy"][timestamp] - tol
        assert soc <= order["max_energy"][timestamp] + tol

        assert charge <= order["max_input_power"][timestamp] + tol
        assert discharge <= order["max_output_power"][timestamp] + tol

        assert order["accepted_volume"][timestamp] == pytest.approx(
            discharge - charge, abs=tol
        )

        previous_soc = soc

    return order


def fbmc_cnec_utilisation(
    clearing_input: dict,
    flows: dict,
    timestamps: list[datetime],
) -> list[dict]:
    """
    Recompute the PTDF-weighted FB flow of every CNEC from reported results.

    Returns one record per ``(timestamp, cnec)`` with ``fb_flow``, ``ram``
    and ``slack = ram - fb_flow``.
    """
    records = []

    for timestamp in timestamps:
        for cnec in clearing_input["cnec_list"]:
            fb_flow = sum(
                clearing_input["ptdf_zonal"][(timestamp, cnec, zone)]
                * get_fb_net_export(flows, timestamp, zone)
                for zone in clearing_input["fbmc_zone_ids"]
            )
            ram = clearing_input["ram"][(timestamp, cnec)]

            records.append(
                {
                    "timestamp": timestamp,
                    "cnec": cnec,
                    "fb_flow": float(fb_flow),
                    "ram": float(ram),
                    "slack": float(ram - fb_flow),
                }
            )

    return records


def assert_fbmc_ram_feasible(
    clearing_input: dict,
    flows: dict,
    timestamps: list[datetime],
    tol: float = 1e-3,
) -> list[dict]:
    """
    Independently assert FBMC RAM feasibility from PTDF, RAM and FB exports.

    Also checks that FB net exports sum to zero inside the FB region.
    """
    for timestamp in timestamps:
        total_export = sum(
            get_fb_net_export(flows, timestamp, zone)
            for zone in clearing_input["fbmc_zone_ids"]
        )
        assert total_export == pytest.approx(0.0, abs=tol), (
            f"FB net exports do not balance at '{timestamp}'."
        )

    records = fbmc_cnec_utilisation(clearing_input, flows, timestamps)

    for record in records:
        assert record["fb_flow"] <= record["ram"] + tol, (
            f"CNEC '{record['cnec']}' at '{record['timestamp']}' violates "
            f"RAM: flow {record['fb_flow']} MW > RAM {record['ram']} MW."
        )

    return records


def assert_simple_order_price_consistency(
    result: dict,
    tol: float = PRICE_TOL,
) -> None:
    """
    Assert official simple-order acceptance rules at the resulting prices.

    - fully accepted supply requires ``price >= bid price``;
    - rejected supply requires ``price <= bid price``;
    - partially accepted orders fix the price to their bid price;
    - demand conditions are mirrored.
    """
    for order in result["orders"]:
        if order["bid_type"] != "SB":
            continue

        volume = order["volume"]

        if abs(volume) <= EPS:
            continue

        price = get_price(
            result["meta"], order["node"], order["start_time"]
        )
        ratio = float(order["accepted_volume"]) / volume

        if volume > 0:
            if ratio <= EPS:
                assert price <= order["price"] + tol
            elif ratio >= 1.0 - EPS:
                assert price >= order["price"] - tol
            else:
                assert math.isclose(price, order["price"], abs_tol=tol)
        else:
            if ratio <= EPS:
                assert price >= order["price"] - tol
            elif ratio >= 1.0 - EPS:
                assert price <= order["price"] + tol
            else:
                assert math.isclose(price, order["price"], abs_tol=tol)


def block_surplus_eur(result: dict, bid_id: str) -> float:
    """
    Compute the surplus of a block bid in EUR at the resulting prices.

        surplus = sum((price - bid price) * volume * 0.25 * ratio)
    """
    order = find_order(result, bid_id)
    ratio = acceptance_ratio(result, bid_id)

    return sum(
        (
            get_price(result["meta"], order["node"], timestamp)
            - order["price"]
        )
        * volume
        * MTU_HOURS
        * ratio
        for timestamp, volume in order["volume"].items()
    )


def assert_block_surplus_conditions(
    result: dict,
    tol: float = 1e-3,
) -> None:
    """
    Assert non-negative surplus for every accepted block or linked family.

    A family consists of an accepted parent block and all of its accepted
    children, which is the condition implemented by the price-feasibility
    model.
    """
    block_ids = [
        order["bid_id"]
        for order in result["orders"]
        if order["bid_type"] in ("BB", "LB")
    ]
    accepted_ids = [
        bid_id
        for bid_id in block_ids
        if acceptance_ratio(result, bid_id) > EPS
    ]
    surplus = {
        bid_id: block_surplus_eur(result, bid_id) for bid_id in accepted_ids
    }

    for bid_id in accepted_ids:
        children = [
            order["bid_id"]
            for order in result["orders"]
            if order.get("parent_bid_id") == bid_id
            and order["bid_id"] in surplus
        ]
        family_surplus = surplus[bid_id] + sum(
            surplus[child] for child in children
        )

        assert family_surplus >= -tol, (
            f"Accepted block family of '{bid_id}' has negative surplus "
            f"{family_surplus} EUR."
        )