# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
EUPHEMIA-like market clearing with NTC, FBMC, and hybrid coupling.

Simple bids are continuous period-order decisions:

- in-the-money simple bids are fully accepted;
- out-of-the-money simple bids are rejected;
- at-the-money simple bids may be accepted, partially accepted, or rejected.

Block bids retain binary activation and minimum-acceptance-ratio logic.
"""

import logging
from collections import defaultdict
from datetime import timedelta
from operator import itemgetter
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pypsa
from mango import AgentAddress
from pyomo.opt import OptSolver, SolverFactory

from assume.common.fbmc_utils_version4 import (
    precompute_fbmc_parameters,
    to_clearing_input,
)
from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import get_supported_solver_pyomo
from assume.markets.base_market import MarketRole

logging.getLogger("pyomo").setLevel(logging.WARNING)
logging.getLogger("gurobipy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

EPS = 1e-5 # TODO Decide on a good value


def get_exclusive_groups(orders: Orderbook) -> dict[object, list[str]]:
    """Return block-bid IDs grouped by exclusive_group_id."""
    groups: dict[object, list[str]] = defaultdict(list)

    for order in orders:
        if order["bid_type"] in ["BB", "LB"]:
            group_id = order.get("exclusive_group_id")

            if group_id is not None:
                groups[group_id].append(order["bid_id"])

    return {
        group_id: bid_ids
        for group_id, bid_ids in groups.items()
        if len(bid_ids) > 1
    }


def load_ntc_connections(
    ntc_connections: str | Path | pd.DataFrame | None,
    snapshots: pd.Index,
) -> dict:
    """Load and validate time-dependent NTC connection data."""
    empty_result = {
        "line_ids": [],
        "zone_ids": [],
        "from_zone": {},
        "to_zone": {},
        "capacity_up": {},
        "capacity_down": {},
    }

    if ntc_connections is None:
        return empty_result

    if isinstance(ntc_connections, (str, Path)):
        ntc_data = pd.read_csv(ntc_connections)
    elif isinstance(ntc_connections, pd.DataFrame):
        ntc_data = ntc_connections.copy()
    else:
        raise TypeError(
            "ntc_connections must be None, a pandas DataFrame, or a CSV path."
        )

    required_columns = {
        "line_id",
        "from_zone",
        "to_zone",
        "timestamp",
        "capacity_up",
        "capacity_down",
    }
    missing_columns = required_columns.difference(ntc_data.columns)

    if missing_columns:
        raise ValueError(
            "NTC data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if pd.api.types.is_datetime64_any_dtype(snapshots):
        ntc_data["timestamp"] = pd.to_datetime(ntc_data["timestamp"])

    ntc_data = ntc_data.set_index(["timestamp", "line_id"], drop=False)

    if ntc_data.index.has_duplicates:
        duplicates = ntc_data.index[ntc_data.index.duplicated()].tolist()
        raise ValueError(
            "NTC data contains duplicate (timestamp, line_id) rows. "
            f"Examples: {duplicates[:5]}"
        )

    line_ids = pd.Index(ntc_data["line_id"].unique())
    zone_ids = pd.Index(
        pd.concat([ntc_data["from_zone"], ntc_data["to_zone"]]).unique()
    )

    from_zone = {}
    to_zone = {}
    capacity_up = {}
    capacity_down = {}

    for line_id in line_ids:
        line_data = ntc_data[ntc_data["line_id"] == line_id]
        from_zones = line_data["from_zone"].unique()
        to_zones = line_data["to_zone"].unique()

        if len(from_zones) != 1 or len(to_zones) != 1:
            raise ValueError(
                f"NTC line '{line_id}' must have one fixed from_zone and "
                "one fixed to_zone for all timestamps."
            )

        from_zone[line_id] = from_zones[0]
        to_zone[line_id] = to_zones[0]

        for timestamp in snapshots:
            key = (timestamp, line_id)

            if key not in ntc_data.index:
                raise ValueError(
                    f"NTC data has no capacity row for line '{line_id}' "
                    f"at timestamp '{timestamp}'."
                )

            row = ntc_data.loc[key]

            if pd.isna(row["capacity_up"]) or pd.isna(row["capacity_down"]):
                raise ValueError(
                    f"NTC capacities must not be null for line '{line_id}' "
                    f"at timestamp '{timestamp}'."
                )

            if row["capacity_up"] < 0 or row["capacity_down"] < 0:
                raise ValueError(
                    f"NTC capacities must be non-negative for line '{line_id}' "
                    f"at timestamp '{timestamp}'."
                )

            capacity_up[key] = float(row["capacity_up"])
            capacity_down[key] = float(row["capacity_down"])

    return {
        "line_ids": list(line_ids),
        "zone_ids": list(zone_ids),
        "from_zone": from_zone,
        "to_zone": to_zone,
        "capacity_up": capacity_up,
        "capacity_down": capacity_down,
    }


def _get_order_net_position_expression(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    zone: str,
    timestamp,
):
    """Build accepted supply-minus-demand volume for one zone/time."""
    expression = 0.0

    for order in orders:
        if (
            order["bid_type"] == "SB"
            and order["node"] == zone
            and order["start_time"] == timestamp
        ):
            expression += order["volume"] * model.xs[order["bid_id"]]

        elif (
            order["bid_type"] in ["BB", "LB"]
            and order["node"] == zone
            and timestamp in order["volume"]
        ):
            expression += (
                order["volume"][timestamp] * model.xb[order["bid_id"]]
            )

    return expression


def market_clearing_opt_constraints(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    market_products: list[MarketProduct],
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    market_zone_ids: list[str],
    fbmc_params: dict | None = None,
    ntc_params: dict | None = None,
):
    """Add market, NTC, FBMC, hybrid-balance, and block constraints."""
    snapshots = [product[0] for product in market_products]
    fbmc_params = fbmc_params or {}
    ntc_params = ntc_params or {}

    fbmc_zone_ids = list(fbmc_params.get("fbmc_zone_ids", []))
    cnec_list = list(fbmc_params.get("cnec_list", []))
    ntc_line_ids = list(ntc_params.get("line_ids", []))

    model.T = pyo.Set(initialize=snapshots, ordered=True)
    model.zones = pyo.Set(initialize=market_zone_ids, ordered=True)
    model.fbmc_zones = pyo.Set(initialize=fbmc_zone_ids, ordered=True)
    model.cnecs = pyo.Set(initialize=cnec_list, ordered=True)
    model.ntc_lines = pyo.Set(initialize=ntc_line_ids, ordered=True)

    model.sBids = pyo.Set(
        initialize=[
            order["bid_id"]
            for order in orders
            if order["bid_type"] == "SB"
        ]
    )
    model.bBids = pyo.Set(
        initialize=[
            order["bid_id"]
            for order in orders
            if order["bid_type"] in ["BB", "LB"]
        ]
    )

    model.xs = pyo.Var(
        model.sBids,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 1.0),
    )
    model.x = pyo.Var(model.bBids, domain=pyo.Binary)
    model.xb = pyo.Var(
        model.bBids,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 1.0),
    )
    model.net_position = pyo.Var(
        model.T,
        model.zones,
        domain=pyo.Reals,
    )

    if fbmc_zone_ids:
        model.net_export_fb = pyo.Var(
            model.T,
            model.fbmc_zones,
            domain=pyo.Reals,
        )

    if ntc_line_ids:
        model.ntc_flow = pyo.Var(
            model.T,
            model.ntc_lines,
            domain=pyo.Reals,
        )

    model.block_mar_constraints = pyo.ConstraintList()

    for order in orders:
        if order["bid_type"] not in ["BB", "LB"]:
            continue

        mar = order.get("min_acceptance_ratio", 1.0)

        if mar is None:
            mar = 1.0

        bid_id = order["bid_id"]
        model.block_mar_constraints.add(
            model.xb[bid_id] >= mar * model.x[bid_id]
        )
        model.block_mar_constraints.add(
            model.xb[bid_id] <= model.x[bid_id]
        )

    if with_linked_bids:
        model.linked_bid_constraints = pyo.ConstraintList()

        for order in orders:
            parent_bid_id = order.get("parent_bid_id")

            if (
                order["bid_type"] in ["BB", "LB"]
                and parent_bid_id in model.bBids
            ):
                model.linked_bid_constraints.add(
                    model.xb[order["bid_id"]]
                    <= model.xb[parent_bid_id]
                )

    if with_exclusive_groups:
        model.exclusive_group_constraints = pyo.ConstraintList()

        for bid_ids in get_exclusive_groups(orders).values():
            model.exclusive_group_constraints.add(
                sum(model.xb[bid_id] for bid_id in bid_ids) <= 1
            )

    def net_position_rule(model, zone, timestamp):
        return model.net_position[timestamp, zone] == (
            _get_order_net_position_expression(
                model=model,
                orders=orders,
                zone=zone,
                timestamp=timestamp,
            )
        )

    model.net_position_definition = pyo.Constraint(
        model.zones,
        model.T,
        rule=net_position_rule,
    )

    if ntc_line_ids:
        model.ntc_capacity_constraints = pyo.ConstraintList()

        for timestamp in model.T:
            for line_id in model.ntc_lines:
                model.ntc_capacity_constraints.add(
                    model.ntc_flow[timestamp, line_id]
                    <= ntc_params["capacity_up"][(timestamp, line_id)]
                )
                model.ntc_capacity_constraints.add(
                    model.ntc_flow[timestamp, line_id]
                    >= -ntc_params["capacity_down"][(timestamp, line_id)]
                )

    if fbmc_zone_ids:

        def fb_balance_rule(model, timestamp):
            return sum(
                model.net_export_fb[timestamp, zone]
                for zone in model.fbmc_zones
            ) == 0.0

        model.fb_balance = pyo.Constraint(model.T, rule=fb_balance_rule)
        model.fbmc_constraints = pyo.ConstraintList()

        for timestamp in model.T:
            for cnec in model.cnecs:
                model.fbmc_constraints.add(
                    sum(
                        fbmc_params["ptdf_zonal"][(timestamp, cnec, zone)]
                        * model.net_export_fb[timestamp, zone]
                        for zone in model.fbmc_zones
                    )
                    <= fbmc_params["ram"][(timestamp, cnec)]
                )

    def zonal_balance_rule(model, zone, timestamp):
        ntc_export = sum(
            model.ntc_flow[timestamp, line_id]
            for line_id in model.ntc_lines
            if ntc_params["from_zone"][line_id] == zone
        ) - sum(
            model.ntc_flow[timestamp, line_id]
            for line_id in model.ntc_lines
            if ntc_params["to_zone"][line_id] == zone
        )

        fb_export = (
            model.net_export_fb[timestamp, zone]
            if zone in model.fbmc_zones
            else 0.0
        )

        return model.net_position[timestamp, zone] == fb_export + ntc_export

    model.zonal_balance = pyo.Constraint(
        model.zones,
        model.T,
        rule=zonal_balance_rule,
    )


def _build_welfare_expression(
    model: pyo.ConcreteModel,
    orders: Orderbook,
):
    welfare = 0.0

    for order in orders:
        if order["bid_type"] == "SB":
            welfare += (
                -order["price"]
                * order["volume"]
                * model.xs[order["bid_id"]]
            )

        elif order["bid_type"] in ["BB", "LB"]:
            welfare += sum(
                -order["price"] * volume * model.xb[order["bid_id"]]
                for volume in order["volume"].values()
            )

    return welfare


def market_clearing_opt_objective(
    model: pyo.ConcreteModel,
    orders: Orderbook,
):
    """Maximise welfare with positive supply and negative demand volumes."""
    model.welfare = pyo.Expression(
        expr=_build_welfare_expression(model, orders)
    )
    model.objective = pyo.Objective(
        expr=model.welfare,
        sense=pyo.maximize,
    )


def build_market_clearing_model(
    orders: Orderbook,
    market_products: list[MarketProduct],
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    market_zone_ids: list[str],
    fbmc_params: dict | None = None,
    ntc_params: dict | None = None,
    func_constraints=market_clearing_opt_constraints,
    func_objective=market_clearing_opt_objective,
):
    """Build the welfare-maximising master model."""
    model = pyo.ConcreteModel()

    func_constraints(
        model=model,
        orders=orders,
        market_products=market_products,
        with_linked_bids=with_linked_bids,
        with_exclusive_groups=with_exclusive_groups,
        market_zone_ids=market_zone_ids,
        fbmc_params=fbmc_params,
        ntc_params=ntc_params,
    )
    func_objective(model=model, orders=orders)
    model.price_feasibility_cuts = pyo.ConstraintList()

    return model


def market_clearing_opt(
    orders: Orderbook,
    market_products: list[MarketProduct],
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    market_zone_ids: list[str],
    fbmc_params: dict | None = None,
    ntc_params: dict | None = None,
    solver: OptSolver | None = None,
    solver_options: dict | None = None,
    func_constraints=market_clearing_opt_constraints,
    func_objective=market_clearing_opt_objective,
):
    """Build and solve one welfare-master optimisation model."""
    if solver is None:
        raise ValueError("A Pyomo solver instance must be provided.")

    model = build_market_clearing_model(
        orders=orders,
        market_products=market_products,
        with_linked_bids=with_linked_bids,
        with_exclusive_groups=with_exclusive_groups,
        market_zone_ids=market_zone_ids,
        fbmc_params=fbmc_params,
        ntc_params=ntc_params,
        func_constraints=func_constraints,
        func_objective=func_objective,
    )
    results = solver.solve(model, options=solver_options or {})

    return model, results


def _result_is_infeasible(results) -> bool:
    termination = str(results.solver.termination_condition).lower()
    return "infeasible" in termination or "unbounded" in termination


def _result_has_solution(results) -> bool:
    termination = str(results.solver.termination_condition).lower()

    return not any(
        failure in termination
        for failure in [
            "infeasible",
            "unbounded",
            "invalid",
            "error",
        ]
    )


def _solve_checked(
    solver: OptSolver,
    model: pyo.ConcreteModel,
    solver_options: dict,
    context: str,
):
    try:
        results = solver.solve(model, options=solver_options)
    except Exception as exc:
        raise Exception(f"{context} failed: {exc}") from exc

    if not _result_has_solution(results):
        raise Exception(
            f"{context} did not produce a feasible solution "
            f"({results.solver.termination_condition})."
        )

    return results


def _collect_master_solution(model: pyo.ConcreteModel) -> dict:
    """Extract the primal solution required by pricing and tie-breaking."""
    selected_x = {
        bid_id: int(pyo.value(model.x[bid_id]) >= 0.5) # Converts numeric value into binary block-activation
        for bid_id in model.bBids
    }
    simple_acceptance = {
        bid_id: min(1.0, max(0.0, float(pyo.value(model.xs[bid_id]))))
        for bid_id in model.sBids
    }
    block_acceptance = {
        bid_id: min(1.0, max(0.0, float(pyo.value(model.xb[bid_id]))))
        for bid_id in model.bBids
    }

    ntc_flow = {}
    fb_net_export = {}

    if hasattr(model, "ntc_flow"):
        ntc_flow = {
            (timestamp, line_id): float(
                pyo.value(model.ntc_flow[timestamp, line_id])
            )
            for timestamp in model.T
            for line_id in model.ntc_lines
        }

    if hasattr(model, "net_export_fb"):
        fb_net_export = {
            (timestamp, zone): float(
                pyo.value(model.net_export_fb[timestamp, zone])
            )
            for timestamp in model.T
            for zone in model.fbmc_zones
        }

    return {
        "selected_x": selected_x,
        "simple_acceptance": simple_acceptance,
        "block_acceptance": block_acceptance,
        "ntc_flow": ntc_flow,
        "fb_net_export": fb_net_export,
    }


def build_price_feasibility_model(
    orders: Orderbook,
    market_products: list[MarketProduct],
    market_zone_ids: list[str],
    fbmc_params: dict | None,
    ntc_params: dict | None,
    master_solution: dict,
    price_floor: float,
    price_cap: float,
):
    """
    Build the fixed-dispatch price-feasibility model.

    The model contains:

    - simple-order price consistency;
    - accepted block surplus constraints;
    - linked-family block-surplus constraints;
    - NTC flow/price complementarity conditions;
    - FBMC PTDF congestion-price conditions.

    FBMC price conditions are:

        price[t, z]
        =
        fb_reference_price[t]
        - sum(
            PTDF[t, cnec, z] * fb_cnec_shadow[t, cnec]
            for cnec
        )

    A CNEC shadow price is forced to zero when the corresponding fixed
    master FB flow is strictly below its RAM. This enforces price
    convergence in an uncongested FB domain.
    """
    fbmc_params = fbmc_params or {}
    ntc_params = ntc_params or {}
    snapshots = [product[0] for product in market_products]

    fbmc_zone_ids = list(fbmc_params.get("fbmc_zone_ids", []))
    cnec_list = list(fbmc_params.get("cnec_list", []))

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=snapshots, ordered=True)
    model.zones = pyo.Set(initialize=market_zone_ids, ordered=True)
    model.fbmc_zones = pyo.Set(initialize=fbmc_zone_ids, ordered=True)
    model.cnecs = pyo.Set(initialize=cnec_list, ordered=True)
    model.ntc_lines = pyo.Set(
        initialize=list(ntc_params.get("line_ids", [])),
        ordered=True,
    )

    model.price = pyo.Var(
        model.T,
        model.zones,
        domain=pyo.Reals,
        bounds=(price_floor, price_cap),
    )
    model.price_constraints = pyo.ConstraintList()

    # ------------------------------------------------------------------
    # Simple bids
    # ------------------------------------------------------------------

    for order in orders:
        if order["bid_type"] != "SB":
            continue

        acceptance = master_solution["simple_acceptance"][order["bid_id"]]
        price = model.price[order["start_time"], order["node"]]
        bid_price = order["price"]
        volume = order["volume"]

        if volume == 0:
            continue

        rejected = acceptance <= EPS
        full = acceptance >= 1.0 - EPS

        if volume > 0:
            if rejected:
                model.price_constraints.add(price <= bid_price)
            elif full:
                model.price_constraints.add(price >= bid_price)
            else:
                model.price_constraints.add(price == bid_price)
        else:
            if rejected:
                model.price_constraints.add(price >= bid_price)
            elif full:
                model.price_constraints.add(price <= bid_price)
            else:
                model.price_constraints.add(price == bid_price)

    # ------------------------------------------------------------------
    # Block and linked-block surplus
    # ------------------------------------------------------------------

    block_orders = {
        order["bid_id"]: order
        for order in orders
        if order["bid_type"] in ["BB", "LB"]
    }
    children_by_parent: dict[str, list[str]] = defaultdict(list)

    for bid_id, order in block_orders.items():
        parent_bid_id = order.get("parent_bid_id")

        if parent_bid_id in block_orders:
            children_by_parent[parent_bid_id].append(bid_id)

    block_surplus = {}

    for bid_id, order in block_orders.items():
        acceptance = master_solution["block_acceptance"][bid_id]

        if acceptance <= EPS:
            continue

        block_surplus[bid_id] = sum(
            (
                model.price[timestamp, order["node"]]
                - order["price"]
            )
            * volume
            * acceptance
            for timestamp, volume in order["volume"].items()
        )

    for bid_id, own_surplus in block_surplus.items():
        accepted_children = [
            child_bid_id
            for child_bid_id in children_by_parent.get(bid_id, [])
            if child_bid_id in block_surplus
        ]

        model.price_constraints.add(
            own_surplus
            + sum(
                block_surplus[child_bid_id]
                for child_bid_id in accepted_children
            )
            >= -EPS
        )

    # ------------------------------------------------------------------
    # NTC price-flow consistency
    # ------------------------------------------------------------------

    for timestamp in model.T:
        for line_id in model.ntc_lines:
            flow = master_solution["ntc_flow"][(timestamp, line_id)]
            upper = ntc_params["capacity_up"][(timestamp, line_id)]
            lower = -ntc_params["capacity_down"][(timestamp, line_id)]

            from_zone = ntc_params["from_zone"][line_id]
            to_zone = ntc_params["to_zone"][line_id]

            from_price = model.price[timestamp, from_zone]
            to_price = model.price[timestamp, to_zone]

            if lower + EPS < flow < upper - EPS:
                model.price_constraints.add(from_price == to_price)
            else:
                if flow >= upper - EPS:
                    model.price_constraints.add(from_price <= to_price)

                if flow <= lower + EPS:
                    model.price_constraints.add(from_price >= to_price)

    # ------------------------------------------------------------------
    # FBMC price-network consistency
    # ------------------------------------------------------------------
    #
    # The primal FB constraint is:
    #
    #   sum(PTDF[cnec, zone] * net_export_fb[zone]) <= RAM[cnec]
    #
    # For a fixed primal dispatch, valid zonal prices satisfy:
    #
    #   price[zone]
    #   =
    #   common_fb_energy_price
    #   - sum(PTDF[cnec, zone] * congestion_price[cnec])
    #
    # where congestion_price[cnec] >= 0 and must be zero if the CNEC
    # is not binding. This is the relevant stationarity/complementarity
    # condition for the simplified FBMC primal model in this file.
    #
    # With all CNECs slack, every FB zone has the same price.
    # ------------------------------------------------------------------

    if fbmc_zone_ids:
        model.fb_reference_price = pyo.Var(
            model.T,
            domain=pyo.Reals,
            bounds=(price_floor, price_cap),
        )
        model.fb_cnec_shadow = pyo.Var(
            model.T,
            model.cnecs,
            domain=pyo.NonNegativeReals,
        )

        for timestamp in model.T:
            for cnec in model.cnecs:
                fb_flow = sum(
                    fbmc_params["ptdf_zonal"][(timestamp, cnec, zone)]
                    * master_solution["fb_net_export"][(timestamp, zone)]
                    for zone in model.fbmc_zones
                )
                ram = fbmc_params["ram"][(timestamp, cnec)]

                if fb_flow < ram - EPS:
                    model.price_constraints.add(
                        model.fb_cnec_shadow[timestamp, cnec] == 0.0
                    )

            for zone in model.fbmc_zones:
                model.price_constraints.add(
                    model.price[timestamp, zone]
                    == model.fb_reference_price[timestamp]
                    - sum(
                        fbmc_params["ptdf_zonal"][(timestamp, cnec, zone)]
                        * model.fb_cnec_shadow[timestamp, cnec]
                        for cnec in model.cnecs
                    )
                )

    model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return model


def solve_master_with_price_cuts(
    master_model: pyo.ConcreteModel,
    orders: Orderbook,
    market_products: list[MarketProduct],
    market_zone_ids: list[str],
    fbmc_params: dict | None,
    ntc_params: dict | None,
    price_floor: float,
    price_cap: float,
    solver: OptSolver,
    solver_options: dict,
):
    """Find a welfare-optimal and price-feasible block selection."""
    cuts_added = 0

    while True:
        try:
            results = solver.solve(master_model, options=solver_options)
        except Exception as exc:
            raise Exception(f"Market-clearing problem failed: {exc}") from exc

        if _result_is_infeasible(results):
            if cuts_added:
                raise Exception(
                    "No price-feasible discrete market-clearing solution exists."
                )
            raise Exception("Market-clearing problem is infeasible.")

        if not _result_has_solution(results):
            raise Exception(
                "Market-clearing problem did not produce a feasible solution "
                f"({results.solver.termination_condition})."
            )

        master_solution = _collect_master_solution(master_model)

        price_model = build_price_feasibility_model(
            orders=orders,
            market_products=market_products,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
            master_solution=master_solution,
            price_floor=price_floor,
            price_cap=price_cap,
        )

        try:
            price_results = solver.solve(
                price_model,
                options=solver_options,
            )
        except Exception as exc:
            raise Exception(
                f"Price-feasibility subproblem failed: {exc}"
            ) from exc

        if _result_has_solution(price_results):
            return master_model, master_solution

        if not _result_is_infeasible(price_results):
            raise Exception(
                "Price-feasibility subproblem did not produce a valid result "
                f"({price_results.solver.termination_condition})."
            )

        if not list(master_model.bBids):
            raise Exception(
                "No price-feasible market-clearing solution exists."
            )

        accepted_ids = [
            bid_id
            for bid_id in master_model.bBids
            if master_solution["selected_x"][bid_id] == 1
        ]
        rejected_ids = [
            bid_id
            for bid_id in master_model.bBids
            if master_solution["selected_x"][bid_id] == 0
        ]

        master_model.price_feasibility_cuts.add(
            sum(1 - master_model.x[bid_id] for bid_id in accepted_ids)
            + sum(master_model.x[bid_id] for bid_id in rejected_ids)
            >= 1
        )
        cuts_added += 1 # TODO Check cut logic


def find_price_bounds(
    orders: Orderbook,
    market_products: list[MarketProduct],
    market_zone_ids: list[str],
    fbmc_params: dict | None,
    ntc_params: dict | None,
    master_solution: dict,
    price_floor: float,
    price_cap: float,
    solver: OptSolver,
    solver_options: dict,
):
    """Find feasible lower and upper prices for each zone and timestamp."""
    lower_price = {}
    upper_price = {}

    for timestamp in [product[0] for product in market_products]:
        for zone in market_zone_ids:
            lower_model = build_price_feasibility_model(
                orders=orders,
                market_products=market_products,
                market_zone_ids=market_zone_ids,
                fbmc_params=fbmc_params,
                ntc_params=ntc_params,
                master_solution=master_solution,
                price_floor=price_floor,
                price_cap=price_cap,
            )
            lower_model.objective.deactivate()
            lower_model.price_bound_objective = pyo.Objective(
                expr=lower_model.price[timestamp, zone],
                sense=pyo.minimize,
            )
            _solve_checked(
                solver=solver,
                model=lower_model,
                solver_options=solver_options,
                context="Lower-price subproblem",
            )
            lower_price[(timestamp, zone)] = float(
                pyo.value(lower_model.price[timestamp, zone])
            )

            upper_model = build_price_feasibility_model(
                orders=orders,
                market_products=market_products,
                market_zone_ids=market_zone_ids,
                fbmc_params=fbmc_params,
                ntc_params=ntc_params,
                master_solution=master_solution,
                price_floor=price_floor,
                price_cap=price_cap,
            )
            upper_model.objective.deactivate()
            upper_model.price_bound_objective = pyo.Objective(
                expr=upper_model.price[timestamp, zone],
                sense=pyo.maximize,
            )
            _solve_checked(
                solver=solver,
                model=upper_model,
                solver_options=solver_options,
                context="Upper-price subproblem",
            )
            upper_price[(timestamp, zone)] = float(
                pyo.value(upper_model.price[timestamp, zone])
            )

            # TODO Check if needed
            logger.debug(
                "Price bounds at %s in %s: lower=%s, upper=%s, midpoint=%s",
                timestamp,
                zone,
                lower_price[(timestamp, zone)],
                upper_price[(timestamp, zone)],
                0.5 * (
                    lower_price[(timestamp, zone)]
                    + upper_price[(timestamp, zone)]
                ),
            )    

    return lower_price, upper_price


def select_midpoint_prices(
    orders: Orderbook,
    market_products: list[MarketProduct],
    market_zone_ids: list[str],
    fbmc_params: dict | None,
    ntc_params: dict | None,
    master_solution: dict,
    price_floor: float,
    price_cap: float,
    solver: OptSolver,
    solver_options: dict,
):
    """Select deterministic feasible prices nearest to interval midpoints."""
    lower_price, upper_price = find_price_bounds(
        orders=orders,
        market_products=market_products,
        market_zone_ids=market_zone_ids,
        fbmc_params=fbmc_params,
        ntc_params=ntc_params,
        master_solution=master_solution,
        price_floor=price_floor,
        price_cap=price_cap,
        solver=solver,
        solver_options=solver_options,
    )
    midpoint = {
        key: 0.5 * (lower_price[key] + upper_price[key])
        for key in lower_price
    }

    price_model = build_price_feasibility_model(
        orders=orders,
        market_products=market_products,
        market_zone_ids=market_zone_ids,
        fbmc_params=fbmc_params,
        ntc_params=ntc_params,
        master_solution=master_solution,
        price_floor=price_floor,
        price_cap=price_cap,
    )
    price_model.objective.deactivate()
    price_model.midpoint_objective = pyo.Objective(
        expr=sum(
            (
                price_model.price[timestamp, zone]
                - midpoint[(timestamp, zone)]
            )
            ** 2
            for timestamp in price_model.T
            for zone in price_model.zones
        ),
        sense=pyo.minimize,
    )

    quadratic_solution_found = False

    try:
        results = solver.solve(price_model, options=solver_options)
        quadratic_solution_found = _result_has_solution(results)
    except Exception:
        quadratic_solution_found = False

    if not quadratic_solution_found:
        price_model = build_price_feasibility_model(
            orders=orders,
            market_products=market_products,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
            master_solution=master_solution,
            price_floor=price_floor,
            price_cap=price_cap,
        )
        price_model.objective.deactivate()
        price_model.abs_deviation = pyo.Var(
            price_model.T,
            price_model.zones,
            domain=pyo.NonNegativeReals,
        )
        price_model.abs_deviation_constraints = pyo.ConstraintList()

        for timestamp in price_model.T:
            for zone in price_model.zones:
                price_model.abs_deviation_constraints.add(
                    price_model.abs_deviation[timestamp, zone]
                    >= price_model.price[timestamp, zone]
                    - midpoint[(timestamp, zone)]
                )
                price_model.abs_deviation_constraints.add(
                    price_model.abs_deviation[timestamp, zone]
                    >= midpoint[(timestamp, zone)]
                    - price_model.price[timestamp, zone]
                )

        price_model.midpoint_objective = pyo.Objective(
            expr=sum(
                price_model.abs_deviation[timestamp, zone]
                for timestamp in price_model.T
                for zone in price_model.zones
            ),
            sense=pyo.minimize,
        )
        _solve_checked(
            solver=solver,
            model=price_model,
            solver_options=solver_options,
            context="Deterministic LP price-selection subproblem",
        )

    return {
        zone: {
            timestamp: float(pyo.value(price_model.price[timestamp, zone]))
            for timestamp in price_model.T
        }
        for zone in price_model.zones
    }


def _replace_objective(
    model: pyo.ConcreteModel,
    name: str,
    expression,
    sense,
):
    """Deactivate active objectives and add one replacement objective."""
    for objective in model.component_objects(pyo.Objective, active=True):
        objective.deactivate()

    setattr(
        model,
        name,
        pyo.Objective(expr=expression, sense=sense),
    )


def solve_volume_flow_tiebreak(
    orders: Orderbook,
    market_products: list[MarketProduct],
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    market_zone_ids: list[str],
    fbmc_params: dict | None,
    ntc_params: dict | None,
    master_solution: dict,
    master_welfare: float,
    market_clearing_prices: dict,
    price_floor: float,
    price_cap: float,
    solver: OptSolver,
    solver_options: dict,
):
    """
    Resolve remaining simple-bid volume indeterminacy deterministically.

    Block statuses, block ratios, NTC flows, and FB net exports are fixed
    to the values used by price feasibility. This is necessary because price
    conditions are valid only for the fixed network state used by the price
    subproblem.

    Therefore, the previous NTC minimum-norm flow stage is deliberately not
    run after prices have been selected: changing an NTC flow could change
    whether a capacity is binding and invalidate the selected prices.
    """
    model = build_market_clearing_model(
        orders=orders,
        market_products=market_products,
        with_linked_bids=with_linked_bids,
        with_exclusive_groups=with_exclusive_groups,
        market_zone_ids=market_zone_ids,
        fbmc_params=fbmc_params,
        ntc_params=ntc_params,
    )
    model.objective.deactivate()

    for bid_id in model.bBids:
        model.x[bid_id].fix(master_solution["selected_x"][bid_id])
        model.xb[bid_id].fix(master_solution["block_acceptance"][bid_id])

    # Keep the NTC flow state used by the NTC price-feasibility rules.
    if hasattr(model, "ntc_flow"):
        for timestamp in model.T:
            for line_id in model.ntc_lines:
                model.ntc_flow[timestamp, line_id].fix(
                    master_solution["ntc_flow"][(timestamp, line_id)]
                )

    # Keep the FB net-export state used to determine binding CNECs and
    # therefore FB congestion-price feasibility.
    if hasattr(model, "net_export_fb"):
        for timestamp in model.T:
            for zone in model.fbmc_zones:
                model.net_export_fb[timestamp, zone].fix(
                    master_solution["fb_net_export"][(timestamp, zone)]
                )

    model.simple_price_consistency = pyo.ConstraintList()
    price_taking_terms = []
    at_money_terms = []

    for order in orders:
        if order["bid_type"] != "SB":
            continue

        bid_id = order["bid_id"]
        bid_price = order["price"]
        volume = order["volume"]
        final_price = market_clearing_prices[order["node"]][
            order["start_time"]
        ]

        if volume > 0:
            if final_price > bid_price + EPS:
                model.simple_price_consistency.add(model.xs[bid_id] == 1.0)
            elif final_price < bid_price - EPS:
                model.simple_price_consistency.add(model.xs[bid_id] == 0.0)

            if abs(bid_price - price_floor) <= EPS:
                price_taking_terms.append(
                    abs(volume) * (1.0 - model.xs[bid_id])
                )

        elif volume < 0:
            if final_price < bid_price - EPS:
                model.simple_price_consistency.add(model.xs[bid_id] == 1.0)
            elif final_price > bid_price + EPS:
                model.simple_price_consistency.add(model.xs[bid_id] == 0.0)

            if abs(bid_price - price_cap) <= EPS:
                price_taking_terms.append(
                    abs(volume) * (1.0 - model.xs[bid_id])
                )

        if abs(bid_price - final_price) <= EPS:
            at_money_terms.append(abs(volume) * model.xs[bid_id])

    # Re-establish welfare-maximising simple-bid volumes under:
    #
    # - fixed block activations and ratios;
    # - fixed NTC flows;
    # - fixed FB net exports;
    # - fixed final prices;
    # - official simple-order acceptance rules.
    _replace_objective(
        model=model,
        name="price_consistent_welfare_objective",
        expression=model.welfare,
        sense=pyo.maximize,
    )
    _solve_checked(
        solver=solver,
        model=model,
        solver_options=solver_options,
        context="Price-consistent welfare tie-break",
    )

    price_consistent_welfare = float(pyo.value(model.welfare))
    welfare_tolerance = max(
        EPS,
        1e-6,
        abs(master_welfare) * 1e-7,
    )

    if price_consistent_welfare < master_welfare - welfare_tolerance:
        raise Exception(
            "Selected prices are not compatible with the welfare-optimal "
            "master dispatch."
        )

    model.primary_welfare_lower = pyo.Constraint(
        expr=model.welfare >= price_consistent_welfare - welfare_tolerance
    )
    model.primary_welfare_upper = pyo.Constraint(
        expr=model.welfare <= price_consistent_welfare + welfare_tolerance
    )

    # Stage 5a: minimise rejected price-taking simple-order volume.
    model.curtailment = pyo.Expression(expr=sum(price_taking_terms))
    _replace_objective(
        model=model,
        name="curtailment_objective",
        expression=model.curtailment,
        sense=pyo.minimize,
    )
    _solve_checked(
        solver=solver,
        model=model,
        solver_options=solver_options,
        context="Price-taking curtailment tie-break",
    )

    curtailment_optimum = float(pyo.value(model.curtailment))
    model.curtailment_fix = pyo.Constraint(
        expr=model.curtailment <= curtailment_optimum + EPS
    )

    # Stage 5b: maximise accepted at-the-money simple-bid volume.
    model.at_money_volume = pyo.Expression(expr=sum(at_money_terms))
    _replace_objective(
        model=model,
        name="at_money_volume_objective",
        expression=model.at_money_volume,
        sense=pyo.maximize,
    )
    _solve_checked(
        solver=solver,
        model=model,
        solver_options=solver_options,
        context="At-the-money volume tie-break",
    )

    at_money_optimum = float(pyo.value(model.at_money_volume))
    model.at_money_volume_fix = pyo.Constraint(
        expr=model.at_money_volume >= at_money_optimum - EPS
    )

    # NTC flows and FB net exports intentionally remain fixed. See docstring.
    return model


class ComplexClearingRole(MarketRole):
    """EUPHEMIA-like market clearing with NTC, FBMC, or hybrid coupling."""

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.solver_name = get_supported_solver_pyomo(
            marketconfig.param_dict.get("solver_name", "appsi_highs")
        )
        self.solver = SolverFactory(self.solver_name)
        self.solver_options = {}

        if self.solver_name == "gurobi":
            self.solver_options = {
                "cutoff": -1.0,
                "MIPGap": EPS,
            }

        self.log_flows = self.marketconfig.param_dict.get(
            "log_flows",
            False,
        )
        self.pricing_mechanism = self.marketconfig.param_dict.get(
            "pricing_mechanism",
            "pay_as_clear",
        )
        self.zone_column = self.marketconfig.param_dict.get(
            "zones_identifier",
            "zone_id",
        )
        self.network: pypsa.Network | None = self.marketconfig.param_dict.get(
            "network"
        )
        self.flowbased_region = list(
            self.marketconfig.param_dict.get("flowbased_region", [])
        )
        self.ntc_connections = self.marketconfig.param_dict.get(
            "ntc_connections"
        )

        self.market_zone_ids: list[str] = []
        self.node_to_zone: dict = {}

        if self.network is not None:
            if self.zone_column not in self.network.buses.columns:
                raise ValueError(
                    f"The PyPSA network does not contain zone column "
                    f"'{self.zone_column}'."
                )

            self.market_zone_ids = list(
                pd.Index(self.network.buses[self.zone_column].unique())
            )
            self.node_to_zone = self.network.buses[self.zone_column].to_dict()

        if self.flowbased_region and self.network is None:
            raise ValueError(
                "A pypsa.Network is required when flowbased_region is non-empty."
            )

        if self.flowbased_region:
            unknown_zones = set(self.flowbased_region).difference(
                self.market_zone_ids
            )

            if unknown_zones:
                raise ValueError(
                    "flowbased_region contains zones not present in "
                    f"network.buses['{self.zone_column}']: "
                    f"{sorted(unknown_zones)}"
                )

        self.line_results = []
        self.all_orders = []

    def _validate_market_product_snapshots(
        self,
        market_products: list[MarketProduct],
    ) -> pd.Index:
        """Validate market-product timestamps."""
        snapshots = pd.Index(
            [market_product[0] for market_product in market_products]
        )

        if snapshots.has_duplicates:
            raise ValueError(
                "Market products contain duplicate start timestamps. "
                "This clearing implementation expects one product per "
                "timestamp."
            )

        if self.network is not None:
            network_snapshots = pd.Index(self.network.snapshots)
            missing_snapshots = snapshots.difference(network_snapshots)

            if not missing_snapshots.empty:
                raise ValueError(
                    "All market-product start times must be contained in "
                    "network.snapshots. Missing timestamps: "
                    f"{list(missing_snapshots)}"
                )

        return snapshots

    def _determine_market_zones(
        self,
        orderbook: Orderbook,
        ntc_params: dict,
    ) -> list[str]:
        """Determine all market zones used by the clearing."""
        zones = list(self.market_zone_ids)

        for zone in ntc_params["zone_ids"]:
            if zone not in zones:
                zones.append(zone)

        for order in orderbook:
            zone = order.get("node")

            if zone is not None and zone not in zones:
                zones.append(zone)

        if not zones:
            zones = ["node0"]

        return zones

    def validate_orderbook(
        self,
        orderbook: Orderbook,
        agent_addr: AgentAddress,
    ) -> None:
        """Validate order types, volumes, exclusive groups, and zones."""
        market_id = self.marketconfig.market_id
        max_volume = self.marketconfig.maximum_bid_volume

        with_exclusive_groups = (
            "exclusive_group_id" in self.marketconfig.additional_fields
        )

        for order in orderbook:
            if order.get("bid_type") is None:
                order["bid_type"] = "SB"
            elif order["bid_type"] not in ["SB", "BB", "LB"]:
                logger.warning(
                    "Market '%s': invalid bid_type '%s' in order '%s'. "
                    "Using SB.",
                    market_id,
                    order["bid_type"],
                    order.get("bid_id"),
                )
                order["bid_type"] = "SB"

        super().validate_orderbook(orderbook, agent_addr)

        for order in orderbook:
            if order["bid_type"] in ["BB", "LB"]:
                for timestamp, volume in order.get("volume", {}).items():
                    if abs(volume) > max_volume:
                        logger.warning(
                            "Market '%s': volume '%s' at '%s' exceeds "
                            "maximum_bid_volume '%s'. Clipping it.",
                            market_id,
                            volume,
                            timestamp,
                            max_volume,
                        )
                        order["volume"][timestamp] = (
                            max_volume if volume > 0 else -max_volume
                        )

            if with_exclusive_groups:
                if order.get("exclusive_group_id") is None:
                    order["exclusive_group_id"] = None
                elif order["bid_type"] == "SB":
                    logger.warning(
                        "Market '%s': simple order '%s' cannot belong to "
                        "an exclusive group. Removing exclusive_group_id.",
                        market_id,
                        order["bid_id"],
                    )
                    order["exclusive_group_id"] = None

            node = order.get("node")

            if node is None:
                order["node"] = (
                    self.market_zone_ids[0]
                    if self.market_zone_ids
                    else "node0"
                )
                logger.warning(
                    "Market '%s': order '%s' has no node. Assigned '%s'.",
                    market_id,
                    order["bid_id"],
                    order["node"],
                )
                continue

            if node in self.node_to_zone:
                order["node"] = self.node_to_zone[node]

    def clear(
        self,
        orderbook: Orderbook,
        market_products,
    ) -> tuple[Orderbook, Orderbook, list[dict], dict]:
        """Clear the market through master, price, and tie-break stages."""
        if len(orderbook) == 0:
            return [], [], [], {}

        orderbook.sort(
            key=itemgetter("start_time", "end_time", "only_hours")
        )

        snapshots = self._validate_market_product_snapshots(
            market_products
        )

        fbmc_params = None
        raw_fbmc_parameters = None

        if self.flowbased_region:
            logger.info(
                "Precomputing FBMC parameters for region %s.",
                self.flowbased_region,
            )
            raw_fbmc_parameters = precompute_fbmc_parameters(
                network=self.network,
                flowbased_region=self.flowbased_region,
                zone_column=self.zone_column,
                snapshots=snapshots,
                solver_name=self.marketconfig.param_dict.get(
                    "fbmc_solver_name"
                ),
            )
            fbmc_params = to_clearing_input(raw_fbmc_parameters)

        ntc_params = load_ntc_connections(
            ntc_connections=self.ntc_connections,
            snapshots=snapshots,
        )
        market_zone_ids = self._determine_market_zones(
            orderbook=orderbook,
            ntc_params=ntc_params,
        )

        if fbmc_params is not None:
            for zone in fbmc_params["fbmc_zone_ids"]:
                if zone not in market_zone_ids:
                    market_zone_ids.append(zone)

        self.market_zone_ids = market_zone_ids
        child_orders = []

        for order in orderbook:
            order["accepted_price"] = {}
            order["accepted_volume"] = {}

            parent_bid_id = order.get("parent_bid_id")

            if parent_bid_id is None:
                continue

            parent_bid = next(
                (
                    bid
                    for bid in orderbook
                    if bid["bid_id"] == parent_bid_id
                ),
                None,
            )

            if parent_bid is None:
                order["parent_bid_id"] = None
                logger.warning(
                    "Parent bid '%s' is not in the orderbook.",
                    parent_bid_id,
                )
            else:
                child_orders.append(order)

        with_linked_bids = bool(child_orders)
        with_exclusive_groups = bool(get_exclusive_groups(orderbook))

        price_floor = self.marketconfig.param_dict.get(
            "price_floor",
            -500.0,
        )
        price_cap = self.marketconfig.param_dict.get(
            "price_cap",
            4000.0,
        )

        master_model = build_market_clearing_model(
            orders=orderbook,
            market_products=market_products,
            with_linked_bids=with_linked_bids,
            with_exclusive_groups=with_exclusive_groups,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
        )
        master_model, master_solution = solve_master_with_price_cuts(
            master_model=master_model,
            orders=orderbook,
            market_products=market_products,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
            price_floor=price_floor,
            price_cap=price_cap,
            solver=self.solver,
            solver_options=self.solver_options,
        )
        master_welfare = float(pyo.value(master_model.welfare))

        market_clearing_prices = select_midpoint_prices(
            orders=orderbook,
            market_products=market_products,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
            master_solution=master_solution,
            price_floor=price_floor,
            price_cap=price_cap,
            solver=self.solver,
            solver_options=self.solver_options,
        )

        final_model = solve_volume_flow_tiebreak(
            orders=orderbook,
            market_products=market_products,
            with_linked_bids=with_linked_bids,
            with_exclusive_groups=with_exclusive_groups,
            market_zone_ids=market_zone_ids,
            fbmc_params=fbmc_params,
            ntc_params=ntc_params,
            master_solution=master_solution,
            master_welfare=master_welfare,
            market_clearing_prices=market_clearing_prices,
            price_floor=price_floor,
            price_cap=price_cap,
            solver=self.solver,
            solver_options=self.solver_options,
        )

        (
            accepted_orders,
            rejected_orders,
            meta,
            flows,
            line_results,
        ) = extract_results(
            model=final_model,
            orders=orderbook,
            rejected_orders=[],
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
            pricing_mechanism=self.pricing_mechanism,
            log_flows=self.log_flows,
            fbmc_parameters=raw_fbmc_parameters,
        )

        self.line_results = line_results
        self.all_orders = []

        return accepted_orders, rejected_orders, meta, flows


def extract_results(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    rejected_orders: Orderbook,
    market_products: list[MarketProduct],
    market_clearing_prices: dict,
    pricing_mechanism: str = "pay_as_clear",
    log_flows: bool = False,
    fbmc_parameters: dict | None = None,
):
    """Extract accepted/rejected orders, prices, and optional flow results."""
    if pricing_mechanism not in ["pay_as_clear", "pay_as_bid"]:
        raise ValueError(f"Invalid pricing mechanism '{pricing_mechanism}'.")

    accepted_orders: Orderbook = []
    meta = []

    supply_volume_dict = {
        zone: {timestamp: 0.0 for timestamp in model.T}
        for zone in model.zones
    }
    demand_volume_dict = {
        zone: {timestamp: 0.0 for timestamp in model.T}
        for zone in model.zones
    }

    for order in orders:
        if order["bid_type"] == "SB":
            acceptance = pyo.value(model.xs[order["bid_id"]])
            acceptance = 0.0 if acceptance < EPS else acceptance

            order["accepted_volume"] = acceptance * order["volume"]

            if pricing_mechanism == "pay_as_clear":
                order["accepted_price"] = market_clearing_prices[
                    order["node"]
                ][order["start_time"]]
            else:
                order["accepted_price"] = order["price"]

            if order["accepted_volume"] > 0:
                supply_volume_dict[order["node"]][
                    order["start_time"]
                ] += order["accepted_volume"]
            else:
                demand_volume_dict[order["node"]][
                    order["start_time"]
                ] += order["accepted_volume"]

        elif order["bid_type"] in ["BB", "LB"]:
            acceptance = pyo.value(model.xb[order["bid_id"]])
            acceptance = 0.0 if acceptance < EPS else acceptance

            for timestamp, volume in order["volume"].items():
                order["accepted_volume"][timestamp] = acceptance * volume

                if pricing_mechanism == "pay_as_clear":
                    order["accepted_price"][timestamp] = (
                        market_clearing_prices[order["node"]][timestamp]
                    )
                else:
                    order["accepted_price"][timestamp] = order["price"]

                if order["accepted_volume"][timestamp] > 0:
                    supply_volume_dict[order["node"]][timestamp] += (
                        order["accepted_volume"][timestamp]
                    )
                else:
                    demand_volume_dict[order["node"]][timestamp] += (
                        order["accepted_volume"][timestamp]
                    )

        if acceptance > 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    for order in rejected_orders:
        if order["bid_type"] == "SB":
            order["accepted_volume"] = 0.0
            order["accepted_price"] = 0.0

        elif order["bid_type"] in ["BB", "LB"]:
            order["accepted_volume"] = {
                timestamp: 0.0
                for timestamp in order["volume"]
            }
            order["accepted_price"] = {
                timestamp: 0.0
                for timestamp in order["volume"]
            }

    for zone in market_clearing_prices:
        for product in market_products:
            timestamp = product[0]
            duration_hours = (
                product[1] - product[0]
            ) / timedelta(hours=1)

            clear_price = market_clearing_prices[zone][timestamp]
            supply_volume = supply_volume_dict[zone][timestamp]
            demand_volume = demand_volume_dict[zone][timestamp]

            meta.append(
                {
                    "supply_volume": supply_volume,
                    "demand_volume": -demand_volume,
                    "demand_volume_energy": -demand_volume * duration_hours,
                    "supply_volume_energy": supply_volume * duration_hours,
                    "price": clear_price,
                    "max_price": clear_price,
                    "min_price": clear_price,
                    "zone": zone,
                    "product_start": product[0],
                    "product_end": product[1],
                    "only_hours": product[2],
                }
            )

    flows = {}
    line_results = []

    if log_flows:
        ntc_flows = {}
        fb_net_exports = {}
        net_positions = {}

        if hasattr(model, "ntc_flow"):
            ntc_flows = {
                (timestamp, line_id): pyo.value(
                    model.ntc_flow[timestamp, line_id]
                )
                for timestamp in model.T
                for line_id in model.ntc_lines
            }

            line_results.extend(
                {
                    "time": timestamp,
                    "line": line_id,
                    "flow": flow,
                    "network_type": "NTC",
                }
                for (timestamp, line_id), flow in ntc_flows.items()
            )

        if hasattr(model, "net_export_fb"):
            fb_net_exports = {
                (timestamp, zone): pyo.value(
                    model.net_export_fb[timestamp, zone]
                )
                for timestamp in model.T
                for zone in model.fbmc_zones
            }

            line_results.extend(
                {
                    "time": timestamp,
                    "line": zone,
                    "flow": flow,
                    "network_type": "FBMC_NET_EXPORT",
                }
                for (timestamp, zone), flow in fb_net_exports.items()
            )

        net_positions = {
            (timestamp, zone): pyo.value(
                model.net_position[timestamp, zone]
            )
            for timestamp in model.T
            for zone in model.zones
        }

        cnec_diagnostics = []

        if (
            fbmc_parameters is not None
            and hasattr(model, "net_export_fb")
            and "cnec_info" in fbmc_parameters
            and "zonal_ptdf" in fbmc_parameters
            and "ram" in fbmc_parameters
        ):
            cnec_info = fbmc_parameters["cnec_info"]
            fb_zones = list(fbmc_parameters["flowbased_region"])

            for timestamp in model.T:
                ptdf = fbmc_parameters["zonal_ptdf"][timestamp]
                ram_values = fbmc_parameters["ram"].loc[timestamp]

                for cnec in model.cnecs:
                    fb_flow = sum(
                        float(ptdf.at[cnec, zone])
                        * pyo.value(
                            model.net_export_fb[timestamp, zone]
                        )
                        for zone in fb_zones
                    )
                    ram = float(ram_values.at[cnec])
                    slack = ram - fb_flow
                    cnec_metadata = cnec_info.loc[cnec]

                    cnec_diagnostics.append(
                        {
                            "time": timestamp,
                            "line": cnec,
                            "flow": fb_flow,
                            "network_type": "FBMC_CNEC",
                            "cnec": cnec,
                            "cne": cnec_metadata["cne"],
                            "contingency": cnec_metadata["contingency"],
                            "direction": cnec_metadata["direction"],
                            "ram": ram,
                            "fb_flow": fb_flow,
                            "slack": slack,
                            "binding": abs(slack) <= EPS,
                            "utilization": (
                                fb_flow / ram
                                if abs(ram) > EPS
                                else None
                            ),
                        }
                    )

            line_results.extend(cnec_diagnostics)

        flows = {
            "ntc_flows": ntc_flows,
            "fb_net_exports": fb_net_exports,
            "net_positions": net_positions,
            "cnec_diagnostics": cnec_diagnostics,
        }

    return (
        accepted_orders,
        rejected_orders,
        meta,
        flows,
        line_results,
    )