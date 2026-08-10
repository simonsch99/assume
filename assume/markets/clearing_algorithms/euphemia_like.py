### This clearing algorithm is an extended version of the comnplex_clearing.py with more extensive grid simulation.


# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict # TODO Check if this can be done in base py
from datetime import timedelta
from operator import itemgetter

import pandas as pd
import pyomo.environ as pyo
from mango import AgentAddress
from pyomo.opt import OptSolver, SolverFactory, TerminationCondition

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import create_incidence_matrix, get_supported_solver_pyomo
from assume.markets.base_market import MarketRole

# Set the log level to WARNING
logging.getLogger("pyomo").setLevel(logging.WARNING)
logging.getLogger("gurobipy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

EPS = 1e-4


def get_exclusive_groups(orders: Orderbook) -> dict[object, list[str]]:
    """
    Collects the exclusive groups of block bids from the orderbook.

    Only block bids ('BB' and 'LB') can be part of an exclusive group. The group
    membership is defined by the optional order field ``exclusive_group_id``.
    Orders without this field or with ``exclusive_group_id`` set to None are ignored.

    Args:
        orders (Orderbook): The list of the orders.

    Returns:
        dict[object, list[str]]: A mapping of the exclusive group id to the list of bid ids
            belonging to this group. Groups containing less than two bids are dropped,
            since the constraint would be redundant.

    Note:
        Exclusive groups model the EUPHEMIA rule that at most one bid of a group of
        mutually exclusive block bids may be accepted.
    """
    groups: dict[object, list[str]] = defaultdict(list)
    for order in orders:
        if order["bid_type"] not in ["BB", "LB"]:
            continue
        group_id = order.get("exclusive_group_id")
        if group_id is None:
            continue
        groups[group_id].append(order["bid_id"])

    # drop groups with a single member, as the constraint is redundant there
    return {
        group_id: bid_ids for group_id, bid_ids in groups.items() if len(bid_ids) > 1
    }


def _get_line_capacity(line_id: str, lines: pd.DataFrame, line_methods: dict, timestep):
    """
    Determines the transmission capacity for a line based on the capacity method.
    
    Args:
        line_id (str): The line identifier.
        lines (pd.DataFrame): The lines DataFrame containing capacity data.
        line_methods (dict): Mapping of line_id to capacity method ("ATC" or "FBMC").
        timestep: The current timestep (for future FBMC time-variant support).
    
    Returns:
        float: The capacity in MW.
    """
    method = line_methods.get(line_id, "ATC") if line_methods else "ATC"
    
    if method == "ATC":
        # ATC: Available Transfer Capacity (current implementation)
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        return lines.at[line_id, "s_nom"] * s_max_pu
    
    elif method == "FBMC":
        # FBMC: Flow-Based Market Coupling (placeholder for future implementation)
        # For now, use same logic as ATC
        # TODO: Implement FBMC-specific logic (e.g., load flow constraints, PTDF matrix)
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        capacity = lines.at[line_id, "s_nom"] * s_max_pu
        logger.debug(f"FBMC capacity for {line_id} at {timestep}: {capacity} MW (placeholder)")
        return capacity
    
    else:
        logger.warning(f"Unknown capacity method '{method}' for line {line_id}. Defaulting to ATC.")
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        return lines.at[line_id, "s_nom"] * s_max_pu

def _add_line_losses_to_balance(
    balance_expr,
    model: pyo.ConcreteModel,
    incidence_matrix: pd.DataFrame,
    line_losses: dict,
    node: str,
    t,
):
    """
    Adds loss terms to the energy balance for a given node and timestep.
    
    Losses are modeled as: loss = loss_factor * |flow|
    For positive flows (A→B): loss occurs at destination node B
    For negative flows (B→A): loss occurs at destination node A
    
    Args:
        balance_expr: The current balance expression
        model: The Pyomo model
        incidence_matrix: The incidence matrix
        line_losses: Dict mapping line_id to loss factor (0.02 = 2%)
        node: The node for which to calculate losses
        t: The timestep
    
    Returns:
        Updated balance expression with loss terms
    """
    if line_losses is None or not any(line_losses.values()):
        return balance_expr
    
    for line in model.lines:
        loss_factor = line_losses.get(line, 0.0)
        if loss_factor == 0.0:
            continue
        
        incidence_value = incidence_matrix.loc[node, line]
        if incidence_value == 0:
            continue
        
        # Loss is proportional to |flow|, which requires absolute value
        # In linear form: loss = loss_factor * flow (for positive flow direction)
        # For bidirectional: we use piecewise linear or accept the approximation
        # Here we use: loss_outflow = loss_factor * max(0, flow)
        # This is implemented via additional variables if needed, or simplified
        
        flow = model.flows[t, line]
        
        # Simplified: assume losses on outflow (positive flow direction)
        # This is a linear approximation; full nonlinear would require abs()
        if incidence_value > 0:  # Outflow from this node
            balance_expr += loss_factor * flow
    
    return balance_expr


def _add_line_tariffs_to_objective(
    obj_expr,
    model: pyo.ConcreteModel,
    line_tariffs: dict,
    market_products: list[MarketProduct],
):
    """
    Adds tariff costs to the objective function.
    
    Tariffs are charged per MWh transported on each line.
    
    Args:
        obj_expr: The current objective expression
        model: The Pyomo model
        line_tariffs: Dict mapping line_id to tariff (€/MWh)
        market_products: List of market products (for duration)
    
    Returns:
        Updated objective expression with tariff costs
    """
    if line_tariffs is None or not any(line_tariffs.values()):
        return obj_expr
    
    for t in model.T:
        for line in model.lines:
            tariff = line_tariffs.get(line, 0.0)
            if tariff == 0.0:
                continue
            
            flow = model.flows[t, line]
            # Tariff on absolute flow
            obj_expr += tariff * pyo.abs(flow)
    
    return obj_expr



    
def market_clearing_opt_constraints(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    market_products: list[MarketProduct],
    mode: str,
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    incidence_matrix: pd.DataFrame,
    lines: pd.DataFrame,
    line_methods: dict = None,
    line_losses: dict = None,
    line_tariffs: dict = None,
    line_ramping: dict = None,
):
    """
    Adds the constraints to the model.
    Used by :meth:`market_clearing_opt`.
    """
    # Set nodes and lines based on the incidence matrix and lines DataFrame
    if incidence_matrix is not None:
        nodes = list(incidence_matrix.index)
        line_ids = list(incidence_matrix.columns)
    else:
        nodes = ["node0"]
        line_ids = ["line0"]

    # add dual suffix to the model (we need this to extract the market clearing prices later)
    # if mode is not 'with_min_acceptance_ratio', otherwise the dual suffix is added later
    if mode != "with_min_acceptance_ratio":
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )

    model.sBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "SB"],
        doc="simple_bids",
    )

    model.bBids = pyo.Set(
        initialize=[
            order["bid_id"] for order in orders if order["bid_type"] in ["BB", "LB"]
        ],
        doc="block_bids",
    )

    model.nodes = pyo.Set(initialize=nodes, doc="nodes")
    model.lines = pyo.Set(initialize=line_ids, doc="lines")

    # decision variables for the acceptance ratio of simple and block bids (including linked bids)
    model.xs = pyo.Var(
        model.sBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="simple_bid_acceptance",
    )

    model.xb = pyo.Var(
        model.bBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="block_bid_acceptance",
    )

    # decision variables that define flows between nodes
    # assuming the orders contain the node and are collected in nodes
    if incidence_matrix is not None:
        # Decision variables for flows on each line at each timestep
        model.flows = pyo.Var(model.T, model.lines, domain=pyo.Reals, doc="power_flows")

    if mode == "with_min_acceptance_ratio":
        model.Bids = pyo.Set(
            initialize=[order["bid_id"] for order in orders], doc="all_bids"
        )
        # decision variables for the acceptance as binary variable
        model.x = pyo.Var(
            model.Bids,
            domain=pyo.Binary,
            doc="bid_accepted",
        )

        # add minimum acceptance ratio constraints
        model.mar_constr = pyo.ConstraintList()
        for order in orders:
            if order["min_acceptance_ratio"] is None:
                continue
            elif order["bid_type"] == "SB":
                model.mar_constr.add(
                    model.xs[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xs[order["bid_id"]] <= model.x[order["bid_id"]]
                )

            elif order["bid_type"] in ["BB", "LB"]:
                model.mar_constr.add(
                    model.xb[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xb[order["bid_id"]] <= model.x[order["bid_id"]]
                )

    # limit the acceptance of child bids by the acceptance of their parent bid
    if with_linked_bids:
        model.linked_bid_constr = pyo.ConstraintList()
        for order in orders:
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                parent_bid_id = order["parent_bid_id"]
                model.linked_bid_constr.add(
                    model.xb[order["bid_id"]] <= model.xb[parent_bid_id]
                )

    # limit the acceptance of mutually exclusive block bids to at most one bid per group
    if with_exclusive_groups:
        exclusive_groups = get_exclusive_groups(orders)
        model.exclusive_group_constr = pyo.ConstraintList()
        for group_id, bid_ids in exclusive_groups.items():
            # only consider bids that are still part of the model
            bid_ids = [bid_id for bid_id in bid_ids if bid_id in model.bBids]
            if len(bid_ids) < 2:
                continue
            if mode == "with_min_acceptance_ratio": # TODO Check if this is correct 
                # if binaries are available, the stricter formulation on the
                # acceptance indicators is used (at most one bid selected)
                model.exclusive_group_constr.add(
                    sum(model.x[bid_id] for bid_id in bid_ids) <= 1
                )
            else:
                # without binaries the acceptance ratios of the group are limited
                model.exclusive_group_constr.add(
                    sum(model.xb[bid_id] for bid_id in bid_ids) <= 1
                )

    # Function to calculate the balance for each node and time
    def energy_balance_rule(model, node, t):
        """
        Calculate the energy balance for a given node and time.

        This function calculates the energy balance for a specific node and time in a complex clearing algorithm. It iterates over the orders and adjusts the balance expression based on the bid type. It also adjusts the flow subtraction to account for actual connections if an incidence matrix is provided.

        Args:
            model: The complex clearing model.
            node: The node for which to calculate the energy balance.
            t: The time for which to calculate the energy balance.

        Returns:
            bool: True if the energy balance is zero, False otherwise.
        """
        balance_expr = 0.0  # Initialize the balance expression
        # Iterate over orders to adjust the balance expression based on bid type
        for order in orders:
            if (
                order["bid_type"] == "SB"
                and order["node"] == node
                and order["start_time"] == t
            ):
                balance_expr += order["volume"] * model.xs[order["bid_id"]]
            elif order["bid_type"] in ["BB", "LB"] and order["node"] == node:
                for start_time, volume in order["volume"].items():
                    if start_time == t:
                        balance_expr += volume * model.xb[order["bid_id"]]

        # Add contributions from line flows based on the incidence matrix
        if incidence_matrix is not None:
            for line in model.lines:
                incidence_value = incidence_matrix.loc[node, line]
                if incidence_value != 0:
                    balance_expr += incidence_value * model.flows[t, line]

        # Add loss terms if losses are defined
        if line_losses is not None and any(line_losses.values()):
            balance_expr = _add_line_losses_to_balance(
                balance_expr, model, incidence_matrix, line_losses, node, t
            )

        return balance_expr == 0

    # Add the energy balance constraints for each node and time period using the rule
    # Define the energy balance constraint using two indices (node and time)
    model.energy_balance = pyo.Constraint(
        model.nodes, model.T, rule=energy_balance_rule
    )

    if incidence_matrix is not None:
        # add capacities and ramping constraints
        model.transmission_constr = pyo.ConstraintList()
        model.ramping_constr = pyo.ConstraintList()
        for t in model.T:
            for line in model.lines:
                capacity = _get_line_capacity(line, lines, line_methods, t)
                # Limit the flow on each line
                model.transmission_constr.add(model.flows[t, line] <= capacity)
                model.transmission_constr.add(model.flows[t, line] >= -capacity)

        if line_ramping is not None and any(v is not None for v in line_ramping.values()):
            t_list = sorted(model.T)
            for idx in range (len(t_list) - 1):
                t_current = t_list[idx]
                t_next = t_list[idx + 1]

                for line in model.lines:
                    ramp_limit = line_ramping.get(line)
                    if ramp_limit is None or ramp_limit <= 0:
                        continue
                    # Calculate time delta in hours TODO check if this is needed
                    time_delta = (t_next - t_current) / timedelta(hours=1)
                    max_change = ramp_limit * time_delta
                    # Ramping constraints: Absolute flow change <= max_change
                    model.ramping_constr.add(
                        model.flows[t_next, line] - model.flows[t_current, line] <= max_change
                    )
                    model.ramping_constr.add(
                        model.flows[t_next, line] - model.flows[t_current, line] >= -max_change
                    )


def market_clearing_opt_objective(model: pyo.ConcreteModel, orders: Orderbook, line_tariffs: dict = None,):
    """
    Define the objective function as cost minimization
    Used by :meth:`market_clearing_opt`
    """
    obj_expr = 0
    for order in orders:
        if order["bid_type"] == "SB":
            obj_expr += order["price"] * order["volume"] * model.xs[order["bid_id"]]
        elif order["bid_type"] in ["BB", "LB"]:
            for start_time, volume in order["volume"].items():
                obj_expr += order["price"] * volume * model.xb[order["bid_id"]]

    # Add tariff costs if tariffs are defined
    if line_tariffs is not None and any(line_tariffs.values()):
        obj_expr = _add_line_tariffs_to_objective(obj_expr, model, line_tariffs, [])
    

    model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)


def market_clearing_opt(
    orders: Orderbook,
    market_products: list[MarketProduct],
    mode: str,
    with_linked_bids: bool,
    with_exclusive_groups: bool,
    incidence_matrix: pd.DataFrame = None,
    lines: pd.DataFrame = None,
    line_methods: dict = None,
    line_losses: dict = None,
    line_tariffs: dict = None,
    line_ramping: dict = None,
    solver: OptSolver = None,
    solver_options: dict = {},
    func_constraints=market_clearing_opt_constraints,
    func_objective=market_clearing_opt_objective,
):
    """
    Sets up and solves the market clearing optimization problem.

    Args:
        orders (Orderbook): The list of the orders.
        market_products (list[MarketProduct]): The products to be traded.
        mode (str): The mode of the market clearing determining whether the minimum acceptance ratio is considered.
        with_linked_bids (bool): Whether the market clearing should include linked bids.
        with_exclusive_groups (bool): Whether the market clearing should include exclusive groups of block bids.
        incidence_matrix (pd.DataFrame): The directed incidence matrix of the network.
        lines (pd.DataFrame): The lines and their capacities of the network.
        solver (pyomo.opt.OptSolver): Specifies the solver instance to be used for the optimization problem.
        solver_options (dict): Additional solver options.
        func_constraints: The function that is executed to add the constraints to the model. Defaults to :meth:`market_clearing_opt_constraints`.
        func_objective: The function that is executed to add the objective function to the model. Defaults to :meth:`market_clearing_opt_objective`.

    Returns:
        tuple[pyomo.core.base.PyomoModel.ConcreteModel, pyomo.opt.results.SolverResults]: The solved pyomo model and the solver results

    Note:
        The problem is formulated as a mixed-integer linear program (MILP) and solved using the pyomo package.
        The objective function is to maximize the social welfare and defined as the sum of the product of the price, volume, and acceptance ratio of each order.
        The decision variables are given by the acceptance ratio of each order bounded by 0 and 1 and the acceptance as a binary variable.

        The energy balance constraint ensures that the sum of the accepted volumes of all orders is zero.
        The acceptance of each order is bounded by 0 and 1.

        If the mode is 'with_min_acceptance_ratio', the minimum acceptance ratio is considered.
        The minimum acceptance ratio is defined as the ratio of the minimum volume to accept to the total volume of the order.

        If linked bids are considered, the acceptance of a child bid is bounded by the acceptance of its parent bid.

        If exclusive groups are considered, at most one block bid of each exclusive group can be accepted.

        The market clearing is solved using pyomo with the specified solver_name (HIGHS is used by default).
        If the specified solver is not available, the model is solved using available solver.
        If none of the solvers are available, an exception is raised.

        After solving the model, the acceptance of each order is fixed to the value in the solution and the model is solved again.
        This removes all binary variables from the model and allows to extract the market clearing prices from the dual variables of the energy balance constraint.

    """
    model = pyo.ConcreteModel()

    func_constraints(
        model,
        orders,
        market_products,
        mode,
        with_linked_bids,
        with_exclusive_groups,
        incidence_matrix,
        lines,
        line_methods,
        line_losses,
        line_tariffs,
        line_ramping,
    )

    func_objective(model, orders, line_tariffs)

    # Solve the model
    instance = model.create_instance()
    results = solver.solve(instance, options=solver_options)

    # Fix all model.x to the values in the solution
    if mode == "with_min_acceptance_ratio":
        # Add dual suffix to the model (needed to extract duals later)
        instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        for bid_id in instance.Bids:
            # Fix the binary variable to its value
            value = instance.x[bid_id].value
            if value is not None:
                value = 1 if value >= 0.99 else 0
            instance.x[bid_id].fix(value)
            # Change the domain to Reals (or appropriate continuous domain)
            instance.x[bid_id].domain = pyo.Reals

        # Resolve the model
        results = solver.solve(instance, options=solver_options)

    return instance, results


class ComplexClearingRole(MarketRole):
    """
    This class defines an optimization-based market clearing algorithm with support for complex bid types,
    including block bids, linked bids, exclusive groups of block bids, minimum acceptance ratios, and profiled
    volumes. It supports network representations (through Net Transfer Capacities) with either zonal or nodal
    configurations, enabling the modeling of complex markets with multiple zones based on a transport model.

    The market clearing algorithm accepts additional arguments via the ``param_dict`` in the market configuration.

    Args:
        marketconfig (MarketConfig): The market configuration object containing all parameters for the market clearing process.

    Attributes:
        marketconfig (MarketConfig): The market configuration.
        incidence_matrix (pd.DataFrame): The directed incidence matrix representing the power network connections (definition according to Wikipedia https://en.wikipedia.org/wiki/Incidence_matrix)
        nodes (list): List of nodes or zones in the network, depending on the selected representation.

    Supported Parameters in ``param_dict``:
        - ``solver_name`` (str): Specifies the solver_name to be used for the optimization problem. Default is `'appsi_highs'`.
        - ``log_flows`` (bool): Indicates whether to log the power flows on the lines. Default is `False`.
        - ``pricing_mechanism`` (str): Defines the pricing mechanism to be used. Default is `'pay_as_clear'`, with an alternative option of `'pay_as_bid'`.
        - ``zones_identifier`` (str): The key in the bus data that identifies the zone each bus belongs to. Used for zonal representation.

    Supported Additional Fields:
        - ``min_acceptance_ratio`` (float): Activates the minimum acceptance ratio mode.
        - ``parent_bid_id`` (str): Activates linked bids.
        - ``exclusive_group_id``: Activates exclusive groups. Only evaluated for block bids ('BB', 'LB').
          At most one block bid per group can be accepted.

    Example market configuration:

    .. code-block:: yaml

        market_mechanism: complex_clearing
        additional_fields:
            - min_acceptance_ratio
            - parent_bid_id
            - exclusive_group_id
        param_dict:
            solver_name: appsi_highs
            log_flows: true
            pricing_mechanism: pay_as_clear
            zones_identifier: zone_id

    Network Representations:
        - **Zonal Representation**: The network is divided into zones, and the incidence matrix represents the connections between these zones.
            - If a ``zones_identifier`` is provided, buses are grouped into zones based on this identifier.
            - The incidence matrix is constructed to represent the power connections between these zones.
            - The total transfer capacity between zones is determined by the sum of the capacities of the lines connecting the zones.
            - The line type can be either set to "ATC" (default) or FBMC (currently only a placeholder).
            - For ATC lines losses, tariffs and ramping constraints can be added.

        - **Nodal Representation**: If no ``zones_identifier`` is provided, each bus is treated as a separate node, and the incidence matrix represents the connections between these nodes.
    """

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

        # Define grid data
        self.nodes = ["node0"]
        self.zones_id = None
        self.incidence_matrix = None
        self.lines = None
        self.line_methods = {}
        self.line_losses = {}
        self.line_tariffs = {}
        self.line_ramping = {}

        if self.grid_data:
            self.lines = self.grid_data["lines"]
            buses = self.grid_data["buses"]

            if "x" in self.lines.columns:
                logger.warning(
                    "Warning: 'lines.csv' contains reactances 'x' but this clearing is based on Net Transfer Capacities only (Transport model). Use 'nodal_clearing' to include a linear OPF."
                )

             # Extract capacity calculation method per line (ATC or FBMC)
            if "capacity_method" in self.lines.columns:
                self.line_methods = self.lines["capacity_method"].to_dict()
            else:
                # Default to ATC if not specified
                self.line_methods = {line_id: "ATC" for line_id in self.lines.index}
                logger.info("No 'capacity_method' column found in lines. Defaulting all lines to ATC.")

            # Extrace losses per line (as percentage)
            if "losses" in self.lines.columns:
                self.line_losses = self.lines["losses"].to_dict()

            # Extract tariffs per line (€/MWh)
            if "tariff" in self.lines.columns:
                self.line_tariffs = self.lines["tariff"].to_dict()

            # Extract ramoping limits per line (MW/h)
            if "ramping_limit" in self.lines.columns:
                self.line_ramping = self.lines["ramping_limit"].to_dict()
            else:
                self.line_ramping = {line_id: None for line_id in self.lines.index}

            self.zones_id = self.marketconfig.param_dict.get("zones_identifier")
            self.node_to_zone = None

            # Generate the incidence matrix and set the nodes based on zones or individual buses
            if self.zones_id:
                # Zonal Case
                self.incidence_matrix = create_incidence_matrix(
                    self.lines, buses, zones_id=self.zones_id
                )
                self.nodes = buses[self.zones_id].unique()
                self.node_to_zone = buses[self.zones_id].to_dict()
            else:
                # Nodal Case
                self.incidence_matrix = create_incidence_matrix(self.lines, buses)
                self.nodes = buses.index.values

        self.log_flows = self.marketconfig.param_dict.get("log_flows", False)
        self.pricing_mechanism = self.marketconfig.param_dict.get(
            "pricing_mechanism", "pay_as_clear"
        )

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr: AgentAddress
    ) -> None:
        """
        Checks whether the bid types are valid and whether the volumes are within the maximum bid volume.
        Additionally, exclusive group ids are validated and only kept for block bids.

        Args:
            orderbook (Orderbook): The orderbook to be validated.
            agent_addr (AgentAddress): The agent address of the market.

        Raises:
            ValueError: If the bid type is invalid.
        """
        market_id = self.marketconfig.market_id
        max_volume = self.marketconfig.maximum_bid_volume

        with_exclusive_groups = (
            "exclusive_group_id" in self.marketconfig.additional_fields
        )

        for order in orderbook:
            # if bid_type is None, set to default bid_type
            if order.get("bid_type") is None:
                order["bid_type"] = "SB"
            # Validate bid_type
            elif order["bid_type"] not in ["SB", "BB", "LB"]:
                logger.warning(
                    f"Market '{market_id}': Invalid bid_type '{order['bid_type']}' in order {order}. Setting to 'SB'."
                )
                order["bid_type"] = "SB"  # Set to default bid_type

        super().validate_orderbook(orderbook, agent_addr)

        for order in orderbook:
            # Validate volumes
            if order["bid_type"] in ["BB", "LB"]:
                for key, volume in order.get("volume", {}).items():
                    if abs(volume) > max_volume:
                        logger.warning(
                            f"Market '{market_id}': Volume '{volume}' for key '{key}' exceeds max_volume {max_volume} in order {order}. Setting to max_volume."
                        )
                        order["volume"][key] = max_volume if volume > 0 else -max_volume

            # Exclusive group validation, only block bids can be part of a group
            if with_exclusive_groups:
                if order.get("exclusive_group_id") is None:
                    order["exclusive_group_id"] = None
                elif order["bid_type"] == "SB":
                    logger.warning(
                        f"Market '{market_id}': Exclusive groups are only supported for block bids, "
                        f"but order {order['bid_id']} is of bid_type 'SB'. Ignoring 'exclusive_group_id'."
                    )
                    order["exclusive_group_id"] = None

            # Node validation
            node = order.get("node")
            if node:
                if self.zones_id:
                    node = self.node_to_zone.get(node, self.nodes[0])
                    order["node"] = node
                if node not in self.nodes:
                    logger.warning(
                        f"Market '{market_id}': Node '{node}' not in nodes list {self.nodes}. Setting to first node '{self.nodes[0]}'. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
            else:
                if self.incidence_matrix is not None:
                    logger.warning(
                        f"Market '{market_id}': Order without a node, setting node to the first node '{self.nodes[0]}'. Please check the bidding strategy if correct node is set. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
                else:
                    logger.warning(
                        f"Market '{market_id}': Order without a node and no incidence matrix, setting node to 'node0'. Order details: {order}"
                    )
                    order["node"] = "node0"

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Implements pay-as-clear with more complex bid structures, including acceptance ratios, bid types, and profiled volumes.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products to be traded.

        Raises:
            Exception: If the problem is infeasible.

        Returns:
            accepted_orders (Orderbook): The accepted orders.
            rejected_orders (Orderbook): The rejected orders.
            meta (list[dict]): The market clearing results.
            flows (dict): The power flows on the lines.

        Note:
            First the market clearing is solved using the cost minimization with the pyomo model market_clearing_opt.
            Then the market clearing prices are extracted from the solved model as dual variables of the energy balance constraint.
            Next the surplus of each order and its children is calculated and orders with negative surplus are removed from the orderbook.
            This is repeated until all orders remaining in the orderbook have positive surplus.
            Optional additional fields are: min_acceptance_ratio, parent_bid_id, exclusive_group_id, node
        """

        if len(orderbook) == 0:
            return [], [], []

        orderbook.sort(key=itemgetter("start_time", "end_time", "only_hours"))

        # create a list of all orders linked as child to a bid
        child_orders = []
        for order in orderbook:
            order["accepted_price"] = {}
            order["accepted_volume"] = {}
            # get child linked bids
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                # check whether the parent bid is in the orderbook
                parent_bid_id = order["parent_bid_id"]
                parent_bid = next(
                    (bid for bid in orderbook if bid["bid_id"] == parent_bid_id), None
                )
                if parent_bid is None:
                    order["parent_bid_id"] = None
                    logger.warning(f"Parent bid {parent_bid_id} not in orderbook")
                else:
                    child_orders.append(order)

        with_linked_bids = bool(child_orders)

        # check whether exclusive groups of block bids are present in the orderbook
        with_exclusive_groups = bool(get_exclusive_groups(orderbook))

        rejected_orders: Orderbook = []

        mode = "default"
        if "min_acceptance_ratio" in self.marketconfig.additional_fields:
            mode = "with_min_acceptance_ratio"

        # solve the market clearing problem
        while True:
            # solve the optimization with the current orderbook
            instance, results = market_clearing_opt(
                orders=orderbook,
                market_products=market_products,
                mode=mode,
                with_linked_bids=with_linked_bids,
                with_exclusive_groups=with_exclusive_groups,
                incidence_matrix=self.incidence_matrix,
                lines=self.lines,
                line_methods=self.line_methods,
                line_losses=self.line_losses,
                line_tariffs=self.line_tariffs,
                line_ramping = self.line_ramping,
                solver=self.solver,
                solver_options=self.solver_options,
            )

            if results.solver.termination_condition == TerminationCondition.infeasible:
                raise Exception("infeasible")

            # extract dual from model.energy_balance
            market_clearing_prices = {}
            for node in self.nodes:
                market_clearing_prices[node] = {
                    t: instance.dual[instance.energy_balance[node, t]]
                    for t in instance.T
                }

            # check the surplus of each order and remove those with negative surplus
            orders_surplus = []
            for order in orderbook:
                children = []
                if with_linked_bids:
                    children = [
                        child
                        for child in child_orders
                        if child["parent_bid_id"] == order["bid_id"]
                    ]

                order_surplus = calculate_order_surplus(
                    order, market_clearing_prices, instance, children
                )

                # correct rounding
                if order_surplus != 0 and abs(order_surplus) < EPS:
                    order_surplus = 0

                orders_surplus.append(order_surplus)

                # remove orders with negative profit
                if order_surplus < 0:
                    rejected_orders.append(order)
                    orderbook.remove(order)
                    rejected_orders.extend(children)
                    for child in children:
                        orderbook.remove(child)

            # check if all orders have positive surplus
            if all(order_surplus >= 0 for order_surplus in orders_surplus):
                break

        accepted_orders, rejected_orders, meta, flows = extract_results(
            model=instance,
            orders=orderbook,
            rejected_orders=rejected_orders,
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
            pricing_mechanism=self.pricing_mechanism,
            log_flows=self.log_flows,
        )

        self.all_orders = []

        return accepted_orders, rejected_orders, meta, flows


def calculate_order_surplus(
    order: dict,
    market_clearing_prices: dict,
    instance: pyo.ConcreteModel,
    children: list[dict],
):
    """
    Calculates the surplus of an order given the market clearing prices and results of the market clearing.

    Args:
        order (dict): The order
        market_clearing_prices (dict): The market clearing prices.
        instance (pyomo.core.base.PyomoModel.ConcreteModel): The solved pyomo model containing the results of the market clearing.
        children (list[dict]): The linked child bids of the given order.

    Returns:
        float: The surplus of the order as (market_clearing_price - order_price) * order_volume * order_acceptance

    Note:
        The surplus of children linked to the given order is added if it is positive to account for the rule that children can 'save' their parent bid.
    """

    order_surplus = 0

    # calculate the surplus of simple bids
    if order["bid_type"] == "SB":
        if (
            pyo.value(instance.xs[order["bid_id"]]) < EPS
            or abs(
                market_clearing_prices[order["node"]][order["start_time"]]
                - order["price"]
            )
            < EPS
        ):
            order_surplus = 0
        else:
            order_surplus = (
                (
                    market_clearing_prices[order["node"]][order["start_time"]]
                    - order["price"]
                )
                * order["volume"]
                * pyo.value(instance.xs[order["bid_id"]])
            )
    # calculate the surplus of block bidx
    elif order["bid_type"] in ["BB", "LB"]:
        bid_volume = sum(order["volume"].values())
        if pyo.value(instance.xb[order["bid_id"]]) < EPS:
            order_surplus = 0
        else:
            order_surplus = (
                sum(
                    market_clearing_prices[order["node"]][t] * v
                    for t, v in order["volume"].items()
                )
                - order["price"] * bid_volume
            ) * pyo.value(instance.xb[order["bid_id"]])

        # add the surplus of child linked bids if it is positive
        for child_order in children:
            child_surplus = (
                sum(
                    market_clearing_prices[child_order["node"]][t] * v
                    for t, v in child_order["volume"].items()
                )
                - child_order["price"] * bid_volume
            ) * pyo.value(instance.xb[child_order["bid_id"]])
            if child_surplus > 0:
                order_surplus += child_surplus

    # correct rounding
    if order_surplus != 0 and abs(order_surplus) < EPS:
        order_surplus = 0

    return order_surplus


def extract_results(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    rejected_orders: Orderbook,
    market_products: list[MarketProduct],
    market_clearing_prices: dict,
    pricing_mechanism: str = "pay_as_clear",
    log_flows: bool = False,
):
    """
    Extracts the results of the market clearing from the solved pyomo model.

    Args:
        model (pyomo.core.base.PyomoModel.ConcreteModel): The solved pyomo model containing the results of the market clearing
        orders (Orderbook): List of the orders
        rejected_orders (Orderbook): List of the rejected orders
        market_products (list[MarketProduct]): The products to be traded
        market_clearing_prices (dict): The market clearing prices

    Returns:
        tuple[Orderbook, Orderbook, list[dict]]: The accepted orders, rejected orders, and meta information

    """
    if pricing_mechanism not in ["pay_as_clear", "pay_as_bid"]:
        raise ValueError(f"Invalid pricing mechanism {pricing_mechanism}")

    accepted_orders: Orderbook = []
    meta = []

    supply_volume_dict = {node: {t: 0.0 for t in model.T} for node in model.nodes}
    demand_volume_dict = {node: {t: 0.0 for t in model.T} for node in model.nodes}

    for order in orders:
        if order["bid_type"] == "SB":
            acceptance = model.xs[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance

            # set the accepted volume and price for each simple bid
            order["accepted_volume"] = acceptance * order["volume"]
            if pricing_mechanism == "pay_as_clear":
                order["accepted_price"] = market_clearing_prices[order["node"]][
                    order["start_time"]
                ]
            elif pricing_mechanism == "pay_as_bid":
                order["accepted_price"] = order["price"]

            # calculate the total cleared supply and demand volume
            if order["accepted_volume"] > 0:
                supply_volume_dict[order["node"]][order["start_time"]] += order[
                    "accepted_volume"
                ]
            else:
                demand_volume_dict[order["node"]][order["start_time"]] += order[
                    "accepted_volume"
                ]

        elif order["bid_type"] in ["BB", "LB"]:
            acceptance = model.xb[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance

            # set the accepted volume and price for each block bid
            for start_time, volume in order["volume"].items():
                order["accepted_volume"][start_time] = acceptance * volume
                if pricing_mechanism == "pay_as_clear":
                    order["accepted_price"][start_time] = market_clearing_prices[
                        order["node"]
                    ][start_time]
                elif pricing_mechanism == "pay_as_bid":
                    order["accepted_price"][start_time] = order["price"]

                # calculate the total cleared supply and demand volume
                if order["accepted_volume"][start_time] > 0:
                    supply_volume_dict[order["node"]][start_time] += order[
                        "accepted_volume"
                    ][start_time]
                else:
                    demand_volume_dict[order["node"]][start_time] += order[
                        "accepted_volume"
                    ][start_time]

        if acceptance > 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    for order in rejected_orders:
        # set the accepted volume and price for each rejected order to zero
        if order["bid_type"] == "SB":
            order["accepted_volume"] = 0
            order["accepted_price"] = 0

        elif order["bid_type"] in ["BB", "LB"]:
            order["accepted_volume"] = {t: 0 for t in order["volume"].keys()}
            order["accepted_price"] = {t: 0 for t in order["volume"].keys()}

    # write the meta information for each hour of the clearing period
    for node in market_clearing_prices.keys():
        for product in market_products:
            t = product[0]

            clear_price = market_clearing_prices[node][t]

            supply_volume = supply_volume_dict[node][t]
            demand_volume = demand_volume_dict[node][t]
            duration_hours = (product[1] - product[0]) / timedelta(hours=1)

            meta.append(
                {
                    "supply_volume": supply_volume,
                    "demand_volume": -demand_volume,
                    "demand_volume_energy": -demand_volume * duration_hours,
                    "supply_volume_energy": supply_volume * duration_hours,
                    "price": clear_price,
                    "max_price": clear_price,
                    "min_price": clear_price,
                    "node": node,
                    "product_start": product[0],
                    "product_end": product[1],
                    "only_hours": product[2],
                }
            )

    flows_filtered = {}

    if log_flows:
        # extract flows

        # Check if the model has the 'flows' attribute
        if hasattr(model, "flows"):
            flows = model.flows

            flows_filtered = {
                index: flow.value for index, flow in flows.items() if not flow.stale
            }

    return accepted_orders, rejected_orders, meta, flows_filtered