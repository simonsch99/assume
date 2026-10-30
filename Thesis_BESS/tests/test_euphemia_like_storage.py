# test_euphemia_like_storage.py
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Validation of Storage Order (SO) logic.

All scenarios use three fixed 15-minute MTUs with an increasing price
pattern of 10, 60 and 100 EUR/MWh. The price pattern is created by
oversized simple supply offers, so each zonal price is pinned by a
partially accepted simple bid and is independent of the storage schedule.

Storage orders are always returned as accepted orders by the current
implementation. The tests therefore assert ``charge``, ``discharge``,
``soc`` and ``accepted_volume`` instead of acceptance status.
"""

from __future__ import annotations

import pytest

from .test_euphemia_like_tools import (
    EPS,
    MTU_HOURS,
    PRICE_TOL,
    assert_isolated_zone_balances,
    assert_ntc_balance,
    assert_ntc_capacity_bounds,
    assert_simple_order_price_consistency,
    assert_storage_schedule_feasible,
    clear_market,
    find_order,
    get_ntc_flow,
    get_price,
    make_market_config,
    make_ntc_dataframe,
    make_products,
    make_simple_bid,
    make_storage_order,
    product_timestamps,
    requires_solver,
)

ZONE = "Z1"
ZONE_A = "ZA"
ZONE_B = "ZB"

#: Marginal supply prices per MTU in EUR/MWh.
SUPPLY_PRICES = (10.0, 60.0, 100.0)

#: Demand per MTU in MW.
DEMANDS = (100.0, 200.0, 200.0)


def _price_pattern_orders(
    timestamps: list,
    node: str = ZONE,
    supply_volume: float = 400.0,
) -> list[dict]:
    """Create the oversized supply and inelastic demand of each MTU."""
    orders = []

    for index, timestamp in enumerate(timestamps):
        orders.append(
            make_simple_bid(
                f"dem_{index}",
                timestamp,
                -DEMANDS[index],
                3000.0,
                node=node,
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_{index}",
                timestamp,
                supply_volume,
                SUPPLY_PRICES[index],
                node=node,
            )
        )

    return orders


def _assert_price_pattern(result: dict, timestamps: list, zone: str = ZONE):
    """Assert the expected pinned price pattern."""
    for index, timestamp in enumerate(timestamps):
        assert get_price(result["meta"], zone, timestamp) == pytest.approx(
            SUPPLY_PRICES[index], abs=PRICE_TOL
        )


# ---------------------------------------------------------------------------
# Arbitrage, SOC recursion and bounds
# ---------------------------------------------------------------------------


@requires_solver
def test_basic_storage_arbitrage():
    """
    Price arbitrage over three MTUs.

    A lossless 25 MWh / 100 MW battery buys energy in the 10 EUR/MWh MTU
    and sells it in the 100 EUR/MWh MTU. Charging 100 MW for 0.25 h fills
    the storage exactly, and discharging 100 MW in the last MTU empties it
    again. Selling in the 60 EUR/MWh MTU is strictly worse, so the schedule
    is unique.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = _price_pattern_orders(timestamps)
    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            max_input_power=100.0,
            max_output_power=100.0,
            node=ZONE,
        )
    )

    result = clear_market(config, orders, products)
    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    assert storage["charge"][timestamps[0]] == pytest.approx(100.0, abs=1e-4)
    assert storage["discharge"][timestamps[2]] == pytest.approx(
        100.0, abs=1e-4
    )

    # Diagnostic: simultaneous operation is clearly dominated here.
    for timestamp in timestamps:
        assert min(
            storage["charge"][timestamp], storage["discharge"][timestamp]
        ) == pytest.approx(0.0, abs=1e-4)

    assert storage["accepted_volume"][timestamps[0]] == pytest.approx(
        -100.0, abs=1e-4
    )
    assert storage["accepted_volume"][timestamps[2]] == pytest.approx(
        100.0, abs=1e-4
    )

    _assert_price_pattern(result, timestamps)
    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


@requires_solver
def test_storage_soc_recursion_and_bounds():
    """
    SOC recursion, energy bounds and power limits.

    The schedule of the arbitrage case is verified independently against

        soc_t = soc_{t-1} + 0.25 * efficiency * charge_t - 0.25 * discharge_t

    with an explicit starting energy of 5 MWh, a minimum energy of 2 MWh
    and asymmetric power limits of 80 MW input and 40 MW output.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = _price_pattern_orders(timestamps)
    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            min_energy=2.0,
            max_input_power=80.0,
            max_output_power=40.0,
            starting_energy=5.0,
            node=ZONE,
        )
    )

    result = clear_market(config, orders, products)
    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    # Explicit recursion check including the starting energy.
    previous = 5.0
    for timestamp in timestamps:
        expected = (
            previous
            + MTU_HOURS * storage["charge"][timestamp]
            - MTU_HOURS * storage["discharge"][timestamp]
        )
        assert storage["soc"][timestamp] == pytest.approx(expected, abs=1e-4)
        assert 2.0 - 1e-4 <= storage["soc"][timestamp] <= 25.0 + 1e-4
        assert storage["charge"][timestamp] <= 80.0 + 1e-4
        assert storage["discharge"][timestamp] <= 40.0 + 1e-4
        previous = storage["soc"][timestamp]

    # Energy is still shifted towards the most expensive MTU.
    assert storage["discharge"][timestamps[2]] > storage["discharge"][
        timestamps[1]
    ] - 1e-4

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


@requires_solver
def test_storage_efficiency_losses():
    """
    Efficiency below one.

    With an efficiency of 0.5, charging 100 MW for 0.25 h adds only
    0.25 * 0.5 * 100 = 12.5 MWh of stored energy. Buying at 60 EUR/MWh for
    a 100 EUR/MWh sale is unprofitable at this efficiency, so only the
    cheapest MTU is used for charging and the discharge power is limited by
    the available 12.5 MWh.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = _price_pattern_orders(timestamps)
    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            max_input_power=100.0,
            max_output_power=100.0,
            efficiency=0.5,
            node=ZONE,
        )
    )

    result = clear_market(config, orders, products)
    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    assert storage["charge"][timestamps[0]] == pytest.approx(100.0, abs=1e-4)
    assert storage["soc"][timestamps[0]] == pytest.approx(
        MTU_HOURS * 0.5 * storage["charge"][timestamps[0]], abs=1e-4
    )
    assert storage["soc"][timestamps[0]] == pytest.approx(12.5, abs=1e-4)
    assert storage["charge"][timestamps[1]] == pytest.approx(0.0, abs=1e-4)
    assert storage["discharge"][timestamps[2]] == pytest.approx(
        50.0, abs=1e-4
    )
    assert storage["soc"][timestamps[2]] == pytest.approx(0.0, abs=1e-4)

    _assert_price_pattern(result, timestamps)
    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


# ---------------------------------------------------------------------------
# Cycling cost and terminal value
# ---------------------------------------------------------------------------


@requires_solver
def test_high_cycling_cost_prevents_discharge():
    """
    Cycling cost above the achievable price spread.

    A cycling cost of 200 EUR/MWh exceeds the maximum market price of
    100 EUR/MWh. Discharging therefore destroys value, and charging without
    any later sale or terminal value is pointless. The storage stays idle
    even though discharging would be technically possible.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = _price_pattern_orders(timestamps)
    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            max_input_power=100.0,
            max_output_power=100.0,
            cycling_cost=200.0,
            node=ZONE,
        )
    )

    result = clear_market(config, orders, products)
    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    total_discharge = sum(storage["discharge"].values())
    total_charge = sum(storage["charge"].values())

    assert total_discharge == pytest.approx(0.0, abs=1e-4)
    assert total_charge == pytest.approx(0.0, abs=1e-4)
    assert storage["total_discharged_energy"] == pytest.approx(0.0, abs=1e-4)

    _assert_price_pattern(result, timestamps)
    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


@requires_solver
def test_high_terminal_value_retains_energy():
    """
    High end-of-day value.

    The storage starts full with 25 MWh and values stored energy at
    500 EUR/MWh at the end of the horizon. Selling the energy at the
    maximum market price of 100 EUR/MWh would yield 2500 EUR, while the
    terminal value of the retained energy is 12500 EUR. The storage
    therefore keeps its energy.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = _price_pattern_orders(timestamps)
    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            max_input_power=100.0,
            max_output_power=100.0,
            starting_energy=25.0,
            end_of_day_value=500.0,
            node=ZONE,
            efficiency=0.5,
        )
    )

    result = clear_market(config, orders, products)
    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    assert storage["soc"][timestamps[-1]] == pytest.approx(25.0, abs=1e-4)
    net_energy = sum(
        MTU_HOURS
        * (storage["discharge"][t] - storage["charge"][t])
        for t in timestamps
    )
    assert net_energy <= 1e-4
    assert storage["total_terminal_value_eur"] == pytest.approx(
        500.0 * 25.0, abs=1e-2
    )

    _assert_price_pattern(result, timestamps)
    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


# ---------------------------------------------------------------------------
# Storage with network coupling
# ---------------------------------------------------------------------------


@requires_solver
def test_storage_with_ntc_coupling():
    """
    Storage behind an NTC border.

    Zone ZA holds the oversized supply offers with the 10/60/100 EUR/MWh
    price pattern. Zone ZB holds only inelastic demand and a lossless
    25 MWh / 100 MW battery, and is connected to ZA by a generous
    1000 MW NTC. The battery charges in the cheapest MTU, which increases
    the import, and discharges in the most expensive MTU, which reduces it.

    The test verifies storage feasibility, zonal balance including the
    commercial exchange, and NTC capacity bounds together.
    """
    products = make_products(3)
    timestamps = product_timestamps(products)

    ntc_data = make_ntc_dataframe(
        timestamps,
        [
            {
                "line_id": "ntc_ab",
                "from_zone": ZONE_A,
                "to_zone": ZONE_B,
                "capacity_up": 1000.0,
                "capacity_down": 1000.0,
            }
        ],
    )

    config = make_market_config(
        products,
        param_dict={"ntc_connections": ntc_data, "log_flows": True},
    )

    orders = []

    for index, timestamp in enumerate(timestamps):
        # Supply and a small local demand in the cheap zone.
        orders.append(
            make_simple_bid(
                f"sup_a_{index}",
                timestamp,
                600.0,
                SUPPLY_PRICES[index],
                node=ZONE_A,
            )
        )
        orders.append(
            make_simple_bid(
                f"dem_a_{index}", timestamp, -50.0, 3000.0, node=ZONE_A
            )
        )
        # Inelastic demand in the storage zone.
        orders.append(
            make_simple_bid(
                f"dem_b_{index}", timestamp, -100.0, 3000.0, node=ZONE_B
            )
        )

    orders.append(
        make_storage_order(
            "storage",
            timestamps,
            max_energy=25.0,
            max_input_power=100.0,
            max_output_power=100.0,
            node=ZONE_B,
        )
    )

    result = clear_market(config, orders, products)

    storage = assert_storage_schedule_feasible(result, "storage", timestamps)

    # Charging in the cheap MTU and discharging in the expensive MTU.
    assert storage["charge"][timestamps[0]] == pytest.approx(100.0, abs=1e-3)
    assert storage["discharge"][timestamps[2]] == pytest.approx(
        100.0, abs=1e-3
    )

    # Prices are pinned in ZA and coupled to ZB by the unconstrained NTC.
    _assert_price_pattern(result, timestamps, zone=ZONE_A)
    for index, timestamp in enumerate(timestamps):
        assert get_price(result["meta"], ZONE_B, timestamp) == pytest.approx(
            SUPPLY_PRICES[index], abs=PRICE_TOL
        )

    # Import into ZB equals its demand plus charging minus discharging.
    for index, timestamp in enumerate(timestamps):
        flow = get_ntc_flow(result["flows"], timestamp, "ntc_ab")
        expected_import = (
            100.0
            + storage["charge"][timestamp]
            - storage["discharge"][timestamp]
        )
        assert flow == pytest.approx(expected_import, abs=1e-3)

    assert_ntc_capacity_bounds(result, ntc_data, timestamps)
    assert_ntc_balance(result, ntc_data, [ZONE_A, ZONE_B], timestamps)
    assert_simple_order_price_consistency(result)

    # The storage order is reported as an accepted order with a dispatch
    # profile rather than an acceptance ratio.
    assert find_order(result, "storage")["bid_type"] == "SO"
    assert set(storage["accepted_volume"]) == set(timestamps)
    assert sum(
        abs(volume) for volume in storage["accepted_volume"].values()
    ) >= EPS
    for timestamp in timestamps:
        assert storage["accepted_volume"][timestamp] == pytest.approx(
            storage["discharge"][timestamp] - storage["charge"][timestamp],
            abs=1e-6,
        )