# test_euphemia_like_coupling.py
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Validation of NTC and flow-based market coupling.

NTC scenarios use two zones with in-memory NTC capacity data. FBMC
scenarios use a minimal deterministic two-bus/two-zone PyPSA network with
one bus per bidding zone. With that topology the GSK is an identity
mapping, the D-2 reference flow explained by FB net positions is exact and
therefore:

    RAM = (1 - FRM) * s_nom = 0.9 * s_nom

for both directional CNECs of the single cross-border line. The feasible
commercial exchange inside the FB region equals this RAM.
"""

from __future__ import annotations

import pytest

from .test_euphemia_like_tools import (
    EPS,
    PRICE_TOL,
    accepted_volume,
    assert_fbmc_ram_feasible,
    assert_ntc_balance,
    assert_ntc_capacity_bounds,
    assert_simple_order_price_consistency,
    clear_market,
    fbmc_cnec_utilisation,
    get_fb_net_export,
    get_ntc_flow,
    get_price,
    make_market_config,
    make_ntc_dataframe,
    make_products,
    make_simple_bid,
    make_two_zone_network,
    precompute_fbmc_clearing_input,
    product_timestamps,
    requires_solver,
)

ZONE_A = "ZA"
ZONE_B = "ZB"
LINE = "ntc_ab"


# ---------------------------------------------------------------------------
# NTC helpers
# ---------------------------------------------------------------------------


def _ntc_orders(timestamp, reverse: bool = False) -> list[dict]:
    """
    Two-zone merit order with a clear export incentive.

    Default direction: the cheap 20 EUR/MWh zone ZA supplies its own 100 MW
    of demand and exports to zone ZB, where 300 MW of demand would
    otherwise be served at 80 EUR/MWh. Both offers are oversized, so any
    partially accepted offer pins its zonal price.
    """
    cheap_zone, expensive_zone = (
        (ZONE_B, ZONE_A) if reverse else (ZONE_A, ZONE_B)
    )

    return [
        make_simple_bid("dem_cheap", timestamp, -100.0, 3000.0, node=cheap_zone),
        make_simple_bid("sup_cheap", timestamp, 800.0, 20.0, node=cheap_zone),
        make_simple_bid(
            "dem_expensive", timestamp, -300.0, 3000.0, node=expensive_zone
        ),
        make_simple_bid(
            "sup_expensive", timestamp, 500.0, 80.0, node=expensive_zone
        ),
    ]


def _run_ntc_case(
    capacity_up: float,
    capacity_down: float,
    reverse: bool = False,
):
    """Clear a one-MTU, two-zone NTC scenario."""
    products = make_products(1)
    timestamp = product_timestamps(products)[0]

    ntc_data = make_ntc_dataframe(
        [timestamp],
        [
            {
                "line_id": LINE,
                "from_zone": ZONE_A,
                "to_zone": ZONE_B,
                "capacity_up": capacity_up,
                "capacity_down": capacity_down,
            }
        ],
    )

    config = make_market_config(
        products,
        param_dict={"ntc_connections": ntc_data, "log_flows": True},
    )
    orders = _ntc_orders(timestamp, reverse=reverse)

    result = clear_market(config, orders, products)

    # Independent feasibility: capacity bounds and zonal balance including
    # commercial exchange.
    assert_ntc_capacity_bounds(result, ntc_data, [timestamp])
    assert_ntc_balance(result, ntc_data, [ZONE_A, ZONE_B], [timestamp])
    assert_simple_order_price_consistency(result)

    return result, ntc_data, timestamp


# ---------------------------------------------------------------------------
# NTC tests
# ---------------------------------------------------------------------------


@requires_solver
def test_unconstrained_ntc_couples_prices():
    """
    Unconstrained NTC exchange.

    With 1000 MW of capacity the economically desired export of 300 MW is
    unconstrained. The cheap zone covers all demand, the expensive offer is
    rejected and both zonal prices converge to 20 EUR/MWh.
    """
    result, _, timestamp = _run_ntc_case(1000.0, 1000.0)

    flow = get_ntc_flow(result["flows"], timestamp, LINE)

    assert flow == pytest.approx(300.0, abs=1e-4)
    assert accepted_volume(result, "sup_cheap") == pytest.approx(
        400.0, abs=1e-4
    )
    assert accepted_volume(result, "sup_expensive") == pytest.approx(
        0.0, abs=EPS
    )

    price_a = get_price(result["meta"], ZONE_A, timestamp)
    price_b = get_price(result["meta"], ZONE_B, timestamp)

    assert price_a == pytest.approx(20.0, abs=PRICE_TOL)
    assert price_b == pytest.approx(price_a, abs=PRICE_TOL)


@requires_solver
def test_binding_ntc_export_limit_separates_prices():
    """
    Binding export limit.

    The export capability is limited to 90 MW. The cheap zone remains
    marginal at 20 EUR/MWh, while the importing zone must cover the
    remaining 210 MW at 80 EUR/MWh. The flow reaches the limit and the
    importing price is at least the exporting price.
    """
    result, _, timestamp = _run_ntc_case(90.0, 1000.0)

    flow = get_ntc_flow(result["flows"], timestamp, LINE)

    assert flow == pytest.approx(90.0, abs=1e-4)
    assert accepted_volume(result, "sup_cheap") == pytest.approx(
        190.0, abs=1e-4
    )
    assert accepted_volume(result, "sup_expensive") == pytest.approx(
        210.0, abs=1e-4
    )

    price_a = get_price(result["meta"], ZONE_A, timestamp)
    price_b = get_price(result["meta"], ZONE_B, timestamp)

    assert price_a == pytest.approx(20.0, abs=PRICE_TOL)
    assert price_b == pytest.approx(80.0, abs=PRICE_TOL)
    assert price_b >= price_a - PRICE_TOL


@requires_solver
def test_reverse_ntc_flow_reaches_capacity_down():
    """
    Reversed economic incentive.

    The cheap generation is now located in the ``to_zone`` of the line, so
    the commercial flow becomes negative. It reaches the configured
    ``capacity_down`` of 90 MW and the price ordering is reversed.
    """
    result, _, timestamp = _run_ntc_case(1000.0, 90.0, reverse=True)

    flow = get_ntc_flow(result["flows"], timestamp, LINE)

    assert flow == pytest.approx(-90.0, abs=1e-4)

    price_a = get_price(result["meta"], ZONE_A, timestamp)
    price_b = get_price(result["meta"], ZONE_B, timestamp)

    # Zone ZA is now the importing, expensive zone.
    assert price_a == pytest.approx(80.0, abs=PRICE_TOL)
    assert price_b == pytest.approx(20.0, abs=PRICE_TOL)
    assert price_a >= price_b - PRICE_TOL


@requires_solver
def test_zero_ntc_capacity_isolates_zones():
    """
    No commercial exchange in the economic direction.

    ``capacity_up`` is zero, so the desired export is impossible and both
    zones clear independently at 20 and 80 EUR/MWh with zero flow. The
    opposite direction remains open, which keeps the price conditions of
    the implementation well posed for a strictly one-sided zero capacity.
    """
    result, _, timestamp = _run_ntc_case(0.0, 0.0)

    flow = get_ntc_flow(result["flows"], timestamp, LINE)

    assert flow == pytest.approx(0.0, abs=1e-4)
    assert accepted_volume(result, "sup_cheap") == pytest.approx(
        100.0, abs=1e-4
    )
    assert accepted_volume(result, "sup_expensive") == pytest.approx(
        300.0, abs=1e-4
    )

    assert get_price(result["meta"], ZONE_A, timestamp) == pytest.approx(
        20.0, abs=PRICE_TOL
    )
    assert get_price(result["meta"], ZONE_B, timestamp) == pytest.approx(
        80.0, abs=PRICE_TOL
    )


# ---------------------------------------------------------------------------
# FBMC helpers
# ---------------------------------------------------------------------------


def _run_fbmc_case(s_nom: float):
    """
    Clear a one-MTU, two-zone flow-based scenario.

    The market orders are identical to the NTC scenarios so that the only
    difference is the flow-based representation of the cross-border
    capacity.
    """
    products = make_products(1)
    timestamp = product_timestamps(products)[0]

    network = make_two_zone_network([timestamp], s_nom=s_nom)

    config = make_market_config(
        products,
        param_dict={
            "network": network,
            "flowbased_region": [ZONE_A, ZONE_B],
            "zones_identifier": "zone_id",
            "log_flows": True,
        },
    )
    orders = _ntc_orders(timestamp)

    result = clear_market(config, orders, products)

    # Independently recompute the FB domain and verify RAM feasibility of
    # the reported FB net exports.
    clearing_input = precompute_fbmc_clearing_input(
        network, [ZONE_A, ZONE_B], [timestamp]
    )
    assert_fbmc_ram_feasible(clearing_input, result["flows"], [timestamp])
    assert_simple_order_price_consistency(result)

    return result, clearing_input, timestamp


# ---------------------------------------------------------------------------
# FBMC tests
# ---------------------------------------------------------------------------


@requires_solver
def test_slack_fbmc_constraints_converge_prices():
    """
    Slack flow-based domain.

    With a 1000 MW cross-border element the RAM of 900 MW does not
    constrain the desired 300 MW exchange. All CNECs stay slack, so every
    FB congestion price is forced to zero and all FB-zone prices converge.
    """
    result, clearing_input, timestamp = _run_fbmc_case(1000.0)

    net_export_a = get_fb_net_export(result["flows"], timestamp, ZONE_A)
    net_export_b = get_fb_net_export(result["flows"], timestamp, ZONE_B)

    assert net_export_a == pytest.approx(300.0, abs=1e-3)
    assert net_export_b == pytest.approx(-300.0, abs=1e-3)

    records = fbmc_cnec_utilisation(clearing_input, result["flows"], [timestamp])
    assert min(record["slack"] for record in records) > 1e-3

    price_a = get_price(result["meta"], ZONE_A, timestamp)
    price_b = get_price(result["meta"], ZONE_B, timestamp)

    assert price_a == pytest.approx(20.0, abs=PRICE_TOL)
    assert price_b == pytest.approx(price_a, abs=PRICE_TOL)


@requires_solver
def test_binding_fbmc_constraint_limits_exchange():
    """
    Binding flow-based constraint.

    A 100 MW cross-border element yields RAM = 90 MW. The PTDF-weighted FB
    net export reaches RAM within tolerance, the exchange is limited to
    90 MW and the zonal prices separate.
    """
    result, clearing_input, timestamp = _run_fbmc_case(100.0)

    net_export_a = get_fb_net_export(result["flows"], timestamp, ZONE_A)
    net_export_b = get_fb_net_export(result["flows"], timestamp, ZONE_B)

    assert net_export_a == pytest.approx(90.0, abs=1e-3)
    assert net_export_b == pytest.approx(-90.0, abs=1e-3)

    records = fbmc_cnec_utilisation(clearing_input, result["flows"], [timestamp])

    # RAM follows from the deterministic one-bus-per-zone topology.
    for record in records:
        assert record["ram"] == pytest.approx(90.0, abs=1e-3)

    # At least one CNEC is binding within tolerance.
    assert min(abs(record["slack"]) for record in records) <= 1e-3

    price_a = get_price(result["meta"], ZONE_A, timestamp)
    price_b = get_price(result["meta"], ZONE_B, timestamp)

    assert price_a == pytest.approx(20.0, abs=PRICE_TOL)
    assert price_b == pytest.approx(80.0, abs=PRICE_TOL)
    assert price_b >= price_a - PRICE_TOL


@requires_solver
def test_ram_sensitivity_reduces_exchange_and_increases_separation():
    """
    RAM sensitivity.

    Tightening the physical capacity of the critical network element
    reduces RAM proportionally. Compared with a relaxed-RAM case, the
    feasible commercial exchange shrinks and the price separation inside
    the FB region increases.
    """
    relaxed, relaxed_input, timestamp = _run_fbmc_case(1000.0)
    tight, tight_input, _ = _run_fbmc_case(100.0)

    relaxed_exchange = get_fb_net_export(relaxed["flows"], timestamp, ZONE_A)
    tight_exchange = get_fb_net_export(tight["flows"], timestamp, ZONE_A)

    relaxed_ram = min(
        record["ram"]
        for record in fbmc_cnec_utilisation(
            relaxed_input, relaxed["flows"], [timestamp]
        )
    )
    tight_ram = min(
        record["ram"]
        for record in fbmc_cnec_utilisation(
            tight_input, tight["flows"], [timestamp]
        )
    )

    assert tight_ram < relaxed_ram - 1e-3
    assert tight_exchange < relaxed_exchange - 1e-3

    relaxed_separation = abs(
        get_price(relaxed["meta"], ZONE_B, timestamp)
        - get_price(relaxed["meta"], ZONE_A, timestamp)
    )
    tight_separation = abs(
        get_price(tight["meta"], ZONE_B, timestamp)
        - get_price(tight["meta"], ZONE_A, timestamp)
    )

    assert tight_separation > relaxed_separation + PRICE_TOL