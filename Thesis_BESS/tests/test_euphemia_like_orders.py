# test_euphemia_like_orders.py
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Validation of simple-bid and block-bid logic without network coupling.

All scenarios use one bidding zone, fixed 15-minute products and
economically unambiguous expected outcomes.
"""

from __future__ import annotations

import pytest

from .test_euphemia_like_tools import (
    EPS,
    PRICE_TOL,
    accepted_volume,
    acceptance_ratio,
    assert_block_surplus_conditions,
    assert_isolated_zone_balances,
    assert_simple_order_price_consistency,
    block_surplus_eur,
    clear_market,
    get_price,
    is_accepted,
    make_block_bid,
    make_linked_bid,
    make_market_config,
    make_products,
    make_simple_bid,
    product_timestamps,
    requires_solver,
)

ZONE = "Z1"


# ---------------------------------------------------------------------------
# Simple orders
# ---------------------------------------------------------------------------


@requires_solver
def test_simple_merit_order_clearing():
    """
    One zone, one MTU, pure merit order.

    Demand of 100 MW at 3000 EUR/MWh is served by 60 MW at 20 EUR/MWh and
    by 40 MW of the 100 MW offer at 50 EUR/MWh. The partially accepted
    offer is marginal, so the clearing price is 50 EUR/MWh.
    """
    products = make_products(1)
    timestamp = product_timestamps(products)[0]
    config = make_market_config(products)

    orders = [
        make_simple_bid("dem", timestamp, -100.0, 3000.0, node=ZONE),
        make_simple_bid("sup_cheap", timestamp, 60.0, 20.0, node=ZONE),
        make_simple_bid("sup_marginal", timestamp, 100.0, 50.0, node=ZONE),
    ]

    result = clear_market(config, orders, products)

    assert accepted_volume(result, "dem") == pytest.approx(-100.0, abs=EPS)
    assert accepted_volume(result, "sup_cheap") == pytest.approx(
        60.0, abs=EPS
    )
    assert accepted_volume(result, "sup_marginal") == pytest.approx(
        40.0, abs=EPS
    )
    assert get_price(result["meta"], ZONE, timestamp) == pytest.approx(
        50.0, abs=PRICE_TOL
    )

    assert_isolated_zone_balances(result, [ZONE], [timestamp])
    assert_simple_order_price_consistency(result)


@requires_solver
def test_marginal_simple_bid_fixes_price():
    """
    A partially accepted simple bid is at-the-money.

    Demand of 80 MW meets a single 100 MW offer at 37.5 EUR/MWh. The offer
    is partially accepted, which fixes the market clearing price exactly to
    its bid price.
    """
    products = make_products(1)
    timestamp = product_timestamps(products)[0]
    config = make_market_config(products)

    orders = [
        make_simple_bid("dem", timestamp, -80.0, 3000.0, node=ZONE),
        make_simple_bid("sup", timestamp, 100.0, 37.5, node=ZONE),
    ]

    result = clear_market(config, orders, products)

    assert accepted_volume(result, "sup") == pytest.approx(80.0, abs=EPS)
    assert get_price(result["meta"], ZONE, timestamp) == pytest.approx(
        37.5, abs=PRICE_TOL
    )

    assert_isolated_zone_balances(result, [ZONE], [timestamp])
    assert_simple_order_price_consistency(result)


# ---------------------------------------------------------------------------
# Block orders
# ---------------------------------------------------------------------------


def _two_mtu_block_scenario(block_price: float):
    """
    Two MTUs with marginal prices 41 and 50 EUR/MWh.

    Demand is 1000 MW per MTU. The marginal offers are oversized and thus
    partially accepted, which pins both clearing prices irrespective of
    block acceptance.
    """
    products = make_products(2)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    marginal_prices = {timestamps[0]: 41.0, timestamps[1]: 50.0}

    orders = []

    for timestamp in timestamps:
        orders.append(
            make_simple_bid(
                f"dem_{timestamp:%H%M}", timestamp, -1000.0, 3000.0, node=ZONE
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_marginal_{timestamp:%H%M}",
                timestamp,
                1100.0,
                marginal_prices[timestamp],
                node=ZONE,
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_backup_{timestamp:%H%M}",
                timestamp,
                1200.0,
                100.0,
                node=ZONE,
            )
        )

    orders.append(
        make_block_bid(
            "block",
            {timestamps[0]: 100.0, timestamps[1]: 100.0},
            price=block_price,
            node=ZONE,
            min_acceptance_ratio=1.0,
        )
    )

    return config, orders, products, timestamps


@requires_solver
def test_accepted_versus_rejected_block_bid():
    """
    Price-feasibility logic for a multi-MTU block.

    With a bid price of 45 EUR/MWh the block surplus is
    ((41-45) + (50-45)) * 100 MW * 0.25 h = +25 EUR, so the block is
    accepted. With a bid price of 48 EUR/MWh the surplus is
    ((41-48) + (50-48)) * 100 MW * 0.25 h = -125 EUR, so the block cannot
    be part of a price-feasible solution and is rejected.
    """
    # In-the-money block.
    config, orders, products, timestamps = _two_mtu_block_scenario(45.0)
    result = clear_market(config, orders, products)

    assert get_price(result["meta"], ZONE, timestamps[0]) == pytest.approx(
        41.0, abs=PRICE_TOL
    )
    assert get_price(result["meta"], ZONE, timestamps[1]) == pytest.approx(
        50.0, abs=PRICE_TOL
    )
    assert is_accepted(result, "block")

    for timestamp in timestamps:
        assert accepted_volume(result, "block", timestamp) == pytest.approx(
            100.0, abs=EPS
        )

    assert block_surplus_eur(result, "block") >= -1e-3
    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)
    assert_block_surplus_conditions(result)

    # Out-of-the-money block.
    config, orders, products, timestamps = _two_mtu_block_scenario(48.0)
    result = clear_market(config, orders, products)

    assert acceptance_ratio(result, "block") == pytest.approx(0.0, abs=EPS)

    for timestamp in timestamps:
        assert accepted_volume(result, "block", timestamp) == pytest.approx(
            0.0, abs=EPS
        )

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


@requires_solver
def test_minimum_acceptance_ratio_is_respected():
    """
    Minimum acceptance ratio of a partially attractive block.

    Welfare alone would accept only half of the 200 MW block, because the
    cheapest 900 MW offer at 10 EUR/MWh is cheaper than the block at
    20 EUR/MWh. The configured minimum acceptance ratio of 0.6 therefore
    forces the block to be either rejected or accepted at a ratio of at
    least 0.6. Any accepted ratio must additionally be supported by
    non-negative surplus at the resulting prices.
    """
    products = make_products(2)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = []

    for timestamp in timestamps:
        orders.append(
            make_simple_bid(
                f"dem_{timestamp:%H%M}", timestamp, -1000.0, 3000.0, node=ZONE
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_cheap_{timestamp:%H%M}", timestamp, 900.0, 10.0, node=ZONE
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_peak_{timestamp:%H%M}", timestamp, 1200.0, 50.0, node=ZONE
            )
        )

    orders.append(
        make_block_bid(
            "block_mar",
            {timestamp: 200.0 for timestamp in timestamps},
            price=20.0,
            node=ZONE,
            min_acceptance_ratio=0.6,
        )
    )

    result = clear_market(config, orders, products)

    ratio = acceptance_ratio(result, "block_mar")

    assert ratio <= EPS or ratio >= 0.6 - 1e-4, (
        f"Block acceptance ratio {ratio} violates the minimum acceptance "
        "ratio of 0.6."
    )

    if ratio > EPS:
        assert block_surplus_eur(result, "block_mar") >= -1e-3

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)
    assert_block_surplus_conditions(result)


# ---------------------------------------------------------------------------
# Linked blocks
# ---------------------------------------------------------------------------


def _linked_block_scenario(parent_price: float, child_price: float):
    """
    Two MTUs with a clearing price of 100 EUR/MWh.

    The 1200 MW offer at 100 EUR/MWh is partially accepted and therefore
    pins the price in both MTUs.
    """
    products = make_products(2)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = []

    for timestamp in timestamps:
        orders.append(
            make_simple_bid(
                f"dem_{timestamp:%H%M}", timestamp, -1000.0, 3000.0, node=ZONE
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_{timestamp:%H%M}", timestamp, 1200.0, 100.0, node=ZONE
            )
        )

    orders.append(
        make_block_bid(
            "parent",
            {timestamp: 100.0 for timestamp in timestamps},
            price=parent_price,
            node=ZONE,
            min_acceptance_ratio=1.0,
        )
    )
    orders.append(
        make_linked_bid(
            "child",
            {timestamp: 100.0 for timestamp in timestamps},
            price=child_price,
            parent_bid_id="parent",
            node=ZONE,
            min_acceptance_ratio=1.0,
        )
    )

    return config, orders, products, timestamps


@requires_solver
def test_linked_block_requires_accepted_parent():
    """
    A child linked block cannot be accepted without its parent.

    The parent bids 500 EUR/MWh against a clearing price of 100 EUR/MWh.
    Its loss of (100-500) * 100 MW * 0.25 h per MTU exceeds the gain of the
    cheap child, so the parent is rejected. The linking constraint then
    forces the child to be rejected as well.
    """
    config, orders, products, timestamps = _linked_block_scenario(
        parent_price=500.0, child_price=10.0
    )
    result = clear_market(config, orders, products)

    assert acceptance_ratio(result, "parent") == pytest.approx(0.0, abs=EPS)
    assert acceptance_ratio(result, "child") == pytest.approx(0.0, abs=EPS)

    for timestamp in timestamps:
        assert get_price(result["meta"], ZONE, timestamp) == pytest.approx(
            100.0, abs=PRICE_TOL
        )

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)


@requires_solver
def test_accepted_parent_with_rejected_child():
    """
    An accepted parent may coexist with a rejected child.

    The parent at 10 EUR/MWh is deeply in the money at a price of
    100 EUR/MWh, while the child at 500 EUR/MWh is not. The child is
    therefore rejected, and the remaining accepted family surplus stays
    non-negative.
    """
    config, orders, products, timestamps = _linked_block_scenario(
        parent_price=10.0, child_price=500.0
    )
    result = clear_market(config, orders, products)

    assert acceptance_ratio(result, "parent") == pytest.approx(1.0, abs=1e-4)
    assert acceptance_ratio(result, "child") == pytest.approx(0.0, abs=EPS)

    for timestamp in timestamps:
        assert accepted_volume(result, "parent", timestamp) == pytest.approx(
            100.0, abs=EPS
        )
        assert accepted_volume(result, "child", timestamp) == pytest.approx(
            0.0, abs=EPS
        )

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)
    assert_block_surplus_conditions(result)


# ---------------------------------------------------------------------------
# Exclusive groups
# ---------------------------------------------------------------------------


@requires_solver
def test_exclusive_block_group_accepts_at_most_one_block():
    """
    Exclusive group with two fully-or-nothing blocks.

    Both blocks are in the money at a clearing price of 100 EUR/MWh, but
    they belong to the same exclusive group. Only the cheaper block at
    50 EUR/MWh is accepted; the 60 EUR/MWh alternative is rejected.
    """
    products = make_products(2)
    timestamps = product_timestamps(products)
    config = make_market_config(products)

    orders = []

    for timestamp in timestamps:
        orders.append(
            make_simple_bid(
                f"dem_{timestamp:%H%M}", timestamp, -1000.0, 3000.0, node=ZONE
            )
        )
        orders.append(
            make_simple_bid(
                f"sup_{timestamp:%H%M}", timestamp, 1200.0, 100.0, node=ZONE
            )
        )

    for bid_id, price in (("block_a", 50.0), ("block_b", 60.0)):
        orders.append(
            make_block_bid(
                bid_id,
                {timestamp: 100.0 for timestamp in timestamps},
                price=price,
                node=ZONE,
                min_acceptance_ratio=1.0,
                exclusive_group_id="group_1",
            )
        )

    result = clear_market(config, orders, products)

    ratio_a = acceptance_ratio(result, "block_a")
    ratio_b = acceptance_ratio(result, "block_b")

    # The exclusive-group constraint limits the sum of acceptance ratios.
    assert ratio_a + ratio_b <= 1.0 + 1e-4
    assert sum(ratio > EPS for ratio in (ratio_a, ratio_b)) <= 1

    # Welfare maximisation selects the cheaper of the two alternatives.
    assert ratio_a == pytest.approx(1.0, abs=1e-4)
    assert ratio_b == pytest.approx(0.0, abs=EPS)

    assert_isolated_zone_balances(result, [ZONE], timestamps)
    assert_simple_order_price_consistency(result)
    assert_block_surplus_conditions(result)