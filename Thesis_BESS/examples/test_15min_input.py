# create_simple_15min_test_orderbook.py

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import logging 

import pandas as pd
import pypsa

from Thesis_BESS.tools.clearing_test_tool import ClearingTestTool

MTU_DURATION = timedelta(minutes=15)
MTU_DURATION_HOURS = 0.25

DEMAND_PRICE = 3000.0
SUPPLY_PRICES = {1: 50.0,2: 60.0,3: 40.0,}
SUPPLY_MARGIN = 1.02

EXPECTED_ZONES = {1, 2, 3}


def resolve_zone_column(network: pypsa.Network) -> str:
    """Return the bidding-zone column stored in the supplied network."""
    for column in ("zone", "zone_id", "Zone"):
        if column in network.buses.columns:
            return column

    raise ValueError(
        "The network must contain one of the zone columns "
        "'zone', 'zone_id', or 'Zone'."
    )


def get_delivery_snapshots(
    network: pypsa.Network,
) -> pd.DatetimeIndex:
    """
    Select the first full delivery day consisting of exactly 96 15-minute MTUs.
    """
    all_snapshots = pd.DatetimeIndex(network.snapshots)

    if all_snapshots.empty:
        raise ValueError("The supplied PyPSA network has no snapshots.")

    if all_snapshots.has_duplicates:
        raise ValueError("The supplied PyPSA network has duplicate snapshots.")

    first_day = all_snapshots[0].normalize()

    snapshots = pd.date_range(
        start=first_day,
        periods=96,
        freq="15min",
        tz=first_day.tz,
    )

    missing_snapshots = snapshots.difference(all_snapshots)

    if not missing_snapshots.empty:
        raise ValueError(
            "The network does not contain a complete 15-minute delivery day. "
            f"Missing snapshots: {list(missing_snapshots)}"
        )

    differences = snapshots[1:] - snapshots[:-1]

    if not (differences == pd.Timedelta(minutes=15)).all():
        raise ValueError(
            "The selected network snapshots are not consecutive "
            "15-minute intervals."
        )

    return snapshots


def get_zonal_loads(
    network: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    zone_column: str,
) -> pd.DataFrame:
    """
    Aggregate PyPSA load time series to zonal MW demand.

    The resulting DataFrame has:
    - index: 96 15-minute timestamps;
    - columns: zones 1, 2, and 3;
    - values: zonal load in MW.
    """
    if network.loads.empty:
        raise ValueError("The supplied network has no loads.")

    if network.loads_t.p_set.empty:
        raise ValueError(
            "The supplied network has no time-dependent loads_t.p_set data."
        )

    load_profile = network.loads_t.p_set.reindex(
        index=snapshots,
        columns=network.loads.index,
        fill_value=0.0,
    ).fillna(0.0)

    load_zones = network.loads["bus"].map(network.buses[zone_column])

    if load_zones.isna().any():
        invalid_loads = load_zones[load_zones.isna()].index.tolist()
        raise ValueError(
            "Some loads are connected to buses without a zone assignment: "
            f"{invalid_loads}"
        )

    network_zones = set(network.buses[zone_column].unique())

    if network_zones != EXPECTED_ZONES:
        raise ValueError(
            "Expected network zones {1, 2, 3}, but found "
            f"{sorted(network_zones)}."
        )

    zonal_loads = pd.DataFrame(
        index=snapshots,
        columns=sorted(EXPECTED_ZONES),
        dtype=float,
    )

    for zone in sorted(EXPECTED_ZONES):
        zone_load_ids = load_zones.index[load_zones == zone]

        zonal_loads[zone] = load_profile.reindex(
            columns=zone_load_ids,
            fill_value=0.0,
        ).sum(axis=1)

    if (zonal_loads < 0.0).any().any():
        raise ValueError("Zonal loads must be non-negative.")

    return zonal_loads

def assign_15_minute_snapshots(
    network: pypsa.Network,
    delivery_date: str = "2026-01-01",
) -> pd.DatetimeIndex:
    """
    Replace positional PyPSA snapshots (0, 1, 2, ...) with fixed
    15-minute timestamps while preserving the existing load profile.

    The supplied network was created with positional integer snapshots.
    Its loads_t.p_set values are preserved by position.
    """
    old_snapshots = pd.Index(network.snapshots)

    if len(old_snapshots) < 96:
        raise ValueError(
            "The network must contain at least 96 snapshot rows for one "
            f"15-minute delivery day, but contains only {len(old_snapshots)}."
        )

    if len(old_snapshots) != 96:
        raise ValueError(
            "This simple one-day test expects exactly 96 network snapshots, "
            f"but found {len(old_snapshots)}."
        )

    new_snapshots = pd.date_range(
        start=pd.Timestamp(delivery_date).normalize(),
        periods=96,
        freq="15min",
    )

    # Preserve time-dependent load values before replacing snapshots.
    old_load_profile = network.loads_t.p_set.copy()

    if len(old_load_profile.index) != 96:
        raise ValueError(
            "loads_t.p_set must contain exactly 96 rows, but contains "
            f"{len(old_load_profile.index)}."
        )

    # The existing data uses positional snapshots. Replace its index by
    # position before assigning it back to the retimed network.
    new_load_profile = old_load_profile.copy()
    new_load_profile.index = new_snapshots

    # set_snapshots changes all dynamic component indices. Therefore assign
    # the preserved load profile after changing the network snapshot index.
    network.set_snapshots(new_snapshots)
    network.loads_t.p_set = new_load_profile

    # The day-ahead model is fixed at 15-minute MTUs.
    network.snapshot_weightings.loc[
        new_snapshots,
        "objective",
    ] = 0.25

    return new_snapshots

def build_simple_test_orderbook(
    snapshots: pd.DatetimeIndex,
    zonal_loads: pd.DataFrame,
) -> list[dict]:
    """
    Create one demand bid and one supply bid per zone and 15-minute MTU.

    Demand:
        -volume = zonal load in MW
        price   = 3000 EUR/MWh

    Supply:
        volume = 1.02 * zonal load in MW
        price  = 50 EUR/MWh

    The supply margin allows supply acceptance to be partial. Since every
    zone has local supply, cleared zonal supply and demand should both be
    close to the corresponding zonal network load.
    """
    orderbook = []

    for interval, timestamp in enumerate(snapshots, start=1):
        end_time = timestamp + MTU_DURATION

        for zone in sorted(EXPECTED_ZONES):
            load_mw = float(zonal_loads.at[timestamp, zone])

            if load_mw <= 0.0:
                continue

            supply_mw = load_mw * SUPPLY_MARGIN
            supply_price = SUPPLY_PRICES[zone]

            orderbook.append(
                {
                    "bid_id": f"DEMAND_Z{zone}_MTU{interval:03d}",
                    "agent_addr": (f"demand_zone_{zone}", "test"),
                    "bid_type": "SB",
                    "price": DEMAND_PRICE,
                    "volume": -load_mw,
                    "node": zone,
                    "start_time": timestamp.to_pydatetime(),
                    "end_time": end_time.to_pydatetime(),
                    "only_hours": None,
                    "min_acceptance_ratio": None,
                    "parent_bid_id": None,
                    "exclusive_group_id": None,
                }
            )

            orderbook.append(
                {
                    "bid_id": f"SUPPLY_Z{zone}_MTU{interval:03d}",
                    "agent_addr": (f"supply_zone_{zone}", "test"),
                    "bid_type": "SB",
                    "price": supply_price,
                    "volume": supply_mw,
                    "node": zone,
                    "start_time": timestamp.to_pydatetime(),
                    "end_time": end_time.to_pydatetime(),
                    "only_hours": None,
                    "min_acceptance_ratio": None,
                    "parent_bid_id": None,
                    "exclusive_group_id": None,
                }
            )

    expected_order_count = 96 * 3 * 2

    if len(orderbook) != expected_order_count:
        raise AssertionError(
            f"Expected {expected_order_count} orders, got {len(orderbook)}. "
            "Every zone must have positive load in every MTU for this test."
        )

    return orderbook


def build_market_products(
    snapshots: pd.DatetimeIndex,
) -> list[tuple]:
    """Create exactly 96 fixed 15-minute market products."""
    products = [
        (
            timestamp.to_pydatetime(),
            (timestamp + MTU_DURATION).to_pydatetime(),
            None,
        )
        for timestamp in snapshots
    ]

    assert len(products) == 96
    assert all(
        end_time - start_time == MTU_DURATION
        for start_time, end_time, _ in products
    )

    return products


def verify_result_against_load(
    market_meta: list[dict],
    zonal_loads: pd.DataFrame,
) -> None:
    """
    Verify 15-minute clearing results for an FBMC-coupled three-zone system.

    Demand must equal the submitted zonal load because local supply is
    available in every zone. Supply may differ from the local load because
    FBMC permits interzonal commercial exchanges.

    All physical quantities remain MW. Reported energy must equal MW * 0.25 h.
    """
    result = pd.DataFrame(market_meta).copy()

    result["product_start"] = pd.to_datetime(result["product_start"])
    result["zone"] = result["zone"].astype(int)

    expected = (
        zonal_loads.rename_axis("product_start")
        .stack()
        .rename("expected_load_mw")
        .reset_index()
        .rename(columns={"level_1": "zone"})
    )

    expected["product_start"] = pd.to_datetime(
        expected["product_start"]
    )
    expected["zone"] = expected["zone"].astype(int)

    comparison = expected.merge(
        result,
        on=["product_start", "zone"],
        how="left",
        validate="one_to_one",
    )

    if comparison["demand_volume"].isna().any():
        missing = comparison.loc[
            comparison["demand_volume"].isna(),
            ["product_start", "zone"],
        ]
        raise AssertionError(
            "Missing market result rows:\n"
            f"{missing.to_string(index=False)}"
        )

    tolerance_mw = 1e-3
    tolerance_mwh = 1e-3

    # Every submitted high-price demand bid should be fully accepted.
    demand_error = (
        comparison["demand_volume"]
        - comparison["expected_load_mw"]
    ).abs()

    assert (demand_error < tolerance_mw).all(), (
        "Cleared zonal demand does not match the expected zonal load:\n"
        f"{comparison.loc[demand_error >= tolerance_mw, [
            'product_start',
            'zone',
            'expected_load_mw',
            'demand_volume',
            'supply_volume',
        ]].to_string(index=False)}"
    )

    # Energy reporting uses the fixed 15-minute conversion factor.
    demand_energy_error = (
        comparison["demand_volume_energy"]
        - comparison["demand_volume"] * MTU_DURATION_HOURS
    ).abs()

    supply_energy_error = (
        comparison["supply_volume_energy"]
        - comparison["supply_volume"] * MTU_DURATION_HOURS
    ).abs()

    assert (demand_energy_error < tolerance_mwh).all(), (
        "Demand energy is not equal to demand MW * 0.25 h."
    )

    assert (supply_energy_error < tolerance_mwh).all(), (
        "Supply energy is not equal to supply MW * 0.25 h."
    )

    # FBMC allows zonal net positions, but the coupled market must remain
    # balanced at every 15-minute timestamp.
    system_balance = (
        result.groupby("product_start", as_index=False)[
            ["supply_volume", "demand_volume"]
        ]
        .sum()
    )

    system_balance["imbalance_mw"] = (
        system_balance["supply_volume"]
        - system_balance["demand_volume"]
    )

    assert (
        system_balance["imbalance_mw"].abs() < tolerance_mw
    ).all(), (
        "System-wide market balance failed:\n"
        f"{system_balance.loc[
            system_balance['imbalance_mw'].abs() >= tolerance_mw
        ].to_string(index=False)}"
    )

    # This is informational only. Under FBMC, individual zones may export
    # or import and are therefore not required to have zero net position.
    zonal_net_positions = result[
        [
            "product_start",
            "zone",
            "supply_volume",
            "demand_volume",
        ]
    ].copy()

    zonal_net_positions["net_position_mw"] = (
        zonal_net_positions["supply_volume"]
        - zonal_net_positions["demand_volume"]
    )

    print("\nFBMC zonal net-position summary in MW:")
    print(
        zonal_net_positions.groupby("zone")["net_position_mw"]
        .agg(["min", "mean", "max"])
        .round(3)
        .to_string()
    )

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    network_path = Path("S:/23502/5/500_ESA/ESW_51520/Aktuell/80_Masterarbeiten/Schiebel_Simon/03_data/test_network/filtered_network.nc")
    network = pypsa.Network(network_path)

    # Smoke-test simplification: avoid physical infeasibility caused by
    # minimum-output restrictions in the source generation fleet.
    network.generators["p_min_pu"] = 0.0
    network.sanitize()

    # The existing NetCDF network uses integer snapshots. Convert them to a
    # real 15-minute delivery day before creating products or running FBMC.
    snapshots = assign_15_minute_snapshots(
        network=network,
        delivery_date="2026-01-01",
    )

    zone_column = resolve_zone_column(network)
    snapshots = get_delivery_snapshots(network)
    zonal_loads = get_zonal_loads(
        network=network,
        snapshots=snapshots,
        zone_column=zone_column,
    )

    orderbook = build_simple_test_orderbook(
        snapshots=snapshots,
        zonal_loads=zonal_loads,
    )
    market_products = build_market_products(snapshots)

    print("Test orderbook created")
    print(f"Delivery date: {snapshots[0].date()}")
    print(f"MTUs: {len(market_products)}")
    print(f"Orders: {len(orderbook)}")
    print("\nZonal load summary in MW:")
    print(zonal_loads.describe().T[["min", "mean", "max"]])

    tool = ClearingTestTool(
        market_id="three_zone_15min_test",
        network=network,
        flowbased_region=[1,2,3],
        zones_identifier=zone_column,
        param_dict={
            "solver_name": "gurobi",
            "fbmc_solver_name": "gurobi",
            "log_flows": True,
            "pricing_mechanism": "pay_as_clear",
            "log_performance": True,
        },
    )

    # The test tool normally creates its orderbook from CSV input. Here the
    # generated orderbook is assigned directly to preserve numeric zone IDs.
    tool.delivery_date = snapshots[0].normalize().to_pydatetime()
    tool.orderbook = orderbook
    tool.market_products = market_products
    tool.submitted_order_count = len(orderbook)

    #tool._validate_network_market_product_snapshots()

    results = tool.run_clearing()
    tool.export_results("C:/assume/assume/testing_tools/results/15min_example")

    assert len(tool.market_products) == 96
    assert network.snapshots[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert network.snapshots[1] == pd.Timestamp("2026-01-01 00:15:00")
    assert network.snapshots[-1] == pd.Timestamp("2026-01-01 23:45:00")
    assert network.loads_t.p_set.index.equals(network.snapshots)
    assert all(
        product[1] - product[0] == MTU_DURATION
        for product in tool.market_products
    )

    verify_result_against_load(
        market_meta=results["market_meta"],
        zonal_loads=zonal_loads,
    )

    tool.print_summary()

    print(
        "Success: 96 fixed 15-minute MTUs cleared and the reported "
        "MW/MWh volumes match the zonal PyPSA loads."
    )


if __name__ == "__main__":
    main()