#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Minimal 3-zone FBMC showcase for the EUPHEMIA-like clearing (Storage Order focus).

Topology (one node per zone, ring, no NTC):

    N_A(ZoneA) --line_AB(400 MW)-- N_B(ZoneB)
           /                            /
      line_CA(150 MW)          line_BC(400 MW)
            /                        /
                  N_C(ZoneC)

* line_CA is the bottleneck -> it is congested for many consecutive MTUs
  (whole midday + evening peak), so zonal prices separate.
* Zone A: cheap, Zone B: medium, Zone C: expensive and import-constrained.
* One Storage Order (efficiency 0.86, cycling cost 10 EUR/MWh) in Zone C:
  charges during the cheap night MTUs, discharges during the evening peak.

Expected qualitative outcome (EUR/MWh in Zone C, MTUs 68..83 = 17:00-21:00):

| case          | night (00-06) | midday | evening peak |
|---------------|---------------|--------|--------------|
| without_SO    | 40            | 90     | 150          |
| with_SO       | 40            | 90     | 120          |

In addition, a demand block order in Zone C (130 EUR/MWh, hour 20) is
rejected without the Storage Order and accepted with it.

Inputs  -> ./test_data/
Results -> ./results/with_SO/ , ./results/without_SO/
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pypsa

from clearing_test_tool import ClearingTestTool

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("storage_order_showcase")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DELIVERY_DATE = datetime(2026, 1, 15)
MTU = timedelta(minutes=15)
N_MTU = 96

TEST_DATA_DIR = Path("assume/testing_tools/test_data")
RESULTS_DIR = Path("assume/testing_tools/results")

SOLVER = "gurobi"          # Pyomo solver for the clearing
FBMC_SOLVER = "gurobi"     # linopy solver for the FBMC D-2 base case

ZONES = ["ZoneA", "ZoneB", "ZoneC"]
BUSES = {"ZoneA": "N_A", "ZoneB": "N_B", "ZoneC": "N_C"}
LINES = {
    "line_AB": ("N_A", "N_B", 400.0),
    "line_BC": ("N_B", "N_C", 400.0),
    "line_CA": ("N_C", "N_A", 150.0),   # bottleneck
}

# (capacity MW, marginal cost EUR/MWh) per zone
SUPPLY_TIERS = {
    "ZoneA": [(400.0, 20.0), (300.0, 65.0)],
    "ZoneB": [(200.0, 40.0), (400.0, 100.0)],
    "ZoneC": [(250.0, 90.0), (100.0, 120.0), (200.0, 150.0), (200.0, 400.0)],
}

DEMAND_PRICE = 3000.0  # inelastic demand

NIGHT = range(0, 24)      # 00:00 - 06:00
MIDDAY = range(24, 68)    # 06:00 - 17:00
PEAK = range(68, 84)      # 17:00 - 21:00  (16 consecutive MTUs)
LATE = range(84, 96)      # 21:00 - 24:00


def zone_load(zone: str, mtu: int) -> float:
    """Return zonal load in MW for a 15-minute MTU index."""
    if zone in ("ZoneA", "ZoneB"):
        return 200.0
    if mtu in NIGHT:
        return 120.0
    if mtu in MIDDAY:
        return 330.0
    if mtu in PEAK:
        return 605.0
    return 220.0


# Hourly (block) orders: hour is 1-based (hour 20 = 19:00-20:00).
# volume_mw follows the ASSUME convention: positive = supply, negative = demand.
HOURLY_BLOCKS = [
    #{"bid_id": "BB_A_H09_SUP", "zone": "ZoneA", "hour": 9, "price": 15.0, "volume_mw": 20.0},
    #{"bid_id": "BB_B_H11_SUP", "zone": "ZoneB", "hour": 11, "price": 50.0, "volume_mw": 15.0},
    {"bid_id": "BB_C_H13_SUP", "zone": "ZoneC", "hour": 13, "price": 100.0, "volume_mw": 20.0},
    {"bid_id": "BB_C_H20_SUP", "zone": "ZoneC", "hour": 20, "price": 100.0, "volume_mw": 20.0},
    {"bid_id": "BB_C_H20_DEM", "zone": "ZoneC", "hour": 20, "price": 130.0, "volume_mw": -5.0},
]

# Storage Order (Zone C)
STORAGE_ORDER = {
    "bid_id": "SO_ZoneC_battery",
    "bidding_zone": "ZoneC",
    "starting_energy": 0.0,
    "efficiency": 0.86,
    "cycling_cost": 10.0,
    "end_of_day_value": 0.0,
    "max_energy": 240.0,        # MWh
    "max_input_power": 50.0,    # MW
    "max_output_power": 60.0,   # MW
}


# ---------------------------------------------------------------------------
# Input data creation
# ---------------------------------------------------------------------------


def snapshots() -> pd.DatetimeIndex:
    start = pd.Timestamp(DELIVERY_DATE)
    return pd.DatetimeIndex([start + i * MTU for i in range(N_MTU)])


def build_network() -> pypsa.Network:
    """Build the 3-zone / 3-node FBMC network (used for FBMC preprocessing)."""
    snaps = snapshots()

    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    n.add("Carrier", "load")
    n.add("Carrier", "thermal")

    for zone, bus in BUSES.items():
        n.add("Bus", bus, zone_id=zone, carrier="AC")

    for line, (bus0, bus1, s_nom) in LINES.items():
        n.add("Line", line, bus0=bus0, bus1=bus1, r= 0.01, x=0.1, s_nom=s_nom, s_max_pu=1.0, carrier="AC")

    # Keep the test case simple: no N-1 CNECs, only directional N-0 CNECs.
    n.lines["contingency"] = False

    for zone, tiers in SUPPLY_TIERS.items():
        for index, (p_nom, marginal_cost) in enumerate(tiers, start=1):
            n.add(
                "Generator",
                f"gen_{zone}_{index}",
                bus=BUSES[zone],
                p_nom=p_nom,
                marginal_cost=marginal_cost,
            )

    load_names = {}
    for zone in ZONES:
        name = f"load_{zone}"
        load_names[zone] = name
        n.add("Load", name, bus=BUSES[zone], p_set=0.0)

    n.loads_t.p_set = pd.DataFrame(
        {
            load_names[zone]: [zone_load(zone, mtu) for mtu in range(N_MTU)]
            for zone in ZONES
        },
        index=snaps,
    )

    return n


def write_simple_orders(path: Path) -> None:
    """Write the EUPHEMIA-like 15-minute simple-order CSV."""
    date_string = DELIVERY_DATE.strftime("%Y-%m-%d")
    iso = DELIVERY_DATE.isocalendar()
    rows = []

    for mtu in range(N_MTU):
        hour = mtu // 4 + 1
        quarter = mtu % 4 + 1

        for zone in ZONES:
            rows.append(
                {
                    "Date": date_string,
                    "Week": iso[1],
                    "Week Day": iso[2],
                    "Hour": hour,
                    "Quarter hour": quarter,
                    "Price": DEMAND_PRICE,
                    "Volume": zone_load(zone, mtu),
                    "Sale/Purchase": "Purchase",
                    "bidding_zone": zone,
                }
            )

            for p_nom, marginal_cost in SUPPLY_TIERS[zone]:
                rows.append(
                    {
                        "Date": date_string,
                        "Week": iso[1],
                        "Week Day": iso[2],
                        "Hour": hour,
                        "Quarter hour": quarter,
                        "Price": marginal_cost,
                        "Volume": p_nom,
                        "Sale/Purchase": "Sell",
                        "bidding_zone": zone,
                    }
                )

    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Wrote %s simple orders to %s.", len(rows), path)


def write_block_orders(path: Path) -> None:
    """
    Write the EUPHEMIA-like block-order CSV (semicolon separated, 105 fields).

    Layout convention of BlockOrderReader:
    0:'BB', 1:id, 2:type, 3:special_id, 4:-, 5:price, 6..101:96 interval
    volumes (sign inverted w.r.t. ASSUME), 102:min acceptance ratio,
    103:-, 104:bidding_zone
    """
    lines = []

    for block in HOURLY_BLOCKS:
        volumes = ["0"] * N_MTU
        first = (block["hour"] - 1) * 4

        for mtu in range(first, first + 4):
            # CSV value is negated by the reader -> invert here.
            volumes[mtu] = f"{-block['volume_mw']:.4f}"

        fields = (
            ["BB", block["bid_id"], "C01", "", ""]
            + [f"{block['price']:.4f}"]
            + volumes
            + ["1.0", "", block["zone"]]
        )

        assert len(fields) == 105
        lines.append(";".join(fields))

    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s hourly block orders to %s.", len(lines), path)


def write_storage_orders(path: Path) -> None:
    """Write the wide-format storage-order CSV (one row = one Storage Order)."""
    row = {
        "bid_id": STORAGE_ORDER["bid_id"],
        "bidding_zone": STORAGE_ORDER["bidding_zone"],
        "starting_energy": STORAGE_ORDER["starting_energy"],
        "efficiency": STORAGE_ORDER["efficiency"],
        "cycling_cost": STORAGE_ORDER["cycling_cost"],
        "end_of_day_value": STORAGE_ORDER["end_of_day_value"],
    }

    for mtu in range(1, N_MTU + 1):
        row[f"max_energy_{mtu:02d}"] = STORAGE_ORDER["max_energy"]
        row[f"max_input_power_{mtu:02d}"] = STORAGE_ORDER["max_input_power"]
        row[f"max_output_power_{mtu:02d}"] = STORAGE_ORDER["max_output_power"]
        row[f"min_energy_{mtu:02d}"] = 0.0

    pd.DataFrame([row]).to_csv(path, index=False)
    logger.info("Wrote 1 storage order to %s.", path)


def create_input_data() -> dict:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "simple": TEST_DATA_DIR / "simple_orders.csv",
        "blocks": TEST_DATA_DIR / "block_orders.csv",
        "storage": TEST_DATA_DIR / "storage_orders.csv",
        "network": TEST_DATA_DIR / "network.nc",
    }

    write_simple_orders(paths["simple"])
    write_block_orders(paths["blocks"])
    write_storage_orders(paths["storage"])

    network = build_network()
    try:
        network.export_to_netcdf(paths["network"])
    except Exception as error:  # netCDF export is optional for the test run
        logger.warning("Could not export the network to NetCDF: %s", error)

    return paths


# ---------------------------------------------------------------------------
# Clearing runs
# ---------------------------------------------------------------------------


def run_case(case: str, paths: dict, with_storage: bool) -> ClearingTestTool:
    logger.info("=" * 70)
    logger.info("Running case '%s' (storage order: %s).", case, with_storage)
    logger.info("=" * 70)

    tool = ClearingTestTool(
        market_id=f"fbmc_3zone_{case}",
        network=build_network(),            # fresh network per run
        flowbased_region=ZONES,             # pure FBMC, no NTC
        ntc_connections=None,
        zones_identifier="zone_id",
        param_dict={
            "solver_name": SOLVER,
            "fbmc_solver_name": FBMC_SOLVER,
            "log_flows": True,
            "log_performance": True,
            "pricing_mechanism": "pay_as_clear",
        },
    )

    tool.load_orders(
        delivery_date=DELIVERY_DATE,
        quarter_hourly_csv=paths["simple"],
        block_csv=paths["blocks"],
        storage_csv=paths["storage"] if with_storage else None,
        default_node="ZoneA",
    )

    tool.run_clearing()
    tool.print_summary()
    tool.export_results(RESULTS_DIR / case)

    return tool


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def price_table(tool: ClearingTestTool) -> pd.DataFrame:
    meta = pd.DataFrame(tool.results["market_meta"])
    return meta.pivot(index="product_start", columns="zone", values="price")


def max_consecutive(indices: list[int]) -> int:
    if not indices:
        return 0

    best = run = 1
    ordered = sorted(indices)

    for previous, current in zip(ordered, ordered[1:]):
        run = run + 1 if current == previous + 1 else 1
        best = max(best, run)

    return best


def congestion_report(tool: ClearingTestTool, case: str) -> pd.DataFrame:
    line_results = pd.DataFrame(tool.results.get("line_results", []))

    if line_results.empty or "network_type" not in line_results:
        return pd.DataFrame()

    cnecs = line_results[line_results["network_type"] == "FBMC_CNEC"].copy()

    if cnecs.empty:
        return pd.DataFrame()

    cnecs["mtu"] = cnecs["time"].map(
        {timestamp: index for index, timestamp in enumerate(snapshots())}
    )

    rows = []
    for (cne, direction), group in cnecs.groupby(["cne", "direction"]):
        binding = group.loc[group["binding"], "mtu"].tolist()
        rows.append(
            {
                "case": case,
                "cne": cne,
                "direction": direction,
                "binding_mtus": len(binding),
                "max_consecutive_binding_mtus": max_consecutive(binding),
                "first_binding_mtu": min(binding) if binding else None,
                "last_binding_mtu": max(binding) if binding else None,
            }
        )

    return pd.DataFrame(rows).sort_values("binding_mtus", ascending=False)


def storage_report(tool: ClearingTestTool) -> pd.DataFrame:
    storage_orders = [
        order
        for order in tool.results["accepted_orders"]
        if order.get("bid_type") == "SO"
    ]

    if not storage_orders:
        return pd.DataFrame()

    order = storage_orders[0]

    return pd.DataFrame(
        {
            "charge_mw": pd.Series(order["charge"]),
            "discharge_mw": pd.Series(order["discharge"]),
            "soc_mwh": pd.Series(order["soc"]),
            "price_eur_mwh": pd.Series(order["accepted_price"]),
            "surplus_eur": pd.Series(order["storage_surplus_eur"]),
        }
    ).sort_index()


def block_report(tool: ClearingTestTool, case: str) -> pd.DataFrame:
    accepted = {
        order["bid_id"]
        for order in tool.results["accepted_orders"]
        if order.get("bid_type") in {"BB", "LB"}
    }

    return pd.DataFrame(
        [
            {
                "case": case,
                "bid_id": block["bid_id"],
                "zone": block["zone"],
                "hour": block["hour"],
                "price": block["price"],
                "volume_mw": block["volume_mw"],
                "accepted": block["bid_id"] in accepted,
            }
            for block in HOURLY_BLOCKS
        ]
    )


def main() -> None:
    paths = create_input_data()

    tool_with = run_case("with_SO", paths, with_storage=True)
    tool_without = run_case("without_SO", paths, with_storage=False)

    prices_with = price_table(tool_with)
    prices_without = price_table(tool_without)

    comparison = prices_without.join(
        prices_with,
        lsuffix="_without_SO",
        rsuffix="_with_SO",
    )

    for zone in ZONES:
        comparison[f"{zone}_delta"] = (
            comparison[f"{zone}_with_SO"] - comparison[f"{zone}_without_SO"]
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "price_comparison.csv")

    storage = storage_report(tool_with)
    if not storage.empty:
        storage.to_csv(RESULTS_DIR / "with_SO" / "storage_dispatch.csv")

    congestion = pd.concat(
        [
            congestion_report(tool_without, "without_SO"),
            congestion_report(tool_with, "with_SO"),
        ],
        ignore_index=True,
    )
    if not congestion.empty:
        congestion.to_csv(RESULTS_DIR / "congestion_summary.csv", index=False)

    blocks = pd.concat(
        [
            block_report(tool_without, "without_SO"),
            block_report(tool_with, "with_SO"),
        ],
        ignore_index=True,
    )
    blocks.to_csv(RESULTS_DIR / "block_order_acceptance.csv", index=False)

    print("\n" + "=" * 78)
    print("STORAGE ORDER PRICE IMPACT (hourly averages, EUR/MWh)")
    print("=" * 78)
    hourly = comparison.resample("1h").mean().round(2)
    print(hourly.to_string())

    print("\n" + "=" * 78)
    print("EVENING PEAK (MTU 68-83) ZONE C PRICES")
    print("=" * 78)
    peak_index = comparison.index[list(PEAK)]
    print(
        comparison.loc[
            peak_index,
            ["ZoneC_without_SO", "ZoneC_with_SO", "ZoneC_delta"],
        ].round(2).to_string()
    )

    if not storage.empty:
        print("\n" + "=" * 78)
        print("STORAGE ORDER DISPATCH (Zone C)")
        print("=" * 78)
        print(storage.round(2).to_string())
        print(
            f"\nTotal discharged energy: "
            f"{sum(storage['discharge_mw']) * 0.25:.1f} MWh"
        )
        print(
            f"Total charged energy:    "
            f"{sum(storage['charge_mw']) * 0.25:.1f} MWh"
        )
        print(f"Max SOC:                 {storage['soc_mwh'].max():.1f} MWh")

    if not congestion.empty:
        print("\n" + "=" * 78)
        print("FBMC CNEC CONGESTION")
        print("=" * 78)
        print(congestion.to_string(index=False))

    print("\n" + "=" * 78)
    print("HOURLY (BLOCK) ORDER ACCEPTANCE")
    print("=" * 78)
    print(blocks.to_string(index=False))

    print(f"\nInputs:  {TEST_DATA_DIR.resolve()}")
    print(f"Results: {RESULTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()