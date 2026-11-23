"""
Benchmark execution: baseline, coupling-regime and line-scaling clearings.

All runners share clear_storage_scenarios(). Preparation (order books, network
loading) is done in preparation.py; this module only builds the clearing tool,
adds storage orders and exports the raw results.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path

import pandas as pd
import subprocess
import sys

from tools.clearing_test_tool import ClearingTestTool
from tools.create_zonal_orderbooks import normalize_zone_label

from . import config as cfg
from . import preparation as prep

logger = logging.getLogger(__name__)


# =============================================================================
# Storage orders
# =============================================================================
def add_storage_orders(
    tool: ClearingTestTool,
    zone: str,
    number_of_orders: int,
    *,
    max_energy: float | Mapping,
    max_input_power: float | Mapping,
    max_output_power: float | Mapping,
    cycling_cost: float,
    min_energy: float | Mapping = 0.0,
    starting_energy: float = 0.0,
    efficiency: float = 1.0,
    end_of_day_value: float = 0.0,
    bid_id_prefix: str = "SO",
    agent_addr: tuple[str, str] | None = None,
) -> list[dict]:
    """
    Append identically configured storage orders to a loaded zonal orderbook.

    Operating limits accept a scalar, expanded across all market MTUs, or a
    timestamp-to-value mapping covering exactly the loaded market horizon.
    Each order has independent limit dictionaries and a unique ID.
    """
    if (
        isinstance(number_of_orders, bool)
        or not isinstance(number_of_orders, int)
        or number_of_orders < 0
    ):
        raise ValueError("number_of_orders must be a non-negative integer.")

    if number_of_orders == 0:
        return []

    if not tool.orderbook or not tool.market_products:
        raise ValueError("Call tool.load_orders() before adding storage orders.")

    if zone is None or not str(zone).strip():
        raise ValueError("zone must be a non-empty zone label.")

    zone = normalize_zone_label(zone)

    if tool.network is not None:
        known_zones = {
            str(value)
            for value in tool.network.buses[tool.zones_identifier].dropna().unique()
        }
    else:
        known_zones = {str(order["node"]) for order in tool.orderbook} | {
            str(value) for value in tool.flowbased_region
        }

    if zone not in known_zones:
        raise ValueError(f"Unknown storage zone '{zone}'. Known zones: {sorted(known_zones)}")

    products = sorted(tool.market_products, key=lambda product: product[0])
    timestamps = [product[0] for product in products]

    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Market products contain duplicate start timestamps.")

    def numeric(value, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric.") from exc
        if not isfinite(result):
            raise ValueError(f"{field} must be finite.")
        return result

    def operating_limit(value, field: str) -> dict:
        if isinstance(value, Mapping):
            missing = [time for time in timestamps if time not in value]
            extra = [time for time in value if time not in set(timestamps)]
            if missing or extra:
                raise ValueError(
                    f"{field} must cover exactly all market MTUs. "
                    f"Missing: {missing[:5]}; extra: {extra[:5]}."
                )
            result = {time: numeric(value[time], f"{field}[{time}]") for time in timestamps}
        else:
            scalar = numeric(value, field)
            result = {time: scalar for time in timestamps}

        if any(limit < 0.0 for limit in result.values()):
            raise ValueError(f"{field} must be non-negative.")
        return result

    limits = {
        "max_energy": operating_limit(max_energy, "max_energy"),
        "min_energy": operating_limit(min_energy, "min_energy"),
        "max_input_power": operating_limit(max_input_power, "max_input_power"),
        "max_output_power": operating_limit(max_output_power, "max_output_power"),
    }

    for time in timestamps:
        if limits["min_energy"][time] > limits["max_energy"][time]:
            raise ValueError(f"min_energy exceeds max_energy at '{time}'.")

    static_values = {
        "starting_energy": numeric(starting_energy, "starting_energy"),
        "efficiency": numeric(efficiency, "efficiency"),
        "cycling_cost": numeric(cycling_cost, "cycling_cost"),
        "end_of_day_value": numeric(end_of_day_value, "end_of_day_value"),
    }

    if static_values["starting_energy"] < 0.0:
        raise ValueError("starting_energy must be non-negative.")
    if not 0.0 <= static_values["efficiency"] <= 1.0:
        raise ValueError("efficiency must be in [0, 1].")
    if static_values["cycling_cost"] < 0.0:
        raise ValueError("cycling_cost must be non-negative.")

    bid_id_prefix = str(bid_id_prefix).strip()
    if not bid_id_prefix:
        raise ValueError("bid_id_prefix must not be empty.")

    if agent_addr is None:
        agent_addr = (f"agent_{zone}", "test")
    elif (
        not isinstance(agent_addr, tuple)
        or len(agent_addr) != 2
        or not all(isinstance(value, str) for value in agent_addr)
    ):
        raise ValueError("agent_addr must be a tuple of two strings.")

    existing_ids = {order["bid_id"] for order in tool.orderbook}
    added_orders = []
    candidate_index = 1

    for _ in range(number_of_orders):
        while True:
            bid_id = f"{bid_id_prefix}_{zone}_{candidate_index:04d}"
            candidate_index += 1
            if bid_id not in existing_ids:
                break

        existing_ids.add(bid_id)

        added_orders.append({
            "bid_id": bid_id,
            "agent_addr": agent_addr,
            "bid_type": "SO",
            "node": zone,
            "price": 0.0,
            "volume": {},
            "start_time": products[0][0],
            "end_time": products[-1][1],
            "only_hours": None,
            "min_acceptance_ratio": None,
            "parent_bid_id": None,
            "exclusive_group_id": None,
            **static_values,
            **{field: values.copy() for field, values in limits.items()},
        })

    # Modify the orderbook only after the full configuration has been validated.
    tool.orderbook.extend(added_orders)
    tool.submitted_order_count = len(tool.orderbook)

    logger.info(
        "Added %s identically configured storage orders to zone '%s'.",
        len(added_orders), zone,
    )
    return added_orders


# =============================================================================
# Shared clearing helper
# =============================================================================
def _require_files(*paths: Path, hint: str) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files {missing}. {hint}")


def _tool_params(calculate_maxbex: bool, fbmc_kwargs: Mapping | None = None) -> dict:
    params = {
        "solver_name": cfg.SOLVER_NAME,
        "log_flows": True,
        "calculate_maxbex": calculate_maxbex,
    }
    if fbmc_kwargs is not None:
        params["fbmc_kwargs"] = dict(fbmc_kwargs)
    return params


def _storage_free(orderbook: list[dict]) -> list[dict]:
    return deepcopy([o for o in orderbook if o.get("bid_type") != "SO"])


def clear_storage_scenarios(
    tool: ClearingTestTool,
    base_orderbook: list[dict],
    zone: str,
    counts: tuple[int, ...],
    result_dir: Path,
    on_result: Callable[[int, dict], None] | None = None,
) -> None:
    """
    Clear one market without and with n identical storage orders.

    Each scenario starts from an independent copy of the storage-free order
    book. n orders are added at once (identical to adding them one by one:
    IDs and parameters are the same). Raw results go to
    result_dir/storage_orders_<n>.
    """
    for count in counts:
        label = cfg.scenario_label(count)
        tool.orderbook = deepcopy(base_orderbook)
        tool.submitted_order_count = len(tool.orderbook)

        if count > 0:
            add_storage_orders(
                tool=tool,
                zone=zone,
                number_of_orders=count,
                **dict(cfg.STORAGE_CONFIG),
            )

        logger.info("Clearing [%s] with %s storage order(s).", label, count)
        results = tool.run_clearing()
        tool.print_summary()
        tool.export_results(result_dir / label)

        if on_result is not None:
            on_result(count, results)


# =============================================================================
# Runners
# =============================================================================
def run_baseline_case(case: cfg.Case) -> None:
    """Full FBMC network; storage orders 0..n (baseline and Dunkelflaute day)."""
    day = case.day_config
    _require_files(day.zonal_simple, day.zonal_blocks,
                   hint="Run the 'prepare' stage first.")

    delivery_date = prep.resolve_delivery_date(day)
    network = prep.load_baseline_network(delivery_date)
    flowbased_region = sorted(network.buses[cfg.ZONE_COLUMN].dropna().unique(), key=str)

    tool = ClearingTestTool(
        market_id=f"{case.name}_market",
        network=network,
        flowbased_region=flowbased_region,
        zones_identifier=cfg.ZONE_COLUMN,
        param_dict=_tool_params(calculate_maxbex=True, fbmc_kwargs=case.fbmc_kwargs),
    )
    tool.load_orders(
        delivery_date=delivery_date,
        quarter_hourly_csv=day.zonal_simple,
        block_csv=day.zonal_blocks,
        default_node=flowbased_region[0],
    )

    clear_storage_scenarios(
        tool=tool,
        base_orderbook=_storage_free(tool.orderbook),
        zone=case.storage_zone,
        counts=case.storage_counts,
        result_dir=case.results_dir,
    )


def run_coupling_case(case: cfg.Case) -> None:
    """Isolated, NTC (from baseline MaxBEX) or copper-plate market."""
    day = case.day_config
    _require_files(day.zonal_simple, day.zonal_blocks,
                   hint="Run the 'prepare' stage first.")

    ntc = None
    if case.market == "ntc":
        if case.maxbex_csv is None:
            raise ValueError(f"[{case.name}] NTC case needs maxbex_csv.")
        _require_files(case.maxbex_csv, hint="Run the baseline case first.")
        ntc = prep.maxbex_to_ntc(case.maxbex_csv)

    delivery_date = prep.resolve_delivery_date(day)

    # No PyPSA network and no FB region are needed for these regimes.
    tool = ClearingTestTool(
        market_id=f"{case.name}_market",
        network=None,
        flowbased_region=[],
        ntc_connections=ntc,
        zones_identifier=cfg.ZONE_COLUMN,
        param_dict=_tool_params(calculate_maxbex=False),
    )
    tool.load_orders(
        delivery_date=delivery_date,
        quarter_hourly_csv=day.zonal_simple,
        block_csv=day.zonal_blocks,
    )

    if case.market == "copper_plate":
        for order in tool.orderbook:
            order["node"] = cfg.COPPER_PLATE_ZONE
            order["agent_addr"] = (f"agent_{cfg.COPPER_PLATE_ZONE}", "test")

    order_zones = {order["node"] for order in tool.orderbook}

    if ntc is not None:
        ntc_zones = set(ntc["from_zone"]) | set(ntc["to_zone"])
        if not ntc_zones <= order_zones:
            raise ValueError(
                f"NTC zones {sorted(ntc_zones - order_zones)} do not exist in "
                f"the order book zones {sorted(order_zones)}."
            )

    clear_storage_scenarios(
        tool=tool,
        base_orderbook=_storage_free(tool.orderbook),
        zone=case.storage_zone,
        counts=case.storage_counts,
        result_dir=case.results_dir,
    )


def _check_against_baseline(count: int, results: dict) -> None:
    """Factor 1.0 must reproduce the baseline prices."""
    baseline = cfg.get_case("fbmc")
    path = baseline.results_dir / cfg.scenario_label(count) / "market_meta.csv"

    if not path.exists():
        logger.warning("Baseline file %s not found; reproduction check skipped.", path)
        return

    reference = pd.read_csv(path, usecols=["zone", "product_start", "price"])
    reference["product_start"] = pd.to_datetime(reference["product_start"])
    current = pd.DataFrame(results["market_meta"])[["zone", "product_start", "price"]]
    current["product_start"] = pd.to_datetime(current["product_start"])

    merged = current.merge(reference, on=["zone", "product_start"],
                           suffixes=("", "_baseline"), how="outer")
    diff = (merged["price"] - merged["price_baseline"]).abs()
    worst = float(diff.max())

    if diff.isna().any() or worst > cfg.BASELINE_PRICE_TOL:
        logger.warning(
            "Factor 1.0 does NOT reproduce the baseline (storage_orders_%s): "
            "max price difference %.3f EUR/MWh, %s unmatched rows.",
            count, worst, int(diff.isna().sum()),
        )
    else:
        logger.info("Factor 1.0 reproduces baseline storage_orders_%s "
                    "(max difference %.3f EUR/MWh).", count, worst)


def run_line_scaling_case(case: cfg.Case) -> None:
    """FBMC with scaled line ratings; the FB domain is recomputed per factor."""
    from clearing_algorithm.euphemia_like import ComplexClearingRole

    if "fbmc_kwargs" not in inspect.getsource(ComplexClearingRole.clear):
        logger.warning(
            "ComplexClearingRole.clear() does not read 'fbmc_kwargs'. "
            "n1_slack_penalty is ignored; low factors may fail in the D-2 case."
        )

    day = case.day_config
    _require_files(day.zonal_simple, day.zonal_blocks,
                   hint="Run the 'prepare' stage first.")

    delivery_date = prep.resolve_delivery_date(day)
    network = prep.load_scaled_network(case.line_factor, delivery_date)
    flowbased_region = sorted(network.buses[cfg.ZONE_COLUMN].dropna().unique(), key=str)

    tool = ClearingTestTool(
        market_id=f"fbmc_{case.name}",
        network=network,
        flowbased_region=flowbased_region,
        zones_identifier=cfg.ZONE_COLUMN,
        param_dict=_tool_params(
            calculate_maxbex=cfg.LINE_SCALING_CALCULATE_MAXBEX,
            fbmc_kwargs=case.fbmc_kwargs,
        ),
    )
    tool.load_orders(
        delivery_date=delivery_date,
        quarter_hourly_csv=day.zonal_simple,
        block_csv=day.zonal_blocks,
        default_node=flowbased_region[0],
    )

    check = (abs(case.line_factor - 1.0) < 1e-9 and cfg.APPLY_BASELINE_LINE_SCALING)

    clear_storage_scenarios(
        tool=tool,
        base_orderbook=_storage_free(tool.orderbook),
        zone=case.storage_zone,
        counts=case.storage_counts,
        result_dir=case.results_dir,
        on_result=_check_against_baseline if check else None,
    )


RUNNERS: dict[str, Callable[[cfg.Case], None]] = {
    "baseline": run_baseline_case,
    "coupling": run_coupling_case,
    "line_scaling": run_line_scaling_case,
}


def run_case(case: cfg.Case) -> None:
    """Run all storage scenarios of one case."""
    if case.runner not in RUNNERS:
        raise ValueError(f"[{case.name}] unknown runner '{case.runner}'.")

    if case.runner == "coupling" and case.market == "ntc":
        # Fail early with a helpful message.
        _require_files(case.maxbex_csv, hint="Run the 'fbmc' baseline case first.")

    logger.info("\n%s\nCase '%s' (%s, day '%s', storage counts %s)\n%s",
                "=" * 70, case.name, case.market, case.day, case.storage_counts, "=" * 70)
    RUNNERS[case.runner](case)
    logger.info("Case '%s' finished. Results in %s.", case.name, case.results_dir)


def run_simulator_case(case: cfg.Case) -> None:
    """
    Run the price simulator on the staged input of one case.

    The runner is started in a subprocess with the current interpreter. It
    reads <SIMULATOR_INPUT_ROOT>/<case>/simulator_case.json and writes
    <SIMULATOR_OUTPUT_ROOT>/<method>/<case>/<mode>/sim_*.csv.
    A non-zero exit code raises, so CONTINUE_ON_ERROR in run_experiments
    applies as for the benchmark cases.
    """
    if not case.sim_methods:
        logger.info("[%s] no simulator methods configured; skipped.", case.name)
        return

    runner = Path(cfg.SIMULATOR_RUNNER)
    spec_file = case.simulator_input_dir / "simulator_case.json"
    _require_files(runner, spec_file,
                   hint="Check SIMULATOR_RUNNER and run the 'simulator_inputs' stage first.")

    command = [
        sys.executable, str(runner),
        "--input-root", str(cfg.SIMULATOR_INPUT_ROOT),
        "--output-root", str(cfg.SIMULATOR_OUTPUT_ROOT),
        "--cases", case.name,
    ]
    logger.info("[%s] running price simulator: %s", case.name, " ".join(command))

    # Run from the simulator project root (parent of tools/).
    completed = subprocess.run(command, cwd=runner.parents[1])
    if completed.returncode != 0:
        raise RuntimeError(
            f"Price simulator failed for case '{case.name}' "
            f"(exit code {completed.returncode})."
        )