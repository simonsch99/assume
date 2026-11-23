"""
Load raw benchmark and simulator outputs into standardised tables.

No research metrics are calculated here. Identities are preserved (for example
the copper-plate zone stays "CP"; derived views such as copying it to the
reference zones or demand-weighted system prices belong to the analysis).

Tables (all carry family, case, day where applicable):

    cases              one row per case (metadata)
    prices             case, scenario, zone, time, price, supply/demand, net position
    storage_orders     one row per case/scenario: storage parameters as cleared
    storage_dispatch   aggregated storage charge/discharge/SOC and zonal price
    constraints        limit flags per element (CNEC or NTC line) and MTU
    maxbex             directed MaxBEX per scenario
    block_acceptance   acceptance ratio of every block bid per scenario
    sim_prices         simulator prices (regime = case name)
    sim_dispatch       simulator dispatch
    sim_summary        simulator summary rows
    validation         checks on coverage and consistency
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)

TABLES = (
    "cases", "prices", "storage_orders", "storage_dispatch", "constraints",
    "maxbex", "block_acceptance", "sim_prices", "sim_dispatch", "sim_summary",
    "validation",
)


# =============================================================================
# Helpers
# =============================================================================
def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV; return an empty frame if it is missing or empty."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} misses required columns: {sorted(missing)}")


def scenario_number(name: str) -> int:
    match = re.fullmatch(r"storage_orders_(\d+)", str(name))
    if match is None:
        raise ValueError(f"Unexpected scenario name '{name}'.")
    return int(match.group(1))


def parse_mapping(value) -> pd.Series:
    """Parse a JSON {timestamp: value} cell of the exported order CSVs."""
    if not isinstance(value, str):
        return pd.Series(dtype=float)
    value = value.strip()
    if not value.startswith("{"):
        return pd.Series(dtype=float)
    mapping = json.loads(value)
    if not mapping:
        return pd.Series(dtype=float)
    series = pd.Series(mapping, dtype=float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def sum_mappings(values) -> pd.Series:
    total = pd.Series(dtype=float)
    for value in values:
        total = total.add(parse_mapping(value), fill_value=0.0)
    return total.sort_index()


def _tag(frame: pd.DataFrame, case: cfg.Case) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "day", case.day)
    frame.insert(0, "case", case.name)
    frame.insert(0, "family", case.family)
    return frame


def _add_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    frame["storage_power_mw"] = frame["storage_order_count"] * cfg.ORDER_POWER_MW
    frame["storage_energy_mwh"] = frame["storage_order_count"] * cfg.ORDER_ENERGY_MWH
    return frame


def _optional(data: pd.DataFrame, column: str, numeric: bool = False) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return pd.to_numeric(data[column], errors="coerce") if numeric else data[column]


def available_scenarios(case: cfg.Case) -> dict[str, Path]:
    """Expected scenarios of a case that exist on disk (stale folders are ignored)."""
    scenarios = {}
    for count in case.storage_counts:
        label = cfg.scenario_label(count)
        directory = case.results_dir / label
        if (directory / "market_meta.csv").exists():
            scenarios[label] = directory
        else:
            logger.warning("[%s] missing %s/market_meta.csv.", case.name, label)
    return scenarios


# =============================================================================
# Benchmark loaders
# =============================================================================
def case_table(cases: list[cfg.Case]) -> pd.DataFrame:
    return pd.DataFrame([{
        "family": c.family, "case": c.name, "day": c.day, "runner": c.runner,
        "market": c.market, "line_factor": c.line_factor,
        "storage_zone": c.storage_zone, "curve_zone": c.curve_zone,
        "report_zone": c.report_zone,
        "storage_counts": ",".join(map(str, c.storage_counts)),
        "sim_storage_counts": ",".join(map(str, c.sim_storage_counts)),
        "sim_methods": ",".join(c.sim_methods),
        "results_dir": str(c.results_dir),
    } for c in cases])


def load_prices(case: cfg.Case) -> pd.DataFrame:
    """Zonal prices and volumes (market_meta.csv) of all scenarios."""
    frames = []
    for scenario, directory in available_scenarios(case).items():
        path = directory / "market_meta.csv"
        meta = pd.read_csv(path)
        require_columns(
            meta, {"zone", "product_start", "price", "supply_volume", "demand_volume"},
            str(path),
        )
        frame = pd.DataFrame({
            "scenario": scenario,
            "storage_order_count": scenario_number(scenario),
            "zone": meta["zone"].astype(str),
            "time": pd.to_datetime(meta["product_start"]),
            "price": pd.to_numeric(meta["price"], errors="raise"),
            "supply_mw": pd.to_numeric(meta["supply_volume"], errors="raise"),
            "demand_mw": pd.to_numeric(meta["demand_volume"], errors="raise"),
        })
        # Net position = supply - demand (positive = export); storage
        # discharge counts as supply and charging as demand.
        frame["net_position_mw"] = frame["supply_mw"] - frame["demand_mw"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return _tag(_add_capacity(pd.concat(frames, ignore_index=True)), case)


_STORAGE_FIELDS = {
    "charge", "discharge", "soc", "node", "efficiency", "cycling_cost",
    "end_of_day_value", "starting_energy", "total_storage_surplus_eur",
}


def load_storage(case: cfg.Case, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Storage orders as cleared: parameter table and aggregated dispatch."""
    meta_rows, dispatch_frames = [], []

    for scenario, directory in available_scenarios(case).items():
        count = scenario_number(scenario)
        if count == 0:
            continue

        accepted = read_csv_safe(directory / "accepted_orders.csv")
        if accepted.empty or "bid_type" not in accepted.columns:
            logger.warning("[%s/%s] accepted_orders.csv missing or empty.", case.name, scenario)
            continue

        storage = accepted.loc[accepted["bid_type"] == "SO"].copy()
        if storage.empty:
            logger.warning("[%s/%s] no accepted storage orders.", case.name, scenario)
            continue

        require_columns(storage, _STORAGE_FIELDS, f"{case.name}/{scenario}/accepted_orders.csv")

        charge = sum_mappings(storage["charge"])
        discharge = sum_mappings(storage["discharge"])
        soc = sum_mappings(storage["soc"])
        index = charge.index.union(discharge.index).union(soc.index)

        dispatch = pd.DataFrame({
            "charge_mwh": charge.reindex(index),
            "discharge_mwh": discharge.reindex(index),
            "soc_mwh": soc.reindex(index),
        }).fillna(0.0).sort_index()
        dispatch.index.name = "time"

        # Charge/discharge are MWh per MTU.
        dispatch["net_discharge_mw"] = (
            dispatch["discharge_mwh"] - dispatch["charge_mwh"]
        ) / cfg.MTU_HOURS

        node = str(storage["node"].iloc[0])
        price = prices.loc[
            (prices["scenario"] == scenario) & (prices["zone"] == node),
            ["time", "price"],
        ].set_index("time")["price"]
        dispatch["price"] = price.reindex(dispatch.index)

        dispatch = dispatch.reset_index()
        dispatch.insert(0, "report_zone", case.report_zone)
        dispatch.insert(0, "zone", node)
        dispatch.insert(0, "storage_order_count", count)
        dispatch.insert(0, "scenario", scenario)
        dispatch_frames.append(dispatch)

        meta_rows.append({
            "scenario": scenario,
            "storage_order_count": count,
            "zone": node,
            "n_accepted_storage_orders": len(storage),
            "efficiency": float(storage["efficiency"].astype(float).iloc[0]),
            "cycling_cost": float(storage["cycling_cost"].astype(float).iloc[0]),
            "end_of_day_value": float(storage["end_of_day_value"].astype(float).iloc[0]),
            "starting_energy": float(storage["starting_energy"].astype(float).sum()),
            "total_storage_surplus_eur": float(
                storage["total_storage_surplus_eur"].astype(float).sum()
            ),
        })

    if not dispatch_frames:
        return pd.DataFrame(), pd.DataFrame()

    meta = _tag(_add_capacity(pd.DataFrame(meta_rows)), case)
    dispatch = _tag(_add_capacity(pd.concat(dispatch_frames, ignore_index=True)), case)
    return meta, dispatch


_CONSTRAINT_COLUMNS = [
    "scenario", "storage_order_count", "time", "element", "element_type", "limited",
    "flow_mw", "ram_mw", "slack_mw", "utilization", "capacity_up_mw",
    "capacity_down_mw", "cne", "contingency", "direction",
]


def _cnec_frame(directory: Path, scenario: str) -> pd.DataFrame:
    data = read_csv_safe(directory / "CNECs.csv")
    if data.empty:
        return pd.DataFrame()
    require_columns(data, {"time", "cnec", "ram", "slack", "binding"},
                    str(directory / "CNECs.csv"))

    slack = pd.to_numeric(data["slack"], errors="coerce")
    flagged = data["binding"].astype(str).str.lower().eq("true")

    return pd.DataFrame({
        "scenario": scenario,
        "storage_order_count": scenario_number(scenario),
        "time": pd.to_datetime(data["time"]),
        "element": data["cnec"],
        "element_type": "cnec",
        "limited": flagged | (slack <= cfg.FLOW_TOL_MW),
        "flow_mw": _optional(data, "flow", numeric=True),
        "ram_mw": pd.to_numeric(data["ram"], errors="coerce"),
        "slack_mw": slack,
        "utilization": _optional(data, "utilization", numeric=True),
        "capacity_up_mw": np.nan,
        "capacity_down_mw": np.nan,
        "cne": _optional(data, "cne"),
        "contingency": _optional(data, "contingency"),
        "direction": _optional(data, "direction"),
    })


def _ntc_frame(directory: Path, scenario: str, capacity: pd.DataFrame) -> pd.DataFrame:
    flows = read_csv_safe(directory / "flows.csv")
    if flows.empty:
        return pd.DataFrame()
    require_columns(flows, {"time", "line", "flow", "network_type"},
                    str(directory / "flows.csv"))

    flows = flows[flows["network_type"] == "NTC"].copy()
    if flows.empty:
        return pd.DataFrame()

    flows["time"] = pd.to_datetime(flows["time"])
    flows = flows.rename(columns={"line": "element"})
    data = flows.merge(capacity, on=["time", "element"], how="left")

    tol = cfg.FLOW_TOL_MW
    at_up = (data["capacity_up_mw"] > tol) & (data["flow"] >= data["capacity_up_mw"] - tol)
    at_down = (data["capacity_down_mw"] > tol) & (data["flow"] <= -data["capacity_down_mw"] + tol)

    return pd.DataFrame({
        "scenario": scenario,
        "storage_order_count": scenario_number(scenario),
        "time": data["time"],
        "element": data["element"],
        "element_type": "ntc_line",
        "limited": at_up | at_down,
        "flow_mw": pd.to_numeric(data["flow"], errors="coerce"),
        "ram_mw": np.nan,
        "slack_mw": np.nan,
        "utilization": np.nan,
        "capacity_up_mw": data["capacity_up_mw"],
        "capacity_down_mw": data["capacity_down_mw"],
        "cne": np.nan, "contingency": np.nan, "direction": np.nan,
    })


def load_constraints(case: cfg.Case) -> pd.DataFrame:
    """Limit flags per CNEC (FBMC) or NTC line (NTC) and MTU."""
    if case.market not in ("fbmc", "ntc"):
        return pd.DataFrame()

    capacity = None
    if case.market == "ntc":
        from .preparation import maxbex_to_ntc   # lazy: avoids importing pypsa here

        if case.maxbex_csv is None or not case.maxbex_csv.exists():
            logger.warning("[%s] MaxBEX file missing; NTC constraints skipped.", case.name)
            return pd.DataFrame()
        capacity = maxbex_to_ntc(case.maxbex_csv).rename(columns={
            "line_id": "element", "timestamp": "time",
            "capacity_up": "capacity_up_mw", "capacity_down": "capacity_down_mw",
        })[["time", "element", "capacity_up_mw", "capacity_down_mw"]]

    frames = []
    for scenario, directory in available_scenarios(case).items():
        frame = (_cnec_frame(directory, scenario) if case.market == "fbmc"
                 else _ntc_frame(directory, scenario, capacity))
        if frame.empty:
            logger.warning("[%s/%s] no constraint data.", case.name, scenario)
            continue
        frames.append(frame[_CONSTRAINT_COLUMNS])

    if not frames:
        return pd.DataFrame()
    return _tag(pd.concat(frames, ignore_index=True), case)


def load_maxbex(case: cfg.Case) -> pd.DataFrame:
    """Directed MaxBEX per scenario (FBMC cases only)."""
    frames = []
    for scenario, directory in available_scenarios(case).items():
        data = read_csv_safe(directory / "maxbex.csv")
        if data.empty:
            continue
        require_columns(data, {"product_start", "from_zone", "to_zone", "maxbex_mw"},
                        str(directory / "maxbex.csv"))
        frames.append(pd.DataFrame({
            "scenario": scenario,
            "storage_order_count": scenario_number(scenario),
            "time": pd.to_datetime(data["product_start"]),
            "from_zone": data["from_zone"].astype(str),
            "to_zone": data["to_zone"].astype(str),
            "maxbex_mw": pd.to_numeric(data["maxbex_mw"], errors="coerce"),
        }))
    if not frames:
        return pd.DataFrame()
    return _tag(pd.concat(frames, ignore_index=True), case)


_BLOCK_COLUMNS = ["bid_id", "zone", "bid_type", "acceptance_ratio", "active"]


def _block_acceptance(directory: Path) -> pd.DataFrame:
    accepted = read_csv_safe(directory / "accepted_orders.csv")
    rejected = read_csv_safe(directory / "rejected_orders.csv")
    orders = pd.concat([accepted, rejected], ignore_index=True)

    if orders.empty:
        return pd.DataFrame(columns=_BLOCK_COLUMNS)

    require_columns(orders, {"bid_id", "bid_type", "node", "volume", "accepted_volume"},
                    f"order files in {directory}")

    rows = []
    for row in orders.loc[orders["bid_type"].isin(["BB", "LB"])].itertuples(index=False):
        submitted = float(parse_mapping(row.volume).abs().sum())
        accepted_energy = float(parse_mapping(row.accepted_volume).abs().sum())
        ratio = (accepted_energy / submitted
                 if submitted > cfg.BLOCK_ACCEPTANCE_TOL else 0.0)
        rows.append({
            "bid_id": str(row.bid_id),
            "zone": str(row.node),
            "bid_type": row.bid_type,
            "acceptance_ratio": ratio,
            "active": ratio > cfg.BLOCK_ACCEPTANCE_TOL,
        })

    result = pd.DataFrame(rows, columns=_BLOCK_COLUMNS)
    if result["bid_id"].duplicated().any():
        raise ValueError(f"Duplicate block IDs detected in {directory}.")
    return result


def load_block_acceptance(case: cfg.Case) -> pd.DataFrame:
    frames = []
    for scenario, directory in available_scenarios(case).items():
        frame = _block_acceptance(directory)
        if frame.empty:
            continue
        frame.insert(0, "storage_order_count", scenario_number(scenario))
        frame.insert(0, "scenario", scenario)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return _tag(pd.concat(frames, ignore_index=True), case)


# =============================================================================
# Simulator loaders (simulator is run manually)
# =============================================================================
def _read_simulator(root: Path, filename: str) -> pd.DataFrame:
    files = sorted(root.glob(f"*/*/*/{filename}"))
    if not files:
        logger.info("No %s files below %s.", filename, root)
        return pd.DataFrame()

    frames = []
    for path in files:
        frame = pd.read_csv(path, parse_dates=["time"] if filename != "sim_summary.csv" else None)
        frame["simulator_source_file"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)

    require_columns(result, {"method", "mode", "regime", "scenario"}, filename)
    result["regime"] = result["regime"].astype(str).str.strip()
    result["scenario"] = result["scenario"].astype(str)
    result.insert(0, "case", result["regime"])

    if filename != "sim_summary.csv":
        require_columns(result, {"zone", "time"}, filename)
        result["zone"] = result["zone"].astype(str)
        result["time"] = pd.to_datetime(result["time"])

        key = ["method", "mode", "regime", "zone", "scenario", "time"]
        duplicate = result.duplicated(key, keep=False)
        if duplicate.any():
            raise ValueError(
                f"{filename} contains {int(duplicate.sum())} duplicate rows. "
                f"Examples: {result.loc[duplicate, key].head(5).to_dict('records')}"
            )
    return result


def load_simulator_results(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = Path(root or cfg.SIMULATOR_OUTPUT_ROOT)
    prices = _read_simulator(root, "sim_prices_long.csv")
    dispatch = _read_simulator(root, "sim_dispatch.csv")
    summary = _read_simulator(root, "sim_summary.csv")

    if not prices.empty:
        require_columns(prices, {"price"}, "sim_prices_long.csv")
        prices["price"] = pd.to_numeric(prices["price"], errors="raise")
    if not dispatch.empty:
        require_columns(dispatch, {"charge_mwh", "discharge_mwh", "price"}, "sim_dispatch.csv")

    return {"sim_prices": prices, "sim_dispatch": dispatch, "sim_summary": summary}


# =============================================================================
# Validation
# =============================================================================
def validate_results(results: dict[str, pd.DataFrame], cases: list[cfg.Case]) -> pd.DataFrame:
    """Coverage and consistency checks. Returns one row per finding."""
    issues: list[dict] = []

    def add(level, case, scenario, check, detail):
        issues.append({"level": level, "case": case, "scenario": scenario,
                       "check": check, "detail": detail})

    prices = results.get("prices", pd.DataFrame())
    meta = results.get("storage_orders", pd.DataFrame())
    dispatch = results.get("storage_dispatch", pd.DataFrame())
    constraints = results.get("constraints", pd.DataFrame())

    for case in cases:
        p = prices[prices["case"] == case.name] if not prices.empty else prices
        loaded = set(p["scenario"].unique()) if not p.empty else set()

        for count in case.storage_counts:
            label = cfg.scenario_label(count)
            if label not in loaded:
                add("error", case.name, label, "scenario_missing",
                    "no market_meta.csv found")

        if cfg.NO_STORAGE not in loaded:
            continue

        reference = p[p["scenario"] == cfg.NO_STORAGE]
        reference_keys = set(zip(reference["zone"], reference["time"]))

        for scenario, g in p.groupby("scenario"):
            if g.duplicated(["zone", "time"]).any():
                add("error", case.name, scenario, "duplicate_price_rows", "")
            if g["price"].isna().any():
                add("error", case.name, scenario, "nan_prices",
                    f"{int(g['price'].isna().sum())} rows")
            if g["time"].nunique() != cfg.EXPECTED_MTUS:
                add("error", case.name, scenario, "mtu_count",
                    f"{g['time'].nunique()} != {cfg.EXPECTED_MTUS}")
            if set(zip(g["zone"], g["time"])) != reference_keys:
                add("error", case.name, scenario, "zone_time_mismatch",
                    "zone/time grid differs from storage_orders_0")

        if case.market == "isolated":
            worst = float(p["net_position_mw"].abs().max())
            if worst > cfg.FLOW_TOL_MW:
                add("warning", case.name, "", "isolated_net_position",
                    f"max |net position| {worst:.4f} MW")

        # Storage consistency
        expected = {
            "efficiency": cfg.EFFICIENCY,
            "cycling_cost": cfg.CYCLING_COST_EUR_PER_MWH,
            "end_of_day_value": cfg.END_OF_DAY_VALUE,
        }
        for count in case.storage_counts:
            if count == 0:
                continue
            label = cfg.scenario_label(count)
            m = meta[(meta["case"] == case.name) & (meta["scenario"] == label)] \
                if not meta.empty else meta
            if m.empty:
                add("error", case.name, label, "storage_missing",
                    "no accepted storage orders found")
                continue
            row = m.iloc[0]
            if int(row["n_accepted_storage_orders"]) != count:
                add("warning", case.name, label, "storage_order_count",
                    f"{int(row['n_accepted_storage_orders'])} accepted, {count} submitted")
            for column, value in expected.items():
                if not np.isclose(float(row[column]), value, atol=1e-6):
                    add("warning", case.name, label, f"storage_{column}",
                        f"{row[column]} differs from config {value}")
            if abs(float(row["starting_energy"]) - count * cfg.STARTING_ENERGY) > 1e-6:
                add("warning", case.name, label, "storage_starting_energy",
                    f"{row['starting_energy']}")

        # SOC at the end of the day (information for the terminal-condition check)
        if not dispatch.empty:
            d = dispatch[dispatch["case"] == case.name]
            for scenario, g in d.groupby("scenario"):
                end_soc = float(g.sort_values("time")["soc_mwh"].iloc[-1])
                if end_soc > 1e-6:
                    add("info", case.name, scenario, "end_soc_nonzero",
                        f"{end_soc:.3f} MWh")

        # Constraint feasibility
        if not constraints.empty:
            c = constraints[constraints["case"] == case.name]
            if not c.empty and c["slack_mw"].notna().any():
                worst = float(-c["slack_mw"].min())
                if worst > cfg.FLOW_TOL_MW:
                    add("warning", case.name, "", "constraint_violation",
                        f"max violation {worst:.4f} MW")
        elif case.market in ("fbmc", "ntc"):
            add("warning", case.name, "", "constraints_missing", "no constraint data")

    # Simulator coverage
    sim_prices = results.get("sim_prices", pd.DataFrame())
    if not sim_prices.empty:
        known = {c.name: c for c in cases}
        for regime in sorted(set(sim_prices["regime"]) - set(known)):
            add("warning", regime, "", "sim_regime_unknown",
                "simulator regime without a configured case")
        for case in cases:
            s = sim_prices[sim_prices["regime"] == case.name]
            if s.empty:
                continue
            expected_scenarios = {cfg.NO_STORAGE} | {
                cfg.scenario_label(n) for n in case.sim_storage_counts
            }
            for method_key in case.sim_methods:
                label = cfg.SIMULATOR_METHODS[method_key]["label"]
                have = set(s.loc[s["method"] == label, "scenario"])
                for scenario in sorted(expected_scenarios - have):
                    add("warning", case.name, scenario, "sim_scenario_missing",
                        f"method '{label}'")

    return pd.DataFrame(issues, columns=["level", "case", "scenario", "check", "detail"])


# =============================================================================
# Top level
# =============================================================================
def load_all_results(
    cases: list[cfg.Case],
    include_simulator: bool = True,
    simulator_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load, standardise and validate all results of the selected cases."""
    collected: dict[str, list[pd.DataFrame]] = {
        k: [] for k in ("prices", "storage_orders", "storage_dispatch",
                        "constraints", "maxbex", "block_acceptance")
    }

    for case in cases:
        logger.info("Loading results of case '%s'.", case.name)
        prices = load_prices(case)
        if prices.empty:
            logger.warning("[%s] no results found.", case.name)
            continue
        meta, dispatch = load_storage(case, prices)

        collected["prices"].append(prices)
        collected["storage_orders"].append(meta)
        collected["storage_dispatch"].append(dispatch)
        collected["constraints"].append(load_constraints(case))
        collected["maxbex"].append(load_maxbex(case))
        collected["block_acceptance"].append(load_block_acceptance(case))

    results: dict[str, pd.DataFrame] = {"cases": case_table(cases)}
    for key, frames in collected.items():
        frames = [f for f in frames if not f.empty]
        results[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if include_simulator:
        results.update(load_simulator_results(simulator_root))
    else:
        results.update({k: pd.DataFrame() for k in ("sim_prices", "sim_dispatch", "sim_summary")})

    results["validation"] = validate_results(results, cases)
    return results


def write_standardised(results: dict[str, pd.DataFrame], directory: Path | None = None) -> Path:
    """Write the standardised tables as CSV for the analysis stage."""
    directory = Path(directory or cfg.STANDARDISED_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        frame = results.get(name)
        path = directory / f"{name}.csv"
        if frame is None or frame.empty:
            if path.exists():
                path.unlink()           # do not leave stale tables behind
            continue
        frame.to_csv(path, index=False)
        logger.info("Wrote %s (%s rows).", path, len(frame))
    return directory


def read_standardised(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read tables written by write_standardised (entry point for analysis.py)."""
    directory = Path(directory or cfg.STANDARDISED_DIR)
    results = {}
    for name in TABLES:
        path = directory / f"{name}.csv"
        if not path.exists():
            results[name] = pd.DataFrame()
            continue
        frame = pd.read_csv(path)
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"])
        results[name] = frame
    return results