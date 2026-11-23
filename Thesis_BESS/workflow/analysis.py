"""
Metric calculation for the benchmark vs. price-simulator comparison.

Input : standardised tables from results.py (dict of DataFrames).
Output: dict of result tables (written by write_analysis).

Structure
    * Metric blocks   small functions registered in ACCURACY / SPIKES. Each takes
                      the rows of one group and returns {metric_name: value}.
                      Add a metric by writing one function with a decorator; drop
                      one by deleting it or listing its name in
                      config.ANALYSIS_DISABLED_METRICS.
    * Tables          functions registered with @table. They group the data, apply
                      a metric registry and assemble a result table. Derived
                      tables read other tables from ctx.tables. Tables can be
                      disabled via config.ANALYSIS_DISABLED_TABLES.

Comparison level: storage zone only (the simulator simulates one zone).
Benchmark zone = case.curve_zone ("CP" for copper plate), simulator zone =
case.report_zone.

Research-question mapping
    RQ1  rq1_accuracy, rq1_by_congestion, case_conditions, gate_p0
    RQ2  rq2_methods, rq2_decomposition, bench_block_changes, bench_spillover
    RQ3  rq3_spikes, rq3_response, rq3_comparison
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .results import read_standardised

logger = logging.getLogger(__name__)

NO = cfg.NO_STORAGE
SHIFT = cfg.SIMULATOR_METHODS["base"]["label"]
BLOCKS = cfg.SIMULATOR_METHODS["cb"]["label"]
COUPLING = cfg.SIMULATOR_METHODS["mc"]["label"]
BOTH = cfg.SIMULATOR_METHODS["mc_cb"]["label"]

CASE_INFO = ["case", "family", "day", "market", "line_factor"]
NAN = float("nan")


# =============================================================================
# Registries
# =============================================================================
class Registry:
    """Ordered collection of metric blocks."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable] = {}

    def register(self, key: str):
        def decorator(function: Callable) -> Callable:
            if key in self._items:
                raise ValueError(f"Metric block '{key}' already registered in {self.name}.")
            self._items[key] = function
            return function
        return decorator

    def keys(self) -> list[str]:
        return list(self._items)

    def compute(self, group: pd.DataFrame, ctx: "Context", only=None) -> dict:
        out: dict = {}
        for key, function in self._items.items():
            if key in cfg.ANALYSIS_DISABLED_METRICS:
                continue
            if only is not None and key not in only:
                continue
            result = function(group, ctx)
            clash = set(result) & set(out)
            if clash:
                raise ValueError(f"Metric block '{key}' redefines {sorted(clash)}.")
            out.update(result)
        return out


ACCURACY = Registry("accuracy")   # groups of paired benchmark/simulator rows
SPIKES = Registry("spikes")       # groups of one price series (benchmark or method)
TABLES: dict[str, Callable] = {}


def table(name: str):
    def decorator(function: Callable) -> Callable:
        if name in TABLES:
            raise ValueError(f"Table '{name}' already registered.")
        TABLES[name] = function
        return function
    return decorator


# =============================================================================
# Context and data alignment
# =============================================================================
@dataclass
class Context:
    results: dict
    cases: pd.DataFrame            # one row per case
    bench: pd.DataFrame            # benchmark storage-zone series
    sim: pd.DataFrame              # simulator series
    pairs: pd.DataFrame            # paired benchmark/simulator rows per MTU
    thresholds: pd.DataFrame       # spike threshold per case
    tables: dict = field(default_factory=dict)


def _mask_missing_dispatch(frame: pd.DataFrame, dispatch: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """No-storage scenarios have zero dispatch; storage scenarios without dispatch data stay NaN."""
    have = set(map(tuple, dispatch[keys].drop_duplicates().to_numpy())) if not dispatch.empty else set()
    ident = list(zip(*[frame[k] for k in keys]))
    known = np.array([(i in have) for i in ident]) | (frame["scenario"] == NO).to_numpy()
    frame["q_mw"] = frame["q_mw"].where(known, np.nan)
    frame.loc[(frame["scenario"] == NO).to_numpy(), "q_mw"] = 0.0
    frame.loc[known & frame["q_mw"].isna().to_numpy(), "q_mw"] = 0.0
    return frame


def build_benchmark_series(results: dict, cases: pd.DataFrame) -> pd.DataFrame:
    """Benchmark price and net dispatch (MW) in the storage zone of each case."""
    prices = results["prices"]
    if prices.empty:
        return pd.DataFrame()

    merged = prices.merge(cases[["case", "curve_zone"]], on="case", how="inner")
    bench = merged.loc[merged["zone"] == merged["curve_zone"],
                       ["case", "scenario", "storage_order_count", "time", "price"]].copy()

    dispatch = results["storage_dispatch"]
    if not dispatch.empty:
        d = dispatch[["case", "scenario", "time", "net_discharge_mw"]].rename(
            columns={"net_discharge_mw": "q_mw"})
        bench = bench.merge(d, on=["case", "scenario", "time"], how="left", validate="one_to_one")
    else:
        bench["q_mw"] = np.nan

    return _mask_missing_dispatch(bench, dispatch, ["case", "scenario"])


def build_simulator_series(results: dict, cases: pd.DataFrame) -> pd.DataFrame:
    """Simulator price and net dispatch (MW) per case, method and mode."""
    prices = results["sim_prices"]
    if prices.empty:
        return pd.DataFrame()

    merged = prices.merge(cases[["case", "report_zone"]], on="case", how="inner")
    sim = merged.loc[merged["zone"] == merged["report_zone"],
                     ["case", "method", "mode", "scenario", "time", "price"]].copy()
    sim["storage_order_count"] = sim["scenario"].str.extract(r"(\d+)$")[0].astype(int)

    dispatch = results["sim_dispatch"]
    keys = ["case", "method", "mode", "scenario", "time"]
    if not dispatch.empty:
        d = dispatch.copy()
        d["q_mw"] = (d["discharge_mwh"] - d["charge_mwh"]) / cfg.MTU_HOURS
        sim = sim.merge(d[keys + ["q_mw"]], on=keys, how="left", validate="one_to_one")
    else:
        sim["q_mw"] = np.nan

    return _mask_missing_dispatch(sim, dispatch, ["case", "method", "mode", "scenario"])


def build_pairs(bench: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """One row per case, method, mode, storage scenario and MTU."""
    if bench.empty or sim.empty:
        return pd.DataFrame()

    b0 = bench.loc[bench["scenario"] == NO, ["case", "time", "price"]].rename(columns={"price": "p0_b"})
    b1 = bench.loc[bench["scenario"] != NO,
                   ["case", "scenario", "storage_order_count", "time", "price", "q_mw"]].rename(
        columns={"price": "p1_b", "q_mw": "q_b_mw"})
    s0 = sim.loc[sim["scenario"] == NO, ["case", "method", "mode", "time", "price"]].rename(
        columns={"price": "p0_s"})
    s1 = sim.loc[sim["scenario"] != NO,
                 ["case", "method", "mode", "scenario", "time", "price", "q_mw"]].rename(
        columns={"price": "p1_s", "q_mw": "q_s_mw"})

    pairs = (
        s1.merge(s0, on=["case", "method", "mode", "time"], how="inner", validate="many_to_one")
          .merge(b1, on=["case", "scenario", "time"], how="inner", validate="many_to_one")
          .merge(b0, on=["case", "time"], how="inner", validate="many_to_one")
    )

    if len(pairs) < len(s1):
        logger.warning("%s of %s simulator storage rows have no benchmark counterpart "
                       "(check scenarios and timestamps).", len(s1) - len(pairs), len(s1))

    pairs["power_mw"] = pairs["storage_order_count"] * cfg.ORDER_POWER_MW
    pairs["dp_b"] = pairs["p1_b"] - pairs["p0_b"]
    pairs["dp_s"] = pairs["p1_s"] - pairs["p0_s"]
    return pairs


def build_thresholds(bench: pd.DataFrame) -> pd.DataFrame:
    """theta = mean + k * std of the benchmark no-storage price (per case)."""
    if bench.empty:
        return pd.DataFrame(columns=["case", "mean_price", "std_price", "theta"])
    base = bench.loc[bench["scenario"] == NO].groupby("case")["price"]
    out = pd.DataFrame({
        "mean_price": base.mean(),
        "std_price": base.std(ddof=cfg.SPIKE_STD_DDOF),
    })
    out["theta"] = out["mean_price"] + cfg.SPIKE_N_STD * out["std_price"]
    return out.reset_index()


def build_context(results: dict) -> Context:
    if results["prices"].empty:
        raise ValueError("No benchmark prices available. Run the 'results' stage first.")
    cases = results["cases"].copy()
    bench = build_benchmark_series(results, cases)
    sim = build_simulator_series(results, cases)
    return Context(results=results, cases=cases, bench=bench, sim=sim,
                   pairs=build_pairs(bench, sim), thresholds=build_thresholds(bench))


# =============================================================================
# Small helpers
# =============================================================================
def run_groups(frame: pd.DataFrame, keys: list[str], registry: Registry,
               ctx: Context, only=None) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["n_mtus"] = len(group)
        row.update(registry.compute(group, ctx, only))
        rows.append(row)
    return pd.DataFrame(rows)


def with_case_info(frame: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    if frame.empty:
        return frame
    info = ctx.cases[CASE_INFO]
    out = frame.merge(info, on="case", how="left")
    lead = [c for c in CASE_INFO if c in out.columns]
    return out[lead + [c for c in out.columns if c not in lead]]


def _has_dispatch(group: pd.DataFrame) -> bool:
    return not group[["q_b_mw", "q_s_mw"]].isna().any().any()


def _need(frame, columns: list[str], name: str) -> bool:
    if frame is None or frame.empty:
        return False
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        logger.warning("Table '%s' skipped: required columns missing (%s). "
                       "A metric block may be disabled.", name, missing)
        return False
    return True


# =============================================================================
# Accuracy metric blocks (paired benchmark/simulator rows of one group)
# =============================================================================
@ACCURACY.register("capacity")
def _capacity(g, ctx):
    return {"storage_order_count": int(g["storage_order_count"].iloc[0]),
            "storage_power_mw": float(g["power_mw"].iloc[0])}


@ACCURACY.register("price_level")
def _price_level(g, ctx):
    error = g["p1_s"] - g["p1_b"]
    return {"mae_p1": error.abs().mean(), "bias_p1": error.mean()}


@ACCURACY.register("effect_error")
def _effect_error(g, ctx):
    error = g["dp_s"] - g["dp_b"]
    return {"mae_dp": error.abs().mean(), "bias_dp": error.mean()}


@ACCURACY.register("effect_size")
def _effect_size(g, ctx):
    bench, sim = g["dp_b"].abs().mean(), g["dp_s"].abs().mean()
    return {
        "mean_dp_bench": g["dp_b"].mean(),
        "mean_dp_sim": g["dp_s"].mean(),
        "mean_abs_dp_bench": bench,
        "mean_abs_dp_sim": sim,
        "abs_effect_ratio": sim / bench if bench > 1e-9 else NAN,
    }


@ACCURACY.register("skill_zero_effect")
def _skill(g, ctx):
    """1 - MAE / mean|dP_bench|; undefined if the benchmark effect is ~0."""
    mae = (g["dp_s"] - g["dp_b"]).abs().mean()
    denominator = g["dp_b"].abs().mean()
    return {"skill_vs_zero_effect": 1.0 - mae / denominator if denominator > 1e-9 else NAN}


@ACCURACY.register("active_mtus")
def _active(g, ctx):
    tol = cfg.ANALYSIS_EFFECT_TOL
    active = g["dp_b"].abs() > tol
    error = (g["dp_s"] - g["dp_b"]).abs()
    return {
        "n_active_bench": int(active.sum()),
        "n_active_sim": int((g["dp_s"].abs() > tol).sum()),
        "mae_dp_active": error[active].mean() if active.any() else NAN,
    }


@ACCURACY.register("price_sensitivity")
def _kappa(g, ctx):
    """Dispatch-normalised price impact: sum|dP| / sum|q| in EUR/MWh per MW."""
    if not _has_dispatch(g):
        return {"kappa_bench": NAN, "kappa_sim": NAN, "kappa_ratio": NAN}

    def kappa(dp, q):
        total = q.abs().sum()
        return dp.abs().sum() / total if total > 1e-9 else NAN

    kb, ks = kappa(g["dp_b"], g["q_b_mw"]), kappa(g["dp_s"], g["q_s_mw"])
    return {"kappa_bench": kb, "kappa_sim": ks, "kappa_ratio": ks / kb if kb and kb > 0 else NAN}


@ACCURACY.register("dispatch_matched")
def _dispatch_matched(g, ctx):
    """MAE of dP where both models dispatch alike (within tolerance of installed power)."""
    if not _has_dispatch(g):
        return {"share_dispatch_matched": NAN, "n_matched_active": 0, "mae_dp_dispatch_matched": NAN}

    tol = cfg.ANALYSIS_DISPATCH_TOL_FRACTION * float(g["power_mw"].iloc[0])
    close = (g["q_s_mw"] - g["q_b_mw"]).abs() <= tol
    busy = (g["q_b_mw"].abs() > cfg.ANALYSIS_ACTION_TOL_MW) | (g["q_s_mw"].abs() > cfg.ANALYSIS_ACTION_TOL_MW)
    use = close & busy
    error = (g["dp_s"] - g["dp_b"]).abs()
    return {
        "share_dispatch_matched": close.mean(),
        "n_matched_active": int(use.sum()),
        "mae_dp_dispatch_matched": error[use].mean() if use.any() else NAN,
    }


@ACCURACY.register("dispatch_agreement")
def _dispatch_agreement(g, ctx):
    if not _has_dispatch(g):
        return {"net_mw_mae": NAN, "discharged_mwh_bench": NAN, "discharged_mwh_sim": NAN,
                "discharged_diff_mwh": NAN, "throughput_overlap": NAN}

    qb, qs = g["q_b_mw"], g["q_s_mw"]
    db, cb = qb.clip(lower=0.0), (-qb).clip(lower=0.0)
    ds, cs = qs.clip(lower=0.0), (-qs).clip(lower=0.0)
    throughput = (db + cb).sum()
    overlap = (np.minimum(ds, db).sum() + np.minimum(cs, cb).sum()) / throughput if throughput > 1e-9 else NAN
    e_b, e_s = db.sum() * cfg.MTU_HOURS, ds.sum() * cfg.MTU_HOURS
    return {"net_mw_mae": (qs - qb).abs().mean(), "discharged_mwh_bench": e_b,
            "discharged_mwh_sim": e_s, "discharged_diff_mwh": e_s - e_b,
            "throughput_overlap": overlap}


# =============================================================================
# Spike metric blocks (one price series: benchmark or one simulator method)
# Columns: price, p0 (same source without storage), theta
# =============================================================================
@SPIKES.register("capacity")
def _spike_capacity(g, ctx):
    n = int(g["storage_order_count"].iloc[0])
    return {"storage_order_count": n, "storage_power_mw": n * cfg.ORDER_POWER_MW,
            "spike_threshold": float(g["theta"].iloc[0])}


@SPIKES.register("duration")
def _spike_duration(g, ctx):
    spike = g["price"] >= g["theta"].iloc[0]
    return {"n_spike_mtus": int(spike.sum()), "spike_hours": float(spike.sum()) * cfg.MTU_HOURS}


@SPIKES.register("exceedance")
def _spike_exceedance(g, ctx):
    return {"mean_exceedance": (g["price"] - g["theta"].iloc[0]).clip(lower=0.0).mean()}


@SPIKES.register("peak")
def _spike_peak(g, ctx):
    return {"peak_price": g["price"].max()}


@SPIKES.register("baseline_set")
def _spike_baseline(g, ctx):
    """Effect on the fixed set S0 of no-storage spike MTUs."""
    theta = g["theta"].iloc[0]
    s0 = g["p0"] >= theta
    dp = g["price"] - g["p0"]
    n0 = int(s0.sum())
    return {
        "n_baseline_spikes": n0,
        "mean_dp_baseline_spikes": dp[s0].mean() if n0 else NAN,
        "share_baseline_spikes_resolved": float(((g["price"] < theta) & s0).sum()) / n0 if n0 else NAN,
    }


@SPIKES.register("new_spikes")
def _spike_new(g, ctx):
    theta = g["theta"].iloc[0]
    return {"n_new_spikes": int(((g["price"] >= theta) & (g["p0"] < theta)).sum())}


# =============================================================================
# Tables: conditions and gate
# =============================================================================
@table("case_conditions")
def _case_conditions(ctx: Context) -> pd.DataFrame:
    """Explanatory variables per case (benchmark, no storage)."""
    prices = ctx.results["prices"]
    constraints = ctx.results["constraints"]
    thresholds = ctx.thresholds.set_index("case")
    rows = []

    for case in ctx.cases.itertuples(index=False):
        p0 = prices[(prices["case"] == case.case) & (prices["scenario"] == NO)]
        if p0.empty:
            continue
        row = {"case": case.case}

        zone_price = p0.loc[p0["zone"] == case.curve_zone, "price"]
        row["mean_price"] = zone_price.mean()
        row["std_price"] = zone_price.std(ddof=cfg.SPIKE_STD_DDOF)
        row["spike_threshold"] = thresholds["theta"].get(case.case, NAN)

        by_time = p0.groupby("time")["price"]
        row["mean_zonal_spread"] = (by_time.max() - by_time.min()).mean() if p0["zone"].nunique() > 1 else NAN

        if not constraints.empty:
            c0 = constraints[(constraints["case"] == case.case) & (constraints["scenario"] == NO)]
            if not c0.empty:
                row["share_mtus_with_limit"] = c0.groupby("time")["limited"].any().mean()
                row["share_limited_element_mtus"] = c0["limited"].mean()
                row["mean_limited_per_mtu"] = c0.groupby("time")["limited"].sum().mean()
                ram = c0["ram_mw"].dropna()
                row["share_zero_ram"] = (ram <= cfg.FLOW_TOL_MW).mean() if len(ram) else NAN
        rows.append(row)

    return with_case_info(pd.DataFrame(rows), ctx)


@table("gate_p0")
def _gate_p0(ctx: Context) -> pd.DataFrame:
    """Reproduction of the no-storage price (validity gate, not a result)."""
    if ctx.pairs.empty:
        return pd.DataFrame()
    unique = ctx.pairs.drop_duplicates(["case", "method", "mode", "time"])
    rows = []
    for (case, method, mode), g in unique.groupby(["case", "method", "mode"], sort=False):
        diff = g["p0_s"] - g["p0_b"]
        rows.append({"case": case, "method": method, "mode": mode, "n_mtus": len(g),
                     "mae": diff.abs().mean(), "max_abs": diff.abs().max(), "bias": diff.mean(),
                     "gate_ok": bool(diff.abs().mean() <= cfg.ANALYSIS_GATE_MAE_LIMIT)})
    return with_case_info(pd.DataFrame(rows), ctx)


# =============================================================================
# RQ1
# =============================================================================
@table("rq1_accuracy")
def _rq1_accuracy(ctx: Context) -> pd.DataFrame:
    """Accuracy per case, method, mode and storage scenario."""
    if ctx.pairs.empty:
        return pd.DataFrame()

    out = run_groups(ctx.pairs, ["case", "method", "mode", "scenario"], ACCURACY, ctx)

    bad = out[out["n_mtus"] != cfg.EXPECTED_MTUS]
    if not bad.empty:
        logger.warning("%s accuracy groups do not have %s MTUs.", len(bad), cfg.EXPECTED_MTUS)

    cond = ctx.tables.get("case_conditions")
    if cond is not None and not cond.empty:
        keep = [c for c in ("case", "mean_zonal_spread", "share_mtus_with_limit",
                            "share_limited_element_mtus") if c in cond.columns]
        out = out.merge(cond[keep], on="case", how="left")
    return with_case_info(out, ctx)


@table("rq1_by_congestion")
def _rq1_by_congestion(ctx: Context) -> pd.DataFrame:
    """Core accuracy metrics split by the no-storage congestion state of the MTU."""
    constraints = ctx.results["constraints"]
    if ctx.pairs.empty or constraints.empty:
        return pd.DataFrame()

    state = (constraints[constraints["scenario"] == NO]
             .groupby(["case", "time"])["limited"].any().rename("congested").reset_index())
    pairs = ctx.pairs.merge(state, on=["case", "time"], how="inner")
    if pairs.empty:
        return pd.DataFrame()

    out = run_groups(pairs, ["case", "method", "mode", "scenario", "congested"],
                     ACCURACY, ctx, only=cfg.ANALYSIS_CONGESTION_METRICS)
    return with_case_info(out, ctx)


# =============================================================================
# RQ2
# =============================================================================
@table("rq2_methods")
def _rq2_methods(ctx: Context) -> pd.DataFrame:
    """Effect size and error of each method relative to the shift method."""
    acc = ctx.tables.get("rq1_accuracy")
    need = ["mean_abs_dp_sim", "mean_abs_dp_bench", "mae_dp"]
    if not _need(acc, need, "rq2_methods"):
        return pd.DataFrame()

    keys = ["case", "mode", "scenario"]
    ref = acc.loc[acc["method"] == SHIFT, keys + ["mean_abs_dp_sim", "mae_dp"]].rename(
        columns={"mean_abs_dp_sim": "mean_abs_dp_shift", "mae_dp": "mae_dp_shift"})
    out = acc.merge(ref, on=keys, how="left")

    out["attenuation_vs_shift"] = 1.0 - out["mean_abs_dp_sim"] / out["mean_abs_dp_shift"].where(out["mean_abs_dp_shift"] > 0)
    out["skill_vs_shift"] = 1.0 - out["mae_dp"] / out["mae_dp_shift"].where(out["mae_dp_shift"] > 0)

    columns = CASE_INFO + keys + ["method", "storage_power_mw", "mean_abs_dp_bench", "mean_abs_dp_sim",
                                  "abs_effect_ratio", "attenuation_vs_shift", "mae_dp", "skill_vs_shift",
                                  "kappa_bench", "kappa_sim"]
    return out[[c for c in columns if c in out.columns]]


@table("rq2_decomposition")
def _rq2_decomposition(ctx: Context) -> pd.DataFrame:
    """2x2 decomposition (blocks x coupling) of effect size and error."""
    acc = ctx.tables.get("rq1_accuracy")
    quantities = {"mean_abs_dp_sim": "mean_abs_effect", "mae_dp": "mae_dp"}
    if not _need(acc, list(quantities), "rq2_decomposition"):
        return pd.DataFrame()

    rows = []
    for (case, mode, scenario), g in acc.groupby(["case", "mode", "scenario"], sort=False):
        by_method = g.set_index("method")
        for column, label in quantities.items():
            v = {m: float(by_method[column].get(m, NAN)) for m in (SHIFT, BLOCKS, COUPLING, BOTH)}
            blocks_effect = v[BLOCKS] - v[SHIFT]
            coupling_effect = v[COUPLING] - v[SHIFT]
            joint = v[BOTH] - v[SHIFT]
            rows.append({
                "case": case, "mode": mode, "scenario": scenario, "quantity": label,
                "shift": v[SHIFT], "blocks": v[BLOCKS], "coupling": v[COUPLING], "blocks_coupling": v[BOTH],
                "blocks_effect": blocks_effect, "coupling_effect": coupling_effect,
                "joint_effect": joint, "interaction": joint - blocks_effect - coupling_effect,
            })
    return with_case_info(pd.DataFrame(rows), ctx)


@table("bench_block_changes")
def _bench_block_changes(ctx: Context) -> pd.DataFrame:
    """Benchmark mechanism: change of block-bid acceptance vs. no storage."""
    blocks = ctx.results["block_acceptance"]
    if blocks.empty:
        return pd.DataFrame()

    base = blocks.loc[blocks["scenario"] == NO, ["case", "bid_id", "acceptance_ratio", "active"]].rename(
        columns={"acceptance_ratio": "ratio_0", "active": "active_0"})
    current = blocks[blocks["scenario"] != NO].merge(
        base, on=["case", "bid_id"], how="left", validate="many_to_one")
    current["ratio_0"] = current["ratio_0"].fillna(0.0)
    current["active_0"] = current["active_0"].fillna(False).astype(bool)
    current["active"] = current["active"].astype(bool)
    current["change"] = current["acceptance_ratio"] - current["ratio_0"]

    rows = []
    for (case, scenario), g in current.groupby(["case", "scenario"], sort=False):
        rows.append({
            "case": case, "scenario": scenario,
            "storage_order_count": int(g["storage_order_count"].iloc[0]),
            "n_block_bids": len(g),
            "n_changed": int((g["change"].abs() > cfg.BLOCK_ACCEPTANCE_TOL).sum()),
            "n_activated": int((~g["active_0"] & g["active"]).sum()),
            "n_deactivated": int((g["active_0"] & ~g["active"]).sum()),
            "total_abs_change": float(g["change"].abs().sum()),
        })
    return with_case_info(pd.DataFrame(rows), ctx)


@table("bench_spillover")
def _bench_spillover(ctx: Context) -> pd.DataFrame:
    """Benchmark mechanism: price and net-position response by zone."""
    prices = ctx.results["prices"]
    if prices.empty:
        return pd.DataFrame()

    p0 = prices.loc[prices["scenario"] == NO, ["case", "zone", "time", "price", "net_position_mw"]].rename(
        columns={"price": "p0", "net_position_mw": "np0"})
    p1 = prices[prices["scenario"] != NO]
    m = p1.merge(p0, on=["case", "zone", "time"], how="inner", validate="many_to_one")
    m["dp"] = m["price"] - m["p0"]
    m["dnp"] = m["net_position_mw"] - m["np0"]

    out = (m.groupby(["case", "scenario", "storage_order_count", "zone"], sort=False)
             .agg(mean_dp=("dp", "mean"),
                  mean_abs_dp=("dp", lambda s: s.abs().mean()),
                  mean_abs_dnp_mw=("dnp", lambda s: s.abs().mean()))
             .reset_index())

    storage_zone = ctx.cases.set_index("case")["storage_zone"]
    out["storage_zone"] = out["case"].map(storage_zone)
    reference = (out[out["zone"] == out["storage_zone"]]
                 .set_index(["case", "scenario"])["mean_abs_dp"].rename("ref"))
    out = out.merge(reference.reset_index(), on=["case", "scenario"], how="left")
    out["relative_to_storage_zone"] = out["mean_abs_dp"] / out["ref"].where(out["ref"] > 0)
    return with_case_info(out.drop(columns="ref"), ctx)


# =============================================================================
# RQ3
# =============================================================================
def _spike_frame(ctx: Context) -> pd.DataFrame:
    parts = []
    if not ctx.bench.empty:
        b = ctx.bench[["case", "scenario", "storage_order_count", "time", "price"]].copy()
        b["source"], b["mode"] = "benchmark", "benchmark"
        parts.append(b)
    if not ctx.sim.empty:
        s = ctx.sim[["case", "method", "mode", "scenario", "storage_order_count", "time", "price"]].copy()
        parts.append(s.rename(columns={"method": "source"}))
    if not parts:
        return pd.DataFrame()

    series = pd.concat(parts, ignore_index=True)
    p0 = series.loc[series["scenario"] == NO, ["case", "source", "mode", "time", "price"]].rename(
        columns={"price": "p0"})
    series = series.merge(p0, on=["case", "source", "mode", "time"], how="left", validate="many_to_one")
    return series.merge(ctx.thresholds[["case", "theta"]], on="case", how="left")


@table("rq3_spikes")
def _rq3_spikes(ctx: Context) -> pd.DataFrame:
    """Spike frequency and magnitude per case, source (benchmark/method) and capacity."""
    series = _spike_frame(ctx)
    if series.empty:
        return pd.DataFrame()
    out = run_groups(series, ["case", "source", "mode", "scenario"], SPIKES, ctx)
    if "storage_order_count" in out.columns:
        out = out.sort_values(["case", "source", "mode", "storage_order_count"]).reset_index(drop=True)
    return with_case_info(out, ctx)


@table("rq3_response")
def _rq3_response(ctx: Context) -> pd.DataFrame:
    """Change of spike duration and exceedance vs. no storage, and marginal effect per order."""
    spikes = ctx.tables.get("rq3_spikes")
    if not _need(spikes, ["spike_hours", "mean_exceedance", "storage_order_count"], "rq3_response"):
        return pd.DataFrame()

    parts = []
    for _, g in spikes.groupby(["case", "source", "mode"], sort=False):
        g = g.sort_values("storage_order_count").copy()
        base = g[g["storage_order_count"] == 0]
        if base.empty:
            continue
        h0, x0 = float(base["spike_hours"].iloc[0]), float(base["mean_exceedance"].iloc[0])

        g["spike_hours_change"] = g["spike_hours"] - h0
        g["spike_hours_rel_change"] = g["spike_hours_change"] / h0 if h0 > 0 else NAN
        g["exceedance_change"] = g["mean_exceedance"] - x0
        g["exceedance_rel_change"] = g["exceedance_change"] / x0 if x0 > 0 else NAN

        orders = g["storage_order_count"].diff()
        g["marginal_spike_hours_per_order"] = g["spike_hours"].diff() / orders
        g["marginal_exceedance_per_order"] = g["mean_exceedance"].diff() / orders
        parts.append(g)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@table("rq3_comparison")
def _rq3_comparison(ctx: Context) -> pd.DataFrame:
    """Simulator vs. benchmark spike response."""
    response = ctx.tables.get("rq3_response")
    quantities = ["spike_hours", "mean_exceedance", "peak_price",
                  "spike_hours_change", "exceedance_change",
                  "spike_hours_rel_change", "exceedance_rel_change"]
    if not _need(response, quantities, "rq3_comparison"):
        return pd.DataFrame()

    bench = response.loc[response["source"] == "benchmark", ["case", "scenario"] + quantities]
    sim = response.loc[response["source"] != "benchmark",
                       ["case", "source", "mode", "scenario", "storage_order_count"] + quantities]
    if bench.empty or sim.empty:
        return pd.DataFrame()

    out = sim.merge(bench, on=["case", "scenario"], how="inner", suffixes=("_sim", "_bench"))
    for q in quantities:
        out[f"{q}_diff"] = out[f"{q}_sim"] - out[f"{q}_bench"]
    out = out.rename(columns={"source": "method"})
    return with_case_info(out.sort_values(["case", "method", "storage_order_count"]), ctx)


# =============================================================================
# Entry points
# =============================================================================
def run_analysis(results: dict | None = None, only=None) -> dict[str, pd.DataFrame]:
    """Calculate all enabled tables. `results=None` reads the standardised CSV files."""
    results = results if results is not None else read_standardised()
    ctx = build_context(results)

    for name, function in TABLES.items():
        if name in cfg.ANALYSIS_DISABLED_TABLES or (only is not None and name not in only):
            continue
        logger.info("Analysis table '%s'.", name)
        ctx.tables[name] = function(ctx)
    return ctx.tables


def write_analysis(tables: dict[str, pd.DataFrame], directory: Path | None = None) -> Path:
    directory = Path(directory or cfg.ANALYSIS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        path = directory / f"{name}.csv"
        if frame is None or frame.empty:
            if path.exists():
                path.unlink()               # no stale tables
            continue
        frame.to_csv(path, index=False)
        logger.info("Wrote %s (%s rows).", path, len(frame))
    return directory