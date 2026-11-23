"""
Reporting: figures and tables for the thesis.

Framework
    * Plots   functions registered with @plot("name"). Each receives the
              ReportContext and saves its figures with ctx.save(...).
    * Tables  functions registered with @report_table("name"). Each returns a
              DataFrame, written to <REPORT_DIR>/tables/<name>.csv.
    Plots and tables can be disabled in config.REPORT_DISABLED_PLOTS /
    config.REPORT_DISABLED_TABLES.

Add a plot: write one function with @plot("my_plot") and use the helpers of
ReportContext (storage_scenarios, benchmark, simulator, save). Nothing else
has to be changed.

Input: standardised tables from results.py (and, for later tables, the
analysis tables from analysis.py). Series are taken from the storage zone of
each case, using the same definitions as analysis.py.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .analysis import build_benchmark_series, build_simulator_series
from .results import read_standardised

logger = logging.getLogger(__name__)

NO = cfg.NO_STORAGE

PLOTS: dict[str, Callable] = {}
REPORT_TABLES: dict[str, Callable] = {}


def plot(name: str):
    def decorator(function: Callable) -> Callable:
        if name in PLOTS:
            raise ValueError(f"Plot '{name}' already registered.")
        PLOTS[name] = function
        return function
    return decorator


def report_table(name: str):
    def decorator(function: Callable) -> Callable:
        if name in REPORT_TABLES:
            raise ValueError(f"Report table '{name}' already registered.")
        REPORT_TABLES[name] = function
        return function
    return decorator


# =============================================================================
# Context
# =============================================================================
@dataclass
class ReportContext:
    results: dict
    analysis: dict                 # analysis tables (may be empty)
    cases: pd.DataFrame
    bench: pd.DataFrame            # benchmark storage-zone series (price, q_mw)
    sim: pd.DataFrame              # simulator storage-zone series (price, q_mw)
    out_dir: Path
    written: list[Path] = field(default_factory=list)

    # ---- data access -------------------------------------------------------
    def storage_scenarios(self) -> Iterator[tuple[object, str]]:
        """Yield (case row, scenario) for every benchmark scenario with storage."""
        for case in self.cases.itertuples(index=False):
            b = self.bench[self.bench["case"] == case.case]
            scenarios = sorted(set(b.loc[b["scenario"] != NO, "scenario"]),
                               key=lambda s: int(s.rsplit("_", 1)[1]))
            for scenario in scenarios:
                yield case, scenario

    def benchmark(self, case, scenario: str) -> pd.DataFrame:
        b = self.bench[(self.bench["case"] == case.case) & (self.bench["scenario"] == scenario)]
        return b.sort_values("time")

    def simulator(self, case, scenario: str) -> tuple[str | None, pd.DataFrame]:
        """Simulator series of the preferred available method, or (None, empty)."""
        if self.sim.empty:
            return None, pd.DataFrame()
        s = self.sim[(self.sim["case"] == case.case)
                     & (self.sim["scenario"] == scenario)
                     & (self.sim["mode"] == cfg.REPORT_SIM_MODE)]
        available = set(s["method"])
        for method in cfg.REPORT_SIM_METHOD_PREFERENCE:
            if method in available:
                return method, s[s["method"] == method].sort_values("time")
        return None, pd.DataFrame()

    # ---- output ------------------------------------------------------------
    def save(self, figure, subdir: str, filename: str) -> Path:
        import matplotlib.pyplot as plt

        folder = self.out_dir / subdir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / _safe(filename)
        figure.savefig(path, dpi=cfg.REPORT_DPI)
        plt.close(figure)
        self.written.append(path)
        return path


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_context(results: dict, analysis_tables: dict | None = None) -> ReportContext:
    if results["prices"].empty:
        raise ValueError("No benchmark prices available. Run the 'results' stage first.")
    cases = results["cases"].copy()
    return ReportContext(
        results=results,
        analysis=analysis_tables or {},
        cases=cases,
        bench=build_benchmark_series(results, cases),
        sim=build_simulator_series(results, cases),
        out_dir=Path(cfg.REPORT_DIR),
    )


# =============================================================================
# Plot helpers
# =============================================================================
def _step_xy(frame: pd.DataFrame, column: str):
    """Step data with the last MTU extended to its end (value valid for 15 min)."""
    times = frame["time"].tolist()
    values = frame[column].tolist()
    times.append(times[-1] + pd.Timedelta(hours=cfg.MTU_HOURS))
    values.append(values[-1])
    return times, values


def _format_time_axis(axis, frame: pd.DataFrame) -> None:
    import matplotlib.dates as mdates

    axis.set_xlim(frame["time"].min(), frame["time"].max() + pd.Timedelta(hours=cfg.MTU_HOURS))
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(alpha=0.3)


def _note(axis, text: str) -> None:
    axis.text(0.01, 0.96, text, transform=axis.transAxes, va="top",
              bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})


# =============================================================================
# Plots
# =============================================================================
@plot("mcp_prices")
def plot_mcp_prices(ctx: ReportContext) -> None:
    """MCP of the storage zone: benchmark without/with storage and simulator."""
    import matplotlib.pyplot as plt

    for case, scenario in ctx.storage_scenarios():
        method, sim = ctx.simulator(case, scenario)
        if sim.empty and cfg.REPORT_REQUIRE_SIMULATOR:
            continue

        b0 = ctx.bench[(ctx.bench["case"] == case.case) & (ctx.bench["scenario"] == NO)].sort_values("time")
        b1 = ctx.benchmark(case, scenario)
        if b0.empty or b1.empty:
            continue

        figure, axis = plt.subplots(figsize=(11, 4.8))

        axis.step(*_step_xy(b0, "price"), where="post", color="black", linestyle="--",
                  linewidth=1.5, label="Benchmark: no storage")
        axis.step(*_step_xy(b1, "price"), where="post", color="tab:blue",
                  linewidth=1.8, label="Benchmark: storage")

        if not sim.empty:
            axis.step(*_step_xy(sim, "price"), where="post", color="tab:red",
                      linewidth=1.4, label=f"Simulator: {method}, storage")
        else:
            _note(axis, "Simulator result unavailable")

        axis.set_title(f"MCP in {case.curve_zone}: {case.case}, {scenario}")
        axis.set_xlabel("Time of day")
        axis.set_ylabel("Price [EUR/MWh]")
        axis.legend()
        _format_time_axis(axis, b1)
        figure.tight_layout()

        ctx.save(figure, "mcp_prices", f"mcp_{case.case}_{scenario}.png")


@plot("bess_dispatch")
def plot_bess_dispatch(ctx: ReportContext) -> None:
    """Net BESS dispatch (discharge positive, charge negative): benchmark and simulator."""
    import matplotlib.pyplot as plt

    width = cfg.MTU_HOURS / 24.0      # matplotlib date units (days)

    for case, scenario in ctx.storage_scenarios():
        method, sim = ctx.simulator(case, scenario)
        if sim.empty and cfg.REPORT_REQUIRE_SIMULATOR:
            continue

        bench = ctx.benchmark(case, scenario)
        if bench.empty:
            continue

        power = int(bench["storage_order_count"].iloc[0]) * cfg.ORDER_POWER_MW
        panels = [("Benchmark", bench),
                  (f"Simulator: {method}" if method else "Simulator", sim)]

        figure, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 6.5))
        for axis, (label, frame) in zip(axes, panels):
            if frame.empty or frame["q_mw"].isna().all():
                _note(axis, "Dispatch unavailable")
            else:
                q = frame["q_mw"].fillna(0.0)
                discharge, charge = q.clip(lower=0.0), q.clip(upper=0.0)
                axis.bar(frame["time"], discharge, width=width, align="edge",
                         color="tab:green", label="Discharge")
                axis.bar(frame["time"], charge, width=width, align="edge",
                         color="tab:red", label="Charge")
                axis.legend(loc="upper right", fontsize=8)

            axis.axhline(power, color="grey", linestyle=":", linewidth=1)
            axis.axhline(-power, color="grey", linestyle=":", linewidth=1)
            axis.axhline(0.0, color="black", linewidth=0.6)
            axis.set_title(label, loc="left", fontsize=9)
            axis.set_ylabel("Net dispatch [MW]")
            _format_time_axis(axis, bench)

        axes[-1].set_xlabel("Time of day")
        figure.suptitle(f"BESS dispatch: {case.case}, {scenario} ({power:.0f} MW)")
        figure.tight_layout()

        ctx.save(figure, "bess_dispatch", f"dispatch_{case.case}_{scenario}.png")


# =============================================================================
# Tables (none yet). Example:
#
# @report_table("rq1_summary")
# def rq1_summary(ctx: ReportContext) -> pd.DataFrame:
#     table = ctx.analysis.get("rq1_accuracy", pd.DataFrame())
#     return table[["case", "method", "scenario", "mae_p1", "mae_dp"]]
# =============================================================================


# =============================================================================
# Entry point
# =============================================================================
def run_reporting(
    results: dict | None = None,
    analysis_tables: dict | None = None,
    only=None,
) -> list[Path]:
    """
    Create all enabled plots and tables. Returns the written files.

    results=None reads the standardised CSV files. analysis_tables=None reads
    the CSV files of the analysis stage if they exist.
    """
    import matplotlib

    matplotlib.use("Agg")

    results = results if results is not None else read_standardised()

    if analysis_tables is None:
        analysis_tables = {}
        folder = Path(cfg.ANALYSIS_DIR)
        if folder.exists():
            analysis_tables = {p.stem: pd.read_csv(p) for p in folder.glob("*.csv")}

    ctx = build_context(results, analysis_tables)

    for name, function in PLOTS.items():
        if name in cfg.REPORT_DISABLED_PLOTS or (only is not None and name not in only):
            continue
        logger.info("Report plot '%s'.", name)
        function(ctx)

    for name, function in REPORT_TABLES.items():
        if name in cfg.REPORT_DISABLED_TABLES or (only is not None and name not in only):
            continue
        logger.info("Report table '%s'.", name)
        frame = function(ctx)
        if frame is None or frame.empty:
            continue
        folder = ctx.out_dir / "tables"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_safe(name)}.csv"
        frame.to_csv(path, index=False)
        ctx.written.append(path)

    logger.info("Reporting finished: %s files written to %s.", len(ctx.written), ctx.out_dir)
    return ctx.written