"""
Central configuration: paths, shared parameters, days and experiment cases.

All other modules read their settings from here. A *case* is one benchmark
market configuration (e.g. fbmc, ntc, lines_0p75) on one delivery day, with
the storage counts to clear and the simulator inputs to prepare.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "output"

NETWORK_FILE = DATA_DIR / "filtered_network.nc"
PRICES_CSV = DATA_DIR / "auction_spot_15_prices_germany_luxembourg_2026.csv"
NET_POSITIONS_CSV = (
    DATA_DIR / "GUI_IMPLICIT_ALLOCATIONS_NET_POSITIONS_202512312300-202612312300.csv"
)

COUPLING_DIR = OUTPUT_ROOT / "coupling_regimes"
LINE_SCALING_DIR = OUTPUT_ROOT / "line_scaling"

# Standardised tables written by the "results" stage (input for analysis.py).
STANDARDISED_DIR = OUTPUT_ROOT / "standardised"

# Price simulator (run manually).
# SIMULATOR_INPUT_ROOT: staged input, one sub-folder per case. Point it to
#   <price_simulation CSV_INPUT_ROOT>/scenarios if the simulator should read
#   it directly.
# SIMULATOR_OUTPUT_ROOT: where the simulator writes
#   <method>/<regime>/<mode>/sim_*.csv
SIMULATOR_INPUT_ROOT = OUTPUT_ROOT / "simulator_input"
SIMULATOR_OUTPUT_ROOT = OUTPUT_ROOT / "price_simulator"

# Runner script of the price simulator (same Python environment as the benchmark).
SIMULATOR_RUNNER = Path("Path to price simulator")

# File-name conventions expected by the simulator. Must match
# price_simulation.config (CSV_CURVES_GLOB_15MIN, COUPLING_CSV_FILE_GLOB).
SIM_CURVE_FILE_TEMPLATE = "auction_15_aggregated_curves_germany_luxembourg_{date}.csv"
SIM_COUPLING_FILE_TEMPLATE = "coupling_inputs_{date}.csv"
SIM_BLOCK_FILE_PREFIX = "all_exchanges_bbof_15"
SIM_KAPPA_BY_ZONE = {"Z2": 1.0, "Z3": 1.0}   # neighbour size factors (coupling)

# =============================================================================
# SHARED MODEL PARAMETERS
# =============================================================================
ZONE_COLUMN = "zone"          # bus column holding the bidding-zone identifier
SOURCE_ZONE = "DE-LU"         # zone label of the published order book
SOLVER_NAME = "gurobi"
ALIGN_SNAPSHOTS = True        # reassign network timestamps to the delivery day
CARRIER_ALPHA = 0.5           # TODO decide on a value (as in the original)

# Line capacity adjustments to ensure binding CNECs: {line: factor on s_nom}
LINE_CAPACITY_SCALING = {
    "Line_114": 0.5,
    "Line_187": 0.5,
}

COPPER_PLATE_ZONE = "CP"
STORAGE_ZONE = "Z1"

# One storage order (MWh, MW, EUR/MWh). Single source of truth for the
# benchmark, the simulator specification and the result checks.
ORDER_ENERGY_MWH = 300.0
ORDER_POWER_MW = 150.0
EFFICIENCY = 0.86
CYCLING_COST_EUR_PER_MWH = 10.0
END_OF_DAY_VALUE = 0.0
STARTING_ENERGY = 0.0

STORAGE_CONFIG = {
    "max_energy": ORDER_ENERGY_MWH,
    "min_energy": 0.0,
    "max_input_power": ORDER_POWER_MW,
    "max_output_power": ORDER_POWER_MW,
    "starting_energy": STARTING_ENERGY,
    "efficiency": EFFICIENCY,
    "cycling_cost": CYCLING_COST_EUR_PER_MWH,
    "end_of_day_value": END_OF_DAY_VALUE,
    "bid_id_prefix": "SO",
    "agent_addr": None,       # None -> (f"agent_{zone}", "test")
}

# Conventions / tolerances used when reading results
MTU_HOURS = 0.25
EXPECTED_MTUS = 96
FLOW_TOL_MW = 1e-3
BLOCK_ACCEPTANCE_TOL = 1e-6

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
NO_STORAGE = "storage_orders_0"

BASELINE_STORAGE_COUNTS = (0, 1, 2, 3, 4)
COUPLING_STORAGE_COUNTS = (0, 1)
LINE_SCALING_STORAGE_COUNTS = (0, 1)

COUPLING_REGIMES = ("ntc", "isolated", "copper_plate")

LINE_SCALING_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
LINE_SCALING_SCOPE = "all"               # "all" or "cross_border"
APPLY_BASELINE_LINE_SCALING = True       # factor multiplies the baseline scaling
# Passed to precompute_fbmc_parameters (needs the edit in euphemia_like.py).
LINE_SCALING_FBMC_KWARGS = {"n1_slack_penalty": None}
LINE_SCALING_CALCULATE_MAXBEX = True
BASELINE_PRICE_TOL = 0.011               # EUR/MWh; baseline CSV is rounded to 0.01

# Simulator methods: key = scenario key of price_simulation, label = method
# name in the result files.
SIMULATOR_METHODS = {
    "base":  {"label": "shift",           "coupling": False, "blocks": False},
    "cb":    {"label": "blocks",          "coupling": False, "blocks": True},
    "mc":    {"label": "coupling",        "coupling": True,  "blocks": False},
    "mc_cb": {"label": "blocks_coupling", "coupling": True,  "blocks": True},
}
SIM_ALL_METHODS = tuple(SIMULATOR_METHODS)
SIM_NO_COUPLING_METHODS = ("base", "cb")


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

ANALYSIS_EFFECT_TOL = 1.0              # EUR/MWh, |dP| above this = "active" MTU
ANALYSIS_DISPATCH_TOL_FRACTION = 0.10  # dispatch "matched" within 10 % of installed power
ANALYSIS_ACTION_TOL_MW = 1e-3          # |net dispatch| above this = storage active
ANALYSIS_GATE_MAE_LIMIT = 1.0          # EUR/MWh, p0 reproduction gate

SPIKE_N_STD = 2.0                      # theta = mean + SPIKE_N_STD * std of the no-storage price
SPIKE_STD_DDOF = 0                     # 0 = population std

# Drop metric blocks or whole tables without touching the code.
ANALYSIS_DISABLED_METRICS: set[str] = set()   # e.g. {"dispatch_agreement"}
ANALYSIS_DISABLED_TABLES: set[str] = set()    # e.g. {"rq1_by_congestion"}

# Metric blocks used in the congestion-split table.
ANALYSIS_CONGESTION_METRICS = ("price_level", "effect_error", "effect_size")


# =============================================================================
# REPORTING SETTINGS
# =============================================================================
REPORT_DIR = OUTPUT_ROOT / "report"
REPORT_DPI = 200

# Simulator series shown in the reports: mode, and the first available method
# of the preference list (per case).
REPORT_SIM_MODE = "own_dispatch"
REPORT_SIM_METHOD_PREFERENCE = ("blocks_coupling", "blocks", "coupling", "shift")
REPORT_REQUIRE_SIMULATOR = False       # True: skip scenarios without simulator results

# Drop plots or tables without touching the code.
REPORT_DISABLED_PLOTS: set[str] = set()     # e.g. {"bess_dispatch"}
REPORT_DISABLED_TABLES: set[str] = set()


# =============================================================================
# DAYS
# =============================================================================
@dataclass(frozen=True)
class DayConfig:
    """Published order books and output location of one delivery day."""

    name: str
    simple_orders: Path
    block_orders: Path
    output_dir: Path
    delivery_date: datetime | None      # None = infer from the simple orders

    @property
    def intermediate_dir(self) -> Path:
        return self.output_dir / "intermediate"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def preprocessed_simple(self) -> Path:
        return self.intermediate_dir / "preprocessed_simple_orders.csv"

    @property
    def preprocessed_blocks(self) -> Path:
        return self.intermediate_dir / "preprocessed_block_orders.csv"

    @property
    def zonal_simple(self) -> Path:
        return self.intermediate_dir / "zonal_simple_orders.csv"

    @property
    def zonal_blocks(self) -> Path:
        return self.intermediate_dir / "zonal_block_orders.csv"


DAYS: dict[str, DayConfig] = {
    "baseline": DayConfig(
        name="baseline",
        simple_orders=DATA_DIR / "auction_15_aggregated_curves_germany_luxembourg_20260120.csv",
        block_orders=DATA_DIR / "all_exchanges_bbof_15_germany_luxembourg_20260120.csv",
        output_dir=OUTPUT_ROOT / "baseline",
        delivery_date=datetime(2026, 1, 20),
    ),
    # TODO: change to a historical day with an actual Dunkelflaute.
    "dunkelflaute": DayConfig(
        name="dunkelflaute",
        simple_orders=DATA_DIR / "auction_15_aggregated_curves_germany_luxembourg_20261029.csv",
        block_orders=DATA_DIR / "all_exchanges_bbof_15_germany_luxembourg_20261029.csv",
        output_dir=OUTPUT_ROOT / "dunkelflaute",
        delivery_date=datetime(2026, 10, 29),
    ),
}


# =============================================================================
# CASES
# =============================================================================
def scenario_label(count: int) -> str:
    return f"storage_orders_{count}"


def factor_label(factor: float) -> str:
    return f"lines_{factor:.2f}".replace(".", "p")


@dataclass(frozen=True)
class Case:
    """One benchmark market configuration on one delivery day."""

    name: str                       # also the simulator "regime" name
    family: str                     # baseline | coupling | line_scaling | dunkelflaute
    runner: str                     # baseline | coupling | line_scaling
    market: str                     # fbmc | ntc | isolated | copper_plate
    day: str                        # key in DAYS
    results_dir: Path               # contains storage_orders_<n>/
    storage_counts: tuple[int, ...]
    storage_zone: str = STORAGE_ZONE      # zone the storage orders are placed in
    curve_zone: str = STORAGE_ZONE        # benchmark zone simulated by the simulator
    report_zone: str = STORAGE_ZONE       # zone label in simulator results
    line_factor: float | None = None
    fbmc_kwargs: Mapping | None = None
    maxbex_csv: Path | None = None        # NTC source and simulator coupling input
    sim_methods: tuple[str, ...] = SIM_ALL_METHODS
    sim_storage_counts: tuple[int, ...] = (1,)

    @property
    def day_config(self) -> DayConfig:
        return DAYS[self.day]

    @property
    def no_storage_dir(self) -> Path:
        return self.results_dir / NO_STORAGE

    @property
    def simulator_input_dir(self) -> Path:
        return SIMULATOR_INPUT_ROOT / self.name

    @property
    def sim_needs_blocks(self) -> bool:
        return any(SIMULATOR_METHODS[m]["blocks"] for m in self.sim_methods)

    @property
    def sim_needs_coupling(self) -> bool:
        return any(SIMULATOR_METHODS[m]["coupling"] for m in self.sim_methods)


def _build_cases() -> list[Case]:
    baseline_day = DAYS["baseline"]
    baseline_maxbex = baseline_day.results_dir / NO_STORAGE / "maxbex.csv"
    cases: list[Case] = []

    # Baseline FBMC with several BESS capacities
    cases.append(Case(
        name="fbmc", family="baseline", runner="baseline", market="fbmc",
        day="baseline", results_dir=baseline_day.results_dir,
        storage_counts=BASELINE_STORAGE_COUNTS, maxbex_csv=baseline_maxbex,
        sim_storage_counts=(1, 2, 3, 4),
    ))

    # Coupling regimes (NTC derived from the baseline MaxBEX)
    for regime in COUPLING_REGIMES:
        is_cp = regime == "copper_plate"
        cases.append(Case(
            name=regime, family="coupling", runner="coupling", market=regime,
            day="baseline", results_dir=COUPLING_DIR / regime,
            storage_counts=COUPLING_STORAGE_COUNTS,
            storage_zone=COPPER_PLATE_ZONE if is_cp else STORAGE_ZONE,
            curve_zone=COPPER_PLATE_ZONE if is_cp else STORAGE_ZONE,
            report_zone=STORAGE_ZONE,
            maxbex_csv=baseline_maxbex if regime == "ntc" else None,
            sim_methods=(SIM_ALL_METHODS if regime == "ntc"
                         else SIM_NO_COUPLING_METHODS),
        ))

    # Line scaling (FBMC)
    for factor in LINE_SCALING_FACTORS:
        label = factor_label(factor)
        run_dir = LINE_SCALING_DIR / label
        cases.append(Case(
            name=label, family="line_scaling", runner="line_scaling",
            market="fbmc", day="baseline", results_dir=run_dir,
            storage_counts=LINE_SCALING_STORAGE_COUNTS, line_factor=factor,
            fbmc_kwargs=LINE_SCALING_FBMC_KWARGS,
            maxbex_csv=run_dir / NO_STORAGE / "maxbex.csv",
        ))

    # Dunkelflaute day: baseline-type run on a different delivery day
    dunkel = DAYS["dunkelflaute"]
    cases.append(Case(
        name="dunkelflaute", family="dunkelflaute", runner="baseline",
        market="fbmc", day="dunkelflaute", results_dir=dunkel.results_dir,
        storage_counts=BASELINE_STORAGE_COUNTS,
        maxbex_csv=dunkel.results_dir / NO_STORAGE / "maxbex.csv",
        sim_storage_counts=(1, 2, 3, 4),
    ))

    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate case names: {names}")
    return cases


CASES: list[Case] = _build_cases()
CASES_BY_NAME: dict[str, Case] = {c.name: c for c in CASES}


def get_case(name: str) -> Case:
    try:
        return CASES_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown case '{name}'. Available: {list(CASES_BY_NAME)}"
        ) from None


def select_cases(families=None, names=None) -> list[Case]:
    """Select cases by family and/or name (None = no restriction)."""
    selected = list(CASES)
    if families is not None:
        unknown = set(families) - {c.family for c in CASES}
        if unknown:
            raise ValueError(f"Unknown families: {sorted(unknown)}")
        selected = [c for c in selected if c.family in families]
    if names is not None:
        for name in names:
            get_case(name)
        selected = [c for c in selected if c.name in names]
    return selected
