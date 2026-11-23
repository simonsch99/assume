"""
Run the experiment workflow. Edit the CONFIGURATION block and run this file.

Stages (each can be switched on/off independently, executed in this order):
    prepare           preprocessing and zonal order books per delivery day
    benchmark         market clearings of the selected cases
    simulator_inputs  stage price-simulator input (config.SIMULATOR_INPUT_ROOT)
    simulator         run the price simulator on the staged input
                      (config.SIMULATOR_RUNNER, same Python environment)
    results           load, standardise and validate results
                      (config.STANDARDISED_DIR), benchmark and simulator
"""

from __future__ import annotations

import logging
import traceback

import pandas as pd

from workflow import analysis, config as cfg, execution, preparation, reporting, results

ALL_STAGES = ("prepare", "benchmark", "simulator_inputs", "simulator",
              "results", "analysis", "reporting")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_experiments")

# =============================================================================
# CONFIGURATION
# =============================================================================
STAGES = ['reporting']  # = ALL_STAGES to run the whole pipeline, ["stage"] to select only one

# Selection (None = all). Families: baseline, coupling, line_scaling, dunkelflaute.
# Case names: fbmc, ntc, isolated, copper_plate, lines_0p50, ..., dunkelflaute.
FAMILIES = ("baseline", "coupling", "line_scaling")            # e.g. ("baseline", "coupling")
CASE_NAMES = None          # e.g. ("fbmc", "ntc")

CONTINUE_ON_ERROR = True
INCLUDE_SIMULATOR_RESULTS = True
FAIL_ON_VALIDATION_ERRORS = False
# =============================================================================


def run_stage(label: str, items, function, failures: list) -> None:
    for item in items:
        name = getattr(item, "name", str(item))
        try:
            function(item)
        except Exception as exc:  # noqa: BLE001
            if not CONTINUE_ON_ERROR:
                raise
            logger.error("[%s] %s failed: %s: %s", label, name, type(exc).__name__, exc)
            traceback.print_exc()
            failures.append(f"{label}:{name}")


def main() -> None:
    loaded = None
    tables = None

    unknown = set(STAGES) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    cases = cfg.select_cases(FAMILIES, CASE_NAMES)
    if not cases:
        raise ValueError("No cases selected.")

    logger.info("Selected cases: %s", ", ".join(c.name for c in cases))
    failures: list[str] = []

    if "prepare" in STAGES:
        days = [cfg.DAYS[d] for d in dict.fromkeys(c.day for c in cases)]
        run_stage("prepare", days, preparation.prepare_day_inputs, failures)

    if "benchmark" in STAGES:
        # Definition order guarantees: baseline before coupling/line scaling.
        run_stage("benchmark", cases, execution.run_case, failures)

    if "simulator_inputs" in STAGES:
        run_stage("simulator_inputs", cases, preparation.stage_simulator_inputs, failures)

    if "simulator" in STAGES:
        run_stage("simulator", cases, execution.run_simulator_case, failures)

    if "results" in STAGES:
        loaded = results.load_all_results(cases, include_simulator=INCLUDE_SIMULATOR_RESULTS)
        results.write_standardised(loaded)

        validation = loaded["validation"]
        pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80)
        if validation.empty:
            print("\nValidation: no findings.")
        else:
            print("\nValidation findings:")
            print(validation.to_string(index=False))
            if FAIL_ON_VALIDATION_ERRORS and (validation["level"] == "error").any():
                failures.append("results:validation")

    if "analysis" in STAGES:
        tables = analysis.run_analysis(loaded)   # None: read the standardised CSV files
        analysis.write_analysis(tables)

    if "reporting" in STAGES:
        reporting.run_reporting(loaded, tables)   # None: read the CSV files
    
    if failures:
        raise SystemExit(f"Failed: {failures}")
    logger.info("Done.")


if __name__ == "__main__":
    main()