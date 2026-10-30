import pandas as pd
import pypsa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


folder = Path("S:/23502/5/500_ESA/ESW_51520/Aktuell/80_Masterarbeiten/Schiebel_Simon/03_data/test_network")

# Read the filtered CSV files
bus = pd.read_csv(folder / "df_bus_filtered.csv")
branch = pd.read_csv(folder / "df_branch_filtered.csv")
gen = pd.read_csv(folder / "df_gen_filtered.csv", sep=";")
load = pd.read_csv(folder / "df_load_filtered_15min.csv")

# Use BusID values consistently as strings
bus["BusID"] = bus["BusID"].astype(str)
branch[["FromBus", "ToBus"]] = branch[["FromBus", "ToBus"]].astype(str)
gen["OnBus"] = gen["OnBus"].astype(str)
load.columns = load.columns.astype(str)

network = pypsa.Network()
network.set_snapshots(load.index)

# Add buses with geographic coordinates
for _, row in bus.iterrows():
    network.add(
        "Bus",
        row["BusID"],
        x=row["Longitude"],
        y=row["Latitude"],
        zone=row["Zone"]
    )

# Add transmission lines
for _, row in branch.iterrows():
    network.add(
        "Line",
        f"Line_{row['BranchID']}",
        bus0=row["FromBus"],
        bus1=row["ToBus"],
        r=row["R"],
        x=row["X"],
        s_nom=row["Pmax"]
    )

# Add generators
for _, row in gen.iterrows():
    network.add(
        "Generator",
        f"Generator_{row['GenID']}",
        bus=row["OnBus"],
        carrier=row["Type"],
        p_nom=row["Pmax"],
        p_min_pu=row["Pmin"] / row["Pmax"],
        marginal_cost=row["Costs"]
    )

# Add one load per bus and assign its time series
for bus_id in network.buses.index:
    if bus_id in load.columns:
        network.add("Load", f"Load_{bus_id}", bus=bus_id)
        network.loads_t.p_set[f"Load_{bus_id}"] = load[bus_id].values

# Target: must-run floor must never exceed this fraction of minimum load.
MUST_RUN_SAFETY_MARGIN = 0.85


def rescale_must_run_floor(
    network: pypsa.Network,
    safety_margin: float = MUST_RUN_SAFETY_MARGIN,
) -> None:
    """
    Proportionally scale down generator p_min_pu to guarantee D-2 feasibility.

    Must be called AFTER snapshots are aligned and load is correctly injected
    (network.loads_t.p_set populated), and BEFORE run_clearing() /
    precompute_fbmc_parameters().
    """
    if not 0.0 < safety_margin <= 1.0:
        raise ValueError("safety_margin must be in (0, 1].")

    static_load = network.loads["p_set"].fillna(0.0)
    load = pd.DataFrame(
        [static_load.to_numpy()] * len(network.snapshots),
        index=network.snapshots,
        columns=network.loads.index,
    )
    if not network.loads_t.p_set.empty:
        dynamic_load = network.loads_t.p_set.reindex(
            index=network.snapshots, columns=network.loads.index
        )
        load = dynamic_load.fillna(load)
    total_load_per_snapshot = load.sum(axis=1)

    min_load = float(total_load_per_snapshot.min())
    target_floor = safety_margin * min_load

    must_run_mask = network.generators["p_min_pu"].fillna(0.0) > 0
    current_floor = (
        network.generators.loc[must_run_mask, "p_min_pu"]
        * network.generators.loc[must_run_mask, "p_nom"]
    ).sum()

    if current_floor <= target_floor:
        logger.info(
            "Must-run floor (%.1f MW) already fits within %.0f%% of "
            "minimum load (%.1f MW); no rescaling needed.",
            current_floor, safety_margin * 100, min_load,
        )
        return

    scale_factor = target_floor / current_floor
    network.generators.loc[must_run_mask, "p_min_pu"] *= scale_factor

    new_floor = (
        network.generators.loc[must_run_mask, "p_min_pu"]
        * network.generators.loc[must_run_mask, "p_nom"]
    ).sum()

    logger.info(
        "Rescaled must-run p_min_pu by factor %.4f: floor %.1f MW -> "
        "%.1f MW (target <= %.1f MW, min load = %.1f MW).",
        scale_factor, current_floor, new_floor, target_floor, min_load,
    )    

rescale_must_run_floor(network)

network.sanitize()

# Export the network without changing the input files
network.export_to_netcdf( "Thesis_BESS/data/filtered_network.nc")

print("Network saved as filtered_network.nc")

print("\nNetwork name:", network.name)
print("PyPSA version:", network.pypsa_version)
print("Number of snapshots:", len(network.snapshots))
print("Snapshot range:", network.snapshots.min(), "to", network.snapshots.max())
#print("Snapshot frequency:", network.snapshots.inferred_freq)

print("\nComponents:")
print("Buses:", len(network.buses))
print("Lines:", len(network.lines))
print("Generators:", len(network.generators))
print("Loads:", len(network.loads))

print("\nBus columns:")
print(network.buses.columns.tolist())

print("\nLine columns:")
print(network.lines.columns.tolist())

print("\nGenerator columns:")
print(network.generators.columns.tolist())

print("\nLoad columns:")
print(network.loads.columns.tolist())

print("\nTime-dependent load profile:")
print("Shape:", network.loads_t.p_set.shape)
print(network.loads_t.p_set.head())

print("\nSnapshot weightings:")
print(network.snapshot_weightings.head())

print("\nTotal system load in MW:")
print(network.loads_t.p_set.sum(axis=1).head())