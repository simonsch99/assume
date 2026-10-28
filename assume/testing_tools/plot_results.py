from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results = Path(__file__).parent / "results"
with_so = pd.read_csv(results / "with_SO" / "market_meta.csv")
without_so = pd.read_csv(results / "without_SO" / "market_meta.csv")

for data in (with_so, without_so):
    data["product_start"] = pd.to_datetime(data["product_start"])

output = results / "price_plots"
output.mkdir(exist_ok=True)

for zone in sorted(with_so["zone"].unique()):
    plt.figure(figsize=(10, 4))

    for data, label in [(without_so, "Without SO"), (with_so, "With SO")]:
        zone_data = data[data["zone"] == zone]
        plt.plot(zone_data["product_start"], zone_data["price"], label=label)

    plt.title(f"Price curve – {zone}")
    plt.xlabel("Time")
    plt.ylabel("Price [EUR/MWh]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / f"price_curve_{zone}.png", dpi=150)
    plt.close()

print(f"Plots saved to {output}")