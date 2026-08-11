"""Erzeugt Testgebote im EPEX-artigen Format für ein 2-Zonen-System (24 h)."""

import csv
from pathlib import Path

OUT = Path("assume/testing_tools/test_data")
OUT.mkdir(exist_ok=True)

# --- Nachfrageprofile (MW) -------------------------------------------------
DEMAND = {
    "DE": [6800, 6400, 6200, 6100, 6300, 7000, 8200, 9200, 9800, 9900, 9800, 9700,
           9500, 9400, 9300, 9400, 9800, 10500, 11000, 10800, 10000, 9000, 8000, 7200],
    "FR": [3600, 3400, 3300, 3250, 3350, 3700, 4300, 4800, 5100, 5150, 5100, 5050,
           4950, 4900, 4850, 4900, 5100, 5450, 5500, 5400, 5100, 4700, 4200, 3800],
}

# --- Erzeugungsstufen: (Label, Preis €/MWh, Kapazität MW) -------------------
SUPPLY_STEPS = {
    "DE": [
        ("de_lignite", 22.0, 2500),
        ("de_coal",    38.0, 2500),
        ("de_gas_cc",  62.0, 3000),
        ("de_gas_ot",  95.0, 2000),
        ("de_oil",    180.0, 1500),
    ],
    "FR": [
        ("fr_hydro",   15.0, 1200),
        ("fr_gas_cc",  58.0, 2000),
        ("fr_gas_ot",  88.0, 1500),
        ("fr_oil",    160.0, 1200),
    ],
}

# --- Preisunelastischer Nachfrageanteil ------------------------------------
INELASTIC_PRICE = 3000.0      # muss <= maximum_bid_price sein
ELASTIC_SHARE = 0.10          # 10 % der Last als Preisstufen
ELASTIC_PRICES = [200.0, 120.0, 70.0]


def write_hourly_orders(path: Path) -> None:
    """Schreibt Stundengebote: 8 Spalten, komma-separiert.

    Spaltenbelegung gemäss hourly_orders_reader:
    0=id, 1=frei, 2=frei, 3=hour(1-basiert), 4=price, 5=quantity,
    6='Purchase'|'Sell', 7=bidding_zone
    """
    rows = []
    oid = 0

    for zone in ("DE", "FR"):
        for h in range(24):
            load = DEMAND[zone][h]
            inelastic = load * (1.0 - ELASTIC_SHARE)
            elastic_block = load * ELASTIC_SHARE / len(ELASTIC_PRICES)

            # Preisunelastische Grundlast
            oid += 1
            rows.append([f"D{oid:05d}", "HOURLY", "", h + 1,
                         INELASTIC_PRICE, round(inelastic, 1),
                         "Purchase", zone])

            # Elastische Nachfragestufen
            for p in ELASTIC_PRICES:
                oid += 1
                rows.append([f"D{oid:05d}", "HOURLY", "", h + 1,
                             p, round(elastic_block, 1),
                             "Purchase", zone])

            # Angebotsstufen
            for label, price, cap in SUPPLY_STEPS[zone]:
                oid += 1
                rows.append([f"S{oid:05d}", "HOURLY", label, h + 1,
                             price, float(cap),
                             "Sell", zone])

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"{path}: {len(rows)} Zeilen")


def _block_row(bid_id, btype, special_id, price, quantities, zone):
    """Baut eine Blockzeile mit 33 Spalten gemäss block_orders_reader.

    0='BB', 1=id, 2=type(C01/C02/C04), 3=special_id, 4=frei,
    5=price, 6..29=24 Stundenmengen, 30..31=frei, 32=bidding_zone
    """
    row = ["BB", bid_id, btype, special_id, "", price]
    row += [float(q) for q in quantities]      # Spalten 6-29
    row += ["", ""]                            # Spalten 30-31
    row.append(zone)                           # Spalte 32
    assert len(row) == 33, len(row)
    return row


def write_block_orders(path: Path) -> None:
    """Schreibt Blockgebote."""
    rows = []

    def profile(hours, value):
        """24er-Vektor, nur in 'hours' mit 'value' besetzt."""
        q = [0.0] * 24
        for h in hours:
            q[h] = float(value)
        return q

    # --- Einfache Blöcke (C01): Grundlast-Kernkraft -----------------------
    # Positive Mengen = Verkauf, muss-laufend über 24 h.
    rows.append(_block_row("B_DE_NUC1", "C01", "", 18.0,
                           profile(range(24), -1600), "DE"))
    rows.append(_block_row("B_DE_NUC2", "C01", "", 20.0,
                           profile(range(24), -1200), "DE"))
    rows.append(_block_row("B_FR_NUC1", "C01", "", 12.0,
                           profile(range(24), -2600), "FR"))
    rows.append(_block_row("B_FR_NUC2", "C01", "", 14.0,
                           profile(range(24), -1800), "FR"))

    # --- Verknüpfte Blöcke (C02): Kohleblock + Aufschlagstufe ------------
    # Parent muss vor dem Child gelesen werden; Reihenfolge ist hier egal,
    # load_orders prüft die Existenz nachträglich.
    rows.append(_block_row("B_DE_COAL_P", "C01", "", 34.0,
                           profile(range(6, 22), -900), "DE"))
    rows.append(_block_row("B_DE_COAL_C", "C02", "B_DE_COAL_P", 45.0,
                           profile(range(8, 20), -400), "DE"))

    # --- Exklusive Gruppe (C04): flexibler Gasblock, 3 Zeitfenster -------
    # Genau eine der drei Varianten darf akzeptiert werden.
    rows.append(_block_row("B_DE_FLEX_M", "C04", "GRP_DE_FLEX", 70.0,
                           profile(range(7, 12), -600), "DE"))
    rows.append(_block_row("B_DE_FLEX_A", "C04", "GRP_DE_FLEX", 68.0,
                           profile(range(12, 17), -600), "DE"))
    rows.append(_block_row("B_DE_FLEX_E", "C04", "GRP_DE_FLEX", 66.0,
                           profile(range(17, 22), -600), "DE"))


    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)
    print(f"{path}: {len(rows)} Zeilen")


if __name__ == "__main__":
    write_hourly_orders(OUT / "test_hourly_2zone.csv")
    write_block_orders(OUT / "test_blocks_2zone.csv")