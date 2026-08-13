import csv
from pathlib import Path


OUTPUT_DIR = Path("assume/testing_tools/test_data")
HOURLY_FILE = OUTPUT_DIR / "test_hourly_block_features_2zone.csv"
BLOCK_FILE = OUTPUT_DIR / "test_blocks_features_2zone.csv"


def create_hourly_orders(path: Path) -> None:
    """
    Creates four simple hourly orders for every delivery hour:

    DE:
      - 100 MW demand at 1000 EUR/MWh
      - 200 MW backstop supply at 100 EUR/MWh

    FR:
      - 100 MW demand at 1000 EUR/MWh
      - 40 MW base supply at 20 EUR/MWh

    Before block orders, the system therefore clears:
      - 200 MW demand
      - 40 MW cheap supply
      - 160 MW backstop supply

    The DE-FR line remains uncongested.
    """
    rows = []

    for hour in range(1, 25):
        rows.extend(
            [
                [f"D_DE_{hour:02d}","HOURLY","",hour,1000.0,100.0,"Purchase","DE",],
                [f"D_FR_{hour:02d}","HOURLY","",hour,1000.0,100.0,"Purchase","FR",],
                [f"S_FR_BASE_{hour:02d}","HOURLY","fr_base",hour,20.0,40.0,"Sell","FR",],
                [f"S_DE_BACKSTOP_{hour:02d}","HOURLY","de_backstop",hour,100.0,200.0,"Sell","DE",],
            ]
        )

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def block_row(
    bid_id: str,
    block_type: str,
    price: float,
    zone: str,
    quantities: dict[int, float],
    minimum_acceptance_ratio: float = 1.0,
    special_id: str = "",
) -> list:
    """
    Builds one 33-column block-order row.

    quantities uses delivery hours 1..24.

    Important:
      The block CSV reader multiplies quantities by -1.
      Therefore, negative CSV quantities represent supply orders.
    """
    hourly_quantities = [
        quantities.get(hour, 0.0)
        for hour in range(1, 25)
    ]

    row = [
        "BB",                         # 0: marker
        bid_id,                       # 1: block ID
        block_type,                   # 2: C01, C02, or C04
        special_id,                   # 3: parent or exclusive-group ID
        "",                           # 4: unused
        price,                        # 5: price
        *hourly_quantities,           # 6..29: 24 hourly quantities
        minimum_acceptance_ratio,     # 30: minimum acceptance ratio
        "",                           # 31: unused
        zone,                         # 32: bidding zone
    ]

    assert len(row) == 33
    return row


def create_block_orders(path: Path) -> None:
    rows = [
        # ---------------------------------------------------------------
        # Hours 1-4: ordinary fill-or-kill blocks
        # ---------------------------------------------------------------
        block_row(
            bid_id="REG_GOOD",
            block_type="C01",
            price=50.0,
            zone="DE",
            quantities={hour: -40.0 for hour in range(1, 5)},
            minimum_acceptance_ratio=1.0,
        ),
        block_row(
            bid_id="REG_BAD",
            block_type="C01",
            price=150.0,
            zone="FR",
            quantities={hour: -40.0 for hour in range(1, 5)},
            minimum_acceptance_ratio=1.0,
        ),

        # ---------------------------------------------------------------
        # Hours 5-6: profiled, partially accepted block
        #
        # At hour 6, only 160 MW remains after the 40 MW base supply.
        # The 200 MW profile therefore limits acceptance to:
        #
        #     160 / 200 = 0.8
        #
        # At hour 5 this produces 0.8 * 160 = 128 MW.
        # ---------------------------------------------------------------
        block_row(
            bid_id="PROFILE_PARTIAL",
            block_type="C01",
            price=60.0,
            zone="FR",
            quantities={
                5: -160.0,
                6: -200.0,
            },
            minimum_acceptance_ratio=0.5,
        ),

        # ---------------------------------------------------------------
        # Hours 7-10: profitable child saves a loss-making parent
        #
        # Parent: 20 MW at 120 EUR/MWh
        # Child:  60 MW at  20 EUR/MWh
        #
        # Their combined weighted price is:
        # (20*120 + 60*20) / 80 = 45 EUR/MWh.
        # ---------------------------------------------------------------
        block_row(
            bid_id="LINK_PARENT_SAVED",
            block_type="C01",
            price=120.0,
            zone="FR",
            quantities={hour: -20.0 for hour in range(7, 11)},
            minimum_acceptance_ratio=1.0,
        ),
        block_row(
            bid_id="LINK_CHILD_GOOD",
            block_type="C02",
            special_id="LINK_PARENT_SAVED",
            price=20.0,
            zone="FR",
            quantities={hour: -60.0 for hour in range(7, 11)},
            minimum_acceptance_ratio=1.0,
        ),

        # ---------------------------------------------------------------
        # Hours 11-12: parent accepted, unprofitable child rejected
        # ---------------------------------------------------------------
        block_row(
            bid_id="LINK_PARENT_GOOD",
            block_type="C01",
            price=30.0,
            zone="DE",
            quantities={hour: -40.0 for hour in range(11, 13)},
            minimum_acceptance_ratio=1.0,
        ),
        block_row(
            bid_id="LINK_CHILD_BAD",
            block_type="C02",
            special_id="LINK_PARENT_GOOD",
            price=150.0,
            zone="DE",
            quantities={hour: -40.0 for hour in range(11, 13)},
            minimum_acceptance_ratio=1.0,
        ),

        # ---------------------------------------------------------------
        # Hours 13-16: mutually exclusive blocks
        #
        # Both are cheaper than the 100 EUR/MWh backstop supply.
        # Without the exclusive constraint, both would be economical.
        # With the constraint, only the cheaper block should clear.
        # ---------------------------------------------------------------
        block_row(
            bid_id="EXCLUSIVE_A",
            block_type="C04",
            special_id="EX_GROUP_1",
            price=30.0,
            zone="DE",
            quantities={hour: -60.0 for hour in range(13, 17)},
            minimum_acceptance_ratio=1.0,
        ),
        block_row(
            bid_id="EXCLUSIVE_B",
            block_type="C04",
            special_id="EX_GROUP_1",
            price=70.0,
            zone="FR",
            quantities={hour: -60.0 for hour in range(13, 17)},
            minimum_acceptance_ratio=1.0,
        ),
    ]

    with path.open("w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    create_hourly_orders(HOURLY_FILE)
    create_block_orders(BLOCK_FILE)

    print(f"Created {HOURLY_FILE}")
    print(f"Created {BLOCK_FILE}")


if __name__ == "__main__":
    main()