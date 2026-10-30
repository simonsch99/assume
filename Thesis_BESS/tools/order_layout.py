"""Shared readers and writers for published EPEX-like order layouts."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NATIVE_SIMPLE_COLUMNS = [
    "Date",
    "Week",
    "Week Day",
    "Hour",
    "Quarter hour",
    "Price",
    "Volume",
    "Sale/Purchase",
]

PROFILE_PATTERN = re.compile(
    r"^Volume H(?P<hour>\d{2})(?:A)? Q(?P<quarter>[1-4])$"
)


@dataclass
class SimpleLayout:
    """Simple-order data and comments preceding the CSV header."""

    frame: pd.DataFrame
    comments: list[str]


@dataclass
class BlockLayout:
    """Block-order data and metadata required to write it back."""

    frame: pd.DataFrame
    header: list[str]
    preamble: list[str]
    post_header_comments: list[str]
    st_rows: list[str]


def read_simple_layout(
    path: str | Path,
    required_columns: set[str] | None = None,
) -> SimpleLayout:
    """Read a comma-delimited simple-order file and retain leading comments."""
    path = Path(path)
    comments: list[str] = []
    data_lines: list[str] = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        for line in file:
            if line.lstrip().startswith("#"):
                comments.append(line.rstrip("\n"))
            elif line.strip():
                data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No simple-order records found in {path}")

    frame = pd.read_csv(io.StringIO("".join(data_lines)))
    frame.columns = [str(column).strip() for column in frame.columns]

    required = set(NATIVE_SIMPLE_COLUMNS)
    if required_columns is not None:
        required.update(required_columns)

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Simple-order file {path} is missing columns: {sorted(missing)}"
        )

    return SimpleLayout(frame=frame, comments=comments)


def write_simple_layout(layout: SimpleLayout, path: str | Path) -> None:
    """Write a simple-order layout while retaining leading comments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        for comment in layout.comments:
            file.write(f"{comment}\n")
        layout.frame.to_csv(file, index=False)


def read_block_layout(path: str | Path) -> BlockLayout:
    """Read a semicolon-delimited native block-order layout."""
    path = Path(path)
    preamble: list[str] = []
    post_header_comments: list[str] = []
    st_rows: list[str] = []
    records: list[list[str]] = []
    header: list[str] | None = None
    header_seen = False

    with path.open(encoding="utf-8-sig", newline="") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith("# Data type(BB);"):
                header = next(csv.reader([line[2:]], delimiter=";"))
                header_seen = True
                continue

            if line.startswith("#"):
                if header_seen:
                    post_header_comments.append(line)
                else:
                    preamble.append(line)
                continue

            if line.startswith("ST;"):
                st_rows.append(line)
                continue

            if line.startswith("BB;"):
                records.append(next(csv.reader([line], delimiter=";")))

    if header is None:
        raise ValueError(f"Missing '# Data type(BB);...' header in {path}")

    if not records:
        raise ValueError(f"No BB records found in {path}")

    for row_number, record in enumerate(records, start=1):
        if len(record) != len(header):
            raise ValueError(
                f"BB record {row_number} contains {len(record)} fields; "
                f"expected {len(header)}."
            )

    return BlockLayout(
        frame=pd.DataFrame(records, columns=header),
        header=header,
        preamble=preamble,
        post_header_comments=post_header_comments,
        st_rows=st_rows,
    )


def write_block_layout(layout: BlockLayout, path: str | Path) -> None:
    """Write a native block-order layout while retaining its metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = list(layout.frame.columns)

    with path.open("w", encoding="utf-8", newline="") as file:
        for line in layout.preamble:
            file.write(f"{line}\n")

        file.write(f"# {';'.join(header)}\n")

        for line in layout.post_header_comments:
            file.write(f"{line}\n")

        for line in layout.st_rows:
            file.write(f"{line}\n")

        writer = csv.writer(file, delimiter=";", lineterminator="\n")
        writer.writerows(layout.frame[header].itertuples(index=False, name=None))


def native_profile_columns(columns) -> list[str]:
    """Return the 96 native profile columns in chronological order."""
    parsed: list[tuple[int, int, str]] = []

    for column in columns:
        match = PROFILE_PATTERN.fullmatch(str(column))
        if match is not None:
            parsed.append(
                (
                    int(match.group("hour")),
                    int(match.group("quarter")),
                    str(column),
                )
            )

    parsed.sort()
    result = [column for _, _, column in parsed]

    if len(result) != 96:
        raise ValueError(
            f"Expected 96 native block-profile columns, found {len(result)}."
        )

    return result
