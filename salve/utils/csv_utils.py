
"""Utilities for reading in CSV or TSV files."""

import csv
from typing import Any, Dict, List


def read_csv(fpath: str, delimiter: str = ",") -> List[Dict[str, Any]]:
    """Read in a .csv or .tsv file as a list of dictionaries."""
    rows = []

    with open(fpath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        for row in reader:
            rows.append(row)

    return rows
