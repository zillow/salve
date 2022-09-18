"""File I/O utilities."""

from pathlib import Path
from typing import Union

import json


def read_json_file(fpath: Union[str, Path]) -> Any:
    """Load dictionary from JSON file.
    
    Args:
        fpath: Path to JSON file.
    
    Returns:
        Deserialized Python dictionary or list.
    """
    with open(fpath, "r") as f:
        return json.load(f)