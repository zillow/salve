"""File I/O utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import json


def read_json_file(fpath: Union[str, Path]) -> Any:
    """Load dictionary from JSON file.
    
    Args:
        fpath: Path to JSON file.
    
    Returns:
        Deserialized Python dictionary or list.
    """
    if not Path(fpath).exists():
        raise ValueError(f"No file found at {fpath}")

    with open(fpath, "r") as f:
        return json.load(f)

def save_json_file(
    json_fpath: str,
    data: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary or list to a JSON file.
    
    Args:
        json_fpath: Path to file to create.
        data: Python dictionary or list to be serialized.
    """
    os.makedirs(os.path.dirname(json_fpath), exist_ok=True)
    with open(json_fpath, "w") as f:
        json.dump(data, f, indent=4)