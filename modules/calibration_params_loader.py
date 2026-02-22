"""
Calibration parameters loader - reads from JSON format.

This module provides a clean interface to load calibration parameters
from the JSON data file, maintaining the same API as the previous
Python-based configuration module.
"""

import json
from pathlib import Path
from typing import Dict, List
import shutil


# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
PARAMS_FILE = MODULE_DIR / 'calibration_parameters.json'


def _load_parameters() -> tuple:
    """
    Load calibration parameters from JSON file.
    
    Returns
    -------
    tuple
        (parameters, is_new_format) where:
        - parameters: dict containing parameter data
        - is_new_format: bool indicating if it uses unique parameter sets
    """
    if not PARAMS_FILE.exists():
        # Attempt to auto-generate the parameters JSON using the
        # original `initial_parameters` script if available.
        try:
            from modules.initial_parameters import generate_init_parameters

            # Ask generator to save into the modules folder. The
            # original generator writes 'calibration_parameters_wcal1.json',
            # so we'll call it and then copy/rename to the expected name.
            try:
                generate_init_parameters(json_path=str(MODULE_DIR), save=True)
            except Exception:
                # If generation fails, continue to final error below
                pass

            # If the generator created an alternate filename, copy it
            alt_file = MODULE_DIR / 'calibration_parameters_wcal1.json'
            if alt_file.exists():
                shutil.copy(alt_file, PARAMS_FILE)
                # remove the alternate file to avoid keeping two copies
                try:
                    alt_file.unlink()
                except Exception:
                    pass

        except Exception:
            # ignore import/generation errors here and raise a clear message below
            pass

        if not PARAMS_FILE.exists():
            raise FileNotFoundError(
                f"Calibration parameters file not found: {PARAMS_FILE}\n"
                f"You can generate it by running 'modules.initial_parameters.generate_init_parameters' or"
                f" create 'modules/calibration_parameters.json' manually."
            )
    
    with open(PARAMS_FILE, 'r') as f:
        data = json.load(f)
    
    # Check format: new format has 'parameter_sets' and 'row_mapping'
    is_new_format = 'parameter_sets' in data and 'row_mapping' in data
    
    # Normalize old-format keys: if top-level keys are numeric strings, convert to ints
    if not is_new_format and isinstance(data, dict):
        try:
            # build new dict with int keys when possible
            normalized = {}
            for k, v in data.items():
                if isinstance(k, str) and k.isdigit():
                    normalized[int(k)] = v
                else:
                    normalized[k] = v
            data = normalized
        except Exception:
            # if anything goes wrong, keep original data
            pass

    return data, is_new_format


# Load parameters once at module import time
_CACHED_PARAMETERS = None
_IS_NEW_FORMAT = None


def _get_cached_parameters() -> tuple:
    """Get cached parameters, loading if necessary."""
    global _CACHED_PARAMETERS, _IS_NEW_FORMAT
    if _CACHED_PARAMETERS is None:
        _CACHED_PARAMETERS, _IS_NEW_FORMAT = _load_parameters()
    return _CACHED_PARAMETERS, _IS_NEW_FORMAT


def get_parameters_for_row(row: int) -> dict:
    """
    Get calibration parameters for a specific row.
    
    Parameters
    ----------
    row : int
        Row number (typically 6-323)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'idx_interval': dict with spectral interval indices
        - 'bckg_fit': float, background fit parameter
        - 'init_parameters': dict with initial gaussian parameters
    
    Raises
    ------
    ValueError
        If the row number is not in the calibration parameters
    """
    params, is_new_format = _get_cached_parameters()
    row_str = str(int(row))
    
    if is_new_format:
        # New format: unique parameter sets with row mapping
        if row_str not in params['row_mapping']:
            raise ValueError(
                f"Parameters not defined for row {row}. "
                f"Available rows: {get_all_rows()}"
            )
        param_set_id = str(params['row_mapping'][row_str])
        return params['parameter_sets'][param_set_id]
    else:
        # Old format: direct row mapping
        row_int = int(row)
        if row_int not in params:
            raise ValueError(
                f"Parameters not defined for row {row}. "
                f"Available rows: {get_all_rows()}"
            )
        return params[row_int]


def get_all_rows() -> List[int]:
    """
    Get list of all rows with defined parameters.
    
    Returns
    -------
    list of int
        Sorted list of row numbers with calibration parameters
    """
    params, is_new_format = _get_cached_parameters()
    
    if is_new_format:
        return sorted([int(r) for r in params['row_mapping'].keys()])
    else:
        return sorted([k for k in params.keys() if isinstance(k, int)])


def get_parameters_dict() -> Dict:
    """
    Get all parameters as a dictionary (for advanced usage).
    
    Returns
    -------
    dict
        Full parameter dictionary
    """
    params, _ = _get_cached_parameters()
    return params


def reload_parameters():
    """
    Reload parameters from disk (useful if file was modified externally).
    
    This function should rarely be needed in normal operation, but can be
    useful if the calibration_parameters.json file is updated and you want
    to reload it without restarting Python.
    """
    global _CACHED_PARAMETERS, _IS_NEW_FORMAT
    _CACHED_PARAMETERS = None
    _IS_NEW_FORMAT = None
    return _get_cached_parameters()


if __name__ == '__main__':
    # Simple test
    rows = get_all_rows()
    print(f"✓ Loaded {len(rows)} rows")
    print(f"Row range: {min(rows)} - {max(rows)}")
    
    # Show sample
    sample_row = rows[0]
    sample_params = get_parameters_for_row(sample_row)
    print(f"\nSample (row {sample_row}):")
    for key, val in sample_params.items():
        print(f"  {key}: {val}")
