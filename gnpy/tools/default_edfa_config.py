#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.default_edfa_configs: loads JSON configuration files at module initialization time
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.tools.default_edfa_config
==============================

Default configs for pre defined amplifiers:
- Juniper-BoosterHG.json,
- std_medium_gain_advanced_config.json
"""

from logging import getLogger
from typing import Dict, Optional
from json import JSONDecodeError, load
from pathlib import Path

from gnpy.core.exceptions import ConfigurationError
from gnpy.tools.convert_legacy_yang import yang_to_legacy


_logger = getLogger(__name__)
_examples_dir = Path(__file__).parent.parent / 'example-data'


def _load_json_file(file_path: Path) -> Optional[Dict]:
    """Load and parse a JSON file.
    :param file_path: Path to the JSON file to load
    :type file_path: Path
    :return: Dict containing the parsed JSON data or None if loading fails
    :rtype: Optional[Dict]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yang_to_legacy(load(file))
    except FileNotFoundError:
        msg = f"Configuration file not found: {file_path}"
        _logger.error(msg)
        return None
    except JSONDecodeError as e:
        msg = f"Invalid JSON in configuration file {file_path}: {e}"
        _logger.error(msg)
        return None


# Default files to load
_files_to_load = {
    "std_medium_gain_advanced_config.json": _examples_dir / "std_medium_gain_advanced_config.json",
    "Juniper-BoosterHG.json": _examples_dir / "Juniper-BoosterHG.json"
}

# Load configurations
_configs: Dict = {}

for key, filepath in _files_to_load.items():
    config_data = _load_json_file(filepath)
    if config_data is not None:
        _configs[key] = config_data
    else:
        _msg = f"Failed to load configuration: {key}. Using empty dict as fallback."
        _logger.error(_msg)
        raise ConfigurationError

# Expose the constant
DEFAULT_EXTRA_CONFIG: Dict[str, Dict] = _configs

DEFAULT_EQPT_CONFIG: Path = _examples_dir / "eqpt_config.json"
