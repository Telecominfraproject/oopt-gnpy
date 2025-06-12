#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.validate_mode: Library compliance to GNPy models validation
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors


"""Validate equipment libraries
"""

import os
import json
from argparse import ArgumentParser
from pathlib import Path
import logging

import oopt_gnpy_libyang as ly
from gnpy.tools.yang_convert_utils import load_data, yang_lib


STANDARD_MODE_ID = [
    # OpenZR+ Specifications, version 3.0, 12 September 2023 https://www.openzrplus.org/documents/
    'OpenZR+ ZR400-OFEC-16QAM',
    'OpenZR+ ZR400-OFEC-16QAM-HA',
    'OpenZR+ ZR400-OFEC-16QAM-HB',
    'OpenZR+ ZR400-OFEC-8QAM-HA',
    'OpenZR+ ZR400-OFEC-8QAM-HB',
    'OpenZR+ ZR300-OFEC-8QAM',
    'OpenZR+ ZR300-OFEC-8QAM-HA',
    'OpenZR+ ZR300-OFEC-8QAM-HB',
    'OpenZR+ ZR200-OFEC-QPSK',
    'OpenZR+ ZR200-OFEC-QPSK-HA',
    'OpenZR+ ZR200-OFEC-QPSK-HB',
    'OpenZR+ ZR100-OFEC-QPSK',
    'OpenZR+ ZR100-OFEC-QPSK-HA',
    'OpenZR+ ZR100-OFEC-QPSK-HB',
    'OIF-100GZR',
    'OIF-400GZR-0x01',
    'OIF-400GZR-0x02',
    'OIF-400GZR-0x03',
    'OIF-800GZR',
    # OpenROADM IPs in catalog https://github.com/OpenROADM/OpenROADM_MSA_Public/blob/master/model/Specifications/apidoc-operational-modes-to-catalog-15_0-optical-spec-6_0.json
    'OpenROADM-OR-W-100G-SC',
    'OpenROADM-OR-W-100G-oFEC-31.6Gbd',
    'OpenROADM-OR-W-200G-oFEC-31.6Gbd',
    'OpenROADM-OR-W-200G-oFEC-63.1Gbd',
    'OpenROADM-OR-W-300G-oFEC-63.1Gbd',
    'OpenROADM-OR-W-400G-oFEC-63.1Gbd',
    'OpenROADM-OR-W-400G-oFEC-124Gbd',
    'OpenROADM-OR-W-400G-oFEC-118Gbd',
    'OpenROADM-OR-W-400G-oFEC-124Gbd_type2',
    'OpenROADM-OR-W-400G-oFEC-118Gbd_type2',
    'OpenROADM-OR-W-800G-oFEC-124Gbd',
    'OpenROADM-OR-W-800G-oFEC-118Gbd',
    'OpenROADM-OR-W-800G-oFEC-124Gbd_type2',
    'OpenROADM-OR-W-800G-oFEC-118Gbd_type2',
    'OpenROADM-OR-W-600G-oFEC-124Gbd',
    'OpenROADM-OR-W-600G-oFEC-118Gbd',
    'OpenROADM-OR-W-800G-oFEC-131GbdE',
    'OpenROADM-OR-W-800G-oFEC-131GbdM',
    'OpenROADM-OR-W-600G-oFEC-124Gbd_type2',
    'OpenROADM-OR-W-600G-oFEC-118Gbd_type2',
    'OpenROADM-OR-W-800G-oFEC-131GbdE_type2',
    'OpenROADM-OR-W-800G-oFEC-131GbdM_type2',
]


def keys():
    return [
        'type_variety',
        'format',
        'frequency',
        'min',
        'max',
        'baud_rate',
        'OSNR',
        'tx_osnr',
        'min_spacing',
        'bit_rate',
        'roll_off',
        'penalties',
        'other_names',
        'additional_comments',
        'chromatic_dispersion',
        'pmd',
        'pdl',
        'cost'
    ]


# Configure logging
def _setup_logging(args):
    """
    """
    logging.basicConfig(level={2: logging.DEBUG, 1: logging.INFO, 0: logging.WARNING}.get(args.verbose, logging.DEBUG),
                        format='%(message)s')


def validate_modes(directory):
    """collect json file list in the directory, collect vendor name, validate each mode
    """
    json_files = [f for f in os.listdir(directory) if f.endswith('.json') or f.endswith('.JSON')]
    for file_name in json_files:
        # get vendor name
        mode_names = []
        msg = f'\nstart reading {file_name}\n------------'
        logging.info(msg)
        with open(directory / file_name, 'r', encoding='utf-8') as file:
            try:
                equipment_dict = json.load(file)
            except json.decoder.JSONDecodeError as e:
                logging.warning(f'{e}\nFile {file_name} has json issues. Analysis stopped.\n')
                continue
            # yang validation
            try:
                _ = load_data(json.dumps(equipment_dict), yang_lib())
            except ly.Error:
                logging.warning(f'File {file_name} not compliant. Analysis stopped.\n')
                continue
        # values verification
        for pluggable in equipment_dict['gnpy-eqpt-config:equipment']['Transceiver']:
            for mode in pluggable['mode']:
                mode['type_variety'] = pluggable['type_variety']
                mode['frequency'] = pluggable.get('frequency', None)
                name = f'{pluggable["type_variety"]}{mode["format"]}'
                logging.warning(
                    f'\n{file_name}\n\tpluggable name: {pluggable["type_variety"]}\n\tmode name: {mode["format"]}')
                if name in mode_names:
                    logging.warning(f'{pluggable["type_variety"]} mode name not unique {mode["format"]}')
                check_mode(mode)
                mode_names.append(name)


def validate_value(key, value, expected_type, range_check=None, units=''):
    """Helper function to validate value type and range."""
    if value is None:
        logging.warning(f'Missing value for {key}.')
        return None  # Return early if value is None
    try:
        if expected_type == float:
            value = float(value)
        elif expected_type == int:
            value = int(value)

        if range_check:
            if not range_check(value):
                logging.warning(f'Invalid {key}. Value: {value} does not meet range criteria ({units}).')

        return value  # Return the validated value for further checks if needed
    except ValueError:
        msg = f'Invalid {key}. Must be a {expected_type.__name__} ' \
            + f'(float or string representation of a {expected_type.__name__})'
        logging.warning(msg)


def check_mode(data):
    """verifies that mode contains all fields, that no new field are added, that values are meaningful

    type_variety: present and unique
    format: present and unique
    frequency/min: present and below /max and  in range 180-200
    frequency/max: present and above /min and  in range 180-200
    frequency/tunability:  present and in range 0.01-10 or in 3.125, 6.25, 12.5, 25, 50
    baud_rate: present and in the range 10-1000
    OSNR: present and in range 0-40
    tx_osnr: present and in the range 10-1000
    min_spacing: present and above baud_rate and in the range 10-1000
    bit_rate: present and in 100, 200, 300, 400, 500, 600
    roll_off: present and in the range 0-1
    penalties: present contains a list of penalty and penalty_value:
    penalties/chromatic_dispersion
    penalties/pmd
    penalties/pdl
    other_names: present, empty or contains standard mode names or proprietary mode names
    """
    # check that all keys are in keys() (only 2 levels checked)
    for key, value in data.items():
        if key not in keys():
            logging.warning(f'Invalid key {key}')
        else:
            if isinstance(value, dict):
                for key2 in value.keys():
                    if key2 not in keys():
                        logging.warning(f'Invalid key {key2} in {key}')

    # Check type_variety and format uniqueness
    if 'type_variety' not in data:
        logging.warning('Missing type_variety')
    if 'format' not in data:
        logging.warning('Missing format')

    # Check frequency
    if 'frequency' in data:
        frequency_min = validate_value('frequency/min', data['frequency'].get('min'), float,
                                       lambda x: 180e12 <= x <= float(data['frequency'].get('max', float('inf'))), 'Hz')
        _ = validate_value('frequency/max', data['frequency'].get('max'), float, lambda x: frequency_min <= x <= 200e12, 'Hz')
    else:
        logging.warning('Missing frequency dictionary')

    # Check presence
    required_keys = [
        'baud_rate',
        'OSNR',
        'tx_osnr',
        'min_spacing', 'bit_rate',
        'roll_off'
    ]

    for key in required_keys:
        if key not in data:
            logging.warning(f'Missing {key}')

    # Check baud_rate range
    validate_value('baud_rate', data.get('baud_rate'), float, lambda x: 10e9 <= x <= 1000e9, 'Bd')

    # Check OSNR range
    validate_value('OSNR', data.get('OSNR'), float, lambda x: 0 <= x <= 40)

    # Check tx_osnr range
    for key in ['tx_osnr']:
        validate_value(key, data.get(key), float, lambda x: 10 <= x <= 1000)

    # Check min_spacing
    min_spacing = validate_value('min_spacing', data.get('min_spacing'), float, None, 'Hz')
    baud_rate = float(data.get('baud_rate'))
    if not (min_spacing > baud_rate and 10e9 <= min_spacing <= 1000e9):
        logging.warning('Invalid min_spacing. Must be above baud_rate and in range 10-1000 GHz')

    # Check bit_rate
    validate_value(key='bit_rate', value=data.get('bit_rate'), expected_type=float,
                   range_check=lambda x: x in [100e9, 200e9, 300e9, 400e9, 500e9, 600e9, 700e9, 800e9],
                   units='bit/s')

    # Check roll_off range
    validate_value('roll_off', data.get('roll_off'), float, lambda x: 0 <= x <= 1)

    # Check penalties keys
    penalties_keys = [
        'chromatic_dispersion',
        'pmd',
        'pdl'
    ]

    if 'penalties' in data:
        for key in penalties_keys:
            key_exist = False
            for elem in data['penalties']:
                if key in elem:
                    key_exist = True
            if not key_exist:
                logging.warning(f'Missing nested key in penalties: {key}')
    else:
        logging.warning('Missing penalties dictionary')

    # check that other_names contains a standard mode
    is_standard = False
    if 'other_names' in data:
        for other_name in data['other_names']:
            if other_name in STANDARD_MODE_ID:
                is_standard = True
    if not is_standard:
        logging.warning('Proprietary mode only')


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('directory', type=Path,
                        help='Input folders with JSON files')
    PARSER.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be specified several times)')
    args = PARSER.parse_args()
    _setup_logging(args)
    validate_modes(args.directory)
