#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# JSON files format conversion legacy <-> YANG
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
YANG formatted to legacy format conversion
==========================================

"""

from argparse import ArgumentParser
from pathlib import Path
from copy import deepcopy
import json
from typing import Dict

from gnpy.tools.yang_convert_utils import convert_degree, convert_back_degree, \
    convert_delta_power_range, convert_back_delta_power_range, \
    convert_dict, convert_back, \
    remove_null_region_city, remove_union_that_fail, \
    convert_design_band, convert_back_design_band, \
    convert_none_to_empty, convert_empty_to_none, \
    convert_loss_coeff_list, convert_back_loss_coeff_list, \
    ELEMENTS_KEY, PATH_REQUEST_KEY, RESPONSE_KEY, SPECTRUM_KEY, EQPT_TYPES, EDFA_CONFIG_KEYS, SIM_PARAMS_KEYS, \
    TOPO_NMSP, SERV_NMSP, EQPT_NMSP, SPECTRUM_NMSP, SIM_PARAMS_NMSP, EDFA_CONFIG_NMSP, RESP_NMSP, \
    dump_data, add_missing_default_type_variety, \
    remove_namespace_context


def legacy_to_yang(json_data: Dict) -> Dict:
    """Convert legacy format to GNPy YANG format.

    This function adds the required namespace if not present and processes the input JSON data
    based on its structure to convert it to the appropriate YANG format.

    :param json_data: The input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data in GNPy YANG format.
    :rtype: Dict
    """
    json_data = convert_none_to_empty(deepcopy(json_data))

    # case of topology file
    if ELEMENTS_KEY in json_data:
        json_data = remove_null_region_city(json_data)
        json_data = convert_degree(json_data)
        json_data = convert_design_band(json_data)
        json_data = convert_loss_coeff_list(json_data)
        json_data = convert_dict(json_data)
        return {TOPO_NMSP: json_data}
    if TOPO_NMSP in json_data:
        # then this is a new format topology file, ensure that there are no issues
        json_data[TOPO_NMSP] = convert_degree(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = convert_design_band(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = convert_loss_coeff_list(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = remove_null_region_city(json_data[TOPO_NMSP])
        json_data = convert_dict(json_data)
        return json_data

    # case of equipment file
    if any(k in json_data for k in EQPT_TYPES):
        json_data = convert_delta_power_range(json_data)
        json_data = add_missing_default_type_variety(json_data)
        json_data = convert_dict(json_data)
        return {EQPT_NMSP: json_data}
    if EQPT_NMSP in json_data:
        # then this is already a new format topology file, ensure that there are no issues
        json_data[EQPT_NMSP] = convert_delta_power_range(json_data[EQPT_NMSP])
        json_data = convert_dict(json_data)
        return json_data

    # case of service file
    if PATH_REQUEST_KEY in json_data:
        json_data = remove_union_that_fail(json_data)
        json_data = convert_dict(json_data)
        return {SERV_NMSP: json_data}
    if SERV_NMSP in json_data:
        # then this is a new format service file, ensure that there are no issues
        json_data = convert_dict(json_data)
        return json_data

    # case of edfa_config file
    if any(k in json_data for k in EDFA_CONFIG_KEYS):
        json_data = convert_dict(json_data)
        return {EDFA_CONFIG_NMSP: json_data}
    if EDFA_CONFIG_NMSP in json_data:
        # then this is a new format edfa_config file, ensure that there are no issues
        json_data = convert_dict(json_data)
        return json_data

    # case of spectrum file
    if SPECTRUM_KEY in json_data:
        json_data = convert_dict(json_data)
        return {SPECTRUM_NMSP: json_data[SPECTRUM_KEY]}
    if SPECTRUM_NMSP in json_data:
        # then this is a new format spectrum file, ensure that there are no issues
        json_data = convert_dict(json_data)
        return json_data

    # case of sim_params file
    if any(k in json_data for k in SIM_PARAMS_KEYS):
        json_data = convert_dict(json_data)
        return {SIM_PARAMS_NMSP: json_data}
    if SIM_PARAMS_NMSP in json_data:
        # then this is a new format sim_params file, ensure that there are no issues
        json_data = convert_dict(json_data)
        return json_data

    # case of response file
    if RESPONSE_KEY in json_data:
        json_data = convert_dict(json_data)
        return {RESP_NMSP: json_data}
    if RESP_NMSP in json_data:
        # then this is a new format response file, ensure that there are no issues
        json_data = convert_dict(json_data)
        return json_data

    raise ValueError('Unrecognized type of content (not topology, service or equipment)')


def yang_to_legacy(json_data: Dict) -> Dict:
    """Convert GNPy YANG format to legacy format.

    This function processes the input JSON data to convert it from the new GNPy YANG format
    back to the legacy format. It handles various types of content, including topology,
    equipment, and service files, ensuring that the necessary conversions are applied.

    :param json_data: The input JSON data in GNPy YANG format to convert.
    :type json_data: Dict
    :return: The converted JSON data in legacy format.
    :rtype: Dict

    :raises ValueError: If the input JSON data does not match any recognized content type
                        (not topology, service, or equipment).
    """
    json_data = convert_empty_to_none(json_data)
    json_data = convert_back(json_data)

    # case of topology file
    if ELEMENTS_KEY in json_data:
        json_data = convert_back_degree(json_data)
        json_data = convert_back_design_band(json_data)
        json_data = convert_back_loss_coeff_list(json_data)
        return json_data
    if TOPO_NMSP in json_data:
        json_data = convert_back_degree(json_data[TOPO_NMSP])
        json_data = convert_back_design_band(json_data)
        json_data = convert_back_loss_coeff_list(json_data)
        return json_data

    # case of equipment file
    if any(k in json_data for k in EQPT_TYPES):
        json_data = convert_back_delta_power_range(json_data)
        json_data = remove_namespace_context(json_data, "gnpy-eqpt-config:")
        return json_data
    if EQPT_NMSP in json_data:
        json_data[EQPT_NMSP] = convert_back_delta_power_range(json_data[EQPT_NMSP])
        json_data = remove_namespace_context(json_data[EQPT_NMSP], "gnpy-eqpt-config:")
        return json_data

    # case of EDFA config file
    if any(k in json_data for k in EDFA_CONFIG_KEYS):
        json_data = convert_back_delta_power_range(json_data)
        return json_data
    if EDFA_CONFIG_NMSP in json_data:
        return json_data

    # case of service file
    if PATH_REQUEST_KEY in json_data:
        return json_data
    if SERV_NMSP in json_data:
        return json_data[SERV_NMSP]

    # case of sim_params file
    if any(k in json_data for k in SIM_PARAMS_KEYS):
        return json_data
    if SIM_PARAMS_NMSP in json_data:
        return json_data[SIM_PARAMS_NMSP]

    # case of spectrum file
    if SPECTRUM_KEY in json_data:
        return json_data
    if SPECTRUM_NMSP in json_data:
        return {SPECTRUM_KEY: json_data[SPECTRUM_NMSP]}

    # case of planning response file
    if RESPONSE_KEY in json_data:
        return json_data
    if RESP_NMSP in json_data:
        return json_data[RESP_NMSP]

    raise ValueError('Unrecognized type of content (not topology, service or equipment)')


def main():
    parser = ArgumentParser()
    parser.add_argument('--legacy-to-yang', nargs='?', type=Path,
                        help='convert file with this name into yangconformedname.json')
    parser.add_argument('--yang-to-legacy', nargs='?', type=Path,
                        help='convert file with this name into gnpy'
                             + ' using decimal instead of strings and null instead of [null]')
    parser.add_argument('-o', '--output', type=Path,
                        help='Stores into file with this name; default = GNPy_legacy_formatted-<file_name>.json or'
                        + 'GNPy_yang_formatted-<file_name>.json')
    args = parser.parse_args()

    if args.legacy_to_yang:
        prefix = 'GNPy_yang_formatted-'
        with open(args.legacy_to_yang, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            converted = dump_data(legacy_to_yang(json_data))
        output = prefix + str(args.legacy_to_yang.name)
    elif args.yang_to_legacy:
        prefix = 'GNPy_legacy_formatted-'
        with open(args.yang_to_legacy, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            converted = json.dumps(yang_to_legacy(json_data), indent=2, ensure_ascii=False)
        output = prefix + str(args.yang_to_legacy.name)
    if args.output:
        output = args.output
    with open(output, 'w', encoding='utf-8') as f:
        f.write(converted)


if __name__ == '__main__':
    main()
