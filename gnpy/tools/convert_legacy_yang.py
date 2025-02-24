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
    remove_namespace_context, load_data, reorder_route_objects, reorder_lumped_losses_objects, \
    reorder_raman_pumps, convert_raman_coef, convert_back_raman_coef, convert_raman_efficiency, \
    convert_back_raman_efficiency, convert_nf_coef, convert_back_nf_coef, \
    convert_nf_fit_coef, convert_back_nf_fit_coef


def legacy_to_yang(json_data: Dict) -> Dict:
    """Convert legacy format to GNPy YANG format.

    This function adds the required namespace if not present and processes the input JSON data
    based on its structure to convert it to the appropriate YANG format. There is no validation
    of yang formatted data.

    :param json_data: The input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data in GNPy YANG format.
    :rtype: Dict
    """
    json_data = convert_none_to_empty(deepcopy(json_data))

    # case of topology json
    if ELEMENTS_KEY in json_data:
        json_data = reorder_raman_pumps(json_data)
        json_data = reorder_lumped_losses_objects(json_data)
        json_data = remove_null_region_city(json_data)
        json_data = convert_degree(json_data)
        json_data = convert_design_band(json_data)
        json_data = convert_loss_coeff_list(json_data)
        json_data = convert_raman_coef(json_data)
        json_data = {TOPO_NMSP: json_data}
    elif TOPO_NMSP in json_data:
        # then this is a new format topology json, ensure that there are no issues
        json_data[TOPO_NMSP] = convert_degree(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = convert_design_band(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = convert_loss_coeff_list(json_data[TOPO_NMSP])
        json_data[TOPO_NMSP] = remove_null_region_city(json_data[TOPO_NMSP])

    # case of equipment json
    elif any(k in json_data for k in EQPT_TYPES):
        json_data = convert_raman_efficiency(json_data)
        json_data = convert_delta_power_range(json_data)
        json_data = convert_nf_coef(json_data)
        json_data = add_missing_default_type_variety(json_data)
        json_data = {EQPT_NMSP: json_data}
    elif EQPT_NMSP in json_data:
        # then this is already a new format topology json, ensure that there are no issues
        json_data[EQPT_NMSP] = convert_raman_efficiency(json_data[EQPT_NMSP])
        json_data[EQPT_NMSP] = convert_delta_power_range(json_data[EQPT_NMSP])
        json_data[EQPT_NMSP] = convert_nf_coef(json_data[EQPT_NMSP])
        json_data[EQPT_NMSP] = add_missing_default_type_variety(json_data[EQPT_NMSP])

    # case of service json
    elif PATH_REQUEST_KEY in json_data:
        json_data = reorder_route_objects(json_data)
        json_data = remove_union_that_fail(json_data)
        json_data = {SERV_NMSP: json_data}

    elif SERV_NMSP in json_data:
        json_data[SERV_NMSP] = reorder_route_objects(json_data[SERV_NMSP])
        json_data[SERV_NMSP] = remove_union_that_fail(json_data[SERV_NMSP])

    # case of edfa_config json
    elif any(k in json_data for k in EDFA_CONFIG_KEYS):
        json_data = convert_nf_fit_coef(json_data)
        json_data = {EDFA_CONFIG_NMSP: json_data}

    elif EDFA_CONFIG_NMSP in json_data:
        json_data[EDFA_CONFIG_NMSP] = convert_nf_fit_coef(json_data[EDFA_CONFIG_NMSP])

    # case of spectrum json
    elif SPECTRUM_KEY in json_data:
        json_data = {SPECTRUM_NMSP: json_data[SPECTRUM_KEY]}

    # case of sim_params json
    elif any(k in json_data for k in SIM_PARAMS_KEYS):
        json_data = {SIM_PARAMS_NMSP: json_data}

    # case of response json
    elif RESPONSE_KEY in json_data:
        json_data = {RESP_NMSP: json_data}

    elif any(k in json_data for k in [SPECTRUM_NMSP, SIM_PARAMS_NMSP, RESP_NMSP]):
        # then this is a new format json, nothing to convert
        pass

    else:
        raise ValueError('Unrecognized type of content (not topology, service or equipment)')

    json_data = convert_dict(json_data)
    return json_data


def yang_to_legacy(json_data: Dict) -> Dict:
    """Convert GNPy YANG format to legacy format.

    This function processes the input JSON data to convert it from the new GNPy YANG format
    back to the legacy format. It handles various types of content, including topology,
    equipment, and service jsons, ensuring that the necessary conversions are applied.
    The input data is validated with oopt-gnpy-libyang.

    :param json_data: The input JSON data in GNPy YANG format to convert.
    :type json_data: Dict
    :return: The converted JSON data in legacy format.
    :rtype: Dict

    :raises ValueError: If the input JSON data does not match any recognized content type
                        (not topology, service, or equipment).
    """
    # validate data compliance: make sure that this is yang formated data before validation.
    load_data(json.dumps(legacy_to_yang(json_data)))
    json_data = convert_empty_to_none(json_data)
    json_data = convert_back(json_data)

    # case of topology json
    if ELEMENTS_KEY in json_data:
        json_data = convert_back_degree(json_data)
        json_data = convert_back_design_band(json_data)
        json_data = convert_back_loss_coeff_list(json_data)
        json_data = convert_back_raman_coef(json_data)
    elif TOPO_NMSP in json_data:
        json_data = convert_back_degree(json_data[TOPO_NMSP])
        json_data = convert_back_design_band(json_data)
        json_data = convert_back_loss_coeff_list(json_data)
        json_data = convert_back_raman_coef(json_data)

    # case of equipment json
    elif any(k in json_data for k in EQPT_TYPES):
        json_data = convert_back_delta_power_range(json_data)
        json_data = convert_back_raman_efficiency(json_data)
        json_data = convert_back_nf_coef(json_data)
        json_data = remove_namespace_context(json_data, "gnpy-eqpt-config:")
    elif EQPT_NMSP in json_data:
        json_data[EQPT_NMSP] = convert_back_delta_power_range(json_data[EQPT_NMSP])
        json_data[EQPT_NMSP] = convert_back_raman_efficiency(json_data[EQPT_NMSP])
        json_data[EQPT_NMSP] = convert_back_nf_coef(json_data[EQPT_NMSP])
        json_data = remove_namespace_context(json_data[EQPT_NMSP], "gnpy-eqpt-config:")

    # case of EDFA config json
    elif any(k in json_data for k in EDFA_CONFIG_KEYS):
        json_data = convert_back_nf_fit_coef(json_data)
    elif EDFA_CONFIG_NMSP in json_data:
        json_data[EDFA_CONFIG_NMSP] = convert_back_nf_fit_coef(json_data[EDFA_CONFIG_NMSP])

    # case of service json
    elif SERV_NMSP in json_data:
        json_data = json_data[SERV_NMSP]

    # case of sim_params json
    elif SIM_PARAMS_NMSP in json_data:
        json_data = json_data[SIM_PARAMS_NMSP]

    # case of spectrum json
    elif SPECTRUM_NMSP in json_data:
        json_data = {SPECTRUM_KEY: json_data[SPECTRUM_NMSP]}

    # case of planning response json
    elif RESP_NMSP in json_data:
        json_data = json_data[RESP_NMSP]
    elif any(k in json_data for k in SIM_PARAMS_KEYS + [SPECTRUM_KEY, RESPONSE_KEY, PATH_REQUEST_KEY]):
        # then this is a legacy format json, nothing to convert
        pass
    else:
        raise ValueError('Unrecognized type of content (not topology, service or equipment)')
    return json_data


def main():
    """Conversion function
    """
    parser = ArgumentParser()
    parser.add_argument('--legacy-to-yang', nargs='?', type=Path,
                        help='convert file with this name into yangconformedname.json')
    parser.add_argument('--yang-to-legacy', nargs='?', type=Path,
                        help='convert file with this name into gnpy'
                             + ' using decimal instead of strings and null instead of [null]')
    parser.add_argument('--validate', nargs='?', type=Path,
                        help='validate yang conformity')
    parser.add_argument('-o', '--output', type=Path,
                        help='Stores into file with this name; default = GNPy_legacy_formatted-<file_name>.json or'
                        + 'GNPy_yang_formatted-<file_name>.json')
    args = parser.parse_args()

    output = None
    converted = None
    if args.validate:
        with open(args.validate, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            load_data(json.dumps(json_data))
        return 0
    elif args.legacy_to_yang:
        prefix = 'GNPy_yang_formatted-'
        with open(args.legacy_to_yang, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # note that dump_data automatically validate date against yang models
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
