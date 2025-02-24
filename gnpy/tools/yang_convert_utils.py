#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# Utils for yang <-> legacy format conversion
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Utils for yang <-> legacy format conversion
===========================================

Format conversion utils.
"""

from pathlib import Path
from copy import deepcopy
from typing import Dict, Union, List, Any, NamedTuple
from importlib import resources
import json
import os

import oopt_gnpy_libyang as ly

from gnpy.yang.precision_dict import PRECISION_DICT

ELEMENTS_KEY = 'elements'
ROADM_KEY = 'Roadm'
PARAMS_KEY = 'params'
METADATA_KEY = 'metadata'
LOCATION_KEY = 'location'
DEGREE_KEY = 'degree_uid'
PATH_REQUEST_KEY = 'path-request'
RESPONSE_KEY = 'response'
SPECTRUM_KEY = 'spectrum'
LOSS_COEF_KEY = 'loss_coef'
LOSS_COEF_KEY_PER_FREQ = 'loss_coef_per_frequency'
EQPT_TYPES = ['Edfa', 'Transceiver', 'Fiber', 'Roadm']
EDFA_CONFIG_KEYS = ['nf_fit_coef', 'nf_ripple', 'gain_ripple', 'dgt']
SIM_PARAMS_KEYS = ['raman_params', 'nli_params']
TOPO_NMSP = 'gnpy-network-topology:topology'
EQPT_NMSP = 'gnpy-eqpt-config:equipment'
SERV_NMSP = 'gnpy-path-computation:services'
RESP_NMSP = 'gnpy-path-computation:responses'
EDFA_CONFIG_NMSP = 'gnpy-edfa-config:edfa-config'
SIM_PARAMS_NMSP = 'gnpy-sim-params:sim-params'
SPECTRUM_NMSP = 'gnpy-spectrum:spectrum'


class PrettyFloat(float):
    """"A float subclass for formatting according to specific fraction digit requirements.

    >>> PrettyFloat(3.1245)
    3.12
    >>> PrettyFloat(100.65, 5)
    100.65
    >>> PrettyFloat(2.1e-5, 8)
    0.000021
    >>> PrettyFloat(10, 3)
    10.0
    >>> PrettyFloat(-0.3110761646066259, 18)
    -0.3110761646066259
    """
    def __new__(cls, value: float, fraction_digit: int = 2):
        """Create a new instance of PrettyFloat"""
        instance = super().__new__(cls, value)
        instance.fraction_digit = fraction_digit
        instance.value = value
        return instance

    def __repr__(self) -> str:
        """Return the string representation of the float formatted to the specified fraction digits. It removes
        scientific notation ("e-x").
        """
        # When fraction digit is over 16, the usual formatting does not works properly because of floating point issues.
        # For example -0.3110761646066259 is represented as "-0.311076164606625905". The following function makes
        # sure that the unwanted floating point issue does not change the value. Maximum fraction digit in YANG is 18.
        if self.fraction_digit in range(0, 19):
            temp = str(self.value)
            if 'e' in temp or '.' not in temp or self.fraction_digit < 17:
                formatted_value = f'{self:.{self.fraction_digit}f}'
                if '.' in formatted_value:
                    formatted_value = formatted_value.rstrip('0')
                    if formatted_value.endswith('.'):
                        formatted_value += '0'
                return formatted_value
            if '.' in temp:
                parts = temp.split('.')
                formatted_value = parts[0] + '.' + parts[1][0:min(self.fraction_digit, len(parts[1]))]
                formatted_value = formatted_value.rstrip('0')
                if formatted_value.endswith('.'):
                    formatted_value += '0'
                return formatted_value
            return temp
        raise ValueError(f'Fraction digit {self.fraction_digit} not handled')


def gnpy_precision_dict() -> Dict[str, int]:
    """Return a dictionary of fraction-digit definitions for GNPy.
    Precision correspond to fraction digit number if it is a decimal64 yang type, or 0 if it is an
    (u)int < 64 or -1 if it is a string or an (u)int64 type.

    :return: Dictionnary mapping key names with digit numbers for values.
    :rtype: Dict[str, int]
    """
    return PRECISION_DICT


def convert_dict(data: Dict, fraction_digit: int = 2, precision: Union[Dict[str, int], None] = None) \
        -> Union[Dict, List, float, int, str, None]:
    """Recursive conversion from float to str, conformed to RFC7951
    does not work for int64 (will not returm str as stated in standard)
    If nothing is stated precision is using gnpy_precision_dict.

    :param data: the input dictionary to convert.
    :type data: data: Dict
    :param fraction_digit: the number of decimal places to format.
    :type fraction_digit: int
    :param precision: A dictionary defining precision for specific keys.
    :type precision: Union[Dict[str, int], None]
    :return: A new dictionary with converted values.
    :rtype: Dict

    >>> convert_dict({"y": "amp", "t": "vn", "g": 25, "gamma": 0.0016, "p": 21.5, "o": True, \
    "output-power": 14.12457896})
    {'y': 'amp', 't': 'vn', 'g': '25.0', 'gamma': '0.0016', 'p': '21.5', 'o': True, 'output-power': '14.12457896'}
    """
    if not precision:
        precision = gnpy_precision_dict()
    if isinstance(data, dict):
        for k, v in data.items():
            fraction_digit = precision.get(k, 2)
            data[k] = convert_dict(v, fraction_digit, precision=precision)
    elif isinstance(data, list):
        temp = deepcopy(data)
        for i, el in enumerate(temp):
            if isinstance(el, float):
                data[i] = PrettyFloat(el, fraction_digit)
                data[i] = str(data[i])
            else:
                data[i] = convert_dict(el, fraction_digit=fraction_digit, precision=precision)
    elif isinstance(data, bool):
        return data
    elif isinstance(data, int):
        data = PrettyFloat(data)
        data.fraction_digit = fraction_digit
        if fraction_digit > 0:
            return str(data)
        if fraction_digit < 0:
            return data
        return int(data)
    elif isinstance(data, float):
        data = PrettyFloat(data)
        data.fraction_digit = fraction_digit
        return str(data)
    return data


def convert_back(data: Dict, fraction_digit: Union[int, None] = None, precision: Union[Dict[str, int], None] = None) \
        -> Union[Dict, List, float, int, str, None]:
    """Recursively convert strings back to their original types int, float according to RFC7951.

    :param data: the input dictionary to convert.
    :type data: Dict
    :param fraction_digit: the number of decimal places to format.
    :type fraction_digit: Union[int, None]
    :param precision: A dictionary defining precision for specific keys.
    :type precision: Union[Dict[str, int], None]
    :return: A new dictionary with converted values.
    :rtype: Dict

    >>> a = {'y': 'amp', 't': 'vn', 'N': '25', 'gamma': '0.0000000000000016', 'p': '21.50', 'o': True, \
      'output-power': '14.12458'}
    >>> convert_back({'a': a, 'delta_power_range_db': ['12.3', '10.6', True]})
    {'a': {'y': 'amp', 't': 'vn', 'N': 25, 'gamma': 1.6e-15, 'p': '21.50', 'o': True, 'output-power': 14.12458}, \
'delta_power_range_db': ['12.3', '10.6', True]}
    """
    if not precision:
        precision = gnpy_precision_dict()
    if isinstance(data, dict):
        for k, v in data.items():
            fraction_digit = None
            if k in precision:
                fraction_digit = precision[k]
            data[k] = convert_back(v, fraction_digit, precision=precision)
    elif isinstance(data, list):
        for i, el in enumerate(data):
            if isinstance(el, str) and fraction_digit not in [None, -1]:
                data[i] = float(data[i])
            else:
                data[i] = convert_back(el, fraction_digit=fraction_digit, precision=precision)
    elif isinstance(data, bool):
        return data
    elif isinstance(data, int):
        return data
    elif isinstance(data, float):
        return data
    elif isinstance(data, str) and fraction_digit is not None:
        if fraction_digit > 0:
            return float(data)
        if fraction_digit < 0:
            return data
        return int(data)
    return data


def model_path() -> Path:
    """Filesystem path to YANG models.

    return: path to the GNPy YANG modules.
    rtype: Path
    """
    return Path(__file__).parent.parent / 'yang'


def external_yang() -> Path:
    """Filesystem to the IETF external yang modules.

    return: path to the IETF modules.
    rtype: Path
    """
    return Path(__file__).parent.parent / 'yang' / 'ext'


def yang_lib() -> Path:
    """Path to the json library of needed yang modules.

    return: path to the library describing all modules and revisions for this gnpy release.
    rtype: Path
    """
    return Path(__file__).parent.parent / 'yang' / 'yang-library-gnpy.json'


def _create_context(yang_library) -> ly.Context:
    """Prepare a libyang context for validating data against GNPy YANG models.

    :param yang_library: path to the library describing all modules and revisions to be considered for the formatted
                         string generation.
    :type yang_library: Path
    :return: Context used to hold all information about schemas.
    :rtype: ly.Context
    """
    ly.set_log_options(ly.LogOptions.Log | ly.LogOptions.Store)
    ctx = ly.Context(str(model_path()) + os.pathsep + str(external_yang()),
                     ly.ContextOptions.AllImplemented | ly.ContextOptions.DisableSearchCwd)
    with open(yang_library, 'r', encoding='utf-8') as file:
        data = json.load(file)
    yang_modules = [{'name': e['name'], 'revision': e['revision']}
                    for e in data['ietf-yang-library:modules-state']['module']]
    for module in yang_modules:
        ctx.load_module(module['name'], revision=module['revision'])
    return ctx


class ErrorMessage(NamedTuple):
    what: str
    where: str


def load_data(s: str, yang_library: Path = yang_lib()) -> ly.DataNode:
    """Load data from YANG-based JSON input and validate them.

    :param data: a string contating the json data to be loaded.
    :type data: str
    :param yang_library: path to the library describing all modules and revisions to be considered for the formatted
                         string generation.
    :type yang_library: Path
    :return: DataNode containing the loaded data
    :rtype: ly.DataNode
    """
    ctx = _create_context(yang_library)
    try:
        data = ctx.parse_data(s, ly.DataFormat.JSON,
                              ly.ParseOptions.Strict | ly.ParseOptions.Ordered,
                              ly.ValidationOptions.Present
                              | ly.ValidationOptions.MultiError)
    except ly.Error as exc:
        raise ly.Error(exc, [ErrorMessage(err.message, err.path) for err in ctx.errors()]) from None
    return data


def dump_data(data: Dict, yang_library: Path = yang_lib()) -> str:
    """Creates a formatted string using oopt-gnpy-libyang.

    :param data: a json dict with data already formatted
    :type data: Dict
    :param yang_library: path to the library describing all modules and revisions to be considered for the formatted
                         string generation.
    :type yang_library: Path
    :return: formatted string data
    :rtype: str
    """
    return load_data(json.dumps(data), yang_library).print(ly.DataFormat.JSON, ly.PrintFlags.WithSiblings)


def convert_degree(json_data: Dict) -> Dict:
    """Convert legacy json topology format to gnpy yang format revision 2025-01-20:

    :param json_data: The input JSON topology data to convert.
    :type json_data: Dict
    :return: the converted JSON data
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if elem['type'] == ROADM_KEY and PARAMS_KEY in elem:
            new_targets = []
            for equalization_type in ['per_degree_pch_out_db', 'per_degree_psd_out_mWperGHz',
                                      'per_degree_psd_out_mWperSlotWidth']:
                targets = elem[PARAMS_KEY].pop(equalization_type, None)
                if targets:
                    new_targets.extend([{DEGREE_KEY: degree, equalization_type: target}
                                        for degree, target in targets.items()])
            if new_targets:
                elem[PARAMS_KEY]['per_degree_power_targets'] = new_targets
    return json_data


def convert_back_degree(json_data: Dict) -> Dict:
    """Convert gnpy yang format back to legacy json topology format.

    :param json_data: The input JSON topology data to convert back.
    :return: the converted back JSON data
    """
    for elem in json_data[ELEMENTS_KEY]:
        if elem['type'] != ROADM_KEY or PARAMS_KEY not in elem:
            continue
        power_targets = elem[PARAMS_KEY].pop('per_degree_power_targets', None)
        if not power_targets:
            continue
        # Process each power target
        process_power_targets(elem, power_targets)
    return json_data


def process_power_targets(elem: Dict, power_targets: List[Dict]) -> None:
    """Process power targets and update element parameters.

    :param elem: The element to update
    :param power_targets: List of power target configurations
    """
    equalization_types = [
        'per_degree_pch_out_db',
        'per_degree_psd_out_mWperGHz',
        'per_degree_psd_out_mWperSlotWidth'
    ]

    for target in power_targets:
        degree_uid = target[DEGREE_KEY]
        for eq_type in equalization_types:
            if eq_type not in target:
                continue
            # Initialize the equalization type dict if needed
            if eq_type not in elem[PARAMS_KEY]:
                elem[PARAMS_KEY][eq_type] = {}
            # Set the value for this degree
            elem[PARAMS_KEY][eq_type][degree_uid] = target[eq_type]


def convert_loss_coeff_list(json_data: Dict) -> Dict:
    """Convert legacy json topology format to gnpy yang format revision 2025-01-20:

    :param json_data: The input JSON topology data to convert.
    :type json_data: Dict
    :return: the converted JSON data
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if PARAMS_KEY in elem and LOSS_COEF_KEY in elem[PARAMS_KEY] \
                and isinstance(elem[PARAMS_KEY][LOSS_COEF_KEY], dict):
            loss_coef_per_frequency = elem[PARAMS_KEY].pop(LOSS_COEF_KEY)
            loss_coef_list = loss_coef_per_frequency.pop('value', None)
            frequency_list = loss_coef_per_frequency.pop('frequency', None)
            if loss_coef_list:
                new_loss_coef_per_frequency = [{'frequency': f, 'value': v}
                                               for f, v in zip(frequency_list, loss_coef_list)]
            elem[PARAMS_KEY][LOSS_COEF_KEY_PER_FREQ] = new_loss_coef_per_frequency
    return json_data


def convert_back_loss_coeff_list(json_data: Dict) -> Dict:
    """Convert gnpy yang format revision 2025-01-20 back to legacy json topology format

    :param json_data: The input JSON topology data to convert back
    :type json_data: Dict
    :return: the converted JSON data
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if PARAMS_KEY in elem and LOSS_COEF_KEY_PER_FREQ in elem[PARAMS_KEY]:
            loss_coef_per_frequency = elem[PARAMS_KEY].pop(LOSS_COEF_KEY_PER_FREQ)
            if loss_coef_per_frequency:
                new_loss_coef_per_frequency = {'frequency': [item['frequency'] for item in loss_coef_per_frequency],
                                               'value': [item['value'] for item in loss_coef_per_frequency]}
            elem[PARAMS_KEY]['loss_coef'] = new_loss_coef_per_frequency
    return json_data


def convert_design_band(json_data: Dict) -> Dict:
    """Convert legacy json topology format to gnpy yang format revision 2025-01-20:

    :param json_data: The input JSON topology data to convert.
    :type json_data: Dict
    :return: the converted JSON data
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if elem['type'] == ROADM_KEY and PARAMS_KEY in elem:
            new_targets = []
            targets = elem[PARAMS_KEY].pop('per_degree_design_bands', None)
            if targets:
                new_targets.extend([{DEGREE_KEY: degree, 'design_bands': target}
                                    for degree, target in targets.items()])
            if new_targets:
                elem[PARAMS_KEY]['per_degree_design_bands_targets'] = new_targets
    return json_data


def convert_back_design_band(json_data: Dict) -> Dict:
    """Convert gnpy yang format revision 2025-01-20 back to legacy json topology format

    :param json_data: The input JSON topology data to convert back
    :type json_data: Dict
    :return: the converted JSON data
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if elem['type'] == ROADM_KEY and PARAMS_KEY in elem:
            targets = elem[PARAMS_KEY].pop('per_degree_design_bands_targets', None)
            if targets:
                design_bands = {}
                for target in targets:
                    design_bands[target[DEGREE_KEY]] = target['design_bands']
                if design_bands:
                    elem[PARAMS_KEY]['per_degree_design_bands'] = design_bands
    return json_data


def convert_range_to_dict(range_values: List[float]) -> Dict[str, float]:
    """Convert a range list to a dictionary format:

    :param range_values: range of loat values defined with the format [min, max, step].
    :type range_value: List[float]
    :return: range formatted as a dict {"min_value": min, "max_value": max, "step": step}
    :rtype: Dict[str, float]
    """
    return {
        'min_value': range_values[0],
        'max_value': range_values[1],
        'step': range_values[2]
    }


def process_span_data(span: Dict) -> None:
    """Convert Span data with range in dict format
    :param span: The span data to process.
    :type span: Dict
    """
    if 'delta_power_range_dict_db' in span:
        return

    if 'delta_power_range_db' not in span:
        raise KeyError('delta_power_range or delta_power_range_dict_db missing in Span dict.')

    delta_power_range_db = span.get('delta_power_range_db', [0, 0, 0])
    span['delta_power_range_dict_db'] = convert_range_to_dict(delta_power_range_db)
    del span['delta_power_range_db']


def process_si_data(si: Dict) -> None:
    """Convert Span data with range in dict format
    :param si: The span data to process.
    :type si: Dict
    """
    if 'power_range_dict_db' in si:
        return

    if 'power_range_db' not in si:
        raise KeyError('power_range_db or power_range_dict_db missing in SI dict.')

    power_range_db = si.get('power_range_db', [0, 0, 0])
    si['power_range_dict_db'] = convert_range_to_dict(power_range_db)
    del si['power_range_db']


def convert_delta_power_range(json_data: Dict) -> Dict:
    """Convert legacy json equipment format to GNPy yang format revision 2025-01-20

    :param json_data: the input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data.
    :rtype: Dict
    """
    if 'Span' in json_data:
        for span in json_data['Span']:
            process_span_data(span)

    if 'SI' in json_data:
        for si in json_data['SI']:
            process_si_data(si)

    return json_data


def convert_back_delta_power_range(json_data: Dict) -> Dict:
    """Convert Yang JSON revision 2025-01-20 equipment format to legacy GNPy format.

    :param json_data: the input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data.
    :rtype: Dict
    """
    if 'Span' in json_data and 'delta_power_range_dict_db' in json_data['Span'][0]:
        delta_power_range_db = json_data['Span'][0]['delta_power_range_dict_db']
        json_data['Span'][0]['delta_power_range_db'] = [
            delta_power_range_db['min_value'],
            delta_power_range_db['max_value'],
            delta_power_range_db['step']]
        del json_data['Span'][0]['delta_power_range_dict_db']
    if 'SI' in json_data and 'power_range_dict_db' in json_data['SI'][0]:
        power_range_db = json_data['SI'][0]['power_range_dict_db']
        json_data['SI'][0]['power_range_db'] = [
            power_range_db['min_value'],
            power_range_db['max_value'],
            power_range_db['step']]
        del json_data['SI'][0]['power_range_dict_db']
    return json_data


def add_missing_default_type_variety(json_data: Dict) -> Dict:
    """Case of ROADM: legacy does not enforce type_variety to be present.
    This utils ensures that 'default' type_variety is inserted if the key is missing.

    :param json_data: the input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data.
    :rtype: Dict
    """
    if 'Roadm' not in json_data:
        return json_data
    for i, elem in enumerate(json_data['Roadm']):
        if 'type_variety' not in elem:
            # make sure type_variety is the first key in the elem
            temp = {'type_variety': 'default'}
            temp.update(elem)
            json_data['Roadm'][i] = temp
            break
    return json_data


def remove_null_region_city(json_data: Dict) -> Dict:
    """if present, name should not be None.

    :param json_data: the input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data.
    :rtype: Dict
    """
    for elem in json_data[ELEMENTS_KEY]:
        if "metadata" in elem and "location" in elem[METADATA_KEY]:
            for name in ['city', 'region']:
                if name in elem[METADATA_KEY][LOCATION_KEY] \
                        and elem[METADATA_KEY][LOCATION_KEY][name] is None:
                    elem[METADATA_KEY][LOCATION_KEY][name] = ""
    return json_data


def remove_union_that_fail(json_data: Dict) -> Dict:
    """Convert GNPy legacy JSON request format to GNPy yang format revision 2025-01-20
    If present "N": or "M": should not contain empy data.
    If present max-nb-of-channel should not contain empty data.

    :param json_data: the input JSON data to convert.
    :type json_data: Dict
    :return: The converted JSON data.
    :rtype: Dict
    """
    for elem in json_data[PATH_REQUEST_KEY]:
        te = elem['path-constraints']['te-bandwidth']
        freq_slot = te.get('effective-freq-slot', None)
        if freq_slot:
            for slot in freq_slot:
                if slot.get('N', None) is None:
                    slot.pop('N', None)
                if slot.get('M', None) is None:
                    slot.pop('M', None)
                if not slot:
                    te['effective-freq-slot'].remove(slot)
            if not te['effective-freq-slot']:
                te.pop('effective-freq-slot', None)
        for attribute in ['max-nb-of-channel', 'trx_mode', 'output-power']:
            if te.get(attribute) is None:
                te.pop(attribute, None)
    return json_data


def convert_none_to_empty(json_data: Any):
    """Convert all instances of None in the input to [None].

    This function recursively traverses the input and replaces any None
    values with a list containing None. If the input is already a list
    containing None, it returns the input unchanged.

    :param json_data: The input data to process, which can be of any type.
    :type json_data: Any
    :return: A new representation of the input with None values replaced by [None].
    :rtype: Any

    :example:
    >>> a = {'uid': '[930/WRT-2-2-SIG=>923/WRT-1-9-SIG]-923/AMP-1-13', 'type_variety': 'AMP',
    ... 'metadata': {'location': {'latitude': 0.0, 'longitude': 0.0, 'city': 'Zion', 'region': ''}},
    ... 'type': 'Multiband_amplifier', 'amplifiers': [{'type_variety': 'AMP_LOW_C',
    ... 'operational': {'gain_target': 12.22, 'delta_p': 4.19, 'out_voa': None, 'tilt_target': 0.0,
    ... 'f_min': 191.3, 'f_max': 196.1}}, {'type_variety': 'AMP_LOW_L',
    ... 'operational': {'gain_target': 12.05, 'delta_p': 4.19, 'out_voa': None, 'tilt_target': 0.0,
    ... 'f_min': 186.1, 'f_max': 190.9}}]}
    >>> convert_none_to_empty(a)
    {'uid': '[930/WRT-2-2-SIG=>923/WRT-1-9-SIG]-923/AMP-1-13', 'type_variety': 'AMP', \
'metadata': {'location': {'latitude': 0.0, 'longitude': 0.0, 'city': 'Zion', 'region': ''}}, \
'type': 'Multiband_amplifier', 'amplifiers': [{'type_variety': 'AMP_LOW_C', \
'operational': {'gain_target': 12.22, 'delta_p': 4.19, 'out_voa': [None], 'tilt_target': 0.0, \
'f_min': 191.3, 'f_max': 196.1}}, {'type_variety': 'AMP_LOW_L', \
'operational': {'gain_target': 12.05, 'delta_p': 4.19, 'out_voa': [None], 'tilt_target': 0.0, \
'f_min': 186.1, 'f_max': 190.9}}]}

    """
    if json_data == [None]:
        # already conformed
        return json_data
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            json_data[key] = convert_none_to_empty(value)
    elif isinstance(json_data, list):
        for i, elem in enumerate(json_data):
            json_data[i] = convert_none_to_empty(elem)
    elif json_data is None:
        return [None]
    return json_data


def convert_empty_to_none(json_data: Any):
    """Convert all instances of [None] in the input to None.

    This function recursively traverses the input data and replaces any
    lists containing a single None element with None. If the input is
    already None, it returns None unchanged.

    :param json_data: The input data to process, which can be of any type.
    :type json_data: Any
    :return: A new representation of the input with [None] replaced by None.
    :rtype: Any

    >>> json_data = {
    ...     "uid": "[930/WRT-2-2-SIG=>923/WRT-1-9-SIG]-923/AMP-1-13",
    ...     "type_variety": "AMP",
    ...     "metadata": {
    ...         "location": {
    ...             "latitude": 0.000000,
    ...             "longitude": 0.000000,
    ...             "city": "Zion",
    ...             "region": ""
    ...         }
    ...     },
    ...     "type": "Multiband_amplifier",
    ...     "amplifiers": [{
    ...             "type_variety": "AMP_LOW_C",
    ...             "operational": {
    ...                 "gain_target": 12.22,
    ...                 "delta_p": 4.19,
    ...                 "out_voa": [None],
    ...                 "tilt_target": 0.00,
    ...                 "f_min": 191.3,
    ...                 "f_max": 196.1
    ...             }
    ...         }, {
    ...             "type_variety": "AMP_LOW_L",
    ...             "operational": {
    ...                 "gain_target": 12.05,
    ...                 "delta_p": 4.19,
    ...                 "out_voa": [None],
    ...                 "tilt_target": 0.00,
    ...                 "f_min": 186.1,
    ...                 "f_max": 190.9
    ...             }
    ...         }
    ...     ]
    ... }
    >>> convert_empty_to_none(json_data)
    {'uid': '[930/WRT-2-2-SIG=>923/WRT-1-9-SIG]-923/AMP-1-13', 'type_variety': 'AMP', \
'metadata': {'location': {'latitude': 0.0, 'longitude': 0.0, 'city': 'Zion', 'region': ''}}, \
'type': 'Multiband_amplifier', 'amplifiers': [{'type_variety': 'AMP_LOW_C', \
'operational': {'gain_target': 12.22, 'delta_p': 4.19, 'out_voa': None, 'tilt_target': 0.0, \
'f_min': 191.3, 'f_max': 196.1}}, {'type_variety': 'AMP_LOW_L', \
'operational': {'gain_target': 12.05, 'delta_p': 4.19, 'out_voa': None, 'tilt_target': 0.0, \
'f_min': 186.1, 'f_max': 190.9}}]}

    """
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            json_data[key] = convert_empty_to_none(value)
    elif isinstance(json_data, list):
        if len(json_data) == 1 and json_data[0] is None:
            return None
        for i, elem in enumerate(json_data):
            json_data[i] = convert_empty_to_none(elem)
    return json_data


def remove_namespace_context(json_data, namespace):
    """Serialisation with yang introduces a namespace in values that
    are defined as identity. this function filter them out.

    >>> a = [{"a": 123, "b": "123:alkdje"}, {"a": 456, "c": "123", "d": "123:123"}]
    >>> remove_namespace_context(a, "123:")
    [{'a': 123, 'b': 'alkdje'}, {'a': 456, 'c': '123', 'd': '123'}]

    """
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            json_data[key] = remove_namespace_context(value, namespace)
    elif isinstance(json_data, list):
        for i, elem in enumerate(json_data):
            json_data[i] = remove_namespace_context(elem, namespace)
    elif isinstance(json_data, str) and namespace in json_data:
        return json_data.split(namespace)[1]
    return json_data


def dict_to_list_range(json_data):
    """Change {"min_value": 0.00, "max_value": 2.00, "step": 0.00} into
    [0.00, 2.00, 0.00]
    keeping the correct order
    """
    return [json_data['min_value'], json_data['max_value'], json_data['step']]


def list_to_dict_range(json_data):
    """Change [0.00, 2.00, 0.00] into
    {"min_value": 0.00, "max_value": 2.00, "step": 0.00}
    keeping the correct order
    """
    return {"min_value": json_data[0], "max_value": json_data[1], "step": json_data[2]}


def restore_ranges_as_list(json_data, dict_keys):
    """Change {"min_value": 0.00, "max_value": 2.00, "step": 0.00} into
    [0.00, 2.00, 0.00] for Span[0][delta_power_range_db] and SI[0][power_range_db]
    """
    for key, attribute in dict_keys.items():
        dict_name = attribute['dict_name']
        list_name = attribute['list_name']
        elem = json_data[key][0].pop(dict_name, None)
        if elem:
            json_data[key][0][list_name] = dict_to_list_range(elem)
    return json_data


def restore_ranges_as_dict(json_data, dict_keys):
    """Change [0.00, 2.00, 0.00] into
    {"min_value": 0.00, "max_value": 2.00, "step": 0.00} for Span[0][delta_power_range_db] and SI[0][power_range_db]
    """
    for key, attribute in dict_keys.items():
        dict_name = attribute['dict_name']
        list_name = attribute['list_name']
        elem = json_data[key][0].pop(list_name, None)
        if elem:
            json_data[key][0][dict_name] = list_to_dict_range(elem)
    return json_data
