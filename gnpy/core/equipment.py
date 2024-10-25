#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.equipment
===================

This module contains functionality for specifying equipment.
"""
from collections import defaultdict
from functools import reduce
from typing import List

from gnpy.core.exceptions import EquipmentConfigError, ConfigurationError


def trx_mode_params(equipment, trx_type_variety='', trx_mode='', error_message=False):
    """return the trx and SI parameters from eqpt_config for a given type_variety and mode (ie format)

    if the type or mode do no match an existing transceiver in the library, then the function
    raises an error if error_message is True else returns a default mode based on equipment['SI']['default']
    If trx_mode is None (but type is valid), it returns an undetermined mode whatever the error message:
    this is a special case for automatic mode selection.
    """
    trx_params = {}
    default_si_data = equipment['SI']['default']
    # default transponder characteristics
    # mainly used with transmission_main_example.py
    default_trx_params = {
        'f_min': default_si_data.f_min,
        'f_max': default_si_data.f_max,
        'baud_rate': default_si_data.baud_rate,
        'spacing': default_si_data.spacing,
        'OSNR': None,
        'penalties': {},
        'bit_rate': None,
        'cost': None,
        'roll_off': default_si_data.roll_off,
        'tx_osnr': default_si_data.tx_osnr,
        'min_spacing': None,
        'equalization_offset_db': 0
    }
    # Undetermined transponder characteristics
    # mainly used with path_request_run.py for the automatic mode computation case
    undetermined_trx_params = {
        "format": "undetermined",
        "baud_rate": None,
        "OSNR": None,
        "penalties": None,
        "bit_rate": None,
        "roll_off": None,
        "tx_osnr": None,
        "min_spacing": None,
        "cost": None,
        "equalization_offset_db": 0
    }

    trxs = equipment['Transceiver']
    if trx_type_variety in trxs:
        modes = {mode['format']: mode for mode in trxs[trx_type_variety].mode}
        trx_frequencies = {'f_min': trxs[trx_type_variety].frequency['min'],
                           'f_max': trxs[trx_type_variety].frequency['max']}
        if trx_mode in modes:
            # if called from transmission_main.py, trx_mode is ''
            trx_params = {**modes[trx_mode], **trx_frequencies}
            if trx_params['baud_rate'] > trx_params['min_spacing']:
                # sanity check: baudrate must be smaller than min spacing
                raise EquipmentConfigError(f'Inconsistency in equipment library:\n Transponder "{trx_type_variety}" '
                                           + f'mode "{trx_params["format"]}" has baud rate '
                                           + f'{trx_params["baud_rate"] * 1e-9:.2f} GHz greater than min_spacing '
                                           + f'{trx_params["min_spacing"] * 1e-9:.2f}.')
            trx_params['equalization_offset_db'] = trx_params.get('equalization_offset_db', 0)
            return trx_params
        if trx_mode is None:
            # if called from path_requests_run.py, trx_mode is filled with None when not specified by user
            trx_params = {**undetermined_trx_params, **trx_frequencies}
            return trx_params
    if trx_type_variety in trxs and error_message:
        raise EquipmentConfigError(f'Could not find transponder "{trx_type_variety}" with mode "{trx_mode}" '
                                   + 'in equipment library')
    if error_message:
        raise EquipmentConfigError(f'Could not find transponder "{trx_type_variety}" in equipment library')

    trx_params = {**default_trx_params}
    return trx_params


def find_type_variety(amps: List[str], equipment: dict) -> List[str]:
    """Returns the multiband type_variety associated with a list of single band type_varieties
    Args:
    amps (List[str]): A list of single band type_varieties.
    equipment (dict): A dictionary containing equipment information.

    Returns:
    str: an amplifier type variety
    """
    listes = find_type_varieties(amps, equipment)

    _found_type = list(reduce(lambda x, y: set(x) & set(y), listes))
    # Given a list of single band amplifiers, find the multiband amplifier whose multi_band group
    # matches. For example, if amps list contains ["a1_LBAND", "a2_CBAND"], with a1.multi_band = [a1_LBAND, a1_CBAND]
    # and a2.multi_band = [a1_LBAND, a2_CBAND], then:
    # possible_type_varieties = {"a1_LBAND": ["a1", "a2"], "a2_CBAND": ["a2"]}
    # listes = [["a1", "a2"],  ["a2"]]
    # and _found_type = [a2]
    if not _found_type:
        msg = f'{amps} amps do not belong to the same amp type {listes}'
        raise ConfigurationError(msg)
    return _found_type


def find_type_varieties(amps: List[str], equipment: dict) -> List[List[str]]:
    """Returns the multiband list of type_varieties associated with a list of single band type_varieties
    Args:
    amps (List[str]): A list of single band type_varieties.
    equipment (dict): A dictionary containing equipment information.

    Returns:
    List[List[str]]: A list of lists containing the multiband type_varieties
    associated with each single band type_variety.
    """
    possible_type_varieties = defaultdict(list)
    for amp_name, amp in equipment['Edfa'].items():
        if amp.multi_band is not None:
            for elem in amp.multi_band:
                # possible_type_varieties stores the list of multiband amp names that list this elem as
                # a possible amplifier of the multiband group. For example, if "std_medium_gain_multiband"
                # and "std_medium_gain_multiband_new" contain "std_medium_gain_C" in their "multi_band" list, then:
                # possible_type_varieties["std_medium_gain_C"] =
                # ["std_medium_gain_multiband", "std_medium_gain_multiband_new"]
                possible_type_varieties[elem].append(amp_name)
    return [possible_type_varieties[a] for a in amps]
