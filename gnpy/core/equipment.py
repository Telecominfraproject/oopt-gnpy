#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.equipment
===================

This module contains functionality for specifying equipment.
'''

from gnpy.core.utils import automatic_nch, db2lin
from gnpy.core.exceptions import EquipmentConfigError


def trx_mode_params(equipment, trx_type_variety='', trx_mode='', error_message=False):
    """return the trx and SI parameters from eqpt_config for a given type_variety and mode (ie format)"""
    trx_params = {}
    default_si_data = equipment['SI']['default']

    try:
        trxs = equipment['Transceiver']
        # if called from path_requests_run.py, trx_mode is filled with None when not specified by user
        # if called from transmission_main.py, trx_mode is ''
        if trx_mode is not None:
            mode_params = next(mode for trx in trxs
                               if trx == trx_type_variety
                               for mode in trxs[trx].mode
                               if mode['format'] == trx_mode)
            trx_params = {**mode_params}
            # sanity check: spacing baudrate must be smaller than min spacing
            if trx_params['baud_rate'] > trx_params['min_spacing']:
                raise EquipmentConfigError(f'Inconsistency in equipment library:\n Transpoder "{trx_type_variety}" mode "{trx_params["format"]}" ' +
                                           f'has baud rate {trx_params["baud_rate"]*1e-9} GHz greater than min_spacing {trx_params["min_spacing"]*1e-9}.')
        else:
            mode_params = {"format": "undetermined",
                           "baud_rate": None,
                           "OSNR": None,
                           "bit_rate": None,
                           "roll_off": None,
                           "tx_osnr": None,
                           "min_spacing": None,
                           "cost": None}
            trx_params = {**mode_params}
        trx_params['f_min'] = equipment['Transceiver'][trx_type_variety].frequency['min']
        trx_params['f_max'] = equipment['Transceiver'][trx_type_variety].frequency['max']

        # TODO: novel automatic feature maybe unwanted if spacing is specified
        # trx_params['spacing'] = _automatic_spacing(trx_params['baud_rate'])
        # temp = trx_params['spacing']
        # print(f'spacing {temp}')
    except StopIteration:
        if error_message:
            raise EquipmentConfigError(f'Could not find transponder "{trx_type_variety}" with mode "{trx_mode}" in equipment library')
        else:
            # default transponder charcteristics
            # mainly used with transmission_main_example.py
            trx_params['f_min'] = default_si_data.f_min
            trx_params['f_max'] = default_si_data.f_max
            trx_params['baud_rate'] = default_si_data.baud_rate
            trx_params['spacing'] = default_si_data.spacing
            trx_params['OSNR'] = None
            trx_params['bit_rate'] = None
            trx_params['cost'] = None
            trx_params['roll_off'] = default_si_data.roll_off
            trx_params['tx_osnr'] = default_si_data.tx_osnr
            trx_params['min_spacing'] = None
            nch = automatic_nch(trx_params['f_min'], trx_params['f_max'], trx_params['spacing'])
            trx_params['nb_channel'] = nch
            print(f'There are {nch} channels propagating')

    trx_params['power'] = db2lin(default_si_data.power_dbm) * 1e-3

    return trx_params
