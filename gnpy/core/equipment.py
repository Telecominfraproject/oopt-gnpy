#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.equipment
===================

This module contains functionality for specifying equipment.
'''

from pathlib import Path
from json import load
from gnpy.core.utils import automatic_nch, db2lin, load_json
from collections import namedtuple
from gnpy.core import ansi_escapes
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.core.science_utils import estimate_nf_model
import time

Model_vg = namedtuple('Model_vg', 'nf1 nf2 delta_p')
Model_fg = namedtuple('Model_fg', 'nf0')
Model_openroadm = namedtuple('Model_openroadm', 'nf_coef')
Model_hybrid = namedtuple('Model_hybrid', 'nf_ram gain_ram edfa_variety')
Model_dual_stage = namedtuple('Model_dual_stage', 'preamp_variety booster_variety')


class common:
    def update_attr(self, default_values, kwargs, name):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in default_values.items():
            setattr(self, k, clean_kwargs.get(k, v))
            if k not in clean_kwargs and name != 'Amp':
                print(ansi_escapes.red +
                      f'\n WARNING missing {k} attribute in eqpt_config.json[{name}]' +
                      f'\n default value is {k} = {v}' +
                      ansi_escapes.reset)
                time.sleep(1)


class SI(common):
    default_values = {
        "f_min": 191.35e12,
        "f_max": 196.1e12,
        "baud_rate": 32e9,
        "spacing": 50e9,
        "power_dbm": 0,
        "power_range_db": [0, 0, 0.5],
        "roll_off": 0.15,
        "tx_osnr": 45,
        "sys_margins": 0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'SI')


class Span(common):
    default_values = {
        'power_mode': True,
        'delta_power_range_db': None,
        'max_fiber_lineic_loss_for_raman': 0.25,
        'target_extended_gain': 2.5,
        'max_length': 150,
        'length_units': 'km',
        'max_loss': None,
        'padding': 10,
        'EOL': 0,
        'con_in': 0,
        'con_out': 0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Span')


class Roadm(common):
    default_values = {
        'target_pch_out_db': -17,
        'add_drop_osnr': 100,
        'restrictions': {
            'preamp_variety_list': [],
            'booster_variety_list': []
        }
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Roadm')


class Transceiver(common):
    default_values = {
        'type_variety': None,
        'frequency': None,
        'mode': {}
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Transceiver')


class Fiber(common):
    default_values = {
        'type_variety': '',
        'dispersion': None,
        'gamma': 0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Fiber')


class RamanFiber(common):
    default_values = {
        'type_variety': '',
        'dispersion': None,
        'gamma': 0,
        'raman_efficiency': None
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'RamanFiber')
        for param in ('cr', 'frequency_offset'):
            if param not in self.raman_efficiency:
                raise EquipmentConfigError(f'RamanFiber.raman_efficiency: missing "{param}" parameter')
        if self.raman_efficiency['frequency_offset'] != sorted(self.raman_efficiency['frequency_offset']):
            raise EquipmentConfigError(f'RamanFiber.raman_efficiency.frequency_offset is not sorted')


class Amp(common):
    default_values = {
        'f_min': 191.35e12,
        'f_max': 196.1e12,
        'type_variety': '',
        'type_def': '',
        'gain_flatmax': None,
        'gain_min': None,
        'p_max': None,
        'nf_model': None,
        'dual_stage_model': None,
        'nf_fit_coeff': None,
        'nf_ripple': None,
        'dgt': None,
        'gain_ripple': None,
        'out_voa_auto': False,
        'allowed_for_design': False,
        'raman': False
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Amp')

    @classmethod
    def from_json(cls, filename, **kwargs):
        config = Path(filename).parent / 'default_edfa_config.json'

        type_variety = kwargs['type_variety']
        type_def = kwargs.get('type_def', 'variable_gain')  # default compatibility with older json eqpt files
        nf_def = None
        dual_stage_def = None

        if type_def == 'fixed_gain':
            try:
                nf0 = kwargs.pop('nf0')
            except KeyError:  # nf0 is expected for a fixed gain amp
                raise EquipmentConfigError(f'missing nf0 value input for amplifier: {type_variety} in equipment config')
            for k in ('nf_min', 'nf_max'):
                try:
                    del kwargs[k]
                except KeyError:
                    pass
            nf_def = Model_fg(nf0)
        elif type_def == 'advanced_model':
            config = Path(filename).parent / kwargs.pop('advanced_config_from_json')
        elif type_def == 'variable_gain':
            gain_min, gain_max = kwargs['gain_min'], kwargs['gain_flatmax']
            try:  # nf_min and nf_max are expected for a variable gain amp
                nf_min = kwargs.pop('nf_min')
                nf_max = kwargs.pop('nf_max')
            except KeyError:
                raise EquipmentConfigError(f'missing nf_min or nf_max value input for amplifier: {type_variety} in equipment config')
            try:  # remove all remaining nf inputs
                del kwargs['nf0']
            except KeyError:
                pass  # nf0 is not needed for variable gain amp
            nf1, nf2, delta_p = estimate_nf_model(type_variety, gain_min, gain_max, nf_min, nf_max)
            nf_def = Model_vg(nf1, nf2, delta_p)
        elif type_def == 'openroadm':
            try:
                nf_coef = kwargs.pop('nf_coef')
            except KeyError:  # nf_coef is expected for openroadm amp
                raise EquipmentConfigError(f'missing nf_coef input for amplifier: {type_variety} in equipment config')
            nf_def = Model_openroadm(nf_coef)
        elif type_def == 'dual_stage':
            try:  # nf_ram and gain_ram are expected for a hybrid amp
                preamp_variety = kwargs.pop('preamp_variety')
                booster_variety = kwargs.pop('booster_variety')
            except KeyError:
                raise EquipmentConfigError(f'missing preamp/booster variety input for amplifier: {type_variety} in equipment config')
            dual_stage_def = Model_dual_stage(preamp_variety, booster_variety)

        with open(config, encoding='utf-8') as f:
            json_data = load(f)

        return cls(**{**kwargs, **json_data,
                      'nf_model': nf_def, 'dual_stage_model': dual_stage_def})


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


def _automatic_spacing(baud_rate):
    """return the min possible channel spacing for a given baud rate"""
    # TODO : this should parametrized in a cfg file
    # list of possible tuples [(max_baud_rate, spacing_for_this_baud_rate)]
    spacing_list = [(33e9, 37.5e9), (38e9, 50e9), (50e9, 62.5e9), (67e9, 75e9), (92e9, 100e9)]
    return min((s[1] for s in spacing_list if s[0] > baud_rate), default=baud_rate * 1.2)


def load_equipment(filename):
    json_data = load_json(filename)
    return _equipment_from_json(json_data, filename)


def _update_trx_osnr(equipment):
    """add sys_margins to all Transceivers OSNR values"""
    for trx in equipment['Transceiver'].values():
        for m in trx.mode:
            m['OSNR'] = m['OSNR'] + equipment['SI']['default'].sys_margins
    return equipment


def _update_dual_stage(equipment):
    edfa_dict = equipment['Edfa']
    for edfa in edfa_dict.values():
        if edfa.type_def == 'dual_stage':
            edfa_preamp = edfa_dict[edfa.dual_stage_model.preamp_variety]
            edfa_booster = edfa_dict[edfa.dual_stage_model.booster_variety]
            for key, value in edfa_preamp.__dict__.items():
                attr_k = 'preamp_' + key
                setattr(edfa, attr_k, value)
            for key, value in edfa_booster.__dict__.items():
                attr_k = 'booster_' + key
                setattr(edfa, attr_k, value)
            edfa.p_max = edfa_booster.p_max
            edfa.gain_flatmax = edfa_booster.gain_flatmax + edfa_preamp.gain_flatmax
            if edfa.gain_min < edfa_preamp.gain_min:
                raise EquipmentConfigError(f'Dual stage {edfa.type_variety} minimal gain is lower than its preamp minimal gain')
    return equipment


def _roadm_restrictions_sanity_check(equipment):
    """ verifies that booster and preamp restrictions specified in roadm equipment are listed
    in the edfa.
    """
    restrictions = equipment['Roadm']['default'].restrictions['booster_variety_list'] + \
        equipment['Roadm']['default'].restrictions['preamp_variety_list']
    for amp_name in restrictions:
        if amp_name not in equipment['Edfa']:
            raise EquipmentConfigError(f'ROADM restriction {amp_name} does not refer to a defined EDFA name')


def _equipment_from_json(json_data, filename):
    """build global dictionnary eqpt_library that stores all eqpt characteristics:
    edfa type type_variety, fiber type_variety
    from the eqpt_config.json (filename parameter)
    also read advanced_config_from_json file parameters for edfa if they are available:
    typically nf_ripple, dfg gain ripple, dgt and nf polynomial nf_fit_coeff
    if advanced_config_from_json file parameter is not present: use nf_model:
    requires nf_min and nf_max values boundaries of the edfa gain range
    """
    equipment = {}
    for key, entries in json_data.items():
        equipment[key] = {}
        for entry in entries:
            subkey = entry.get('type_variety', 'default')
            if key == 'Edfa':
                equipment[key][subkey] = Amp.from_json(filename, **entry)
            elif key == 'Fiber':
                equipment[key][subkey] = Fiber(**entry)
            elif key == 'Span':
                equipment[key][subkey] = Span(**entry)
            elif key == 'Roadm':
                equipment[key][subkey] = Roadm(**entry)
            elif key == 'SI':
                equipment[key][subkey] = SI(**entry)
            elif key == 'Transceiver':
                equipment[key][subkey] = Transceiver(**entry)
            elif key == 'RamanFiber':
                equipment[key][subkey] = RamanFiber(**entry)
            else:
                raise EquipmentConfigError(f'Unrecognized network element type "{key}"')
    equipment = _update_trx_osnr(equipment)
    equipment = _update_dual_stage(equipment)
    _roadm_restrictions_sanity_check(equipment)
    return equipment
