#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.json_io
==================

Loading and saving data from JSON files in GNPy's internal data format
'''

from networkx import DiGraph
from logging import getLogger
from os import path
from pathlib import Path
import json
from collections import namedtuple
from gnpy.core import ansi_escapes, elements
from gnpy.core.exceptions import ConfigurationError, EquipmentConfigError, NetworkTopologyError
from gnpy.core.science_utils import estimate_nf_model
from gnpy.core.utils import merge_amplifier_restrictions
from gnpy.tools.convert import convert_file
import time


_logger = getLogger(__name__)


Model_vg = namedtuple('Model_vg', 'nf1 nf2 delta_p')
Model_fg = namedtuple('Model_fg', 'nf0')
Model_openroadm = namedtuple('Model_openroadm', 'nf_coef')
Model_hybrid = namedtuple('Model_hybrid', 'nf_ram gain_ram edfa_variety')
Model_dual_stage = namedtuple('Model_dual_stage', 'preamp_variety booster_variety')


class _JsonThing:
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


class SI(_JsonThing):
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


class Span(_JsonThing):
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


class Roadm(_JsonThing):
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


class Transceiver(_JsonThing):
    default_values = {
        'type_variety': None,
        'frequency': None,
        'mode': {}
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Transceiver')


class Fiber(_JsonThing):
    default_values = {
        'type_variety': '',
        'dispersion': None,
        'gamma': 0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Fiber')


class RamanFiber(_JsonThing):
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


class Amp(_JsonThing):
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
            json_data = json.load(f)

        return cls(**{**kwargs, **json_data,
                      'nf_model': nf_def, 'dual_stage_model': dual_stage_def})


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


def load_network(filename, equipment, name_matching=False):
    json_filename = ''
    if filename.suffix.lower() in ('.xls', '.xlsx'):
        _logger.info('Automatically generating topology JSON file')
        json_filename = convert_file(filename, name_matching)
    elif filename.suffix.lower() == '.json':
        json_filename = filename
    else:
        raise ValueError(f'unsuported topology filename extension {filename.suffix.lower()}')
    json_data = load_json(json_filename)
    return network_from_json(json_data, equipment)


def save_network(filename, network):
    filename_output = path.splitext(filename)[0] + '_auto_design.json'
    json_data = network_to_json(network)
    save_json(json_data, filename_output)


def _cls_for(equipment_type):
    if equipment_type == 'Edfa':
        return elements.Edfa
    if equipment_type == 'Fused':
        return elements.Fused
    elif equipment_type == 'Roadm':
        return elements.Roadm
    elif equipment_type == 'Transceiver':
        return elements.Transceiver
    elif equipment_type == 'Fiber':
        return elements.Fiber
    elif equipment_type == 'RamanFiber':
        return elements.RamanFiber
    else:
        raise ConfigurationError(f'Unknown network equipment "{equipment_type}"')


def network_from_json(json_data, equipment):
    # NOTE|dutc: we could use the following, but it would tie our data format
    #            too closely to the graph library
    # from networkx import node_link_graph
    g = DiGraph()
    for el_config in json_data['elements']:
        typ = el_config.pop('type')
        variety = el_config.pop('type_variety', 'default')
        cls = _cls_for(typ)
        if typ == 'Fused':
            # well, there's no variety for the 'Fused' node type
            pass
        elif variety in equipment[typ]:
            extra_params = equipment[typ][variety]
            temp = el_config.setdefault('params', {})
            temp = merge_amplifier_restrictions(temp, extra_params.__dict__)
            el_config['params'] = temp
            el_config['type_variety'] = variety
        elif typ in ['Edfa', 'Fiber', 'RamanFiber']:  # catch it now because the code will crash later!
            raise ConfigurationError(f'The {typ} of variety type {variety} was not recognized:'
                                     '\nplease check it is properly defined in the eqpt_config json file')
        el = cls(**el_config)
        g.add_node(el)

    nodes = {k.uid: k for k in g.nodes()}

    for cx in json_data['connections']:
        from_node, to_node = cx['from_node'], cx['to_node']
        try:
            if isinstance(nodes[from_node], elements.Fiber):
                edge_length = nodes[from_node].params.length
            else:
                edge_length = 0.01
            g.add_edge(nodes[from_node], nodes[to_node], weight=edge_length)
        except KeyError:
            raise NetworkTopologyError(f'can not find {from_node} or {to_node} defined in {cx}')

    return g


def network_to_json(network):
    data = {
        'elements': [n.to_json for n in network]
    }
    connections = {
        'connections': [{"from_node": n.uid,
                         "to_node": next_n.uid}
                        for n in network
                        for next_n in network.successors(n) if next_n is not None]
    }
    data.update(connections)
    return data


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
