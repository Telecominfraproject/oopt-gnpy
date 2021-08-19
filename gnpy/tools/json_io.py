#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.tools.json_io
==================

Loading and saving data from JSON files in GNPy's internal data format
"""

from networkx import DiGraph
from logging import getLogger
from pathlib import Path
import json
from collections import namedtuple
from numpy import arange

from gnpy.core import elements
from gnpy.core.equipment import trx_mode_params
from gnpy.core.exceptions import ConfigurationError, EquipmentConfigError, NetworkTopologyError, ServiceError
from gnpy.core.science_utils import estimate_nf_model
from gnpy.core.info import Carrier
from gnpy.core.utils import automatic_nch, automatic_fmax, merge_amplifier_restrictions
from gnpy.core.parameters import DEFAULT_RAMAN_COEFFICIENT
from gnpy.topology.request import PathRequest, Disjunction, compute_spectrum_slot_vs_bandwidth
from gnpy.topology.spectrum_assignment import mvalue_to_slots
from gnpy.tools.convert import xls_to_json_data
from gnpy.tools.service_sheet import read_service_sheet


_logger = getLogger(__name__)


Model_vg = namedtuple('Model_vg', 'nf1 nf2 delta_p orig_nf_min orig_nf_max')
Model_fg = namedtuple('Model_fg', 'nf0')
Model_openroadm_ila = namedtuple('Model_openroadm_ila', 'nf_coef')
Model_hybrid = namedtuple('Model_hybrid', 'nf_ram gain_ram edfa_variety')
Model_dual_stage = namedtuple('Model_dual_stage', 'preamp_variety booster_variety')


class Model_openroadm_preamp:
    pass


class Model_openroadm_booster:
    pass


class _JsonThing:
    def update_attr(self, default_values, kwargs, name):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in default_values.items():
            setattr(self, k, clean_kwargs.get(k, v))
            if k not in clean_kwargs and name != 'Amp':
                msg = f'\n WARNING missing {k} attribute in eqpt_config.json[{name}]' \
                    + f'\n default value is {k} = {v}'
                _logger.warning(msg)


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
        'add_drop_osnr': 100,
        'pmd': 0,
        'pdl': 0,
        'restrictions': {
            'preamp_variety_list': [],
            'booster_variety_list': []
        }
    }

    def __init__(self, **kwargs):
        # If equalization is not defined in equipment, then raise an error.
        # Only one type of equalization must be defined.
        allowed_equalisations = ['target_pch_out_db', 'target_psd_out_mWperGHz', 'target_out_mWperSlotWidth']
        requested_eq_mask = [eq in kwargs for eq in allowed_equalisations]
        if sum(requested_eq_mask) > 1:
            msg = 'Only one equalization type should be set in ROADM, found: ' \
                  + ', '.join(eq for eq in allowed_equalisations if eq in kwargs)
            raise EquipmentConfigError(msg)
        if not any(requested_eq_mask):
            msg = 'No equalization type set in ROADM'
            raise EquipmentConfigError(msg)
        for key in allowed_equalisations:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                break
        self.update_attr(self.default_values, kwargs, 'Roadm')


class Transceiver(_JsonThing):
    default_values = {
        'type_variety': None,
        'frequency': None,
        'mode': {}
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, 'Transceiver')
        for mode_params in self.mode:
            penalties = mode_params.get('penalties')
            mode_params['penalties'] = {}
            mode_params['equalization_offset_db'] = mode_params.get('equalization_offset_db', 0)
            if not penalties:
                continue
            for impairment in ('chromatic_dispersion', 'pmd', 'pdl'):
                imp_penalties = [p for p in penalties if impairment in p]
                if not imp_penalties:
                    continue
                if all(p[impairment] > 0 for p in imp_penalties):
                    # make sure the list of penalty values include a proper lower boundary
                    # (we assume 0 penalty for 0 impairment)
                    imp_penalties.insert(0, {impairment: 0, 'penalty_value': 0})
                # make sure the list of penalty values are sorted by impairment value
                imp_penalties.sort(key=lambda i: i[impairment])
                # rearrange as dict of lists instead of list of dicts
                mode_params['penalties'][impairment] = {
                    'up_to_boundary': [p[impairment] for p in imp_penalties],
                    'penalty_value': [p['penalty_value'] for p in imp_penalties]
                }


class Fiber(_JsonThing):
    default_values = {
        'type_variety': '',
        'dispersion': None,
        'effective_area': None,
        'pmd_coef': 0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs, self.__class__.__name__)
        if 'gamma' in kwargs:
            setattr(self, 'gamma', kwargs['gamma'])
        if 'raman_efficiency' in kwargs:
            raman_coefficient = kwargs['raman_efficiency']
            cr = raman_coefficient.pop('cr')
            raman_coefficient['g0'] = cr
            raman_coefficient['reference_frequency'] = DEFAULT_RAMAN_COEFFICIENT['reference_frequency']
            setattr(self, 'raman_coefficient', raman_coefficient)


class RamanFiber(Fiber):
    pass


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
        'preamp_variety': None,
        'booster_variety': None,
        'nf_min': None,
        'nf_max': None,
        'nf_coef': None,
        'nf0': None,
        'nf_fit_coeff': None,
        'nf_ripple': 0,
        'dgt': None,
        'gain_ripple': 0,
        'tilt_ripple': 0,
        'f_ripple_ref': None,
        'out_voa_auto': False,
        'allowed_for_design': False,
        'raman': False,
        'pmd': 0,
        'pdl': 0,
        'advance_configurations_from_json': None
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
                msg = f'missing nf0 value input for amplifier: {type_variety} in equipment config'
                raise EquipmentConfigError(msg)
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
                msg = f'missing nf_min or nf_max value input for amplifier: {type_variety} in equipment config'
                raise EquipmentConfigError(msg)
            try:  # remove all remaining nf inputs
                del kwargs['nf0']
            except KeyError:
                pass  # nf0 is not needed for variable gain amp
            nf1, nf2, delta_p = estimate_nf_model(type_variety, gain_min, gain_max, nf_min, nf_max)
            nf_def = Model_vg(nf1, nf2, delta_p, nf_min, nf_max)
        elif type_def == 'openroadm':
            try:
                nf_coef = kwargs.pop('nf_coef')
            except KeyError:  # nf_coef is expected for openroadm amp
                raise EquipmentConfigError(f'missing nf_coef input for amplifier: {type_variety} in equipment config')
            nf_def = Model_openroadm_ila(nf_coef)
        elif type_def == 'openroadm_preamp':
            nf_def = Model_openroadm_preamp()
        elif type_def == 'openroadm_booster':
            nf_def = Model_openroadm_booster()
        elif type_def == 'dual_stage':
            try:  # nf_ram and gain_ram are expected for a hybrid amp
                preamp_variety = kwargs.pop('preamp_variety')
                booster_variety = kwargs.pop('booster_variety')
            except KeyError:
                msg = f'missing preamp/booster variety input for amplifier: {type_variety} in equipment config'
                raise EquipmentConfigError(msg)
            dual_stage_def = Model_dual_stage(preamp_variety, booster_variety)
        else:
            raise EquipmentConfigError(f'Edfa type_def {type_def} does not exist')

        json_data = load_json(config)

        return cls(**{**kwargs, **json_data,
                      'nf_model': nf_def, 'dual_stage_model': dual_stage_def})


def _automatic_spacing(baud_rate):
    """return the min possible channel spacing for a given baud rate"""
    # TODO : this should parametrized in a cfg file
    # list of possible tuples [(max_baud_rate, spacing_for_this_baud_rate)]
    spacing_list = [(33e9, 37.5e9), (38e9, 50e9), (50e9, 62.5e9), (67e9, 75e9), (92e9, 100e9)]
    return min((s[1] for s in spacing_list if s[0] > baud_rate), default=baud_rate * 1.2)


def _spectrum_from_json(json_data):
    """JSON_data is a list of spectrum partitions each with
    {f_min, f_max, baud_rate, roll_off, delta_pdb, slot_width, tx_osnr, label}
    Creates the per freq Carrier's dict.
    f_min, f_max, baud_rate, slot_width and roll_off are mandatory
    label, tx_osnr and delta_pdb are created if not present
    label should be different for each partition
    >>> json_data = {'spectrum': \
        [{'f_min': 193.2e12, 'f_max': 193.4e12, 'slot_width': 50e9, 'baud_rate': 32e9, 'roll_off': 0.15, \
            'delta_pdb': 1, 'tx_osnr': 45},\
        {'f_min': 193.4625e12, 'f_max': 193.9875e12, 'slot_width': 75e9, 'baud_rate': 64e9, 'roll_off': 0.15},\
        {'f_min': 194.075e12, 'f_max': 194.075e12, 'slot_width': 100e9, 'baud_rate': 90e9, 'roll_off': 0.15},\
        {'f_min': 194.2e12, 'f_max': 194.35e12, 'slot_width': 50e9, 'baud_rate': 32e9, 'roll_off': 0.15}]}
    >>> spectrum = _spectrum_from_json(json_data['spectrum'])
    >>> for k, v in spectrum.items():
    ...     print(f'{k}: {v}')
    ...
    193200000000000.0: Carrier(delta_pdb=1, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=45, label='0-32.00G')
    193250000000000.0: Carrier(delta_pdb=1, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=45, label='0-32.00G')
    193300000000000.0: Carrier(delta_pdb=1, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=45, label='0-32.00G')
    193350000000000.0: Carrier(delta_pdb=1, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=45, label='0-32.00G')
    193400000000000.0: Carrier(delta_pdb=1, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=45, label='0-32.00G')
    193462500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193537500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193612500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193687500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193762500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193837500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193912500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    193987500000000.0: Carrier(delta_pdb=0, baud_rate=64000000000.0, slot_width=75000000000.0, roll_off=0.15, tx_osnr=40, label='1-64.00G')
    194075000000000.0: Carrier(delta_pdb=0, baud_rate=90000000000.0, slot_width=100000000000.0, roll_off=0.15, tx_osnr=40, label='2-90.00G')
    194200000000000.0: Carrier(delta_pdb=0, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=40, label='3-32.00G')
    194250000000000.0: Carrier(delta_pdb=0, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=40, label='3-32.00G')
    194300000000000.0: Carrier(delta_pdb=0, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=40, label='3-32.00G')
    194350000000000.0: Carrier(delta_pdb=0, baud_rate=32000000000.0, slot_width=50000000000.0, roll_off=0.15, tx_osnr=40, label='3-32.00G')
    """
    spectrum = {}
    json_data = sorted(json_data, key=lambda x: x['f_min'])
    # min freq of occupation is f_min - slot_width/2 (numbering starts at 0)
    previous_part_max_freq = 0.0
    for index, part in enumerate(json_data):
        # default delta_pdb is 0 dB
        if 'delta_pdb' not in part:
            part['delta_pdb'] = 0
        # add a label to the partition for the printings
        if 'label' not in part:
            part['label'] = f'{index}-{part["baud_rate"] * 1e-9 :.2f}G'
        # default tx_osnr is set to 40 dB
        if 'tx_osnr' not in part:
            part['tx_osnr'] = 40
        # starting freq is exactly f_min to be consistent with utils.automatic_nch
        # first partition min occupation is f_min - slot_width / 2 (central_frequency is f_min)
        # supposes that carriers are centered on frequency
        if previous_part_max_freq > (part['f_min'] - part['slot_width'] / 2):
            # check that previous part last channel does not overlap on next part first channel
            # max center of the part should be below part['f_max'] and aligned on the slot_width
            msg = 'Not a valid initial spectrum definition:\nprevious spectrum last carrier max occupation ' +\
                f'{previous_part_max_freq * 1e-12 :.5f}GHz ' +\
                'overlaps on next spectrum first carrier occupation ' +\
                f'{(part["f_min"] - part["slot_width"] / 2) * 1e-12 :.5f}GHz'
            raise ValueError(msg)

        max_range = ((part['f_max'] - part['f_min']) // part['slot_width'] + 1) * part['slot_width']
        for current_freq in arange(part['f_min'],
                                   part['f_min'] + max_range,
                                   part['slot_width']):
            spectrum[current_freq] = Carrier(delta_pdb=part['delta_pdb'], baud_rate=part['baud_rate'],
                                             slot_width=part['slot_width'], roll_off=part['roll_off'],
                                             tx_osnr=part['tx_osnr'], label=part['label'])
        previous_part_max_freq = current_freq + part['slot_width'] / 2
    return spectrum


def load_equipment(filename):
    json_data = load_json(filename)
    return _equipment_from_json(json_data, filename)


def load_initial_spectrum(filename):
    json_data = load_json(filename)
    return _spectrum_from_json(json_data['spectrum'])


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
    """verifies that booster and preamp restrictions specified in roadm equipment are listed in the edfa."""
    restrictions = equipment['Roadm']['default'].restrictions['booster_variety_list'] + \
        equipment['Roadm']['default'].restrictions['preamp_variety_list']
    for amp_name in restrictions:
        if amp_name not in equipment['Edfa']:
            raise EquipmentConfigError(f'ROADM restriction {amp_name} does not refer to a defined EDFA name')


def _check_fiber_vs_raman_fiber(equipment):
    """Ensure that Fiber and RamanFiber with the same name define common properties equally"""
    if 'RamanFiber' not in equipment:
        return
    for fiber_type in set(equipment['Fiber'].keys()) & set(equipment['RamanFiber'].keys()):
        for attr in ('dispersion', 'dispersion-slope', 'effective_area', 'gamma', 'pmd-coefficient'):
            fiber = equipment['Fiber'][fiber_type]
            raman = equipment['RamanFiber'][fiber_type]
            a = getattr(fiber, attr, None)
            b = getattr(raman, attr, None)
            if a != b:
                raise EquipmentConfigError(f'WARNING: Fiber and RamanFiber definition of "{fiber_type}" '
                                           f'disagrees for "{attr}": {a} != {b}')


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
    _check_fiber_vs_raman_fiber(equipment)
    equipment = _update_dual_stage(equipment)
    _roadm_restrictions_sanity_check(equipment)
    return equipment


def load_network(filename, equipment):
    if filename.suffix.lower() in ('.xls', '.xlsx'):
        json_data = xls_to_json_data(filename)
    elif filename.suffix.lower() == '.json':
        json_data = load_json(filename)
    else:
        raise ValueError(f'unsupported topology filename extension {filename.suffix.lower()}')
    return network_from_json(json_data, equipment)


def save_network(network: DiGraph, filename: str):
    """Dump the network into a JSON file

    :param network: network to work on
    :param filename: file to write to
    """
    save_json(network_to_json(network), filename)


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
            extra_params = equipment[typ][variety].__dict__
            temp = el_config.setdefault('params', {})
            if typ == 'Roadm':
                # if equalization is defined, remove default equalization from the extra_params
                # If equalisation is not defined in the element config, then use the default one from equipment
                # if more than one equalization was defined in element config, then raise an error
                extra_params = merge_equalization(temp, extra_params)
                if not extra_params:
                    msg = f'ROADM {el_config["uid"]}: invalid equalization settings'
                    raise ConfigurationError(msg)
            temp = merge_amplifier_restrictions(temp, extra_params)
            el_config['params'] = temp
            el_config['type_variety'] = variety
        elif (typ in ['Fiber', 'RamanFiber']):
            raise ConfigurationError(f'The {typ} of variety type {variety} was not recognized:'
                                     '\nplease check it is properly defined in the eqpt_config json file')
        elif typ == 'Edfa':
            if variety in ['default', '']:
                el_config['params'] = Amp.default_values
            else:
                raise ConfigurationError(f'The Edfa of variety type {variety} was not recognized:'
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
            msg = f'can not find {from_node} or {to_node} defined in {cx}'
            raise NetworkTopologyError(msg)

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


def load_requests(filename, eqpt, bidir, network, network_filename):
    """loads the requests from a json or an excel file into a data string"""
    if filename.suffix.lower() in ('.xls', '.xlsx'):
        _logger.info('Automatically converting requests from XLS to JSON')
        try:
            return convert_service_sheet(filename, eqpt, network, network_filename=network_filename, bidir=bidir)
        except ServiceError as this_e:
            raise ServiceError(f'Service error: {this_e}')
    else:
        return load_json(filename)


def requests_from_json(json_data, equipment):
    """Extract list of requests from data parsed from JSON"""
    requests_list = []

    for req in json_data['path-request']:
        # init all params from request
        params = {}
        params['request_id'] = f'{req["request-id"]}'
        params['source'] = req['source']
        params['bidir'] = req['bidirectional']
        params['destination'] = req['destination']
        params['trx_type'] = req['path-constraints']['te-bandwidth']['trx_type']
        if params['trx_type'] is None:
            msg = f'Request {req["request-id"]} has no transceiver type defined.'
            raise ServiceError(msg)
        params['trx_mode'] = req['path-constraints']['te-bandwidth'].get('trx_mode', None)
        params['format'] = params['trx_mode']
        params['spacing'] = req['path-constraints']['te-bandwidth']['spacing']
        try:
            nd_list = sorted(req['explicit-route-objects']['route-object-include-exclude'], key=lambda x: x['index'])
        except KeyError:
            nd_list = []
        params['nodes_list'] = [n['num-unnum-hop']['node-id'] for n in nd_list]
        params['loose_list'] = [n['num-unnum-hop']['hop-type'] for n in nd_list]
        # recover trx physical param (baudrate, ...) from type and mode
        # in trx_mode_params optical power is read from equipment['SI']['default'] and
        # nb_channel is computed based on min max frequency and spacing
        try:
            trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
        except EquipmentConfigError as e:
            msg = f'Equipment Config error in {req["request-id"]}: {e}'
            raise EquipmentConfigError(msg) from e
        params.update(trx_params)
        # optical power might be set differently in the request. if it is indicated then the
        # params['power'] is updated
        try:
            if req['path-constraints']['te-bandwidth']['output-power']:
                params['power'] = req['path-constraints']['te-bandwidth']['output-power']
        except KeyError:
            pass
        # same process for nb-channel
        f_min = params['f_min']
        f_max_from_si = params['f_max']
        try:
            if req['path-constraints']['te-bandwidth']['max-nb-of-channel'] is not None:
                nch = req['path-constraints']['te-bandwidth']['max-nb-of-channel']
                params['nb_channel'] = nch
                spacing = params['spacing']
                params['f_max'] = automatic_fmax(f_min, spacing, nch)
            else:
                params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
        except KeyError:
            params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
        params['effective_freq_slot'] = \
            req['path-constraints']['te-bandwidth'].get('effective-freq-slot', [{'N': None, 'M': None}])
        try:
            params['path_bandwidth'] = req['path-constraints']['te-bandwidth']['path_bandwidth']
        except KeyError:
            pass
        _check_one_request(params, f_max_from_si)
        requests_list.append(PathRequest(**params))
    return requests_list


def _check_one_request(params, f_max_from_si):
    """Checks that the requested parameters are consistant (spacing vs nb channel vs transponder mode...)"""
    f_min = params['f_min']
    f_max = params['f_max']
    max_recommanded_nb_channels = automatic_nch(f_min, f_max_from_si, params['spacing'])
    if params['baud_rate'] is not None:
        # implicitly means that a mode is defined with min_spacing
        if params['min_spacing'] > params['spacing']:
            msg = f'Request {params["request_id"]} has spacing below transponder ' +\
                  f'{params["trx_type"]} {params["trx_mode"]} min spacing value ' +\
                  f'{params["min_spacing"]*1e-9}GHz.\nComputation stopped'
            raise ServiceError(msg)
        if f_max > f_max_from_si:
            msg = f'Requested channel number {params["nb_channel"]}, baud rate {params["baud_rate"] * 1e-9} GHz' \
                  + f' and requested spacing {params["spacing"]*1e-9}GHz is not consistent with frequency range' \
                  + f' {f_min*1e-12} THz, {f_max_from_si*1e-12} THz.' \
                  + f' Max recommanded nb of channels is {max_recommanded_nb_channels}.'
            raise ServiceError(msg)
    # Transponder mode already selected; will it fit to the requested bandwidth?
    if params['trx_mode'] is not None and params['effective_freq_slot'] is not None:
        required_nb_of_channels, requested_m = compute_spectrum_slot_vs_bandwidth(params['path_bandwidth'],
                                                                                  params['spacing'],
                                                                                  params['bit_rate'])
        _, per_channel_m = compute_spectrum_slot_vs_bandwidth(params['bit_rate'],
                                                              params['spacing'],
                                                              params['bit_rate'])
        # each M should fit one or more channels if it is not None
        # spectrum slots should not overlap
        # resulting nb of channels should be bigger than the nb computed with path_bandwidth
        # without being splitted
        # TODO: elaborate a more accurate estimate with nb_wl * tx_osnr + possibly guardbands in case of
        # superchannel closed packing.
        nb_of_channels = 0
        # order slots
        slots = sorted(params['effective_freq_slot'], key=lambda x: float('inf') if x['N'] is None else x['N'])
        for slot in slots:
            nb_of_channels = nb_of_channels + slot['M'] // per_channel_m if slot['M'] is not None \
                and nb_of_channels is not None else None
            if slot['M'] is not None and slot['M'] < per_channel_m:
                msg = f'Requested M {slot} number of slots for request' +\
                      f' {params["request_id"]} should be greater than {per_channel_m} to support request' +\
                      f'with {params["trx_type"]} {params["trx_mode"]}'
                _logger.critical(msg)
        if nb_of_channels is not None and nb_of_channels < required_nb_of_channels:
            msg = f'Requested M {slots} number of slots for request {params["request_id"]} support {nb_of_channels}' +\
                  f' nb of channels while {required_nb_of_channels} are required to support request' +\
                  f' {params["path_bandwidth"] * 1e-9} Gbit/s with {params["trx_type"]} {params["trx_mode"]}'
            raise ServiceError(msg)
        if nb_of_channels is not None:
            _, stop0n = mvalue_to_slots(slots[0]['N'], slots[0]['M'])
            i = 1
            while i < len(slots):
                slot = slots[i]
                startn, stopn = mvalue_to_slots(slot['N'], slot['M'])
                if startn <= stop0n:
                    msg = f'Requested M {slots} for request {params["request_id"]} overlap'
                    raise ServiceError(msg)
                _, stop0n = startn, stopn
                i += 1


def disjunctions_from_json(json_data):
    """reads the disjunction requests from the json dict and create the list
    of requested disjunctions for this set of requests
    """
    disjunctions_list = []
    if 'synchronization' in json_data:
        for snc in json_data['synchronization']:
            params = {}
            params['disjunction_id'] = snc['synchronization-id']
            params['relaxable'] = snc['svec']['relaxable']
            params['link_diverse'] = 'link' in snc['svec']['disjointness']
            params['node_diverse'] = 'node' in snc['svec']['disjointness']
            params['disjunctions_req'] = snc['svec']['request-id-number']
            disjunctions_list.append(Disjunction(**params))

    return disjunctions_list


def convert_service_sheet(
        input_filename,
        eqpt,
        network,
        network_filename=None,
        output_filename='',
        bidir=False):
    if output_filename == '':
        output_filename = f'{str(input_filename)[0:len(str(input_filename))-len(str(input_filename.suffixes[0]))]}_services.json'
    data = read_service_sheet(input_filename, eqpt, network, network_filename, bidir)
    save_json(data, output_filename)
    return data


def find_equalisation(params, equalization_types):
    """Find the equalization(s) defined in params. params can be a dict or a Roadm object.

    >>> roadm = {'add_drop_osnr': 100, 'pmd': 1, 'pdl': 0.5,
    ...     'restrictions': {'preamp_variety_list': ['a'], 'booster_variety_list': ['b']},
    ...     'target_psd_out_mWperGHz': 4e-4}
    >>> equalization_types = ['target_pch_out_db', 'target_psd_out_mWperGHz']
    >>> find_equalisation(roadm, equalization_types)
    {'target_pch_out_db': False, 'target_psd_out_mWperGHz': True}
    """
    equalization = {e: False for e in equalization_types}
    for equ in equalization_types:
        if equ in params:
            equalization[equ] = True
    return equalization


def merge_equalization(params, extra_params):
    """params contains ROADM element config and extra_params default values from equipment library.
    If equalization is not defined in ROADM element use the one defined in equipment library.
    Only one type of equalization must be defined: power (target_pch_out_db) or PSD (target_psd_out_mWperGHz)
    or PSW (target_out_mWperSlotWidth)
    params and extra_params are dict
    """
    equalization_types = ['target_pch_out_db', 'target_psd_out_mWperGHz', 'target_out_mWperSlotWidth']
    roadm_equalizations = find_equalisation(params, equalization_types)
    if sum(roadm_equalizations.values()) > 1:
        # if ROADM config contains more than one equalization type then this is an error
        return None
    if sum(roadm_equalizations.values()) == 1:
        # if ROADM config contains one equalization
        # don't use the default equalization
        return {k: v for k, v in extra_params.items() if k not in equalization_types}
    if sum(roadm_equalizations.values()) == 0:
        # If ROADM config doesn't contain any equalization type, keep the default one
        return extra_params
    return None
