#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.json_io
==================

Loading and saving data from JSON files in GNPy's internal data format
'''

from networkx import DiGraph
from scipy.interpolate import interp1d
from logging import getLogger
from os import path
from operator import attrgetter
from pathlib import Path
import json
from collections import namedtuple
from gnpy.core import ansi_escapes, elements
from gnpy.core.exceptions import ConfigurationError, EquipmentConfigError, NetworkTopologyError
from gnpy.core.science_utils import estimate_nf_model
from gnpy.core.utils import round2float, merge_amplifier_restrictions, convert_length
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


def edfa_nf(gain_target, variety_type, equipment):
    amp_params = equipment['Edfa'][variety_type]
    amp = elements.Edfa(
        uid='calc_NF',
        params=amp_params.__dict__,
        operational={
            'gain_target': gain_target,
            'tilt_target': 0
        }
    )
    amp.pin_db = 0
    amp.nch = 88
    return amp._calc_nf(True)


def select_edfa(raman_allowed, gain_target, power_target, equipment, uid, restrictions=None):
    """amplifer selection algorithm
    @Orange Jean-Luc Augé
    """
    Edfa_list = namedtuple('Edfa_list', 'variety power gain_min nf')
    TARGET_EXTENDED_GAIN = equipment['Span']['default'].target_extended_gain

    # for roadm restriction only: create a dict including not allowed for design amps
    # because main use case is to have specific radm amp which are not allowed for ILA
    # with the auto design
    edfa_dict = {name: amp for (name, amp) in equipment['Edfa'].items()
                 if restrictions is None or name in restrictions}

    pin = power_target - gain_target

    # create 2 list of available amplifiers with relevant attributes for their selection

    # edfa list with:
    # extended gain min allowance of 3dB: could be parametrized, but a bit complex
    # extended gain max allowance TARGET_EXTENDED_GAIN is coming from eqpt_config.json
    # power attribut include power AND gain limitations
    edfa_list = [Edfa_list(
        variety=edfa_variety,
        power=min(
            pin
            + edfa.gain_flatmax
            + TARGET_EXTENDED_GAIN,
            edfa.p_max
        )
        - power_target,
        gain_min=gain_target + 3
        - edfa.gain_min,
        nf=edfa_nf(gain_target, edfa_variety, equipment))
        for edfa_variety, edfa in edfa_dict.items()
        if ((edfa.allowed_for_design or restrictions is not None) and not edfa.raman)]

    # consider a Raman list because of different gain_min requirement:
    # do not allow extended gain min for Raman
    raman_list = [Edfa_list(
        variety=edfa_variety,
        power=min(
            pin
            + edfa.gain_flatmax
            + TARGET_EXTENDED_GAIN,
            edfa.p_max
        )
        - power_target,
        gain_min=gain_target
        - edfa.gain_min,
        nf=edfa_nf(gain_target, edfa_variety, equipment))
        for edfa_variety, edfa in edfa_dict.items()
        if (edfa.allowed_for_design and edfa.raman)] \
        if raman_allowed else []

    # merge raman and edfa lists
    amp_list = edfa_list + raman_list

    # filter on min gain limitation:
    acceptable_gain_min_list = [x for x in amp_list if x.gain_min > 0]

    if len(acceptable_gain_min_list) < 1:
        # do not take this empty list into account for the rest of the code
        # but issue a warning to the user and do not consider Raman
        # Raman below min gain should not be allowed because i is meant to be a design requirement
        # and raman padding at the amplifier input is impossible!

        if len(edfa_list) < 1:
            raise ConfigurationError(f'auto_design could not find any amplifier \
                    to satisfy min gain requirement in node {uid} \
                    please increase span fiber padding')
        else:
            # TODO: convert to logging
            print(
                f'{ansi_escapes.red}WARNING:{ansi_escapes.reset} target gain in node {uid} is below all available amplifiers min gain: \
                  amplifier input padding will be assumed, consider increase span fiber padding instead'
            )
            acceptable_gain_min_list = edfa_list

    # filter on gain+power limitation:
    # this list checks both the gain and the power requirement
    # because of the way .power is calculated in the list
    acceptable_power_list = [x for x in acceptable_gain_min_list if x.power > 0]
    if len(acceptable_power_list) < 1:
        # no amplifier satisfies the required power, so pick the highest power(s):
        power_max = max(acceptable_gain_min_list, key=attrgetter('power')).power
        # check and pick if other amplifiers may have a similar gain/power
        # allow a 0.3dB power range
        # this allows to chose an amplifier with a better NF subsequentely
        acceptable_power_list = [x for x in acceptable_gain_min_list
                                 if x.power - power_max > -0.3]

    # gain and power requirements are resolved,
    #       =>chose the amp with the best NF among the acceptable ones:
    selected_edfa = min(acceptable_power_list, key=attrgetter('nf'))  # filter on NF
    # check what are the gain and power limitations of this amp
    power_reduction = round(min(selected_edfa.power, 0), 2)
    if power_reduction < -0.5:
        print(
            f'{ansi_escapes.red}WARNING:{ansi_escapes.reset} target gain and power in node {uid}\n \
    is beyond all available amplifiers capabilities and/or extended_gain_range:\n\
    a power reduction of {power_reduction} is applied\n'
        )

    return selected_edfa.variety, power_reduction


def target_power(network, node, equipment):  # get_fiber_dp
    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    dp_range = list(equipment['Span']['default'].delta_power_range_db)
    node_loss = span_loss(network, node)

    try:
        dp = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except KeyError:
        raise ConfigurationError(f'invalid delta_power_range_db definition in eqpt_config[Span]'
                                 f'delta_power_range_db: [lower_bound, upper_bound, step]')

    if isinstance(node, elements.Roadm):
        dp = 0

    return dp


def prev_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are Fused or Fiber type"""
    try:
        prev_node = next(n for n in network.predecessors(node))
    except StopIteration:
        raise NetworkTopologyError(f'Node {node.uid} is not properly connected, please check network topology')
    # yield and re-iterate
    if isinstance(prev_node, elements.Fused) or isinstance(node, elements.Fused):
        yield prev_node
        yield from prev_node_generator(network, prev_node)
    else:
        StopIteration


def next_node_generator(network, node):
    """fused spans interest:
    iterate over all successors while they are Fused or Fiber type"""
    try:
        next_node = next(n for n in network.successors(node))
    except StopIteration:
        raise NetworkTopologyError('Node {node.uid} is not properly connected, please check network topology')
    # yield and re-iterate
    if isinstance(next_node, elements.Fused) or isinstance(node, elements.Fused):
        yield next_node
        yield from next_node_generator(network, next_node)
    else:
        StopIteration


def span_loss(network, node):
    """Fused span interest:
    return the total span loss of all the fibers spliced by a Fused node"""
    loss = node.loss if node.passive else 0
    try:
        prev_node = next(n for n in network.predecessors(node))
        if isinstance(prev_node, elements.Fused):
            loss += sum(n.loss for n in prev_node_generator(network, node))
    except StopIteration:
        pass
    try:
        next_node = next(n for n in network.successors(node))
        if isinstance(next_node, elements.Fused):
            loss += sum(n.loss for n in next_node_generator(network, node))
    except StopIteration:
        pass
    return loss


def find_first_node(network, node):
    """Fused node interest:
    returns the 1st node at the origin of a succession of fused nodes
    (aka no amp in between)"""
    this_node = node
    for this_node in prev_node_generator(network, node):
        pass
    return this_node


def find_last_node(network, node):
    """Fused node interest:
    returns the last node in a succession of fused nodes
    (aka no amp in between)"""
    this_node = node
    for this_node in next_node_generator(network, node):
        pass
    return this_node


def set_amplifier_voa(amp, power_target, power_mode):
    VOA_MARGIN = 1  # do not maximize the VOA optimization
    if amp.out_voa is None:
        if power_mode:
            voa = min(amp.params.p_max - power_target,
                      amp.params.gain_flatmax - amp.effective_gain)
            voa = max(round2float(max(voa, 0), 0.5) - VOA_MARGIN, 0) if amp.params.out_voa_auto else 0
            amp.delta_p = amp.delta_p + voa
            amp.effective_gain = amp.effective_gain + voa
        else:
            voa = 0  # no output voa optimization in gain mode
        amp.out_voa = voa


def set_egress_amplifier(network, roadm, equipment, pref_total_db):
    power_mode = equipment['Span']['default'].power_mode
    next_oms = (n for n in network.successors(roadm) if not isinstance(n, elements.Transceiver))
    for oms in next_oms:
        # go through all the OMS departing from the Roadm
        node = roadm
        prev_node = roadm
        next_node = oms
        # if isinstance(next_node, elements.Fused): #support ROADM wo egress amp for metro applications
        #     node = find_last_node(next_node)
        #     next_node = next(n for n in network.successors(node))
        #     next_node = find_last_node(next_node)
        prev_dp = getattr(node.params, 'target_pch_out_db', 0)
        dp = prev_dp
        prev_voa = 0
        voa = 0
        while True:
            # go through all nodes in the OMS (loop until next Roadm instance)
            if isinstance(node, elements.Edfa):
                node_loss = span_loss(network, prev_node)
                voa = node.out_voa if node.out_voa else 0
                if node.delta_p is None:
                    dp = target_power(network, next_node, equipment)
                else:
                    dp = node.delta_p
                gain_from_dp = node_loss + dp - prev_dp + prev_voa
                if node.effective_gain is None or power_mode:
                    gain_target = gain_from_dp
                else:  # gain mode with effective_gain
                    gain_target = node.effective_gain
                    dp = prev_dp - node_loss + gain_target

                power_target = pref_total_db + dp

                raman_allowed = False
                if isinstance(prev_node, elements.Fiber):
                    max_fiber_lineic_loss_for_raman = \
                        equipment['Span']['default'].max_fiber_lineic_loss_for_raman
                    raman_allowed = prev_node.params.loss_coef < max_fiber_lineic_loss_for_raman

                # implementation of restrictions on roadm boosters
                if isinstance(prev_node, elements.Roadm):
                    if prev_node.restrictions['booster_variety_list']:
                        restrictions = prev_node.restrictions['booster_variety_list']
                    else:
                        restrictions = None
                elif isinstance(next_node, elements.Roadm):
                    # implementation of restrictions on roadm preamp
                    if next_node.restrictions['preamp_variety_list']:
                        restrictions = next_node.restrictions['preamp_variety_list']
                    else:
                        restrictions = None
                else:
                    restrictions = None

                if node.params.type_variety == '':
                    edfa_variety, power_reduction = select_edfa(raman_allowed, gain_target, power_target, equipment, node.uid, restrictions)
                    extra_params = equipment['Edfa'][edfa_variety]
                    node.params.update_params(extra_params.__dict__)
                    dp += power_reduction
                    gain_target += power_reduction
                elif node.params.raman and not raman_allowed:
                    print(f'{ansi_escapes.red}WARNING{ansi_escapes.reset}: raman is used in node {node.uid}\n but fiber lineic loss is above threshold\n')

                node.delta_p = dp if power_mode else None
                node.effective_gain = gain_target
                set_amplifier_voa(node, power_target, power_mode)
            if isinstance(next_node, elements.Roadm) or isinstance(next_node, elements.Transceiver):
                break
            prev_dp = dp
            prev_voa = voa
            prev_node = node
            node = next_node
            # print(f'{node.uid}')
            next_node = next(n for n in network.successors(node))


def add_egress_amplifier(network, node):
    next_nodes = [n for n in network.successors(node)
                  if not (isinstance(n, elements.Transceiver) or isinstance(n, elements.Fused) or isinstance(n, elements.Edfa))]
    # no amplification for fused spans or TRX
    for i, next_node in enumerate(next_nodes):
        network.remove_edge(node, next_node)
        amp = elements.Edfa(
            uid=f'Edfa{i}_{node.uid}',
            params={},
            metadata={
                'location': {
                    'latitude': (node.lat * 2 + next_node.lat * 2) / 4,
                    'longitude': (node.lng * 2 + next_node.lng * 2) / 4,
                    'city': node.loc.city,
                    'region': node.loc.region,
                }
            },
            operational={
                'gain_target': None,
                'tilt_target': 0,
            })
        network.add_node(amp)
        if isinstance(node, elements.Fiber):
            edgeweight = node.params.length
        else:
            edgeweight = 0.01
        network.add_edge(node, amp, weight=edgeweight)
        network.add_edge(amp, next_node, weight=0.01)


def calculate_new_length(fiber_length, bounds, target_length):
    if fiber_length < bounds.stop:
        return fiber_length, 1

    n_spans = int(fiber_length // target_length)

    length1 = fiber_length / (n_spans + 1)
    delta1 = target_length - length1
    result1 = (length1, n_spans + 1)

    length2 = fiber_length / n_spans
    delta2 = length2 - target_length
    result2 = (length2, n_spans)

    if (bounds.start <= length1 <= bounds.stop) and not(bounds.start <= length2 <= bounds.stop):
        result = result1
    elif (bounds.start <= length2 <= bounds.stop) and not(bounds.start <= length1 <= bounds.stop):
        result = result2
    else:
        result = result1 if delta1 < delta2 else result2

    return result


def split_fiber(network, fiber, bounds, target_length, equipment):
    new_length, n_spans = calculate_new_length(fiber.params.length, bounds, target_length)
    if n_spans == 1:
        return

    try:
        next_node = next(network.successors(fiber))
        prev_node = next(network.predecessors(fiber))
    except StopIteration:
        raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')

    network.remove_node(fiber)

    fiber.params.length = new_length

    f = interp1d([prev_node.lng, next_node.lng], [prev_node.lat, next_node.lat])
    xpos = [prev_node.lng + (next_node.lng - prev_node.lng) * (n + 1) / (n_spans + 1) for n in range(n_spans)]
    ypos = f(xpos)
    for span, lng, lat in zip(range(n_spans), xpos, ypos):
        new_span = elements.Fiber(uid=f'{fiber.uid}_({span+1}/{n_spans})',
                                  type_variety=fiber.type_variety,
                                  metadata={
                                    'location': {
                                        'latitude': lat,
                                        'longitude': lng,
                                        'city': fiber.loc.city,
                                        'region': fiber.loc.region,
                                    }
                                  }, params=fiber.params.asdict())
        if isinstance(prev_node, elements.Fiber):
            edgeweight = prev_node.params.length
        else:
            edgeweight = 0.01
        network.add_edge(prev_node, new_span, weight=edgeweight)
        prev_node = new_span
    if isinstance(prev_node, elements.Fiber):
        edgeweight = prev_node.params.length
    else:
        edgeweight = 0.01
    network.add_edge(prev_node, next_node, weight=edgeweight)


def add_connector_loss(network, fibers, default_con_in, default_con_out, EOL):
    for fiber in fibers:
        if fiber.params.con_in is None:
            fiber.params.con_in = default_con_in
        if fiber.params.con_out is None:
            fiber.params.con_out = default_con_out
        next_node = next(n for n in network.successors(fiber))
        if not isinstance(next_node, elements.Fused):
            fiber.params.con_out += EOL


def add_fiber_padding(network, fibers, padding):
    """last_fibers = (fiber for n in network.nodes()
                         if not (isinstance(n, elements.Fiber) or isinstance(n, elements.Fused))
                         for fiber in network.predecessors(n)
                         if isinstance(fiber, elements.Fiber))"""
    for fiber in fibers:
        this_span_loss = span_loss(network, fiber)
        try:
            next_node = next(network.successors(fiber))
        except StopIteration:
            raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')
        if this_span_loss < padding and not (isinstance(next_node, elements.Fused)):
            # add a padding att_in at the input of the 1st fiber:
            # address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            # in order to support no booster , fused might be placed
            # just after a roadm: need to check that first_fiber is really a fiber
            if isinstance(first_fiber, elements.Fiber):
                if first_fiber.params.att_in is None:
                    first_fiber.params.att_in = padding - this_span_loss
                else:
                    first_fiber.params.att_in = first_fiber.params.att_in + padding - this_span_loss


def build_network(network, equipment, pref_ch_db, pref_total_db):
    default_span_data = equipment['Span']['default']
    max_length = int(convert_length(default_span_data.max_length, default_span_data.length_units))
    min_length = max(int(default_span_data.padding / 0.2 * 1e3), 50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, 90_000)
    default_con_in = default_span_data.con_in
    default_con_out = default_span_data.con_out
    padding = default_span_data.padding

    # set roadm loss for gain_mode before to build network
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    add_connector_loss(network, fibers, default_con_in, default_con_out, default_span_data.EOL)
    add_fiber_padding(network, fibers, padding)
    # don't group split fiber and add amp in the same loop
    # =>for code clarity (at the expense of speed):
    for fiber in fibers:
        split_fiber(network, fiber, bounds, target_length, equipment)

    amplified_nodes = [n for n in network.nodes() if isinstance(n, elements.Fiber) or isinstance(n, elements.Roadm)]

    for node in amplified_nodes:
        add_egress_amplifier(network, node)

    roadms = [r for r in network.nodes() if isinstance(r, elements.Roadm)]
    for roadm in roadms:
        set_egress_amplifier(network, roadm, equipment, pref_total_db)

    # support older json input topology wo Roadms:
    if len(roadms) == 0:
        trx = [t for t in network.nodes() if isinstance(t, elements.Transceiver)]
        for t in trx:
            set_egress_amplifier(network, t, equipment, pref_total_db)


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
