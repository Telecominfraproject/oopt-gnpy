#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.network
=================

Working with networks which consist of network elements
"""

from copy import deepcopy
from operator import attrgetter
from collections import namedtuple
from logging import getLogger

from gnpy.core import elements
from gnpy.core.exceptions import ConfigurationError, NetworkTopologyError
from gnpy.core.utils import round2float, convert_length, psd2powerdbm, lin2db, watt2dbm, dbm2watt
from gnpy.core.info import ReferenceCarrier, create_input_spectral_information
from gnpy.tools import json_io
from gnpy.core.parameters import SimParams


logger = getLogger(__name__)


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
    amp.slot_width = 50e9
    return amp._calc_nf(True)


def select_edfa(raman_allowed, gain_target, power_target, equipment, uid, restrictions=None, verbose=True):
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
        power=min(pin + edfa.gain_flatmax + TARGET_EXTENDED_GAIN, edfa.p_max) - power_target,
        gain_min=gain_target + 3 - edfa.gain_min,
        nf=edfa_nf(gain_target, edfa_variety, equipment))
        for edfa_variety, edfa in edfa_dict.items()
        if ((edfa.allowed_for_design or restrictions is not None) and not edfa.raman)]

    # consider a Raman list because of different gain_min requirement:
    # do not allow extended gain min for Raman
    raman_list = [Edfa_list(
        variety=edfa_variety,
        power=min(pin + edfa.gain_flatmax + TARGET_EXTENDED_GAIN, edfa.p_max) - power_target,
        gain_min=gain_target - edfa.gain_min,
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
            if verbose:
                logger.warning(f'\n\tWARNING: target gain in node {uid} is below all available amplifiers min gain: '
                               + '\n\tamplifier input padding will be assumed, consider increase span fiber padding '
                               + 'instead.\n')
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
    power_reduction = min(selected_edfa.power, 0)
    if power_reduction < -0.5 and verbose:
        logger.warning(f'\n\tWARNING: target gain and power in node {uid}\n'
                       + '\tis beyond all available amplifiers capabilities and/or extended_gain_range:\n'
                       + f'\ta power reduction of {round(power_reduction, 2)} is applied\n')
    return selected_edfa.variety, power_reduction


def target_power(network, node, equipment):  # get_fiber_dp
    """Computes target power using J. -L. Auge, V. Curri and E. Le Rouzic,
    Open Design for Multi-Vendor Optical Networks, OFC 2019.
    equation 4
    """
    if isinstance(node, elements.Roadm):
        return 0

    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    dp_range = list(equipment['Span']['default'].delta_power_range_db)
    node_loss = span_loss(network, node, equipment)

    try:
        dp = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except IndexError:
        raise ConfigurationError('invalid delta_power_range_db definition in eqpt_config[Span]'
                                 'delta_power_range_db: [lower_bound, upper_bound, step]')

    return dp


_fiber_fused_types = (elements.Fused, elements.Fiber)


def prev_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are either Fused or Fibers succeeded by Fused"""
    try:
        prev_node = next(network.predecessors(node))
    except StopIteration:
        if isinstance(node, elements.Transceiver):
            return
        raise NetworkTopologyError(f'Node {node.uid} is not properly connected, please check network topology')
    if ((isinstance(prev_node, elements.Fused) and isinstance(node, _fiber_fused_types)) or
            (isinstance(prev_node, _fiber_fused_types) and isinstance(node, elements.Fused))):
        yield prev_node
        yield from prev_node_generator(network, prev_node)


def next_node_generator(network, node):
    """fused spans interest:
    iterate over all predecessors while they are either Fused or Fibers preceded by Fused"""
    try:
        next_node = next(network.successors(node))
    except StopIteration:
        if isinstance(node, elements.Transceiver):
            return
        raise NetworkTopologyError(f'Node {node.uid} is not properly connected, please check network topology')

    if ((isinstance(next_node, elements.Fused) and isinstance(node, _fiber_fused_types)) or
            (isinstance(next_node, _fiber_fused_types) and isinstance(node, elements.Fused))):
        yield next_node
        yield from next_node_generator(network, next_node)


def estimate_raman_gain(node, equipment):
    """If node is RamanFiber, then estimate the possible Raman gain if any
    for this purpose propagate a fake signal in a copy.
    to be accurate the nb of channel should be the same as in SI, but this increases computation time
    """
    f_min = equipment['SI']['default'].f_min
    f_max = equipment['SI']['default'].f_max
    roll_off = equipment['SI']['default'].roll_off
    baud_rate = equipment['SI']['default'].baud_rate
    power_dbm = equipment['SI']['default'].power_dbm
    power = dbm2watt(equipment['SI']['default'].power_dbm)
    spacing = equipment['SI']['default'].spacing
    tx_osnr = equipment['SI']['default'].tx_osnr

    sim_params = {
        "raman_params": {
            "flag": True,
            "result_spatial_resolution": 10e3,
            "solver_spatial_resolution": 50
        },
        "nli_params": {
            "method": "ggn_spectrally_separated",
            "dispersion_tolerance": 1,
            "phase_shift_tolerance": 0.1,
            "computed_channels": [1, 18, 37, 56, 75]
        }
    }
    if isinstance(node, elements.RamanFiber):
        # in order to take into account gain generated in RamanFiber, propagate in the RamanFiber with
        # SI reference channel.
        spectral_info_input = create_input_spectral_information(f_min=f_min, f_max=f_max, roll_off=roll_off,
                                                                baud_rate=baud_rate, power=power, spacing=spacing,
                                                                tx_osnr=tx_osnr)
        n_copy = deepcopy(node)
        # need to set ref_pch_in_dbm in order to correctly run propagate of the element, because this
        # setting has not yet been done by autodesign
        n_copy.ref_pch_in_dbm = power_dbm
        SimParams.set_params(sim_params)
        pin = watt2dbm(sum(spectral_info_input.signal))
        spectral_info_out = n_copy(spectral_info_input)
        pout = watt2dbm(sum(spectral_info_out.signal))
        estimated_gain = pout - pin + node.loss
        return round(estimated_gain, 2)
    else:
        return 0.0


def span_loss(network, node, equipment):
    """Total loss of a span (Fiber and Fused nodes) which contains the given node"""
    loss = node.loss if node.passive else 0
    loss += sum(n.loss for n in prev_node_generator(network, node))
    loss += sum(n.loss for n in next_node_generator(network, node))
    # add the possible Raman gain
    gain = estimate_raman_gain(node, equipment)
    gain += sum(estimate_raman_gain(n, equipment) for n in prev_node_generator(network, node))
    gain += sum(estimate_raman_gain(n, equipment) for n in next_node_generator(network, node))

    return loss - gain


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
        if power_mode and amp.params.out_voa_auto:
            voa = min(amp.params.p_max - power_target,
                      amp.params.gain_flatmax - amp.effective_gain)
            voa = max(round2float(voa, 0.5) - VOA_MARGIN, 0)
            amp.delta_p = amp.delta_p + voa
            amp.effective_gain = amp.effective_gain + voa
        else:
            voa = 0  # no output voa optimization in gain mode
        amp.out_voa = voa


def set_egress_amplifier(network, this_node, equipment, pref_ch_db, pref_total_db, verbose):
    """This node can be a transceiver or a ROADM (same function called in both cases).
    go through each link staring from this_node until next Roadm or Transceiver and
    set gain and delta_p according to configurations set by user.
    power_mode = True, set amplifiers delta_p and effective_gain
    power_mode = False, set amplifiers effective_gain and ignore delta_p config: set it to None
    """
    power_mode = equipment['Span']['default'].power_mode
    next_oms = (n for n in network.successors(this_node) if not isinstance(n, elements.Transceiver))
    for oms in next_oms:
        # go through all the OMS departing from the ROADM
        prev_node = this_node
        node = oms
        if isinstance(this_node, elements.Transceiver):
            # for the time being use the same power for the target of roadms and for transceivers
            # TODO: This should be changed when introducing a power parameter dedicated to transceivers
            this_node_out_power = pref_ch_db
        if isinstance(this_node, elements.Roadm):
            # get target power out from ROADM for the reference carrier based on equalization settings
            this_node_out_power = this_node.get_per_degree_ref_power(degree=node.uid)
        # use the target power on this degree
        prev_dp = this_node_out_power - pref_ch_db
        dp = prev_dp
        prev_voa = 0
        voa = 0
        visited_nodes = []
        while not (isinstance(node, elements.Roadm) or isinstance(node, elements.Transceiver)):
            # go through all nodes in the OMS (loop until next Roadm instance)
            next_node = get_next_node(node, network)
            visited_nodes.append(node)
            if next_node in visited_nodes:
                raise NetworkTopologyError(f'Loop detected for {type(node).__name__} {node.uid}, '
                                           + 'please check network topology')
            if isinstance(node, elements.Edfa):
                node_loss = span_loss(network, prev_node, equipment)
                voa = node.out_voa if node.out_voa else 0
                if node.operational.delta_p is None:
                    dp = target_power(network, next_node, equipment) + voa
                else:
                    dp = node.operational.delta_p
                if node.effective_gain is None or power_mode:
                    gain_target = node_loss + dp - prev_dp + prev_voa
                else:  # gain mode with effective_gain
                    gain_target = node.effective_gain
                    dp = prev_dp - node_loss - prev_voa + gain_target

                power_target = pref_total_db + dp

                if isinstance(prev_node, elements.Fiber):
                    max_fiber_lineic_loss_for_raman = \
                        equipment['Span']['default'].max_fiber_lineic_loss_for_raman * 1e-3  # dB/m
                    raman_allowed = (prev_node.params.loss_coef < max_fiber_lineic_loss_for_raman).all()
                else:
                    raman_allowed = False

                if node.params.type_variety == '':
                    if node.variety_list and isinstance(node.variety_list, list):
                        restrictions = node.variety_list
                    elif isinstance(prev_node, elements.Roadm) and prev_node.restrictions['booster_variety_list']:
                        # implementation of restrictions on roadm boosters
                        restrictions = prev_node.restrictions['booster_variety_list']
                    elif isinstance(next_node, elements.Roadm) and next_node.restrictions['preamp_variety_list']:
                        # implementation of restrictions on roadm preamp
                        restrictions = next_node.restrictions['preamp_variety_list']
                    else:
                        restrictions = None
                    edfa_variety, power_reduction = select_edfa(raman_allowed, gain_target, power_target, equipment,
                                                                node.uid, restrictions, verbose)
                    extra_params = equipment['Edfa'][edfa_variety]
                    node.params.update_params(extra_params.__dict__)
                    dp += power_reduction
                    gain_target += power_reduction
                else:
                    # Check power saturation also in this case
                    p_max = equipment['Edfa'][node.params.type_variety].p_max
                    if power_mode:
                        power_reduction = min(0, p_max - (pref_total_db + dp))
                    else:
                        pout = pref_total_db + prev_dp - node_loss - prev_voa + gain_target
                        power_reduction = min(0, p_max - pout)
                    dp += power_reduction
                    gain_target += power_reduction
                    if node.params.raman and not raman_allowed:
                        if isinstance(prev_node, elements.Fiber):
                            logger.warning(f'\n\tWARNING: raman is used in node {node.uid}\n '
                                           + '\tbut fiber lineic loss is above threshold\n')
                        else:
                            logger.critical(f'\n\tWARNING: raman is used in node {node.uid}\n '
                                            + '\tbut previous node is not a fiber\n')
                    # if variety is imposed by user, and if the gain_target (computed or imposed) is also above
                    # variety max gain + extended range, then warn that gain > max_gain + extended range
                    if gain_target - equipment['Edfa'][node.params.type_variety].gain_flatmax - \
                            equipment['Span']['default'].target_extended_gain > 1e-2 and verbose:
                        # 1e-2 to allow a small margin according to round2float min step
                        logger.warning(f'\n\tWARNING: effective gain in Node {node.uid}\n'
                                       + f'\tis above user specified amplifier {node.params.type_variety}\n'
                                       + '\tmax flat gain: '
                                       + f'{equipment["Edfa"][node.params.type_variety].gain_flatmax}dB ; '
                                       + f'required gain: {round(gain_target, 2)}dB. Please check amplifier type.\n')

                node.delta_p = dp if power_mode else None
                node.effective_gain = gain_target
                # if voa is not set, then set it and possibly optimize it with gain and update delta_p and
                # effective_gain values
                set_amplifier_voa(node, power_target, power_mode)
                # set_amplifier_voa may change delta_p in power_mode
                node._delta_p = node.delta_p if power_mode else dp

                # target_pch_out_dbm records target power for design: If user defines one, then this is displayed,
                # else display the one computed during design
                if node.delta_p is not None and node.operational.delta_p is not None:
                    # use the user defined target
                    node.target_pch_out_dbm = round(node.operational.delta_p + pref_ch_db, 2)
                elif node.delta_p is not None:
                    # use the design target if no target were set
                    node.target_pch_out_dbm = round(node.delta_p + pref_ch_db, 2)
                elif node.delta_p is None:
                    node.target_pch_out_dbm = None

            prev_dp = dp
            prev_voa = voa
            prev_node = node
            node = next_node


def set_roadm_ref_carrier(roadm, equipment):
    """ref_carrier records carrier information used for design and usefull for equalization
    """
    roadm.ref_carrier = ReferenceCarrier(baud_rate=equipment['SI']['default'].baud_rate,
                                         slot_width=equipment['SI']['default'].spacing)


def set_roadm_per_degree_targets(roadm, network):
    """Set target powers/PSD on all degrees
    This is needed to populate per_degree_pch_out_dbm or per_degree_pch_psd or per_degree_pch_psw dicts when
    they are not initialized by users.
    """
    next_oms = (n for n in network.successors(roadm) if not isinstance(n, elements.Transceiver))

    for node in next_oms:
        # go through all the OMS departing from the ROADM
        if node.uid not in roadm.per_degree_pch_out_dbm and node.uid not in roadm.per_degree_pch_psd and \
                node.uid not in roadm.per_degree_pch_psw:
            # if no target power is defined on this degree or no per degree target power is given use the global one
            if roadm.params.target_pch_out_db:
                roadm.per_degree_pch_out_dbm[node.uid] = roadm.params.target_pch_out_db
            elif roadm.params.target_psd_out_mWperGHz:
                roadm.per_degree_pch_psd[node.uid] = roadm.params.target_psd_out_mWperGHz
            elif roadm.params.target_out_mWperSlotWidth:
                roadm.per_degree_pch_psw[node.uid] = roadm.params.target_out_mWperSlotWidth
            else:
                raise ConfigurationError(roadm.uid, 'needs an equalization target')


def set_roadm_input_powers(network, roadm, equipment, pref_ch_db):
    """Set reference powers at ROADM input for a reference channel and based on the adjacent OMS.
    This supposes that there is no dependency on path. For example, the succession:
    node                             power out of element
    roadm A (target power -10dBm)   -10dBm
    fiber A (16 dB loss)            -26dBm
    roadm B (target power -12dBm)   -26dBm
    fiber B (10 dB loss)            -36dBm
    roadm C (target power -14dBm)   -36dBm
    is not consistent because target powers in roadm B and roadm C can not be met.
    input power for the reference channel will be set -26 dBm in roadm B and -22dBm in roadm C,
    because at design time we can not know about path.
    The function raises a warning if target powers can not be met with the design.
    User should be aware that design was not successfull and that power reduction was applied.
    Note that this value is only used for visualisation purpose (to compute ROADM loss in elements).
    """
    previous_elements = [n for n in network.predecessors(roadm)]
    roadm.ref_pch_in_dbm = {}
    for element in previous_elements:
        node = element
        loss = 0.0
        while isinstance(node, (elements.Fiber, elements.Fused, elements.RamanFiber)):
            # go through all predecessors until a power target is found either in an amplifier, a ROADM or a transceiver
            # then deduce power at ROADM input from this degree based on this target and crossed losses
            loss += node.loss
            previous_node = node
            node = next(network.predecessors(node))
        if isinstance(node, elements.Edfa):
            roadm.ref_pch_in_dbm[element.uid] = pref_ch_db + node._delta_p - node.out_voa - loss
        elif isinstance(node, elements.Roadm):
            roadm.ref_pch_in_dbm[element.uid] = \
                node.get_per_degree_ref_power(degree=previous_node.uid) - loss
        elif isinstance(node, elements.Transceiver):
            roadm.ref_pch_in_dbm[element.uid] = pref_ch_db - loss
    # check if target power can be met
    temp = []
    if roadm.per_degree_pch_out_dbm:
        temp.append(max([p for p in roadm.per_degree_pch_out_dbm.values()]))
    if roadm.per_degree_pch_psd:
        temp.append(max([psd2powerdbm(p, roadm.ref_carrier.baud_rate) for p in roadm.per_degree_pch_psd.values()]))
    if roadm.per_degree_pch_psw:
        temp.append(max([psd2powerdbm(p, roadm.ref_carrier.slot_width) for p in roadm.per_degree_pch_psw.values()]))
    if roadm.params.target_pch_out_db:
        temp.append(roadm.params.target_pch_out_db)
    if roadm.params.target_psd_out_mWperGHz:
        temp.append(psd2powerdbm(roadm.params.target_psd_out_mWperGHz, roadm.ref_carrier.baud_rate))
    if roadm.params.target_out_mWperSlotWidth:
        temp.append(psd2powerdbm(roadm.params.target_out_mWperSlotWidth, roadm.ref_carrier.slot_width))
    if not temp:
        raise ConfigurationError(f'Could not find target power/PSD/PSW in ROADM "{roadm.uid}"')
    target_to_be_supported = max(temp)
    for from_degree, in_power in roadm.ref_pch_in_dbm.items():
        if in_power < target_to_be_supported:
            logger.warning(
                f'WARNING: maximum target power {target_to_be_supported}dBm '
                + f'in ROADM "{roadm.uid}" can not be met for at least one crossing path. Min input power '
                + f'from "{from_degree}" direction is {round(in_power, 2)}dBm. Please correct input topology.'
            )


def set_fiber_input_power(network, fiber, equipment, pref_ch_db):
    """Set reference powers at fiber input for a reference channel.
    Supposes that target power out of ROADMs and amplifiers are consistent.
    This is only for visualisation purpose
    """
    loss = 0.0
    node = next(network.predecessors(fiber))
    while isinstance(node, elements.Fused):
        loss += node.loss
        previous_node = node
        node = next(network.predecessors(node))
    if isinstance(node, (elements.Fiber, elements.RamanFiber)) and node.ref_pch_in_dbm is not None:
        fiber.ref_pch_in_dbm = node.ref_pch_in_dbm - loss - node.loss
    if isinstance(node, (elements.Fiber, elements.RamanFiber)) and node.ref_pch_in_dbm is None:
        set_fiber_input_power(network, node, equipment, pref_ch_db)
        fiber.ref_pch_in_dbm = node.ref_pch_in_dbm - loss - node.loss
    elif isinstance(node, elements.Roadm):
        fiber.ref_pch_in_dbm = \
            node.get_per_degree_ref_power(degree=previous_node.uid) - loss
    elif isinstance(node, elements.Edfa):
        fiber.ref_pch_in_dbm = pref_ch_db + node._delta_p - node.out_voa - loss
    elif isinstance(node, elements.Transceiver):
        fiber.ref_pch_in_dbm = pref_ch_db - loss


def add_roadm_booster(network, roadm):
    next_nodes = [n for n in network.successors(roadm)
                  if not (isinstance(n, elements.Transceiver) or isinstance(n, elements.Fused)
                  or isinstance(n, elements.Edfa))]
    # no amplification for fused spans or TRX
    for next_node in next_nodes:
        network.remove_edge(roadm, next_node)
        amp = elements.Edfa(
            uid=f'Edfa_booster_{roadm.uid}_to_{next_node.uid}',
            params=json_io.Amp.default_values,
            metadata={
                'location': {
                    'latitude': roadm.lat,
                    'longitude': roadm.lng,
                    'city': roadm.loc.city,
                    'region': roadm.loc.region,
                }
            },
            operational={
                'gain_target': None,
                'tilt_target': 0,
            })
        network.add_node(amp)
        network.add_edge(roadm, amp, weight=0.01)
        network.add_edge(amp, next_node, weight=0.01)


def add_roadm_preamp(network, roadm):
    prev_nodes = [n for n in network.predecessors(roadm)
                  if not (isinstance(n, elements.Transceiver) or isinstance(n, elements.Fused) or isinstance(n, elements.Edfa))]
    # no amplification for fused spans or TRX
    for prev_node in prev_nodes:
        network.remove_edge(prev_node, roadm)
        amp = elements.Edfa(
            uid=f'Edfa_preamp_{roadm.uid}_from_{prev_node.uid}',
            params=json_io.Amp.default_values,
            metadata={
                'location': {
                    'latitude': roadm.lat,
                    'longitude': roadm.lng,
                    'city': roadm.loc.city,
                    'region': roadm.loc.region,
                }
            },
            operational={
                'gain_target': None,
                'tilt_target': 0,
            })
        network.add_node(amp)
        if isinstance(prev_node, elements.Fiber):
            edgeweight = prev_node.params.length
        else:
            edgeweight = 0.01
        network.add_edge(prev_node, amp, weight=edgeweight)
        network.add_edge(amp, roadm, weight=0.01)


def add_inline_amplifier(network, fiber):
    next_node = get_next_node(fiber, network)
    if isinstance(next_node, elements.Fiber) or isinstance(next_node, elements.RamanFiber):
        # no amplification for fused spans or TRX
        network.remove_edge(fiber, next_node)
        amp = elements.Edfa(
            uid=f'Edfa_{fiber.uid}',
            params=json_io.Amp.default_values,
            metadata={
                'location': {
                    'latitude': (fiber.lat + next_node.lat) / 2,
                    'longitude': (fiber.lng + next_node.lng) / 2,
                    'city': fiber.loc.city,
                    'region': fiber.loc.region,
                }
            },
            operational={
                'gain_target': None,
                'tilt_target': 0,
            })
        network.add_node(amp)
        network.add_edge(fiber, amp, weight=fiber.params.length)
        network.add_edge(amp, next_node, weight=0.01)


def calculate_new_length(fiber_length, bounds, target_length):
    """If fiber is over boundary, then assume this is a link "intent" and computes the set of
    identical fiber spans this link should be composed of.
    """
    if fiber_length < bounds.stop:
        return fiber_length, 1

    n_spans2 = int(fiber_length // target_length)
    n_spans1 = n_spans2 + 1

    length1 = fiber_length / n_spans1
    length2 = fiber_length / n_spans2

    if (bounds.start <= length1 <= bounds.stop) and not(bounds.start <= length2 <= bounds.stop):
        return (length1, n_spans1)
    elif (bounds.start <= length2 <= bounds.stop) and not(bounds.start <= length1 <= bounds.stop):
        return (length2, n_spans2)
    elif length2 - target_length <= target_length - length1 and length2 <= bounds.stop:
        return (length2, n_spans2)
    else:
        return (length1, n_spans1)


def get_next_node(node, network):
    """get_next node else raise tha appropriate error
    """
    try:
        next_node = next(network.successors(node))
        return next_node
    except StopIteration:
        raise NetworkTopologyError(
            f'{type(node).__name__} {node.uid} is not properly connected, please check network topology')


def split_fiber(network, fiber, bounds, target_length):
    """If fiber length exceeds boundary then assume this is a link "intent", and replace this one-span link
    with an n_spans link, with identical fiber types.
    """
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

    xpos = [prev_node.lng + (next_node.lng - prev_node.lng) * (n + 0.5) / n_spans for n in range(n_spans)]
    ypos = [prev_node.lat + (next_node.lat - prev_node.lat) * (n + 0.5) / n_spans for n in range(n_spans)]
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
                         },
                         params=fiber.params.asdict())
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
    """Add default connector loss if no loss are defined. EOL repair margin is added as a connector loss
    """
    for fiber in fibers:
        next_node = get_next_node(fiber, network)
        if fiber.params.con_in is None:
            fiber.params.con_in = default_con_in
        if fiber.params.con_out is None:
            fiber.params.con_out = default_con_out
        if not isinstance(next_node, elements.Fused):
            fiber.params.con_out += EOL


def add_fiber_padding(network, fibers, padding, equipment):
    """Add a padding att_in at the input of the 1st fiber of a succession of fibers and fused
    """
    for fiber in fibers:
        next_node = get_next_node(fiber, network)
        if isinstance(next_node, elements.Fused):
            continue
        this_span_loss = span_loss(network, fiber, equipment)
        if this_span_loss < padding:
            # add a padding att_in at the input of the 1st fiber:
            # address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            # in order to support no booster , fused might be placed
            # just after a roadm: need to check that first_fiber is really a fiber
            if isinstance(first_fiber, elements.Fiber):
                first_fiber.params.att_in = first_fiber.params.att_in + padding - this_span_loss


def add_missing_elements_in_network(network, equipment):
    """Autodesign network: add missing elements. split fibers if their length is too big
    add ROADM preamp or booster and inline amplifiers between fibers
    """
    default_span_data = equipment['Span']['default']
    max_length = int(convert_length(default_span_data.max_length, default_span_data.length_units))
    min_length = max(int(default_span_data.padding / 0.2 * 1e3), 50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, min(max_length, 90_000))
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    for fiber in fibers:
        split_fiber(network, fiber, bounds, target_length)
    roadms = [r for r in network.nodes() if isinstance(r, elements.Roadm)]
    for roadm in roadms:
        add_roadm_preamp(network, roadm)
        add_roadm_booster(network, roadm)
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    for fiber in fibers:
        add_inline_amplifier(network, fiber)


def add_missing_fiber_attributes(network, equipment):
    """Fill in connector loss with default values. Add the padding loss is required.
    EOL is added as a connector loss
    """
    default_span_data = equipment['Span']['default']
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    add_connector_loss(network, fibers, default_span_data.con_in, default_span_data.con_out, default_span_data.EOL)
    # don't group split fiber and add amp in the same loop
    # =>for code clarity (at the expense of speed):
    add_fiber_padding(network, fibers, default_span_data.padding, equipment)


def build_network(network, equipment, pref_ch_db, pref_total_db, set_connector_losses=True, verbose=True):
    """Set roadm equalization target and amplifier gain and power
    """
    roadms = [r for r in network.nodes() if isinstance(r, elements.Roadm)]
    transceivers = [t for t in network.nodes() if isinstance(t, elements.Transceiver)]

    if set_connector_losses:
        add_missing_fiber_attributes(network, equipment)
    # set roadm equalization targets first
    for roadm in roadms:
        set_roadm_ref_carrier(roadm, equipment)
        set_roadm_per_degree_targets(roadm, network)
    # then set amplifiers gain, delta_p and out_voa on each OMS
    for roadm in roadms + transceivers:
        set_egress_amplifier(network, roadm, equipment, pref_ch_db, pref_total_db, verbose)
    for roadm in roadms:
        set_roadm_input_powers(network, roadm, equipment, pref_ch_db)
    for fiber in [f for f in network.nodes() if isinstance(f, (elements.Fiber, elements.RamanFiber))]:
        set_fiber_input_power(network, fiber, equipment, pref_ch_db)


def design_network(reference_channel, network, equipment, set_connector_losses=True, verbose=True):
    """Network is designed according to reference channel. Verbose indicate if the function should
    print all warnings or not
    """
    pref_ch_db = watt2dbm(reference_channel.power)  # reference channel power
    pref_total_db = pref_ch_db + lin2db(reference_channel.nb_channel)  # reference total power
    build_network(network, equipment, pref_ch_db, pref_total_db, set_connector_losses=set_connector_losses,
                  verbose=verbose)
