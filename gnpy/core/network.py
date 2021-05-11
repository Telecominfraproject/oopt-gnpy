#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.network
=================

Working with networks which consist of network elements
'''

from operator import attrgetter
from gnpy.core import ansi_escapes, elements
from gnpy.core.exceptions import ConfigurationError, NetworkTopologyError
from gnpy.core.utils import round2float, convert_length
from collections import namedtuple


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
    if isinstance(node, elements.Roadm):
        return 0

    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    dp_range = list(equipment['Span']['default'].delta_power_range_db)
    node_loss = span_loss(network, node)

    try:
        dp = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except IndexError:
        raise ConfigurationError(f'invalid delta_power_range_db definition in eqpt_config[Span]'
                                 f'delta_power_range_db: [lower_bound, upper_bound, step]')

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


def span_loss(network, node):
    """Total loss of a span (Fiber and Fused nodes) which contains the given node"""
    loss = node.loss if node.passive else 0
    loss += sum(n.loss for n in prev_node_generator(network, node))
    loss += sum(n.loss for n in next_node_generator(network, node))
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
        if power_mode and amp.params.out_voa_auto:
            voa = min(amp.params.p_max - power_target,
                      amp.params.gain_flatmax - amp.effective_gain)
            voa = max(round2float(voa, 0.5) - VOA_MARGIN, 0)
            amp.delta_p = amp.delta_p + voa
            amp.effective_gain = amp.effective_gain + voa
        else:
            voa = 0  # no output voa optimization in gain mode
        amp.out_voa = voa


def set_egress_amplifier(network, this_node, equipment, pref_ch_db, pref_total_db):
    """ this node can be a transceiver or a ROADM (same function called in both cases)
    """
    power_mode = equipment['Span']['default'].power_mode
    next_oms = (n for n in network.successors(this_node) if not isinstance(n, elements.Transceiver))
    this_node_degree = {k: v for k, v in this_node.per_degree_pch_out_db.items()} if hasattr(this_node, 'per_degree_pch_out_db') else {}
    for oms in next_oms:
        # go through all the OMS departing from the ROADM
        prev_node = this_node
        node = oms
        # if isinstance(next_node, elements.Fused): #support ROADM wo egress amp for metro applications
        #     node = find_last_node(next_node)
        #     next_node = next(n for n in network.successors(node))
        #     next_node = find_last_node(next_node)
        if node.uid not in this_node_degree:
            # if no target power is defined on this degree or no per degree target power is given use the global one
            # if target_pch_out_db  is not an attribute, then the element must be a transceiver
            this_node_degree[node.uid] = getattr(this_node.params, 'target_pch_out_db', 0)
        # use the target power on this degree
        prev_dp = this_node_degree[node.uid] - pref_ch_db
        dp = prev_dp
        prev_voa = 0
        voa = 0
        visited_nodes = []
        while not (isinstance(node, elements.Roadm) or isinstance(node, elements.Transceiver)):
            # go through all nodes in the OMS (loop until next Roadm instance)
            try:
                next_node = next(network.successors(node))
            except StopIteration:
                raise NetworkTopologyError(f'{type(node).__name__} {node.uid} is not properly connected, please check network topology')
            visited_nodes.append(node)
            if next_node in visited_nodes:
                raise NetworkTopologyError(f'Loop detected for {type(node).__name__} {node.uid}, please check network topology')
            if isinstance(node, elements.Edfa):
                node_loss = span_loss(network, prev_node)
                voa = node.out_voa if node.out_voa else 0
                if node.delta_p is None:
                    dp = target_power(network, next_node, equipment)
                else:
                    dp = node.delta_p
                if node.effective_gain is None or power_mode:
                    gain_target = node_loss + dp - prev_dp + prev_voa
                else:  # gain mode with effective_gain
                    gain_target = node.effective_gain
                    dp = prev_dp - node_loss - prev_voa + gain_target

                power_target = pref_total_db + dp

                if isinstance(prev_node, elements.Fiber):
                    max_fiber_lineic_loss_for_raman = \
                        equipment['Span']['default'].max_fiber_lineic_loss_for_raman
                    raman_allowed = prev_node.params.loss_coef < max_fiber_lineic_loss_for_raman
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
                    edfa_variety, power_reduction = select_edfa(raman_allowed, gain_target, power_target, equipment, node.uid, restrictions)
                    extra_params = equipment['Edfa'][edfa_variety]
                    node.params.update_params(extra_params.__dict__)
                    dp += power_reduction
                    gain_target += power_reduction
                elif node.params.raman and not raman_allowed:
                    print(f'{ansi_escapes.red}WARNING{ansi_escapes.reset}: raman is used in node {node.uid}\n but fiber lineic loss is above threshold\n')
                else:
                    # if variety is imposed by user, and if the gain_target (computed or imposed) is also above
                    # variety max gain + extended range, then warn that gain > max_gain + extended range
                    if gain_target - equipment['Edfa'][node.params.type_variety].gain_flatmax - \
                            equipment['Span']['default'].target_extended_gain > 1e-2:
                        # 1e-2 to allow a small margin according to round2float min step
                        print(f'{ansi_escapes.red}WARNING{ansi_escapes.reset}: '
                              f'WARNING: effective gain in Node {node.uid} is above user '
                              f'specified amplifier {node.params.type_variety}\n'
                              f'max flat gain: {equipment["Edfa"][node.params.type_variety].gain_flatmax}dB ; '
                              f'required gain: {gain_target}dB. Please check amplifier type.')

                node.delta_p = dp if power_mode else None
                node.effective_gain = gain_target
                set_amplifier_voa(node, power_target, power_mode)

            prev_dp = dp
            prev_voa = voa
            prev_node = node
            node = next_node
            # print(f'{node.uid}')

    if isinstance(this_node, elements.Roadm):
        this_node.per_degree_pch_out_db = {k: v for k, v in this_node_degree.items()}


def add_roadm_booster(network, roadm):
    next_nodes = [n for n in network.successors(roadm)
                  if not (isinstance(n, elements.Transceiver) or isinstance(n, elements.Fused) or isinstance(n, elements.Edfa))]
    # no amplification for fused spans or TRX
    for next_node in next_nodes:
        network.remove_edge(roadm, next_node)
        amp = elements.Edfa(
            uid=f'Edfa_booster_{roadm.uid}_to_{next_node.uid}',
            params={},
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
            params={},
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
    next_node = next(network.successors(fiber))
    if isinstance(next_node, elements.Fiber) or isinstance(next_node, elements.RamanFiber):
        # no amplification for fused spans or TRX
        network.remove_edge(fiber, next_node)
        amp = elements.Edfa(
            uid=f'Edfa_{fiber.uid}',
            params={},
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
    elif target_length - length1 < length2 - target_length:
        return (length1, n_spans1)
    else:
        return (length2, n_spans2)


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
    for fiber in fibers:
        try:
            next_node = next(network.successors(fiber))
        except StopIteration:
            raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')
        if fiber.params.con_in is None:
            fiber.params.con_in = default_con_in
        if fiber.params.con_out is None:
            fiber.params.con_out = default_con_out
        if not isinstance(next_node, elements.Fused):
            fiber.params.con_out += EOL


def add_fiber_padding(network, fibers, padding):
    """last_fibers = (fiber for n in network.nodes()
                         if not (isinstance(n, elements.Fiber) or isinstance(n, elements.Fused))
                         for fiber in network.predecessors(n)
                         if isinstance(fiber, elements.Fiber))"""
    for fiber in fibers:
        try:
            next_node = next(network.successors(fiber))
        except StopIteration:
            raise NetworkTopologyError(f'Fiber {fiber.uid} is not properly connected, please check network topology')
        if isinstance(next_node, elements.Fused):
            continue
        this_span_loss = span_loss(network, fiber)
        if this_span_loss < padding:
            # add a padding att_in at the input of the 1st fiber:
            # address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            # in order to support no booster , fused might be placed
            # just after a roadm: need to check that first_fiber is really a fiber
            if isinstance(first_fiber, elements.Fiber):
                first_fiber.params.att_in = first_fiber.params.att_in + padding - this_span_loss


def build_network(network, equipment, pref_ch_db, pref_total_db):
    default_span_data = equipment['Span']['default']
    max_length = int(convert_length(default_span_data.max_length, default_span_data.length_units))
    min_length = max(int(default_span_data.padding / 0.2 * 1e3), 50_000)
    bounds = range(min_length, max_length)
    target_length = max(min_length, 90_000)

    # set roadm loss for gain_mode before to build network
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    add_connector_loss(network, fibers, default_span_data.con_in, default_span_data.con_out, default_span_data.EOL)
    add_fiber_padding(network, fibers, default_span_data.padding)
    # don't group split fiber and add amp in the same loop
    # =>for code clarity (at the expense of speed):
    for fiber in fibers:
        split_fiber(network, fiber, bounds, target_length, equipment)

    roadms = [r for r in network.nodes() if isinstance(r, elements.Roadm)]
    for roadm in roadms:
        add_roadm_preamp(network, roadm)
        add_roadm_booster(network, roadm)

    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    for fiber in fibers:
        add_inline_amplifier(network, fiber)

    for roadm in roadms:
        set_egress_amplifier(network, roadm, equipment, pref_ch_db, pref_total_db)

    trx = [t for t in network.nodes() if isinstance(t, elements.Transceiver)]
    for t in trx:
        next_node = next(network.successors(t), None)
        if next_node and not isinstance(next_node, elements.Roadm):
            set_egress_amplifier(network, t, equipment, 0, pref_total_db)
