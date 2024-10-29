#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.core.network: Working with networks which consist of network elements
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.core.network
=================

Working with networks which consist of network elements
"""

from operator import attrgetter
from collections import namedtuple
from functools import reduce
from logging import getLogger
from typing import Tuple, List, Optional, Union, Dict, Iterator
from networkx import DiGraph
from numpy import allclose
import warnings

from gnpy.core import elements
from gnpy.core.equipment import find_type_variety, find_type_varieties
from gnpy.core.exceptions import ConfigurationError, NetworkTopologyError
from gnpy.core.utils import round2float, convert_length, psd2powerdbm, lin2db, watt2dbm, dbm2watt, automatic_nch, \
    find_common_range, get_spacing_from_band, reorder_per_degree_design_bands
from gnpy.core.info import ReferenceCarrier, create_input_spectral_information
from gnpy.core.parameters import SimParams, EdfaParams, find_band_name, FrequencyBand, MultiBandParams
from gnpy.core.science_utils import RamanSolver


logger = getLogger(__name__)

ELEMENT_TYPES = Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                      elements.Transceiver, elements.Transceiver]
PASSIVE_ELEMENT_TYPES = Union[elements.Fiber, elements.Roadm, elements.Fused]


def edfa_nf(gain_target: float, amp_params) -> float:
    """Calculates the noise figure (NF) of an EDFA (Erbium-Doped Fiber Amplifier)
    based on the specified gain target and amplifier parameters.

    This function creates an EDFA instance with the given parameters and
    computes its noise figure using the internal calculation method.

    :param gain_target: The target gain for which the noise figure is to be calculated.
    :type gain_target: float
    :param amp_params: An object containing the amplifier parameters.
    :type amp_params: gnpy.tools.json_io.Amp

    :return: The calculated noise figure (NF) of the EDFA in dB.
    :rtype: float

    :raises ValueError: If the gain_target is not achievable with the given parameters.
    """
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


def select_edfa(raman_allowed: bool, gain_target: float, power_target: float, edfa_eqpt: dict, uid: str,
                target_extended_gain: float, verbose: Optional[bool] = True) -> Tuple[str, float]:
    """Selects an amplifier within a library based on specified parameters.

    This function implements an amplifier selection algorithm that considers
    various constraints, including gain and power targets.
    It can also handle Raman amplifiers if allowed.
    edfa_eqpt dict has already filtered out the amplifiers that do not match any other restrictions
    such as ROADM booster or preamp restrictions or frequency constraints.

    :param raman_allowed: Indicates whether Raman amplifiers are permitted in the selection.
    :type raman_allowed: bool
    :param gain_target: The target gain that the selected amplifier should achieve.
    :type gain_target: float
    :param power_target: The target power level for the amplifier.
    :type power_target: float
    :param edfa_eqpt: A dictionary containing available EDFA equipment, where keys are
                      amplifier names and values are amplifier objects.
    :type edfa_eqpt: dict
    :param uid: A unique identifier for the node where the amplifier will be used.
    :type uid: str
    :param target_extended_gain: The extended gain target derived from configuration settings.
    :type target_extended_gain: float
    :param verbose: If True, enables verbose logging of warnings and information.
    :type verbose: Optional[bool], default=True

    :return: A tuple containing the selected amplifier's variety and the power
             reduction applied (if any).
    :rtype: Tuple[str, float]

    :raises ConfigurationError: If no amplifiers meet the minimum gain requirement for the specified node.

    Notes:
    ------
    - The function considers both gain and power limitations when selecting
      an amplifier.
    - If no suitable amplifier is found or if the target gain exceeds the
      capabilities of available amplifiers, a warning is logged.

    Author:
    -------
    Jean-Luc Augé
    """
    try:
        tilt_target = 0
        with warnings.catch_warnings(record=True) as caught_warnings:
            acceptable_power_list = \
                filter_edfa_list_based_on_targets(uid, edfa_eqpt, power_target, gain_target,
                                                  tilt_target, target_extended_gain,
                                                  raman_allowed, verbose)
            if caught_warnings:
                msg = f'In node {uid}: {caught_warnings[0].message}'
                logger.warning(msg)
    except ConfigurationError as e:
        raise ConfigurationError(f'in node {uid}, {e}')
    # gain and power requirements are resolved,
    #       =>chose the amp with the best NF among the acceptable ones:
    selected_edfa = min(acceptable_power_list, key=attrgetter('nf'))  # filter on NF
    # check what are the gain and power limitations of this amp
    power_reduction = min(selected_edfa.power, 0.0)
    if power_reduction < -0.5 and verbose:
        logger.warning(f'\n\tWARNING: target gain and power in node {uid}\n'
                       + '\tis beyond all available amplifiers capabilities and/or extended_gain_range:\n'
                       + f'\ta power reduction of {round(power_reduction, 2)} is applied\n')
    return selected_edfa.variety, power_reduction


def target_power(network: DiGraph, node: PASSIVE_ELEMENT_TYPES,
                 equipment: Dict, deviation_db: float) -> float:  # get_fiber_dp
    """Computes the target power deviation (delta_p) for a given node in the network.

    This function uses the methodology described by J. -L. Auge, V. Curri, and E. Le Rouzic
    in "Open Design for Multi-Vendor Optical Networks," OFC 2019, specifically equation 4.

    :param network: The network object containing the topology and elements.
    :type network: DiGraph
    :param node: The node for which the target power is being calculated.
    :type node: elements.Fiber
    :param equipment: A dictionary containing equipment parameters and configurations.
    :type equipment: dict
    :param deviation_db: The deviation in dB to be applied to the calculated power, for example due to SRS power tilt.
    :type deviation_db: float

    :return: The computed target power deviation in dB.
    :rtype: float

    :raises ConfigurationError: If the delta power range is not defined correctly in the equipment configuration.

    Notes:
    ------
    - If the node is a ROADM, the function returns 0.
    - The target power is calculated based on the span loss and adjusted by the specified deviation.
    """
    if isinstance(node, elements.Roadm):
        return 0

    SPAN_LOSS_REF = 20
    POWER_SLOPE = 0.3
    dp_range = list(equipment['Span']['default'].delta_power_range_db)
    node_loss = span_loss(network, node, equipment) + deviation_db

    try:
        dp = round2float((node_loss - SPAN_LOSS_REF) * POWER_SLOPE, dp_range[2])
        dp = max(dp_range[0], dp)
        dp = min(dp_range[1], dp)
    except IndexError:
        raise ConfigurationError('invalid delta_power_range_db definition in eqpt_config[Span]'
                                 'delta_power_range_db: [lower_bound, upper_bound, step]')

    return dp


_fiber_fused_types = (elements.Fused, elements.Fiber)


def prev_node_generator(network: DiGraph, node: ELEMENT_TYPES) -> Iterator[Union[elements.Fiber, elements.Fused]]:
    """Generates predecessor nodes in the network that are either Fused or Fibers succeeded by Fused.

    This function iterates over all predecessors of the specified node, yielding nodes that meet the criteria
    of being either a Fused element or a Fiber that is succeeded by a Fused element.

    :param network: The directed graph representing the network topology.
    :type network: DiGraph
    :param node: The current node for which predecessors are being generated.
    :type node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                      elements.Transceiver, elements.Transceiver]

    :yield: The predecessor nodes that are either Fused or Fibers succeeded by Fused.
    :rtype: Iterator[Union[elements.Fiber, elements.Fused]

    :raises NetworkTopologyError: If the node is not properly connected in the network topology.

    Notes:
    ------
    - If the node is a Transceiver and has no predecessors, the function will return without yielding any nodes.
    """
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


def next_node_generator(network: DiGraph, node: ELEMENT_TYPES) -> Iterator[Union[elements.Fiber, elements.Fused]]:
    """Generates successor nodes in the network that are either Fused or Fibers preceded by Fused.

    This function iterates over all successors of the specified node, yielding nodes that meet the criteria
    of being either a Fused element or a Fiber that is preceded by a Fused element.

    :param network: The directed graph representing the network topology.
    :type network: DiGraph
    :param node: The current node for which successors are being generated.
    :type node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                      elements.Transceiver, elements.Transceiver]

    :yield: The successor nodes that are either Fused or Fibers preceded by Fused.
    :rtype: Iterator[Union[elements.Fiber, elements.Fused]

    :raises NetworkTopologyError: If the node is not properly connected in the network topology.

    Notes:
    ------
    - If the node is a Transceiver and has no successors, the function will return without yielding any nodes.
    """
    try:
        next_node = next(network.successors(node))
    except StopIteration:
        if isinstance(node, elements.Transceiver):
            return
        raise NetworkTopologyError(f'Node {node.uid} is not properly connected, please check network topology')

    if ((isinstance(next_node, elements.Fused) and isinstance(node, _fiber_fused_types))
            or (isinstance(next_node, _fiber_fused_types) and isinstance(node, elements.Fused))):
        yield next_node
        yield from next_node_generator(network, next_node)


def estimate_raman_gain(node: ELEMENT_TYPES, equipment: DiGraph, power_dbm: float) -> float:
    """Estimates the possible Raman gain for a given node if it is a RamanFiber.

    This function computes the stimulated Raman scattering loss profile to estimate the gain.
    Note that this computation may be time-consuming.

    :param node: The node for which the Raman gain is to be estimated.
    :type node: elements
    :param equipment: A dictionary containing equipment parameters and configurations.
    :type equipment: dict
    :param power_dbm: The input power in dBm to be used for the gain estimation.
    :type power_dbm: float

    :return: The estimated Raman gain in dB, or 0.0 if the node is not a RamanFiber.
    :rtype: float

    :raises ValueError: If the node is not properly configured or if required parameters are missing.

    Notes:
    ------
    - If the node is a RamanFiber and has already estimated gain, it will return the cached value.
    - The function applies various parameters from the equipment configuration to compute the gain.
    """

    if isinstance(node, elements.RamanFiber):
        if hasattr(node, "estimated_gain"):
            return node.estimated_gain
        f_min = equipment['SI']['default'].f_min
        f_max = equipment['SI']['default'].f_max
        roll_off = equipment['SI']['default'].roll_off
        baud_rate = equipment['SI']['default'].baud_rate
        power = dbm2watt(power_dbm)
        spacing = equipment['SI']['default'].spacing
        tx_osnr = equipment['SI']['default'].tx_osnr

        # reduce the nb of channels to speed up
        spacing = spacing * 3
        power = power * 3

        sim_params = {
            "raman_params": {
                "flag": True,
                "result_spatial_resolution": 50e3,
                "solver_spatial_resolution": 100
            }
        }

        # in order to take into account gain generated in RamanFiber, propagate in the RamanFiber with
        if hasattr(node, "estimated_gain"):
            # do not compute twice to save on time
            return node.estimated_gain
        spectral_info = create_input_spectral_information(f_min=f_min, f_max=f_max, roll_off=roll_off,
                                                          baud_rate=baud_rate, tx_power=power, spacing=spacing,
                                                          tx_osnr=tx_osnr)
        pin = watt2dbm(sum(spectral_info.signal))
        attenuation_in_db = node.params.con_in + node.params.att_in
        spectral_info.apply_attenuation_db(attenuation_in_db)
        save_sim_params = {"raman_params": SimParams._shared_dict['raman_params'].to_json(),
                           "nli_params": SimParams._shared_dict['nli_params'].to_json()}
        SimParams.set_params(sim_params)
        stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, node)
        attenuation_fiber = stimulated_raman_scattering.loss_profile[:spectral_info.number_of_channels, -1]
        spectral_info.apply_attenuation_lin(attenuation_fiber)
        attenuation_out_db = node.params.con_out
        spectral_info.apply_attenuation_db(attenuation_out_db)
        pout = watt2dbm(sum(spectral_info.signal))
        estimated_loss = pin - pout
        estimated_gain = node.loss - estimated_loss
        node.estimated_gain = estimated_gain
        SimParams.set_params(save_sim_params)
        return round(estimated_gain, 2)
    return 0.0


def span_loss(network: DiGraph, node: Union[elements.Fiber, elements.Fused], equipment: Dict,
              input_power: Union[float, None] = None) -> float:
    """Calculates the total loss of a span (Fiber and Fused nodes) containing the specified node.

    This function computes the total loss by considering the losses of the node itself,
    its predecessors, and successors in the network. It also accounts for any Raman gain
    that may be present in the span.

    :param network: The directed graph representing the network topology.
    :type network: DiGraph
    :param node: The node for which the span loss is being calculated.
    :type node: Union[elements.Fiber, elements.Fused]
    :param equipment: A dictionary containing equipment parameters and configurations.
    :type equipment: dict
    :param input_power: The input power in dBm, used for estimating Raman gain (optional).
    :type input_power: float, optional

    :return: The total span loss in dB.
    :rtype: float

    :raises ValueError: If the node is not properly configured or if required parameters are missing.

    Notes:
    ------
    - The function caches the computed span loss in the `design_span_loss` attribute of the node
      to avoid redundant calculations.
    - If the node is passive, its loss is considered; otherwise, it is treated as zero.
    """
    if hasattr(node, "design_span_loss"):
        return node.design_span_loss
    loss = node.loss if node.passive else 0
    loss += sum(n.loss for n in prev_node_generator(network, node))
    loss += sum(n.loss for n in next_node_generator(network, node))
    # add the possible Raman gain
    gain = estimate_raman_gain(node, equipment, input_power)
    gain += sum(estimate_raman_gain(n, equipment, input_power) for n in prev_node_generator(network, node))
    gain += sum(estimate_raman_gain(n, equipment, input_power) for n in next_node_generator(network, node))
    return loss - gain


def estimate_srs_power_deviation(network: DiGraph, last_node, equipment: Dict, design_bands: Dict, input_powers: Dict) \
        -> List[Dict]:
    """Estimate tilt of power accross the design bands.

    If Raman flag is on (sim-params), this function estimates the center frequency power and the
    power tilt within each design band using the stimulated Raman scattering loss profile.
    Note that this computation may be time-consuming.

    :param network: The network object representing the topology.
    :type network: DiGraph
    :param last_node: The last node (Fiber or RamanFiber) of the considered span, which may consist of
                      a succession of fiber and fused elements.
    :type last_node: elements
    :param equipment: A dictionary containing equipment parameters.
    :type equipment: dict
    :param design_bands: A dictionary of design bands with frequency ranges.
    :type design_bands: dict
    :param input_powers: A dictionary of input powers in the fiber span for each design band.
    :type input_powers: dict

    :return: A list of dictionaries containing the power at band centers and the tilt within each band.
    :rtype: List[dict]

    :raises ValueError: If an unexpected type is encountered for nodes during processing.

    Notes:
    ------
    - The function collects preceding nodes (Fiber and Fused) and computes the stimulated Raman scattering
      for each fiber span to estimate the power deviation.
    - The results include center frequency powers normalized to the input powers and the in-band power deviation in dB.
    """
    # Get reference channel parameters
    roll_off = equipment['SI']['default'].roll_off
    baud_rate = equipment['SI']['default'].baud_rate
    spacing = equipment['SI']['default'].spacing
    tx_osnr = equipment['SI']['default'].tx_osnr

    # Create input spectral information for the first design band
    band_name0 = list(design_bands.keys())[0]
    band0 = design_bands[band_name0]
    spectral_information = \
        create_input_spectral_information(f_min=band0['f_min'], f_max=band0['f_max'], roll_off=roll_off,
                                          baud_rate=baud_rate, spacing=spacing,
                                          tx_osnr=tx_osnr, tx_power=input_powers[band_name0])

    # Create input spectral information for the remaining design bands
    for band_name, band in list(design_bands.items())[1:]:
        spectral_information = spectral_information + \
            create_input_spectral_information(f_min=band['f_min'], f_max=band['f_max'], roll_off=roll_off,
                                              baud_rate=baud_rate, spacing=spacing,
                                              tx_osnr=tx_osnr, tx_power=input_powers[band_name])

    # collect preceding nodes Fiber and Fused
    prev_nodes = [n for n in prev_node_generator(network, last_node)]
    prev_nodes.append(last_node)

    for elem in prev_nodes:
        # compute SRS tilt
        if isinstance(elem, elements.Fiber):
            # computes the power profile and resulting srs_power_deviation after each fiber span
            srs = RamanSolver.calculate_stimulated_raman_scattering(spectral_information, fiber=elem)
            # records per band
            srs_power_deviation = []
            center_frequency_powers = []
            for band_name, band in design_bands.items():
                # find center frequency power
                center_frequency = (band['f_max'] + band['f_min']) / 2
                center_frequency_index = abs(srs.frequency - center_frequency).argmin()
                center_frequency_power = srs.power_profile[center_frequency_index][-1]
                center_frequency_powers.append(center_frequency_power / input_powers[band_name])
                index_f_min = abs(srs.frequency - band['f_min']).argmin()
                index_f_max = abs(srs.frequency - band['f_max']).argmin()
                srs_power_deviation.append({'center_frequency_power': center_frequency_power / input_powers[band_name],
                                            'in_band_power_deviation_db': watt2dbm(srs.power_profile[index_f_min][-1])
                                            - watt2dbm(srs.power_profile[index_f_max][-1])})
            # apply the attenuation due to the fiber losses
            # apply attenuation for possible next fiber in the list
            # (computes the srs_power_deviation for the whole list)
            attenuation_fiber = srs.loss_profile[:spectral_information.number_of_channels, -1]
            spectral_information.apply_attenuation_lin(attenuation_fiber)
        elif isinstance(elem, elements.Fused):
            spectral_information.apply_attenuation_db(elem.loss)
        else:
            # to be removed when patch is finished
            raise ValueError('unexpected type. supported types for srs_power_deviation estimation are Fiber and Fused')
    return srs_power_deviation


def compute_band_power_deviation_and_tilt(srs_power_deviation: List[Dict], design_bands: Dict, ratio: float = 0.8):
    """Computes the power difference between bands at center frequency and the power tilt within each band.

    This function calculates the relative power deviation with respect to the maximum value for each band
    and determines the tilt target to apply for each band based on the stimulated Raman scattering (SRS) power
    deviation.

    :param srs_power_deviation: A list of dictionaries containing the power at band centers and the tilt within
                                each band.
    :type srs_power_deviation: List[dict]
    :param design_bands: A dictionary of design bands with frequency ranges.
    :type design_bands: dict
    :param ratio: The ratio applied to compute the band tilt (default is 0.8).
    :type ratio: float

    :return: A tuple containing:
             - A dictionary with the relative power deviation in dB for each band.
             - A dictionary with the tilt target to apply for each band.
    :rtype: Tuple[dict, dict]

    Notes:
    ------
    - If there is no SRS computed, the tilt estimation will be zero for all bands.
    - The function uses the maximum center frequency power to compute the deviations and tilt targets.
    """
    # if there is no SRS computed, there is no tilt, and the result should be zero for tilt estimation
    # else, let's use the power difference between bands (due to SRS) to estimate the tilt between bands,
    # and apply these values with a ratio to the next amplifier gain target, to compensate for this difference.
    deviation_db = {}
    tilt_target = {}
    max_center_frequency_powers = max([e['center_frequency_power'] for e in srs_power_deviation])
    for band_name, tilt_elem in zip(design_bands.keys(), srs_power_deviation):
        deviation_db[band_name] = \
            watt2dbm(ratio * max_center_frequency_powers) - watt2dbm(tilt_elem['center_frequency_power'])
        tilt_target[band_name] = tilt_elem['in_band_power_deviation_db']
    if allclose([t['in_band_power_deviation_db'] for t in srs_power_deviation], 0, atol=1e-9):
        for band_name in design_bands.keys():
            deviation_db[band_name] = 0.0
            tilt_target[band_name] = 0.0
    return deviation_db, tilt_target


def compute_tilt_using_previous_and_next_spans(
        prev_node: ELEMENT_TYPES, next_node: ELEMENT_TYPES,
        design_bands: Dict[str, float], input_powers: Dict[str, float], equipment: dict,
        network: DiGraph, prev_weight: float = 1.0,
        next_weight: float = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Computes the power deviation per band and the tilt target based on previous and next spans.

    This function estimates the power deviation between center frequencies due to the previous span and
    the tilt within each band using the previous and next fiber spans, applying specified weights.

    :param prev_node: The previous node in the network (Fiber or Fused).
    :type prev_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param next_node: The next node in the network (Fiber or Fused).
    :type next_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param design_bands: A dictionary of design bands for which the tilt is computed.
    :type design_bands: Dict[str, float]
    :param input_powers: A dictionary of input powers for each design band.
    :type input_powers: Dict[str, float]
    :param equipment: Equipment specifications.
    :type equipment: dict
    :param network: The network graph representing the topology.
    :type network: DiGraph
    :param prev_weight: Weight for the previous tilt in the target calculation (default is 1.0).
    :type prev_weight: float
    :param next_weight: Weight for the next tilt in the target calculation (default is 0.0).
    :type next_weight: float

    :return: A tuple containing:
             - A dictionary with the tilt estimation for each design band.
             - A dictionary with the tilt target for each design band.
    :rtype: Tuple[Dict[str, float], Dict[str, float]]

    Notes:
    ------
    - The function computes the tilt based on the estimated stimulated Raman scattering (SRS) power deviation
      from the previous and next spans.
    - If either node is not a Fiber, the corresponding tilt estimation will not be computed.
    """
    tilt_estimation = {band: 0 for band in design_bands}
    prev_tilt_target = {band: 0 for band in design_bands}
    next_tilt_target = {band: 0 for band in design_bands}
    if isinstance(prev_node, (elements.Fiber)):
        # get the estimated tilt on previous span
        srs_power_deviation = estimate_srs_power_deviation(network, prev_node, equipment, design_bands=design_bands,
                                                           input_powers=input_powers)
        tilt_estimation, prev_tilt_target = compute_band_power_deviation_and_tilt(srs_power_deviation,
                                                                                  design_bands=design_bands, ratio=0.86)
    if isinstance(next_node, (elements.Fiber)):
        # get estimated tilt on next span
        # use the same input powers (approximation!) since current amp dp and voa have not yet been computed
        srs_power_deviation = estimate_srs_power_deviation(network, find_last_node(network, next_node), equipment,
                                                           design_bands=design_bands, input_powers=input_powers)
        _, next_tilt_target = compute_band_power_deviation_and_tilt(srs_power_deviation, design_bands=design_bands,
                                                                    ratio=0.86)
    tilt_target = {band_name: prev_weight * prev_t + next_weight * next_tilt_target[band_name]
                   for band_name, prev_t in prev_tilt_target.items()}
    return tilt_estimation, tilt_target


def find_first_node(network, node):
    """Finds the first node at the origin of a succession of fused nodes.

    This function traverses the network in reverse to locate the first node
    in a sequence of fused nodes, ensuring that there are no amplifiers in between.

    :param network: The network object representing the topology.
    :type network: Any
    :param node: The starting node from which to find the first fused node.
    :type node: elements

    :return: The first node in the succession of fused nodes.
    :rtype: elements
    """
    this_node = node
    for this_node in prev_node_generator(network, node):
        pass
    return this_node


def find_last_node(network, node):
    """Finds the last node in a succession of fused nodes.

    This function traverses the network to locate the last node in a sequence
    of fused nodes, ensuring that there are no amplifiers in between.

    :param network: The network object representing the topology.
    :type network: Any
    :param node: The starting node from which to find the last fused node.
    :type node: elements

    :return: The last node in the succession of fused nodes.
    :rtype: elements
    """
    this_node = node
    for this_node in next_node_generator(network, node):
        pass
    return this_node


def set_amplifier_voa(amp: elements.Edfa, power_target: float, power_mode: bool):
    """Sets the output variable optical attenuator (VOA) for the amplifier.

    This function adjusts the VOA based on the target power and the operating mode
    of the amplifier. It ensures that the VOA is not maximized beyond a certain margin.

    :param amp: The amplifier object for which the VOA is being set.
    :type amp: elements.Edfa
    :param power_target: The target output power level for the amplifier.
    :type power_target: float
    :param power_mode: The mode for the design (power_mode True means use delta_p targets, while
                       power_mode False means use gain targets).
    :type power_mode: bool

    :return: None

    Notes:
    ------
    - If the amplifier's output VOA is set to None and the power mode is active,
      the function calculates the appropriate VOA value based on the amplifier's
      parameters and the target power.
    - If power_mode is False, the output VOA optimization is not applied.
    """
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
    if amp.in_voa is None:
        amp.in_voa = 0


def get_oms_edge_list(oms_ingress_node: Union[elements.Roadm, elements.Transceiver], network: DiGraph) \
        -> List[Tuple]:
    """get the list of OMS edges (node, neighbour next node) starting from its ingress down to its egress
    oms_ingress_node can be a ROADM or a Transceiver
    """
    oms_edges = []
    node = oms_ingress_node
    visited_nodes = []
    # collect the OMS element list (ROADM to ROADM, or Transceiver to ROADM)
    while not (isinstance(node, elements.Roadm) or isinstance(node, elements.Transceiver)):
        next_node = get_next_node(node, network)
        visited_nodes.append(node.uid)
        if next_node.uid in visited_nodes:
            raise NetworkTopologyError(f'Loop detected for {type(node).__name__} {node.uid}, '
                                       + 'please check network topology')
        oms_edges.append((node, next_node))
        node = next_node

    return oms_edges


def get_oms_edge_list_from_egress(oms_egress_node, network: DiGraph) -> List[Tuple]:
    """get the list of OMS edges (node, neighbour next node) starting from its ingress down to its egress
    oms_ingress_node can be a ROADM or a Transceiver
    """
    oms_edges = []
    node = oms_egress_node
    visited_nodes = []
    # collect the OMS element list (ROADM to ROADM, or Transceiver to ROADM)
    while not (isinstance(node, elements.Roadm) or isinstance(node, elements.Transceiver)):
        previous_node = get_previous_node(node, network)
        visited_nodes.append(node.uid)
        if previous_node.uid in visited_nodes:
            raise NetworkTopologyError(f'Loop detected for {type(node).__name__} {node.uid}, '
                                       + 'please check network topology')
        oms_edges.append((node, previous_node))
        node = previous_node

    return oms_edges


def check_oms_single_type(oms_edges: List[Tuple]) -> List[str]:
    """Verifies that the OMS only contains all single band amplifiers or all multi band amplifiers
    No mixed OMS are permitted for the time being.
    returns the amplifiers'type of the OMS
    """
    oms_types = {}
    for node, _ in oms_edges:
        if isinstance(node, elements.Edfa):
            oms_types[node.uid] = 'Edfa'
        elif isinstance(node, elements.Multiband_amplifier):
            oms_types[node.uid] = 'Multiband_amplifier'
    # checks that the element in the OMS are consistant (no multi-band mixed with single band)
    types = set(list(oms_types.values()))
    if len(types) > 1:
        msg = 'type_variety Multiband ("Multiband_amplifier") and single band ("Edfa") cannot be mixed;\n' \
            + f'Multiband amps: {[e for e in oms_types.keys() if oms_types[e] == "Multiband_amplifier"]}\n' \
            + f'single band amps: {[e for e in oms_types.keys() if oms_types[e] == "Edfa"]}'
        raise NetworkTopologyError(msg)
    return list(types)


def compute_gain_power_and_tilt_target(node: elements.Edfa, prev_node: ELEMENT_TYPES, next_node: ELEMENT_TYPES,
                                       power_mode: bool, prev_voa: float,
                                       prev_dp: float, pref_total_db: float, network: DiGraph, equipment: dict,
                                       deviation_db: float, tilt_target: float) \
        -> Tuple[float, float, float, float, float]:
    """Computes the gain and power targets for a given EDFA node.

    This function calculates the gain target, power target, delta power, output variable optical attenuation (VOA),
    and span loss for the specified EDFA node based on the network configuration and previous amplifier settings.

    :param node: The current EDFA node for which the targets are being computed.
    :type node: elements.Edfa
    :param prev_node: The previous node in the network.
    :type prev_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param next_node: The next node in the network.
    :type next_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param power_mode: Indicates if the computation is in power mode.
    :type power_mode: bool
    :param prev_voa: The previous amplifier variable optical attenuation.
    :type prev_voa: float
    :param prev_dp: The previous amplifier delta power.
    :type prev_dp: float
    :param pref_total_db: The reference total power in dBm.
    :type pref_total_db: float
    :param network: The network graph representing the topology.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications.
    :type equipment: dict
    :param deviation_db: Power deviation due to band tilt during propagation before crossing this node.
    :type deviation_db: float
    :param tilt_target: Tilt target to be configured on this amplifier for its amplification band.
    :type tilt_target: float

    :return: A tuple containing:
             - gain_target (float): The computed gain target.
             - power_target (float): The computed power target.
             - tilt_target (float): The tilt target to be configured on this amplifier.
             - dp (float): The computed delta power.
             - voa (float): The output variable optical attenuation.
             - node_loss (float): The span loss from the previous amplifier.
    :rtype: Tuple[float, float, float, float, float]

    Notes:
    ------
    - The function considers both power mode and gain mode when calculating the gain target and delta power.
    - If the operational parameters are not set, default values are used for calculations.
    """
    node_loss = span_loss(network, prev_node, equipment)
    voa = node.out_voa if node.out_voa else 0
    in_voa = node.in_voa if node.in_voa else 0
    if node.operational.delta_p is None:
        dp = target_power(network, next_node, equipment, deviation_db) + voa
    else:
        dp = node.operational.delta_p
    if node.effective_gain is None or power_mode:
        gain_target = node_loss + deviation_db + dp - prev_dp + prev_voa + in_voa
    else:  # gain mode with effective_gain
        gain_target = node.effective_gain
        dp = prev_dp - (node_loss + deviation_db) - prev_voa + gain_target - in_voa

    if node.operational.tilt_target is None:
        _tilt_target = -tilt_target
    else:
        _tilt_target = node.operational.tilt_target
    power_target = pref_total_db + dp

    return gain_target, power_target, _tilt_target, dp, voa, node_loss


def filter_edfa_list_based_on_targets(uid: str, edfa_eqpt: dict, power_target: float, gain_target: float,
                                      tilt_target: float, target_extended_gain: float,
                                      raman_allowed: bool = True, verbose: bool = False):
    """Filters amplifiers based on power, gain, and tilt targets.

    This function evaluates a list of EDFA amplifiers and filters them according to specified power,
    gain, and tilt targets. It considers both standard and Raman amplifiers based on the provided parameters.

    :param uid: The unique identifier for the node being processed.
    :type uid: str
    :param edfa_eqpt: A dictionary containing the amplifiers' equipment specifications.
    :type edfa_eqpt: dict
    :param power_target: The target power level for the amplifiers.
    :type power_target: float
    :param gain_target: The target gain level for the amplifiers.
    :type gain_target: float
    :param tilt_target: The target tilt level for the amplifiers.
    :type tilt_target: float
    :param target_extended_gain: The extended gain target for the amplifiers.
    :type target_extended_gain: float
    :param raman_allowed: Flag indicating whether to include Raman amplifiers in the selection (default is True).
    :type raman_allowed: bool
    :param verbose: Flag for enabling verbose logging (default is False).
    :type verbose: bool

    :return: A list of amplifiers that satisfy the specified power, gain, and tilt targets.
    :rtype: list

    :raises ConfigurationError: If no amplifiers meet the minimum gain requirement.

    Notes:
    ------
    - The function creates two lists of amplifiers: one for standard EDFAs and another for Raman amplifiers,
      applying different criteria for each.
    - If no amplifiers meet the minimum gain requirement, a warning is issued, and the function raises an error.
    """
    Edfa_list = namedtuple('Edfa_list', 'variety power gain_min nf f_min f_max')
    edfa_dict = {name: amp for (name, amp) in edfa_eqpt.items()}

    pin = power_target - gain_target

    # create 2 list of available amplifiers with relevant attributes for their selection

    # edfa list with:
    # extended gain min allowance of 3dB: could be parametrized, but a bit complex
    # extended gain max allowance target_extended_gain is coming from eqpt_config.json
    # power attribut include power AND gain limitations
    edfa_list = [Edfa_list(
        variety=edfa_variety,
        power=min(pin + edfa.gain_flatmax + target_extended_gain, edfa.p_max) - power_target,
        gain_min=gain_target + 3 - edfa.gain_min,
        nf=edfa_nf(gain_target, edfa_eqpt[edfa_variety]),
        f_min=edfa.f_min,
        f_max=edfa.f_max)
        for edfa_variety, edfa in edfa_dict.items()
        if not edfa.raman]

    # consider a Raman list because of different gain_min requirement:
    # do not allow extended gain min for Raman
    raman_list = [Edfa_list(
        variety=edfa_variety,
        power=min(pin + edfa.gain_flatmax + target_extended_gain, edfa.p_max) - power_target,
        gain_min=gain_target - edfa.gain_min,
        nf=edfa_nf(gain_target, edfa_eqpt[edfa_variety]),
        f_min=edfa.f_min,
        f_max=edfa.f_max)
        for edfa_variety, edfa in edfa_dict.items()
        if edfa.raman] \
        if raman_allowed else []

    # merge raman and edfa lists
    amp_list = edfa_list + raman_list

    # Filter on min gain limitation:
    acceptable_gain_min_list = [x for x in amp_list if x.gain_min > 0]

    if len(acceptable_gain_min_list) < 1:
        # do not take this empty list into account for the rest of the code
        # but issue a warning to the user and do not consider Raman
        # Raman below min gain should not be allowed because i is meant to be a design requirement
        # and raman padding at the amplifier input is impossible!

        if len(edfa_list) < 1:
            raise ConfigurationError('auto_design could not find any amplifier \
                    to satisfy min gain requirement \
                    please increase span fiber padding')
        else:
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
        # No amplifier satisfies the required power, so pick the highest power(s):
        power_max = max(acceptable_gain_min_list, key=attrgetter('power')).power
        # Check and pick if other amplifiers may have a similar gain/power
        # Allow a 0.3dB power range
        # This allows to chose an amplifier with a better NF subsequentely
        acceptable_power_list = [x for x in acceptable_gain_min_list
                                 if x.power - power_max > -0.3]
    return acceptable_power_list


def preselect_multiband_amps(uid: str, _amplifiers: dict, prev_node: ELEMENT_TYPES, next_node: ELEMENT_TYPES,
                             power_mode: bool, prev_voa: dict, prev_dp: dict,
                             pref_total_db: dict, network: DiGraph, equipment: dict, restrictions: List,
                             _design_bands: dict, deviation_db: dict, tilt_target: dict):
    """Preselects multiband amplifiers eligible based on power, gain, and tilt targets across all bands.

    This function filters multiband amplifiers according to specified constraints, ensuring that they meet
    the power, gain, and tilt targets for each design band.

    :param uid: The unique identifier for the node being processed.
    :type uid: str
    :param _amplifiers: A dictionary containing the amplifiers of the multiband amplifier.
    :type _amplifiers: dict
    :param prev_node: The previous node in the network.
    :type prev_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param next_node: The next node in the network.
    :type next_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param power_mode: Indicates if the computation is in power mode.
    :type power_mode: bool
    :param prev_voa: A dictionary containing the previous amplifier output VOA settings per band.
    :type prev_voa: dict
    :param prev_dp: A dictionary containing the previous amplifier delta power settings per band.
    :type prev_dp: dict
    :param pref_total_db: The total power used for design in dBm.
    :type pref_total_db: float
    :param network: The network graph representing the topology.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications.
    :type equipment: dict
    :param restrictions: A list of restrictions related to amplifier selection.
    :type restrictions: List
    :param _design_bands: A dictionary of design bands with frequency ranges.
    :type _design_bands: dict
    :param deviation_db: A dictionary of tilt power per band.
    :type deviation_db: dict
    :param tilt_target: A dictionary of tilt targets for each band.
    :type tilt_target: dict

    :return: A list of preselected multiband amplifiers that are eligible for all the bands.
    :rtype: list

    Notes:
    ------
    - The function iteratively filters amplifiers based on the constraints and updates the selection
      based on the results from each band.
    - If no amplifiers meet the criteria, an empty list is returned.
    """
    # Initialize the list for the loop
    target_extended_gain = equipment['Span']['default'].target_extended_gain
    _selected_type_varieties = [n for n in restrictions]
    for band, amp in _amplifiers.items():
        # In the loop, keep only the set of amps that match the constraints on all the bands
        # Use the subset of EDFA library that are multiband and fits the restriction
        # filter those amps that match this amp band and that are within the selection made during the loop
        edfa_eqpt = {t: equipment['Edfa'][t]
                     for m in _selected_type_varieties for t in equipment['Edfa'][m].multi_band
                     if equipment['Edfa'][t].f_min <= _design_bands[band]['f_min']
                     and equipment['Edfa'][t].f_max >= _design_bands[band]['f_max']}
        # get the target gain, power and tilt based on previous propagation
        gain_target, power_target, _tilt_target, _, _, _ = \
            compute_gain_power_and_tilt_target(amp, prev_node, next_node, power_mode, prev_voa[band], prev_dp[band],
                                               pref_total_db[band], network, equipment, deviation_db[band], tilt_target[band])
        _selection = [a.variety
                      for a in filter_edfa_list_based_on_targets(uid, edfa_eqpt, power_target, gain_target,
                                                                 _tilt_target, target_extended_gain)]
        listes = find_type_varieties(_selection, equipment)
        _selected_type_varieties = []
        if listes:
            # get the union of the lists
            _selected_type_varieties = list(reduce(lambda x, y: set(x) | set(y), listes))

    return [t for m in _selected_type_varieties for t in equipment['Edfa'][m].multi_band]


def set_one_amplifier(node: elements.Edfa, prev_node: ELEMENT_TYPES, next_node: ELEMENT_TYPES,
                      power_mode: bool, prev_voa: float, prev_dp: float,
                      pref_ch_db: float, pref_total_db: float, network: DiGraph, restrictions: List[str],
                      equipment: dict, verbose: bool, deviation_db: float = 0.0, tilt_target: float = 0.0) \
        -> Tuple[float, float]:
    """Sets the EDFA amplifier configuration based on power targets.

    This function adjusts the amplifier settings according to specified parameters and ensures compliance
    with power and gain targets. It handles both cases where the amplifier type is specified or needs
    to be selected based on restrictions.

    :param node: The EDFA amplifier node to configure.
    :type node: elements.Edfa
    :param prev_node: The previous node in the network.
    :type prev_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param next_node: The next node in the network.
    :type next_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param power_mode: Indicates if the amplifier is in power mode.
    :type power_mode: bool
    :param prev_voa: The previous amplifier variable optical attenuator value.
    :type prev_voa: float
    :param prev_dp: The previous amplifier delta power.
    :type prev_dp: float
    :param pref_ch_db: Reference per channel power in dBm.
    :type pref_ch_db: float
    :param pref_total_db: Reference total power in dBm.
    :type pref_total_db: float
    :param network: The network graph representing the topology.
    :type network: DiGraph
    :param restrictions: A list of amplifiers authorized for this configuration.
    :type restrictions: List[str]
    :param equipment: Equipment library containing amplifier specifications.
    :type equipment: dict
    :param verbose: Flag for verbose logging.
    :type verbose: bool
    :param deviation_db: Tilt power per band (default is 0.0).
    :type deviation_db: float
    :param tilt_target: Tilt target in each band (default is 0.0).
    :type tilt_target: float

    :return: A tuple containing the updated delta power and variable optical attenuator values.
    :rtype: Tuple[float, float]

    Notes:
    ------
    - The function computes the gain and power targets based on previous amplifier settings and the network state.
    - It selects the appropriate amplifier type based on the specified restrictions and updates the amplifier
      parameters accordingly.
    - Warnings are logged if the effective gain exceeds the specified amplifier limits.
    """
    gain_target, power_target, _tilt_target, dp, voa, node_loss = \
        compute_gain_power_and_tilt_target(node, prev_node, next_node, power_mode, prev_voa, prev_dp,
                                           pref_total_db, network, equipment, deviation_db, tilt_target)
    if isinstance(prev_node, elements.Fiber):
        max_fiber_lineic_loss_for_raman = \
            equipment['Span']['default'].max_fiber_lineic_loss_for_raman * 1e-3  # dB/m
        raman_allowed = (prev_node.params.loss_coef < max_fiber_lineic_loss_for_raman).all()
    else:
        raman_allowed = False

    if node.params.type_variety == '':
        edfa_eqpt = {n: a for n, a in equipment['Edfa'].items() if a.type_def != 'multi_band'}
        if restrictions:
            edfa_eqpt = {n: a for n, a in edfa_eqpt.items() if n in restrictions}
        edfa_variety, power_reduction = \
            select_edfa(raman_allowed, gain_target, power_target, edfa_eqpt,
                        node.uid,
                        target_extended_gain=equipment['Span']['default'].target_extended_gain,
                        verbose=verbose)
        extra_params = equipment['Edfa'][edfa_variety]
        node.params.update_params(extra_params.__dict__)
        node.type_variety = node.params.type_variety
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
            if isinstance(prev_node, elements.Fiber) and verbose:
                logger.warning(f'\n\tWARNING: raman is used in node {node.uid}\n '
                               + '\tbut fiber lineic loss is above threshold\n')
            else:
                logger.warning(f'\n\tWARNING: raman is used in node {node.uid}\n '
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
        if gain_target - equipment['Edfa'][node.params.type_variety].gain_min < 0 and verbose:
            logger.warning(f'\n\tWARNING: effective gain in Node {node.uid}\n'
                           + f'\tis below user specified amplifier {node.params.type_variety}\n'
                           + '\tmin gain: '
                           + f'{equipment["Edfa"][node.params.type_variety].gain_min}dB ; '
                           + f'required gain: {round(gain_target, 2)}dB. Please check amplifier type.\n')
    node.delta_p = dp if power_mode else None
    node.effective_gain = gain_target
    node.tilt_target = _tilt_target
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
    return dp, voa


def get_node_restrictions(node: Union[elements.Edfa, elements.Multiband_amplifier],
                          prev_node: ELEMENT_TYPES, next_node: ELEMENT_TYPES,
                          equipment: dict, _design_bands: dict) -> List:
    """Returns a list of eligible amplifiers that comply with restrictions and design bands.

    If the node is a multiband amplifier, only multiband amplifiers will be considered.

    :param node: The current amplifier node, which can be either an EDFA or a multiband amplifier.
    :type node: Union[elements.Edfa, elements.Multiband_amplifier]
    :param prev_node: The previous node in the network.
    :type prev_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param next_node: The next node in the network.
    :type next_node: Union[elements.Fiber, elements.Roadm, elements.Fused, elements.Edfa,
                           elements.Transceiver, elements.Transceiver]
    :param equipment: A dictionary containing equipment specifications.
    :type equipment: dict
    :param _design_bands: A dictionary of design bands with frequency limits.
    :type _design_bands: dict

    :return: A list of eligible amplifier types that meet the specified restrictions.
    :rtype: List[str]

    Notes:
    ------
    - If the amplifier node has a specified type variety, that variety takes precedence over other restrictions.
    - The function checks for restrictions based on the previous and next nodes if applicable.
    - For multiband amplifiers, it filters the available amplifiers to ensure they are eligible for all design bands.
    """
    if node.params.type_variety != '' and node.params.type_variety:
        # type_variety takes precedence over any other restrictions
        return [node.params.type_variety]
    restrictions = []
    if node.variety_list and isinstance(node.variety_list, list):
        restrictions = node.variety_list
    elif isinstance(prev_node, elements.Roadm) and prev_node.restrictions['booster_variety_list']:
        # implementation of restrictions on roadm boosters
        restrictions = prev_node.restrictions['booster_variety_list']
    elif isinstance(next_node, elements.Roadm) and next_node.restrictions['preamp_variety_list']:
        # implementation of restrictions on roadm preamp
        restrictions = next_node.restrictions['preamp_variety_list']
    if isinstance(node, elements.Multiband_amplifier):
        # Only keep multiband amps that are eligible for all the bands
        # use the subset of EDFA library that are multiband, fits the design band and are either imposed
        # in restriction list or allowed for design
        multiband_eqpt = [n for n, a in equipment['Edfa'].items()
                          if a.type_def == 'multi_band'
                          and (n in restrictions or (not restrictions and a.allowed_for_design))]
        # collect the individual amps part of the multiband amps that match the bands
        edfa_eqpt = [t for m in multiband_eqpt
                     for t in equipment['Edfa'][m].multi_band
                     for band in _design_bands.values()
                     if equipment['Edfa'][t].f_min <= band['f_min']
                     and equipment['Edfa'][t].f_max >= band['f_max']]
        # then filter all multi band amps whose amps group belong to the previous list
        multiband_eqpt = [m for m in multiband_eqpt if all(t in edfa_eqpt for t in equipment['Edfa'][m].multi_band)]
        # and returns the list of type_variety of multiband amps built with this single band amps
        return multiband_eqpt
    if isinstance(node, elements.Edfa):
        band = next(b for b in _design_bands.values())
        # preselect amps which are either part of restrictions or allowed for design, and compliant to the band.
        edfa_eqpt = [n for n, a in equipment['Edfa'].items()
                     if (a.type_def != 'multi_band' and a.f_min <= band['f_min'] and a.f_max >= band['f_max'])
                     and (n in restrictions or (not restrictions and a.allowed_for_design))]
        return edfa_eqpt


def set_egress_amplifier(network: DiGraph, this_node: Union[elements.Roadm, elements.Transceiver], equipment: dict,
                         pref_ch_db: float, verbose: bool, reference_channel):
    """Configures the egress amplifiers for a given node in the network.

    Go through each link starting from this_node until next Roadm or Transceiver and
    set the amplifiers (Edfa and multiband) according to configurations set by user.
    Computes the gain for Raman finers and records it as the gain for reference design.
    
    - power_mode = True, set amplifiers delta_p and effective_gain
    - power_mode = False, set amplifiers effective_gain and ignore delta_p config: set it to None.
    
    Records the computed dp in an internal variable for autodesign purpose.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param this_node: The starting node for OMS link configuration, which can be a ROADM or Transceiver.
    :type this_node: Union[elements.Roadm, elements.Transceiver]
    :param equipment: Equipment specifications.
    :type equipment: dict
    :param pref_ch_db: Reference channel power in dBm.
    :type pref_ch_db: float
    :param pref_total_db: Reference total power in dBm.
    :type pref_total_db: float
    :param verbose: Flag for verbose logging.
    :type verbose: bool

    :raises NetworkTopologyError: If a loop is detected in the network topology.
    :raises ConfigurationError: If no amplifiers are found that match the design bands and restrictions.

    Notes:
    ------
    - The function initializes power and VOA settings for each design band and updates them as it traverses
      through the network nodes.
    - It handles different types of nodes (EDFA, RamanFiber, Multiband_amplifier) and applies appropriate
      configurations based on their characteristics.
    """
    power_mode = equipment['Span']['default'].power_mode
    next_oms = (n for n in network.successors(this_node) if not isinstance(n, elements.Transceiver))
    for oms in next_oms:
        _design_bands = {find_band_name(FrequencyBand(f_min=e["f_min"], f_max=e["f_max"])): e
                         for e in this_node.per_degree_design_bands[oms.uid]}
        oms_nodes = get_oms_edge_list(oms, network)
        # go through all the OMS departing from the ROADM
        prev_node = this_node
        node = oms
        # initialize dp and prev_dp with roadm out target or transceiver power. Use design bands.
        dp = {}
        prev_dp = {}
        voa = {}
        prev_voa = {}
        pref_total_db = {}
        for band_name, band in _design_bands.items():
            this_node_out_power = None
            if isinstance(this_node, elements.Transceiver):
                if equipment['SI']['default'].tx_power_dbm is not None:
                    this_node_out_power = equipment['SI']['default'].tx_power_dbm
                else:
                    this_node_out_power = pref_ch_db
            if isinstance(this_node, elements.Roadm):
                # get target power out from ROADM for the reference carrier based on equalization settings
                this_node_out_power = this_node.get_per_degree_ref_power(degree=node.uid)
            # use the target power on this degree
            prev_dp[band_name] = this_node_out_power - pref_ch_db
            dp[band_name] = prev_dp[band_name]
            prev_voa[band_name] = 0
            voa[band_name] = 0
            nb_channels_per_band = reference_channel.nb_channel if reference_channel.nb_channel \
                else automatic_nch(band['f_min'], band['f_max'], band['spacing'])
            pref_total_db[band_name] = pref_ch_db + lin2db(nb_channels_per_band)

        for node, next_node in oms_nodes:
            # go through all nodes in the OMS
            input_powers = {band_name: dbm2watt(pref_ch_db + prev_dp[band_name] - prev_voa[band_name])
                            for band_name in _design_bands}
            deviation_db, tilt_target = \
                compute_tilt_using_previous_and_next_spans(prev_node, next_node, _design_bands, input_powers,
                                                           equipment, network)
            if isinstance(node, elements.Edfa):
                band_name, _ = next((n, b) for n, b in _design_bands.items())
                restrictions = get_node_restrictions(node, prev_node, next_node, equipment, _design_bands)
                if not restrictions:
                    raise ConfigurationError(f'{node.uid}: Auto_design could not find any amplifier in equipment '
                                             + f'library matching the design bands{_design_bands} '
                                             + 'and the restrictions (roadm or amplifier restictions)')
                dp[band_name], voa[band_name] = set_one_amplifier(node, prev_node, next_node, power_mode,
                                                                  prev_voa[band_name], prev_dp[band_name],
                                                                  pref_ch_db, pref_total_db[band_name],
                                                                  network, restrictions, equipment, verbose)
            elif isinstance(node, elements.RamanFiber):
                # this is to record the expected gain in Raman fiber in its .estimated_gain attribute.
                band_name, _ = next((n, b) for n, b in _design_bands.items())
                _ = span_loss(network, node, equipment, input_power=pref_ch_db + dp[band_name])
            elif isinstance(node, elements.Multiband_amplifier):
                if len(node.amplifiers) == 0:
                    # creates one amp per design band.
                    for band_name, band in _design_bands.items():
                        node.amplifiers[band_name] = elements.Edfa(params=EdfaParams.default_values, uid=node.uid)
                if node.params.type_variety:
                    restrictions_edfa = equipment['Edfa'][node.type_variety].multi_band
                else:
                    # only select amplifiers which match the design bands
                    restrictions_multi = get_node_restrictions(node, prev_node, next_node, equipment, _design_bands)
                    restrictions_edfa = \
                        preselect_multiband_amps(node.uid, node.amplifiers, prev_node, next_node, power_mode,
                                                 prev_voa, prev_dp, pref_total_db,
                                                 network, equipment, restrictions_multi, _design_bands,
                                                 deviation_db=deviation_db, tilt_target=tilt_target)
                for band_name, amp in node.amplifiers.items():
                    _restrictions = [n for n in restrictions_edfa
                                     if equipment['Edfa'][n].f_min <= _design_bands[band_name]['f_min']
                                     and equipment['Edfa'][n].f_max >= _design_bands[band_name]['f_max']]
                    dp[band_name], voa[band_name] = \
                        set_one_amplifier(amp, prev_node, next_node, power_mode,
                                          prev_voa[band_name], prev_dp[band_name], pref_ch_db, pref_total_db[band_name],
                                          network, _restrictions, equipment, verbose,
                                          deviation_db=deviation_db[band_name], tilt_target=tilt_target[band_name])
                amps_type_varieties = [a.type_variety for a in node.amplifiers.values()]
                amps_bands = [a.params.bands[0] for a in node.amplifiers.values()]
                try:
                    node.type_variety = find_type_variety(amps_type_varieties, equipment)[0]
                    # overwrite type_variety and bands to be consistent with selected type_variety
                    node.params.type_variety = node.type_variety
                    node.params.bands = amps_bands
                except ConfigurationError as e:
                    # should never come here... only for debugging
                    msg = f'In {node.uid}: {e}'
                    raise ConfigurationError(msg)

            prev_dp.update(**dp)
            prev_voa.update(**voa)
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


def set_per_degree_design_band(node: Union[elements.Roadm, elements.Transceiver], network: DiGraph, equipment: dict):
    """Configures the design bands for each degree of a node based on network and equipment constraints.
    This function determines the design bands for each degree of a node (either a ROADM or a Transceiver)
    based on the existing amplifier types and spectral information (SI) constraints. It uses a default
    design band derived from the SI or ROADM bands if no specific bands are defined by the user.
    node.params.x contains the values initially defined by user (with x in design_bands,
    per_degree_design_bands). node.x contains the autodesign values.

    :param node: The node for which design bands are being set (ROADM or Transceiver).
    :type node: Union[elements.Roadm, elements.Transceiver]
    :param network: The network containing the node and its connections.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment data, including spectral information.
    :type equipment: dict

    :raises NetworkTopologyError: If there is an inconsistency in band definitions or unsupported configurations.

    Notes:
        - The function prioritizes user-defined bands in `node.params` if available.
        - It checks for consistency between default bands and amplifier types.
        - Mixed single-band and multi-band configurations are not supported and will raise an error.
        - The function ensures that all bands are ordered by their minimum frequency.
    """
    next_oms = (n for n in network.successors(node))
    if len(node.design_bands) == 0:
        node.design_bands = [{'f_min': si.f_min, 'f_max': si.f_max, 'spacing': si.spacing}
                             for si in equipment['SI'].values()]
    # complete node.per_degree_design_bands with node.design_bands spacing when it is missing.
    # Use the spacing from SI if nothing is specified.
    update_design_bands_spacing(node, equipment['SI']['default'].spacing)
    default_is_single_band = len(node.design_bands) == 1
    for next_node in next_oms:
        # get all the elements from the OMS and retrieve their amps types and bands
        oms_edges = get_oms_edge_list(next_node, network)
        amps_type = check_oms_single_type(oms_edges)
        oms_is_single_band = "Edfa" in amps_type if len(amps_type) == 1 else None
        # oms_is_single_band can be True (single band OMS), False (Multiband OMS) or None (undefined: waiting for
        # autodesign).
        el_list = [n for n, _ in oms_edges]
        amp_bands = [n.params.bands for n in el_list if isinstance(n, (elements.Edfa, elements.Multiband_amplifier))
                     and n.params.bands]
        # Use node.design_bands constraints if they are consistent with the amps type
        if oms_is_single_band == default_is_single_band:
            amp_bands.append(node.design_bands)

        common_range = find_common_range(amp_bands, None, None, equipment['SI']['default'].spacing, node.design_bands)
        # node.per_degree_design_bands has already been populated with node.params.per_degree_design_bands loaded
        # from the json.
        # let's complete the dict with the design band of degrees for which there was no definition
        if next_node.uid not in node.per_degree_design_bands:
            if common_range:
                # if degree design band was not defined, then use the common_range computed with the oms amplifiers
                # already defined.
                node.per_degree_design_bands[next_node.uid] = common_range
            elif oms_is_single_band is None or (oms_is_single_band == default_is_single_band):
                # else if no amps are defined (no bands) then use default ROADM bands
                # use default ROADM bands only if this is consistent with the oms amps type
                node.per_degree_design_bands[next_node.uid] = node.design_bands
            else:
                # unsupported case: single band OMS with default multiband design band
                raise NetworkTopologyError(f"in {node.uid} degree {next_node.uid}: inconsistent design multiband/"
                                           + " single band definition on a single band/ multiband OMS")
    # reorder per_degree_design_bands.
    reorder_per_degree_design_bands(node.per_degree_design_bands)
    reorder_per_degree_design_bands(node.params.per_degree_design_bands)
    # check node.params.per_degree_design_bands keys
    check_per_degree_design_bands_keys(node, network)


def update_design_bands_spacing(node: Union[elements.Roadm, elements.Transceiver],
                                default_spacing: float):
    """
    Update the spacing of design bands for a given node.

    This function iterates through the design bands associated with the node and updates
    their spacing based on the frequency range defined by 'f_min' and 'f_max'. If a specific
    spacing cannot be determined from the design bands, the default spacing is used.

    :param node: The node object which can be either a Roadm or Transceiver instance.
    :type node: Union[elements.Roadm, elements.Transceiver]
    :param default_spacing: The default spacing to use if no specific spacing can be determined.
    :type default_spacing: float
    """
    for design_bands in node.per_degree_design_bands.values():
        for design_band in design_bands:
            temp = get_spacing_from_band(node.design_bands, design_band['f_min'], design_band['f_max'])
            default_spacing = temp if temp is not None else default_spacing
            design_band['spacing'] = design_band.get('spacing', default_spacing)


def check_per_degree_design_bands_keys(node: Union[elements.Roadm, elements.Transceiver], network: DiGraph):
    """Checks that per_degree_design_bands keys are existing uid of elements in the network

    :param node: a ROADM or a Transceiver element
    :type node: Union[elements.Roadm, elements.Transceiver]
    :param network: the network containing the node and its connections
    :type network: DiGraph
    """
    if node.params.per_degree_design_bands:
        next_oms_uid = [n.uid for n in network.successors(node)]
        for degree in node.params.per_degree_design_bands.keys():
            if degree not in next_oms_uid:
                raise NetworkTopologyError(f"in {node.uid} degree {degree} does not match any degree"
                                           + f"{list(node.per_degree_design_bands.keys())}")


def set_roadm_input_powers(network: DiGraph, roadm: elements.Roadm, equipment: dict, pref_ch_db: float):
    """Sets the reference powers at the input of a ROADM for a specified reference channel based on adjacent OMS.

    This function calculates the input power for the ROADM based on the losses incurred through connected nodes.
    It assumes that there is no dependency on the path, meaning that the power output from one element may not
    be consistent with the target powers of subsequent elements. If the target powers cannot be met, a warning is
    raised.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param roadm: The ROADM node for which the input powers are being set.
    :type roadm: elements.Roadm
    :param equipment: A dictionary containing equipment specifications.
    :type equipment: dict
    :param pref_ch_db: The preferred channel power in dBm to be used as a reference.
    :type pref_ch_db: float

    :return: None

    :raises ConfigurationError: If no target power, PSD, or PSW can be found in the ROADM.

    Notes:
    ------
    - The function traverses the network to calculate the cumulative loss from the predecessor nodes.
    - It sets the reference power based on the type of the preceding node, adjusting for losses accordingly.
    - If the calculated input power is below the maximum target power, a warning is logged to inform the user.
    - This value is primarily used for visualization purposes to compute resulting ROADM loss in elements.

    Example:
    --------
    For a ROADM with the following configuration:

    ::
        ``
        node                             power out of element
        roadm A (target power -10dBm)   -10dBm
        fiber A (16 dB loss)            -26dBm
        roadm B (target power -12dBm)   -26dBm
        fiber B (10 dB loss)            -36dBm
        roadm C (target power -14dBm)   -36dBm
        ``

    is not consistent because target powers in roadm B and roadm C can not be met.
    The input power for the reference channel will be set to -26 dBm in roadm B and -22 dBm in roadm C,
    because at design time we cannot know about the path. The function raises a warning if target powers
    cannot be met with the design, indicating that the user should be aware that the design was not successful
    and that power reduction was applied.
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
        elif isinstance(node, elements.Multiband_amplifier):
            # use the worst (min) value among amps
            roadm.ref_pch_in_dbm[element.uid] = min([pref_ch_db + amp._delta_p - amp.out_voa - loss
                                                     for amp in node.amplifiers.values()])
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


def set_fiber_input_power(network: DiGraph, fiber: elements.Fiber, equipment: dict, pref_ch_db: float):
    """Sets the reference power at the input of a fiber for a specified reference channel.

    This function calculates the input power for a fiber based on the losses incurred through connected nodes,
    including ROADMs, amplifiers, and other fibers. It assumes that the target power out of ROADMs and amplifiers
    is consistent. This is primarily for visualization purposes.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param fiber: The fiber node for which the input power is being set.
    :type fiber: elements.Fiber
    :param equipment: A dictionary containing equipment specifications.
    :type equipment: dict
    :param pref_ch_db: The preferred channel power in dBm to be used as a reference.
    :type pref_ch_db: float

    :return: None

    Notes:
    ------
    - The function traverses the network to calculate the cumulative loss from the predecessor nodes.
    - It sets the reference power based on the type of the preceding node, adjusting for losses accordingly.
    - The function handles various node types, including fibers, ROADMs, EDFAs, transceivers, and multiband amplifiers.
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
    elif isinstance(node, elements.Multiband_amplifier):
        # use the worst (min) value among amps
        fiber.ref_pch_in_dbm = min([pref_ch_db + amp._delta_p - amp.out_voa - loss for amp in node.amplifiers.values()])


def set_roadm_internal_paths(roadm: elements.Roadm, network: DiGraph):
    """Sets the internal path types (express, add, drop) for a ROADM.

    This function determines the path types for the ROADM based on its connections to other nodes in the network.
    It uses implicit assumptions if no specific information is provided for the ROADM's paths.

    :param roadm: The ROADM node for which the internal paths are being set.
    :type roadm: elements.Roadm
    :param network: The network graph containing nodes and links.
    :type network: DiGraph

    :return: None

    :raises NetworkTopologyError: If the path types are inconsistent with the expected types based on the connections.

    Notes:
    ------
    - The function identifies the next and previous nodes connected to the ROADM, distinguishing between
      transceivers and other elements.
    - It sets the path types based on the connections and raises errors if any inconsistencies are found.
    - The function ensures that paths connected to transceivers are marked as 'add' or 'drop', while other
      paths can be 'express', 'add', or 'drop'.
    """
    next_oms = [n.uid for n in network.successors(roadm) if not isinstance(n, elements.Transceiver)]
    previous_oms = [n.uid for n in network.predecessors(roadm) if not isinstance(n, elements.Transceiver)]
    drop_port = [n.uid for n in network.successors(roadm) if isinstance(n, elements.Transceiver)]
    add_port = [n.uid for n in network.predecessors(roadm) if isinstance(n, elements.Transceiver)]

    default_express = 'express'
    default_add = 'add'
    default_drop = 'drop'
    # take user defined element impairment id if it exists
    correct_from_degrees = []
    correct_add = []
    correct_to_degrees = []
    correct_drop = []
    for from_degree in previous_oms:
        correct_from_degrees.append(from_degree)
        for to_degree in next_oms:
            correct_to_degrees.append(to_degree)
            impairment_id = roadm.get_per_degree_impairment_id(from_degree, to_degree)
            roadm.set_roadm_paths(from_degree=from_degree, to_degree=to_degree, path_type=default_express,
                                  impairment_id=impairment_id)
        for drop in drop_port:
            correct_drop.append(drop)
            impairment_id = roadm.get_per_degree_impairment_id(from_degree, drop)
            path_type = roadm.get_path_type_per_id(impairment_id)
            # a degree connected to a transceiver MUST be add or drop
            # but a degree connected  to something else could be an express, add or drop
            # (for example case of external shelves)
            if path_type and path_type != 'drop':
                msg = f'Roadm {roadm.uid} path_type is defined as {path_type} but it should be drop'
                raise NetworkTopologyError(msg)
            roadm.set_roadm_paths(from_degree=from_degree, to_degree=drop, path_type=default_drop,
                                  impairment_id=impairment_id)
    for to_degree in next_oms:
        for add in add_port:
            correct_add.append(add)
            impairment_id = roadm.get_per_degree_impairment_id(add, to_degree)
            path_type = roadm.get_path_type_per_id(impairment_id)
            if path_type and path_type != 'add':
                msg = f'Roadm {roadm.uid} path_type is defined as {path_type} but it should be add'
                raise NetworkTopologyError(msg)
            roadm.set_roadm_paths(from_degree=add, to_degree=to_degree, path_type=default_add,
                                  impairment_id=impairment_id)
    # sanity check: raise an error if per_degree from or to degrees are not in the correct list
    # raise an error if user defined path_type is not consistent with inferred path_type:
    for item in roadm.per_degree_impairments.values():
        if item['from_degree'] not in correct_from_degrees + correct_add or \
                item['to_degree'] not in correct_to_degrees + correct_drop:
            msg = f'Roadm {roadm.uid} has wrong from-to degree uid {item["from_degree"]} - {item["to_degree"]}'
            raise NetworkTopologyError(msg)


def add_roadm_booster(network: DiGraph, roadm: elements.Roadm):
    """Adds a booster amplifier to the specified ROADM in the network.

    This function identifies the successor nodes of the ROADM that are not transceivers, fused elements, or amplifiers.
    For each valid successor, it removes the existing edge to the ROADM and adds a booster amplifier (either a
    multiband amplifier or an EDFA) based on the type of the connected nodes and the design bands of the ROADM.
    The amplifier is positioned at the location of the ROADM.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param roadm: The ROADM node to which the booster amplifier will be added.
    :type roadm: elements.Roadm

    :return: None

    Notes:
    ------
    - The function determines the type of amplifier to add based on the connected nodes and the design bands
      of the ROADM.
    - It updates the network by adding the amplifier node and connecting it to the ROADM and the successor node.
    - The amplifier's location metadata is set to the location of the ROADM.
    """
    next_nodes = [n for n in network.successors(roadm)
                  if not isinstance(n, (elements.Transceiver, elements.Fused, elements.Edfa,
                                        elements.Multiband_amplifier))]
    # no amplification for fused spans or TRX
    for next_node in next_nodes:
        network.remove_edge(roadm, next_node)
        oms_edges = get_oms_edge_list(next_node, network)
        amps_type = check_oms_single_type(oms_edges)
        if 'Multiband_amplifier' in amps_type or ('Edfa' not in amps_type and len(roadm.design_bands) > 1):
            amp = elements.Multiband_amplifier(
                uid=f'Edfa_booster_{roadm.uid}_to_{next_node.uid}',
                params=MultiBandParams.default_values,
                metadata={
                    'location': {
                        'latitude': roadm.lat,
                        'longitude': roadm.lng,
                        'city': roadm.loc.city,
                        'region': roadm.loc.region,
                    }
                },
                amplifiers=[])
        else:
            # if 'Edfa' or no amplifier type is set in the OMS, then assumes single band
            amp = elements.Edfa(
                uid=f'Edfa_booster_{roadm.uid}_to_{next_node.uid}',
                params=EdfaParams.default_values,
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


def add_roadm_preamp(network: DiGraph, roadm: elements.Roadm):
    """Adds a preamplifier to the specified ROADM in the network.

    This function identifies the predecessor nodes of the ROADM that are not transceivers,
    fused elements, or amplifiers.
    For each valid predecessor, it removes the existing edge to the ROADM and adds a preamplifier
    (either a multiband amplifier or an EDFA) based on the type of the connected nodes. The amplifier
    is positioned at the location of the ROADM.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param roadm: The ROADM node to which the preamplifier will be added.
    :type roadm: elements.Roadm

    :return: None

    Notes:
    ------
    - The function determines the type of amplifier to add based on the connected nodes.
    - It updates the network by adding the amplifier node and connecting it to the predecessor node and the ROADM.
    - The amplifier's location metadata is set to the location of the ROADM.
    """
    prev_nodes = [n for n in network.predecessors(roadm)
                  if not isinstance(n, (elements.Transceiver, elements.Fused, elements.Edfa,
                                        elements.Multiband_amplifier))]
    # no amplification for fused spans or TRX
    for prev_node in prev_nodes:
        network.remove_edge(prev_node, roadm)
        oms_edges = get_oms_edge_list_from_egress(prev_node, network)
        amps_type = check_oms_single_type(oms_edges)
        if 'Multiband_amplifier' in amps_type:
            amp = elements.Multiband_amplifier(
                uid=f'Edfa_preamp_{roadm.uid}_from_{prev_node.uid}',
                params=MultiBandParams.default_values,
                metadata={
                    'location': {
                        'latitude': roadm.lat,
                        'longitude': roadm.lng,
                        'city': roadm.loc.city,
                        'region': roadm.loc.region,
                    }
                },
                amplifiers=[])
        else:
            amp = elements.Edfa(
                uid=f'Edfa_preamp_{roadm.uid}_from_{prev_node.uid}',
                params=EdfaParams.default_values,
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


def add_inline_amplifier(network: DiGraph, fiber: elements.Fiber):
    """Adds an inline amplifier to the network between a fiber and its next node.

    This function checks the next node connected to the specified fiber. If the next node is a fiber or a Raman
    fiber, it removes the existing edge and adds an inline amplifier (either a multiband amplifier or an EDFA)
    based on the type of the connected nodes. The amplifier is positioned at the midpoint between the fiber and
    the next node.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param fiber: The fiber node to which the inline amplifier will be added.
    :type fiber: elements.Fiber

    :return: None

    Notes:
    ------
    - The function determines the type of amplifier to add based on the connected nodes.
    - It updates the network by adding the amplifier node and connecting it to the fiber and the next node.
    - The amplifier's location metadata is calculated as the midpoint between the fiber and the next node.
    """
    next_node = get_next_node(fiber, network)
    if isinstance(next_node, elements.Fiber) or isinstance(next_node, elements.RamanFiber):
        # no amplification for fused spans or TRX
        network.remove_edge(fiber, next_node)
        oms_edges = get_oms_edge_list(next_node, network)
        amps_type = check_oms_single_type(oms_edges)
        if 'Multiband_amplifier' in amps_type:
            amp = elements.Multiband_amplifier(
                uid=f'Edfa_{fiber.uid}',
                params=MultiBandParams.default_values,
                metadata={
                    'location': {
                        'latitude': (fiber.lat + next_node.lat) / 2,
                        'longitude': (fiber.lng + next_node.lng) / 2,
                        'city': fiber.loc.city,
                        'region': fiber.loc.region,
                    }
                },
                amplifiers=[])
        else:
            amp = elements.Edfa(
                uid=f'Edfa_{fiber.uid}',
                params=EdfaParams.default_values,
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

    if (bounds.start <= length1 <= bounds.stop) and not (bounds.start <= length2 <= bounds.stop):
        return (length1, n_spans1)
    elif (bounds.start <= length2 <= bounds.stop) and not (bounds.start <= length1 <= bounds.stop):
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


def get_previous_node(node, network):
    """get previous node else raise the appropriate error
    """
    try:
        previous_node = next(network.predecessors(node))
        return previous_node
    except StopIteration:
        raise NetworkTopologyError(
            f'{type(node).__name__} {node.uid} is not properly connected, please check network topology')


def split_fiber(network, fiber, bounds, target_length):
    """Splits a fiber into multiple spans if its length exceeds specified boundaries.

    If the length of the fiber exceeds the defined bounds, this function replaces the single span link
    with multiple spans of identical fiber types. It calculates the new lengths and positions for each span.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param fiber: The fiber node to be split into multiple spans.
    :type fiber: elements.Fiber
    :param bounds: The range of lengths to determine if the fiber should be split.
    :type bounds: range
    :param target_length: The target length for each span after splitting.
    :type target_length: float

    :return: None

    :raises NetworkTopologyError: If the fiber is not properly connected in the network topology.

    Notes:
    ------
    - The function calculates the number of spans required and their respective lengths.
    - It updates the network by removing the original fiber and adding the new spans with updated positions.
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
        new_span = elements.Fiber(uid=f'{fiber.uid}_({span + 1}/{n_spans})',
                                  type_variety=fiber.type_variety,
                                  metadata={'location': {
                                            'latitude': lat,
                                            'longitude': lng,
                                            'city': fiber.loc.city,
                                            'region': fiber.loc.region, }
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


def add_connector_loss(network: DiGraph, fibers: List[elements.Fiber], default_con_in: float, 
                       default_con_out: float, EOL: float):
    """Adds default connector loss to fibers in the network if not already defined.

    This function sets the input and output connector losses for each fiber. If no connector loss is defined,
    it assigns default values. Additionally, it adds an end-of-line (EOL) repair margin as a connector loss
    unless the next node is a fused element.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param fibers: A list of fiber nodes to which connector losses will be added.
    :type fibers: List[elements.Fiber]
    :param default_con_in: The default input connector loss to apply if not defined.
    :type default_con_in: float
    :param default_con_out: The default output connector loss to apply if not defined.
    :type default_con_out: float
    :param EOL: The end-of-line repair margin to be added as connector loss.
    :type EOL: float

    :return: None

    Notes:
    ------
    - The function checks each fiber's parameters and updates them accordingly.
    - It skips adding EOL loss if the next node is a fused element.
    """
    for fiber in fibers:
        next_node = get_next_node(fiber, network)
        if fiber.params.con_in is None:
            fiber.params.con_in = default_con_in
        if fiber.params.con_out is None:
            fiber.params.con_out = default_con_out
        if not isinstance(next_node, elements.Fused):
            fiber.params.con_out += EOL


def add_fiber_padding(network: DiGraph, fibers: List[elements.Fiber], padding: float, equipment: dict):
    """Adds padding attenuation at the input of the first fiber in a succession of fibers.

    This function checks each fiber in the network and adds a specified padding loss at the input of the 
    first fiber in a series of connected fibers if the calculated span loss is less than the specified padding.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param fibers: A list of fiber nodes to which padding will be added.
    :type fibers: List[elements.Fiber]
    :param padding: The amount of padding attenuation to apply.
    :type padding: float
    :param equipment: A dictionary containing equipment specifications, including span loss data.
    :type equipment: dict

    :return: None

    Notes:
    ------
    - The function skips padding for Raman fibers and fused elements.
    - It updates the design span loss for each fiber accordingly.
    """
    for fiber in fibers:
        next_node = get_next_node(fiber, network)
        if isinstance(next_node, elements.Fused):
            continue
        # do not pad if this is a Raman Fiber
        if isinstance(fiber, elements.RamanFiber):
            continue
        this_span_loss = span_loss(network, fiber, equipment)
        fiber.design_span_loss = this_span_loss
        if this_span_loss < padding:
            # add a padding att_in at the input of the 1st fiber:
            # address the case when several fibers are spliced together
            first_fiber = find_first_node(network, fiber)
            # in order to support no booster , fused might be placed
            # just after a roadm: need to check that first_fiber is really a fiber
            if isinstance(first_fiber, elements.Fiber):
                first_fiber.params.att_in = first_fiber.params.att_in + padding - this_span_loss
                fiber.design_span_loss += first_fiber.params.att_in


def add_missing_elements_in_network(network: DiGraph, equipment: dict):
    """Autodesigns the network by adding missing elements.

    This function splits fibers that are too long, adds ROADM preamplifiers or boosters,
    and inserts inline amplifiers between fibers to ensure proper network design.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications, including span data.
    :type equipment: dict

    :return: None

    Notes:
    ------
    - The function calculates the maximum and minimum lengths for fibers based on equipment specifications.
    - It processes all fiber nodes to ensure they are within acceptable length limits and adds necessary amplifiers.
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


def add_missing_fiber_attributes(network: DiGraph, equipment: dict):
    """Fills in missing connector loss attributes for fibers in the network.

    This function sets default connector loss values for fibers and adds padding loss where required.
    It also includes end-of-line (EOL) losses as part of the connector loss configuration.

    :param network: The network graph containing nodes and links.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications, including default span data.
    :type equipment: dict

    :return: None

    Notes:
    ------
    - The function processes all fiber nodes in the network to ensure they have the necessary attributes.
    - It separates the addition of connector losses and padding losses for clarity in the code.
    """
    default_span_data = equipment['Span']['default']
    fibers = [f for f in network.nodes() if isinstance(f, elements.Fiber)]
    add_connector_loss(network, fibers, default_span_data.con_in, default_span_data.con_out, default_span_data.EOL)
    # don't group split fiber and add amp in the same loop
    # =>for code clarity (at the expense of speed):
    add_fiber_padding(network, fibers, default_span_data.padding, equipment)


def build_network(network: DiGraph, equipment: dict, reference_channel,
                  set_connector_losses: bool = True, verbose: bool = True):
    """Sets the ROADM equalization targets and amplifier gain and power.

    This function configures the network by setting the equalization targets for ROADMs,
    configuring the design bands for each node, and adjusting the gain, delta power, and output
    variable optical attenuation (VOA) for amplifiers along the optical paths.

    :param network: The network graph representing the topology to be configured.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications and configurations.
    :type equipment: dict
    :param pref_ch_db: The reference channel power in dBm.
    :type pref_ch_db: float
    :param pref_total_db: The reference total power in dBm.
    :type pref_total_db: float
    :param set_connector_losses: Flag indicating whether to set connector losses (default is True).
    :type set_connector_losses: bool
    :param verbose: Flag indicating whether to print warnings and information (default is True).
    :type verbose: bool

    :return: None

    Notes:
    ------
    - The function processes ROADMs and transceivers to set their design bands and equalization targets.
    - It also configures amplifiers and sets input powers for fibers in the network.
    """
    roadms = [r for r in network.nodes() if isinstance(r, elements.Roadm)]
    transceivers = [t for t in network.nodes() if isinstance(t, elements.Transceiver)]

    if set_connector_losses:
        add_missing_fiber_attributes(network, equipment)
    # set roadm equalization targets first
    for roadm in roadms:
        set_roadm_ref_carrier(roadm, equipment)
        set_roadm_per_degree_targets(roadm, network)
        set_per_degree_design_band(roadm, network, equipment)
    for transceiver in transceivers:
        set_per_degree_design_band(transceiver, network, equipment)
    # then set amplifiers gain, delta_p and out_voa on each OMS
    pref_ch_db = watt2dbm(reference_channel.power)
    for roadm in roadms + transceivers:
        set_egress_amplifier(network, roadm, equipment, pref_ch_db, verbose, reference_channel)
    for roadm in roadms:
        set_roadm_input_powers(network, roadm, equipment, pref_ch_db)
        set_roadm_internal_paths(roadm, network)
    for fiber in [f for f in network.nodes() if isinstance(f, (elements.Fiber, elements.RamanFiber))]:
        set_fiber_input_power(network, fiber, equipment, pref_ch_db)


def design_network(reference_channel, network: DiGraph, equipment: Dict, set_connector_losses: bool = True,
                   verbose: bool = True):
    """Designs the network according to the specified reference channel.

    This function configures the network based on the properties of the reference channel, including
    the number of channels and their power levels. It also allows for the setting of connector losses
    and controls the verbosity of logging for warnings and information.

    :param reference_channel: The reference channel used for designing the network.
    :type reference_channel: gnpy.topology.request.PathRequest
    :param network: The network graph representing the topology to be designed.
    :type network: DiGraph
    :param equipment: A dictionary containing equipment specifications and configurations.
    :type equipment: Dict
    :param set_connector_losses: Flag indicating whether to set connector losses (default is True).
    :type set_connector_losses: bool
    :param verbose: Flag indicating whether to print warnings and information (default is True).
    :type verbose: bool

    :return: None

    Notes:
    ------
    - The function calculates the reference total power based on the number of channels and their power levels.
    """
    if verbose:
        logger.info(f'\nReference used for design: (Input optical power reference in span = {watt2dbm(reference_channel.power):.2f}dBm\n'   # noqa E501
                    + f'                            spacing = {reference_channel.spacing * 1e-9:.3f}GHz\n'
                    + f'                            nb_channels = {reference_channel.nb_channel}')
    build_network(network, equipment, reference_channel, set_connector_losses=set_connector_losses,
                  verbose=verbose)
