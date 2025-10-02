#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.topology.spectrum_assignment: spectrum assignment functionality
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors


"""
gnpy.topology.topology_parameters
========================

This module contains all parameters for topology modules
"""

from copy import deepcopy
from typing import Dict

from gnpy.core.exceptions import ParametersError


class BaseParams:
    """Base class for handling parameter initialization and updating.

    :ivar default_values: A dictionary of default parameter values.
    :vartype default_values: Dict[str, any]
    """
    default_values = {}

    def __init__(self, **params):
        """Initialize the parameters with provided values or defaults.

        :param params: Keyword arguments representing parameter values.

        :raises ParametersError: If required parameters are missing.
        """
        try:
            self.update_attr(params)
        except KeyError as e:
            raise ParametersError(f'Request json must include {e}. Configuration: {params}')

    def update_attr(self, kwargs: Dict):
        """Update the attributes of the instance based on provided keyword arguments.

        :param kwargs: A dictionary of parameters to update.
        :type kwargs: Dict[str, Any]
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            # use deepcopy to avoid sharing same object amongst all instance when v is a list or a dict!
            if isinstance(v, (list, dict)):
                setattr(self, k, clean_kwargs.get(k, deepcopy(v)))
            else:
                setattr(self, k, clean_kwargs.get(k, v))


class RequestParams(BaseParams):
    """Class to handle request parameters for topology modules.

    :ivar request_id: Identifier for the request.
    :vartype request_id: Optional[str]
    :ivar trx_type: Transceiver type.
    :vartype trx_type: Optional[str]
    :ivar trx_mode: Transceiver mode.
    :vartype trx_mode: Optional[str]
    :ivar source: Source node.
    :vartype source: Optional[str]
    :ivar destination: Destination node.
    :vartype destination: Optional[str]
    :ivar bidir: Indicates if the request feasibility should be also computed in the opposite direction.
    :vartype bidir: bool
    :ivar nodes_list: List of nodes that must be included in the path.
    :vartype nodes_list: List[str]
    :ivar loose_list: list of loose condition on the include nodes_list.
    :vartype loose_list: List[str]
    :ivar format: Format of the request.
    :vartype format: str
    :ivar baud_rate: Baud rate.
    :vartype baud_rate: Optional[float]
    :ivar bit_rate: Bit rate.
    :vartype bit_rate: Optional[float]
    :ivar roll_off: Roll-off factor.
    :vartype roll_off: Optional[float]
    :ivar OSNR: Optical Signal-to-Noise Ratio.
    :vartype OSNR: Optional[float]
    :ivar penalties: Penalties applied.
    :vartype penalties: Optional[float]
    :ivar path_bandwidth: Bandwidth of the path.
    :vartype path_bandwidth: Optional[float]
    :ivar effective_freq_slot: Effective frequency slot.
    :vartype effective_freq_slot: Optional[float]
    :ivar f_min: Minimum frequency.
    :vartype f_min: Optional[float]
    :ivar f_max: Maximum frequency.
    :vartype f_max: Optional[float]
    :ivar spacing: Spacing between channels.
    :vartype spacing: Optional[float]
    :ivar min_spacing: Minimum spacing.
    :vartype min_spacing: Optional[float]
    :ivar cost: Cost associated with the request.
    :vartype cost: Optional[float]
    :ivar nb_channel: Number of channels.
    :vartype nb_channel: Optional[int]
    :ivar power: Power level.
    :vartype power: Optional[float]
    :ivar equalization_offset_db: Equalization offset in dB.
    :vartype equalization_offset_db: Optional[float]
    :ivar tx_power: Transmit power out of transceiver.
    :vartype tx_power: Optional[float]
    :ivar tx_osnr: Transmit OSNR out of transceiver.
    :vartype tx_osnr: Optional[float]
    """
    default_values = {
        'request_id': None,
        'trx_type': None,
        'trx_mode': None,
        'source': None,
        'destination': None,
        'bidir': False,
        'nodes_list': [],
        'loose_list': [],
        'format': '',
        'baud_rate': None,
        'bit_rate': None,
        'roll_off': None,
        'OSNR': None,
        'penalties': None,
        'path_bandwidth': None,
        'effective_freq_slot': None,
        'f_min': None,
        'f_max': None,
        'spacing': None,
        'min_spacing': None,
        'cost': None,
        'nb_channel': None,
        'power': None,
        'equalization_offset_db': None,
        'tx_power': None,
        'tx_osnr': None,
        'tx_channel_power_min': None,
        'tx_channel_power_max': None,
        'rx_channel_power_min': None,
        'rx_channel_power_max': None,
        'rx_ref_channel_power': None
    }


class DisjunctionParams(BaseParams):
    """Class to handle disjunction parameters for topology modules.

    :ivar disjunction_id: Identifier for the disjunction group of requests.
    :vartype disjunction_id: Optional[str]
    :ivar relaxable: Indicates if the disjunction is relaxable.
    :vartype relaxable: bool
    :ivar link_diverse: Indicates if link diversity is required.
    :vartype link_diverse: bool
    :ivar node_diverse: Indicates if node diversity is required.
    :vartype node_diverse: bool
    :ivar disjunctions_req: List of request that must be routed disjointly.
    :vartype disjunctions_req: List[str]
    """
    default_values = {
        'disjunction_id': None,
        'relaxable': False,
        'link_diverse': True,
        'node_diverse': True,
        'disjunctions_req': []
    }
