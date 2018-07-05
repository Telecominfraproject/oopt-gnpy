#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : path_requests_run.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
@author: jeanluc-auge
read json request file in accordance with:
    Yang model for requesting Path Computation
    draft-ietf-teas-yang-path-computation-01.txt. 
and returns path results in terms of path and feasibility

"""

from sys import exit
from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps, loads
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path, NetworkXNoPath)
from numpy import mean
from examples.convert_service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.utils import load_json
from gnpy.core.network import load_network, build_network
from gnpy.core.equipment import load_equipment
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power
from copy import copy, deepcopy
from numpy import log10


RequestParams = namedtuple('RequestParams','request_id source destination trx_type'+
' trx_mode nodes_list loose_list spacing power nb_channel format baudrate OSNR bit_rate')

class Path_request:
    def __init__(self, *args, **params):
        params = RequestParams(**params)
        self.request_id = params.request_id
        self.source = params.source
        self.destination = params.destination
        self.tsp        = params.trx_type
        self.tsp_mode   = params.trx_mode
        # retrieve baudrate out of transponder type and mode (format)

        self.baudrate = params.baudrate
        self.nodes_list = params.nodes_list
        self.loose_list = params.loose_list
        self.spacing    = params.spacing
        self.power      = params.power
        self.nb_channel = params.nb_channel
        self.format     = params.format
        self.OSNR       = params.OSNR
        self.bit_rate   = params.bit_rate
        

    def __str__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source:       {self.source}',
                            f'destination:  {self.destination}'])
    def __repr__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source:       {self.source}',
                            f'destination:  {self.destination}',
                            f'trx type:     {self.tsp}',
                            f'baudrate:     {self.baudrate}',
                            f'spacing:      {self.spacing}',
                            f'power:        {self.power}'
                            '\n'])

class Result_element(Element):
    def __init__(self,path_request,computed_path):
        self.path_id = int(path_request.request_id)
        self.path_request = path_request
        self.computed_path = computed_path
        hop_type = []
        for e in computed_path :
            if isinstance(e, Transceiver) : 
                hop_type.append(' - '.join([path_request.tsp,path_request.tsp_mode])) 
            else:
                hop_type.append('not recorded')
        self.hop_type = hop_type
    uid = property(lambda self: repr(self))
    @property
    def pathresult(self):
        return {
               'path-id': self.path_id,
               'path-properties':{
                   'path-metric': [
                       {
                       'metric-type': 'SNR@bandwidth',
                       'accumulative-value': round(mean(self.computed_path[-1].snr),2)
                       },
                       {
                       'metric-type': 'SNR@0.1nm',
                       'accumulative-value': round(mean(self.computed_path[-1].snr+10*log10(self.path_request.baudrate/12.5)),2)
                       }
                    ],
                    'path-srlgs': {
                        'usage': 'not used yet',
                        'values': 'not used yet'
                    },
                    'path-route-objects': [
                        {
                        'path-route-object': {
                            'index': self.computed_path.index(n),
                            'unnumbered-hop': {
                                'node-id': n.uid,
                                'link-tp-id': n.uid,
                                'hop-type': self.hop_type[self.computed_path.index(n)],
                                'direction': 'not used'
                            },
                            'label-hop': {
                                'te-label': {
                                    'generic': 'not used yet',
                                    'direction': 'not used yet'
                                    }
                                }
                            }
                        } for n in self.computed_path
                        ]
                }
            }
                    
    @property
    def json(self):
        return self.pathresult 

