#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.request
=================

This module contains path request functionality.

This functionality allows the user to provide a JSON request
file in accordance with a Yang model for requesting path
computations and returns path results in terms of path
and feasibility

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from networkx import (dijkstra_path, NetworkXNoPath)
from numpy import mean
from examples.convert_service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused
from gnpy.core.network import set_roadm_loss
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power
from copy import copy, deepcopy
from csv import writer

logger = getLogger(__name__)


RequestParams = namedtuple('RequestParams','request_id source destination trx_type'+
' trx_mode nodes_list loose_list spacing power nb_channel frequency format baud_rate OSNR bit_rate roll_off')

class Path_request:
    def __init__(self, *args, **params):
        params = RequestParams(**params)
        self.request_id = params.request_id
        self.source     = params.source
        self.destination = params.destination
        self.tsp        = params.trx_type
        self.tsp_mode   = params.trx_mode
        self.baud_rate  = params.baud_rate
        self.nodes_list = params.nodes_list
        self.loose_list = params.loose_list
        self.spacing    = params.spacing
        self.power      = params.power
        self.nb_channel = params.nb_channel
        self.frequency  = params.frequency
        self.format     = params.format
        self.OSNR       = params.OSNR
        self.bit_rate   = params.bit_rate
        self.roll_off   = params.roll_off

    def __str__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source:       {self.source}',
                            f'destination:  {self.destination}'])
    def __repr__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source: \t{self.source}',
                            f'destination:\t{self.destination}',
                            f'trx type:\t{self.tsp}',
                            f'trx mode:\t{self.tsp_mode}',
                            f'baud_rate:\t{self.baud_rate * 1e-9} Gbaud',
                            f'bit_rate:\t{self.bit_rate * 1e-9} Gb/s',
                            f'spacing:\t{self.spacing * 1e-9} GHz',
                            f'power:  \t{round(lin2db(self.power)+30,2)} dBm'
                            '\n'])

class Result_element(Element):
    def __init__(self,path_request,computed_path):
        self.path_id = path_request.request_id
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
        if not self.computed_path:
            return {
                   'path-id': self.path_id,
                   'path-properties':{
                       'path-metric': [
                           {
                           'metric-type': 'SNR@bandwidth',
                           'accumulative-value': 'None'
                           },
                           {
                           'metric-type': 'SNR@0.1nm',
                           'accumulative-value': 'None'
                           },
                           {
                           'metric-type': 'OSNR@bandwidth',
                           'accumulative-value': 'None'
                           },
                           {
                           'metric-type': 'OSNR@0.1nm',
                           'accumulative-value': 'None'
                           },
                           {
                           'metric-type': 'reference_power',
                           'accumulative-value': self.path_request.power
                           }
                        ],
                        'path-srlgs': {
                            'usage': 'not used yet',
                            'values': 'not used yet'
                        },
                        'path-route-objects': [
                            {
                            'path-route-object': {
                                'index': 0,
                                'unnumbered-hop': {
                                    'node-id': self.path_request.source,
                                    'link-tp-id': self.path_request.source,
                                    'hop-type': ' - '.join([self.path_request.tsp, self.path_request.tsp_mode]),
                                    'direction': 'not used'
                                },
                                'label-hop': {
                                    'te-label': {
                                        'generic': 'not used yet',
                                        'direction': 'not used yet'
                                        }
                                    }
                                }
                            },
                            {
                            'path-route-object': {
                                'index': 1,
                                'unnumbered-hop': {
                                    'node-id': self.path_request.destination,
                                    'link-tp-id': self.path_request.destination,
                                    'hop-type': ' - '.join([self.path_request.tsp, self.path_request.tsp_mode]),
                                    'direction': 'not used'
                                },
                                'label-hop': {
                                    'te-label': {
                                        'generic': 'not used yet',
                                        'direction': 'not used yet'
                                        }
                                    }
                                }
                            }
                            ]
                    }
                }
        else:
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
                           'accumulative-value': round(mean(self.computed_path[-1].snr+lin2db(self.path_request.baud_rate/12.5e9)),2)
                           },
                           {
                           'metric-type': 'OSNR@bandwidth',
                           'accumulative-value': round(mean(self.computed_path[-1].osnr_ase),2)
                           },
                           {
                           'metric-type': 'OSNR@0.1nm',
                           'accumulative-value': round(mean(self.computed_path[-1].osnr_ase_01nm),2)
                           },
                           {
                           'metric-type': 'reference_power',
                           'accumulative-value': self.path_request.power
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

def compute_constrained_path(network, req):
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)]
    roadm = [n for n in network.nodes() if isinstance(n, Roadm)]
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)]
    source = next(el for el in trx if el.uid == req.source)
    # start the path with its source
    # TODO : avoid loops due to constraints , guess name base on string,
    # avoid crashing if on req is not correct
    total_path = [source]
    for n in req.nodes_list:
        # print(n)
        try :
            node = next(el for el in trx if el.uid == n)
        except StopIteration:
            try:
                node = next(el for el in roadm if el.uid == f'roadm {n}')
            except StopIteration:
                try:
                    node = next(el for el in edfa
                        if el.uid.startswith(f'egress edfa in {n}'))
                except StopIteration:
                    msg = f'could not find node : {n} in network topology: \
                        not a trx, roadm, edfa or fused element'
                    logger.critical(msg)
                    raise ValueError(msg)
        # extend path list without repeating source -> skip first element in the list
        try:
            total_path.extend(dijkstra_path(network, source, node)[1:])
            source = node
        except NetworkXNoPath:
            # for debug
            # print(req.loose_list)
            # print(req.nodes_list.index(n))
            if req.loose_list[req.nodes_list.index(n)] == 'loose':
                print(f'could not find a path from {source.uid} to loose node : {n} in network topology')
                print(f'node  {n} is skipped')
            else:
                msg = f'could not find a path from {source.uid} to node : {n} in network topology'
                logger.critical(msg)
                #raise ValueError(msg)
                print(msg)
                total_path = []

# preparing disjonction feature
    # for p in all_simple_paths(network,\
    #     source=next(el for el in trx if el.uid == req.source),\
    #     target=next(el for el in trx if el.uid == req.destination)):
    #     print([e.uid for e in p if isinstance(e,Roadm)])

    return total_path

def propagate(path, req, equipment, show=False):
    #update roadm loss in case of power sweep (power mode only)
    set_roadm_loss(path, equipment, lin2db(req.power*1e3))
    si = create_input_spectral_information(
        req.frequency['min'], req.roll_off,
        req.baud_rate, req.power, req.spacing, req.nb_channel)
    for el in path:
        si = el(si)
        if show :
            print(el)
    return path


def jsontocsv(json_data,equipment,fileout):
    # read json path result file in accordance with:
    # Yang model for requesting Path Computation
    # draft-ietf-teas-yang-path-computation-01.txt.
    # and write results in an CSV file

    mywriter = writer(fileout)
    mywriter.writerow(('path-id','source','destination','transponder-type',\
        'transponder-mode','baud rate (Gbaud)', 'input power (dBm)','path',\
        'OSNR@bandwidth','OSNR@0.1nm','SNR@bandwidth','SNR@0.1nm','Pass?'))
    tspjsondata = equipment['Transceiver']
    #print(tspjsondata)
    for p in json_data['path']:
        path_id     = p['path-id']
        source      = p['path-properties']['path-route-objects'][0]\
        ['path-route-object']['unnumbered-hop']['node-id']
        destination = p['path-properties']['path-route-objects'][-1]\
        ['path-route-object']['unnumbered-hop']['node-id']
        pth        = ' | '.join([ e['path-route-object']['unnumbered-hop']['node-id']
                 for e in p['path-properties']['path-route-objects']])

        [tsp,mode] = p['path-properties']['path-route-objects'][0]\
        ['path-route-object']['unnumbered-hop']['hop-type'].split(' - ')

        # find the min  acceptable OSNR, baud rate from the eqpt library based on tsp (tupe) and mode (format)
        try:
            [minosnr, baud_rate] = next([m['OSNR'] , m['baud_rate']]
                for m in equipment['Transceiver'][tsp].mode if  m['format']==mode)

        # for debug
        # print(f'coucou {baud_rate}')
        except IndexError:
            msg = f'could not find tsp : {self.tsp} with mode: {self.tsp_mode} in eqpt library'

            raise ValueError(msg)
        output_snr = next(e['accumulative-value']
            for e in p['path-properties']['path-metric'] if e['metric-type'] == 'SNR@0.1nm')
        output_snrbandwidth = next(e['accumulative-value']
            for e in p['path-properties']['path-metric'] if e['metric-type'] == 'SNR@bandwidth')
        output_osnr = next(e['accumulative-value']
            for e in p['path-properties']['path-metric'] if e['metric-type'] == 'OSNR@0.1nm')
        output_osnrbandwidth = next(e['accumulative-value']
            for e in p['path-properties']['path-metric'] if e['metric-type'] == 'OSNR@bandwidth')
        power = next(e['accumulative-value']
            for e in p['path-properties']['path-metric'] if e['metric-type'] == 'reference_power')
        if isinstance(output_snr, str):
            isok = ''
        else:
            isok = output_snr >= minosnr
        mywriter.writerow((path_id,
            source,
            destination,
            tsp,
            mode,
            baud_rate*1e-9,
            round(lin2db(power)+30,2),
            pth,
            output_osnrbandwidth,
            output_osnr,
            output_snrbandwidth,
            output_snr,
            isok
            ))
