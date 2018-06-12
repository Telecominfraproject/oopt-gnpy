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
from gnpy.core import network_from_json, build_network
from gnpy.core.equipment import read_eqpt_library
from examples.convert import convert_file
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.info import SpectralInformation, Channel, Power
from copy import copy, deepcopy
from numpy import log10

#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

logger = getLogger(__name__)

parser = ArgumentParser(description = 'A function that computes performances for a list of services provided in a json file or an excel sheet.')
parser.add_argument('network_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls')
parser.add_argument('service_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls')
parser.add_argument('eqpt_filename', nargs='?', type = Path, default=Path(__file__).parent / 'eqpt_config.json')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-o', '--output', default=None)


class Path_request():
    def __init__(self,jsondata,tspjsondata):
        self.request_id = jsondata['request-id']
        self.source = jsondata['src-tp-id']
        self.destination = jsondata['dst-tp-id']
        # retrieving baudrate out of transponder type and mode (format)
        self.tsp = jsondata['path-constraints']['te-bandwidth']['trx_type']
        self.tsp_mode = jsondata['path-constraints']['te-bandwidth']['trx_mode']
        # for debug
        # print(tsp)
        # refactoring into a simple expression
        # tsp_found = False
        # mode_found = False
        # for t in tspjsondata:
        #     if t['type_variety']== tsp :
        #         #    print('coucou')
        #         tsp_found = True
        #         for m in t['mode']:
        #             if m['format']==tsp_mode:
        #                 mode_found = True
        #                 b = m['baudrate']
        #                 # for debug
        #                 print(b)
        #                 self.baudrate = b 
        try:
            baudrate = [m['baudrate'] 
                for t in  tspjsondata if t['type_variety']== self.tsp
                for m in t['mode']  if  m['format']==self.tsp_mode][0]
            # for debug
            # print(f'coucou {baudrate}')
        except IndexError:
            msg = f'could not find tsp : {self.tsp} with mode: {self.tsp_mode} in eqpt library'
            logger.critical(msg)
            raise ValueError(msg)
        self.baudrate = baudrate

        nodes_list = jsondata['optimizations']['explicit-route-include-objects']
        self.nodes_list = [n['unnumbered-hop']['node-id'] for n in nodes_list]
        # create a list for individual loose capability for each node ... even if convert_service_sheet fills with the same value
        self.loose_list = [n['unnumbered-hop']['hop-type'] for n in nodes_list]

        self.spacing = jsondata['path-constraints']['te-bandwidth']['spacing']
        self.power = jsondata['path-constraints']['te-bandwidth']['output-power']
        self.nb_channel = jsondata['path-constraints']['te-bandwidth']['max-nb-of-channel']

    def __str__(self):
        return '\t'.join([f'{self.source}',
            f'{self.destination}'])
    def __repr__(self):
        return '\t'.join([f'{self.source}',
            f'{self.destination}',
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

def load_SI(filename):
    with open(filename) as f:
        json_data = loads(f.read())
        return json_data['SI'][0]

def load_Transceiver(filename):
    with open(filename) as f:
        json_data = loads(f.read())
        return json_data['Transceiver']

def requests_from_json(json_data,eqpt_filename):
    requests_list = []
    tspjsondata = load_Transceiver(eqpt_filename)
    for req in json_data['path-request']:
        #print(f'{req}')
        requests_list.append(Path_request(req,tspjsondata))

    return requests_list

# def create_input_spectral_information(sidata,baudrate):
#     si = SpectralInformation() # !! SI units W, Hz
#     si = si.update(carriers=tuple(Channel(f, (sidata['f_min']+sidata['spacing']*f), 
#             baudrate*1e9, sidata['roll_off'], Power(sidata['power'], 0, 0)) for f in range(1,sidata['Nch'])))
#     return si

def create_input_spectral_information(sidata,baudrate,power,spacing,nb_channel):
    si = SpectralInformation() # !! SI units W, Hz
    si = si.update(carriers=tuple(Channel(f, (sidata['f_min']+spacing*f), 
            baudrate*1e9, sidata['roll_off'], Power(power, 0, 0)) for f in range(1,nb_channel)))
    return si

def load_requests(filename,eqpt_filename):
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically converting requests from XLS to JSON')
        json_data = convert_service_sheet(filename,eqpt_filename)
    else:
        with open(filename) as f:
            json_data = loads(f.read())
    return json_data

def load_network(filename,eqpt_filename):
    # to be replaced with the good load_network
    # important note: network should be created only once for a given 
    # simulation. Note that it only generates infrastructure information. 
    # Only one transceiver element is attached per roadm: it represents the 
    # logical starting point / stopping point for the propagation of 
    # the spectral information to be prpagated along a path. 
    # at that point it is not meant to represent the capacity of add drop ports
    # As a result transponder type is not part of the network info. it is related to 
    # the list of services requests

    input_filename = str(filename)
    suffix_filename = str(filename.suffixes[0])
    split_filename = [input_filename[0:len(input_filename)-len(suffix_filename)] , suffix_filename[1:]]
    json_filename = split_filename[0]+'.json'
    try:
        assert split_filename[1] in ('json','xls','csv','xlsm')
    except AssertionError as e:
        print(f'invalid file extension .{split_filename[1]}')
        raise e
    if split_filename[1] != 'json':
        print(f'parse excel input to {json_filename}')
        convert_file(filename)

    json_data = load_json(json_filename)
    read_eqpt_library(eqpt_filename)
    #print(json_data)

    network = network_from_json(json_data)
    build_network(network)
    return network

def compute_path(network, pathreqlist):
    # temporary : repeats calls from transmission_main_example
    # to be merged when ready
    # final function should only be compute_path(network,pathreqlist)
    
    path_res_list = []
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)]
    roadm = [n for n in network.nodes() if isinstance(n, Roadm)]
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)]
    # TODO include also fused in the element check : too difficult because of direction
    # fused = [n for n in network.nodes() if isinstance(n, Fused)]
    sidata = load_SI(args.eqpt_filename)
    for pathreq in pathreqlist:
        pathreq.nodes_list.append(pathreq.destination)
        #we assume that the destination is a strict constraint
        pathreq.loose_list.append('strict')
        #pathreq.nodes_list.insert(0,pathreq.source)
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with explicit path: {pathreq.nodes_list}')

        source = next(el for el in trx if el.uid == pathreq.source)
        # print(source.uid)
        # destination = next(el for el in trx if el.uid == pathreq.destination)
        # start the path with its source
        total_path = [source]
        for n in pathreq.nodes_list:
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
                # print(pathreq.loose_list)
                # print(pathreq.nodes_list.index(n))
                if pathreq.loose_list[pathreq.nodes_list.index(n)] == 'loose':
                    print(f'could not find a path from {source.uid} to loose node : {n} in network topology')
                    print(f'node  {n} is skipped')
                else:
                    msg = f'could not find a path from {source.uid} to node : {n} in network topology'
                    logger.critical(msg)
                    raise ValueError(msg)
        
        # si = create_input_spectral_information(sidata,pathreq.baudrate)
        # for debug
        # print(f'{pathreq.baudrate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        si = create_input_spectral_information(sidata,pathreq.baudrate,pathreq.power,pathreq.spacing,pathreq.nb_channel)
        for el in total_path:
            si = el(si)
            # print(el)
        # we record the last tranceiver object in order to have th whole 
        # information about spectrum. Important Note: since transceivers 
        # attached to roadms are actually logical elements to simulate
        # performance, several demands having the same destination may use 
        # the same transponder for the performance simaulation. This is why 
        # we use deepcopy: to ensure each propagation is recorded and not 
        # overwritten 
        
        # path_res_list.append(deepcopy(destination))
        path_res_list.append(deepcopy(total_path))
    return path_res_list

def path_result_json(pathresult):
    data = {
        'path': [n.json for n in pathresult]
    }
    return data


if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(args.verbose, CRITICAL))
    logger.info(f'Computing path requests {args.service_filename} into JSON format')
    # for debug
    # print( args.eqpt_filename)
    data = load_requests(args.service_filename,args.eqpt_filename)
    network = load_network(args.network_filename,args.eqpt_filename)
    pths = requests_from_json(data, args.eqpt_filename)
    test = compute_path(network,pths)

    if args.output is None:
        print("todo write results")
        print("demand\t\t\t\tsnr@bandwidth\tsnr@0.1nm")
        i = 0
        
        for p in test:
            print(f'{pths[i].source} to {pths[i].destination} : {round(mean(p[-1].snr),2)} ,\
                {round(mean(p[-1].snr+10*log10(pths[i].baudrate/12.5)),2)}')
            i = i+1
    else:
        result = []
        for p in test:
            result.append(Result_element(pths[test.index(p)],p))
        with open(args.output, 'w') as f:
            f.write(dumps(path_result_json(result), indent=2))