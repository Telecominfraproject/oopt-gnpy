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
                      draw_networkx_labels, dijkstra_path)
from numpy import mean
from examples.convert_service_sheet import convert_service_sheet, Request_element
from gnpy.core.utils import load_json
from gnpy.core import network_from_json, build_network
from gnpy.core.equipment import read_eqpt_library
from examples.convert import convert_file
from gnpy.core.elements import Transceiver
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.info import SpectralInformation, Channel, Power
from copy import copy, deepcopy

EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'
NETWORK_FILENAME = Path(__file__).parent / 'meshTopologyExampleV2.xls'

logger = getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('filename', nargs='?', type = Path, default='meshTopologyExampleV2.xls')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-o', '--output', default=None)


class Path_request():
    def __init__(self,jsondata):
        self.request_id = jsondata['request-id']
        self.source = jsondata['src-tp-id']
        self.destination = jsondata['dst-tp-id']
        self.signalwidth = 32 #TODO analyse eqptsheet
    def __str__(self):
    	return '\t'.join([f'{self.source}',
    		f'{self.destination}'])
    def __repr__(self):
    	return '\t'.join([f'{self.source}',
    		f'{self.destination}',
    		'\n'])

def requests_from_json(json_data):
    requests_list = []
    for req in json_data['path-request']:
        #print(f'{req}')
        requests_list.append(Path_request(req))

    return requests_list


def load_requests(filename):
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically converting requests from XLS to JSON')
        json_data = convert_service_sheet(filename)
    else:
        with open(filename) as f:
            json_data = loads(f.read())
    return json_data

def load_network(filename):
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
    read_eqpt_library(EQPT_LIBRARY_FILENAME)
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
    for pathreq in pathreqlist:
        print(f'Computing feasibility of path request from' +
        	f'{pathreq.source} to {pathreq.destination}')

        source = next(el for el in trx if el.uid == pathreq.source)
        destination = next(el for el in trx if el.uid == pathreq.destination)
        path = dijkstra_path(network, source, destination)
        p=db2lin(2)*1e-3
        spacing = 0.05 #THz
        si = SpectralInformation() # !! SI units W, Hz
        si = si.update(carriers=tuple(Channel(f, (191.3+spacing*f)*1e12, 
            32e9, 0.15, Power(p, 0, 0)) for f in range(1,80)))
        for el in path:
            si = el(si)
        # we record the last tranceiver object in order to have th whole 
        # information about spectrum. Important Note: since transceivers 
        # attached to roadms are actually logical elements to simulate
        # performance, several demands having the same destination may use 
        # the same transponder for the performance simaulation. This is why 
        # we use deepcopy: to ensure each propagation is recorded and not 
        # overwritten 
        path_res_list.append(deepcopy(destination))
    return path_res_list

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(args.verbose, CRITICAL))
    logger.info(f'Computing path requests {args.filename} into JSON format')
    data = load_requests(args.filename)
    network = load_network(NETWORK_FILENAME)
    pths = requests_from_json(data)
    test = compute_path(network,pths)
    
    if args.output is None:
        print("todo write results")
        for p in test:
	        print(round(mean(p.snr),2))

    else:
        with open(args.output, 'w') as f:
            f.write(test)