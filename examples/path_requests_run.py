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
        tsp = jsondata['path-constraints']['te-bandwidth']['trx_type']
        tsp_mode = jsondata['path-constraints']['te-bandwidth']['trx_mode']
        # for debug
        print(tsp)
        tsp_found = False
        mode_found = False
        for t in tspjsondata:
            if t['type_variety']== tsp :
                #	print('coucou')
                tsp_found = True
                for m in t['mode']:
                    if m['format']==tsp_mode:
                        mode_found = True
                        b = m['baudrate']
                        # for debug
                        print(b)
                        self.baudrate = b 
        print(tsp_found and mode_found)
        if not (tsp_found and mode_found) : 
            msg = f'could not find tsp : {tsp} with mode: {tsp_mode} in eqpt library'
            logger.critical(msg)
            raise ValueError(msg)

    def __str__(self):
        return '\t'.join([f'{self.source}',
            f'{self.destination}'])
    def __repr__(self):
        return '\t'.join([f'{self.source}',
            f'{self.destination}',
            '\n'])

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

def create_input_spectral_information(sidata,baudrate):
    si = SpectralInformation() # !! SI units W, Hz
    si = si.update(carriers=tuple(Channel(f, (sidata['f_min']+sidata['spacing']*f), 
            baudrate*1e9, sidata['roll_off'], Power(sidata['power'], 0, 0)) for f in range(1,sidata['Nch'])))
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
    sidata = load_SI(args.eqpt_filename)
    for pathreq in pathreqlist:
        print(f'Computing feasibility of path request from' +
            f'{pathreq.source} to {pathreq.destination}')

        source = next(el for el in trx if el.uid == pathreq.source)
        destination = next(el for el in trx if el.uid == pathreq.destination)
        path = dijkstra_path(network, source, destination)
        si = create_input_spectral_information(sidata,pathreq.baudrate)
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
    logger.info(f'Computing path requests {args.service_filename} into JSON format')
    print( args.eqpt_filename)
    data = load_requests(args.service_filename,args.eqpt_filename)
    network = load_network(args.network_filename,args.eqpt_filename)
    pths = requests_from_json(data, args.eqpt_filename)
    test = compute_path(network,pths)

    if args.output is None:
        print("todo write results")
        print("demand\tsnr@bandwidth\tosnr@0.1nm")
        i = 0
        for p in test:
            print(f'{pths[i].source} to {pths[i].destination} : {round(mean(p.snr),2)} , {round(mean(p.osnr_ase_01nm),2)}')
            i = i+1
    else:
        with open(args.output, 'w') as f:
            f.write(test)