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
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power, load_SI
from gnpy.core.request import Path_request, Result_element
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


def load_Transceiver(filename):
    with open(filename) as f:
        json_data = loads(f.read())
        return json_data['Transceiver']

def requests_from_json(json_data,eqpt_filename):
    requests_list = []
    tspjsondata = load_Transceiver(eqpt_filename)

    for req in json_data['path-request']:
        #print(f'{req}')
        params = {}
        params['request_id'] = req['request-id']
        params['source'] = req['src-tp-id']
        params['destination'] = req['dst-tp-id']
        params['trx_type'] = req['path-constraints']['te-bandwidth']['trx_type']
        params['trx_mode'] = req['path-constraints']['te-bandwidth']['trx_mode']
        try:
            extra_params = next(m 
            for t in  tspjsondata if t['type_variety'] == params['trx_type']
            for m in t['mode']  if  m['format'] == params['trx_mode'])
        except StopIteration :
            msg = f'could not find tsp : {params} with mode: {params} in eqpt library'
            raise ValueError(msg)
        nd_list = req['optimizations']['explicit-route-include-objects']
        params['nodes_list'] = [n['unnumbered-hop']['node-id'] for n in nd_list]
        params['loose_list'] = [n['unnumbered-hop']['hop-type'] for n in nd_list]
        params['spacing'] = req['path-constraints']['te-bandwidth']['spacing']
        params['power'] = req['path-constraints']['te-bandwidth']['output-power']
        params['nb_channel'] = req['path-constraints']['te-bandwidth']['max-nb-of-channel']

        params.update(extra_params)
        requests_list.append(Path_request(**params))

    return requests_list


def load_requests(filename,eqpt_filename):
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically converting requests from XLS to JSON')
        json_data = convert_service_sheet(filename,eqpt_filename)
    else:
        with open(filename) as f:
            json_data = loads(f.read())
    return json_data

def compute_path(network, pathreqlist):
    # temporary : repeats calls from transmission_main_example
    # to be merged when ready
    
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
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with explicit path: {pathreq.nodes_list}')

        source = next(el for el in trx if el.uid == pathreq.source)
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
        # for debug
        # print(f'{pathreq.baudrate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        si = create_input_spectral_information(
            sidata['f_min'], sidata['roll_off'],
            pathreq.baudrate, pathreq.power, pathreq.spacing, pathreq.nb_channel)
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
    equipment = load_equipment(args.eqpt_filename)
    network = load_network(args.network_filename,equipment)
    build_network(network, equipment=equipment)
    pths = requests_from_json(data, args.eqpt_filename)
    print(pths)
    test = compute_path(network,pths)

    if args.output is None:
        #TODO write results
        print("demand\t\t\t\tsnr@bandwidth\tsnr@0.1nm")
        
        for i, p in enumerate(test):
            print(f'{pths[i].source} to {pths[i].destination} : {round(mean(p[-1].snr),2)} ,\
                {round(mean(p[-1].snr+10*log10(pths[i].baudrate/(12.5e9))),2)}')
    else:
        result = []
        for p in test:
            result.append(Result_element(pths[test.index(p)],p))
        with open(args.output, 'w') as f:
            f.write(dumps(path_result_json(result), indent=2))