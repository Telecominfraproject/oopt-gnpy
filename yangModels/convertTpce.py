#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
path_requests_run.py
====================

Reads a JSON request file in accordance with the Yang model
for requesting path computation and returns path results in terms
of path and feasibilty.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from argparse import ArgumentParser
from pathlib import Path
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps, loads

logger = getLogger(__name__)

parser = ArgumentParser(description = 'A function that translate requests from ietf model to simplified model for transportPCE.')
parser.add_argument('service_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2_service.json')
parser.add_argument('-o', '--output')


def from_gnpy(json_data):

    r = []
    for req in json_data['path-request']:
        # init all params from request
        params = {}
        params['request-id']   = req['request-id']
        params['source']       = req['source']
        params['destination']  = req['destination']
        params['src-tp-id']    = req['src-tp-id']
        params['dst-tp-id']    = req['dst-tp-id']

        nd_list = req['optimizations']['explicit-route-include-objects']
        rio = []


        params['explicit-route-objects'] = {}
        nd_list = req['optimizations']['explicit-route-include-objects']
        for e in nd_list:
            t = {}
            t['explicit-route-usage'] = "route-include-ero"
            t['index'] = e['index']
            t['num-unnum-hop'] = {}
            t['num-unnum-hop']['node-id']    = e['unnumbered-hop']['node-id']
            t['num-unnum-hop']['link-tp-id'] = e['unnumbered-hop']['link-tp-id']
            t['num-unnum-hop']['hop-type']   = e['unnumbered-hop']['hop-type']
            rio.append(t)
        params['explicit-route-objects']['route-object-include-exclude'] = rio
        
        params['path-constraints'] = req['path-constraints']    
        r.append(params)
    s = []
    for sync in json_data['synchronization']:
        params = {}
        params['synchronization-id'] = sync['synchronization-id']
        params['svec'] = {}
        params['svec']['relaxable'] = sync['svec']['relaxable']
        if sync['svec']['node-diverse'] == 'False' and sync['svec']['link-diverse'] == 'False' : 
            params['svec']['disjointness'] = 0
        if sync['svec']['node-diverse'] == 'True' and sync['svec']['link-diverse'] == 'False' : 
            params['svec']['disjointness'] = 1
        if sync['svec']['node-diverse'] == 'False' and sync['svec']['link-diverse'] == 'True' : 
            params['svec']['disjointness'] = 2
        if sync['svec']['node-diverse'] == 'True' and sync['svec']['link-diverse'] == 'True' : 
            params['svec']['disjointness'] = 3                    
        params['svec']['request-id-number'] = sync['svec']['request-id-number']
        s.append(params)

    return {'path-request' : r,
            'synchronization' : s}



if __name__ == '__main__':

    args = parser.parse_args()
    logger.info(f'Computing path requests {args.service_filename} into JSON format')
    # for debug
    # print( args.eqpt_filename)
    with open(args.service_filename, encoding='utf-8') as f:
        gnpy_data = loads(f.read())
    data = from_gnpy(gnpy_data)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(dumps(data, indent=2, ensure_ascii=False))
    print(data)