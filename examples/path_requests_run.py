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

from sys import exit
from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps, loads
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path, NetworkXNoPath, all_simple_paths)
from networkx.utils import pairwise 
from numpy import mean
from gnpy.core.service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.utils import load_json
from gnpy.core.network import load_network, build_network, set_roadm_loss
from gnpy.core.equipment import load_equipment, trx_mode_params, automatic_nch
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.request import (Path_request, Result_element, compute_constrained_path,
                              propagate, jsontocsv, Disjunction)
from copy import copy, deepcopy

#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

logger = getLogger(__name__)

parser = ArgumentParser(description = 'A function that computes performances for a list of services provided in a json file or an excel sheet.')
parser.add_argument('network_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls')
parser.add_argument('service_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls')
parser.add_argument('eqpt_filename', nargs='?', type = Path, default=Path(__file__).parent / 'eqpt_config.json')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-o', '--output', default=None)


def requests_from_json(json_data,equipment):
    requests_list = []

    for req in json_data['path-request']:
        # print(f'{req}')
        # init all params from request
        params = {}
        params['request_id'] = req['request-id']
        params['source'] = req['src-tp-id']
        params['destination'] = req['dst-tp-id']
        params['trx_type'] = req['path-constraints']['te-bandwidth']['trx_type']
        params['trx_mode'] = req['path-constraints']['te-bandwidth']['trx_mode']
        params['format'] = params['trx_mode']
        nd_list = req['optimizations']['explicit-route-include-objects']
        params['nodes_list'] = [n['unnumbered-hop']['node-id'] for n in nd_list]
        params['loose_list'] = [n['unnumbered-hop']['hop-type'] for n in nd_list]
        params['spacing'] = req['path-constraints']['te-bandwidth']['spacing']

        # recover trx physical param (baudrate, ...) from type and mode
        trx_params = trx_mode_params(equipment,params['trx_type'],params['trx_mode'],True)
        params.update(trx_params)
        params['power'] = req['path-constraints']['te-bandwidth']['output-power']
        params['nb_channel'] = req['path-constraints']['te-bandwidth']['max-nb-of-channel']
        requests_list.append(Path_request(**params))
    return requests_list

def disjunctions_from_json(json_data):
    disjunctions_list = []

    for snc in json_data['synchronization']:
        params = {}
        params['relaxable'] = snc['svec']['relaxable']
        params['link_diverse'] = snc['svec']['link-diverse']
        params['node_diverse'] = snc['svec']['node-diverse']
        params['disjunctions_req'] = snc['svec']['request-id-number']
        disjunctions_list.append(Disjunction(**params))
    return disjunctions_list


def load_requests(filename,eqpt_filename):
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically converting requests from XLS to JSON')
        json_data = convert_service_sheet(filename,eqpt_filename)
    else:
        with open(filename) as f:
            json_data = loads(f.read())
    return json_data

def compute_path(network, equipment, pathreqlist):

    path_res_list = []

    for pathreq in pathreqlist:
        #need to rebuid the network for each path because the total power
        #can be different and the choice of amplifiers in autodesign is power dependant
        #but the design is the same if the total power is the same
        #TODO parametrize the total spectrum power so the same design can be shared
        p_db = lin2db(pathreq.power*1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        build_network(network, equipment, p_db, p_total_db)
        pathreq.nodes_list.append(pathreq.destination)
        #we assume that the destination is a strict constraint
        pathreq.loose_list.append('strict')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}') #adding first node to be clearer on the output
        total_path = compute_constrained_path(network, pathreq)
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}\n')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if total_path :
            total_path = propagate(total_path,pathreq,equipment, show=False)
        else:
            total_path = []
        # we record the last tranceiver object in order to have th whole
        # information about spectrum. Important Note: since transceivers
        # attached to roadms are actually logical elements to simulate
        # performance, several demands having the same destination may use
        # the same transponder for the performance simaulation. This is why
        # we use deepcopy: to ensure each propagation is recorded and not
        # overwritten

        path_res_list.append(deepcopy(total_path))
    return path_res_list

def compute_path_dsjctn(network, equipment, pathreqlist, disjunctions_list):
    
    path_res_list = []

    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    # todo : disjctn must be computed at once together to avoid blocking
    #         1     1
    # eg    a----b-----c
    #       |1   |0.5  |1
    #       e----f--h--g
    #         1  0.5 0.5
    # if I have to compute a to g and a to h 
    # I must not compute a-b-f-h-g, otherwise there is no disjoint path remaining for a to h
    # instead I should list all most disjoint path and select the one that have the less
    # number of commonalities
    #     \     path abfh  aefh   abcgh 
    #      \___cost   2     2.5    3.5
    #   path| cost  
    #  abfhg|  2.5    x      x      x
    #  abcg |  3      x             x
    #  aefhg|  3      x      x      x
    # from this table abcg and aefh have no common links and should be preferred 
    # even they are not the shorpths path
    # build the list of pathreqlist elements not concerned by disjunction
    global_disjunctions_list = [e for d in disjunctions_list for e in d.disjunctions_req ]
    pathreqlist_simple = [e for e in pathreqlist if e.request_id not in global_disjunctions_list]
    pathreqlist_disjt = [e for e in pathreqlist if e.request_id in global_disjunctions_list]
    # compute paths for this simple path
    pathreslist_simple = compute_path(network, equipment, pathreqlist_simple)

    # build a conflict table where each path has a count for 
    # conflicts with the paths from the requests to be disjoint
    # step 1
    # for each remaining request compute a set of simple path
    rqs = {}
    simple_rqs = {}
    for pathreq in pathreqlist_disjt :
        print(pathreq.request_id)
        all_simp_pths = list(all_simple_paths(network,\
            source=next(el for el in network.nodes() if el.uid == pathreq.source),\
            target=next(el for el in network.nodes() if el.uid == pathreq.destination)))
        rqs[pathreq.request_id] = all_simp_pths 
        temp =[]
        for p in all_simp_pths :
            # build a short list representing each roadm+direction with the first item
            # start enumeration at 1 to avoid Trx in the list
            temp.append([e.uid for i,e in enumerate(p[1:-1]) \
                if (isinstance(e,Roadm) | (isinstance(p[i],Roadm) ))] )
        simple_rqs[pathreq.request_id] = temp
        for p in all_simp_pths :
            print ([e.uid for e in p if isinstance (e,Roadm)])
    tab = {}
    tab2 = {}
    # step 2 
    # for each pair in the set of requests that need to be disjoint
    # count the non disjoint cases tab[path] = list of disjoint path
    for d in disjunctions_list :
        print(d)
        temp = d.disjunctions_req.copy()
        for e1 in temp :
            for i,p1 in enumerate(simple_rqs[e1]):
                if temp:
                    for e2 in temp :
                        if e1 != e2 :
                            for j,p2 in enumerate(simple_rqs[e2]):
                                # print(f'{id(p1)}    {id(p2)}')
                                try :
                                    tab[id(p1)] += isdisjoint(p1,p2)
                                    if isdisjoint(p1,p2)==0:
                                        tab2[id(p1)] = tab2[id(p1)].append(p2)
                                    print(f'{e1} is {isdisjoint(p1,p2)} {e2}    tab : {tab2[id(p1),e2]}')
                                except KeyError:
                                    tab[id(p1)] = isdisjoint(p1,p2)
                                    if isdisjoint(p1,p2)==0:
                                        tab2[id(p1)] = [p2]
                                try :
                                    tab[id(p2)] += isdisjoint(p1,p2)
                                    if isdisjoint(p1,p2)==0:
                                        tab2[id(p2)] = tab2[id(p2)].append(p1)
                                except KeyError:
                                    tab[id(p2)] = isdisjoint(p1,p2)
                                    if isdisjoint(p1,p2)==0:
                                        tab2[id(p2)] = [p1]
            # remove the request from the list to avoid computind ij and ji cases
            temp = temp.remove(e1)


    # print(tab)
    # print(len(tab))
    el = disjunctions_list[0].disjunctions_req[0]
    # print(tab[id(simple_rqs[el][0])])

    # now for each request, select the path that has the least nb of disjunction
    # and completely disjoined from the already selected paths in the constraint
    for pathreq in pathreqlist_disjt :
        pths = [ tab[id(e)] for e in simple_rqs[pathreq.request_id]]
        pths2 = []
        for p in simple_rqs[pathreq.request_id] :
            print(f'{p} disjoint de {tab2[id(p)]}')
                        
        i = pths.index(min(pths)) 
        print(simple_rqs[pathreq.request_id][i])
        print(pths)
        print(pths2)


def isdisjoint(p1,p2) :
    # returns 0 if disjoint
    # TODO add reverse direction in the test
    edge1 = list(pairwise(p1))
    edge2 = list(pairwise(p2))
    edge3 = list(pairwise(reversed(p2)))
    for e in edge1 :
        if (e in edge2) | (e in edge3) :
            return 1
    return 0


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

    rqs = requests_from_json(data, equipment)
    print('The following services have been requested:')
    print(rqs)
    pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)
    toto = compute_path_dsjctn(network, equipment, rqs,dsjn)
    
    #TODO write results

    header = ['demand','snr@bandwidth','snr@0.1nm','Receiver minOSNR']
    data = []
    data.append(header)
    for i, p in enumerate(pths):
        if p:
            line = [f'{rqs[i].source} to {rqs[i].destination} : ', f'{round(mean(p[-1].snr),2)}',\
                f'{round(mean(p[-1].snr+lin2db(rqs[i].baud_rate/(12.5e9))),2)}',\
                f'{rqs[i].OSNR}']
        else:
            line = [f'no path from {rqs[i].source} to {rqs[i].destination} ']
        data.append(line)

    col_width = max(len(word) for row in data for word in row)   # padding
    for row in data:
        print(''.join(word.ljust(col_width) for word in row))



    if args.output :
        result = []
        for p in pths:
            result.append(Result_element(rqs[pths.index(p)],p))
        with open(args.output, 'w') as f:
            f.write(dumps(path_result_json(result), indent=2))
            fnamecsv = next(s for s in args.output.split('.')) + '.csv'
            with open(fnamecsv,"w") as fcsv :
                jsontocsv(path_result_json(result),equipment,fcsv)
