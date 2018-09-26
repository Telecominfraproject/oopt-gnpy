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
        params['disjunction_id'] = snc['synchronization-id']
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

def compute_path_2(network, equipment, pathreqlist, pathlist):
    
    path_res_list = []
    # # Build the network once using the default power defined in SI in eqpt config
    # # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # # spacing, f_min and f_max 
    # p_db = equipment['SI']['default'].power_dbm
    
    # p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
    #     equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    # build_network(network, equipment, p_db, p_total_db)
    # TODO : get the designed power to set it when it is not an input
    # pathreq.power to be adapted
    for i,pathreq in enumerate(pathreqlist):

        p_db = lin2db(pathreq.power*1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}') #adding first node to be clearer on the output

        total_path = pathlist[i]
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

def correct_route_list(network, pathreqlist):

    roadms = [n.uid for n in network.nodes() if isinstance(n, Roadm)]
    for pathreq in pathreqlist:
        for i,n_id in enumerate(pathreq.nodes_list):
            # replace possibly wrong name with a formated roadm name
            if n_id not in [n.uid for n in network.nodes()]:
                nodes_suggestion = [uid for uid in roadms \
                    if n_id.lower() in uid.lower()]
                if len(nodes_suggestion)>0 :
                    new_n = nodes_suggestion[0]
                    print(f'invalid route node specified:\
                    \n\'{n_id}\', replaced with \'{new_n}\'')
                    pathreq.nodes_list[i] = new_n
                else:
                    print(f'invalid route node specified \'{n_id}\', could not use it as constraint, skipped!')
                    pathreq.nodes_list.remove(n_id)
                    pathreq.loose_list.pop(i)
        # remove endpoints from this list in case they were added by the user in the xls or json files
    return pathreqlist

def compute_path_dsjctn(network, equipment, pathreqlist, disjunctions_list):
    
    # need to return list
    path_res_list = []

    # all disjctn must be computed at once together to avoid blocking
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
    # even they are not the shortest paths

    # build the list of pathreqlist elements not concerned by disjunction
    global_disjunctions_list = [e for d in disjunctions_list for e in d.disjunctions_req ]
    pathreqlist_simple = [e for e in pathreqlist if e.request_id not in global_disjunctions_list]
    pathreqlist_disjt = [e for e in pathreqlist if e.request_id in global_disjunctions_list]

    # use a mirror class to record path and the corresponding requests
    class Pth:
        def __init__(self, req, pth, simplepth):
            self.req = req
            self.pth = pth
            self.simplepth = simplepth

    # build a conflict table where each path has a count for 
    # conflicts with the paths from the requests to be disjoint
    # step 1
    # for each remaining request compute a set of simple path
    allpaths = {}
    rqs = {}
    simple_rqs = {}
    simple_rqs_reversed = {}
    for pathreq in pathreqlist_disjt :
        all_simp_pths = list(all_simple_paths(network,\
            source=next(el for el in network.nodes() if el.uid == pathreq.source),\
            target=next(el for el in network.nodes() if el.uid == pathreq.destination)))
        # sort them
        all_simp_pths = sorted(all_simp_pths, key=lambda path: len(path))
        # reversed direction paths required to check disjunction on both direction
        all_simp_pths_reversed = []
        for pth in all_simp_pths:
            all_simp_pths_reversed.append(find_reversed_path(pth,network))
        rqs[pathreq.request_id] = all_simp_pths 
        temp =[]
        for p in all_simp_pths :
            # build a short list representing each roadm+direction with the first item
            # start enumeration at 1 to avoid Trx in the list
            s = [e.uid for i,e in enumerate(p[1:-1]) \
                if (isinstance(e,Roadm) | (isinstance(p[i],Roadm) ))] 
            temp.append(s)
            # id(s) is unique even if path is the same: two objects with same 
            # path have two different ids
            allpaths[id(s)] = Pth(pathreq,p,s)
        simple_rqs[pathreq.request_id] = temp
        temp =[]
        for p in all_simp_pths_reversed :
            # build a short list representing each roadm+direction with the first item
            # start enumeration at 1 to avoid Trx in the list
            temp.append([e.uid for i,e in enumerate(p[1:-1]) \
                if (isinstance(e,Roadm) | (isinstance(p[i],Roadm) ))] )
        simple_rqs_reversed[pathreq.request_id] = temp
    # step 2 
    # for each set of requests that need to be disjoint
    # select the disjoint path combination

    candidates = {}
    for d in disjunctions_list :
        dlist = d.disjunctions_req.copy()
        # each line of dpath is one combination of path that satisfies disjunction
        dpath = []
        for i,p in enumerate(simple_rqs[dlist[0]]):
            dpath.append([p])
            # allpaths[id(p)].d_id = d.disjunction_id
        # in each loop, dpath is updated with a path for rq that satisfies 
        # disjunction with each path in dpath
        # for example, assume set of requests in the vector (disjunction_list) is  {rq1,rq2, rq3}
        # rq1  p1: abfhg
        #      p2: aefhg
        #      p3: abcg
        # rq2  p8: bf
        # rq3  p4: abcgh
        #      p6: aefh
        #      p7: abfh
        # initiate with rq1
        #  dpath = [[p1]
        #           [p2]
        #           [p3]]
        #  after first loop:
        #  dpath = [[p1 p8]
        #           [p3 p8]]
        #  since p2 and p8 are not disjoint
        #  after second loop:
        #  dpath = [ p3 p8 p6 ]
        #  since p1 and p4 are not disjoint 
        #        p1 and p7 are not disjoint
        #        p3 and p4 are not disjoint
        #        p3 and p7 are not disjoint

        for e1 in dlist[1:] :
            temp = []
            for j,p1 in enumerate(simple_rqs[e1]):
                # allpaths[id(p1)].d_id = d.disjunction_id
                # can use index j in simple_rqs_reversed because index 
                # of direct and reversed paths have been kept identical
                p1_reversed = simple_rqs_reversed[e1][j]
                # print(p1_reversed)
                # print('\n\n')
                for k,c in enumerate(dpath) :
                    # print(f' c: \t{c}')
                    temp2 = c.copy()
                    all_disjoint = 0
                    for p in c :
                        all_disjoint += isdisjoint(p1,p)+ isdisjoint(p1_reversed,p)
                    if all_disjoint ==0:
                        temp2.append(p1)
                        temp.append(temp2)
                            # print(f' coucou {e1}: \t{temp}')
            dpath = temp
        # print(dpath)
        candidates[d.disjunction_id] = dpath

    # for i in disjunctions_list  :
    #     print(f'\n{candidates[i.disjunction_id]}')

    # step 3
    # now for each request, select the path that satisfies all disjunctions
    # path must be in candidates[id] for all concerned ids
    # for example, assume set of sync vectors (disjunction groups) is
    #   s1 = {rq1 rq2}   s2 = {rq1 rq3}
    #   candidate[s1] = [[p1 p8]
    #                    [p3 p8]]
    #   candidate[s2] = [[p3 p6]]
    #   for rq1 p3 should be preferred


    for pathreq in pathreqlist_disjt:
        concerned_d_id = [d.disjunction_id for d in disjunctions_list if pathreq.request_id in d.disjunctions_req]
        # for each set of solution, verify that the same path is used for the same request
        candidate_paths = simple_rqs[pathreq.request_id]
        # print('coucou')
        # print(pathreq.request_id)
        for p in candidate_paths :
            iscandidate = 0
            for sol in concerned_d_id :
                test = 1
                # for each solution test if p is part of the solution
                # if yes, then p can remain a candidate
                for i,m in enumerate(candidates[sol]) :
                    if p in m:
                        if allpaths[id(m[m.index(p)])].req.request_id == pathreq.request_id :
                            test = 0
                            break
                iscandidate += test
            if iscandidate != 0:
                for l in concerned_d_id :
                    for m in candidates[l] :
                        if p in m :
                            candidates[l].remove(m)

#    for i in disjunctions_list  :
#        print(i.disjunction_id)
#        print(f'\n{candidates[i.disjunction_id]}')

    # step 4 apply route constraints : remove candidate path that do not satisfy the constraint
    # only in  the case of disjounction: the simple path is processed in request.compute_constrained_path
    # TODO : keep a version without the loose constraint
    for d in disjunctions_list  :
        temp = []
        for j,sol in enumerate(candidates[d.disjunction_id]) :
            testispartok = True
            for i,p in enumerate(sol) :
                # print(f'test {allpaths[id(p)].req.request_id}')
                # print(f'length of route {len(allpaths[id(p)].req.nodes_list)}')
                if allpaths[id(p)].req.nodes_list :
                    # if p does not containt the ordered list node, remove sol from the candidate
                    # except if this was the last solution: then check if the constraint is loose or not
                    if not ispart(allpaths[id(p)].req.nodes_list, p) : 
                        # print(f'nb of solutions {len(temp)}')
                        if j < len(candidates[d.disjunction_id])-1 :
                            msg = f'removing {sol}'
                            logger.info(msg)
                            testispartok = False
                            #break
                        else:
                            if 'loose' in allpaths[id(p)].req.loose_list:
                                logger.info(f'Could not apply route constraint'+
                                    f'{allpaths[id(p)].req.nodes_list} on request {allpaths[id(p)].req.request_id}')
                            else :
                                logger.info(f'removing last solution from candidate paths\n{sol}')
                                testispartok = False
            if testispartok :
                temp.append(sol)
        candidates[d.disjunction_id] = temp

    # step 5 select the first combination that works
    pathreslist_disjoint = {}
    for d in disjunctions_list  :
        test_sol = True
        while test_sol:
            if candidates[d.disjunction_id] :
                for p in candidates[d.disjunction_id][0]:
                    if allpaths[id(p)].req in pathreqlist_disjt: 
                        # print(f'selected path :{p} for req {allpaths[id(p)].req.request_id}')
                        pathreslist_disjoint[allpaths[id(p)].req] = allpaths[id(p)].pth
                        pathreqlist_disjt.remove(allpaths[id(p)].req)
                        candidates = remove_candidate(candidates, allpaths, allpaths[id(p)].req, p)
                        test_sol = False
            else:
                msg = f'No disjoint path found with added constraint'
                logger.critical(msg)
                print(f'{msg}\nComputation stopped.')
                # TODO in this case: replay step 5  with the candidate without constraints
                exit()
    
    # for i in disjunctions_list  :
    #     print(i.disjunction_id)
    #     print(f'\n{candidates[i.disjunction_id]}')

    # list the results in the same order as initial pathreqlist        
    for req in pathreqlist :
        req.nodes_list.append(req.destination)
        # we assume that the destination is a strict constraint
        req.loose_list.append('strict')
        if req in pathreqlist_simple:
            path_res_list.append(compute_constrained_path(network, req))
        else:
            path_res_list.append(pathreslist_disjoint[req])
    return path_res_list

def isdisjoint(p1,p2) :
    # returns 0 if disjoint
    edge1 = list(pairwise(p1))
    edge2 = list(pairwise(p2))
    for e in edge1 :
        if e in edge2 :
            return 1
    return 0

def find_reversed_path(p,network) :
    # select of intermediate roadms and find the path between them
    reversed_roadm_path = list(reversed([e for e in p if isinstance (e,Roadm)]))
    source = p[-1]
    destination = p[0]
    total_path = [source]
    for node in reversed_roadm_path :
        total_path.extend(dijkstra_path(network, source, node)[1:])
        source = node
    total_path.append(destination)
    return total_path

def ispart(a,b) :
    j = 0
    for i, el in enumerate(a):
        if el in b :
            if b.index(el) >= j :
                j = b.index(el)
            else: 
                return False
        else:
            return False
    return True

def remove_candidate(candidates, allpaths, rq, pth) :
    # print(f'coucou {rq.request_id}')
    for key, candidate  in candidates.items() :
        temp = candidate.copy()
        for i,sol in enumerate(candidate) :
            for p in sol :
                if allpaths[id(p)].req.request_id == rq.request_id :
                    if id(p) != id(pth) :
                        temp.remove(sol)
                        break
        candidates[key] = temp
    return candidates

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

    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)

    rqs = requests_from_json(data, equipment)
    rqs = correct_route_list(network, rqs)
    print('The following services have been requested:')
    print(rqs)
    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)
    pths = compute_path_dsjctn(network, equipment, rqs,dsjn)
    propagatedpths = compute_path_2(network, equipment, rqs, pths)

    
    header = ['demand','snr@bandwidth','snr@0.1nm','Receiver minOSNR']
    data = []
    data.append(header)
    for i, p in enumerate(propagatedpths):
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
