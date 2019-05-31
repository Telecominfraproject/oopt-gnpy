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

from sys import exit
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from networkx import (dijkstra_path, NetworkXNoPath, all_simple_paths,shortest_path_length)
from networkx.utils import pairwise 
from numpy import mean
from gnpy.core.service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power
from copy import copy, deepcopy
from csv import writer
from math import ceil

logger = getLogger(__name__)


RequestParams = namedtuple('RequestParams','request_id source destination trx_type'+
' trx_mode nodes_list loose_list spacing power nb_channel f_min f_max format baud_rate OSNR bit_rate roll_off tx_osnr min_spacing cost path_bandwidth')
DisjunctionParams = namedtuple('DisjunctionParams','disjunction_id relaxable link_diverse node_diverse disjunctions_req')

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
        self.f_min       = params.f_min
        self.f_max       = params.f_max
        self.format     = params.format
        self.OSNR       = params.OSNR
        self.bit_rate   = params.bit_rate
        self.roll_off   = params.roll_off
        self.tx_osnr    = params.tx_osnr
        self.min_spacing = params.min_spacing
        self.cost       = params.cost
        self.path_bandwidth  = params.path_bandwidth

    def __str__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source:       {self.source}',
                            f'destination:  {self.destination}'])
    def __repr__(self):
        if self.baud_rate is not None:
            temp = self.baud_rate * 1e-9
            temp2 = self.bit_rate * 1e-9
        else:
            temp = self.baud_rate
            temp2 = self.bit_rate

        return '\n\t'.join([  f'{type(self).__name__} {self.request_id}',
                            f'source: \t{self.source}',
                            f'destination:\t{self.destination}',
                            f'trx type:\t{self.tsp}',
                            f'trx mode:\t{self.tsp_mode}',
                            f'baud_rate:\t{temp} Gbaud',
                            f'bit_rate:\t{temp2} Gb/s',
                            f'spacing:\t{self.spacing * 1e-9} GHz',
                            f'power:  \t{round(lin2db(self.power)+30,2)} dBm',
                            f'nb channels: \t{self.nb_channel}',
                            f'path_bandwidth: \t{round(self.path_bandwidth * 1e-9,2)} Gbit/s',
                            f'nodes-list:\t{self.nodes_list}',
                            f'loose-list:\t{self.loose_list}'
                            '\n'])
class Disjunction:
    def __init__(self, *args, **params):
        params = DisjunctionParams(**params)
        self.disjunction_id = params.disjunction_id
        self.relaxable = params.relaxable
        self.link_diverse = params.link_diverse
        self.node_diverse = params.node_diverse
        self.disjunctions_req = params.disjunctions_req
        
    def __str__(self):
        return '\n\t'.join([f'relaxable:    {self.relaxable}',
                            f'link-diverse:       {self.link_diverse}',
                            f'node-diverse:  {self.node_diverse}',
                            f'request-id-numbers: {self.disjunctions_req}']
                            )
    def __repr__(self):
        return '\n\t'.join([  f'{type(self).__name__} {self.disjunction_id}',
                            f'relaxable:    {self.relaxable}',
                            f'link-diverse:       {self.link_diverse}',
                            f'node-diverse:  {self.node_diverse}',
                            f'request-id-numbers: {self.disjunctions_req}'
                            '\n'])

class Result_element(Element):
    def __init__(self,path_request,computed_path):
        self.path_id = path_request.request_id
        self.path_request = path_request
        self.computed_path = computed_path
    uid = property(lambda self: repr(self))
    @property
    def pathresult(self):
        if not self.computed_path:
            return {
                   'response-id': self.path_id,
                   'no-path': "Response without path information, due to failure performing the path computation"
                   }
        else:
            index = 0
            pro_list = [] 
            for n in self.computed_path :
                temp = {
                    'path-route-object': {
                        'index': index,
                        'num-unnum-hop': {
                            'node-id': n.uid,
                            'link-tp-id': n.uid,
                            # TODO change index in order to insert transponder attribute
                            }
                        }
                    }
                pro_list.append(temp) 
                index += 1
                if isinstance(n, Transceiver) :
                    temp = {
                        'path-route-object': {
                            'index': index,
                            'transponder' : {
                               'transponder-type' : self.path_request.tsp,
                               'transponder-mode' : self.path_request.tsp_mode,
                                }
                            }
                        }
                    pro_list.append(temp) 
                    index += 1

            response = {
                'response-id': self.path_id,
                'path-properties':{
                    'path-metric': [
                        {
                            'metric-type': 'SNR-bandwidth',
                            'accumulative-value': round(mean(self.computed_path[-1].snr), 2)
                        },
                        {
                            'metric-type': 'SNR-0.1nm',
                            'accumulative-value': round(mean(self.computed_path[-1]. snr + \
                                                            lin2db(self.path_request.baud_rate/12.5e9)), 2)
                        },
                        {
                            'metric-type': 'OSNR-bandwidth',
                            'accumulative-value': round(mean(self.computed_path[-1].osnr_ase), 2)
                        },
                        {
                            'metric-type': 'OSNR-0.1nm',
                            'accumulative-value': round(mean(self.computed_path[-1].osnr_ase_01nm), 2)
                        },
                        {
                            'metric-type': 'reference_power',
                            'accumulative-value': self.path_request.power
                        },
                        {
                            'metric-type': 'path_bandwidth',
                            'accumulative-value': self.path_request.path_bandwidth
                        }
                        ],
                    'path-route-objects': pro_list
                    }
                }
        return response

    @property
    def json(self):
        return self.pathresult

def compute_constrained_path(network, req):
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)]
    roadm = [n for n in network.nodes() if isinstance(n, Roadm)]
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)]
    anytypenode = [n for n in network.nodes()]

    source = next(el for el in trx if el.uid == req.source)

    # This method ensures that the constraint can be satisfied without loops
    # except when it is not possible : eg if constraints makes a loop
    # It requires that the source, dest and nodes are correct (no error in the names)
    destination = next(el for el in trx if el.uid == req.destination)
    nodes_list = []
    for n in req.nodes_list :
        # for debug excel print(n)
        nodes_list.append(next(el for el in anytypenode if el.uid == n))
    # nodes_list contains at least the destination
    if nodes_list is None :
        msg = f'Request {req.request_id} problem in the constitution of nodes_list: should at least include destination'
        logger.critical(msg)
        exit()
    if req.nodes_list[-1] != req.destination:
        msg = f'Request {req.request_id} malformed list of nodes: last node should be destination trx'
        logger.critical(msg)
        exit()

    if len(nodes_list) == 1  :
        try :
            total_path = dijkstra_path(network, source, destination, weight = 'weight')
            # print('checking edges length is correct')
            # print(shortest_path_length(network,source,destination))
            # print(shortest_path_length(network,source,destination,weight ='weight'))
            # s = total_path[0]
            # for e in total_path[1:]:
            #     print(s.uid)
            #     print(network.get_edge_data(s,e))
            #     s = e
        except NetworkXNoPath:
            msg = f'\x1b[1;33;40m'+f'Request {req.request_id} could not find a path from {source.uid} to node : {destination.uid} in network topology'+ '\x1b[0m'
            logger.critical(msg)
            print(msg)
            total_path = []        
    else : 
        all_simp_pths = list(all_simple_paths(network,source=source,\
            target=destination, cutoff=120))
        candidate = []
        for p in all_simp_pths :
            if ispart(nodes_list, p) :
                # print(f'selection{[el.uid for el in p if el in roadm]}')
                candidate.append(p)
        # select the shortest path (in nb of hops) -> changed to shortest path in km length
        if len(candidate)>0 :
            # candidate.sort(key=lambda x: len(x))
            candidate.sort(key=lambda x: sum(network.get_edge_data(x[i],x[i+1])['weight'] for i in range(len(x)-2)))
            total_path = candidate[0]
        else:
            if req.loose_list[req.nodes_list.index(n)] == 'loose':
                print(f'\x1b[1;33;40m'+f'Request {req.request_id} could not find a path crossing {nodes_list} in network topology'+ '\x1b[0m')
                print(f'constraint ignored')
                total_path = dijkstra_path(network, source, destination, weight = 'weight')
            else:
                msg = f'\x1b[1;33;40m'+f'Request {req.request_id} could not find a path crossing {nodes_list}.\nNo path computed'+ '\x1b[0m'
                logger.critical(msg)
                print(msg)
                total_path = []

    # obsolete method: this does not guaranty to avoid loops or correct results
    # Here is the demonstration :
    #         1     1
    # eg    a----b-----c
    #       |1   |0.5  |1
    #       e----f--h--g
    #         1  0.5 0.5
    # if I have to compute a to g with constraint f-c
    # result will be a concatenation of: a-b-f and f-b-c and c-g
    # which means a loop. 
    # if to avoid loops I iteratively suppress edges of the segmenst in the topo
    # segment 1 = a-b-f
    #               1
    # eg    a    b-----c
    #       |1         |1
    #       e----f--h--g
    #         1  0.5 0.5
    # then 
    # segment 2 = f-h-g-c
    #               1
    # eg    a    b-----c
    #       |1          
    #       e----f  h  g
    #         1  
    # then there is no more path to g destination
    #
    # 
    # total_path = [source]

    # for n in req.nodes_list:
    #     try :
    #         node = next(el for el in trx if el.uid == n)
    #     except StopIteration:
    #         try:
    #             node = next(el for el in anytypenode if el.uid == n)
    #         except StopIteration:
    #             try:
    #                 # TODO this test is not giving good results: full name of the 
    #                 # amp is required to avoid ambiguity on the direction
    #                 node = next(el for el in anytypenode 
    #                     if n in el.uid)
    #             except StopIteration:
    #                 msg = f'could not find node : {n} in network topology: \
    #                     not a trx, roadm, edfa, fiber or fused element'
    #                 logger.critical(msg)
    #                 raise ValueError(msg)
    #     # extend path list without repeating source -> skip first element in the list
    #     try:
    #         # to avoid looping back: use an alternate graph were current path edges and vertex are suppressed

    #         total_path.extend(dijkstra_path(network, source, node)[1:])
    #         source = node
    #     except NetworkXNoPath:
    #         if req.loose_list[req.nodes_list.index(n)] == 'loose':
    #             print(f'could not find a path from {source.uid} to loose node : {n} in network topology')
    #             print(f'node  {n} is skipped')
    #         else:
    #             msg = f'could not find a path from {source.uid} to node : {n} in network topology'
    #             logger.critical(msg)
    #             print(msg)
    #             total_path = []

    return total_path

def propagate(path, req, equipment, show=False):
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate,
        req.power, req.spacing)
    for el in path:
        si = el(si)
        if show :
            print(el)
    path[-1].update_snr(req.tx_osnr, equipment['Roadm']['default'].add_drop_osnr)
    return path

def propagate2(path, req, equipment, show=False):
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate,
        req.power, req.spacing)
    infos = {}
    for el in path:
        before_si = si
        after_si  = si = el(si)
        infos[el] = before_si, after_si
        if show :
            print(el)
    path[-1].update_snr(req.tx_osnr, equipment['Roadm']['default'].add_drop_osnr)
    return infos

def propagate_and_optimize_mode(path, req, equipment, show=False):
    # if mode is unknown : loops on the modes starting from the highest baudrate fiting in the
    # step 1: create an ordered list of modes based on baudrate
    baudrate_to_explore = list(set([m['baud_rate'] for m in equipment['Transceiver'][req.tsp].mode 
        if float(m['min_spacing'])<= req.spacing]))  
        # TODO be carefull on limits cases if spacing very close to req spacing eg 50.001 50.000
    baudrate_to_explore = sorted(baudrate_to_explore, reverse=True)
    if baudrate_to_explore : 
        # at least 1 baudrate can be tested wrt spacing
        for b in baudrate_to_explore :
            modes_to_explore = [m for m in equipment['Transceiver'][req.tsp].mode 
                if m['baud_rate'] == b and float(m['min_spacing'])<= req.spacing]
            modes_to_explore = sorted(modes_to_explore, 
                key = lambda x: x['bit_rate'], reverse=True)
            # print(modes_to_explore)
            # step2 : computes propagation for each baudrate: stop and select the first that passes
            found_a_feasible_mode = False
            # TODO : the case of roll of is not included: for now use SI one
            # TODO : if the loop in mode optimization does not have a feasible path, then bugs
            si = create_input_spectral_information(
            req.f_min, req.f_max, equipment['SI']['default'].roll_off,
            b, req.power, req.spacing)
            for el in path:
                si = el(si)
                if show:
                    print(el)
            for m in modes_to_explore :
                if path[-1].snr is not None:
                    path[-1].update_snr(m['tx_osnr'], equipment['Roadm']['default'].add_drop_osnr)
                    if round(min(path[-1].snr+lin2db(b/(12.5e9))),2) > m['OSNR'] :
                        found_a_feasible_mode = True
                        return path, m
                else:  
                    return [], None
        # only get to this point if no baudrate/mode satisfies OSNR requirement

        # returns the last propagated path and mode
        msg = f'\tWarning! Request {req.request_id}: no mode satisfies path SNR requirement.\n'
        print(msg)
        logger.info(msg)
        return [],None
    else :
    #  no baudrate satisfying spacing
        msg = f'\tWarning! Request {req.request_id}: no baudrate satisfies spacing requirement.\n'
        print(msg)
        logger.info(msg)
        return [], None


def jsontocsv(json_data,equipment,fileout):
    # read json path result file in accordance with:
    # Yang model for requesting Path Computation
    # draft-ietf-teas-yang-path-computation-01.txt.
    # and write results in an CSV file

    mywriter = writer(fileout)
    mywriter.writerow(('response-id','source','destination','path_bandwidth','Pass?',\
        'nb of tsp pairs','total cost','transponder-type','transponder-mode',\
        'OSNR-0.1nm','SNR-0.1nm','SNR-bandwidth','baud rate (Gbaud)',\
        'input power (dBm)','path'))
    tspjsondata = equipment['Transceiver']
    #print(tspjsondata)

    for pth_el in json_data['response']:
        path_id     = pth_el['response-id']
        try:
            if pth_el['no-path'] :
                isok = False
                nb_tsp = 0
                pthbdbw = round(path_bandwidth*1e-9,2)
                rosnr = ''
                rsnr = ''
                rsnrb = ''
                br = ''
                pw = ''
                total_cost = ''
                pth = ''
        except KeyError:
            source      = pth_el['path-properties']['path-route-objects'][0]\
            ['path-route-object']['num-unnum-hop']['node-id']
            destination = pth_el['path-properties']['path-route-objects'][-2]\
            ['path-route-object']['num-unnum-hop']['node-id']
            # selects only roadm nodes
            temp = []
            for e in pth_el['path-properties']['path-route-objects'] :
                try :
                    temp .append(e['path-route-object']['num-unnum-hop']['node-id'])
                except KeyError:
                    pass
            pth        = ' | '.join(temp)

            [tsp,mode] = [pth_el['path-properties']['path-route-objects'][1]\
            ['path-route-object']['transponder']['transponder-type'],\
            pth_el['path-properties']['path-route-objects'][1]\
            ['path-route-object']['transponder']['transponder-mode'] ]

            # find the min  acceptable OSNR, baud rate from the eqpt library based on tsp (tupe) and mode (format)
            # loading equipment already tests the existence of tsp type and mode:
            if mode !='not feasible with this transponder' :
                [minosnr, baud_rate, bit_rate, cost] = next([m['OSNR'] , m['baud_rate'] , m['bit_rate'], m['cost']]  
                    for m in equipment['Transceiver'][tsp].mode if  m['format']==mode)
            # else:
            #     [minosnr, baud_rate, bit_rate] = ['','','','']
            output_snr = next(e['accumulative-value'] 
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'SNR-0.1nm')
            output_snrbandwidth = next(e['accumulative-value']
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'SNR-bandwidth')
            output_osnr = next(e['accumulative-value']
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'OSNR-0.1nm')
            output_osnrbandwidth = next(e['accumulative-value']
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'OSNR-bandwidth')
            power = next(e['accumulative-value']
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'reference_power')
            path_bandwidth = next(e['accumulative-value']
                for e in pth_el['path-properties']['path-metric'] if e['metric-type'] == 'path_bandwidth')
            if isinstance(output_snr, str):
                isok = False
                nb_tsp = 0
                pthbdbw = round(path_bandwidth*1e-9,2)
                rosnr = ''
                rsnr = ''
                rsnrb = ''
                br = ''
                pw = ''
                total_cost = ''
            else:
                isok   = output_snr >= minosnr
                nb_tsp = ceil(path_bandwidth / bit_rate)
                pthbdbw = round(path_bandwidth*1e-9,2)
                rosnr  = round(output_osnr,2)
                rsnr   = round(output_snr,2)
                rsnrb  = round(output_snrbandwidth,2)
                br     = round(baud_rate*1e-9,2) 
                pw     = round(lin2db(power)+30,2)
                total_cost = nb_tsp * cost

        mywriter.writerow((path_id,
            source,
            destination,
            pthbdbw,
            isok,
            nb_tsp,
            total_cost,
            tsp,
            mode,
            rosnr,
            rsnr,
            rsnrb,
            br,
            pw,
            pth
            ))


def compute_path_dsjctn(network, equipment, pathreqlist, disjunctions_list):
    # pathreqlist is a list of Path_request objects
    # disjunctions_list a list of Disjunction objects
    
    # given a network, a list of requests with the set of disjunction features between
    # request, the function computes the set of path satisfying : first the disjunction
    # constraint and second the routing constraint if the request include an explicit 
    # set of elements to pass through.
    # the algorithm used allows to specify disjunction for demands not sharing source or
    # destination.
    # a request might be declared as disjoint from several requests
    # it is a iterative process:
    # first computes a list of all shortest path (this may add computation time)
    # second elaborate the set of path solution for each synchronization vector
    # third select only the candidates that satisfy all synchronization vectors they belong to
    # fourth apply route constraints : remove candidate path that do not satisfy the constraint
    # fifth select the first candidate among the set of candidates.
    # the example network used in comments has been added to the set of data tests files

    # define the list to be returned
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

    # step 1
    # for each remaining request compute a set of simple path
    allpaths = {}
    rqs = {}
    simple_rqs = {}
    simple_rqs_reversed = {}
    for pathreq in pathreqlist_disjt :
        all_simp_pths = list(all_simple_paths(network,\
            source=next(el for el in network.nodes() if el.uid == pathreq.source),\
            target=next(el for el in network.nodes() if el.uid == pathreq.destination),\
            cutoff=80))
        # sort them in km length instead of hop
        # all_simp_pths = sorted(all_simp_pths, key=lambda path: len(path))
        all_simp_pths = sorted(all_simp_pths, key=lambda \
            x: sum(network.get_edge_data(x[i],x[i+1])['weight'] for i in range(len(x)-2)))
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
            # print('coucou')
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
    # note that this function may not give an exact result in case of multiple
    # links between two adjacent nodes. 
    # TODO add some indication on elements to indicate from which other they 
    # are the reversed direction
    reversed_roadm_path = list(reversed([e for e in p if isinstance (e,Roadm)]))
    source = p[-1]
    destination = p[0]
    total_path = [source]
    for node in reversed_roadm_path :
        total_path.extend(dijkstra_path(network, source, node, weight = 'weight')[1:])
        source = node
    total_path.append(destination)
    return total_path

def ispart(a,b) :
    # the functions takes two paths a and b and retrns True
    # if all a elements are part of b and in the same order
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

def compare_reqs(req1,req2,disjlist) :
    dis1 = [d for d in disjlist if req1.request_id in d.disjunctions_req]
    dis2 = [d for d in disjlist if req2.request_id in d.disjunctions_req]
    same_disj = False
    if dis1 and dis2 :
        temp1 = []
        for d in dis1:
            temp1.extend(d.disjunctions_req)
            temp1.remove(req1.request_id)
        temp2 = []
        for d in dis2:
            temp2.extend(d.disjunctions_req)
            temp2.remove(req2.request_id)
        if set(temp1) == set(temp2) :
            same_disj = True
    elif not dis2 and not dis1:
        same_disj = True

    if req1.source     == req2.source and \
        req1.destination == req2.destination and  \
        req1.tsp        == req2.tsp and \
        req1.tsp_mode   == req2.tsp_mode and \
        req1.baud_rate  == req2.baud_rate and \
        req1.nodes_list == req2.nodes_list and \
        req1.loose_list == req2.loose_list and \
        req1.spacing    == req2.spacing and \
        req1.power      == req2.power and \
        req1.nb_channel == req2.nb_channel and \
        req1.f_min  == req2.f_min and \
        req1.f_max  == req2.f_max and \
        req1.format     == req2.format and \
        req1.OSNR       == req2.OSNR and \
        req1.roll_off   == req2.roll_off and \
        same_disj :
        return True
    else:
        return False

def requests_aggregation(pathreqlist,disjlist) :
    # this function aggregates requests so that if several requests
    # exist between same source and destination and with same transponder type
    # todo maybe add conditions on mode ??, spacing ...
    # currently if undefined takes the default values
    local_list = pathreqlist.copy()
    for req in pathreqlist:
        for r in local_list : 
            if  req.request_id != r.request_id and compare_reqs(req, r, disjlist):
                # aggregate
                r.path_bandwidth += req.path_bandwidth
                temp_r_id = r.request_id
                r.request_id = ' | '.join((r.request_id,req.request_id))
                # remove request from list
                local_list.remove(req)
                # todo change also disjunction req with new demand

                for d in disjlist :
                    if req.request_id in d.disjunctions_req :
                        d.disjunctions_req.remove(req.request_id)
                        d.disjunctions_req.append(r.request_id)
                for d in disjlist :        
                    if temp_r_id in d.disjunctions_req :
                        disjlist.remove(d)
                break
    return local_list, disjlist
