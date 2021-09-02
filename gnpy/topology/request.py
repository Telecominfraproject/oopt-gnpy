#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.topology.request
=====================

This module contains path request functionality.

This functionality allows the user to provide a JSON request
file in accordance with a Yang model for requesting path
computations and returns path results in terms of path
and feasibility

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from collections import namedtuple, OrderedDict
from logging import getLogger
from networkx import (dijkstra_path, NetworkXNoPath,
                      all_simple_paths, shortest_simple_paths)
from networkx.utils import pairwise
from numpy import mean
from gnpy.core.elements import Transceiver, Roadm, Regenerator
from gnpy.core.equipment import trx_mode_params
from gnpy.core.utils import lin2db, unique_ordered
from gnpy.core.info import create_input_spectral_information
from gnpy.core.exceptions import ServiceError, DisjunctionError
import gnpy.core.ansi_escapes as ansi_escapes
from copy import deepcopy
from csv import writer
from math import ceil

LOGGER = getLogger(__name__)

RequestParams = namedtuple('RequestParams', 'request_id source destination bidir trx_type' +
                           ' trx_mode nodes_list loose_list regen_list spacing power nb_channel f_min' +
                           ' f_max format baud_rate OSNR bit_rate roll_off tx_osnr' +
                           ' min_spacing cost path_bandwidth effective_freq_slot regen_preference')
DisjunctionParams = namedtuple('DisjunctionParams', 'disjunction_id relaxable link' +
                               '_diverse node_diverse disjunctions_req')


class PathRequest:
    """ the class that contains all attributes related to a request
    """

    def __init__(self, *args, **params):
        if 'regen_preference' not in params.keys():
            params['regen_preference'] = []
        params = RequestParams(**params)
        self.request_id = params.request_id
        self.source = params.source
        self.destination = params.destination
        self.bidir = params.bidir
        self.trx_type = params.trx_type
        self.trx_mode = params.trx_mode
        self.baud_rate = params.baud_rate
        self.nodes_list = params.nodes_list
        self.loose_list = params.loose_list
        self.regen_list = params.regen_list
        self.spacing = params.spacing
        self.power = params.power
        self.nb_channel = params.nb_channel
        self.f_min = params.f_min
        self.f_max = params.f_max
        self.format = params.format
        self.OSNR = params.OSNR
        self.bit_rate = params.bit_rate
        self.roll_off = params.roll_off
        self.tx_osnr = params.tx_osnr
        self.min_spacing = params.min_spacing
        self.cost = params.cost
        self.path_bandwidth = params.path_bandwidth
        if params.effective_freq_slot is not None:
            self.N = params.effective_freq_slot['N']
            self.M = params.effective_freq_slot['M']
        self.regen_preference = params.regen_preference

    def __str__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.request_id}',
                            f'source:       {self.source}',
                            f'destination:  {self.destination}'])

    @property
    def regen_list_print(self):
        """ Build a pretty string to summarize regenerator list in request
        """
        pretty_regen = ''
        for regen in self.regen_list:
            baud_rate = f'{regen["baud_rate"]*1e-9}  Gbaud' if regen["baud_rate"] is not None else 'Undetermined'
            if 'trx_type' in regen.keys() and  'trx_mode' in regen.keys():
                mode = f'{regen["trx_mode"]}' if regen["trx_mode"] is not None else 'Undetermined'
                pretty_regen = pretty_regen + f'{regen["regen_uid"]} {regen["trx_type"]} {mode}\n\t\t' + \
                               f'\t\tbaud rate: {baud_rate}\n\t\t' + \
                               f'\t\tspacing: {regen["spacing"]*1e-9} GHz\n\t\t\t'
            elif 'trx_type' in regen.keys():
                pretty_regen = pretty_regen + f'{regen["regen_uid"]} {regen["trx_type"]}\n\t\t' + \
                               f'\t\tbaud rate: {baud_rate} Gbaud\n\t\t' + \
                               f'\t\tspacing: {regen["spacing"]*1e-9} GHz\n\t\t\t'
            else:
                pretty_regen = pretty_regen + f'{regen["regen_uid"]}\n\t\t\t'

        return pretty_regen
    
    def __repr__(self):
        if self.baud_rate is not None and self.bit_rate is not None:
            temp = self.baud_rate * 1e-9
            temp2 = self.bit_rate * 1e-9
        else:
            temp = self.baud_rate
            temp2 = self.bit_rate

        pretty_print = [f'{type(self).__name__} {self.request_id}',
                        f'source: \t{self.source}',
                        f'destination:\t{self.destination}',
                        f'trx type:\t{self.trx_type}',
                        f'trx mode:\t{self.trx_mode}',
                        f'baud_rate:\t{temp} Gbaud',
                        f'bit_rate:\t{temp2} Gb/s',
                        f'spacing:\t{self.spacing * 1e-9} GHz',
                        f'power:  \t{round(lin2db(self.power)+30, 2)} dBm',
                        f'nb channels: \t{self.nb_channel}',
                        f'path_bandwidth: \t{round(self.path_bandwidth * 1e-9, 2)} Gbit/s',
                        f'nodes-list:\t{self.nodes_list}',
                        f'loose-list:\t{self.loose_list}\n']
        if self.regen_list_print != '':
            pretty_print.extend([f'regen-list:\t{self.regen_list_print}', '\n'])

        return '\n\t'.join(pretty_print)


class Disjunction:
    """ the class that contains all attributes related to disjunction constraints
    """

    def __init__(self, *args, **params):
        params = DisjunctionParams(**params)
        self.disjunction_id = params.disjunction_id
        self.relaxable = params.relaxable
        self.link_diverse = params.link_diverse
        self.node_diverse = params.node_diverse
        self.disjunctions_req = params.disjunctions_req

    def __str__(self):
        return '\n\t'.join([f'relaxable:     {self.relaxable}',
                            f'link-diverse:  {self.link_diverse}',
                            f'node-diverse:  {self.node_diverse}',
                            f'request-id-numbers: {self.disjunctions_req}'])

    def __repr__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.disjunction_id}',
                            f'relaxable:    {self.relaxable}',
                            f'link-diverse: {self.link_diverse}',
                            f'node-diverse: {self.node_diverse}',
                            f'request-id-numbers: {self.disjunctions_req}'
                            '\n'])


BLOCKING_NOPATH = ['NO_PATH', 'NO_PATH_WITH_CONSTRAINT',
                   'NO_FEASIBLE_BAUDRATE_WITH_SPACING',
                   'NO_COMPUTED_SNR']
BLOCKING_NOMODE = ['NO_FEASIBLE_MODE', 'MODE_NOT_FEASIBLE']
BLOCKING_NOSPECTRUM = ['NO_SPECTRUM', 'NOT_ENOUGH_RESERVED_SPECTRUM']


class ResultElement:
    def __init__(self, path_request, computed_path, reversed_computed_path=None):
        self.path_id = path_request.request_id
        self.path_request = path_request
        self.computed_path = computed_path
        # starting implementing reversed properties in case of bidir demand
        if reversed_computed_path is not None:
            self.reversed_computed_path = reversed_computed_path

    uid = property(lambda self: repr(self))

    @property
    def detailed_path_json(self):
        """ a function that builds path object for normal and blocking cases
        """
        index = 0
        pro_list = []
        for element in self.computed_path:
            temp = {
                'path-route-object': {
                    'index': index,
                    'num-unnum-hop': {
                        'node-id': element.uid,
                        'link-tp-id': element.uid,
                        # TODO change index in order to insert transponder attribute
                    }
                }
            }
            pro_list.append(temp)
            index += 1
            if not hasattr(self.path_request, 'blocking_reason'):
                # M and N values should not be None at this point
                if self.path_request.M is None or self.path_request.N is None:
                    raise ServiceError('request {self.path_id} should have positive non null n and m values.')

                temp = {
                    'path-route-object': {
                        'index': index,
                        "label-hop": {
                            "N": self.path_request.N,
                            "M": self.path_request.M
                        },
                    }
                }
                pro_list.append(temp)
                index += 1
            else:
                # if the path is blocked, no label object is created, but
                # the json response includes a detailed path for user information.
                # M and N values should be None at this point
                if self.path_request.M is not None or self.path_request.N is not None:
                    raise ServiceError('request {self.path_id} should not have label M and N values at this point.')


            if isinstance(element, Transceiver):
                temp = {
                    'path-route-object': {
                        'index': index,
                        'transponder': {
                            'transponder-type': self.path_request.trx_type,
                            'transponder-mode': self.path_request.trx_mode
                        }
                    }
                }
                pro_list.append(temp)
                index += 1
        return pro_list

    @property
    def path_properties(self):
        """ a function that returns the path properties (metrics, crossed elements) into a dict
        """
        def path_metric(pth, req):
            """ creates the metrics dictionary
            """
            return [
                {
                    'metric-type': 'SNR-bandwidth',
                    'accumulative-value': round(mean(pth[-1].snr), 2)
                },
                {
                    'metric-type': 'SNR-0.1nm',
                    'accumulative-value': round(mean(pth[-1].snr + lin2db(req.baud_rate / 12.5e9)), 2)
                },
                {
                    'metric-type': 'OSNR-bandwidth',
                    'accumulative-value': round(mean(pth[-1].osnr_ase), 2)
                },
                {
                    'metric-type': 'OSNR-0.1nm',
                    'accumulative-value': round(mean(pth[-1].osnr_ase_01nm), 2)
                },
                {
                    'metric-type': 'reference_power',
                    'accumulative-value': req.power
                },
                {
                    'metric-type': 'path_bandwidth',
                    'accumulative-value': req.path_bandwidth
                }
            ]
        if self.path_request.bidir:
            path_properties = {
                'path-metric': path_metric(self.computed_path, self.path_request),
                'z-a-path-metric': path_metric(self.reversed_computed_path, self.path_request),
                'path-route-objects': self.detailed_path_json
            }
        else:
            path_properties = {
                'path-metric': path_metric(self.computed_path, self.path_request),
                'path-route-objects': self.detailed_path_json
            }
        return path_properties

    @property
    def pathresult(self):
        """ create the result dictionnary (response for a request)
        """
        try:
            if self.path_request.blocking_reason in BLOCKING_NOPATH:
                response = {
                    'response-id': self.path_id,
                    'no-path': {
                        'no-path': self.path_request.blocking_reason
                    }
                }
                return response
            else:
                response = {
                    'response-id': self.path_id,
                    'no-path': {
                        'no-path': self.path_request.blocking_reason,
                        'path-properties': self.path_properties
                    }
                }
                return response
        except AttributeError:
            response = {
                'response-id': self.path_id,
                'path-properties': self.path_properties
            }
            return response

    @property
    def json(self):
        return self.pathresult


def compute_constrained_path(network, req):
    # nodes_list contains at least the destination
    if req.nodes_list[-1] != req.destination:
        # only arrive here if there is a bug in the program because route lists have
        # been corrected and harmonized before
        msg = (f'Request {req.request_id} malformed list of nodes: last node should '
               'be destination trx')
        LOGGER.critical(msg)
        raise ValueError()

    trx = [n for n in network if isinstance(n, Transceiver)]
    source = next(el for el in trx if el.uid == req.source)
    destination = next(el for el in trx if el.uid == req.destination)

    nodes_list = []
    for node in req.nodes_list[:-1]:
        nodes_list.append(next(el for el in network if el.uid == node))

    try:
        path_generator = shortest_simple_paths(network, source, destination, weight='weight')
        total_path = next(path for path in path_generator if ispart(nodes_list, path))
    except NetworkXNoPath:
        msg = (f'{ansi_escapes.yellow}Request {req.request_id} could not find a path from'
               f' {source.uid} to node: {destination.uid} in network topology{ansi_escapes.reset}')
        LOGGER.critical(msg)
        print(msg)
        req.blocking_reason = 'NO_PATH'
        total_path = []
    except StopIteration:
        # TODO: better account for individual loose and strict node
        # to ease: suppose that one strict makes the whole liste strict (except for the
        # last node which is the transceiver)
        # if all nodes i n node_list are LOOSE constraint, skip the constraints and find
        # a path w/o constraints, else there is no possible path
        print(f'{ansi_escapes.yellow}Request {req.request_id} could not find a path crossing '
              f'{[el.uid for el in nodes_list[:-1]]} in network topology{ansi_escapes.reset}')

        if 'STRICT' not in req.loose_list[:-1]:
            msg = (f'{ansi_escapes.yellow}Request {req.request_id} could not find a path with user_'
                   f'include node constraints{ansi_escapes.reset}')
            LOGGER.info(msg)
            print(f'constraint ignored')
            total_path = dijkstra_path(network, source, destination, weight='weight')
        else:
            # one STRICT makes the whole list STRICT
            msg = (f'{ansi_escapes.yellow}Request {req.request_id} could not find a path with user '
                   f'include node constraints.\nNo path computed{ansi_escapes.reset}')
            LOGGER.critical(msg)
            print(msg)
            req.blocking_reason = 'NO_PATH_WITH_CONSTRAINT'
            total_path = []

    return total_path


def propagate(path, req, equipment):
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate,
        req.power, req.spacing)
    for i, el in enumerate(path):
        if isinstance(el, Roadm):
            si = el(si, degree=path[i+1].uid)
        else:
            si = el(si)
        if isinstance(el, Regenerator):
            el.update_snr(req.tx_osnr, equipment['Roadm']['default'].add_drop_osnr)
            regen_spec = next(r for r in req.regen_list if r['regen_uid'] == el.uid)
            si = create_input_spectral_information(
                regen_spec['f_min'], regen_spec['f_max'], regen_spec['roll_off'], regen_spec['baud_rate'],
                regen_spec['power'], regen_spec['spacing'])
            el.update_emitter(si)
    path[0].update_snr(req.tx_osnr)
    if any(isinstance(el, Roadm) for el in path):
        path[-1].update_snr(req.tx_osnr, equipment['Roadm']['default'].add_drop_osnr)
    else:
        path[-1].update_snr(req.tx_osnr)
    return si


def propagate_and_optimize_mode(path, req, equipment):
    # if mode is unknown : loops on the modes starting from the highest baudrate fiting in the
    # step 1: create an ordered list of modes based on baudrate
    baudrate_to_explore = list(set([this_mode['baud_rate']
                                    for this_mode in equipment['Transceiver'][req.trx_type].mode
                                    if float(this_mode['min_spacing']) <= req.spacing]))
    # TODO be carefull on limits cases if spacing very close to req spacing eg 50.001 50.000
    baudrate_to_explore = sorted(baudrate_to_explore, reverse=True)
    if baudrate_to_explore:
        # at least 1 baudrate can be tested wrt spacing
        for this_br in baudrate_to_explore:
            modes_to_explore = [this_mode for this_mode in equipment['Transceiver'][req.trx_type].mode
                                if this_mode['baud_rate'] == this_br and
                                float(this_mode['min_spacing']) <= req.spacing]
            modes_to_explore = sorted(modes_to_explore,
                                      key=lambda x: x['bit_rate'], reverse=True)
            # print(modes_to_explore)
            # step2: computes propagation for each baudrate: stop and select the first that passes
            # TODO: the case of roll of is not included: for now use SI one
            # TODO: if the loop in mode optimization does not have a feasible path, then bugs
            spc_info = create_input_spectral_information(req.f_min, req.f_max,
                                                         equipment['SI']['default'].roll_off,
                                                         this_br, req.power, req.spacing)
            for i, el in enumerate(path):
                if isinstance(el, Roadm):
                    spc_info = el(spc_info, degree=path[i+1].uid)
                else:
                    spc_info = el(spc_info)
            for this_mode in modes_to_explore:
                if path[-1].snr is not None:
                    path[0].update_snr(this_mode['tx_osnr'])
                    if any(isinstance(el, Roadm) for el in path):
                        path[-1].update_snr(this_mode['tx_osnr'], equipment['Roadm']['default'].add_drop_osnr)
                    else:
                        path[-1].update_snr(this_mode['tx_osnr'])
                    if round(min(path[-1].snr + lin2db(this_br / (12.5e9))), 2) \
                            > this_mode['OSNR'] + equipment['SI']['default'].sys_margins:
                        return path, this_mode
                    else:
                        last_explored_mode = this_mode
                else:
                    req.blocking_reason = 'NO_COMPUTED_SNR'
                    return path, None
        # only get to this point if no baudrate/mode satisfies OSNR requirement

        # returns the last propagated path and mode
        msg = f'\tWarning! Request {req.request_id}: no mode satisfies path SNR requirement.\n'
        print(msg)
        LOGGER.info(msg)
        req.blocking_reason = 'NO_FEASIBLE_MODE'
        return path, last_explored_mode
    else:
        # no baudrate satisfying spacing
        msg = f'\tWarning! Request {req.request_id}: no baudrate satisfies spacing requirement.\n'
        print(msg)
        LOGGER.info(msg)
        req.blocking_reason = 'NO_FEASIBLE_BAUDRATE_WITH_SPACING'
        return [], None


def jsontopath_metric(path_metric):
    """ a functions that reads resulting metric  from json string
    """
    output_snr = next(e['accumulative-value']
                      for e in path_metric if e['metric-type'] == 'SNR-0.1nm')
    output_snrbandwidth = next(e['accumulative-value']
                               for e in path_metric if e['metric-type'] == 'SNR-bandwidth')
    output_osnr = next(e['accumulative-value']
                       for e in path_metric if e['metric-type'] == 'OSNR-0.1nm')
    # ouput osnr@bandwidth is not used
    # output_osnrbandwidth = next(e['accumulative-value']
    #     for e in path_metric if e['metric-type'] == 'OSNR-bandwidth')
    power = next(e['accumulative-value']
                 for e in path_metric if e['metric-type'] == 'reference_power')
    path_bandwidth = next(e['accumulative-value']
                          for e in path_metric if e['metric-type'] == 'path_bandwidth')
    return output_snr, output_snrbandwidth, output_osnr, power, path_bandwidth


def jsontoparams(my_p, tsp, mode, equipment):
    """ a function that derives optical params from transponder type and mode
        supports the no mode case
    """
    temp = []
    for elem in my_p['path-properties']['path-route-objects']:
        if 'num-unnum-hop' in elem['path-route-object']:
            temp.append(elem['path-route-object']['num-unnum-hop']['node-id'])
    pth = ' | '.join(temp)

    temp2 = []
    for elem in my_p['path-properties']['path-route-objects']:
        if 'label-hop' in elem['path-route-object'].keys():
            temp2.append(f'{elem["path-route-object"]["label-hop"]["N"]}, ' +
                         f'{elem["path-route-object"]["label-hop"]["M"]}')
    # OrderedDict.fromkeys returns the unique set of strings.
    # TODO: if spectrum changes along the path, we should be able to give the segments
    #       eg for regeneration case
    temp2 = list(OrderedDict.fromkeys(temp2))
    sptrm = ' | '.join(temp2)

    # find the tsp minOSNR, baud rate... from the eqpt library based
    # on tsp (type) and mode (format).
    # loading equipment already tests the existence of tsp type and mode:
    if mode is not None:
        [minosnr, baud_rate, bit_rate, cost] = \
            next([m['OSNR'], m['baud_rate'], m['bit_rate'], m['cost']]
                 for m in equipment['Transceiver'][tsp].mode if m['format'] == mode)
    else:
        [minosnr, baud_rate, bit_rate, cost] = ['', '', '', '']
    output_snr, output_snrbandwidth, output_osnr, power, path_bandwidth = \
        jsontopath_metric(my_p['path-properties']['path-metric'])

    return pth, minosnr, baud_rate, bit_rate, cost, output_snr, \
        output_snrbandwidth, output_osnr, power, path_bandwidth, sptrm


def jsontocsv(json_data, equipment, fileout):
    """ reads json path result file in accordance with:
        Yang model for requesting Path Computation
        draft-ietf-teas-yang-path-computation-01.txt.
        and write results in an CSV file
    """
    mywriter = writer(fileout)
    mywriter.writerow(('response-id', 'source', 'destination', 'path_bandwidth', 'Pass?',
                       'nb of tsp pairs', 'total cost', 'transponder-type', 'transponder-mode',
                       'OSNR-0.1nm', 'SNR-0.1nm', 'SNR-bandwidth', 'baud rate (Gbaud)',
                       'input power (dBm)', 'path', 'spectrum (N,M)', 'reversed path OSNR-0.1nm',
                       'reversed path SNR-0.1nm', 'reversed path SNR-bandwidth'))

    for pth_el in json_data['response']:
        path_id = pth_el['response-id']
        if 'no-path' in pth_el.keys():
            total_cost = ''
            nb_tsp = ''
            sptrm = ''
            if pth_el['no-path']['no-path'] in BLOCKING_NOPATH:
                source = ''
                destination = ''
                pthbdbw = ''
                isok = pth_el['no-path']['no-path']
                tsp = ''
                mode = ''
                rosnr = ''
                rsnr = ''
                rsnrb = ''
                brate = ''
                pwr = ''
                pth = ''
                revosnr = ''
                revsnr = ''
                revsnrb = ''
            else:
                # the objects are listed with this order:
                # - id of hop
                # - label (N,M)
                # - transponder for source and destination only
                # as spectrum assignment is not performed for blocked demands: there is no label object in the answer
                # so the hop_attribute with tsp and mode is second object or last object, while id of hop is first and
                # penultimate
                source = pth_el['no-path']['path-properties']['path-route-objects'][0]['path-route-object']['num-unnum-hop']['node-id']
                destination = pth_el['no-path']['path-properties']['path-route-objects'][-2]['path-route-object']['num-unnum-hop']['node-id']
                temp_tsp = pth_el['no-path']['path-properties']['path-route-objects'][1]['path-route-object']['transponder']
                tsp = temp_tsp['transponder-type']
                mode = temp_tsp['transponder-mode']
                isok = pth_el['no-path']['no-path']
                if pth_el['no-path']['no-path'] in BLOCKING_NOMODE or \
                        pth_el['no-path']['no-path'] in BLOCKING_NOSPECTRUM:
                    pth, minosnr, baud_rate, bit_rate, cost, output_snr, output_snrbandwidth, \
                        output_osnr, power, path_bandwidth, sptrm = \
                        jsontoparams(pth_el['no-path'], tsp, mode, equipment)
                    pthbdbw = ''
                    rosnr = round(output_osnr, 2)
                    rsnr = round(output_snr, 2)
                    rsnrb = round(output_snrbandwidth, 2)
                    brate = round(baud_rate * 1e-9, 2)
                    pwr = round(lin2db(power) + 30, 2)
                    if 'z-a-path-metric' in pth_el['no-path']['path-properties'].keys():
                        output_snr, output_snrbandwidth, output_osnr, power, path_bandwidth = \
                            jsontopath_metric(pth_el['no-path']['path-properties']['z-a-path-metric'])
                        revosnr = round(output_osnr, 2)
                        revsnr = round(output_snr, 2)
                        revsnrb = round(output_snrbandwidth, 2)
                    else:
                        revosnr = ''
                        revsnr = ''
                        revsnrb = ''
        else:
            # when label will be assigned destination will be with index -3, and transponder with index 2
            source = pth_el['path-properties']['path-route-objects'][0]['path-route-object']['num-unnum-hop']['node-id']
            destination = pth_el['path-properties']['path-route-objects'][-3]['path-route-object']['num-unnum-hop']['node-id']
            # selects only roadm nodes
            temp_tsp = pth_el['path-properties']['path-route-objects'][2]['path-route-object']['transponder']
            tsp = temp_tsp['transponder-type']
            mode = temp_tsp['transponder-mode']

            # find the min  acceptable OSNR, baud rate from the eqpt library based
            # on tsp (type) and mode (format).
            # loading equipment already tests the existence of tsp type and mode:
            pth, minosnr, baud_rate, bit_rate, cost, output_snr, output_snrbandwidth, \
                output_osnr, power, path_bandwidth, sptrm = \
                jsontoparams(pth_el, tsp, mode, equipment)
            # this part only works if the request has a blocking_reason atribute, ie if it could not be satisfied
            isok = output_snr >= minosnr
            nb_tsp = ceil(path_bandwidth / bit_rate)
            pthbdbw = round(path_bandwidth * 1e-9, 2)
            rosnr = round(output_osnr, 2)
            rsnr = round(output_snr, 2)
            rsnrb = round(output_snrbandwidth, 2)
            brate = round(baud_rate * 1e-9, 2)
            pwr = round(lin2db(power) + 30, 2)
            total_cost = nb_tsp * cost
            if 'z-a-path-metric' in pth_el['path-properties'].keys():
                output_snr, output_snrbandwidth, output_osnr, power, path_bandwidth = \
                    jsontopath_metric(pth_el['path-properties']['z-a-path-metric'])
                revosnr = round(output_osnr, 2)
                revsnr = round(output_snr, 2)
                revsnrb = round(output_snrbandwidth, 2)
            else:
                revosnr = ''
                revsnr = ''
                revsnrb = ''
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
                           brate,
                           pwr,
                           pth,
                           sptrm,
                           revosnr,
                           revsnr,
                           revsnrb
                           ))


def compute_path_dsjctn(network, equipment, pathreqlist, disjunctions_list):
    # pathreqlist is a list of PathRequest objects
    # disjunctions_list a list of Disjunction objects

    # given a network, a list of requests with the set of disjunction features between
    # request, the function computes the set of path satisfying: first the disjunction
    # constraint and second the routing constraint if the request include an explicit
    # set of elements to pass through.
    # the algorithm used allows to specify disjunction for demands not sharing source or
    # destination.
    # a request might be declared as disjoint from several requests
    # it is a iterative process:
    # first computes a list of all shortest path (this may add computation time)
    # second elaborate the set of path solution for each synchronization vector
    # third select only the candidates that satisfy all synchronization vectors they belong to
    # fourth apply route constraints: remove candidate path that do not satisfy the constraint
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
    #  abfhg|  2.5    x      x
    #  abcg |  3      x             x
    #  aefhg|  3      x      x      x
    # from this table abcg and aefh have no common links and should be preferred
    # even they are not the shortest paths

    # build the list of pathreqlist elements not concerned by disjunction
    global_disjunctions_list = [e for d in disjunctions_list for e in d.disjunctions_req]
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
    for pathreq in pathreqlist_disjt:
        all_simp_pths = list(all_simple_paths(network,
                                              source=next(el for el in network.nodes() if el.uid == pathreq.source),
                                              target=next(el for el in network.nodes()
                                                          if el.uid == pathreq.destination),
                                              cutoff=80))
        # sort them in km length instead of hop
        # all_simp_pths = sorted(all_simp_pths, key=lambda path: len(path))
        all_simp_pths = sorted(all_simp_pths, key=lambda
                               x: sum(network.get_edge_data(x[i], x[i + 1])['weight'] for i in range(len(x) - 2)))
        # reversed direction paths required to check disjunction on both direction
        all_simp_pths_reversed = []
        for pth in all_simp_pths:
            all_simp_pths_reversed.append(find_reversed_path(pth))
        rqs[pathreq.request_id] = all_simp_pths
        temp = []
        for pth in all_simp_pths:
            # build a short list representing each roadm+direction with the first item
            # start enumeration at 1 to avoid Trx in the list
            short_list = [e.uid for i, e in enumerate(pth[1:-1])
                          if isinstance(e, Roadm) | (isinstance(pth[i], Roadm))]
            temp.append(short_list)
            # id(short_list) is unique even if path is the same: two objects with same
            # path have two different ids
            allpaths[id(short_list)] = Pth(pathreq, pth, short_list)
        simple_rqs[pathreq.request_id] = temp
        temp = []
        for pth in all_simp_pths_reversed:
            # build a short list representing each roadm+direction with the first item
            # start enumeration at 1 to avoid Trx in the list
            temp.append([e.uid for i, e in enumerate(pth[1:-1])
                         if isinstance(e, Roadm) | (isinstance(pth[i], Roadm))])
        simple_rqs_reversed[pathreq.request_id] = temp
    # step 2
    # for each set of requests that need to be disjoint
    # select the disjoint path combination

    candidates = {}
    for dis in disjunctions_list:
        dlist = dis.disjunctions_req.copy()
        # each line of dpath is one combination of path that satisfies disjunction
        dpath = []
        for i, pth in enumerate(simple_rqs[dlist[0]]):
            dpath.append([pth])
        # in each loop, dpath is updated with a path for rq that satisfies
        # disjunction with each path in dpath
        # for example, assume set of requests in the vector (disjunction_list) is  {rq1,rq2, rq3}
        # rq1  p1: aefhg
        #      p2: abfhg
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
        #        p1 and p6 are not disjoint
        #        p1 and p7 are not disjoint
        #        p3 and p4 are not disjoint
        #        p3 and p7 are not disjoint

        for elem1 in dlist[1:]:
            temp = []
            for j, pth1 in enumerate(simple_rqs[elem1]):
                # can use index j in simple_rqs_reversed because index
                # of direct and reversed paths have been kept identical
                pth1_reversed = simple_rqs_reversed[elem1][j]
                # print(pth1_reversed)
                # print('\n\n')
                for cndt in dpath:
                    # print(f' c: \t{c}')
                    temp2 = cndt.copy()
                    all_disjoint = 0
                    for pth in cndt:
                        all_disjoint += isdisjoint(pth1, pth) + isdisjoint(pth1_reversed, pth)
                    if all_disjoint == 0:
                        temp2.append(pth1)
                        temp.append(temp2)
                        # print(f' coucou {elem1}: \t{temp}')
            dpath = temp
        candidates[dis.disjunction_id] = dpath

    # for i in disjunctions_list:
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
        concerned_d_id = [d.disjunction_id for d in disjunctions_list
                          if pathreq.request_id in d.disjunctions_req]
        # for each set of solution, verify that the same path is used for the same request
        candidate_paths = simple_rqs[pathreq.request_id]
        # print('coucou')
        # print(pathreq.request_id)
        for pth in candidate_paths:
            iscandidate = 0
            for sol in concerned_d_id:
                test = 1
                # for each solution test if pth is part of the solution
                # if yes, then pth can remain a candidate
                for cndt in candidates[sol]:
                    if pth in cndt:
                        if allpaths[id(cndt[cndt.index(pth)])].req.request_id == pathreq.request_id:
                            test = 0
                            break
                iscandidate += test
            if iscandidate != 0:
                for this_id in concerned_d_id:
                    for cndt in candidates[this_id]:
                        if pth in cndt:
                            candidates[this_id].remove(cndt)

    # for i in disjunctions_list:
    #     print(i.disjunction_id)
    #     print(f'\n{candidates[i.disjunction_id]}')

    # step 4 apply route constraints: remove candidate path that do not satisfy
    # the constraint only in  the case of disjounction: the simple path is processed in
    # request.compute_constrained_path
    # TODO: keep a version without the loose constraint
    for this_d in disjunctions_list:
        temp = []
        alternatetemp = []
        for j, sol in enumerate(candidates[this_d.disjunction_id]):
            testispartok = True
            testispartnokloose = True
            for pth in sol:
                # print(f'test {allpaths[id(pth)].req.request_id}')
                # print(f'length of route {len(allpaths[id(pth)].req.nodes_list)}')
                if allpaths[id(pth)].req.nodes_list:
                    # if any pth from sol does not contain the ordered list node,
                    # remove sol from the candidate, except if constraint was loose:
                    # then keep sol as an alternate solution
                    if not ispart(allpaths[id(pth)].req.nodes_list, pth):
                        testispartok = False
                        if 'STRICT' in allpaths[id(pth)].req.loose_list:
                            LOGGER.info(f'removing solution from candidate paths\n{pth}')
                            testispartnokloose = False
                            break
            if testispartok:
                temp.append(sol)
            elif testispartnokloose:
                LOGGER.info(f'Adding solution as alternate solution not satisfying constraint\n{pth}')
                alternatetemp.append(sol)
        if temp:
            candidates[this_d.disjunction_id] = temp
        elif alternatetemp:
            candidates[this_d.disjunction_id] = alternatetemp
        else:
            candidates[this_d.disjunction_id] = []

    # step 5 select the first combination that works
    pathreslist_disjoint = {}
    for dis in disjunctions_list:
        if candidates[dis.disjunction_id]:
            for pth in candidates[dis.disjunction_id][0]:
                if allpaths[id(pth)].req in pathreqlist_disjt:
                    # print(f'selected path:{pth} for req {allpaths[id(pth)].req.request_id}')
                    pathreslist_disjoint[allpaths[id(pth)].req] = allpaths[id(pth)].pth
                    # remove request from list of requests (in case of duplicate)
                    pathreqlist_disjt.remove(allpaths[id(pth)].req)
                    # remove duplicated candidates
                    candidates = remove_candidate(candidates, allpaths, allpaths[id(pth)].req, pth)
        else:
            msg = f'No disjoint path found with added constraint'
            LOGGER.critical(msg)
            print(f'{msg}\nComputation stopped.')
            # TODO in this case: replay step 5  with the candidate without constraints
            raise DisjunctionError(msg)

    # for i in disjunctions_list:
    #     print(i.disjunction_id)
    #     print(f'\n{candidates[i.disjunction_id]}')

    # list the results in the same order as initial pathreqlist
    for req in pathreqlist:
        req.nodes_list.append(req.destination)
        # we assume that the destination is a strict constraint
        req.loose_list.append('STRICT')
        if req in pathreqlist_simple:
            path_res_list.append(compute_constrained_path(network, req))
        else:
            path_res_list.append(pathreslist_disjoint[req])
    return path_res_list


def isdisjoint(pth1, pth2):
    """ returns 0 if disjoint
    """
    edge1 = list(pairwise(pth1))
    edge2 = list(pairwise(pth2))
    for edge in edge1:
        if edge in edge2:
            return 1
    return 0



def decompose_path(path, nb_decomposition=0):
    """ Returns a list of non regenerated sections of a given path
    first and last elements are either transceiver or a regenerator
    Use nb_decomposition to return the same number of empty section as nb_decompsition
    if it is provided.

    >>> from gnpy.core.elements import Transceiver, Fused, Regenerator
    >>> source = Transceiver('trxa')
    >>> destination = Transceiver('trxc')
    >>> a = Fused('fuseda')
    >>> c = Fused('fusedc')
    >>> regen = Regenerator('regenb')
    >>> path = [source, a, regen, c, destination]
    >>> sections = decompose_path(path)
    >>> for section in sections:
    ...     [e.uid for e in section]
    ... 
    ['trxa', 'fuseda', 'regenb']
    ['regenb', 'fusedc', 'trxc']

    >>> decompose_path([], 2)
    [[], []]
    """
    decomposed_path = []
    section = []
    if path:
        for elem in path:
            section.append(elem)
            if isinstance(elem, Regenerator):
                decomposed_path.append(section)
                section = []
                section.append(elem)
        decomposed_path.append(section)
    else:
        for i in range(nb_decomposition):
            decomposed_path.append([])
    return decomposed_path


def find_reversed_path(pth):
    """ select of intermediate roadms and find the path between them
        note that this function may not give an exact result in case of multiple
        links between two adjacent nodes.
    """
    # TODO add some indication on elements to indicate from which other they
    # are the reversed direction. This is partly done with oms indication

    # we want the list of crossed oms and each item must be unique in the list:
    # since a succession of elements of the path can be in the same oms, a 'unique'
    # function is needed
    # the OrderedDict.fromkeys function does this. eg
    # pth = [el1_oms1 el2_oms1 el3_oms1 el1_oms2 el2_oms2 el3_oms2]
    # p_oms should be = [oms1 oms2]

    # first decompose in regenerated sections
    regen_sections = decompose_path(pth)
    # for each section (reverse order) find the reverse path
    reversed_path = []
    for section in reversed(regen_sections):
        p_oms = list(OrderedDict.fromkeys(reversed([el.oms.reversed_oms for el in section
                                                    if not isinstance(el, (Transceiver, Roadm))])))
        reversed_section = [section[-1]]
        for oms in p_oms:
            if oms is not None:
                reversed_section.extend(oms.el_list)
                # similarly each oms starts and ends with a roadm so roadm may be repeated
                # if we don't use the OrderedDict.fromkeys function. eg:
                # if oms1 = [roadma el1 el2 roadmb] and oms2 = [roadmb el3 el4 roadmc]
                # concatenation should be [roadma el1 el2 roadmb el3 el4 roadmc]
                reversed_section = list(OrderedDict.fromkeys(reversed_section))
            else:
                msg = f'Error while handling reversed path {section[-1].uid} to {section[0].uid}:' +\
                    ' can not handle unidir topology. TO DO.'
                LOGGER.critical(msg)
                raise ValueError(msg)
        reversed_path.extend(reversed_section)
    reversed_path.append(pth[0])
    return reversed_path


def ispart(ptha, pthb):
    """ the functions takes two paths a and b and retrns True
        if all a elements are part of b and in the same order
    """
    j = 0
    for elem in ptha:
        if elem in pthb:
            if pthb.index(elem) >= j:
                j = pthb.index(elem)
            else:
                return False
        else:
            return False
    return True


def remove_candidate(candidates, allpaths, rqst, pth):
    """ filter duplicate candidates
    """
    # print(f'coucou {rqst.request_id}')
    for key, candidate in candidates.items():
        temp = candidate.copy()
        for sol in candidate:
            for this_p in sol:
                if allpaths[id(this_p)].req.request_id == rqst.request_id:
                    if id(this_p) != id(pth):
                        temp.remove(sol)
                        break
        candidates[key] = temp
    return candidates


def compare_reqs(req1, req2, disjlist):
    """ compare two requests: returns True or False
    """
    dis1 = [d for d in disjlist if req1.request_id in d.disjunctions_req]
    dis2 = [d for d in disjlist if req2.request_id in d.disjunctions_req]
    same_disj = False
    if dis1 and dis2:
        temp1 = []
        for this_d in dis1:
            temp1.extend(this_d.disjunctions_req)
            temp1.remove(req1.request_id)
        temp2 = []
        for this_d in dis2:
            temp2.extend(this_d.disjunctions_req)
            temp2.remove(req2.request_id)
        if set(temp1) == set(temp2):
            same_disj = True
    elif not dis2 and not dis1:
        same_disj = True

    if req1.source == req2.source and \
            req1.destination == req2.destination and  \
            req1.trx_type == req2.trx_type and \
            req1.trx_mode == req2.trx_mode and \
            req1.baud_rate == req2.baud_rate and \
            req1.nodes_list == req2.nodes_list and \
            req1.loose_list == req2.loose_list and \
            req1.regen_list == req2.regen_list and \
            req1.spacing == req2.spacing and \
            req1.power == req2.power and \
            req1.nb_channel == req2.nb_channel and \
            req1.f_min == req2.f_min and \
            req1.f_max == req2.f_max and \
            req1.format == req2.format and \
            req1.OSNR == req2.OSNR and \
            req1.roll_off == req2.roll_off and \
            same_disj and \
            getattr(req1, 'N', None) is None and getattr(req2, 'N', None) is None and \
            getattr(req1, 'M', None) is None and getattr(req2, 'M', None) is None:
        return True
    else:
        return False


def requests_aggregation(pathreqlist, disjlist):
    """ this function aggregates requests so that if several requests
        exist between same source and destination and with same transponder type
    """
    # todo maybe add conditions on mode ??, spacing ...
    # currently if undefined takes the default values
    local_list = pathreqlist.copy()
    for req in pathreqlist:
        for this_r in local_list:
            if req.request_id != this_r.request_id and compare_reqs(req, this_r, disjlist):
                # aggregate
                this_r.path_bandwidth += req.path_bandwidth
                temp_r_id = this_r.request_id
                this_r.request_id = ' | '.join((this_r.request_id, req.request_id))
                # remove request from list
                local_list.remove(req)
                # todo change also disjunction req with new demand

                for this_d in disjlist:
                    if req.request_id in this_d.disjunctions_req:
                        this_d.disjunctions_req.remove(req.request_id)
                        this_d.disjunctions_req.append(this_r.request_id)
                for this_d in disjlist:
                    if temp_r_id in this_d.disjunctions_req:
                        disjlist.remove(this_d)
                break
    return local_list, disjlist


def correct_json_route_list(network, pathreqlist):
    """ all names in list should be exact name in the network, and there is no ambiguity
        This function only checks that list is correct, warns user if the name is incorrect and
        suppresses the constraint it it is loose or raises an error if it is strict
    """
    all_uid = [n.uid for n in network.nodes()]
    # do not record regenerators in the transponder list, so use type instead of isinstance
    transponders = [n.uid for n in network.nodes() if type(n) is Transceiver]
    for pathreq in pathreqlist:
        if pathreq.source not in transponders:
            msg = f'{ansi_escapes.red}Request: {pathreq.request_id}: could not find transponder' +\
                f' source : {pathreq.source}.{ansi_escapes.reset}'
            LOGGER.critical(msg)
            raise ServiceError(msg)

        if pathreq.destination not in transponders:
            msg = f'{ansi_escapes.red}Request: {pathreq.request_id}: could not find transponder' +\
                f' destination : {pathreq.destination}.{ansi_escapes.reset}'
            LOGGER.critical(msg)
            raise ServiceError(msg)

        # silently remove source and dest nodes from the list
        if pathreq.nodes_list and pathreq.source == pathreq.nodes_list[0]:
            pathreq.loose_list.pop(0)
            pathreq.nodes_list.pop(0)
        if pathreq.nodes_list and pathreq.destination == pathreq.nodes_list[-1]:
            pathreq.loose_list.pop(-1)
            pathreq.nodes_list.pop(-1)
        temp = deepcopy(pathreq)
        for i, n_id in enumerate(temp.nodes_list):
            # a node within this list must be part of the topology and should not be a transceiver,
            # because only source and dest are transceivers
            if n_id not in all_uid or n_id in transponders:
                if temp.loose_list[i] == 'LOOSE':
                    # if no matching can be found in the network just ignore this constraint
                    # if it is a loose constraint
                    # warns the user that this node is not part of the topology
                    msg = f'{ansi_escapes.yellow}invalid route node specified:\n\t\'{n_id}\',' +\
                        f' could not use it as constraint, skipped!{ansi_escapes.reset}'
                    print(msg)
                    LOGGER.info(msg)
                    pathreq.loose_list.pop(pathreq.nodes_list.index(n_id))
                    pathreq.nodes_list.remove(n_id)
                else:
                    msg = f'{ansi_escapes.red}could not find node:\n\t \'{n_id}\' in network' +\
                        f' topology. Strict constraint can not be applied.{ansi_escapes.reset}'
                    LOGGER.critical(msg)
                    raise ServiceError(msg)

    return pathreqlist


def correct_json_regen_list(network, equipment, pathreqlist):
    """ all names in list should be exact name in the network, and there is no ambiguity
    This function checks that list is correct, warns user if the name is incorrect and
    suppresses the constraint if it is loose or raises an error if it is strict
    it also swap the node_name for the attached roadm name in the route constraint.
    This is to avoid loops in path search: a path trxa-roadma-fiber1-roadmb-regenb-roadmb-fiber2-roadmc-trxc
    is changed to rxa-roadma-fiber1-roadmb-fiber2-roadmc-trxc for computation purpose
    """
    regenerators = [n.uid for n in network.nodes() if isinstance(n, Regenerator)]
    for pathreq in pathreqlist:
        # first add in regen_list the regen nodes in node_list that were not declared explicitely as regenerators
        regen_uid_list = [r['regen_uid'] for r in pathreq.regen_list]
        for node_uid in pathreq.nodes_list:
            try:
                node = next(n for n in network.nodes() if n.uid == node_uid)
                if isinstance(node, Regenerator) and node_uid not in regen_uid_list and \
                        node_uid != pathreq.source and node_uid != pathreq.destination:
                    regen = {'regen_uid': node_uid, 'spacing': pathreq.spacing, 'trx_type': pathreq.trx_type,
                             'trx_mode': pathreq.trx_mode}
                    # since no transceiver is specified for the regen use the request's type and mode
                    regen_params = trx_mode_params(equipment,  pathreq.trx_type, pathreq.trx_mode, True)
                    regen.update(regen_params)
                    pathreq.regen_list.append(regen)
                elif node_uid == pathreq.source or node_uid == pathreq.destination:
                    raise ServiceError('Can not place a regenerator as source or destination node', pathreq.request_id)
            except StopIteration:
                raise ServiceError(f'could not find {node_uid} in topology')
        # then check that all regen in regen_list are really of Regenerator types
        for regen in pathreq.regen_list:
            # a node within this list must be part of the topology and should be a regen
            if regen['regen_uid'] not in regenerators:
                raise ServiceError('Inconsistent regenerator definition: not a regenerator', regen['regen_uid'])
    return pathreqlist


def remove_regen_from_list(network, pathreqlist):
    """ Swap the regen node_name with the attached roadm name in the route constraint.
    This is to avoid loops in path search: a path trxa-roadma-fiber1-roadmb-regenb-roadmb-fiber2-roadmc-trxc
    is changed to rxa-roadma-fiber1-roadmb-fiber2-roadmc-trxc for computation purpose
    """
    regenerators = [n.uid for n in network.nodes() if isinstance(n, Regenerator)]
    for pathreq in pathreqlist: 
        # Prepare the node_lis to be usable by networkx functions: remove Regenerator from th node list and replace it
        # with its attached ROADM : trxa-roadma-fiber1-roadmb-regenb-roadmb-fiber2-roadmc-trxc  changed to
        # trxa-roadma-fiber1-roadmb-roadmb-roadmb-fiber2-roadmc-trxc (same loose_list)
        for i, n_id in enumerate(pathreq.nodes_list):
            # change regen into its attached roadm
            if n_id in regenerators:
                regenerator = next(n for n in network.nodes() if n.uid == n_id)
                attached_roadm = next(network.successors(regenerator))
                pathreq.nodes_list[i] = attached_roadm.uid
        # remove duplicate nodes in nodes_list and correct loose_list accordingly:
        # trxa-roadma-fiber1-roadmb-fiber2-roadmc-trxc
        pathreq.nodes_list, pathreq.loose_list = unique_ordered(pathreq.nodes_list, pathreq.loose_list)
    return pathreqlist


def restore_regen_in_path(network, pathreqlist, pathlist):
    """ restore the regenerator in the path, once a path is computed
    """
    for req, path in zip(pathreqlist, pathlist):
        regen_uid_list = [r['regen_uid'] for r in req.regen_list]
        for i, node_id in enumerate(regen_uid_list):
            regenerator = next(n for n in network.nodes() if n.uid == node_id)
            attached_roadm = next(network.successors(regenerator))
            for i, elem in enumerate(path):
                if elem == attached_roadm:
                    path.insert(i + 1, regenerator)
                    path.insert(i + 2, attached_roadm)
                    break
    return pathlist


def deduplicate_disjunctions(disjn):
    """ clean disjunctions to remove possible repetition
    """
    local_disjn = disjn.copy()
    for elem in local_disjn:
        for dis_elem in local_disjn:
            if set(elem.disjunctions_req) == set(dis_elem.disjunctions_req) and \
                    elem.disjunction_id != dis_elem.disjunction_id:
                local_disjn.remove(dis_elem)
    return local_disjn


def decompose_req(req):
    """ Returns a list of requests corresponding to the regenerated sections of the given path.
    for example request corresponding to trxa-regenb-trxc will be decomposed into
    [trxa to regenb , regenb to trx]. also returns the nb of sections
    """
    decomposed_rqs = []
    source = req.source
    temp = deepcopy(req)
    if req.regen_list:
        for regen in req.regen_list:
            temp.source = source
            temp.destination = regen['regen_uid']
            decomposed_rqs.append(temp)
            temp = deepcopy(req)
            source = regen['regen_uid']
            for key, val in regen.items():
                if hasattr(temp, key):
                    setattr(temp, key, val)
    temp.source = source
    temp.destination = req.destination
    decomposed_rqs.append(temp)
    return decomposed_rqs


def recompose_regenerated_sections(sections, pathreq, modes, equipment):
    """ Recompose a path with all its regenerated sections after it has been propagated.
    """
    # start the path with first node of the first section
    total_path = [sections[0][0]]
    trx_type = pathreq.trx_type
    pathreq.trx_mode = modes[0]
    for i, section in enumerate(sections[:-1]):
        # append all nodes in the section except last node (which is the first node of next section)
        total_path.extend(section[1:])
        # Update regen_list with the selected mode for the given regen according to what has been computed.
        regen = {'spacing': pathreq.spacing, 'trx_mode': modes[i + 1]['format']}
        # use the request's type
        regen_params = trx_mode_params(equipment,  pathreq.trx_type, modes[i + 1]['format'], True)
        regen.update(regen_params)
        pathreq.regen_list[i].update(regen)
    # Append the nodes from the final section, except for first node (which is already in the path of previous section)
    total_path.extend(sections[-1][1:])
    return total_path, pathreq


def compute_path_with_disjunction(network, equipment, pathreqlist, pathlist):
    """ use a list but a dictionnary might be helpful to find path based on request_id
        TODO change all these req, dsjct, res lists into dict !
    """
    path_res_list = []
    reversed_path_res_list = []
    propagated_reversed_path_res_list = []

    for i, pathreq in enumerate(pathreqlist):

        # use the power specified in requests but might be different from the one
        # specified for design the power is an optional parameter for requests
        # definition if optional, use the one defines in eqt_config.json
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        # adding first node to be clearer on the output
        print(f'with path constraint: {[pathreq.source] + pathreq.nodes_list}')

        # pathlist[i] contains the whole path information for request i
        # last element is a transciver and where the result of the propagation is
        # recorded.
        # Important Note: since transceivers attached to roadms are actually logical
        # elements to simulate performance, several demands having the same destination
        # may use the same transponder for the performance simulation. This is why
        # we use deepcopy: to ensure that each propagation is recorded and not overwritten
        total_path = deepcopy(pathlist[i])
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, (Roadm, Regenerator))]}')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if total_path:
            if pathreq.baud_rate is not None:
                # means that at this point the mode was entered/forced by user and thus a
                # baud_rate was defined
                propagate(total_path, pathreq, equipment)

                regen_site = [e.uid for e in total_path if isinstance(e, Regenerator)] + [total_path[-1].uid]
                regen_snr = [round(mean(e.receiver_snr_01nm), 2) for e in total_path if isinstance(e, Regenerator)] + \
                            [round(mean(total_path[-1].snr_01nm),2)]
                required_OSNR = [pathreq.OSNR + equipment['SI']['default'].sys_margins] + \
                                [r['OSNR'] + equipment['SI']['default'].sys_margins for r in pathreq.regen_list]
                # check that each regen section is feasible else if any is below min required OSNR,
                # raise the warning
                for snr, required_osnr, site in zip(regen_snr, required_OSNR, regen_site):
                    if snr < required_osnr:
                        msg = f'\tWarning! Request {pathreq.request_id} computed path from' +\
                              f' {pathreq.source} to {pathreq.destination} does not pass with' +\
                              f' {site}\n\tcomputedSNR in 0.1nm = {snr} - required osnr {required_osnr}'
                        print(msg)
                        LOGGER.warning(msg)
                        pathreq.blocking_reason = 'MODE_NOT_FEASIBLE'
            else:
                # select mode per regenerated section
                req_sections = decompose_req(pathreq)
                sections = decompose_path(total_path, len(req_sections))
                modes = []
                copy_sections = []
                for section, req_section in zip(sections, req_sections):
                    section, mode = propagate_and_optimize_mode(section, req_section, equipment)
                    # since regenerator and attached ROADM are propagated twice, need to use deepcopy to reproduce
                    # the objects, and avoid loosing the performance information (GSNR)
                    copy_sections.append(deepcopy(section))
                    modes.append(mode)
                total_path, pathreq = recompose_regenerated_sections(copy_sections, pathreq, modes, equipment)


                # if no baudrate satisfies spacing, no mode is returned and the last explored mode
                # a warning is shown in the propagate_and_optimize_mode
                # propagate_and_optimize_mode function returns the mode with the highest bitrate
                # that passes. if no mode passes, then a attribute blocking_reason is added on
                # pathreq that contains the reason for blocking: 'NO_PATH', 'NO_FEASIBLE_MODE', ...
                try:
                    if pathreq.blocking_reason in BLOCKING_NOPATH:
                        total_path = []
                    elif pathreq.blocking_reason in BLOCKING_NOMODE:
                        pathreq.baud_rate = mode['baud_rate']
                        pathreq.trx_mode = mode['format']
                        pathreq.format = mode['format']
                        pathreq.OSNR = mode['OSNR']
                        pathreq.tx_osnr = mode['tx_osnr']
                        pathreq.bit_rate = mode['bit_rate']
                    # other blocking reason should not appear at this point
                except AttributeError:
                    pathreq.baud_rate = mode['baud_rate']
                    pathreq.trx_mode = mode['format']
                    pathreq.format = mode['format']
                    pathreq.OSNR = mode['OSNR']
                    pathreq.tx_osnr = mode['tx_osnr']
                    pathreq.bit_rate = mode['bit_rate']

            # reversed path is needed for correct spectrum assignment
            reversed_path = find_reversed_path(pathlist[i])
            if pathreq.bidir and pathreq.baud_rate is not None:
                # Both directions requested, and a feasible mode was found
                rev_p = deepcopy(reversed_path)

                print(f'\n\tPropagating Z to A direction {pathreq.destination} to {pathreq.source}')
                print(f'\tPath (roadsm) {[r.uid for r in rev_p if isinstance(r,(Roadm, Regenerator))]}\n')
                propagate(rev_p, pathreq, equipment)
                # if there are some regen, need to change req.regen_list and req initial values to support
                # regen of different mode/type eg suppose following path
                # tx(voyager, mode1, 50Ghz)-ROADMA-ROADMB-Regen(cassini, mode2, 50Ghz)-ROADMB-ROADMC-rx
                # then rx must be (cassini, mode2, 50Ghz) (so different from source) and reversed path must be:
                # tx(cassini, mode2, 50Ghz)-ROADMC-ROADMB-Regen(voyager, mode1, 50Ghz)-ROADMB-ROADMA-rx(voyager, mode1, 50Ghz)
                propagated_reversed_path = rev_p
                #use the regeneration information in reverse order
                rev_regen_site = [e.uid for e in propagated_reversed_path if isinstance(e, Regenerator)] +\
                                 [propagated_reversed_path[-1].uid]
                rev_regen_snr = [round(mean(e.receiver_snr_01nm), 2) for e in propagated_reversed_path
                                 if isinstance(e, Regenerator)] +\
                                [round(mean(propagated_reversed_path[-1].snr_01nm),2)]
                # don't forget: required_OSNR already integrates system_margin
                rev_required_OSNR = required_OSNR[::-1]
                for site, snr, required_snr in zip(rev_regen_site, rev_regen_snr, rev_required_OSNR):
                    if snr < required_snr:
                        msg = f'\tWarning! Request {pathreq.request_id} computed path from' +\
                              f' {pathreq.destination} to {pathreq.source} does not pass with' +\
                              f' {site}\n\tcomputedSNR in 0.1nm = {snr} -' +\
                              f' required osnr {required_snr + equipment["SI"]["default"].sys_margins}'
                        print(msg)
                        LOGGER.warning(msg)
                        # TODO selection of mode should also be on reversed direction !!
                        if not hasattr(pathreq, 'blocking_reason'):
                            pathreq.blocking_reason = 'MODE_NOT_FEASIBLE'
            else:
                propagated_reversed_path = []
        else:
            msg = 'Total path is empty. No propagation'
            print(msg)
            LOGGER.info(msg)
            reversed_path = []
            propagated_reversed_path = []

        path_res_list.append(total_path)
        reversed_path_res_list.append(reversed_path)
        propagated_reversed_path_res_list.append(propagated_reversed_path)
        # print to have a nice output
        print('')
    return path_res_list, reversed_path_res_list, propagated_reversed_path_res_list


def compute_spectrum_slot_vs_bandwidth(bandwidth, spacing, bit_rate, slot_width=0.0125e12):
    """ Compute the number of required wavelengths and the M value (number of consumed slots)
    Each wavelength consumes one `spacing`, and the result is rounded up to consume a natural number of slots.

    >>> compute_spectrum_slot_vs_bandwidth(400e9, 50e9, 200e9)
    (2, 8)
    """
    number_of_wavelengths = ceil(bandwidth / bit_rate)
    total_number_of_slots = ceil(spacing / slot_width) * number_of_wavelengths
    return number_of_wavelengths, total_number_of_slots


def select_regenerator_place(req, path, equipment, network):
    """ given a path and a mode, cut a path so that each segment is feasible
    minimizes the number of segments
    """
    si = create_input_spectral_information(
        req.f_min, req.f_max, req.roll_off, req.baud_rate,
        req.power, req.spacing)
    regen_place = []
    fini = False
    while not fini:
        for i, el in enumerate(path):
            if isinstance(el, Roadm):
                si = el(si, degree=path[i + 1].uid)
            else:
                si = el(si)
            if el == path[-1]:
                fini = True
            if isinstance (el, Roadm):
                # place a Rx here to check performance
                opm_rx = Transceiver({'uid': 'opm_rx'})
                opm_rx(si)
                opm_rx.update_snr(req.tx_osnr, equipment['Roadm']['default'].add_drop_osnr)
                snr = round(mean(opm_rx.snr_01nm), 2)
                if snr < req.OSNR + equipment['SI']['default'].sys_margins:
                    if not regen_place or regen_place[-1] != path[previous_roadm_index].uid:
                        # place a regen on the previous node
                        # only if there is one regen attached there !
                        try:
                            # find previous roadm in network (because elem in path are copies, not the actual
                            # network elements)
                            roadm = next(n for n in network.nodes() if n.uid == path[previous_roadm_index].uid)
                            regen = next(n for n in network.successors(roadm) if isinstance(n, Regenerator))
                            regen_place.append(path[previous_roadm_index].uid)
                            path = path[previous_roadm_index:]
                            si = create_input_spectral_information(
                                req.f_min, req.f_max, req.roll_off, req.baud_rate,
                                req.power, req.spacing)
                            break
                        except StopIteration:
                            msg = f'could not find a regen in {path[previous_roadm_index].uid}'
                            print(msg)
                            LOGGER.warning(msg)
                            return []
                    elif regen_place and regen_place[-1] == path[previous_roadm_index].uid:
                        # the OMS is not feasible at all
                        return []
                else:
                    previous_roadm_index = i
            if isinstance(el, Regenerator):
                regen_spec = next(r for r in req.regen_list if r['regen_uid'] == el.uid)
                si = create_input_spectral_information(
                    regen_spec['f_min'], regen_spec['f_max'], regen_spec['roll_off'], regen_spec['baud_rate'],
                    regen_spec['power'], regen_spec['spacing'])
    return regen_place


def place_regenerator(path, req, regen_place, network, equipment):
    """ Update the given path with the regenerators selected in the regen_place
    """
    regen_path = []
    req.regen_list = []
    for node in path:
        regen_path.append(node)
        if node.uid in regen_place:
            # node must be a roadm: then insert the regen
            if isinstance(node, Roadm):
                roadm = next(n for n in network.nodes() if n.uid == node.uid)
                regen = next(n for n in network.successors(roadm) if isinstance(n, Regenerator))
                regen_path.extend([regen, roadm])
                regen_dict = {'regen_uid': regen.uid,
                              'trx_type': req.trx_type,
                              'trx_mode': req.trx_mode,
                              'spacing': req.spacing}
                regen_params = trx_mode_params(equipment, req.trx_type, req.trx_mode, True)
                regen_params['spacing'] = req.spacing
                regen_params['request_id'] = req.request_id
                regen_params['trx_type'] = req.trx_type
                regen_params['trx_mode'] = req.trx_mode
                regen_dict.update(regen_params)
                req.regen_list.append(regen_dict)
            else:
                msg = f'Place of regeneration not supported in {node.uid}. Should be a ROADM node.'
                print(msg)
                LOGGER.warning(msg)
    return regen_path
