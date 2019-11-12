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
from numpy import mean
from gnpy.core.service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.utils import load_json
from gnpy.core.network import load_network, build_network, save_network, network_from_json
from gnpy.core.equipment import load_equipment, trx_mode_params, automatic_nch
from gnpy.core.elements import Transceiver, Roadm
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.request import (Path_request, Result_element,
                               propagate, jsontocsv, Disjunction, compute_path_dsjctn,
                               requests_aggregation, propagate_and_optimize_mode,
                               BLOCKING_NOPATH, BLOCKING_NOMODE,
                               find_reversed_path)
from gnpy.core.exceptions import (ConfigurationError, EquipmentConfigError, NetworkTopologyError,
                                  ServiceError, DisjunctionError)
import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.core.spectrum_assignment import (build_oms_list, pth_assign_spectrum)
from copy import copy, deepcopy
from textwrap import dedent
from math import ceil

from flask import Flask, jsonify, make_response, request
from flask_restful import Api, Resource, reqparse, fields

#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

LOGGER = getLogger(__name__)

PARSER = ArgumentParser(description='A function that computes performances for a list of ' +
                        'services provided in a json file or an excel sheet.')
PARSER.add_argument('network_filename', nargs='?', type=Path,\
                    default=Path(__file__).parent / 'meshTopologyExampleV2.xls',\
                    help='input topology file in xls or json')
PARSER.add_argument('service_filename', nargs='?', type=Path,\
                    default=Path(__file__).parent / 'meshTopologyExampleV2.xls',\
                    help='input service file in xls or json')
PARSER.add_argument('eqpt_filename', nargs='?', type=Path,\
                    default=Path(__file__).parent / 'eqpt_config.json',\
                    help='input equipment library in json. Default is eqpt_config.json')
PARSER.add_argument('-bi', '--bidir', action='store_true',\
                    help='considers that all demands are bidir')
PARSER.add_argument('-v', '--verbose', action='count', default=0,\
                    help='increases verbosity for each occurence')
PARSER.add_argument('-o', '--output', type=Path)
PARSER.add_argument('-r', '--rest', action='count', default=0, help='use the REST API')

NETWORK_FILENAME = 'topoDemov1.json' #'disagregatedTopoDemov1.json' #

APP = Flask(__name__, static_url_path="")
API = Api(APP)

def requests_from_json(json_data, equipment):
    """ converts the json data into a list of requests elements
    """
    requests_list = []

    for req in json_data['path-request']:
        # init all params from request
        params = {}
        params['request_id'] = req['request-id']
        params['source'] = req['source']
        params['bidir'] = req['bidirectional']
        params['destination'] = req['destination']
        params['trx_type'] = req['path-constraints']['te-bandwidth']['trx_type']
        params['trx_mode'] = req['path-constraints']['te-bandwidth']['trx_mode']
        params['format'] = params['trx_mode']
        params['spacing'] = req['path-constraints']['te-bandwidth']['spacing']
        try:
            nd_list = req['explicit-route-objects']['route-object-include-exclude']
        except KeyError:
            nd_list = []
        params['nodes_list'] = [n['num-unnum-hop']['node-id'] for n in nd_list]
        params['loose_list'] = [n['num-unnum-hop']['hop-type'] for n in nd_list]
        # recover trx physical param (baudrate, ...) from type and mode
        # in trx_mode_params optical power is read from equipment['SI']['default'] and
        # nb_channel is computed based on min max frequency and spacing
        trx_params = trx_mode_params(equipment, params['trx_type'], params['trx_mode'], True)
        params.update(trx_params)
        # print(trx_params['min_spacing'])
        # optical power might be set differently in the request. if it is indicated then the
        # params['power'] is updated
        try:
            if req['path-constraints']['te-bandwidth']['output-power']:
                params['power'] = req['path-constraints']['te-bandwidth']['output-power']
        except KeyError:
            pass
        # same process for nb-channel
        f_min = params['f_min']
        f_max_from_si = params['f_max']
        try:
            if req['path-constraints']['te-bandwidth']['max-nb-of-channel'] is not None:
                nch = req['path-constraints']['te-bandwidth']['max-nb-of-channel']
                params['nb_channel'] = nch
                spacing = params['spacing']
                params['f_max'] = f_min + nch*spacing
            else:
                params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
        except KeyError:
            params['nb_channel'] = automatic_nch(f_min, f_max_from_si, params['spacing'])
        consistency_check(params, f_max_from_si)

        try:
            params['path_bandwidth'] = req['path-constraints']['te-bandwidth']['path_bandwidth']
        except KeyError:
            pass
        requests_list.append(Path_request(**params))
    return requests_list

def consistency_check(params, f_max_from_si):
    """ checks that the requested parameters are consistant (spacing vs nb channel,
        vs transponder mode...)
    """
    f_min = params['f_min']
    f_max = params['f_max']
    max_recommanded_nb_channels = automatic_nch(f_min, f_max, params['spacing'])
    if params['baud_rate'] is not None:
        #implicitely means that a mode is defined with min_spacing
        if params['min_spacing'] > params['spacing']:
            msg = f'Request {params["request_id"]} has spacing below transponder ' +\
                  f'{params["trx_type"]} {params["trx_mode"]} min spacing value ' +\
                  f'{params["min_spacing"]*1e-9}GHz.\nComputation stopped'
            print(msg)
            LOGGER.critical(msg)
            raise ServiceError(msg)
        if f_max > f_max_from_si:
            msg = dedent(f'''
            Requested channel number {params["nb_channel"]}, baud rate {params["baud_rate"]} GHz and requested spacing {params["spacing"]*1e-9}GHz 
            is not consistent with frequency range {f_min*1e-12} THz, {f_max*1e-12} THz, min recommanded spacing {params["min_spacing"]*1e-9}GHz.
            max recommanded nb of channels is {max_recommanded_nb_channels}
            Computation stopped.''')
            LOGGER.critical(msg)
            raise ServiceError(msg)


def disjunctions_from_json(json_data):
    """ reads the disjunction requests from the json dict and create the list
        of requested disjunctions for this set of requests
    """
    disjunctions_list = []
    try:
        temp_test = json_data['synchronization']
    except KeyError:
        temp_test = []
    if temp_test:
        for snc in json_data['synchronization']:
            params = {}
            params['disjunction_id'] = snc['synchronization-id']
            params['relaxable'] = snc['svec']['relaxable']
            params['link_diverse'] = 'link' in snc['svec']['disjointness']
            params['node_diverse'] = 'node' in snc['svec']['disjointness']
            params['disjunctions_req'] = snc['svec']['request-id-number']
            disjunctions_list.append(Disjunction(**params))

    return disjunctions_list


def load_requests(filename, eqpt_filename, bidir):
    """ loads the requests from a json or an excel file into a data string
    """
    if filename.suffix.lower() == '.xls':
        LOGGER.info('Automatically converting requests from XLS to JSON')
        try:
            json_data = convert_service_sheet(filename, eqpt_filename, bidir=bidir)
        except ServiceError as this_e:
            print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {this_e}')
            exit(1)
    else:
        with open(filename, encoding='utf-8') as my_f:
            json_data = loads(my_f.read())
    return json_data

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
        p_db = lin2db(pathreq.power*1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
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
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if total_path:
            if pathreq.baud_rate is not None:
                # means that at this point the mode was entered/forced by user and thus a
                # baud_rate was defined
                total_path = propagate(total_path, pathreq, equipment)
                temp_snr01nm = round(mean(total_path[-1].snr+lin2db(pathreq.baud_rate/(12.5e9))), 2)
                if temp_snr01nm < pathreq.OSNR:
                    msg = f'\tWarning! Request {pathreq.request_id} computed path from' +\
                          f' {pathreq.source} to {pathreq.destination} does not pass with' +\
                          f' {pathreq.tsp_mode}\n\tcomputedSNR in 0.1nm = {temp_snr01nm} ' +\
                          f'- required osnr {pathreq.OSNR}'
                    print(msg)
                    LOGGER.warning(msg)
                    pathreq.blocking_reason = 'MODE_NOT_FEASIBLE'
            else:
                total_path, mode = propagate_and_optimize_mode(total_path, pathreq, equipment)
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
                        pathreq.tsp_mode = mode['format']
                        pathreq.format = mode['format']
                        pathreq.OSNR = mode['OSNR']
                        pathreq.tx_osnr = mode['tx_osnr']
                        pathreq.bit_rate = mode['bit_rate']
                    # other blocking reason should not appear at this point
                except AttributeError:
                    pathreq.baud_rate = mode['baud_rate']
                    pathreq.tsp_mode = mode['format']
                    pathreq.format = mode['format']
                    pathreq.OSNR = mode['OSNR']
                    pathreq.tx_osnr = mode['tx_osnr']
                    pathreq.bit_rate = mode['bit_rate']

            # reversed path is needed for correct spectrum assignment
            reversed_path = find_reversed_path(pathlist[i])
            if pathreq.bidir:
                # only propagate if bidir is true, but needs the reversed path anyway for
                # correct spectrum assignment
                rev_p = deepcopy(reversed_path)

                print(f'\n\tPropagating Z to A direction {pathreq.destination} to {pathreq.source}')
                print(f'\tPath (roadsm) {[r.uid for r in rev_p if isinstance(r,Roadm)]}\n')
                propagated_reversed_path = propagate(rev_p, pathreq, equipment)
                temp_snr01nm = round(mean(propagated_reversed_path[-1].snr +\
                                          lin2db(pathreq.baud_rate/(12.5e9))), 2)
                if temp_snr01nm < pathreq.OSNR:
                    msg = f'\tWarning! Request {pathreq.request_id} computed path from' +\
                          f' {pathreq.source} to {pathreq.destination} does not pass with' +\
                          f' {pathreq.tsp_mode}\n' +\
                          f'\tcomputedSNR in 0.1nm = {temp_snr01nm} - required osnr {pathreq.OSNR}'
                    print(msg)
                    LOGGER.warning(msg)
                    # TODO selection of mode should also be on reversed direction !!
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

def correct_route_list(network, pathreqlist):
    """ prepares the format of route list of nodes to be consistant
        remove wrong names, remove endpoints
        also correct source and destination
    """
    anytype = [n.uid for n in network.nodes()]
    # TODO there is a problem of identification of fibers in case of parallel fibers
    # between two adjacent roadms so fiber constraint is not supported
    transponders = [n.uid for n in network.nodes() if isinstance(n, Transceiver)]
    for pathreq in pathreqlist:
        for i, n_id in enumerate(pathreq.nodes_list):
            # replace possibly wrong name with a formated roadm name
            # print(n_id)
            if n_id not in anytype:
                # find nodes name that include constraint among all possible names except
                # transponders (not yet supported as constraints).
                nodes_suggestion = [uid for uid in anytype \
                    if n_id.lower() in uid.lower() and uid not in transponders]
                if pathreq.loose_list[i] == 'LOOSE':
                    if len(nodes_suggestion) > 0:
                        new_n = nodes_suggestion[0]
                        print(f'invalid route node specified:\
                        \n\'{n_id}\', replaced with \'{new_n}\'')
                        pathreq.nodes_list[i] = new_n
                    else:
                        print(f'\x1b[1;33;40m'+f'invalid route node specified \'{n_id}\',' +\
                              f' could not use it as constraint, skipped!'+'\x1b[0m')
                        pathreq.nodes_list.remove(n_id)
                        pathreq.loose_list.pop(i)
                else:
                    msg = f'\x1b[1;33;40m'+f'could not find node: {n_id} in network topology.' +\
                          f' Strict constraint can not be applied.' + '\x1b[0m'
                    LOGGER.critical(msg)
                    raise ValueError(msg)
        if pathreq.source not in transponders:
            msg = f'\x1b[1;31;40m' + f'Request: {pathreq.request_id}: could not find' +\
                  f' transponder source: {pathreq.source}.'+'\x1b[0m'
            LOGGER.critical(msg)
            print(f'{msg}\nComputation stopped.')
            raise ServiceError(msg)

        if pathreq.destination not in transponders:
            msg = f'\x1b[1;31;40m'+f'Request: {pathreq.request_id}: could not find' +\
                  f' transponder destination: {pathreq.destination}.'+'\x1b[0m'
            LOGGER.critical(msg)
            print(f'{msg}\nComputation stopped.')
            raise ServiceError(msg)

        # TODO remove endpoints from this list in case they were added by the user
        # in the xls or json files
    return pathreqlist

def correct_disjn(disjn):
    """ clean disjunctions to remove possible repetition
    """
    local_disjn = disjn.copy()
    for elem in local_disjn:
        for dis_elem in local_disjn:
            if set(elem.disjunctions_req) == set(dis_elem.disjunctions_req) and\
             elem.disjunction_id != dis_elem.disjunction_id:
                local_disjn.remove(dis_elem)
    return local_disjn


def path_result_json(pathresult):
    """ create the response dictionnary
    """
    data = {
        'response': [n.json for n in pathresult]
    }
    return data

def compute_requests(network, data, equipment):
    """ Main program calling functions
    """
    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(ARGS.network_filename, network)

    oms_list = build_oms_list(network, equipment)

    try:
        rqs = requests_from_json(data, equipment)
    except ServiceError as this_e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {this_e}')
        raise this_e
    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        LOGGER.critical(msg)
        raise ServiceError(msg)
    try:
        rqs = correct_route_list(network, rqs)
    except ServiceError as this_e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {this_e}')
        raise this_e
        #exit(1)
    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)

    print('\x1b[1;34;40m' + f'List of disjunctions' + '\x1b[0m')
    print(dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = correct_disjn(dsjn)

    # Aggregate demands with same exact constraints
    print('\x1b[1;34;40m' + f'Aggregating similar requests' + '\x1b[0m')

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    print('\x1b[1;34;40m' + 'The following services have been requested:' + '\x1b[0m')
    print(rqs)

    print('\x1b[1;34;40m' + f'Computing all paths with constraints' + '\x1b[0m')
    try:
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    except DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        raise this_e

    print('\x1b[1;34;40m' + f'Propagating on selected path' + '\x1b[0m')
    propagatedpths, reversed_pths, reversed_propagatedpths = \
        compute_path_with_disjunction(network, equipment, rqs, pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    print('\x1b[1;34;40m'+f'Result summary'+ '\x1b[0m')
    header = ['req id', '  demand', '  snr@bandwidth A-Z (Z-A)', '  snr@0.1nm A-Z (Z-A)',\
              '  Receiver minOSNR', '  mode', '  Gbit/s', '  nb of tsp pairs',\
              'N,M or blocking reason']
    data = []
    data.append(header)
    for i, this_p in enumerate(propagatedpths):
        rev_pth = reversed_propagatedpths[i]
        if rev_pth and this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)} ({round(mean(rev_pth[-1].snr),2)})'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' +\
                   f' ({round(mean(rev_pth[-1].snr_01nm),2)})'
        elif this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)}'
            psnr = f'{round(mean(this_p[-1].snr_01nm),2)}'

        try :
            if rqs[i].blocking_reason in  BLOCKING_NOPATH:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} :',\
                        f'-', f'-', f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',\
                        f'-', f'{rqs[i].blocking_reason}']
            else:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                        psnr, f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',\
                        f'-', f'{rqs[i].blocking_reason}']
        except AttributeError:
            line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                    psnr, f'{rqs[i].OSNR}', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',\
                    f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate) }', f'({rqs[i].N},{rqs[i].M})']
        data.append(line)

    col_width = max(len(word) for row in data for word in row[2:])   # padding
    firstcol_width = max(len(row[0]) for row in data)   # padding
    secondcol_width = max(len(row[1]) for row in data)   # padding
    for row in data:
        firstcol = ''.join(row[0].ljust(firstcol_width))
        secondcol = ''.join(row[1].ljust(secondcol_width))
        remainingcols = ''.join(word.center(col_width, ' ') for word in row[2:])
        print(f'{firstcol} {secondcol} {remainingcols}')
    print('\x1b[1;33;40m'+f'Result summary shows mean SNR and OSNR (average over all channels)' +\
          '\x1b[0m')

    return propagatedpths, reversed_propagatedpths, rqs


def launch_cli(network, data, equipment):
    """ Compute requests using network, data and equipment with client line interface
    """
    propagatedpths, reversed_propagatedpths, rqs = compute_requests(network, data, equipment)
    #Generate the output
    if ARGS.output :
        result = []
        # assumes that list of rqs and list of propgatedpths have same order
        for i, pth in enumerate(propagatedpths):
            result.append(Result_element(rqs[i], pth, reversed_propagatedpths[i]))
        temp = path_result_json(result)
        fnamecsv = f'{str(ARGS.output)[0:len(str(ARGS.output))-len(str(ARGS.output.suffix))]}.csv'
        fnamejson = f'{str(ARGS.output)[0:len(str(ARGS.output))-len(str(ARGS.output.suffix))]}.json'
        with open(fnamejson, 'w', encoding='utf-8') as fjson:
            fjson.write(dumps(path_result_json(result), indent=2, ensure_ascii=False))
            with open(fnamecsv, "w", encoding='utf-8') as fcsv:
                jsontocsv(temp, equipment, fcsv)
                print('\x1b[1;34;40m'+f'saving in {ARGS.output} and {fnamecsv}'+ '\x1b[0m')

class GnpyAPI(Resource):
    """ Compute requests using network, data and equipment with rest api
    """
    def get(self):
        return {"ping": True}, 200

    def post(self):
        data = request.get_json()
        equipment = load_equipment('examples/2019-demo-equipment.json')
        topo_json = load_json('examples/2019-demo-topology.json')
        network = network_from_json(topo_json, equipment)
        try:
            propagatedpths, reversed_propagatedpths, rqs = compute_requests(network, data, equipment)
            # Generate the output
            result = []
            #assumes that list of rqs and list of propgatedpths have same order
            for i, pth in enumerate(propagatedpths):
                result.append(Result_element(rqs[i], pth, reversed_propagatedpths[i]))

            return {"result":path_result_json(result)}, 201
        except ServiceError as this_e:
            msg = f'Service error: {this_e}'
            return {"result": msg}, 400

API.add_resource(GnpyAPI, '/gnpy-experimental')

def main(args):
    """ main function that calls all functions
    """
    LOGGER.info(f'Computing path requests {args.service_filename} into JSON format')
    print('\x1b[1;34;40m' +\
          f'Computing path requests {args.service_filename} into JSON format'+ '\x1b[0m')
    # for debug
    # print( args.eqpt_filename)

    try:
        data = load_requests(args.service_filename, args.eqpt_filename, args.bidir)
        equipment = load_equipment(args.eqpt_filename)
        network = load_network(args.network_filename, equipment)
    except EquipmentConfigError as this_e:
        print(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {this_e}')
        exit(1)
    except NetworkTopologyError as this_e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {this_e}')
        exit(1)
    except ConfigurationError as this_e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {this_e}')
        exit(1)
    except ServiceError as this_e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {this_e}')
        exit(1)
    # input_str = raw_input("How will you use your program: c:[cli] , a:[api] ?")
    # print(input_str)
    #
    if ((args.rest == 1) and (args.output is None)):
        print('you have chosen the rest mode')
        APP.run(host='0.0.0.0', port=5000, debug=True)
    elif ((args.rest > 1) or ((args.rest == 1) and (args.output is not None))):
        print('command is not well formulated')
    else:
        launch_cli(network, data, equipment)

if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(ARGS.verbose, DEBUG))
    main(ARGS)
