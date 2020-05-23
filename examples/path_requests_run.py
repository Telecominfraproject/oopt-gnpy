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
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps
from numpy import mean
from gnpy.core import ansi_escapes
from gnpy.core.utils import automatic_nch
from gnpy.core.network import build_network
from gnpy.core.elements import Roadm
from gnpy.core.utils import lin2db
from gnpy.core.exceptions import (ConfigurationError, EquipmentConfigError, NetworkTopologyError,
                                  ServiceError, DisjunctionError)
from gnpy.topology.request import (ResultElement,
                                   propagate, jsontocsv, Disjunction, compute_path_dsjctn,
                                   requests_aggregation, propagate_and_optimize_mode,
                                   BLOCKING_NOPATH, BLOCKING_NOMODE,
                                   find_reversed_path, correct_json_route_list)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import load_equipment, load_network, load_requests, save_network, requests_from_json, disjunctions_from_json
from copy import deepcopy
from math import ceil

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

def main(args):
    """ main function that calls all functions
    """
    LOGGER.info(f'Computing path requests {args.service_filename} into JSON format')
    print(f'{ansi_escapes.blue}Computing path requests {args.service_filename} into JSON format{ansi_escapes.reset}')
    # for debug
    # print( args.eqpt_filename)

    try:
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

    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(args.network_filename, network)
    oms_list = build_oms_list(network, equipment)

    try:
        data = load_requests(args.service_filename, equipment, bidir=args.bidir, network=network, network_filename=args.network_filename)
        rqs = requests_from_json(data, equipment)
    except ServiceError as this_e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {this_e}')
        exit(1)
    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        LOGGER.critical(msg)
        exit()
    rqs = correct_json_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)

    print(f'{ansi_escapes.blue}List of disjunctions{ansi_escapes.reset}')
    print(dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = correct_disjn(dsjn)

    # Aggregate demands with same exact constraints
    print(f'{ansi_escapes.blue}Aggregating similar requests{ansi_escapes.reset}')

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
    print(rqs)

    print(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    try:
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    except DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        exit(1)

    print(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = \
        compute_path_with_disjunction(network, equipment, rqs, pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    print(f'{ansi_escapes.blue}Result summary{ansi_escapes.reset}')
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
    print(f'{ansi_escapes.yellow}Result summary shows mean SNR and OSNR (average over all channels){ansi_escapes.reset}')

    if args.output:
        result = []
        # assumes that list of rqs and list of propgatedpths have same order
        for i, pth in enumerate(propagatedpths):
            result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
        temp = path_result_json(result)
        fnamecsv = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.csv'
        fnamejson = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.json'
        with open(fnamejson, 'w', encoding='utf-8') as fjson:
            fjson.write(dumps(path_result_json(result), indent=2, ensure_ascii=False))
            with open(fnamecsv, "w", encoding='utf-8') as fcsv:
                jsontocsv(temp, equipment, fcsv)
                print('\x1b[1;34;40m'+f'saving in {args.output} and {fnamecsv}'+ '\x1b[0m')


if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(ARGS.verbose, DEBUG))
    main(ARGS)
