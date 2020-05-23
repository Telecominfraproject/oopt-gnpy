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
from gnpy.core.utils import lin2db
import gnpy.core.exceptions as exceptions
from gnpy.topology.request import (ResultElement, jsontocsv, compute_path_dsjctn, requests_aggregation,
                                   BLOCKING_NOPATH, correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import load_equipment, load_network, load_requests, save_network, requests_from_json, disjunctions_from_json
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
    except exceptions.EquipmentConfigError as e:
        print(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
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
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
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
    dsjn = deduplicate_disjunctions(dsjn)

    # Aggregate demands with same exact constraints
    print(f'{ansi_escapes.blue}Aggregating similar requests{ansi_escapes.reset}')

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
    print(rqs)

    print(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    try:
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    except exceptions.DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        exit(1)

    print(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)
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
