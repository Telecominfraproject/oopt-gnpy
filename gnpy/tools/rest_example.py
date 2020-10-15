#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.rest_example
=======================

GNPy as a rest API example
'''

import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

import werkzeug
from flask import Flask, request
from numpy import mean
from werkzeug.exceptions import InternalServerError

import gnpy.core.ansi_escapes as ansi_escapes
import gnpy.core.exceptions as exceptions
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.tools.json_io import requests_from_json, disjunctions_from_json, _equipment_from_json, network_from_json
from gnpy.topology.request import (ResultElement, compute_path_dsjctn, requests_aggregation,
                                   correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum

_logger = logging.getLogger(__name__)
_examples_dir = Path(__file__).parent.parent / 'example-data'
_reaesc = re.compile(r'\x1b[^m]*m')
app = Flask(__name__)


@app.route('/api/v1/path-computation', methods=['POST'])
def compute_path():
    data = request.json
    service = data['service']
    topology = data['topology']
    equipment = _equipment_from_json(data['equipment'],
                                     os.path.join(_examples_dir, 'std_medium_gain_advanced_config.json'))
    network = network_from_json(topology, equipment)

    propagatedpths, reversed_propagatedpths, rqs = path_requests_run(service, network, equipment)
    # Generate the output
    result = []
    # assumes that list of rqs and list of propgatedpths have same order
    for i, pth in enumerate(propagatedpths):
        result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
    return {"result": {"response": [n.json for n in result]}}, 201


@app.route('/api/v1/status', methods=['GET'])
def api_status():
    return {"version": "v1", "status": "ok"}, 200


def _init_logger():
    handler = RotatingFileHandler('api.log', maxBytes=1024 * 1024, backupCount=5, encoding='utf-8')
    ch = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[handler, ch],
                        format="%(asctime)s %(levelname)s %(name)s(%(lineno)s) [%(threadName)s - %(thread)d] - %("
                               "message)s")


def path_requests_run(service, network, equipment):
    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    rqs = requests_from_json(service, equipment)

    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        _logger.critical(msg)
        raise ValueError('Requests id ' + all_ids + ' are not unique')
    rqs = correct_json_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(service)

    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = deduplicate_disjunctions(dsjn)

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    _logger.info(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}' + str(rqs))

    _logger.info(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)

    _logger.info(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs,
                                                                                           pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)
    return propagatedpths, reversed_propagatedpths, rqs


def common_error_handler(exception):
    """

    :type exception: Exception

    """
    status_code = 500
    if not isinstance(exception, werkzeug.exceptions.HTTPException):
        exception = werkzeug.exceptions.InternalServerError()
        exception.description = "Something went wrong on our side."

    response = {
        'message': exception.name,
        'description': exception.description,
        'code': exception.code
    }

    return werkzeug.Response(response=json.dumps(response), status=status_code, mimetype='application/json')


def bad_request_handler(exception):
    response = {
        'message': 'bad request',
        'description': _reaesc.sub('', str(exception)),
        'code': 400
    }
    return werkzeug.Response(response=json.dumps(response), status=400, mimetype='application/json')


def _init_app():
    app.register_error_handler(KeyError, bad_request_handler)
    app.register_error_handler(TypeError, bad_request_handler)
    app.register_error_handler(ValueError, bad_request_handler)
    app.register_error_handler(exceptions.ConfigurationError, bad_request_handler)
    app.register_error_handler(exceptions.DisjunctionError, bad_request_handler)
    app.register_error_handler(exceptions.EquipmentConfigError, bad_request_handler)
    app.register_error_handler(exceptions.NetworkTopologyError, bad_request_handler)
    app.register_error_handler(exceptions.ServiceError, bad_request_handler)
    app.register_error_handler(exceptions.SpectrumError, bad_request_handler)
    app.register_error_handler(exceptions.ParametersError, bad_request_handler)
    app.register_error_handler(AssertionError, bad_request_handler)
    app.register_error_handler(InternalServerError, common_error_handler)
    for error_code in werkzeug.exceptions.default_exceptions:
        app.register_error_handler(error_code, common_error_handler)


def main():
    _init_logger()
    _init_app()
    app.run(port=8080)


if __name__ == '__main__':
    main()
