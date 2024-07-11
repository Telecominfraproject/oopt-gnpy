# coding: utf-8
import http
import os
from pathlib import Path

from flask import request

from gnpy.api import app
from gnpy.api.exception.equipment_error import EquipmentError
from gnpy.api.exception.topology_error import TopologyError
from gnpy.api.service import topology_service
from gnpy.api.service.equipment_service import EquipmentService
from gnpy.api.service.path_request_service import PathRequestService
from gnpy.tools.json_io import _equipment_from_json, network_from_json
from gnpy.topology.request import ResultElement

PATH_COMPUTATION_BASE_PATH = '/api/v1/path-computation'
AUTODESIGN_PATH = PATH_COMPUTATION_BASE_PATH + '/<path_computation_id>/autodesign'

_examples_dir = Path(__file__).parent.parent.parent / 'example-data'


@app.route(PATH_COMPUTATION_BASE_PATH, methods=['POST'])
def compute_path(equipment_service: EquipmentService, path_request_service: PathRequestService):
    data = request.json
    service = data['gnpy-api:service']
    if 'gnpy-api:topology' in data:
        topology = data['gnpy-api:topology']
    elif 'gnpy-api:topology_id' in data:
        topology = topology_service.get_topology(data['gnpy-api:topology_id'])
    else:
        raise TopologyError('No topology found in request')
    if 'gnpy-api:equipment' in data:
        equipment = data['gnpy-api:equipment']
    elif 'gnpy-api:equipment_id' in data:
        equipment = equipment_service.get_equipment(data['gnpy-api:equipment_id'])
    else:
        raise EquipmentError('No equipment found in request')
    equipment = _equipment_from_json(equipment,
                                     os.path.join(_examples_dir, 'std_medium_gain_advanced_config.json'))
    network = network_from_json(topology, equipment)

    propagatedpths, reversed_propagatedpths, rqs, path_computation_id = path_request_service.path_requests_run(service,
                                                                                                               network,
                                                                                                               equipment)
    # Generate the output
    result = []
    # assumes that list of rqs and list of propgatedpths have same order
    for i, pth in enumerate(propagatedpths):
        result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
    return {"result": {"response": [n.json for n in result]}}, 201, {
        'location': AUTODESIGN_PATH.replace('<path_computation_id>', path_computation_id)}


@app.route(AUTODESIGN_PATH, methods=['GET'])
def get_autodesign(path_computation_id, path_request_service: PathRequestService):
    return path_request_service.get_autodesign(path_computation_id), http.HTTPStatus.OK


@app.route(AUTODESIGN_PATH, methods=['DELETE'])
def delete_autodesign(path_computation_id, path_request_service: PathRequestService):
    path_request_service.delete_autodesign(path_computation_id)
    return '', http.HTTPStatus.NO_CONTENT
