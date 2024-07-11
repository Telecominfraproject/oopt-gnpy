# coding: utf-8
import http
import json

from flask import request

from gnpy.api import app
from gnpy.api.exception.topology_error import TopologyError
from gnpy.api.model.result import Result
from gnpy.api.service import topology_service

TOPOLOGY_BASE_PATH = '/api/v1/topologies'
TOPOLOGY_ID_PATH = TOPOLOGY_BASE_PATH + '/<topology_id>'


@app.route(TOPOLOGY_BASE_PATH, methods=['POST'])
def create_topology():
    if not request.is_json:
        raise TopologyError('Request body is not json')
    topology_identifier = topology_service.save_topology(request.json)
    response = Result(message='Topology creation ok', description=topology_identifier)
    return json.dumps(response.__dict__), 201, {'location': TOPOLOGY_BASE_PATH + '/' + topology_identifier}


@app.route(TOPOLOGY_ID_PATH, methods=['PUT'])
def update_topology(topology_id):
    if not request.is_json:
        raise TopologyError('Request body is not json')
    topology_identifier = topology_service.update_topology(request.json, topology_id)
    response = Result(message='Topology update ok', description=topology_identifier)
    return json.dumps(response.__dict__), http.HTTPStatus.OK, {
        'location': TOPOLOGY_BASE_PATH + '/' + topology_identifier}


@app.route(TOPOLOGY_ID_PATH, methods=['GET'])
def get_topology(topology_id):
    return topology_service.get_topology(topology_id), http.HTTPStatus.OK


@app.route(TOPOLOGY_ID_PATH, methods=['DELETE'])
def delete_topology(topology_id):
    topology_service.delete_topology(topology_id)
    return '', http.HTTPStatus.NO_CONTENT
