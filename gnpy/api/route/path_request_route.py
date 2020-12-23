# coding: utf-8
import os
from pathlib import Path

from flask import request

from gnpy.api import app
from gnpy.api.exception.equipment_error import EquipmentError
from gnpy.api.exception.topology_error import TopologyError
from gnpy.api.service import topology_service
from gnpy.api.service.equipment_service import EquipmentService
from gnpy.api.service.path_request_service import path_requests_run
from gnpy.tools.json_io import _equipment_from_json, network_from_json
from gnpy.topology.request import ResultElement

_examples_dir = Path(__file__).parent.parent.parent / 'example-data'


@app.route('/api/v1/path-computation', methods=['POST'])
def compute_path(equipment_service: EquipmentService):
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
    elif 'gnpy-api:topology_id' in data:
        equipment = equipment_service.get_equipment(data['gnpy-api:equipment_id'])
    else:
        raise EquipmentError('No equipment found in request')
    equipment = _equipment_from_json(equipment,
                                     os.path.join(_examples_dir, 'std_medium_gain_advanced_config.json'))
    network = network_from_json(topology, equipment)

    propagatedpths, reversed_propagatedpths, rqs = path_requests_run(service, network, equipment)
    # Generate the output
    result = []
    # assumes that list of rqs and list of propgatedpths have same order
    for i, pth in enumerate(propagatedpths):
        result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
    return {"result": {"response": [n.json for n in result]}}, 201
