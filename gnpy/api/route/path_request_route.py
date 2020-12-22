import os
from pathlib import Path

from flask import request

from gnpy.api import app
from gnpy.api.service.path_request_service import path_requests_run
from gnpy.tools.json_io import _equipment_from_json, network_from_json
from gnpy.topology.request import ResultElement

_examples_dir = Path(__file__).parent.parent.parent / 'example-data'


@app.route('/api/v1/path-computation', methods=['POST'])
def compute_path():
    data = request.json
    service = data['gnpy-api:service']
    topology = data['gnpy-api:topology']
    equipment = _equipment_from_json(data['gnpy-api:equipment'],
                                     os.path.join(_examples_dir, 'std_medium_gain_advanced_config.json'))
    network = network_from_json(topology, equipment)

    propagatedpths, reversed_propagatedpths, rqs = path_requests_run(service, network, equipment)
    # Generate the output
    result = []
    # assumes that list of rqs and list of propgatedpths have same order
    for i, pth in enumerate(propagatedpths):
        result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
    return {"result": {"response": [n.json for n in result]}}, 201
