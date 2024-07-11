# coding: utf-8
import http
import json

from flask import request

from gnpy.api import app
from gnpy.api.exception.equipment_error import EquipmentError
from gnpy.api.model.result import Result
from gnpy.api.service.equipment_service import EquipmentService

EQUIPMENT_BASE_PATH = '/api/v1/equipments'
EQUIPMENT_ID_PATH = EQUIPMENT_BASE_PATH + '/<equipment_id>'


@app.route(EQUIPMENT_BASE_PATH, methods=['POST'])
def create_equipment(equipment_service: EquipmentService):
    if not request.is_json:
        raise EquipmentError('Request body is not json')
    equipment_identifier = equipment_service.save_equipment(request.json)
    response = Result(message='Equipment creation ok', description=equipment_identifier)
    return json.dumps(response.__dict__), 201, {'location': EQUIPMENT_BASE_PATH + '/' + equipment_identifier}


@app.route(EQUIPMENT_ID_PATH, methods=['PUT'])
def update_equipment(equipment_id, equipment_service: EquipmentService):
    if not request.is_json:
        raise EquipmentError('Request body is not json')
    equipment_identifier = equipment_service.update_equipment(request.json, equipment_id)
    response = Result(message='Equipment update ok', description=equipment_identifier)
    return json.dumps(response.__dict__), http.HTTPStatus.OK, {
        'location': EQUIPMENT_BASE_PATH + '/' + equipment_identifier}


@app.route(EQUIPMENT_ID_PATH, methods=['DELETE'])
def delete_equipment(equipment_id, equipment_service: EquipmentService):
    equipment_service.delete_equipment(equipment_id)
    return '', http.HTTPStatus.NO_CONTENT
