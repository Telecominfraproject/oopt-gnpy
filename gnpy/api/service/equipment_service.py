# coding: utf-
import json
import os
import uuid

from injector import Inject

from gnpy.api.exception.equipment_error import EquipmentError
from gnpy.api.service import config_service
from gnpy.api.service.encryption_service import EncryptionService


class EquipmentService:

    def __init__(self, encryption_service: EncryptionService):
        self.encryption = encryption_service

    def save_equipment(self, equipment):
        """
        Save equipment to file.
        @param equipment: json content
        @return: a UUID identifier to identify the equipment
        """
        equipment_identifier = str(uuid.uuid4())
        # TODO: validate json content
        self._write_equipment(equipment, equipment_identifier)
        return equipment_identifier

    def update_equipment(self, equipment, equipment_identifier):
        """
        Update equipment with identifier equipment_identifier.
        @param equipment_identifier: the identifier of the equipment to be updated
        @param equipment: json content
        @return: a UUID identifier to identify the equipment
        """
        # TODO: validate json content
        self._write_equipment(equipment, equipment_identifier)
        return equipment_identifier

    def _write_equipment(self, equipment, equipment_identifier):
        equipment_dir = config_service.get_equipment_dir()
        with(open(os.path.join(equipment_dir, '.'.join([equipment_identifier, 'json'])), 'wb')) as file:
            file.write(self.encryption.encrypt(json.dumps(equipment).encode()))

    def get_equipment(self, equipment_id: str) -> dict:
        """
        Get the equipment with id equipment_id
        @param equipment_id:
        @return: the equipment in json format
        """
        equipment_dir = config_service.get_equipment_dir()
        equipment_file = os.path.join(equipment_dir, '.'.join([equipment_id, 'json']))
        if not os.path.exists(equipment_file):
            raise EquipmentError('Equipment with id {} does not exist '.format(equipment_id))
        with(open(equipment_file, 'rb')) as file:
            return json.loads(self.encryption.decrypt(file.read()))

    def delete_equipment(self, equipment_id: str):
        """
        Delete equipment with id equipment_id
        @param equipment_id:
        """
        equipment_dir = config_service.get_equipment_dir()
        equipment_file = os.path.join(equipment_dir, '.'.join([equipment_id, 'json']))
        if os.path.exists(equipment_file):
            os.remove(equipment_file)
