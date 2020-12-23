# coding: utf-
import json
import os
import uuid

from gnpy.api.exception.topology_error import TopologyError
from gnpy.api.service import config_service


def save_topology(topology):
    """
    Save topology to file.
    @param topology: json content
    @return: a UUID identifier to identify the topology
    """
    topology_identifier = str(uuid.uuid4())
    # TODO: validate json content
    _write_topology(topology, topology_identifier)
    return topology_identifier


def update_topology(topology, topology_identifier):
    """
    Update topology with identifier topology_identifier.
    @param topology_identifier: the identifier of the topology to be updated
    @param topology: json content
    @return: a UUID identifier to identify the topology
    """
    # TODO: validate json content
    _write_topology(topology, topology_identifier)
    return topology_identifier


def _write_topology(topology, topology_identifier):
    topology_dir = config_service.get_topology_dir()
    with(open(os.path.join(topology_dir, '.'.join([topology_identifier, 'json'])), 'w')) as file:
        json.dump(topology, file)


def get_topology(topology_id: str) -> dict:
    """
    Get the topology with id topology_id
    @param topology_id:
    @return: the topology in json format
    """
    topology_dir = config_service.get_topology_dir()
    topology_file = os.path.join(topology_dir, '.'.join([topology_id, 'json']))
    if not os.path.exists(topology_file):
        raise TopologyError('Topology with id {} does not exist '.format(topology_id))
    with(open(topology_file, 'r')) as file:
        return json.load(file)


def delete_topology(topology_id: str):
    """
    Delete topology with id topology_id
    @param topology_id:
    """
    topology_dir = config_service.get_topology_dir()
    topology_file = os.path.join(topology_dir, '.'.join([topology_id, 'json']))
    if os.path.exists(topology_file):
        os.remove(topology_file)
