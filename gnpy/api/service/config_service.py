# coding: utf-8
import configparser
import os

from flask import current_app

from gnpy.api.exception.config_error import ConfigError


def init_config(properties_file_path: str = os.path.join(os.path.dirname(__file__),
                                                         'properties.ini')) -> configparser.ConfigParser:
    """
    Read config from properties_file_path
    @param properties_file_path: the properties file to read
    @return: config parser
    """
    if not os.path.exists(properties_file_path):
        raise ConfigError('Properties file does not exist ' + properties_file_path)
    config = configparser.ConfigParser()
    config.read(properties_file_path)
    return config


def get_topology_dir() -> str:
    """
    Get the base dir where topologies are saved
    @return: the directory of topologies
    """
    return current_app.config['properties'].get('DIRECTORY', 'topology')


def get_equipment_dir() -> str:
    """
    Get the base dir where equipments are saved
    @return: the directory of equipments
    """
    return current_app.config['properties'].get('DIRECTORY', 'equipment')
