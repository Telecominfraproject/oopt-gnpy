#! /bin/usr/python3

'''
gnpy.core.node
==============

This module contains the base class for a network element.

Strictly, a network element is any callable which accepts an immutable
.info.SpectralInformation object and returns a .info.SpectralInformation object
(a copy.)

Network elements MUST implement two attributes .uid and .name representing a
unique identifier and a printable name.

This base class provides a mode convenient way to define a network element
via subclassing.
'''


from uuid import uuid4
from gnpy.core.utils import load_json


class ConfigStruct:

    def __init__(self, **config):
        if config is None:
            return None
        if 'config_from_json' in config:
            json_config = load_json(config['config_from_json'])
            self.set_config_attr(json_config)

        self.set_config_attr(config)

    def set_config_attr(self, config):
        for k, v in config.items():
            setattr(self, k, ConfigStruct(**v)
                    if isinstance(v, dict) else v)

    def __repr__(self):
        return f'{self.__dict__}'


class Node:

    def __init__(self, config=None):
        self.config = ConfigStruct(**config)
        if self.config is None or not hasattr(self.config, 'uid'):
            self.uid = uuid4()
        else:
            self.uid = self.config.uid
        if hasattr(self.config, 'params'):
            self.params = self.config.params
        if hasattr(self.config, 'metadata'):
            self.metadata = self.config.metadata
        if hasattr(self.config, 'operational'):
            self.operational = self.config.operational

    @property
    def coords(self):
        return tuple(self.lng, self.lat)

    @property
    def location(self):
        return self.config.metadata.location

    @property
    def loc(self):  # Aliases .location
        return self.location

    @property
    def lng(self):
        return self.config.metadata.location.longitude

    @property
    def lat(self):
        return self.config.metadata.location.latitude
