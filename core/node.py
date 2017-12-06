#! /bin/usr/python3

from uuid import uuid4

# helpers


class ConfigStruct:

    def __init__(self, **entries):
        if entries is None:
            return None
        for k, v in entries.items():
            setattr(self, k, ConfigStruct(**v) if isinstance(v, dict) else v)

    def __repr__(self):
        return f'{self.__dict__}'


class Node:

    def __init__(self, config=None):
        self.config = ConfigStruct(**config)
        if self.config is None or not hasattr(self.config, 'uid'):
            self.uid = uuid4()
        else:
            self.uid = self.config.uid

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
