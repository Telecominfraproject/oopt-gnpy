#! /bin/usr/python3
# -*- coding: utf-8 -*-

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
from collections import namedtuple

class Location(namedtuple('Location', 'latitude longitude city region')):
    def __new__(cls, latitude=0, longitude=0, city=None, region=None):
        return super().__new__(cls, latitude, longitude, city, region)

class Node:
    def __init__(self, uid, name=None, params=None, metadata={'location':{}}, operational=None):
        if name is None:
            name = uid
        self.uid, self.name = uid, name
        if metadata and not isinstance(metadata.get('location'), Location):
            metadata['location'] = Location(**metadata.pop('location', {}))
        self.params, self.metadata, self.operational = params, metadata, operational

    @property
    def coords(self):
        return (self.lng, self.lat)

    @property
    def location(self):
        return self.metadata['location']
    loc = location

    @property
    def longitude(self):
        return self.location.longitude
    lng = longitude

    @property
    def latitude(self):
        return self.location.latitude
    lat = latitude
