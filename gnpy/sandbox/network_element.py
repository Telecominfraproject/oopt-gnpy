#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:08:37 2017

@author: briantaylor
"""
from uuid import uuid4


class NetworkElement:

    def __init__(self, **kwargs):
        """
        self.direction = [("E", "Z"), ("E", "Z"), ("E", "Z"), ("W", "Z")]
        self.port_mapping = [(1, 5), (2, 5), (3, 5), (4, 5)]
        self.uid = uuid4()
        self.coordinates = (29.9792, 31.1342)
        """
        try:
            for key in ('port_mapping', 'direction', 'coordinates', 'name',
                        'description', 'manufacturer', 'model', 'sn', 'id'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                else:
                    setattr(self, key, None)
                    # print('No Value defined for :', key)
                    # TODO: add logging functionality
        except KeyError as e:
            if 'name' in kwargs:
                s = kwargs['name']
                print('Missing Required Network Element Key!', 'name:=', s)
#           TODO Put log here instead of print
            print(e)
            raise

    def get_output_ports(self):
        """Translate the port mapping into list of output ports
        """
        return None

    def get_input_ports(self):
        """Translate the port mapping into list of output ports
        """
        return None

    def __repr__(self):
        return self.__class__.__name__
