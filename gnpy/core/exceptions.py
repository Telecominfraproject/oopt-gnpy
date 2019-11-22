#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.exceptions
====================

Exceptions thrown by other gnpy modules
'''


class ConfigurationError(Exception):
    '''User-provided configuration contains an error'''

class EquipmentConfigError(ConfigurationError):
    '''Incomplete or wrong configuration within the equipment library'''

class NetworkTopologyError(ConfigurationError):
    '''Topology of user-provided network is wrong'''

class ParametersError(ConfigurationError):
    '''Incomplete or wrong configurations within parameters json'''


