#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.core.exceptions: Exceptions thrown by other gnpy modules
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.core.exceptions
====================

Exceptions thrown by other gnpy modules
"""


class ConfigurationError(Exception):
    """User-provided configuration contains an error"""


class EquipmentConfigError(ConfigurationError):
    """Incomplete or wrong configuration within the equipment library"""


class NetworkTopologyError(ConfigurationError):
    """Topology of user-provided network is wrong"""


class ServiceError(Exception):
    """Service of user-provided request is wrong"""


class DisjunctionError(ServiceError):
    """Disjunction of user-provided request can not be satisfied"""


class SpectrumError(Exception):
    """Spectrum errors of the program"""


class ParametersError(ConfigurationError):
    """Incomplete or wrong configurations within parameters json"""
