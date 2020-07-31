# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors

"""
Scaling factors for unit conversion
===================================

In YANG, the data model defines units for each possible value explicitly.
This makes it possible for users to input data using the customary, common units.
The :py:mod:`gnpy.yang.conversion` module holds scaling factors for conversion of SI units into YANG units and back.
By convention, each items is used for multiplication when going from YANG to the legacy JSON.
When converting from legacy JSON to YANG, use division.
"""

import math

FIBER_DISPERSION = 1e-6
FIBER_DISPERSION_SLOPE = 1e3
FIBER_GAMMA = 1e-3
FIBER_PMD_COEF = 1e-14 * math.sqrt(10)
THZ = 1e12
GIGA = 1000 * 1000 * 1000
