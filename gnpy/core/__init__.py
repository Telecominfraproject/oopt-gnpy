#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
#                _____ ___ ____    ____  ____  _____                   #
#               |_   _|_ _|  _ \  |  _ \/ ___|| ____|                  #
#                 | |  | || |_) | | |_) \___ \|  _|                    #
#                 | |  | ||  __/  |  __/ ___) | |___                   #
#                 |_| |___|_|     |_|   |____/|_____|                  #
#                                                                      #
#                == Physical Simulation Environment ==                 #
#                                                                      #
########################################################################


'''
gnpy route planning and optimization library
============================================

gnpy is a route planning and optimization library, written in Python, for
operators of large-scale mesh optical networks.

:copyright: © 2018, Telecom Infra Project
:license: BSD 3-Clause, see LICENSE for more details.
'''

from . import elements
from .execute import *
from .network import *
from .utils import *
