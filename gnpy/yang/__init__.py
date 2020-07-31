# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors

'''
Working with YANG-encoded data
'''

from pathlib import Path


def model_path() -> Path:
    '''Filesystem path to TIP's own YANG models'''
    return Path(__file__).parent / 'tip'


def external_path() -> Path:
    '''Filesystem path to third-party YANG models that are shipped with GNPy'''
    return Path(__file__).parent / 'ext'


def _yang_library() -> Path:
    '''Filesystem path the the ietf-yanglib JSON file'''
    return Path(__file__).parent / 'yanglib.json'
