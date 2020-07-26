# SPDX-License-Identifier: BSD-3-Clause
#
# Working with YANG-encoded data
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
from yangson import DataModel


def model_path() -> Path:
    '''Filesystem path to TIP's own YANG models'''
    return Path(__file__).parent / 'tip'


def external_path() -> Path:
    '''Filesystem path to third-party YANG models that are shipped with GNPy'''
    return Path(__file__).parent / 'ext'


def _yang_library() -> Path:
    '''Filesystem path the the ietf-yanglib JSON file'''
    return Path(__file__).parent / 'yanglib.json'


def create_datamodel() -> DataModel:
    '''Create a new yangson.DataModel'''
    return DataModel.from_file(_yang_library(), (external_path(), model_path()))
