# SPDX-License-Identifier: BSD-3-Clause
#
# Working with YANG-encoded data
#
# Copyright (C) 2020-2022 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

import oopt_gnpy_libyang as ly
import os
from pathlib import Path
from typing import NamedTuple


def model_path() -> Path:
    '''Filesystem path to TIP's own YANG models'''
    return Path(__file__).parent / 'tip'


def external_path() -> Path:
    '''Filesystem path to third-party YANG models that are shipped with GNPy'''
    return Path(__file__).parent / 'ext'


def _create_context() -> ly.Context:
    '''Prepare a libyang context for validating data against GNPy YANG models'''
    ctx = ly.Context(str(model_path()) + os.pathsep + str(external_path()),
                     ly.ContextOptions.AllImplemented | ly.ContextOptions.NoYangLibrary)
    for m in ('ietf-network', 'ietf-network-topology', 'tip-photonic-equipment'):
        ctx.load_module(m)
    return ctx


class ErrorMessage(NamedTuple):
    what: str
    where: str
    # FIXME: separate data path, schema path and line number


class Error(Exception):
    '''YANG handling error'''
    def __init__(self, orig_exception: Exception, errors: [ErrorMessage]):
        self.errors = errors
        buf = [str(orig_exception)]
        for err in errors:
            buf.append(f'{err.what} {err.where}')
        super().__init__('\n'.join(buf))


def load_data(s: str) -> ly.DataNode:
    '''Load data from YANG-based JSON input and validate them'''
    ctx = _create_context()
    try:
        data = ctx.parse_data_str(s, ly.DataFormat.JSON,
                                  ly.ParseOptions.Strict | ly.ParseOptions.Ordered,
                                  ly.ValidationOptions.Present | ly.ValidationOptions.NoState)
    except ly.Error as exc:
        raise Error(exc, [ErrorMessage(err.message, err.path) for err in ctx.errors()]) from None
    return data
