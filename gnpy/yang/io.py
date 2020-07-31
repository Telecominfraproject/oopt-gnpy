# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020-2023 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#
"""
Reading and writing YANG data
=============================
Module :py:mod:`gnpy.yang.io` enables loading of data that are formatted according to the YANG+JSON rules.
Use :func:`_load_equipment` to parse and validate the data.
"""

from networkx import DiGraph
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from gnpy.core import exceptions
import gnpy.tools.json_io as _ji
import gnpy.core.science_utils as _sci
import gnpy.yang.conversion as _conv
import oopt_gnpy_libyang as ly


def _load_equipment(data: ly.DataNode) -> Dict[str, Dict[str, Any]]:
    """Load the equipment library from parsed and validated YANG data"""
    root = ly.search_at_root(data)
    equipment = {
        'Fiber': {
            _string(x['type']):
            _transform_fiber(x) for x in root('tip-photonic-equipment:fiber')
        },
        'RamanFiber': {
            _string(x['type']):
            _transform_fiber(x) for x in root('tip-photonic-equipment:fiber')
            if 'raman-coefficient' in x
        },
        # 'Span' and 'SI' are actually simulation options, not something from tip-photonic-equipment
        'Edfa': {}, # FIXME
        'Roadm': {
            _string(x['type']):
            _transform_roadm(x) for x in root('tip-photonic-equipment:roadm')
        },
        'Transceiver': {
            _string(x['type']):
            _transform_transceiver(x) for x in root('tip-photonic-equipment:transceiver')
        }

    }
    return equipment


def _number_auto_default(node: ly.DataNode, scaling=None) -> Optional[float]:
    """Process a decimal64 number to float, with a scaling factor, or a magic `automatic-default` value to `None`"""
    term = node.as_term()
    if isinstance(term.value, ly.Enum) and term.value.name == "automatic-default":
        return None
    else:
        return _number(node, scaling)


def _number(node: ly.DataNode, scaling=None) -> Optional[float]:
    """Process a decimal64 number to float, with a scaling factor"""
    res = float(_string(node))
    return res * scaling if scaling is not None else res


def _string(node: ly.DataNode) -> str:
    """Extract a string from a libyang DataNode"""
    return str(node.as_term())


def _transform_fiber(fiber: ly.DataNode) -> _ji.Fiber:
    """Transform a YANG representation of Fiber equipment model into an intermediate data structure"""
    assert fiber.schema.path == '/tip-photonic-equipment:fiber'
    if len(fiber.find('per-frequency')) > 1:
        # FIXME: implement https://review.gerrithub.io/c/Telecominfraproject/oopt-gnpy/+/554949
        raise ConfigurationError('FIXME: per-frequency fiber parameters are not supported yet')
    return _ji.Fiber(
        type_variety=_string(fiber['type']),
        effective_area=_number_auto_default(fiber['effective-area'], _conv.FIBER_EFFECTIVE_AREA),
        gamma=_number_auto_default(fiber['gamma'], _conv.FIBER_GAMMA),
        pmd_coef=_number(fiber['pmd-coefficient'], _conv.FIBER_PMD_COEF),
        dispersion=_number(fiber['per-frequency']['chromatic-dispersion'], _conv.FIBER_DISPERSION),
        dispersion_slope=_number(fiber['per-frequency']['chromatic-dispersion-slope'], _conv.FIBER_DISPERSION_SLOPE),
        raman_efficiency={'cr': [
            _number(x['g_0']) for x in fiber['raman-coefficient'].find('gamma-efficiency')
        ], 'frequency_offset': [
            _number(x['delta-frequency']) for x in fiber['raman-coefficient'].find('gamma-efficiency')
        ]} if 'raman-coefficient' in fiber else None,
    )


def _eq_strategy(roadm: ly.DataNode) -> Dict:
    """Helper for reading the ROADM model's equalization strategy"""
    if 'per-channel-output-power' in roadm:
        return {'target_pch_out_db': _number(roadm['per-channel-output-power'])}
    elif 'power-spectral-density' in roadm:
        return {'target_psd_out_mWperGHz': _number(roadm['power-spectral-density'])}
    elif 'power-by-dwdm-slot-width' in roadm:
        return {'target_out_mWperSlotWidth': _number(roadm['power-by-dwdm-slot-width'])}
    else:
        assert False  # enforced by the YANG model


def _transform_roadm(roadm: ly.DataNode) -> _ji.Roadm:
    """Transform a YANG representation of a ROADM equipment model into an intermediate data structure"""
    assert roadm.schema.path == '/tip-photonic-equipment:roadm'
    return _ji.Roadm(
        type_variety=_string(roadm['type']),
        pmd=_number(roadm['polarization-mode-dispersion']),
        pdl=_number(roadm['polarization-dependent-loss']),
        add_drop_osnr=_number(roadm['add-drop-osnr']),
        **_eq_strategy(roadm),
        restrictions={
            'preamp_variety_list': [_string(x) for x in roadm.find('compatible-preamp')],
            'booster_variety_list': [_string(x) for x in roadm.find('compatible-booster')],
        }
    )


def _transform_transceiver_mode(mode: ly.DataNode) -> Dict[str, object]:
    """Helper for _transform_transceiver which takes care of transponder's operating modes"""
    assert mode.schema.path == '/tip-photonic-equipment:transceiver/mode'
    return {
        'format': _string(mode['name']),
        'bit_rate': _number(mode['bit-rate'], _conv.GIGA),
        'baud_rate': _number(mode['baud-rate'], _conv.GIGA),
        'OSNR': _number(mode['required-osnr']), # FIXME
        'tx_osnr': _number(mode['in-band-tx-osnr']),
        'min_spacing': _number(mode['grid-spacing'], _conv.GIGA),
        'roll_off': _number(mode['tx-roll-off']),
        # FIXME: max-cd
        # FIXME: max-pmd
        # FIXME: max-pdl
        # FIXME: penalty matrix
        # FIXME: enable through tip-simulation-options: 'cost': _number(mode['tip-photonic-simulation:cost'].as_term()),
    }


def _transform_transceiver(txp: ly.DataNode) -> _ji.Transceiver:
    """Transform a YANG representation of a transponder equipment model into an intermediate data structure"""
    assert txp.schema.path == '/tip-photonic-equipment:transceiver'
    return _ji.Transceiver(
        type_variety=_string(txp['type']),
        frequency={
            "min": _number(txp['laser-frequency-min'], _conv.TERA),
            "max": _number(txp['laser-frequency-max'], _conv.TERA),
        },
        mode=[_transform_transceiver_mode(mode) for mode in txp.find('mode')],
    )

