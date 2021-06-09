# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

"""
Reading and writing YANG data
=============================

Module :py:mod:`gnpy.yang.io` enables loading of data that are formatted according to the YANG+JSON rules.
Use :func:`load_from_yang` to parse and validate the data, and :func:`save_equipment` to store the equipment library.
"""

from networkx import DiGraph
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import yangson as _y
from gnpy.core import exceptions
import gnpy.tools.json_io as _ji
import gnpy.core.science_utils as _sci
import gnpy.yang
import gnpy.yang.conversion as _conv


def create_datamodel() -> _y.DataModel:
    '''Create a new yangson.DataModel'''
    return _y.DataModel.from_file(gnpy.yang._yang_library(), (gnpy.yang.external_path(), gnpy.yang.model_path()))


def _extract_common_fiber(fiber: _y.instance.ArrayEntry) -> Dict:
    return {
        'dispersion': float(fiber['chromatic-dispersion'].value) * _conv.FIBER_DISPERSION,
        'dispersion_slope': float(fiber['chromatic-dispersion-slope'].value) * _conv.FIBER_DISPERSION_SLOPE,
        'gamma': float(fiber['gamma'].value) * _conv.FIBER_GAMMA,
        'pmd_coef': float(fiber['pmd-coefficient'].value) * _conv.FIBER_PMD_COEF,
    }


def _transform_fiber(fiber: _y.instance.ArrayEntry) -> _ji.Fiber:
    '''Turn yangson's ``tip-photonic-equipment:fiber`` into a Fiber equipment type representation'''
    return _ji.Fiber(
        type_variety=fiber['type'].value,
        **_extract_common_fiber(fiber),
    )


def _transform_raman_fiber(fiber: _y.instance.ArrayEntry) -> _ji.RamanFiber:
    '''Turn yangson's ``tip-photonic-equipment:fiber`` with a Raman section into a RamanFiber equipment type representation'''
    return _ji.RamanFiber(
        type_variety=fiber['type'].value,
        raman_efficiency={  # FIXME: check the order here, the existing code is picky, and YANG doesn't guarantee any particular order here
            'cr': [x['cr'].value for x in fiber['raman-efficiency']],
            'frequency_offset': [float(x['delta-frequency'].value) for x in fiber['raman-efficiency']],
        },
        **_extract_common_fiber(fiber),
    )


def _extract_per_spectrum(key: str, yang) -> List[float]:
    '''Extract per-frequency offsets from a freq->offset YANG list and store them as a list interpolated at a 50 GHz grid'''
    if key not in yang:
        return (0, )
    data = [(int(x['frequency'].value), float(x[key].value)) for x in yang[key]]
    data.sort(key=lambda tup: tup[0])

    # FIXME: move this to gnpy.core.elements
    # FIXME: we're also probably doing the interpolation wrong in elements.py (C-band grid vs. actual carrier frequencies)
    keys = [x[0] for x in data]
    values = [x[1] for x in data]
    frequencies = [int(191.3e12 + channel * 50e9) for channel in range(96)]
    data = [x for x in np.interp(frequencies, keys, values)]  # force back Python's native list to silence a FutureWarning: elementwise comparison failed
    return data


def _transform_edfa(edfa: _y.instance.ArrayEntry) -> _ji.Amp:
    '''Turn yangson's ``tip-photonic-equipment:amplifier`` into an EDFA equipment type representation'''

    POLYNOMIAL_NF = 'polynomial-NF'
    OPENROADM_ILA = 'OpenROADM-ILA'
    OPENROADM_PREAMP = 'OpenROADM-preamp'
    OPENROADM_BOOSTER = 'OpenROADM-booster'
    MIN_MAX_NF = 'min-max-NF'
    COMPOSITE = 'composite'
    RAMAN_APPROX = 'raman-approximation'
    GAIN_RIPPLE = 'gain-ripple'
    NF_RIPPLE = 'nf-ripple'
    DYNAMIC_GAIN_TILT = 'dynamic-gain-tilt'

    name = edfa['type'].value
    type_def = None
    nf_model = None
    dual_stage_model = None
    f_min = None
    f_max = None
    gain_flatmax = None
    p_max = None
    nf_fit_coeff = None
    nf_ripple = [0]
    dgt = [0]
    gain_ripple = [0]

    if COMPOSITE in edfa:
        # this model will be postprocessed in _fixup_dual_stage, so just save some placeholders here
        model = edfa[COMPOSITE]
        type_def = 'dual_stage'
        dual_stage_model = _ji.Model_dual_stage(model['preamp'].value, model['booster'].value)
    else:
        if POLYNOMIAL_NF in edfa:
            model = edfa[POLYNOMIAL_NF]
            nf_fit_coeff = (float(model['a'].value), float(model['b'].value), float(model['c'].value), float(model['d'].value))
            type_def = 'advanced_model'
        elif OPENROADM_ILA in edfa:
            model = edfa[OPENROADM_ILA]
            nf_model = _ji.Model_openroadm_ila(nf_coef=(float(model['a'].value), float(model['b'].value),
                                                        float(model['c'].value), float(model['d'].value)))
            type_def = 'openroadm'
        elif OPENROADM_PREAMP in edfa:
            type_def = 'openroadm_preamp'
        elif OPENROADM_BOOSTER in edfa:
            type_def = 'openroadm_booster'
        elif MIN_MAX_NF in edfa:
            model = edfa[MIN_MAX_NF]
            nf_min = float(model['nf-min'].value)
            nf_max = float(model['nf-max'].value)
            nf1, nf2, delta_p = _sci.estimate_nf_model(name, float(edfa['gain-min'].value), float(edfa['gain-flatmax'].value),
                                                       nf_min, nf_max)
            nf_model = _ji.Model_vg(nf1, nf2, delta_p, nf_min, nf_max)
            type_def = 'variable_gain'
        elif RAMAN_APPROX in edfa:
            model = edfa[RAMAN_APPROX]
            nf_fit_coeff = (0., 0., 0., float(model['nf'].value))
            type_def = 'advanced_model'
        else:
            raise NotImplementedError(f'Internal error: EDFA model {name}: unrecognized amplifier NF model for EDFA. '
                                      'Error in the YANG validation code.')

        gain_flatmax = float(edfa['gain-flatmax'].value)
        f_min = float(edfa['frequency-min'].value) * _conv.THZ
        f_max = float(edfa['frequency-max'].value) * _conv.THZ
        p_max = float(edfa['max-power-out'].value)

        gain_ripple = _extract_per_spectrum(GAIN_RIPPLE, edfa)
        dgt = _extract_per_spectrum(DYNAMIC_GAIN_TILT, edfa)
        nf_ripple = _extract_per_spectrum(NF_RIPPLE, edfa)

    return _ji.Amp(
        type_variety=name,
        type_def=type_def,
        f_min=f_min,
        f_max=f_max,
        gain_min=float(edfa['gain-min'].value),
        gain_flatmax=gain_flatmax,
        p_max=p_max,
        nf_fit_coeff=nf_fit_coeff,
        nf_ripple=nf_ripple,
        dgt=dgt,
        gain_ripple=gain_ripple,
        out_voa_auto=None,  # FIXME
        allowed_for_design=True,  # FIXME
        raman=False,
        nf_model=nf_model,
        dual_stage_model=dual_stage_model,
    )


def _fixup_dual_stage(amps: Dict[str, _ji.Amp]) -> Dict[str, _ji.Amp]:
    '''Replace preamp/booster string model IDs with references to actual objects'''
    for name, amp in amps.items():
        if amp.dual_stage_model is None:
            continue
        preamp = amps[amp.dual_stage_model.preamp_variety]
        booster = amps[amp.dual_stage_model.booster_variety]
        this_amp = amps[name]
        # FIXME: the old JSON code copies each and every attr, do we need that here?
        for attr in preamp.__dict__.keys():
            setattr(this_amp, f'preamp_{attr}', getattr(preamp, attr))
        for attr in booster.__dict__.keys():
            setattr(this_amp, f'booster_{attr}', getattr(booster, attr))
    return amps


def _transform_roadm(roadm: _y.instance.ArrayEntry) -> _ji.Roadm:
    '''Turn yangson's ``tip-photonic-equipment:roadm`` into a ROADM equipment type representation'''
    return _ji.Roadm(
        target_pch_out_db=float(roadm['target-channel-out-power'].value),
        add_drop_osnr=float(roadm['add-drop-osnr'].value),
        pmd=float(roadm['polarization-mode-dispersion'].value),
        restrictions={
            'preamp_variety_list': [amp.value for amp in roadm['compatible-preamp']] if 'compatible-preamp' in roadm else [],
            'booster_variety_list': [amp.value for amp in roadm['compatible-booster']] if 'compatible-booster' in roadm else [],
        },
    )


def _transform_transceiver_mode(mode: _y.instance.ArrayEntry) -> Dict[str, object]:
    return {
        'format': mode['name'].value,
        'baud_rate': float(mode['baud-rate'].value) * _conv.GIGA,
        'OSNR': float(mode['required-osnr'].value),
        'bit_rate': float(mode['bit-rate'].value) * _conv.GIGA,
        'roll_off': float(mode['tx-roll-off'].value),
        'tx_osnr': float(mode['in-band-tx-osnr'].value),
        'min_spacing': float(mode['grid-spacing'].value) * _conv.GIGA,
        'cost': float(mode['tip-photonic-simulation:cost'].value),
    }


def _transform_transceiver(txp: _y.instance.ArrayEntry) -> _ji.Transceiver:
    '''Turn yangson's ``tip-photonic-equipment:transceiver`` into a Transceiver equipment type representation'''
    return _ji.Transceiver(
        type_variety=txp['type'].value,
        frequency={
            "min": float(txp['frequency-min'].value) * _conv.THZ,
            "max": float(txp['frequency-max'].value) * _conv.THZ,
        },
        mode=[_transform_transceiver_mode(mode) for mode in txp['mode']],
    )


def _optional_float(yangish, key, default=None):
    '''Retrieve a decimal64 value as a float, or None if not present'''
    return float(yangish[key].value) if key in yangish else default


def _load_equipment(data: _y.instance.RootNode, sim_data: _y.instance.InstanceNode) -> Dict[str, Dict[str, Any]]:
    '''Load the equipment library from YANG data'''
    equipment = {
        'Edfa': _fixup_dual_stage({x['type'].value: _transform_edfa(x) for x in data['tip-photonic-equipment:amplifier']}),
        'Fiber': {x['type'].value: _transform_fiber(x) for x in data['tip-photonic-equipment:fiber']},
        'RamanFiber': {x['type'].value: _transform_raman_fiber(x) for x in data['tip-photonic-equipment:fiber'] if 'raman-efficiency' in x},
        'Span': {'default': _ji.Span(
            power_mode='power-mode' in sim_data['autodesign'],
            delta_power_range_db=[
                float(sim_data['autodesign']['power-adjustment-for-span-loss']['maximal-reduction'].value),
                float(sim_data['autodesign']['power-adjustment-for-span-loss']['maximal-boost'].value),
                float(sim_data['autodesign']['power-adjustment-for-span-loss']['excursion-step-size'].value),
            ],
            max_fiber_lineic_loss_for_raman=0,  # FIXME: can we deprecate this?
            target_extended_gain=2.5,  # FIXME
            max_length=150,  # FIXME
            length_units='km',  # FIXME
            max_loss=None,  # FIXME
            padding=0,  # FIXME
            EOL=0,  # FIXME
            con_in=0,
            con_out=0,
        )
        },
        'Roadm': {x['type'].value: _transform_roadm(x) for x in data['tip-photonic-equipment:roadm']},
        'SI': {
            'default': _ji.SI(
                f_min=float(sim_data['grid']['frequency-min'].value) * _conv.THZ,
                f_max=float(sim_data['grid']['frequency-max'].value) * _conv.THZ,
                baud_rate=float(sim_data['grid']['baud-rate'].value) * _conv.GIGA,
                spacing=float(sim_data['grid']['spacing'].value) * _conv.GIGA,
                power_dbm=float(sim_data['grid']['power'].value),
                power_range_db=(
                    [ # start, stop, step
                        float(sim_data['autodesign']['power-mode']['power-sweep']['start'].value),
                        float(sim_data['autodesign']['power-mode']['power-sweep']['stop'].value),
                        float(sim_data['autodesign']['power-mode']['power-sweep']['step-size'].value),
                    ] if 'power-sweep' in sim_data['autodesign']['power-mode'] else [0, 0, 0]
                ) if ('power-mode' in sim_data['autodesign']) else None,
                roll_off=float(sim_data['grid']['tx-roll-off'].value),
                sys_margins=float(sim_data['system-margin'].value),
                tx_osnr=float(sim_data['grid']['tx-osnr'].value),
            ),
        },
        'Transceiver': {x['type'].value: _transform_transceiver(x) for x in data['tip-photonic-equipment:transceiver']},
    }
    return equipment


def load_from_yang(json_data: Dict) -> Tuple[Dict[str, Dict[str, Any]], DiGraph]:
    '''Load equipment library, (FIXME: nothing for now, will be the network topology) and simulation options from a YANG-formatted JSON-like object'''
    dm = create_datamodel()

    data = dm.from_raw(json_data)
    data.validate(ctype=_y.enumerations.ContentType.config)
    data = data.add_defaults()
    # No warnings are given for "missing data". In YANG, it is either an error if some required data are missing,
    # or there are default values which in turn mean that it is safe to not specify those data. There's no middle
    # ground like "please yell at me when I missed that, but continue with the simulation". I have to admit I like that.

    SIMULATION = 'tip-photonic-simulation:simulation'
    if SIMULATION not in data:
        raise exceptions.ConfigurationError(f'YANG data does not contain the /{SIMULATION} element')

    sim_data = data[SIMULATION]
    equipment = _load_equipment(data, sim_data)
    # FIXME: adjust all Simulation's parameters

    network = None

    return (equipment, network)


def _store_equipment_edfa(name: str, edfa: _ji.Amp) -> Dict:
    '''Save in-memory representation of an EDFA amplifier type into a YANG-formatted dict'''
    res = {
        'type': name,
        'gain-min': str(edfa.gain_min),
    }

    if edfa.dual_stage_model is not None:
        res['composite'] = {
            'preamp': edfa.dual_stage_model.preamp_variety,
            'booster': edfa.dual_stage_model.booster_variety,
        }
    else:
        res['frequency-min'] = str(edfa.f_min / _conv.THZ)
        res['frequency-max'] = str(edfa.f_max / _conv.THZ)
        res['gain-flatmax'] = str(edfa.gain_flatmax)
        res['max-power-out'] = str(edfa.p_max)
        res['has-output-voa'] = edfa.out_voa_auto

        if isinstance(edfa.nf_model, _ji.Model_fg):
            if edfa.nf_model.nf0 < 3:
                res['raman-approximation'] = {
                    'nf': str(edfa.nf_model.nf0)
                }
            else:
                res['polynomial-NF'] = {
                    'a': '0',
                    'b': '0',
                    'c': '0',
                    'd': str(edfa.nf_model.nf0),
                }
        elif isinstance(edfa.nf_model, _ji.Model_vg):
            res['min-max-NF'] = {
                'nf-min': str(edfa.nf_model.orig_nf_min),
                'nf-max': str(edfa.nf_model.orig_nf_max),
            }
        elif isinstance(edfa.nf_model, _ji.Model_openroadm_ila):
            res['OpenROADM-ILA'] = {
                'a': str(edfa.nf_model.nf_coef[0]),
                'b': str(edfa.nf_model.nf_coef[1]),
                'c': str(edfa.nf_model.nf_coef[2]),
                'd': str(edfa.nf_model.nf_coef[3]),
            }
        elif isinstance(edfa.nf_model, _ji.Model_openroadm_preamp):
            res['OpenROADM-preamp'] = {}
        elif isinstance(edfa.nf_model, _ji.Model_openroadm_booster):
            res['OpenROADM-booster'] = {}
        elif edfa.type_def == 'advanced_model':
            res['polynomial-NF'] = {
                'a': str(edfa.nf_fit_coeff[0]),
                'b': str(edfa.nf_fit_coeff[1]),
                'c': str(edfa.nf_fit_coeff[2]),
                'd': str(edfa.nf_fit_coeff[3]),
            }

        # FIXME: implement these
        # 'nf_ripple': None,
        # 'dgt': None,
        # 'gain_ripple': None,
    return res


def _store_equipment_fiber(name: str, fiber: Union[_ji.Fiber, _ji.RamanFiber]) -> Dict:
    '''Save in-memory representation of a single fiber type into a YANG-formatted dict'''
    res = {
        'type': name,
        'chromatic-dispersion': str(fiber.dispersion / _conv.FIBER_DISPERSION),
        'gamma': str(fiber.gamma / _conv.FIBER_GAMMA),
        'pmd-coefficient': str(fiber.pmd_coef / _conv.FIBER_PMD_COEF),
    }

    # FIXME: do we support setting 'dispersion-slope' via JSON setting in the first place? There are no examples...
    try:
        res['dispersion-slope'] = str(fiber.dispersion_slope / _conv.FIBER_DISPERSION_SLOPE)
    except AttributeError:
        pass

    if isinstance(fiber, _ji.RamanFiber):
        res['raman-efficiency'] = [
            {
                'delta-frequency': str(freq / _conv.THZ),
                'cr': str(float(cr)),
            } for (cr, freq) in zip(fiber.raman_efficiency['cr'], fiber.raman_efficiency['frequency_offset'])
        ]

    return res


def _store_equipment_txp_mode(mode: Dict) -> Dict:
    res = {
        'name': mode['format'],
        'bit-rate': int(mode['bit_rate'] / _conv.GIGA),
        'baud-rate': str(float(mode['baud_rate'] / _conv.GIGA)),
        'required-osnr': str(float(mode['OSNR'])),
        'in-band-tx-osnr': str(float(mode['tx_osnr'])),
        'grid-spacing': str(float(mode['min_spacing'] / _conv.GIGA)),
        'tip-photonic-simulation:cost': mode['cost'],
    }
    if mode['roll_off'] is not None:
        res['tx-roll-off'] = str(float(mode['roll_off']))
    return res


def _store_equipment_transceiver(name: str, txp: _ji.Transceiver) -> Dict:
    '''Save in-memory representation of a transceiver type into a YANG-formatted dict'''
    return {
        'type': name,
        'frequency-min': str(txp.frequency['min'] / _conv.THZ),
        'frequency-max': str(txp.frequency['max'] / _conv.THZ),
        'mode': [_store_equipment_txp_mode(mode) for mode in txp.mode],
    }


def _store_equipment_roadm(name: str, roadm: _ji.Roadm) -> Dict:
    '''Save in-memory representation of a ROADM type into a YANG-formatted dict'''
    return {
        'type': name,
        'add-drop-osnr': str(roadm.add_drop_osnr),
        'polarization-mode-dispersion': str(roadm.pmd),
        'target-channel-out-power': str(roadm.target_pch_out_db),
        'compatible-preamp': [amp for amp in roadm.restrictions.get('preamp_variety_list', [])],
        'compatible-booster': [amp for amp in roadm.restrictions.get('booster_variety_list', [])],
    }


def save_equipment(equipment: Dict[str, Dict[str, Any]]) -> Dict:
    '''Save the in-memory equipment library into a dict with YANG-formatted data'''
    dm = create_datamodel()

    for k in ('Edfa', 'Fiber', 'Span', 'SI', 'Transceiver', 'Roadm'):
        if k not in equipment:
            raise exceptions.ConfigurationError(f'No "{k}" in the equipment library')
    for k in ('Span', 'SI'):
        if 'default' not in equipment[k]:
            raise exceptions.ConfigurationError('No ["{k}"]["default"] in the equipment library')

    # FIXME: what do we do with these amps? Is this detection a good thing, btw?
    # legacy_raman = [name for (name, amp) in equipment['Edfa'].items() if amp.raman]
    # if legacy_raman:
    #     raise exceptions.ConfigurationError(
    #         f'Legacy Raman amplifiers are not supported, remove them from configuration: {legacy_raman}')

    span: _ji.Span = equipment['Span']['default']
    spectrum: _ji.SI = equipment['SI']['default']

    raw = {
        "tip-photonic-equipment:amplifier": [_store_equipment_edfa(k, v) for (k, v) in equipment['Edfa'].items()],
        "tip-photonic-equipment:fiber":
            [_store_equipment_fiber(k, v) for (k, v) in equipment['Fiber'].items() if k not in equipment.get('RamanFiber', {})] +
            [_store_equipment_fiber(k, v) for (k, v) in equipment.get('RamanFiber', {}).items()],
        "tip-photonic-equipment:transceiver": [_store_equipment_transceiver(k, v) for (k, v) in equipment['Transceiver'].items()],
        "tip-photonic-equipment:roadm": [_store_equipment_roadm(k, v) for (k, v) in equipment['Roadm'].items()],
        "tip-photonic-simulation:simulation": {
            'grid': {
                'frequency-min': str(spectrum.f_min / _conv.THZ),
                'frequency-max': str(spectrum.f_max / _conv.THZ),
                'spacing': str(spectrum.spacing / _conv.GIGA),
                'power': str(spectrum.power_dbm),
                'tx-roll-off': str(spectrum.roll_off),
                'tx-osnr': str(spectrum.tx_osnr),
                'baud-rate': str(spectrum.baud_rate / _conv.GIGA),
            },
            'autodesign': {
                'allowed-inline-edfa': [k for (k, v) in equipment['Edfa'].items() if v.allowed_for_design],
                'power-adjustment-for-span-loss': {
                    'maximal-reduction': str(span.delta_power_range_db[0]),
                    'maximal-boost': str(span.delta_power_range_db[1]),
                    'excursion-step-size': str(span.delta_power_range_db[2]),
                },
            },
            'system-margin': str(spectrum.sys_margins),
        },
    }
    if span.power_mode:
        raw['tip-photonic-simulation:simulation']['autodesign']['power-mode'] = {
            'power-sweep': {
                'start': str(spectrum.power_range_db[0]),
                'stop': str(spectrum.power_range_db[1]),
                'step-size': str(spectrum.power_range_db[2]),
            },
        }
    else:
        raw['tip-photonic-simulation:simulation']['autodesign']['gain-mode'] = [None]

    data = dm.from_raw(raw)
    data.validate()
    return data.raw_value()
