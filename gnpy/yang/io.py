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
from gnpy.core import elements, exceptions
import gnpy.tools.json_io as _ji
import gnpy.core.science_utils as _sci
import gnpy.yang
import gnpy.yang.conversion as _conv


def create_datamodel() -> _y.DataModel:
    '''Create a new yangson.DataModel'''
    return _y.DataModel.from_file(gnpy.yang._yang_library(), (gnpy.yang.external_path(), gnpy.yang.model_path()))


def _extract_common_fiber(fiber: _y.instance.ArrayEntry) -> Dict:
    return {
        'dispersion': float(fiber['dispersion'].value) * _conv.FIBER_DISPERSION,
        'dispersion_slope': float(fiber['dispersion-slope'].value) * _conv.FIBER_DISPERSION_SLOPE,
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
    OPENROADM_OSNR = 'polynomial-OSNR-OpenROADM'
    MIN_MAX_NF = 'min-max-NF'
    DUAL_STAGE = 'dual-stage'
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

    if DUAL_STAGE in edfa:
        # this model will be postprocessed in _fixup_dual_stage, so just save some placeholders here
        model = edfa[DUAL_STAGE]
        type_def = 'dual_stage'
        dual_stage_model = _ji.Model_dual_stage(model['preamp'].value, model['booster'].value)
    else:
        if POLYNOMIAL_NF in edfa:
            model = edfa[POLYNOMIAL_NF]
            nf_fit_coeff = (float(model['a'].value), float(model['b'].value), float(model['c'].value), float(model['d'].value))
            type_def = 'advanced_model'
        elif OPENROADM_OSNR in edfa:
            model = edfa[OPENROADM_OSNR]
            nf_model = _ji.Model_openroadm(nf_coef=(float(model['a'].value), float(model['b'].value),
                                                    float(model['c'].value), float(model['d'].value)))
            type_def = 'openroadm'
        elif MIN_MAX_NF in edfa:
            model = edfa[MIN_MAX_NF]
            nf_min = float(model['nf-min'].value)
            nf_max = float(model['nf-max'].value)
            nf1, nf2, delta_p = _sci.estimate_nf_model(name, float(edfa['gain-min'].value), float(edfa['gain-flatmax'].value),
                                                       nf_min, nf_max)
            nf_model = _ji.Model_vg(nf1, nf2, delta_p, nf_min, nf_max)
            type_def = 'variable_gain'
        else:
            raise NotImplementedError(f'Internal error: EDFA model {name}: unrecognized amplifier NF model for EDFA. '
                                      'Error in the YANG validation code.')

        gain_flatmax = float(edfa['gain-flatmax'].value)
        f_min = float(edfa['frequency-min'].value)
        f_max = float(edfa['frequency-max'].value)
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
        this_amp.preamp = preamp
        this_amp.booster = booster
        this_amp.f_min = max(preamp.f_min, booster.f_min)
        this_amp.f_max = min(preamp.f_max, booster.f_max)
        this_amp.gain_flatmax = preamp.gain_flatmax + booster.gain_flatmax
        this_amp.p_max = booster.p_max
        # FIXME: the old JSON code just copied each and every attr, do we need that?
        for attr in ('type_def', 'gain_flatmax', 'gain_min', 'nf_model', 'nf_fit_coeff'):
            setattr(this_amp, f'preamp_{attr}', getattr(preamp, attr))
            setattr(this_amp, f'booster_{attr}', getattr(booster, attr))
    return amps


def _transform_roadm(roadm: _y.instance.ArrayEntry) -> _ji.Roadm:
    '''Turn yangson's ``tip-photonic-equipment:roadm`` into a ROADM equipment type representation'''
    return _ji.Roadm(
        target_pch_out_db=roadm['target-channel-out-power'].value,
        add_drop_osnr=roadm['add-drop-osnr'].value,
        pmd=roadm['pmd'].value,
        restrictions={
            'preamp_variety_list': [amp.value for amp in roadm['compatible-preamp']] if 'compatible-preamp' in roadm else [],
            'booster_variety_list': [amp.value for amp in roadm['compatible-booster']] if 'compatible-booster' in roadm else [],
        },
    )


def _transform_transceiver_mode(mode: _y.instance.ArrayEntry) -> Dict[str, object]:
    return {
        'format': mode['name'].value,
        'baud_rate': mode['baud-rate'].value,
        'OSNR': mode['required-osnr'].value,
        'bit_rate': mode['bit-rate'].value,
        'roll_off': mode['tx-roll-off'].value,
        'tx_osnr': mode['tx-osnr'].value,
        'min_spacing': mode['grid-spacing'].value,
        'cost': mode['tip-photonic-simulation:cost'].value,
    }


def _transform_transceiver(txp: _y.instance.ArrayEntry) -> _ji.Transceiver:
    '''Turn yangson's ``tip-photonic-equipment:transceiver`` into a Transceiver equipment type representation'''
    return _ji.Transceiver(
        type_variety=txp['type'].value,
        frequency={"min": txp['frequency-min'].value, "max": txp['frequency-max'].value},
        mode={mode['name']: _transform_transceiver_mode(mode) for mode in txp['mode']},
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
                float(sim_data['autodesign']['power-mode']['short-span-power-reduction'].value),
                float(sim_data['autodesign']['power-mode']['long-span-power-reduction'].value),
                float(sim_data['autodesign']['power-mode']['power-sweep-stepping'].value),
            ] if ('power-mode' in sim_data['autodesign']) else [0, 0, 0],
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
                f_min=float(sim_data['grid']['frequency-min'].value),
                f_max=float(sim_data['grid']['frequency-max'].value),
                baud_rate=float(sim_data['grid']['baud-rate'].value),
                spacing=float(sim_data['grid']['spacing'].value),
                power_dbm=float(sim_data['grid']['power'].value),
                power_range_db=[0, 0, 0],  # FIXME
                roll_off=sim_data['grid']['tx-roll-off'].value,
                sys_margins=sim_data['system-margin'].value,
                tx_osnr=40,  # FIXME
            ),
        },
        'Transceiver': {x['type'].value: _transform_transceiver(x) for x in data['tip-photonic-equipment:transceiver']},
    }
    return equipment


def _load_network(data: _y.instance.RootNode, equipment: Dict[str, Dict[str, Any]]) -> DiGraph:
    '''Load the network topology from YANG data'''

    network = DiGraph()
    nodes = {}
    for net in data['ietf-network:networks']['ietf-network:network']:
        if 'network-types' not in net:
            continue
        if 'tip-photonic-topology:photonic-topology' not in net['network-types']:
            continue
        for node in net['ietf-network:node']:
            uid = node['node-id'].value
            location = None
            if 'tip-photonic-topology:geo-location' in node:
                loc = node['tip-photonic-topology:geo-location']
                if 'x' in loc and 'y' in loc:
                    location = elements.Location(
                        longitude=float(loc['tip-photonic-topology:x'].value),
                        latitude=float(loc['tip-photonic-topology:y'].value)
                    )
            metadata = {'location': location} if location is not None else None

            if 'tip-photonic-topology:amplifier' in node:
                amp = node['tip-photonic-topology:amplifier']
                type_variety = amp['model'].value
                params = equipment['Edfa'][type_variety].__dict__
                el = elements.Edfa(
                    uid=uid,
                    type_variety=type_variety,
                    params=params,
                    metadata=metadata,
                    operational={
                        'gain_target': _optional_float(amp, 'gain-target'),
                        'tilt_target': _optional_float(amp, 'tilt-target'),
                        'out_voa': _optional_float(amp, 'out-voa-target'),
                    },
                )
            elif 'tip-photonic-topology:roadm' in node:
                roadm = node['tip-photonic-topology:roadm']
                el = elements.Roadm(
                    uid=uid,
                    type_variety=roadm['model'].value,
                    metadata={'location': location} if location is not None else None,
                    # FIXME
                )
            elif 'tip-photonic-topology:transceiver' in node:
                txp = node['tip-photonic-topology:transceiver']
                el = elements.Transceiver(
                    uid=uid,
                    type_variety=txp['model'].value,
                    metadata={'location': location} if location is not None else None,
                    # FIXME
                )
            else:
                raise ValueError(f'Internal error: unrecognized network node {node} which was expected to belong to the photonic-topology')
            network.add_node(el)
            nodes[el.uid] = el
        for link in net['ietf-network-topology:link']:
            source = link['source']['source-node'].value
            target = link['destination']['dest-node'].value
            if 'tip-photonic-topology:fiber' in link:
                fiber = link['tip-photonic-topology:fiber']
                params = {
                    'length_units': 'km',  # FIXME
                    'length': float(fiber['length'].value),
                    'loss_coef': float(fiber['loss-per-km'].value),
                }
                specs = equipment['Fiber'][fiber['type'].value]
                for key in ('dispersion', 'gamma', 'pmd_coef'):
                    params[key] = getattr(specs, key)
                location = elements.Location(
                    latitude=(nodes[source].metadata['location'].latitude + nodes[target].metadata['location'].latitude) / 2,
                    longitude=(nodes[source].metadata['location'].longitude + nodes[target].metadata['location'].longitude) / 2,
                )
                el = elements.Fiber(
                    uid=link['link-id'].value,
                    type_variety=fiber['type'].value,
                    params=params,
                    metadata={'location': location},
                    # FIXME
                )
                network.add_node(el)
                nodes[el.uid] = el
                network.add_edge(nodes[source], nodes[el.uid], weight=float(fiber['length'].value))
                network.add_edge(nodes[el.uid], nodes[target], weight=0.01)
            elif 'tip-photonic-topology:patch':
                pass # FIXME
            # FIXME: handle all others

    # FIXME: read set_egress_amplifier and make it do what I want to do here
    # FIXME: be super careful with autodesign!, the assumptions in "legacy JSON" and in "YANG JSON" are very different

    return network


def load_from_yang(json_data: Dict) -> Tuple[Dict[str, Dict[str, Any]], DiGraph]:
    '''Load equipment library, (FIXME: nothing for now, will be the network topology) and simulation options from a YANG-formatted JSON-like object'''
    dm = create_datamodel()

    data = dm.from_raw(json_data)
    data.validate()
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
    network = _load_network(data, equipment)

    return (equipment, network)


def _store_edfa(name: str, edfa: _ji.Amp) -> Dict:
    '''Save in-memory representation of an EDFA amplifier type into a YANG-formatted dict'''
    res = {
        'type': name,
        'gain-min': str(edfa.gain_min),
    }

    if edfa.dual_stage_model is not None:
        res['dual-stage'] = {
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
                raise exceptions.ConfigurationError(f'Amplifier "{name}" claims to have a NF={edfa.nf_model.nf0} '
                                                    'too good to be true. Legacy Raman amps are not supported, '
                                                    'migrate to the Raman-aware simulation engine.')
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
        elif isinstance(edfa.nf_model, _ji.Model_openroadm):
            res['polynomial-OSNR-OpenROADM'] = {
                'a': str(edfa.nf_model.nf_coef[0]),
                'b': str(edfa.nf_model.nf_coef[1]),
                'c': str(edfa.nf_model.nf_coef[2]),
                'd': str(edfa.nf_model.nf_coef[3]),
            }
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


def _store_fiber(name: str, fiber: Union[_ji.Fiber, _ji.RamanFiber]) -> Dict:
    '''Save in-memory representation of a single fiber type into a YANG-formatted dict'''
    res = {
        'type': name,
        'dispersion': str(fiber.dispersion / _conv.FIBER_DISPERSION),
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


def _store_transceiver(name: str, txp: _ji.Transceiver) -> Dict:
    '''Save in-memory representation of a transceiver type into a YANG-formatted dict'''
    return {
        'type': name,
        'frequency-min': str(txp.frequency['min'] / _conv.THZ),
        'frequency-max': str(txp.frequency['max'] / _conv.THZ),
        'mode': [
            {
                'name': mode['format'],
                'bit-rate': int(mode['bit_rate'] / _conv.GIGA),
                'baud-rate': str(float(mode['baud_rate'] / _conv.GIGA)),
                'required-osnr': str(float(mode['OSNR'])),
                'tx-osnr': str(float(mode['tx_osnr'])),
                'grid-spacing': str(float(mode['min_spacing'] / _conv.GIGA)),
                'tx-roll-off': str(float(mode['roll_off'])),
                'tip-photonic-simulation:cost': mode['cost'],
            } for mode in txp.mode
        ],
    }


def _store_roadm(name: str, roadm: _ji.Roadm) -> Dict:
    '''Save in-memory representation of a ROADM type into a YANG-formatted dict'''
    return {
        'type': name,
        'add-drop-osnr': str(roadm.add_drop_osnr),
        'pmd': str(roadm.pmd),
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

    legacy_raman = [name for (name, amp) in equipment['Edfa'].items() if amp.raman]
    if legacy_raman:
        raise exceptions.ConfigurationError(
            f'Legacy Raman amplifiers are not supported, remove them from configuration: {legacy_raman}')

    span: _ji.Span = equipment['Span']['default']
    spectrum: _ji.SI = equipment['SI']['default']

    raw = {
        "tip-photonic-equipment:amplifier": [_store_edfa(k, v) for (k, v) in equipment['Edfa'].items()],
        "tip-photonic-equipment:fiber":
            [_store_fiber(k, v) for (k, v) in equipment['Fiber'].items() if k not in equipment.get('RamanFiber', {})] +
            [_store_fiber(k, v) for (k, v) in equipment.get('RamanFiber', {}).items()],
        "tip-photonic-equipment:transceiver": [_store_transceiver(k, v) for (k, v) in equipment['Transceiver'].items()],
        "tip-photonic-equipment:roadm": [_store_roadm(k, v) for (k, v) in equipment['Roadm'].items()],
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
                'gain-mode': [None], # FIXME
            },
            'system-margin': str(spectrum.sys_margins),
        },
    }
    # FIXME: spectrum.power_range_db
    # FIXME: span.*

    data = dm.from_raw(raw)
    data.validate()
    return data.raw_value()
