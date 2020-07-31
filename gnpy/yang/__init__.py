# SPDX-License-Identifier: BSD-3-Clause
#
# Working with YANG-encoded data
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from networkx import DiGraph
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import yangson as _y
from gnpy.core import elements
from gnpy.core.exceptions import *
import gnpy.tools.json_io as _ji
import gnpy.core.science_utils as _sci


def model_path() -> Path:
    '''Filesystem path to TIP's own YANG models'''
    return Path(__file__).parent / 'tip'


def external_path() -> Path:
    '''Filesystem path to third-party YANG models that are shipped with GNPy'''
    return Path(__file__).parent / 'ext'


def _yang_library() -> Path:
    '''Filesystem path the the ietf-yanglib JSON file'''
    return Path(__file__).parent / 'yanglib.json'


def create_datamodel() -> _y.DataModel:
    '''Create a new yangson.DataModel'''
    return _y.DataModel.from_file(_yang_library(), (external_path(), model_path()))


def _transform_fiber(fiber: _y.instance.ArrayEntry) -> _ji.Fiber:
    '''Turn yangson's ``tip-photonic-equipment:fiber`` into a Fiber equipment type representation'''
    return _ji.Fiber(
        type_variety=fiber['type'].value,
        dispersion=float(fiber['dispersion'].value),
        gamma=float(fiber['gamma'].value),
        pmd_coef=float(fiber['pmd-coefficient'].value),
    )


def _transform_raman_fiber(fiber: _y.instance.ArrayEntry) -> _ji.RamanFiber:
    '''Turn yangson's ``tip-photonic-equipment:fiber`` with a Raman section into a RamanFiber equipment type representation'''
    return _ji.RamanFiber(
        type_variety=fiber['type'].value,
        dispersion=float(fiber['dispersion'].value),
        gamma=float(fiber['gamma'].value),
        pmd_coef=float(fiber['pmd-coefficient'].value),
        raman_efficiency={ # FIXME: check the order here, the existing code is picky, and YANG doesn't guarantee any particular order here
            'cr': [x['cr'].value for x in fiber['raman-efficiency']],
            'frequency_offset': [float(x['delta-frequency'].value) for x in fiber['raman-efficiency']],
        },
    )


def _extract_per_spectrum(key: str, yang) -> List[float]:
    '''Exctract per-frequency offsets from a freq->offset YANG list and store them as a list interpolated at a 50 GHz grid'''
    if not key in yang:
        return (0, )
    data = [(int(x['frequency'].value), float(x[key].value)) for x in yang[key]]
    data.sort(key=lambda tup: tup[0])

    # FIXME: move this to gnpy.core.elements
    # FIXME: we're also probably doing the interpolation wrong (C-band grid vs. actual carrier frequencies)
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
            nf1, nf2, delta_p = _sci.estimate_nf_model(name, float(edfa['gain-min'].value), float(edfa['gain-flatmax'].value),
                                                       float(model['nf-min'].value), float(model['nf-max'].value))
            nf_model = _ji.Model_vg(nf1, nf2, delta_p)
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
        out_voa_auto=None, # FIXME
        allowed_for_design=True, # FIXME
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
        target_pch_out_db=roadm['channel-tx-power'].value,
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
    '''Turn yangson's ``tip-photonic-equipment:transceiver`` into a Transciever equipment type representation'''
    return _ji.Transceiver(
        type_variety=txp['type'].value,
        frequency={"min": txp['frequency-min'].value, "max": txp['frequency-max'].value},
        mode={mode['name']: _transform_transceiver_mode(mode) for mode in txp['mode']},
    )


def load_from_yang(json_data: Dict) -> Tuple[Dict[str, Dict[str, Any]], DiGraph]:
    '''Load equipment library, network topology and simulation options from a YANG-formatted JSON-like object'''
    dm = create_datamodel()

    data = dm.from_raw(json_data)
    data.validate()
    data = data.add_defaults()
    # No warnings are given for "missing data". In YANG, it is either an error if some required data are missing,
    # or there are default values which in turn mean that it is safe to not specify those data. There's no middle
    # ground like "please yell at me when I missed that, but continue with the simulation". I have to admit I like that.

    SIMULATION = 'tip-photonic-simulation:simulation'

    if not SIMULATION in data:
        raise ConfigurationError(f'YANG data does not contain the /{SIMULATION} element')

    sim_data = data[SIMULATION]
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
            max_fiber_lineic_loss_for_raman=0, # FIXME: can we deprecate this?
            target_extended_gain=2.5, # FIXME
            max_length=150, # FIXME
            length_units='km', # FIXME
            max_loss=None, # FIXME
            padding=0, # FIXME
            EOL=0, # FIXME
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
                power_range_db=[0, 0, 0], # FIXME
                roll_off=sim_data['grid']['tx-roll-off'].value,
                sys_margins=sim_data['system-margin'].value,
                tx_osnr=40, # FIXME
            ),
        },
        'Transceiver': {x['type'].value: _transform_transceiver(x) for x in data['tip-photonic-equipment:transceiver']},
    }

    # FIXME: adjust all Simulation's parameters

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

            if 'tip-photonic-topology:amplifier' in node:
                amp = node['tip-photonic-topology:amplifier']
                el = elements.Edfa(
                    uid=uid,
                    type_variety=amp['model'].value,
                    metadata={'location': location} if location is not None else None,
                    # FIXME
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
                    'length_units': 'km', # FIXME
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

    return (equipment, network)
