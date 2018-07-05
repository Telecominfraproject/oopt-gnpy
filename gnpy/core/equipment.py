#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
nf model parameters calculation
calculate nf1, nf2 and Delta_P of a 2 coils edfa with internal VOA
from nf_min and nf_max inputs 
'''
from numpy import clip, polyval
from operator import itemgetter
from math import isclose
from pathlib import Path
from json import loads
from gnpy.core.utils import lin2db, db2lin, load_json
from collections import namedtuple

Model = namedtuple('Model', 'nf1 nf2 delta_p')
Fiber = namedtuple('Fiber', 'type_variety dispersion gamma')
Spans = namedtuple('Spans', 'power_mode max_length length_units max_loss padding EOL con_loss')
Transceiver = namedtuple('Transceiver', 'type_variety frequency mode')
Roadms = namedtuple('Roadms', 'gain_mode_default_loss power_mode_pref')
SI = namedtuple('SI', 'f_min Nch baud_rate spacing roll_off power')
EdfaBase = namedtuple(
    'EdfaBase',
    'type_variety gain_flatmax gain_min p_max nf_min nf_max'
    ' nf_model nf_fit_coeff nf_ripple dgt gain_ripple')
class Edfa(EdfaBase):
    def __new__(cls,
            type_variety, gain_flatmax, gain_min, p_max, nf_min=None, nf_max=None,
            nf_model=None, nf_fit_coeff=None, nf_ripple=None, dgt=None, gain_ripple=None):
        return super().__new__(cls,
            type_variety, gain_flatmax, gain_min, p_max, nf_min, nf_max,
            nf_model, nf_fit_coeff, nf_ripple, dgt, gain_ripple)

    @classmethod
    def from_advanced_json(cls, filename, **kwargs):
        with open(filename) as f:
            json_data = loads(f.read())
        return cls(**{**kwargs, **json_data, 'nf_model': None})

    @classmethod
    def from_default_json(cls, filename, **kwargs):
        with open(filename) as f:
            json_data = loads(f.read())
        type_variety = kwargs['type_variety']
        gain_min, gain_max = kwargs['gain_min'], kwargs['gain_flatmax']
        nf_min, nf_max = kwargs['nf_min'], kwargs['nf_max']
        nf1, nf2, delta_p = nf_model(type_variety, gain_min, gain_max, nf_min, nf_max)
        return cls(**{**kwargs, **json_data, 'nf_model': Model(nf1, nf2, delta_p)})


def nf_model(type_variety, gain_min, gain_max, nf_min, nf_max):
    if nf_min < -10:
        raise ValueError(f'Invalid nf_min value {nf_min!r}')
    if nf_max < -10:
        raise ValueError(f'Invalid nf_max value {nf_max!r}')

    # NF estimation model based on nf_min and nf_max
    # delta_p:  max power dB difference between first and second stage coils
    # dB g1a:   first stage gain - internal VOA attenuation
    # nf1, nf2: first and second stage coils
    #           calculated by solving nf_{min,max} = nf1 + nf2 / g1a{min,max}
    delta_p = 5
    g1a_min = gain_min - (gain_max - gain_min) - delta_p
    g1a_max = gain_max - delta_p
    nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) /
                 (1/db2lin(g1a_max) - 1/db2lin(g1a_min)))
    nf1 = lin2db(db2lin(nf_min) - db2lin(nf2)/db2lin(g1a_max))

    if nf1 < 4:
        raise ValueError(f'First coil value too low {nf1}')

    # Check 1 dB < delta_p < 6 dB to ensure nf_min and nf_max values make sense.
    # There shouldn't be high nf differences between the two coils:
    #    nf2 should be nf1 + 0.3 < nf2 < nf1 + 2
    # If not, recompute and check delta_p
    if not nf1 + 0.3 < nf2 < nf1 + 2:
        nf2 = clip(nf2, nf1 + 0.3, nf1 + 2)
        g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1)))
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max-gain_min) - delta_p
        if not 1 < delta_p < 6:
            raise ValueError(f'Computed \N{greek capital letter delta}P invalid \
                \n 1st coil vs 2nd coil calculated DeltaP {delta_p:.2f} for \
                \n amp {type_variety} is not valid: revise inputs \
                \n calculated 1st coil NF = {nf1:.2f}, 2nd coil NF = {nf2:.2f}')
    # Check calculated values for nf1 and nf2
    calc_nf_min = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_max))
    if not isclose(nf_min, calc_nf_min, abs_tol=0.01):
        raise ValueError(f'nf_min does not match calc_nf_min, {nf_min} vs {calc_nf_min} for amp {type_variety}')
    calc_nf_max = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_min))
    if not isclose(nf_max, calc_nf_max, abs_tol=0.01):
        raise ValueError(f'nf_max does not match calc_nf_max, {nf_max} vs {calc_nf_max} for amp {type_variety}')

    return nf1, nf2, delta_p

def edfa_nf(gain, variety_type, equipment):
    edfa = equipment['Edfa'][variety_type]
    'input VOA padding at low gain = worst case strategy'
    'not necessary when output VOA/att padding strategy will be implemented'
    pad = max(edfa.gain_min - gain, 0)
    gain = gain + pad
    dg = max(edfa.gain_flatmax - gain, 0)
    if edfa.nf_model:
        g1a = gain - edfa.nf_model.delta_p - dg
        nf_avg = lin2db(db2lin(edfa.nf_model.nf1) + db2lin(edfa.nf_model.nf2)/db2lin(g1a))
    else:
        nf_avg = polyval(edfa.nf_fit_coeff, dg)
    return nf_avg + pad # input VOA = 1 for 1 NF degradation


def load_equipment(filename):
    json_data = load_json(filename)
    return equipment_from_json(json_data, filename)

def equipment_from_json(json_data, filename):
    """build global dictionnary eqpt_library that stores all eqpt characteristics:
    edfa type type_variety, fiber type_variety
    from the eqpt_config.json (filename parameter)
    also read advanced_config_from_json file parameters for edfa if they are available:
    typically nf_ripple, dfg gain ripple, dgt and nf polynomial nf_fit_coeff
    if advanced_config_from_json file parameter is not present: use nf_model:
    requires nf_min and nf_max values boundaries of the edfa gain range
    """
    equipment = {}
    for key, entries in json_data.items():
        for entry in entries:
            if key not in equipment:
                equipment[key] = {}
            subkey = entry.get('type_variety', 'default')
            typ = globals()[key]
            if key == 'Edfa':
                if 'advanced_config_from_json' in entry:
                    config = Path(filename).parent / entry.pop('advanced_config_from_json')
                    typ = lambda **kws: Edfa.from_advanced_json(config, **kws)
                else:
                    config = Path(filename).parent / 'default_edfa_config.json'
                    typ = lambda **kws: Edfa.from_default_json(config, **kws)
            equipment[key][subkey] = typ(**entry)
    return equipment
