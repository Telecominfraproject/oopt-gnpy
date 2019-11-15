#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
==================

This module contains all parameters to configure standard network elements.

"""

from logging import getLogger

from gnpy.core.exceptions import ParametersError


logger = getLogger(__name__)

class Unique:
    _shared_dict = {}
    @classmethod
    def init(cls):
        return cls

class Parameters:
    def asdict(self):
        class_dict = self.__class__.__dict__
        instance_dict = self.__dict__
        new_dict = {}
        for key in class_dict:
            if isinstance(class_dict[key],property):
                new_dict[key] = instance_dict['_' + key]
        return new_dict

class PumpParams(Parameters):
    def __init__(self, power, frequency, propagation_direction):
        self._power = power
        self._frequency = frequency
        self._propagation_direction = propagation_direction

    @property
    def power(self):
        return self._power

    @property
    def frequency(self):
        return self._frequency

    @property
    def propagation_direction(self):
        return self._propagation_direction


class RamanParams(Parameters):
    def __init__(self, params):
        self._flag_raman = params['flag_raman']
        self._space_resolution = params['space_resolution']
        self._tolerance = params['tolerance']
        self._raman_computed_channels = params["raman_computed_channels"]

    @property
    def flag_raman(self):
        return self._flag_raman

    @property
    def space_resolution(self):
        return self._space_resolution

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def raman_computed_channels(self):
        return self.raman_computed_channels


class NLIParams(Parameters):
    def __init__(self, params):
        self._nli_method_name = params['nli_method_name']
        self._wdm_grid_size = params['wdm_grid_size']
        self._dispersion_tolerance = params['dispersion_tolerance']
        self._phase_shift_tollerance = params['phase_shift_tollerance']
        self._f_cut_resolution = None
        self._f_pump_resolution = None

    @property
    def nli_method_name(self):
        return self._nli_method_name

    @property
    def wdm_grid_size(self):
        return self._wdm_grid_size

    @property
    def dispersion_tolerance(self):
        return self._dispersion_tolerance

    @property
    def phase_shift_tollerance(self):
        return self._phase_shift_tollerance

    @property
    def f_cut_resolution(self):
        return self._f_cut_resolution

    @f_cut_resolution.setter
    def f_cut_resolution(self, f_cut_resolution):
        self._f_cut_resolution = f_cut_resolution

    @property
    def f_pump_resolution(self):
        return self._f_pump_resolution

    @f_pump_resolution.setter
    def f_pump_resolution(self, f_pump_resolution):
        self._f_pump_resolution = f_pump_resolution


class SimParams(Parameters):
    """A private mutable variable is shared across all class instances. Cannot implement Parameters.asdict because
    attribute is _shared_dict"""
    _shared_dict = {}
    def __init__(self, params=None):
        if params:
            if 'nli_parameters' in params:
                self._shared_dict['nli_parameters'] = params['nli_parameters']
            if 'raman_parameters' in params:
                self._shared_dict['raman_parameters'] = params['raman_parameters']

    @property
    def nli_params(self):
        return NLIParams(self._shared_dict['nli_parameters'])

    @property
    def raman_params(self):
        return RamanParams(self._shared_dict['raman_parameters'])


class FiberParams(Parameters):
    def __init__(self, params):
        try:
            self._type_variety = params['type_variety']
            self._length = params['length']
            self._loss_coef = params['loss_coef']
            self._length_units = params['length_units']
            # fixed attenuator for padding
            self._att_in = params['att_in'] if 'att_in' in params else 0
            # if not defined in the network json connector loss in/out
            # the None value will be updated in network.py[build_network]
            # with default values from eqpt_config.json[Spans]
            self._con_in = params['con_in'] if 'con_in' in params else None
            self._con_out = params['con_out'] if 'con_out' in params else None
            self._dispersion = params['dispersion']
            self._gamma = params['gamma']
            self._ref_wavelength = params['ref_wavelength'] if 'ref_wavelength' in params else 1550e-9
            self._beta3 = params['beta3'] if 'beta3' in params else 0
        except KeyError:
            raise ParametersError

    @property
    def type_variety(self):
        return self._type_variety

    @property
    def loss_coef(self):
        return self._loss_coef

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def length_units(self):
        return self._length_units

    @property
    def att_in(self):
        return self._att_in

    @property
    def con_in(self):
        return self._con_in

    @con_in.setter
    def con_in(self, con_in):
        self._con_in = con_in

    @property
    def con_out(self):
        return self._con_out

    @con_out.setter
    def con_out(self, con_out):
        self._con_out = con_out

    @property
    def dispersion(self):
        return self._dispersion

    @property
    def gamma(self):
        return self._gamma

    @property
    def ref_wavelength(self):
        return self._ref_wavelength

    @property
    def beta3(self):
        return self._beta3


class RamanFiberParams(FiberParams):
    def __init__(self, params):
        super().__init__(params)
        self._raman_efficiency = params['raman_efficiency'] if 'raman_efficiency' else None

    @property
    def raman_efficiency(self):
        return self._raman_efficiency
