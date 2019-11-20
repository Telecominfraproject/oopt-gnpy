#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
==================

This module contains all parameters to configure standard network elements.

"""

from logging import getLogger
from scipy.constants import c,pi


from gnpy.core.units import UNITS
from gnpy.core.exceptions import ParametersError


logger = getLogger(__name__)


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
    def __init__(self, **kwargs):
        self._flag_raman = kwargs['flag_raman']
        self._space_resolution = kwargs['space_resolution']
        self._tolerance = kwargs['tolerance']

    @property
    def flag_raman(self):
        return self._flag_raman

    @property
    def space_resolution(self):
        return self._space_resolution

    @property
    def tolerance(self):
        return self._tolerance


class NLIParams(Parameters):
    def __init__(self, **kwargs):
        self._nli_method_name = kwargs['nli_method_name']
        self._wdm_grid_size = kwargs['wdm_grid_size']
        self._dispersion_tolerance = kwargs['dispersion_tolerance']
        self._phase_shift_tollerance = kwargs['phase_shift_tollerance']
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


    @property
    def computed_channels(self):
        return self.computed_channels


class SimParams(Parameters):
    def __init__(self, **kwargs):
        if kwargs:
            if 'nli_parameters' in kwargs:
                self._nli_params = NLIParams(**kwargs['nli_parameters'])
            else:
                self._nli_params = None
            if 'raman_parameters' in kwargs:
                self._raman_params = RamanParams(**kwargs['raman_parameters'])
            else:
                self._raman_params = None

    @property
    def nli_params(self):
        return self._nli_params

    @property
    def raman_params(self):
        return self._raman_params


class FiberParams(Parameters):
    def __init__(self, **kwargs):
        try:
            self.length = kwargs['length'] * UNITS[kwargs['length_units']]  # m
            self._loss_coef = kwargs['loss_coef']
            # fixed attenuator for padding
            self._att_in = kwargs['att_in'] if 'att_in' in kwargs else 0
            # if not defined in the network json connector loss in/out
            # the None value will be updated in network.py[build_network]
            # with default values from eqpt_config.json[Spans]
            self._con_in = kwargs['con_in'] if 'con_in' in kwargs else None
            self._con_out = kwargs['con_out'] if 'con_out' in kwargs else None
            self._gamma = kwargs['gamma']  # 1/W/m
            self._dispersion = kwargs['dispersion']  # s/m/m
            if 'ref_wavelength' in kwargs:
                self._ref_wavelength = kwargs['ref_wavelength']
                self._ref_frequency = c / self._ref_wavelength
            elif 'ref_frequency' in kwargs:
                self._ref_frequency = kwargs['ref_frequency']
                self._ref_wavelength = c / self._ref_frequency
            else:
                self._ref_wavelength = 1550e-9
                self._ref_frequency = c / self._ref_wavelength
            self._beta2 = (self._ref_wavelength ** 2) * abs(self._dispersion) / (2 * pi * c)  # 10^21 scales [ps^2/km]
            self._beta3 = kwargs['beta3'] if 'beta3' in kwargs else 0
        except KeyError:
            raise ParametersError

    @property
    def loss_coef(self):
        return self._loss_coef

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        """length must be in m"""
        self._length = length

    @property
    def att_in(self):
        return self._att_in

    @att_in.setter
    def att_in(self, att_in):
        self._att_in = att_in

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
    def ref_frequency(self):
        return self._ref_frequency

    @property
    def beta2(self):
        return self._beta2

    @property
    def beta3(self):
        return self._beta3


class RamanFiberParams(FiberParams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raman_efficiency = kwargs['raman_efficiency'] if 'raman_efficiency' else None

    @property
    def raman_efficiency(self):
        return self._raman_efficiency
