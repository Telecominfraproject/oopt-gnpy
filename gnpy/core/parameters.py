#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
==================

This module contains all parameters to configure standard network elements.

"""

from logging import getLogger
from scipy.constants import c, pi
from numpy import squeeze, log10, exp, delete


from gnpy.core.units import UNITS
from gnpy.core.utils import db2lin
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
        self._space_resolution = kwargs['space_resolution'] if 'space_resolution' in kwargs else None
        self._tolerance = kwargs['tolerance'] if 'tolerance' in kwargs else None

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
        self._phase_shift_tolerance = kwargs['phase_shift_tolerance']
        self._f_cut_resolution = None
        self._f_pump_resolution = None
        self._computed_channels = kwargs['computed_channels'] if 'computed_channels' in kwargs else None

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
    def phase_shift_tolerance(self):
        return self._phase_shift_tolerance

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
        return self._computed_channels


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
            self._length_units_factor = UNITS[kwargs['length_units']]
            self._length = kwargs['length'] * self._length_units_factor  # m
            self._length_units = 'm'
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
            self._beta2 = (self._ref_wavelength ** 2) * abs(self._dispersion) / (2 * pi * c)  # 1/(m * Hz^2)
            self._beta3 = kwargs['beta3'] if 'beta3' in kwargs else 0
            if type(kwargs['loss_coef']) == dict:
                self._loss_coef = squeeze(kwargs['loss_coef']['loss_coef_power']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = squeeze(kwargs['loss_coef']['frequency'])  # Hz
            else:
                self._loss_coef = kwargs['loss_coef'] * 1e-3  # lineic loss dB/m
                self._f_loss_ref = 193.5e12  # Hz
            self._lin_attenuation = db2lin(self._length * self._loss_coef)
            self._lin_loss_exp = self._loss_coef / (10 * log10(exp(1)))  # linear power exponent loss Neper/m
            self._effective_length = (1 - exp(- self._lin_loss_exp * self._length)) / self._lin_loss_exp
            self._asymptotic_length = 1 / self._lin_loss_exp
            # raman parameters (not compulsory)
            self._raman_efficiency = kwargs['raman_efficiency'] if 'raman_efficiency' in kwargs else None
            self._pumps_loss_coef = kwargs['pumps_loss_coef'] if 'pumps_loss_coef' in kwargs else None
            if 'lumped_losses' in kwargs:
                try:
                    z_lumped_losses_ref = squeeze([lumped['position'] for lumped in kwargs['lumped_losses']]) * 1e3
                    lumped_losses_power = squeeze([lumped['loss'] for lumped in kwargs['lumped_losses']])
                except TypeError:
                    raise ParametersError(f'lumped losses has to be a list of dictionaries.')
                except KeyError as e:
                    raise ParametersError(f'An element in lumped losses list lacks the compulsory field: {e}.')
                # Sorting elements along the z axis
                indices = z_lumped_losses_ref.argsort()
                lumped_losses_power = lumped_losses_power[indices]
                z_lumped_losses_ref = z_lumped_losses_ref[indices]
                # Checking if any loss position is within the span length
                if z_lumped_losses_ref[0] < 0 or z_lumped_losses_ref[-1] > self.length:
                    raise ParametersError('Lumped losses exceed the size of the fiber.')
                # Check the match of input connector loss
                if z_lumped_losses_ref[0] == 0:
                    if lumped_losses_power[0] != self.con_in:
                        raise ParametersError('Parameter mismatch: con_in differs from lumped_losses in position z=0.')
                    else:
                        lumped_losses_power = delete(lumped_losses_power, 0)
                        z_lumped_losses_ref = delete(z_lumped_losses_ref, 0)
                # Check the match of output connector loss
                if z_lumped_losses_ref[-1] == self.length:
                    if lumped_losses_power[-1] != self.con_out:
                        message = f'Parameter mismatch: con_out differs from lumped_losses in position z={self.length}.'
                        raise ParametersError(message)
                    else:
                        lumped_losses_power = delete(lumped_losses_power, -1)
                        z_lumped_losses_ref = delete(z_lumped_losses_ref, -1)

                lumped_losses_power_lin = db2lin(lumped_losses_power)  # [linear units]
                self._lumped_losses = {}
                self._lumped_losses['loss'] = lumped_losses_power_lin
                self._lumped_losses['position'] = z_lumped_losses_ref
            else:
                self._lumped_losses = None
        except KeyError as e:
            raise ParametersError(f'Fiber configurations json must include {e}')

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        """length must be in m"""
        self._length = length

    @property
    def length_units(self):
        return self._length_units

    @property
    def length_units_factor(self):
        return self._length_units_factor

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

    @property
    def loss_coef(self):
        return self._loss_coef

    @property
    def f_loss_ref(self):
        return self._f_loss_ref

    @property
    def lin_loss_exp(self):
        return self._lin_loss_exp

    @property
    def lin_attenuation(self):
        return self._lin_attenuation

    @property
    def effective_length(self):
        return self._effective_length

    @property
    def asymptotic_length(self):
        return self._asymptotic_length

    @property
    def raman_efficiency(self):
        return self._raman_efficiency

    @property
    def pumps_loss_coef(self):
        return self._pumps_loss_coef

    @property
    def lumped_losses(self):
        return self._lumped_losses

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        return dictionary

