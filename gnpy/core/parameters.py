#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""

from scipy.constants import c, pi
from numpy import squeeze

from gnpy.core.utils import convert_length
from gnpy.core.exceptions import ParametersError


class Parameters:
    def asdict(self):
        class_dict = self.__class__.__dict__
        instance_dict = self.__dict__
        new_dict = {}
        for key in class_dict:
            if isinstance(class_dict[key], property):
                new_dict[key] = instance_dict['_' + key]
        return new_dict


class PumpParams(Parameters):
    def __init__(self, power, frequency, propagation_direction):
        self.power = power
        self.frequency = frequency
        self.propagation_direction = propagation_direction


class RamanParams(Parameters):
    def __init__(self, raman_params):
        self.flag = raman_params['flag'] if 'flag' in raman_params else False
        self.solver_spatial_step = \
            raman_params['solver_spatial_step'] if 'solver_spatial_step' in raman_params else 50  # [m]
        self.spatial_resolution = \
            raman_params['spatial_resolution'] if 'spatial_resolution' in raman_params else 10e3  # [m]


class NLIParams(Parameters):
    def __init__(self, nli_params):
        self.method = nli_params['method'] if 'method' in nli_params else 'gn_model_analytic'
        self.dispersion_tolerance = nli_params['dispersion_tolerance'] if 'dispersion_tolerance' in nli_params else 1
        self.phase_shift_tolerance = nli_params['phase_shift_tolerance'] if 'phase_shift_tolerance' in nli_params else 0.1
        self.computed_channels = nli_params['computed_channels'] if 'computed_channels' in nli_params else None


class SimParams(Parameters):
    _shared_dict = {'nli_params': NLIParams({}), 'raman_params': RamanParams({})}

    def __init__(self):
        if type(self) == SimParams:
            raise NotImplementedError('Instances of SimParams cannot be generated')

    @classmethod
    def set_params(cls, sim_params):
        if sim_params is not None:
            if 'nli_params' in sim_params:
                cls._shared_dict['nli_params'] = NLIParams(sim_params['nli_params'])
            if 'raman_params' in sim_params:
                cls._shared_dict['raman_params'] = RamanParams(sim_params['raman_params'])

    @classmethod
    def get(cls):
        self = cls.__new__(cls)
        return self

    @property
    def nli_params(self):
        return self._shared_dict['nli_params']

    @property
    def raman_params(self):
        return self._shared_dict['raman_params']

    @classmethod
    def reset(cls):
        cls._shared_dict = {'nli_params': NLIParams({}), 'raman_params': RamanParams({})}


class FiberParams(Parameters):
    def __init__(self, **kwargs):
        try:
            self._length = convert_length(kwargs['length'], kwargs['length_units'])
            # fixed attenuator for padding
            self._att_in = kwargs['att_in'] if 'att_in' in kwargs else 0
            # if not defined in the network json connector loss in/out
            # the None value will be updated in network.py[build_network]
            # with default values from eqpt_config.json[Spans]
            self._con_in = kwargs['con_in'] if 'con_in' in kwargs else None
            self._con_out = kwargs['con_out'] if 'con_out' in kwargs else None
            if 'ref_wavelength' in kwargs:
                self._ref_wavelength = kwargs['ref_wavelength']
                self._ref_frequency = c / self.ref_wavelength
            elif 'ref_frequency' in kwargs:
                self._ref_frequency = kwargs['ref_frequency']
                self._ref_wavelength = c / self.ref_frequency
            else:
                self._ref_wavelength = 1550e-9
                self._ref_frequency = c / self.ref_wavelength
            self._dispersion = kwargs['dispersion']  # s/m/m
            self._dispersion_slope = kwargs['dispersion_slope'] if 'dispersion_slope' in kwargs else \
                -2 * self._dispersion/self.ref_wavelength  # s/m/m/m
            self._beta2 = -(self.ref_wavelength ** 2) * self.dispersion / (2 * pi * c)  # 1/(m * Hz^2)
            # Eq. (3.23) in  Abramczyk, Halina. "Dispersion phenomena in optical fibers." Virtual European University
            # on Lasers. Available online: http://mitr.p.lodz.pl/evu/lectures/Abramczyk3.pdf
            # (accessed on 25 March 2018) (2005).
            self._beta3 = ((self.dispersion_slope - (4*pi*c/self.ref_wavelength**3) * self.beta2) /
                           (2*pi*c/self.ref_wavelength**2)**2)
            self._gamma = kwargs['gamma']  # 1/W/m
            self._pmd_coef = kwargs['pmd_coef']  # s/sqrt(m)
            if type(kwargs['loss_coef']) == dict:
                self._loss_coef = squeeze(kwargs['loss_coef']['loss_coef_power']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = squeeze(kwargs['loss_coef']['frequency'])  # Hz
            else:
                self._loss_coef = kwargs['loss_coef'] * 1e-3  # lineic loss dB/m
                self._f_loss_ref = 193.5e12  # Hz
            # raman parameters (not compulsory)
            self._raman_efficiency = kwargs['raman_efficiency'] if 'raman_efficiency' in kwargs else None
            # lumped losses
            self._lumped_losses = kwargs['lumped_losses'] if 'lumped_losses' in kwargs else None
        except KeyError as e:
            raise ParametersError(f'Fiber configurations json must include {e}. Configuration: {kwargs}')

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
    def lumped_losses(self):
        return self._lumped_losses

    @property
    def dispersion(self):
        return self._dispersion

    @property
    def dispersion_slope(self):
        return self._dispersion_slope

    @property
    def gamma(self):
        return self._gamma

    @property
    def pmd_coef(self):
        return self._pmd_coef

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
    def raman_efficiency(self):
        return self._raman_efficiency

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        dictionary['length_units'] = 'm'
        if not self.lumped_losses:
            dictionary.pop('lumped_losses')
        if not self.raman_efficiency:
            dictionary.pop('raman_efficiency')
        return dictionary
