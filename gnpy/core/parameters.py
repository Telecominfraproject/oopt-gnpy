#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""

from scipy.constants import c, pi
from numpy import squeeze, log10, exp

from gnpy.core.utils import db2lin, convert_length
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
        self.propagation_direction = propagation_direction.lower()


class RamanParams(Parameters):
    def __init__(self, flag=False, space_resolution=10e3, tolerance=None):
        """ Simulation parameters used within the Raman Solver
        :params flag: boolean for enabling/disable the evaluation of the Raman Power profile in frequency and position
        :params space_resolution: spatial resolution of the evaluated Raman Power profile
        :params tolerance: tuning parameter for scipy.integrate.solve_bvp solution
        """
        self.flag = flag
        self.space_resolution = space_resolution  # [m]
        self.tolerance = tolerance


class NLIParams(Parameters):
    def __init__(self, method='gn_model_analytic', dispersion_tolerance=1, phase_shift_tolerance=0.1,
                 computed_channels=None, wdm_grid_size=None, f_cut_resolution=None, f_pump_resolution=None):
        """ Simulation parameters used within the Nli Solver
        :params method: formula for NLI calculation
        :params dispersion_tolerance: tuning parameter for ggn model solution
        :params phase_shift_tolerance: tuning parameter for ggn model solution
        :params computed_channels: the NLI is evaluated for these channels and extrapolated for the others
        """
        self.method = method.lower()
        self.dispersion_tolerance = dispersion_tolerance
        self.phase_shift_tolerance = phase_shift_tolerance
        self.computed_channels = computed_channels
        self.wdm_grid_size = wdm_grid_size
        self.f_cut_resolution = f_cut_resolution
        self.f_pump_resolution = f_pump_resolution


class SimParams(Parameters):
    _shared_dict = {'nli_params': NLIParams(), 'raman_params': RamanParams()}

    def __init__(self):
        if type(self) == SimParams:
            raise NotImplementedError('Instances of SimParams cannot be generated')

    @classmethod
    def set_params(cls, sim_params):
        cls._shared_dict['nli_params'] = NLIParams(**sim_params.get('nli_params', {}))
        cls._shared_dict['raman_params'] = RamanParams(**sim_params.get('raman_params', {}))

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
    def default(cls):
        cls._shared_dict = {'nli_params': NLIParams(), 'raman_params': RamanParams()}


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
            self._lin_attenuation = db2lin(self.length * self.loss_coef)
            self._lin_loss_exp = self.loss_coef / (10 * log10(exp(1)))  # linear power exponent loss Neper/m
            self._effective_length = (1 - exp(- self.lin_loss_exp * self.length)) / self.lin_loss_exp
            self._asymptotic_length = 1 / self.lin_loss_exp
            # raman parameters (not compulsory)
            self._raman_efficiency = kwargs['raman_efficiency'] if 'raman_efficiency' in kwargs else None
            self._pumps_loss_coef = kwargs['pumps_loss_coef'] if 'pumps_loss_coef' in kwargs else None
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

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        dictionary['length_units'] = 'm'
        return dictionary
