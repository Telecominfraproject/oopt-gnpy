#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""

from scipy.constants import c, pi
from numpy import asarray, array

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
        self.propagation_direction = propagation_direction.lower()


class RamanParams(Parameters):
    def __init__(self, flag=False, result_spatial_resolution=10e3, solver_spatial_resolution=50):
        """Simulation parameters used within the Raman Solver

        :params flag: boolean for enabling/disable the evaluation of the Raman power profile in frequency and position
        :params result_spatial_resolution: spatial resolution of the evaluated Raman power profile
        :params solver_spatial_resolution: spatial step for the iterative solution of the first order ode
        """
        self.flag = flag
        self.result_spatial_resolution = result_spatial_resolution  # [m]
        self.solver_spatial_resolution = solver_spatial_resolution  # [m]


class NLIParams(Parameters):
    def __init__(self, method='gn_model_analytic', dispersion_tolerance=1, phase_shift_tolerance=0.1,
                 computed_channels=None):
        """Simulation parameters used within the Nli Solver

        :params method: formula for NLI calculation
        :params dispersion_tolerance: tuning parameter for ggn model solution
        :params phase_shift_tolerance: tuning parameter for ggn model solution
        :params computed_channels: the NLI is evaluated for these channels and extrapolated for the others
        """
        self.method = method.lower()
        self.dispersion_tolerance = dispersion_tolerance
        self.phase_shift_tolerance = phase_shift_tolerance
        self.computed_channels = computed_channels


class SimParams(Parameters):
    _shared_dict = {'nli_params': NLIParams(), 'raman_params': RamanParams()}

    @classmethod
    def set_params(cls, sim_params):
        cls._shared_dict['nli_params'] = NLIParams(**sim_params.get('nli_params', {}))
        cls._shared_dict['raman_params'] = RamanParams(**sim_params.get('raman_params', {}))

    @property
    def nli_params(self):
        return self._shared_dict['nli_params']

    @property
    def raman_params(self):
        return self._shared_dict['raman_params']


class RoadmParams(Parameters):
    def __init__(self, **kwargs):
        self.target_pch_out_db = kwargs.get('target_pch_out_db')
        self.target_psd_out_mWperGHz = kwargs.get('target_psd_out_mWperGHz')
        self.target_out_mWperSlotWidth = kwargs.get('target_out_mWperSlotWidth')
        equalisation_type = ['target_pch_out_db', 'target_psd_out_mWperGHz', 'target_out_mWperSlotWidth']
        temp = [kwargs.get(k) is not None for k in equalisation_type]
        if sum(temp) > 1:
            raise ParametersError('ROADM config contains more than one equalisation type.'
                                  + 'Please choose only one', kwargs)
        self.per_degree_pch_out_db = kwargs.get('per_degree_pch_out_db', {})
        self.per_degree_pch_psd = kwargs.get('per_degree_psd_out_mWperGHz', {})
        self.per_degree_pch_psw = kwargs.get('per_degree_psd_out_mWperSlotWidth', {})
        try:
            self.add_drop_osnr = kwargs['add_drop_osnr']
            self.pmd = kwargs['pmd']
            self.pdl = kwargs['pdl']
            self.restrictions = kwargs['restrictions']
        except KeyError as e:
            raise ParametersError(f'ROADM configurations must include {e}. Configuration: {kwargs}')


class FusedParams(Parameters):
    def __init__(self, **kwargs):
        self.loss = kwargs['loss'] if 'loss' in kwargs else 1


# SSMF Raman coefficient profile normalized with respect to the effective area (Cr * A_eff)
CR_NORM = array([
    0., 7.802e-16, 2.4236e-15, 4.0504e-15, 5.6606e-15, 6.8973e-15, 7.802e-15, 8.4162e-15, 8.8727e-15, 9.2877e-15,
    1.01011e-14, 1.05244e-14, 1.13295e-14, 1.2367e-14, 1.3695e-14, 1.5023e-14, 1.64091e-14, 1.81936e-14, 2.04927e-14,
    2.28167e-14, 2.48917e-14, 2.66098e-14, 2.82615e-14, 2.98136e-14, 3.1042e-14, 3.17558e-14, 3.18803e-14, 3.17558e-14,
    3.15566e-14, 3.11748e-14, 2.94567e-14, 3.14985e-14, 2.8552e-14, 2.43439e-14, 1.67992e-14, 9.6114e-15, 7.02180e-15,
    5.9262e-15, 5.6938e-15, 7.055e-15, 7.4119e-15, 7.4783e-15, 6.7645e-15, 5.5361e-15, 3.6271e-15, 2.7224e-15,
    2.4568e-15, 2.1995e-15, 2.1331e-15, 2.3323e-15, 2.5564e-15, 3.0461e-15, 4.8555e-15, 5.5029e-15, 5.2788e-15,
    4.565e-15, 3.3698e-15, 2.2991e-15, 2.0086e-15, 1.5521e-15, 1.328e-15, 1.162e-15, 9.379e-16, 8.715e-16, 8.134e-16,
    8.134e-16, 9.379e-16, 1.3612e-15, 1.6185e-15, 1.9754e-15, 1.8758e-15, 1.6849e-15, 1.2284e-15, 9.047e-16, 8.134e-16,
    8.715e-16, 9.711e-16, 1.0375e-15, 1.0043e-15, 9.047e-16, 8.134e-16, 6.806e-16, 5.478e-16, 3.901e-16, 2.241e-16,
    1.577e-16, 9.96e-17, 3.32e-17, 1.66e-17, 8.3e-18])

# Note the non-uniform spacing of this range; this is required for properly capturing the Raman peak shape.
FREQ_OFFSET = array([
    0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
    12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75, 19.,
    19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5, 29., 29.5,
    30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5, 39., 39.5, 40.,
    40.5, 41., 41.5, 42.]) * 1e12


class FiberParams(Parameters):
    def __init__(self, **kwargs):
        try:
            self._length = convert_length(kwargs['length'], kwargs['length_units'])
            # fixed attenuator for padding
            self._att_in = kwargs.get('att_in', 0)
            # if not defined in the network json connector loss in/out
            # the None value will be updated in network.py[build_network]
            # with default values from eqpt_config.json[Spans]
            self._con_in = kwargs.get('con_in')
            self._con_out = kwargs.get('con_out')
            if 'ref_wavelength' in kwargs:
                self._ref_wavelength = kwargs['ref_wavelength']
                self._ref_frequency = c / self._ref_wavelength
            elif 'ref_frequency' in kwargs:
                self._ref_frequency = kwargs['ref_frequency']
                self._ref_wavelength = c / self._ref_frequency
            else:
                self._ref_wavelength = 1550e-9  # conventional central C band wavelength [m]
                self._ref_frequency = c / self._ref_wavelength
            self._dispersion = kwargs['dispersion']  # s/m/m
            self._dispersion_slope = \
                kwargs.get('dispersion_slope', -2 * self._dispersion / self.ref_wavelength)  # s/m/m/m
            self._beta2 = -(self.ref_wavelength ** 2) * self.dispersion / (2 * pi * c)  # 1/(m * Hz^2)
            # Eq. (3.23) in  Abramczyk, Halina. "Dispersion phenomena in optical fibers." Virtual European University
            # on Lasers. Available online: http://mitr.p.lodz.pl/evu/lectures/Abramczyk3.pdf
            # (accessed on 25 March 2018) (2005).
            self._beta3 = ((self.dispersion_slope - (4*pi*c/self.ref_wavelength**3) * self.beta2) /
                           (2*pi*c/self.ref_wavelength**2)**2)
            self._effective_area = kwargs.get('effective_area')  # m^2
            self._n1 = 1.468
            n2 = 2.6e-20  # m^2/W
            if self._effective_area:
                self._gamma = kwargs.get('gamma', 2 * pi * n2 / (self.ref_wavelength * self._effective_area))  # 1/W/m
            elif 'gamma' in kwargs:
                self._gamma = kwargs['gamma']  # 1/W/m
                self._effective_area = 2 * pi * n2 / (self.ref_wavelength * self._gamma)  # m^2
            else:
                self._gamma = 0  # 1/W/m
                self._effective_area = 83e-12  # m^2
            default_raman_efficiency = {'cr': CR_NORM / self._effective_area, 'frequency_offset': FREQ_OFFSET}
            self._raman_efficiency = kwargs.get('raman_efficiency', default_raman_efficiency)
            self._pmd_coef = kwargs['pmd_coef']  # s/sqrt(m)
            if isinstance(kwargs['loss_coef'], dict):
                self._loss_coef = asarray(kwargs['loss_coef']['value']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = asarray(kwargs['loss_coef']['frequency'])  # Hz
            else:
                self._loss_coef = asarray(kwargs['loss_coef']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = asarray(self._ref_frequency)  # Hz
            self._lumped_losses = kwargs['lumped_losses'] if 'lumped_losses' in kwargs else []
            self._latency = self._length / (c / self._n1)  # s
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

    @property
    def lumped_losses(self):
        return self._lumped_losses

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
    def raman_efficiency(self):
        return self._raman_efficiency

    @property
    def latency(self):
        return self._latency

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        dictionary['length_units'] = 'm'
        if not self.lumped_losses:
            dictionary.pop('lumped_losses')
        if not self.raman_efficiency:
            dictionary.pop('raman_efficiency')
        return dictionary


class EdfaParams:
    def __init__(self, **params):
        self.update_params(params)
        if params == {}:
            self.type_variety = ''
            self.type_def = ''
            # self.gain_flatmax = 0
            # self.gain_min = 0
            # self.p_max = 0
            # self.nf_model = None
            # self.nf_fit_coeff = None
            # self.nf_ripple = None
            # self.dgt = None
            # self.gain_ripple = None
            # self.out_voa_auto = False
            # self.allowed_for_design = None

    def update_params(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, self.update_params(**v) if isinstance(v, dict) else v)


class EdfaOperational:
    default_values = {
        'gain_target': None,
        'delta_p': None,
        'out_voa': None,
        'tilt_target': 0
    }

    def __init__(self, **operational):
        self.update_attr(operational)

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            setattr(self, k, clean_kwargs.get(k, v))

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'gain_target={self.gain_target!r}, '
                f'tilt_target={self.tilt_target!r})')
