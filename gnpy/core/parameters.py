#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""
from collections import namedtuple

from scipy.constants import c, pi
from numpy import asarray, array, exp, sqrt, log, outer, ones, squeeze, append, flip, full
from scipy.interpolate import interp1d

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


# SSMF Raman coefficient profile normalized with respect to the effective area overlap (g0 * A_eff(f_probe, f_pump))
GAMMA_RAMAN = array([0.0, 8.534901288127787e-16, 2.646827813788261e-15, 4.416003401040722e-15, 6.161056275749986e-15,
                     7.494237769589302e-15, 8.462609339567226e-15, 9.11294342322821e-15, 9.590404370405774e-15,
                     1.0021251013723174e-14, 1.087950524751747e-14, 1.131510363991119e-14, 1.2158684135288494e-14,
                     1.324795022992073e-14, 1.464362619733337e-14, 1.603390233556577e-14, 1.7480623413437718e-14,
                     1.9345231227060483e-14, 2.1748587210777245e-14, 2.4168776188862157e-14, 2.63159853239411e-14,
                     2.807780420755324e-14, 2.976228762274511e-14, 3.133488594922605e-14, 3.256108937444955e-14,
                     3.3243035093096746e-14, 3.333968452295323e-14, 3.317582998325658e-14, 3.293417277435325e-14,
                     3.250245742862791e-14, 3.0648057577270605e-14, 3.2704501600868567e-14, 2.961424994596558e-14,
                     2.5223129790218825e-14, 1.7369240606559912e-14, 9.916409297327601e-15, 7.229093672193387e-15,
                     6.087952291268887e-15, 5.836446485277513e-15, 7.215831655471871e-15, 7.572465774463005e-15,
                     7.631799094076855e-15, 6.8956291145063906e-15, 5.6370790186966005e-15, 3.684912980244359e-15,
                     2.7594872531250562e-15, 2.484540744792372e-15, 2.2191728882267535e-15, 2.1471386938167345e-15,
                     2.3421016180033473e-15, 2.5610211422217693e-15, 3.0442618266933278e-15, 4.840782970868059e-15,
                     5.472770780858868e-15, 5.236906994259037e-15, 4.517460177253048e-15, 3.3262991642446138e-15,
                     2.2636446394638674e-15, 1.9725429789287957e-15, 1.5202835739817775e-15, 1.2973703347467444e-15,
                     1.1321968153593657e-15, 9.114041614983967e-16, 8.445961873329561e-16, 7.861430005749028e-16,
                     7.839808126801721e-16, 9.014667657496654e-16, 1.3046526189165825e-15, 1.5468673466350297e-15,
                     1.882566342368162e-15, 1.7824768387425245e-15, 1.5963961118763875e-15, 1.1604390575702768e-15,
                     8.520983685630188e-16, 7.637976503903208e-16, 8.158620155451668e-16, 9.063050148211562e-16,
                     9.652622839341152e-16, 9.31435852628614e-16, 8.363952142289423e-16, 7.495722977639876e-16,
                     6.251560014098206e-16, 5.015220251561241e-16, 3.559586939964851e-16, 2.03800409264246e-16,
                     1.4292822878203605e-16, 8.996054308352263e-17, 2.988272600699254e-17, 1.4888889089272364e-17,
                     7.417998462402506e-18]
                    )  # [m/W]

# Note the non-uniform spacing of this range; this is required for properly capturing the Raman peak shape.
FREQ_OFFSET = array([
    0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
    12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75, 19.,
    19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5, 29., 29.5,
    30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5, 39., 39.5, 40.,
    40.5, 41., 41.5, 42.]) * 1e12  # [Hz]

# Raman profile reference frequency
RAMAN_REF_FREQ = 206184634112792  # [Hz] (1454 nm)


class RamanGainCoefficient(namedtuple('RamanGainCoefficient', 'normalized_gamma_raman frequency_offset')):
    """ Raman Gain Coefficient Parameters

        Based on:
            Andrea D’Amico, Bruno Correia, Elliot London, Emanuele Virgillito, Giacomo Borraccini, Antonio Napoli,
            and Vittorio Curri, "Scalable and Disaggregated GGN Approximation Applied to a C+L+S Optical Network,"
            J. Lightwave Technol. 40, 3499-3511 (2022)
            Section III.D
    """


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

            # Reference frequency (unique for all parameters: beta2, beta3, gamma, effective_area)
            if 'ref_wavelength' in kwargs:
                self._ref_wavelength = kwargs['ref_wavelength']
                self._ref_frequency = c / self._ref_wavelength
            elif 'ref_frequency' in kwargs:
                self._ref_frequency = kwargs['ref_frequency']
                self._ref_wavelength = c / self._ref_frequency
            else:
                self._ref_wavelength = 1550e-9  # conventional central C band wavelength [m]
                self._ref_frequency = c / self._ref_wavelength

            # Chromatic Dispersion
            if 'dispersion' in kwargs:
                # Frequency-dependent dispersion
                if type(kwargs['dispersion']) == dict:
                    self._dispersion = asarray(kwargs['dispersion']['value'])
                    self._f_dispersion_ref = asarray(kwargs['dispersion']['frequency'])  # Hz
                    if 'slope' in kwargs['dispersion']:
                        self._dispersion_slope = asarray(kwargs['dispersion']['slope'])  # s/m/m/m
                    elif 'dispersion_slope' in kwargs:
                        self._dispersion_slope = asarray(kwargs['dispersion_slope'])
                    else:
                        w_disperion_ref = flip(c / self._f_dispersion_ref)
                        w_dispesrion = flip(self._dispersion)
                        derivative = (w_dispesrion[1:] - w_dispesrion[:-1]) / (w_disperion_ref[1:] -
                                                                               w_disperion_ref[:-1])
                        averaged_derivative = (derivative[1:] + derivative[:-1]) / 2
                        self._dispersion_slope = flip(append(derivative[0],
                                                             append(averaged_derivative, derivative[-1])))

                    if self._dispersion_slope.size != self._f_dispersion_ref.size:
                        raise ParametersError(
                            'When the dispersion is defined along the frequency, the dispersion slope '
                            'must be defined for each dispersion reference frequency or just omitted.')
                # Single value dispersion
                else:
                    self._dispersion = asarray(kwargs['dispersion'])  # s/m/m
                    self._dispersion_slope = kwargs.get('dispersion_slope')  # s/m/m/m
                    self._f_dispersion_ref = asarray(self._ref_frequency)  # Hz
            else:
                self._dispersion = asarray(1.67e-05)  # s/m/m
                self._dispersion_slope = None
                self._f_dispersion_ref = asarray(self.ref_frequency)

            # Effective Area and Nonlinear Coefficient
            self._effective_area = kwargs.get('effective_area')  # m^2
            self._n1 = 1.45
            self._core_radius = 4.2e-6  # m
            self._n2 = 2.6e-20  # m^2/W
            if self._effective_area is not None:
                default_gamma = 2 * pi * self._n2 / (self._ref_wavelength * self._effective_area)
                self._gamma = kwargs.get('gamma', default_gamma)  # 1/W/m
            elif 'gamma' in kwargs:
                self._gamma = kwargs['gamma']  # 1/W/m
                self._effective_area = 2 * pi * self._n2 / (self._ref_wavelength * self._gamma)  # m^2
            else:
                self._effective_area = 83e-12  # m^2
                self._gamma = 2 * pi * self._n2 / (self._ref_wavelength * self._effective_area)  # 1/W/m
            self._contrast = 0.5 * (c / (2 * pi * self._ref_frequency * self._core_radius * self._n1) * exp(
                pi * self._core_radius ** 2 / self._effective_area)) ** 2

            # Raman Gain Coefficient
            default_raman_coefficient = \
                {
                    'gamma_raman': GAMMA_RAMAN,
                    'frequency_offset': FREQ_OFFSET,
                    'reference_frequency': RAMAN_REF_FREQ
                }
            raman_coefficient = kwargs.get('raman_coefficient', default_raman_coefficient)
            if 'g0' in raman_coefficient:
                g0 = asarray(raman_coefficient['g0'])
                raman_reference_frequency = raman_coefficient['reference_frequency']
                frequency_offset = asarray(raman_coefficient['frequency_offset'])
                stokes_wave = raman_reference_frequency - frequency_offset
                gamma_raman = g0 * self.effective_area_overlap(stokes_wave, raman_reference_frequency)
                gamma_raman = raman_coefficient.get('gamma_raman', gamma_raman)
            else:
                gamma_raman = asarray(raman_coefficient['gamma_raman'])
                frequency_offset = asarray(raman_coefficient['frequency_offset'])
                raman_reference_frequency = raman_coefficient['reference_frequency']

            normalized_gamma_raman = gamma_raman / raman_reference_frequency  # m / (W Hz)
            self._raman_reference_frequency = raman_reference_frequency

            # Raman gain coefficient array of the frequency offset constructed such that positive frequency values
            # represent a positive power transfer from higher frequency and vice versa
            frequency_offset = append(-flip(frequency_offset[1:]), frequency_offset)
            normalized_gamma_raman = append(- flip(normalized_gamma_raman[1:]), normalized_gamma_raman)
            self._raman_coefficient = RamanGainCoefficient(normalized_gamma_raman, frequency_offset)

            # Polarization Mode Dispersion
            self._pmd_coef = kwargs['pmd_coef']  # s/sqrt(m)

            # Loss Coefficient
            if isinstance(kwargs['loss_coef'], dict):
                self._loss_coef = asarray(kwargs['loss_coef']['value']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = asarray(kwargs['loss_coef']['frequency'])  # Hz
            else:
                self._loss_coef = asarray(kwargs['loss_coef']) * 1e-3  # lineic loss dB/m
                self._f_loss_ref = asarray(self._ref_frequency)  # Hz

            # Lumped Losses
            self._lumped_losses = kwargs['lumped_losses'] if 'lumped_losses' in kwargs else array([])
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
    def f_dispersion_ref(self):
        return self._f_dispersion_ref

    @property
    def dispersion_slope(self):
        return self._dispersion_slope

    def dispersion_scaling(self, frequency):
        if self.dispersion.size > 1:
            dispersion = interp1d(self.f_dispersion_ref, self.dispersion)(frequency)
        elif self.dispersion_slope is not None:
            wavelength = c / frequency
            dispersion = self.dispersion + self.dispersion_slope * (wavelength - c / self.f_dispersion_ref)
        else:
            dispersion = (frequency / self.f_dispersion_ref) ** 2 * self.dispersion
        return dispersion

    @property
    def gamma(self):
        return self._gamma

    def effective_area_scaling(self, frequency):
        V = 2 * pi * frequency / c * self._core_radius * self._n1 * sqrt(2 * self._contrast)
        w = self._core_radius / sqrt(log(V))
        return asarray(pi * w ** 2)

    def effective_area_overlap(self, frequency_stokes_wave, frequency_pump):
        effective_area_stokes_wave = self.effective_area_scaling(frequency_stokes_wave)
        effective_area_pump = self.effective_area_scaling(frequency_pump)
        return squeeze(outer(effective_area_stokes_wave, ones(effective_area_pump.size)) + outer(
            ones(effective_area_stokes_wave.size), effective_area_pump)) / 2

    def gamma_scaling(self, frequency):
        return asarray(2 * pi * self._n2 * frequency / (c * self.effective_area_scaling(frequency)))

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
    def loss_coef(self):
        return self._loss_coef

    @property
    def f_loss_ref(self):
        return self._f_loss_ref

    @property
    def raman_coefficient(self):
        return self._raman_coefficient

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        dictionary['length_units'] = 'm'
        if len(self.lumped_losses) == 0:
            dictionary.pop('lumped_losses')
        if not self.raman_coefficient:
            dictionary.pop('raman_coefficient')
        else:
            gamma_raman = self.raman_coefficient.normalized_gamma_raman * self._raman_reference_frequency
            dictionary['raman_coefficient'] = {'gamma_raman': gamma_raman.tolist(),
                                               'frequency_offset': self.raman_coefficient.frequency_offset.tolist(),
                                               'reference_frequency': self._raman_reference_frequency}
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
