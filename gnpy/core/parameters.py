#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""
from collections import namedtuple

from scipy.constants import c, pi
from numpy import asarray, array, exp, sqrt, log, outer, ones, squeeze, append, flip, linspace, full

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


DEFAULT_RAMAN_COEFFICIENT = {
    # SSMF Raman coefficient profile normalized with respect to the effective area overlap (g0 * A_eff(f_probe, f_pump))
    'g0': array(
        [0.00000000e+00, 1.12351610e-05, 3.47838074e-05, 5.79356636e-05, 8.06921680e-05, 9.79845709e-05, 1.10454361e-04,
         1.18735302e-04, 1.24736889e-04, 1.30110053e-04, 1.41001273e-04, 1.46383247e-04, 1.57011792e-04, 1.70765865e-04,
         1.88408911e-04, 2.05914127e-04, 2.24074028e-04, 2.47508283e-04, 2.77729174e-04, 3.08044243e-04, 3.34764439e-04,
         3.56481704e-04, 3.77127256e-04, 3.96269124e-04, 4.10955175e-04, 4.18718761e-04, 4.19511263e-04, 4.17025384e-04,
         4.13565369e-04, 4.07726048e-04, 3.83671291e-04, 4.08564283e-04, 3.69571936e-04, 3.14442090e-04, 2.16074535e-04,
         1.23097823e-04, 8.95457457e-05, 7.52470400e-05, 7.19806145e-05, 8.87961158e-05, 9.30812065e-05, 9.37058268e-05,
         8.45719619e-05, 6.90585286e-05, 4.50407159e-05, 3.36521245e-05, 3.02292475e-05, 2.69376939e-05, 2.60020897e-05,
         2.82958958e-05, 3.08667558e-05, 3.66024657e-05, 5.80610307e-05, 6.54797937e-05, 6.25022715e-05, 5.37806442e-05,
         3.94996621e-05, 2.68120644e-05, 2.33038554e-05, 1.79140757e-05, 1.52472424e-05, 1.32707565e-05, 1.06541760e-05,
         9.84649374e-06, 9.13999627e-06, 9.08971012e-06, 1.04227525e-05, 1.50419271e-05, 1.77838232e-05, 2.15810815e-05,
         2.03744008e-05, 1.81939341e-05, 1.31862121e-05, 9.65352116e-06, 8.62698322e-06, 9.18688016e-06, 1.01737784e-05,
         1.08017817e-05, 1.03903588e-05, 9.30040333e-06, 8.30809173e-06, 6.90650401e-06, 5.52238029e-06, 3.90648708e-06,
         2.22908227e-06, 1.55796177e-06, 9.77218716e-07, 3.23477236e-07, 1.60602454e-07, 7.97306386e-08]
    ),  # [m/W]

    # Note the non-uniform spacing of this range; this is required for properly capturing the Raman peak shape.
    'frequency_offset': array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5,
        12.,
        12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75, 19.,
        19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5, 29.,
        29.5,
        30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5, 39., 39.5,
        40.,
        40.5, 41., 41.5, 42.]
    ) * 1e12,  # [Hz]

    # Raman profile reference frequency
    'reference_frequency': 206184634112792  # [Hz] (1454 nm)}
}


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
            if 'dispersion_per_frequency' in kwargs:
                # Frequency-dependent dispersion
                self._dispersion = asarray(kwargs['dispersion_per_frequency']['value'])  # s/m/m
                self._f_dispersion_ref = asarray(kwargs['dispersion_per_frequency']['frequency'])  # Hz
                self._dispersion_slope = None
            elif 'dispersion' in kwargs:
                # Single value dispersion
                self._dispersion = asarray(kwargs['dispersion'])  # s/m/m
                self._dispersion_slope = kwargs.get('dispersion_slope')  # s/m/m/m
                self._f_dispersion_ref = asarray(self._ref_frequency)  # Hz
            else:
                # Default single value dispersion
                self._dispersion = asarray(1.67e-05)  # s/m/m
                self._dispersion_slope = None
                self._f_dispersion_ref = asarray(self.ref_frequency)  # Hz

            # Effective Area and Nonlinear Coefficient
            self._effective_area = kwargs.get('effective_area')  # m^2
            self._n1 = 1.468
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
            raman_coefficient = kwargs.get('raman_coefficient', DEFAULT_RAMAN_COEFFICIENT)
            self._g0 = asarray(raman_coefficient['g0'])
            raman_reference_frequency = raman_coefficient['reference_frequency']
            frequency_offset = asarray(raman_coefficient['frequency_offset'])
            stokes_wave = raman_reference_frequency - frequency_offset
            gamma_raman = self._g0 * self.effective_area_overlap(stokes_wave, raman_reference_frequency)
            normalized_gamma_raman = gamma_raman / raman_reference_frequency  # 1 / m / W / Hz
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
    def f_dispersion_ref(self):
        return self._f_dispersion_ref

    @property
    def dispersion_slope(self):
        return self._dispersion_slope

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

    @property
    def latency(self):
        return self._latency

    def asdict(self):
        dictionary = super().asdict()
        dictionary['loss_coef'] = self.loss_coef * 1e3
        dictionary['length_units'] = 'm'
        if len(self.lumped_losses) == 0:
            dictionary.pop('lumped_losses')
        if not self.raman_coefficient:
            dictionary.pop('raman_coefficient')
        else:
            raman_frequency_offset = \
                self.raman_coefficient.frequency_offset[self.raman_coefficient.frequency_offset >= 0]
            dictionary['raman_coefficient'] = {'g0': self._g0.tolist(),
                                               'frequency_offset': raman_frequency_offset.tolist(),
                                               'reference_frequency': self._raman_reference_frequency}
        return dictionary


class EdfaParams:
    def __init__(self, **params):
        try:
            self.type_variety = params['type_variety']
            self.type_def = params['type_def']

            # Bandwidth
            self.f_min = params['f_min']
            self.f_max = params['f_max']
            self.bandwidth = self.f_max - self.f_min
            self.f_cent = (self.f_max + self.f_min) / 2
            self.f_ripple_ref = params['f_ripple_ref']

            # Gain
            self.gain_flatmax = params['gain_flatmax']
            self.gain_min = params['gain_min']

            gain_ripple = params['gain_ripple']
            if gain_ripple == 0:
                self.gain_ripple = asarray([0, 0])
                self.f_ripple_ref = asarray([self.f_min, self.f_max])
            else:
                self.gain_ripple = asarray(gain_ripple)
                if self.f_ripple_ref is not None:
                    if (self.f_ripple_ref[0] != self.f_min) or (self.f_ripple_ref[-1] != self.f_max):
                        raise ParametersError("The reference ripple frequency maximum and minimum have to coincide "
                                              "with the EDFA frequency maximum and minimum.")
                    elif self.gain_ripple.size != self.f_ripple_ref.size:
                        raise ParametersError("The reference ripple frequency and the gain ripple must have the same "
                                              "size.")
                else:
                    self.f_ripple_ref = linspace(self.f_min, self.f_max, self.gain_ripple.size)

            tilt_ripple = params['tilt_ripple']

            if tilt_ripple == 0:
                self.tilt_ripple = full(self.gain_ripple.size, 0)
            else:
                self.tilt_ripple = asarray(tilt_ripple)
                if self.tilt_ripple.size != self.gain_ripple.size:
                    raise ParametersError("The tilt ripple and the gain ripple must have the same size.")

            # Power
            self.p_max = params['p_max']

            # Noise Figure
            self.nf_model = params['nf_model']
            self.nf_min = params['nf_min']
            self.nf_max = params['nf_max']
            self.nf_coef = params['nf_coef']
            self.nf0 = params['nf0']
            self.nf_fit_coeff = params['nf_fit_coeff']

            nf_ripple = params['nf_ripple']
            if nf_ripple == 0:
                self.nf_ripple = full(self.gain_ripple.size, 0)
            else:
                self.nf_ripple = asarray(nf_ripple)
                if self.nf_ripple.size != self.gain_ripple.size:
                    raise ParametersError("The noise figure ripple and the gain ripple must have the same size.")

            # VOA
            self.out_voa_auto = params['out_voa_auto']

            # Dual Stage
            self.dual_stage_model = params['dual_stage_model']
            if self.dual_stage_model is not None:
                # Preamp
                self.preamp_variety = params['preamp_variety']
                self.preamp_type_def = params['preamp_type_def']
                self.preamp_nf_model = params['preamp_nf_model']
                self.preamp_nf_fit_coeff = params['preamp_nf_fit_coeff']
                self.preamp_gain_min = params['preamp_gain_min']
                self.preamp_gain_flatmax = params['preamp_gain_flatmax']

                # Booster
                self.booster_variety = params['booster_variety']
                self.booster_type_def = params['booster_type_def']
                self.booster_nf_model = params['booster_nf_model']
                self.booster_nf_fit_coeff = params['booster_nf_fit_coeff']
                self.booster_gain_min = params['booster_gain_min']
                self.booster_gain_flatmax = params['booster_gain_flatmax']

            # Others
            self.pmd = params['pmd']
            self.pdl = params['pdl']
            self.raman = params['raman']
            self.dgt = params['dgt']
            self.advance_configurations_from_json = params['advance_configurations_from_json']

            # Design
            self.allowed_for_design = params['allowed_for_design']

        except KeyError as e:
            raise ParametersError(f'Edfa configurations json must include {e}. Configuration: {params}')

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
