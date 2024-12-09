#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.parameters
====================

This module contains all parameters to configure standard network elements.
"""
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
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
    def __init__(self, flag=False, method='perturbative', order=2, result_spatial_resolution=10e3,
                 solver_spatial_resolution=10e3):
        """Simulation parameters used within the Raman Solver

        :params flag: boolean for enabling/disable the evaluation of the Raman power profile in frequency and position
        :params method: Raman solver method
        :params order: solution order for perturbative method
        :params result_spatial_resolution: spatial resolution of the evaluated Raman power profile
        :params solver_spatial_resolution: spatial step for the iterative solution of the first order ode
        """
        self.flag = flag
        self.method = method
        self.order = order
        self.result_spatial_resolution = result_spatial_resolution  # [m]
        self.solver_spatial_resolution = solver_spatial_resolution  # [m]

    def to_json(self):
        return {"flag": self.flag,
                "method": self.method,
                "order": self.order,
                "result_spatial_resolution": self.result_spatial_resolution,
                "solver_spatial_resolution": self.solver_spatial_resolution}


class NLIParams(Parameters):
    def __init__(self, method='gn_model_analytic', dispersion_tolerance=4, phase_shift_tolerance=0.1,
                 computed_channels=None, computed_number_of_channels=None):
        """Simulation parameters used within the Nli Solver

        :params method: formula for NLI calculation
        :params dispersion_tolerance: tuning parameter for ggn model solution
        :params phase_shift_tolerance: tuning parameter for ggn model solution
        :params computed_channels: the NLI is evaluated for these channels and extrapolated for the others
        :params computed_number_of_channels: the NLI is evaluated for this number of channels equally distributed
        in the spectrum and extrapolated for the others
        """
        self.method = method.lower()
        self.dispersion_tolerance = dispersion_tolerance
        self.phase_shift_tolerance = phase_shift_tolerance
        self.computed_channels = computed_channels
        self.computed_number_of_channels = computed_number_of_channels

    def to_json(self):
        return {"method": self.method,
                "dispersion_tolerance": self.dispersion_tolerance,
                "phase_shift_tolerance": self.phase_shift_tolerance,
                "computed_channels": self.computed_channels,
                "computed_number_of_channels": self.computed_number_of_channels}


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
            self.roadm_path_impairments = self.get_roadm_path_impairments(kwargs['roadm-path-impairments'])
        except KeyError as e:
            raise ParametersError(f'ROADM configurations must include {e}. Configuration: {kwargs}')
        self.per_degree_impairments = kwargs.get('per_degree_impairments', [])
        self.design_bands = kwargs.get('design_bands', [])
        self.per_degree_design_bands = kwargs.get('per_degree_design_bands', {})

    def get_roadm_path_impairments(self, path_impairments_list):
        """Get the ROADM list of profiles for impairments definition

        transform the ietf model into gnpy internal model: add a path-type in the attributes
        """
        if not path_impairments_list:
            return {}
        authorized_path_types = {
            'roadm-express-path': 'express',
            'roadm-add-path': 'add',
            'roadm-drop-path': 'drop',
        }
        roadm_path_impairments = {}
        for path_impairment in path_impairments_list:
            index = path_impairment['roadm-path-impairments-id']
            path_type = next(key for key in path_impairment if key in authorized_path_types.keys())
            impairment_dict = {'path-type': authorized_path_types[path_type], 'impairment': path_impairment[path_type]}
            roadm_path_impairments[index] = RoadmImpairment(impairment_dict)
        return roadm_path_impairments


class RoadmPath:
    def __init__(self, from_degree, to_degree, path_type, impairment_id=None, impairment=None):
        """Records roadm internal paths, types and impairment

        path_type must be in "express", "add", "drop"
        impairment_id must be one of the id detailed in equipement
        """
        self.from_degree = from_degree
        self.to_degree = to_degree
        self.path_type = path_type
        self.impairment_id = impairment_id
        self.impairment = impairment


class RoadmImpairment:
    """Generic definition of impairments for express, add and drop"""
    default_values = {
        'roadm-pmd': None,
        'roadm-cd': None,
        'roadm-pdl': None,
        'roadm-inband-crosstalk': None,
        'roadm-maxloss': 0,
        'roadm-osnr': None,
        'roadm-pmax': None,
        'roadm-noise-figure': None,
        'minloss': None,
        'typloss': None,
        'pmin': None,
        'ptyp': None
    }

    def __init__(self, params):
        self.path_type = params.get('path-type')
        self.impairments = params['impairment']


class FusedParams(Parameters):
    def __init__(self, **kwargs):
        self.loss = kwargs['loss'] if 'loss' in kwargs else 1


DEFAULT_RAMAN_COEFFICIENT = {
    # SSMF Raman coefficient profile in terms of mode intensity (g0 * A_ff_overlap)
    'gamma_raman': array(
        [0.0, 8.524419934705497e-16, 2.643567866245371e-15, 4.410548410941305e-15, 6.153422961291078e-15,
         7.484924703044943e-15, 8.452060808349209e-15, 9.101549322698156e-15, 9.57837595158966e-15,
         1.0008642675474562e-14, 1.0865773569905647e-14, 1.1300776305865833e-14, 1.2143238647099625e-14,
         1.3231065750676068e-14, 1.4624900971525384e-14, 1.6013330554840492e-14, 1.7458119359310242e-14,
         1.9320241330434762e-14, 2.1720395392873534e-14, 2.4137337406734775e-14, 2.628163218460466e-14,
         2.8041019963285974e-14, 2.9723155447089933e-14, 3.129353531005888e-14, 3.251796163324624e-14,
         3.3198839487612773e-14, 3.329527690685666e-14, 3.313155691238456e-14, 3.289013852154548e-14,
         3.2458917188506916e-14, 3.060684277937575e-14, 3.2660349473783173e-14, 2.957419109657689e-14,
         2.518894321396672e-14, 1.734560485857344e-14, 9.902860761605233e-15, 7.219176385099358e-15,
         6.079565990401311e-15, 5.828373065963427e-15, 7.20580801091692e-15, 7.561924351387493e-15,
         7.621152352332206e-15, 6.8859886780643254e-15, 5.629181047471162e-15, 3.679727598966185e-15,
         2.7555869742500355e-15, 2.4810133942597675e-15, 2.2160080532403624e-15, 2.1440626024765557e-15,
         2.33873070799544e-15, 2.557317929858713e-15, 3.039839048226572e-15, 4.8337165515610065e-15,
         5.4647431818257436e-15, 5.229187813711269e-15, 4.510768525811313e-15, 3.3213473130607794e-15,
         2.2602577027996455e-15, 1.969576495866441e-15, 1.5179853954188527e-15, 1.2953988551200156e-15,
         1.1304672156251838e-15, 9.10004390675213e-16, 8.432919922183503e-16, 7.849224069008326e-16,
         7.827568196032024e-16, 9.000514440646232e-16, 1.3025926460013665e-15, 1.5444108938497558e-15,
         1.8795594063060786e-15, 1.7796130169921014e-15, 1.5938159865046653e-15, 1.1585522355108287e-15,
         8.507044444633358e-16, 7.625404663756823e-16, 8.14510750925789e-16, 9.047944693473188e-16,
         9.636431901702084e-16, 9.298633899602105e-16, 8.349739503637023e-16, 7.482901278066085e-16,
         6.240794767134268e-16, 5.00652535687506e-16, 3.553373263685851e-16, 2.0344217706119682e-16,
         1.4267522642294203e-16, 8.980016576743517e-17, 2.9829068181832594e-17, 1.4861959129014824e-17,
         7.404482113326137e-18]
    ),  # m/W
    # SSMF Raman coefficient profile
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
    ),  # [1 / (W m)]

    # Note the non-uniform spacing of this range; this is required for properly capturing the Raman peak shape.
    'frequency_offset': array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5,
        12., 12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75,
        19., 19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5,
        29., 29.5, 30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5,
        39., 39.5, 40., 40.5, 41., 41.5, 42.]) * 1e12,  # [Hz]

    # Raman profile reference frequency
    'reference_frequency': 206.184634112792e12,  # [Hz] (1454 nm)

    # Raman profile reference effective area
    'reference_effective_area': 75.74659443542413e-12  # [m^2] (@1454 nm)
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
            raman_coefficient = kwargs.get('raman_coefficient')
            if raman_coefficient is None:
                self._raman_reference_frequency = DEFAULT_RAMAN_COEFFICIENT['reference_frequency']
                frequency_offset = asarray(DEFAULT_RAMAN_COEFFICIENT['frequency_offset'])
                gamma_raman = asarray(DEFAULT_RAMAN_COEFFICIENT['gamma_raman'])
                stokes_wave = self._raman_reference_frequency - frequency_offset
                normalized_gamma_raman = gamma_raman / self._raman_reference_frequency  # 1 / m / W / Hz
                self._g0 = gamma_raman / self.effective_area_overlap(stokes_wave, self._raman_reference_frequency)
            else:
                self._raman_reference_frequency = raman_coefficient['reference_frequency']
                frequency_offset = asarray(raman_coefficient['frequency_offset'])
                stokes_wave = self._raman_reference_frequency - frequency_offset
                self._g0 = asarray(raman_coefficient['g0'])
                gamma_raman = self._g0 * self.effective_area_overlap(stokes_wave, self._raman_reference_frequency)
                normalized_gamma_raman = gamma_raman / self._raman_reference_frequency  # 1 / m / W / Hz

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
    default_values = {
        'f_min': None,
        'f_max': None,
        'multi_band': None,
        'bands': None,
        'type_variety': '',
        'type_def': '',
        'gain_flatmax': None,
        'gain_min': None,
        'p_max': None,
        'nf_model': None,
        'dual_stage_model': None,
        'preamp_variety': None,
        'booster_variety': None,
        'nf_min': None,
        'nf_max': None,
        'nf_coef': None,
        'nf0': None,
        'nf_fit_coeff': None,
        'nf_ripple': 0,
        'dgt': None,
        'gain_ripple': 0,
        'tilt_ripple': 0,
        'f_ripple_ref': None,
        'out_voa_auto': False,
        'allowed_for_design': False,
        'raman': False,
        'pmd': 0,
        'pdl': 0,
        'advance_configurations_from_json': None
    }

    def __init__(self, **params):
        try:
            self.type_variety = params['type_variety']
            self.type_def = params['type_def']

            # Bandwidth
            self.f_min = params['f_min']
            self.f_max = params['f_max']
            self.bandwidth = self.f_max - self.f_min if self.f_max and self.f_min else None
            self.f_cent = (self.f_max + self.f_min) / 2 if self.f_max and self.f_min else None
            self.f_ripple_ref = params['f_ripple_ref']
            self.bands = [{'f_min': params['f_min'],
                           'f_max': params['f_max']}]

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
                    raise ParametersError(
                        "The noise figure ripple and the gain ripple must have the same size. %s, %s",
                        self.nf_ripple.size, self.gain_ripple.size)

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
            setattr(self, k, v)


class EdfaOperational:
    default_values = {
        'gain_target': None,
        'delta_p': None,
        'out_voa': None,
        'tilt_target': None
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


DEFAULT_EDFA_CONFIG = {
    "nf_ripple": [
        0.0
    ],
    "gain_ripple": [
        0.0
    ],
    "f_min": 191.275e12,
    "f_max": 196.125e12,
    "dgt": [
        1.0, 1.017807767853702, 1.0356155337864215, 1.0534217504465226, 1.0712204022764056, 1.0895983485572227,
        1.108555289615659, 1.1280891949729075, 1.1476135933863398, 1.1672278304018044, 1.1869318618366975,
        1.2067249615595257, 1.2264996957264114, 1.2428104897182262, 1.2556591482982988, 1.2650555289898042,
        1.2744470198196236, 1.2838336236692311, 1.2932153453410835, 1.3040618749785347, 1.316383926863083,
        1.3301807335621048, 1.3439818461440451, 1.3598972673004606, 1.3779439775587023, 1.3981208704326855,
        1.418273806730323, 1.4340878115214444, 1.445565137158368, 1.45273959485914, 1.4599103316162523,
        1.4670307626366115, 1.474100442252211, 1.48111939735681, 1.488134243479226, 1.495145456062699,
        1.502153039909686, 1.5097346239790443, 1.5178910621476225, 1.5266220576235803, 1.5353620432989845,
        1.545374152761467, 1.5566577309558969, 1.569199764184379, 1.5817353179379183, 1.5986915141218316,
        1.6201194134191075, 1.6460167077689267, 1.6719047669939942, 1.6918150918099673, 1.7057507692361864,
        1.7137640932265894, 1.7217732861435076, 1.7297783508684146, 1.737780757913635, 1.7459181197626403,
        1.7541903672600494, 1.7625959636196327, 1.7709972329654864, 1.7793941781790852, 1.7877868031023945,
        1.7961751115773796, 1.8045606557581335, 1.8139629377087627, 1.824381436842932, 1.835814081380705,
        1.847275503201129, 1.862235672444246, 1.8806927939516411, 1.9026104247588487, 1.9245345552113182,
        1.9482128147680253, 1.9736443063300082, 2.0008103857988204, 2.0279625371819305, 2.055100772005235,
        2.082225099873648, 2.1183028432496016, 2.16337565384239, 2.2174389328192197, 2.271520771371253,
        2.322373696229342, 2.3699990328716107, 2.414398437185221, 2.4587748041127506, 2.499446286796604,
        2.5364027376452056, 2.5696460593920065, 2.602860350286428, 2.630396440815385, 2.6521732021128046,
        2.6681935771243177, 2.6841217449620203, 2.6947834587664494, 2.705443819238505, 2.714526681131686
    ]
}


class MultiBandParams:
    default_values = {
        'bands': [],
        'type_variety': '',
        'type_def': None,
        'allowed_for_design': False
    }

    def __init__(self, **params):
        try:
            self.update_attr(params)
        except KeyError as e:
            raise ParametersError(f'Multiband configurations json must include {e}. Configuration: {params}')

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            # use deepcopy to avoid sharing same object amongst all instance when v is a list or a dict!
            if isinstance(v, (list, dict)):
                setattr(self, k, clean_kwargs.get(k, deepcopy(v)))
            else:
                setattr(self, k, clean_kwargs.get(k, v))


class TransceiverParams:
    def __init__(self, **params):
        self.design_bands = params.get('design_bands', [])
        self.per_degree_design_bands = params.get('per_degree_design_bands', {})


@dataclass
class FrequencyBand:
    """Frequency band
    """
    f_min: float
    f_max: float


DEFAULT_BANDS_DEFINITION = {
    "LBAND": FrequencyBand(f_min=187e12, f_max=189e12),
    "CBAND": FrequencyBand(f_min=191.3e12, f_max=196.0e12)
}
# use this definition to index amplifiers'element of a multiband amplifier.
# this is not the design band


def find_band_name(band: FrequencyBand) -> str:
    """return the default band name (CBAND, LBAND, ...) that corresponds to the band frequency range
    Use the band center frequency: if center frequency is inside the band then returns CBAND.
    This is to flexibly encompass all kind of bands definitions.
    returns the first matching band name.
    """
    for band_name, frequency_range in DEFAULT_BANDS_DEFINITION.items():
        center_frequency = (band.f_min + band.f_max) / 2
        if center_frequency >= frequency_range.f_min and center_frequency <= frequency_range.f_max:
            return band_name
    return 'unknown_band'
