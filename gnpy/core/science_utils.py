#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.science_utils
=======================

Solver definitions to calculate the Raman effect and the nonlinear interference noise

The solvers take as input instances of the spectral information, the fiber and the simulation parameters
"""

from numpy import interp, pi, zeros, shape, where, cos, array, append, ones, exp, arange, sqrt, trapz, arcsinh, \
    clip, abs, sum, concatenate, flip, outer, inner, transpose, max, format_float_scientific, diag, sort, unique, \
    argsort, cumprod
from logging import getLogger
from scipy.constants import k, h
from scipy.interpolate import interp1d
from math import isclose

from gnpy.core.utils import db2lin, lin2db
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.core.parameters import SimParams
from gnpy.core.info import SpectralInformation

logger = getLogger(__name__)
sim_params = SimParams()

def raised_cosine_comb(f, *carriers):
    """Returns an array storing the PSD of a WDM comb of raised cosine shaped
    channels at the input frequencies defined in array f

    :param f: numpy array of frequencies in Hz
    :param carriers: namedtuple describing the WDM comb
    :return: PSD of the WDM comb evaluated over f
    """
    psd = zeros(shape(f))
    for carrier in carriers:
        f_nch = carrier.frequency
        g_ch = carrier.power.signal / carrier.baud_rate
        ts = 1 / carrier.baud_rate
        pass_band = (1 - carrier.roll_off) / (2 / carrier.baud_rate)
        stop_band = (1 + carrier.roll_off) / (2 / carrier.baud_rate)
        ff = abs(f - f_nch)
        tf = ff - pass_band
        if carrier.roll_off == 0:
            psd = where(tf <= 0, g_ch, 0.) + psd
        else:
            psd = g_ch * (where(tf <= 0, 1., 0.) + 1 / 2 * (1 + cos(pi * ts / carrier.roll_off * tf)) *
                          where(tf > 0, 1., 0.) * where(abs(ff) <= stop_band, 1., 0.)) + psd
    return psd


class StimulatedRamanScattering:
    def __init__(self, power_profile, loss_profile, frequency, z):
        """
        :params power_profile: power profile matrix along frequency and z [W]
        :params loss_profile: power profile matrix along frequency and z [linear units]
        :params frequency: channels frequencies array [Hz]
        :params z: positions array [m]
        """
        self.power_profile = power_profile
        self.loss_profile = loss_profile
        # Field loss profile matrix along frequency and z
        self.rho = sqrt(loss_profile)
        self.frequency = frequency
        self.z = z


class RamanSolver:
    """This class contains the methods to calculate the Raman scattering effect."""

    @staticmethod
    def _create_lumped_losses(z, lumped_losses, z_lumped_losses):
        lumped_losses = concatenate((lumped_losses, ones(z.size)))
        z, unique_indices = unique(concatenate((z_lumped_losses, z)), return_index=True)
        order = argsort(z)
        lumped_losses = (lumped_losses[unique_indices])[order]
        z = z[order]
        return z, lumped_losses

    @staticmethod
    def calculate_attenuation_profile(spectral_info: SpectralInformation, fiber):
        """Evaluates the attenuation profile along the z axis for all the frequency propagating in the
        fiber without considering the stimulated Raman scattering.
        """
        # z array definition
        z = array([0, fiber.params.length])

        # Lumped losses array definition
        z, lumped_losses = RamanSolver._create_lumped_losses(z, fiber.lumped_losses, fiber.z_lumped_losses)

        lumped_loss_acc = cumprod(lumped_losses)
        frequency = spectral_info.frequency
        alpha = fiber.alpha(frequency)
        loss_profile = exp(- outer(alpha, z)) * lumped_loss_acc
        power_profile = outer(spectral_info.signal, ones(z.size)) * loss_profile
        return StimulatedRamanScattering(power_profile, loss_profile, spectral_info.frequency, z)

    @staticmethod
    def calculate_stimulated_raman_scattering(spectral_info: SpectralInformation, fiber):
        """Evaluates the Raman profile along the z axis for all the frequency propagated in the fiber
        including the Raman pumps co- and counter-propagating
        """
        logger.debug('Start computing fiber Stimulated Raman Scattering')

        if sim_params.raman_params.flag:
            # Raman parameters
            z_resolution = sim_params.raman_params.result_spatial_resolution
            z_step = sim_params.raman_params.solver_spatial_resolution
            z = append(arange(0, fiber.params.length, z_step), fiber.params.length)
            z_final = append(arange(0, fiber.params.length, z_resolution), fiber.params.length)
            z_final = sort(unique(concatenate((fiber.z_lumped_losses, z_final))))

            # Lumped losses array definition
            z, lumped_losses = RamanSolver._create_lumped_losses(z, fiber.lumped_losses, fiber.z_lumped_losses)

            if hasattr(fiber, 'raman_pumps'):
                # TODO: verify co-propagating pumps computation and in general unsorted frequency
                # Co-propagating spectrum definition
                co_raman_pump_power = array([pump.power for pump in fiber.raman_pumps
                                             if pump.propagation_direction == 'coprop'])
                co_raman_pump_frequency = array([pump.frequency for pump in fiber.raman_pumps
                                                 if pump.propagation_direction == 'coprop'])

                co_power = concatenate((spectral_info.signal, co_raman_pump_power))
                co_frequency = concatenate((spectral_info.frequency, co_raman_pump_frequency))

                # Counter-propagating spectrum definition
                cnt_power = array([pump.power for pump in fiber.raman_pumps
                                   if pump.propagation_direction == 'counterprop'])
                cnt_frequency = array([pump.frequency for pump in fiber.raman_pumps
                                       if pump.propagation_direction == 'counterprop'])
                # Co-propagating profile initialization
                co_power_profile = zeros([co_frequency.size, z.size])
                if co_frequency.size:
                    co_cr = fiber.cr(co_frequency)
                    co_alpha = fiber.alpha(co_frequency)
                    co_power_profile = \
                        RamanSolver.first_order_derivative_solution(co_power, co_alpha, co_cr, z, lumped_losses)
                # Counter-propagating profile initialization
                cnt_power_profile = zeros([cnt_frequency.size, z.size])
                if cnt_frequency.size:
                    cnt_cr = fiber.cr(cnt_frequency)
                    cnt_alpha = fiber.alpha(cnt_frequency)
                    cnt_power_profile = \
                        flip(RamanSolver.first_order_derivative_solution(cnt_power, cnt_alpha, cnt_cr,
                                                                         z[-1] - flip(z), flip(lumped_losses)), axis=1)
                # Co-propagating and Counter-propagating Profile Computation
                if co_frequency.size and cnt_frequency.size:
                    co_power_profile, cnt_power_profile = \
                        RamanSolver.iterative_algorithm(co_power_profile, cnt_power_profile,
                                                        co_frequency, cnt_frequency, z, fiber, lumped_losses)
                # Complete Power Profile
                power_profile = concatenate((co_power_profile, cnt_power_profile), axis=0)
                # Complete Loss Profile
                co_loss_profile = co_power_profile / outer(co_power, ones(z.size))
                cnt_loss_profile = cnt_power_profile / outer(cnt_power, ones(z.size))
                loss_profile = concatenate((co_loss_profile, cnt_loss_profile), axis=0)
                # Complete frequency
                frequency = concatenate((co_frequency, cnt_frequency))
            else:
                # Without Raman pumps
                alpha = fiber.alpha(spectral_info.frequency)
                cr = fiber.cr(spectral_info.frequency)
                # Power profile
                power_profile = \
                    RamanSolver.first_order_derivative_solution(spectral_info.signal, alpha, cr, z, lumped_losses)
                # Loss profile
                loss_profile = power_profile / outer(spectral_info.signal, ones(z.size))
                frequency = spectral_info.frequency
            power_profile = interp1d(z, power_profile, axis=1)(z_final)
            loss_profile = interp1d(z, loss_profile, axis=1)(z_final)
            stimulated_raman_scattering = StimulatedRamanScattering(power_profile, loss_profile, frequency, z_final)
        else:
            stimulated_raman_scattering = \
                RamanSolver.calculate_attenuation_profile(spectral_info, fiber)
        return stimulated_raman_scattering

    @staticmethod
    def calculate_spontaneous_raman_scattering(spectral_info: SpectralInformation, srs: StimulatedRamanScattering,
                                               fiber):
        """Evaluates the Raman profile along the z axis for all the frequency propagated in the fiber
        including the Raman pumps co- and counter-propagating.
        """
        logger.debug('Start computing fiber Spontaneous Raman Scattering')
        z = srs.z
        baud_rate = spectral_info.baud_rate
        frequency = spectral_info.frequency
        channels_loss = srs.loss_profile[:spectral_info.number_of_channels, :]

        # calculate ase power
        ase = zeros(spectral_info.number_of_channels)
        for i, pump in enumerate(fiber.raman_pumps):
            pump_power = srs.power_profile[spectral_info.number_of_channels + i, :]
            df = pump.frequency - frequency
            eta = - 1 / (1 - exp(h * df / (k * fiber.temperature)))
            cr = fiber._cr_function(df)
            integral = trapz(pump_power / channels_loss, z, axis=1)
            ase += 2 * h * baud_rate * frequency * (1 + eta) * cr * (df > 0) * integral  # 2 factor for double pol
        return ase

    @staticmethod
    def first_order_derivative_solution(power_in, alpha, cr, z, lumped_losses):
        """Solves the Raman first order derivative equation

        :param power_in: launch power array
        :param alpha: loss coefficient array
        :param cr: Raman efficiency coefficients matrix
        :param z: z position array
        :param lumped_losses: concentrated losses array along the fiber span
        :return: power profile matrix
        """
        dz = z[1:] - z[:-1]
        power = outer(power_in, ones(z.size))
        for i in range(1, z.size):
            power[:, i] = \
                power[:, i - 1] * (1 + (- alpha + sum(cr * power[:, i - 1], 1)) * dz[i - 1]) * lumped_losses[i - 1]
        return power

    @staticmethod
    def iterative_algorithm(co_initial_guess_power, cnt_initial_guess_power, co_frequency, cnt_frequency, z, fiber,
                            lumped_losses):
        """Solves the Raman first order derivative equation in case of both co- and counter-propagating
        frequencies

        :param co_initial_guess_power: co-propagationg Raman first order derivative equation solution
        :param cnt_initial_guess_power: counter-propagationg Raman first order derivative equation solution
        :param co_frequency: co-propagationg frequencies
        :param cnt_frequency: counter-propagationg frequencies
        :param z: z position array
        :param fiber: instance of gnpy.core.elements.Fiber or gnpy.core.elements.RamanFiber
        :param lumped_losses: concentrated losses array along the fiber span
        :return: co- and counter-propagatng power profile matrix
        """
        logger.debug('  Start iterative algorithm')
        residue = 1
        residue_tol = 1e-6
        accuracy = 1
        accuracy_tol = 1e-3
        iteration = 0
        num_max_iter = 1000
        prev_power = concatenate((co_initial_guess_power, cnt_initial_guess_power))
        frequency = concatenate((co_frequency, cnt_frequency))
        dz = z[1:] - z[:-1]
        cr = fiber.cr(frequency)
        alpha = fiber.alpha(frequency)
        next_power = array(prev_power)
        while residue > residue_tol and accuracy > accuracy_tol and iteration < num_max_iter:
            iteration += 1
            for i in range(1, z.size):
                dpdz = - alpha + sum(cr * next_power[:, i - 1], 1)
                next_power[:co_frequency.size, i] = \
                    next_power[:co_frequency.size, i - 1] * (1 + dpdz[:co_frequency.size] * dz[i - 1]) * \
                    lumped_losses[i - 1]
            for i in range(1, z.size):
                dpdz = - alpha + sum(cr * next_power[:, -i], 1)
                next_power[co_frequency.size:, -i - 1] = \
                    next_power[co_frequency.size:, -i] * (1 + dpdz[co_frequency.size:] * dz[-i]) * \
                    lumped_losses[-i]

            dpdz_num = (next_power[:co_frequency.size, 1:] - next_power[:co_frequency.size, :-1]) / dz
            dpdz_exp = next_power[:co_frequency.size, :-1] * \
                (- outer(alpha, ones(z.size)) + inner(cr, transpose(next_power)))[:co_frequency.size, :-1] * \
                lumped_losses[:-1]

            residue = max(abs((next_power - prev_power) / next_power))
            accuracy = max(abs((dpdz_exp - dpdz_num) / dpdz_exp))
            prev_power = array(next_power)
            logger.debug(f'     Iteration: {iteration}  Accuracy: {format_float_scientific(accuracy, precision=3)}')
        return next_power[:co_frequency.size, :], next_power[co_frequency.size:, :]


class NliSolver:
    """This class implements the NLI models.
    Model and method can be specified in `sim_params.nli_params.method`.
    List of implemented methods:
    'gn_model_analytic': eq. 120 from arXiv:1209.0394
    'ggn_spectrally_separated': eq. 21 from arXiv: 1710.02225 spectrally separated
    """

    @staticmethod
    def effective_length(alpha, length):
        """The effective length identify the region in which the NLI has a significant contribution to
        the signal degradation.
        """
        return (1 - exp(- alpha * length)) / alpha

    @staticmethod
    def compute_nli(spectral_info: SpectralInformation, srs: StimulatedRamanScattering, fiber):
        """Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier`
        at the end of the fiber span.
        """
        logger.debug('Start computing fiber NLI noise')
        # Physical fiber parameters
        alpha = fiber.alpha(spectral_info.frequency)
        beta2 = fiber.params.beta2
        beta3 = fiber.params.beta3
        f_ref_beta = fiber.params.ref_frequency
        gamma = fiber.params.gamma
        length = fiber.params.length

        if 'gn_model_analytic' == sim_params.nli_params.method:
            nli = NliSolver._gn_analytic(spectral_info, alpha, beta2, gamma, length)
        elif 'ggn_spectrally_separated' in sim_params.nli_params.method:
            nli = NliSolver._ggn_spectrally_separated(spectral_info, srs, alpha, beta2, beta3, f_ref_beta, gamma)
        else:
            raise ValueError(f'Method {sim_params.nli_params.method} not implemented.')
        return nli

    # Methods for computing GN-model
    @staticmethod
    def _gn_analytic(spectral_info: SpectralInformation, alpha, beta2, gamma, length):
        """Computes the nonlinear interference power evaluated at the fiber input.
        The method uses eq. 120 from arXiv:1209.0394
        """
        spm_weight = (16.0 / 27.0) * gamma ** 2
        xpm_weight = 2 * (16.0 / 27.0) * gamma ** 2

        nch = spectral_info.number_of_channels
        identity = diag(ones(nch))
        weight = spm_weight * identity + xpm_weight * (ones([nch, nch]) - identity)

        effective_length = NliSolver.effective_length(alpha, length)
        asymptotic_length = 1 / alpha

        df = spectral_info.df
        baud_rate = spectral_info.baud_rate

        psd = spectral_info.signal / baud_rate
        ggg = outer(psd, psd**2)

        psi = NliSolver._psi(df, baud_rate, beta2, effective_length, asymptotic_length)
        g_nli = sum(weight * ggg * psi, 1)
        nli = spectral_info.baud_rate * g_nli  # Local white noise
        return nli

    @staticmethod
    def _psi(df, baud_rate, beta2, effective_length, asymptotic_length):
        """Calculates eq. 123 from `arXiv:1209.0394 <https://arxiv.org/abs/1209.0394>`__"""
        cut_baud_rate = outer(baud_rate, ones(baud_rate.size))
        pump_baud_rate = baud_rate
        right_extreme = df + pump_baud_rate / 2
        left_extreme = df - pump_baud_rate / 2
        psi = (arcsinh(pi ** 2 * asymptotic_length * abs(beta2) * cut_baud_rate * right_extreme) -
               arcsinh(pi ** 2 * asymptotic_length * abs(beta2) * cut_baud_rate * left_extreme)) / 2
        psi *= effective_length ** 2 / (2 * pi * abs(beta2) * asymptotic_length)
        return psi

    # Methods for computing the GGN-model
    @staticmethod
    def _ggn_spectrally_separated(spectral_info: SpectralInformation, srs: StimulatedRamanScattering,
                                  alpha, beta2, beta3, f_ref_beta, gamma):
        """Computes the nonlinear interference power evaluated at the fiber input.
        The method uses eq. 21 from arXiv: 1710.02225
        """
        dispersion_tolerance = sim_params.nli_params.dispersion_tolerance
        phase_shift_tolerance = sim_params.nli_params.phase_shift_tolerance
        slot_width = max(spectral_info.slot_width)
        delta_z = sim_params.raman_params.result_spatial_resolution
        spm_weight = (16.0 / 27.0) * gamma ** 2
        xpm_weight = 2 * (16.0 / 27.0) * gamma ** 2
        cuts = [carrier for carrier in spectral_info.carriers if carrier.channel_number
                in sim_params.nli_params.computed_channels] if sim_params.nli_params.computed_channels \
            else spectral_info.carriers

        g_nli = array([])
        f_nli = array([])
        for cut_carrier in cuts:
            logger.debug(f'Start computing fiber NLI noise of cut: {cut_carrier}')
            f_eval = cut_carrier.frequency
            g_nli_computed = 0
            g_cut = (cut_carrier.power.signal / cut_carrier.baud_rate)
            for j, pump_carrier in enumerate(spectral_info.carriers):
                dn = abs(pump_carrier.channel_number - cut_carrier.channel_number)
                delta_f = abs(cut_carrier.frequency - pump_carrier.frequency)
                k_tol = dispersion_tolerance * abs(alpha[j])
                phi_tol = phase_shift_tolerance / delta_z
                f_cut_resolution = min(k_tol, phi_tol) / abs(beta2) / (4 * pi ** 2 * (1 + dn) * slot_width)
                f_pump_resolution = min(k_tol, phi_tol) / abs(beta2) / (4 * pi ** 2 * slot_width)
                if dn == 0:  # SPM
                    ggg = g_cut ** 3
                    g_nli_computed += \
                        spm_weight * ggg * NliSolver._generalized_psi(f_eval, cut_carrier, pump_carrier,
                                                                      f_cut_resolution, f_pump_resolution,
                                                                      srs, alpha[j], beta2, beta3, f_ref_beta)
                else:  # XPM
                    g_pump = (pump_carrier.power.signal / pump_carrier.baud_rate)
                    ggg = g_cut * g_pump ** 2
                    frequency_offset_threshold = NliSolver._frequency_offset_threshold(beta2, pump_carrier.baud_rate)
                    if abs(delta_f) <= frequency_offset_threshold:
                        g_nli_computed += \
                            xpm_weight * ggg * NliSolver._generalized_psi(f_eval, cut_carrier, pump_carrier,
                                                                          f_cut_resolution, f_pump_resolution,
                                                                          srs, alpha[j], beta2, beta3, f_ref_beta)
                    else:
                        g_nli_computed += \
                            xpm_weight * ggg * NliSolver._fast_generalized_psi(f_eval, cut_carrier, pump_carrier,
                                                                               f_cut_resolution, srs, alpha[j], beta2,
                                                                               beta3, f_ref_beta)
            f_nli = append(f_nli, cut_carrier.frequency)
            g_nli = append(g_nli, g_nli_computed)
        g_nli = interp(spectral_info.frequency, f_nli, g_nli)
        nli = spectral_info.baud_rate * g_nli  # Local white noise
        return nli

    @staticmethod
    def _fast_generalized_psi(f_eval, cut_carrier, pump_carrier, f_cut_resolution, srs, alpha, beta2, beta3,
                              f_ref_beta):
        """Computes the generalized psi function similarly to the one used in the GN model."""
        z = srs.z
        rho_norm = srs.rho * exp(outer(alpha/2, z))
        rho_pump = interp1d(srs.frequency, rho_norm, axis=0)(pump_carrier.frequency)

        f1_array = array([pump_carrier.frequency - (pump_carrier.baud_rate * (1 + pump_carrier.roll_off) / 2),
                          pump_carrier.frequency + (pump_carrier.baud_rate * (1 + pump_carrier.roll_off) / 2)])
        f2_array = arange(cut_carrier.frequency,
                          cut_carrier.frequency + (cut_carrier.baud_rate * (1 + cut_carrier.roll_off) / 2),
                          f_cut_resolution)  # Only positive f2 is used since integrand_f2 is symmetric

        integrand_f1 = zeros(len(f1_array))
        for f1_index, f1 in enumerate(f1_array):
            delta_beta = 4 * pi ** 2 * (f1 - f_eval) * (f2_array - f_eval) * \
                (beta2 + pi * beta3 * (f1 + f2_array - 2 * f_ref_beta))
            integrand_f2 = NliSolver._generalized_rho_nli(delta_beta, rho_pump, z, alpha)
            integrand_f1[f1_index] = 2 * trapz(integrand_f2, f2_array)  # 2x since integrand_f2 is symmetric in f2
        generalized_psi = 0.5 * sum(integrand_f1) * pump_carrier.baud_rate
        return generalized_psi

    @staticmethod
    def _generalized_psi(f_eval, cut_carrier, pump_carrier, f_cut_resolution, f_pump_resolution, srs, alpha, beta2,
                         beta3, f_ref_beta):
        """Computes the generalized psi function similarly to the one used in the GN model."""
        z = srs.z
        rho_norm = srs.rho * exp(outer(alpha / 2, z))
        rho_pump = interp1d(srs.frequency, rho_norm, axis=0)(pump_carrier.frequency)

        f1_array = arange(pump_carrier.frequency - (pump_carrier.baud_rate * (1 + pump_carrier.roll_off) / 2),
                          pump_carrier.frequency + (pump_carrier.baud_rate * (1 + pump_carrier.roll_off) / 2),
                          f_pump_resolution)
        f2_array = arange(cut_carrier.frequency - (cut_carrier.baud_rate * (1 + cut_carrier.roll_off) / 2),
                          cut_carrier.frequency + (cut_carrier.baud_rate * (1 + cut_carrier.roll_off) / 2),
                          f_cut_resolution)
        psd1 = raised_cosine_comb(f1_array, pump_carrier) * (pump_carrier.baud_rate / pump_carrier.power.signal)

        integrand_f1 = zeros(len(f1_array))
        for f1_index, (f1, psd1_sample) in enumerate(zip(f1_array, psd1)):
            f3_array = f1 + f2_array - f_eval
            psd2 = raised_cosine_comb(f2_array, cut_carrier) * (cut_carrier.baud_rate / cut_carrier.power.signal)
            psd3 = raised_cosine_comb(f3_array, pump_carrier) * (pump_carrier.baud_rate / pump_carrier.power.signal)
            ggg = psd1_sample * psd2 * psd3
            delta_beta = 4 * pi**2 * (f1 - f_eval) * (f2_array - f_eval) * \
                (beta2 + pi * beta3 * (f1 + f2_array - 2 * f_ref_beta))
            integrand_f2 = ggg * NliSolver._generalized_rho_nli(delta_beta, rho_pump, z, alpha)
            integrand_f1[f1_index] = trapz(integrand_f2, f2_array)
        generalized_psi = trapz(integrand_f1, f1_array)
        return generalized_psi

    @staticmethod
    def _generalized_rho_nli(delta_beta, rho_pump, z, alpha):
        w = 1j * delta_beta - alpha
        generalized_rho_nli = (rho_pump[-1]**2 * exp(w * z[-1]) - rho_pump[0]**2 * exp(w * z[0])) / w
        for z_ind in range(0, len(z) - 1):
            derivative_rho = (rho_pump[z_ind + 1]**2 - rho_pump[z_ind]**2) / (z[z_ind + 1] - z[z_ind])
            generalized_rho_nli -= derivative_rho * (exp(w * z[z_ind + 1]) - exp(w * z[z_ind])) / (w**2)
        generalized_rho_nli = abs(generalized_rho_nli)**2
        return generalized_rho_nli

    @staticmethod
    def _frequency_offset_threshold(beta2, symbol_rate):
        k_ref = 5
        beta2_ref = 21.3e-27
        delta_f_ref = 50e9
        rs_ref = 32e9
        beta2 = abs(beta2)
        freq_offset_th = ((k_ref * delta_f_ref) * rs_ref * beta2_ref) / (beta2 * symbol_rate)
        return freq_offset_th


def estimate_nf_model(type_variety, gain_min, gain_max, nf_min, nf_max):
    if nf_min < -10:
        raise EquipmentConfigError(f'Invalid nf_min value {nf_min!r} for amplifier {type_variety}')
    if nf_max < -10:
        raise EquipmentConfigError(f'Invalid nf_max value {nf_max!r} for amplifier {type_variety}')

    # NF estimation model based on nf_min and nf_max
    # delta_p:  max power dB difference between first and second stage coils
    # dB g1a:   first stage gain - internal VOA attenuation
    # nf1, nf2: first and second stage coils
    #           calculated by solving nf_{min,max} = nf1 + nf2 / g1a{min,max}
    delta_p = 5
    g1a_min = gain_min - (gain_max - gain_min) - delta_p
    g1a_max = gain_max - delta_p
    nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) /
                 (1 / db2lin(g1a_max) - 1 / db2lin(g1a_min)))
    nf1 = lin2db(db2lin(nf_min) - db2lin(nf2) / db2lin(g1a_max))

    if nf1 < 4:
        raise EquipmentConfigError(f'First coil value too low {nf1} for amplifier {type_variety}')

    # Check 1 dB < delta_p < 6 dB to ensure nf_min and nf_max values make sense.
    # There shouldn't be high nf differences between the two coils:
    #    nf2 should be nf1 + 0.3 < nf2 < nf1 + 2
    # If not, recompute and check delta_p
    if not nf1 + 0.3 < nf2 < nf1 + 2:
        nf2 = clip(nf2, nf1 + 0.3, nf1 + 2)
        g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1)))
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max - gain_min) - delta_p
        if not 1 < delta_p < 11:
            raise EquipmentConfigError(f'Computed \N{greek capital letter delta}P invalid \
                \n 1st coil vs 2nd coil calculated DeltaP {delta_p:.2f} for \
                \n amplifier {type_variety} is not valid: revise inputs \
                \n calculated 1st coil NF = {nf1:.2f}, 2nd coil NF = {nf2:.2f}')
    # Check calculated values for nf1 and nf2
    calc_nf_min = lin2db(db2lin(nf1) + db2lin(nf2) / db2lin(g1a_max))
    if not isclose(nf_min, calc_nf_min, abs_tol=0.01):
        raise EquipmentConfigError(f'nf_min does not match calc_nf_min, {nf_min} vs {calc_nf_min} for amp {type_variety}')
    calc_nf_max = lin2db(db2lin(nf1) + db2lin(nf2) / db2lin(g1a_min))
    if not isclose(nf_max, calc_nf_max, abs_tol=0.01):
        raise EquipmentConfigError(f'nf_max does not match calc_nf_max, {nf_max} vs {calc_nf_max} for amp {type_variety}')

    return nf1, nf2, delta_p
