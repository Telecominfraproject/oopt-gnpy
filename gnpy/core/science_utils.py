#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.science_utils
=======================

Solver definitions to calculate the Raman effect and the nonlinear interference noise

The solvers take as input instances of the spectral information, the fiber and the simulation parameters
"""

from numpy import interp, pi, zeros, shape, where, cos, reshape, array, append, ones, argsort, nan, exp, arange, sqrt, \
    empty, vstack, trapz, arcsinh, clip, abs, sum, outer, diag
from operator import attrgetter
from logging import getLogger
import scipy.constants as ph
from scipy.integrate import solve_bvp
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult
from math import isclose

from gnpy.core.utils import db2lin, lin2db
from gnpy.core.exceptions import EquipmentConfigError
from gnpy.core.parameters import SimParams
from gnpy.core.info import SpectralInformation

logger = getLogger(__name__)
sim_params = SimParams.get()


def raised_cosine_comb(f, *carriers):
    """ Returns an array storing the PSD of a WDM comb of raised cosine shaped
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


class SpontaneousRamanScattering:
    def __init__(self, frequency, z, power):
        self.frequency = frequency
        self.z = z
        self.power = power


class StimulatedRamanScattering:
    def __init__(self, frequency, z, rho, power):
        self.frequency = frequency
        self.z = z
        self.rho = rho
        self.power = power


class RamanSolver:
    def __init__(self, fiber=None):
        """ Initialize the Raman solver object.
        :param fiber: instance of elements.py/Fiber.
        :param carriers: tuple of carrier objects
        :param raman_pumps: tuple containing pumps characteristics
        """
        self._fiber = fiber
        self._carriers = None
        self._raman_pumps = None
        self._stimulated_raman_scattering = None
        self._spontaneous_raman_scattering = None

    @property
    def fiber(self):
        return self._fiber

    @property
    def carriers(self):
        return self._carriers

    @carriers.setter
    def carriers(self, carriers):
        self._carriers = carriers
        self._spontaneous_raman_scattering = None
        self._stimulated_raman_scattering = None

    @property
    def raman_pumps(self):
        return self._raman_pumps

    @raman_pumps.setter
    def raman_pumps(self, raman_pumps):
        self._raman_pumps = raman_pumps
        self._stimulated_raman_scattering = None

    @property
    def stimulated_raman_scattering(self):
        if self._stimulated_raman_scattering is None:
            self.calculate_stimulated_raman_scattering(self.carriers, self.raman_pumps)
        return self._stimulated_raman_scattering

    @property
    def spontaneous_raman_scattering(self):
        if self._spontaneous_raman_scattering is None:
            self.calculate_spontaneous_raman_scattering(self.carriers, self.raman_pumps)
        return self._spontaneous_raman_scattering

    def calculate_spontaneous_raman_scattering(self, carriers, raman_pumps):
        raman_efficiency = self.fiber.params.raman_efficiency
        temperature = self.fiber.operational['temperature']

        logger.debug('Start computing fiber Spontaneous Raman Scattering')
        power_spectrum, freq_array, prop_direct, bn_array = self._compute_power_spectrum(carriers, raman_pumps)

        alphap_fiber = self.fiber.alpha(freq_array)

        freq_diff = abs(freq_array - reshape(freq_array, (len(freq_array), 1)))
        interp_cr = interp1d(raman_efficiency['frequency_offset'], raman_efficiency['cr'])
        cr = interp_cr(freq_diff)

        # z propagation axis
        z_array = self.stimulated_raman_scattering.z
        ase_bc = zeros(freq_array.shape)

        # calculate ase power
        int_spontaneous_raman = self._int_spontaneous_raman(z_array, self._stimulated_raman_scattering.power,
                                                            alphap_fiber, freq_array, cr, freq_diff, ase_bc,
                                                            bn_array, temperature)

        spontaneous_raman_scattering = SpontaneousRamanScattering(freq_array, z_array, int_spontaneous_raman.x)
        logger.debug("Spontaneous Raman Scattering evaluated successfully")
        self._spontaneous_raman_scattering = spontaneous_raman_scattering

    @staticmethod
    def _compute_power_spectrum(carriers, raman_pumps=None):
        """
        Rearrangement of spectral and Raman pump information to make them compatible with Raman solver
        :param carriers: a tuple of namedtuples describing the transmitted channels
        :param raman_pumps: a namedtuple describing the Raman pumps
        :return:
        """

        # Signal power spectrum
        pow_array = array([])
        f_array = array([])
        noise_bandwidth_array = array([])
        for carrier in sorted(carriers, key=attrgetter('frequency')):
            f_array = append(f_array, carrier.frequency)
            pow_array = append(pow_array, carrier.power.signal)
            ref_bw = carrier.baud_rate
            noise_bandwidth_array = append(noise_bandwidth_array, ref_bw)

        propagation_direction = ones(len(f_array))

        # Raman pump power spectrum
        if raman_pumps:
            for pump in raman_pumps:
                pow_array = append(pow_array, pump.power)
                f_array = append(f_array, pump.frequency)
                direction = +1 if pump.propagation_direction == 'coprop' else -1
                propagation_direction = append(propagation_direction, direction)
                noise_bandwidth_array = append(noise_bandwidth_array, ref_bw)

        # Final sorting
        ind = argsort(f_array)
        f_array = f_array[ind]
        pow_array = pow_array[ind]
        propagation_direction = propagation_direction[ind]

        return pow_array, f_array, propagation_direction, noise_bandwidth_array

    def _int_spontaneous_raman(self, z_array, raman_matrix, alphap_fiber, freq_array,
                               cr_raman_matrix, freq_diff, ase_bc, bn_array, temperature):
        spontaneous_raman_scattering = OptimizeResult()

        dx = sim_params.raman_params.space_resolution
        h = ph.value('Planck constant')
        kb = ph.value('Boltzmann constant')

        power_ase = nan * ones(raman_matrix.shape)
        int_pump = cumtrapz(raman_matrix, z_array, dx=dx, axis=1, initial=0)

        for f_ind, f_ase in enumerate(freq_array):
            cr_raman = cr_raman_matrix[f_ind, :]
            vibrational_loss = f_ase / freq_array[:f_ind]
            eta = 1 / (exp((h * freq_diff[f_ind, f_ind + 1:]) / (kb * temperature)) - 1)

            int_fiber_loss = -alphap_fiber[f_ind] * z_array
            int_raman_loss = sum((cr_raman[:f_ind] * vibrational_loss * int_pump[:f_ind, :].transpose()).transpose(),
                                    axis=0)
            int_raman_gain = sum((cr_raman[f_ind + 1:] * int_pump[f_ind + 1:, :].transpose()).transpose(), axis=0)

            int_gain_loss = int_fiber_loss + int_raman_gain + int_raman_loss

            new_ase = sum((cr_raman[f_ind + 1:] * (1 + eta) * raman_matrix[f_ind + 1:, :].transpose()).transpose()
                             * h * f_ase * bn_array[f_ind], axis=0)

            bc_evolution = ase_bc[f_ind] * exp(int_gain_loss)
            ase_evolution = exp(int_gain_loss) * cumtrapz(new_ase * exp(-int_gain_loss), z_array, dx=dx, initial=0)

            power_ase[f_ind, :] = bc_evolution + ase_evolution

        spontaneous_raman_scattering.x = 2 * power_ase
        return spontaneous_raman_scattering

    def calculate_stimulated_raman_scattering(self, carriers, raman_pumps):
        """ Returns stimulated Raman scattering solution including
        fiber gain/loss profile.
        :return: None
        """
        # fiber parameters
        fiber_length = self.fiber.params.length
        raman_efficiency = self.fiber.params.raman_efficiency

        if not sim_params.raman_params.flag:
            raman_efficiency['cr'] = zeros(len(raman_efficiency['cr']))
        # raman solver parameters
        z_resolution = sim_params.raman_params.space_resolution
        tolerance = sim_params.raman_params.tolerance

        logger.debug('Start computing fiber Stimulated Raman Scattering')

        power_spectrum, freq_array, prop_direct, _ = self._compute_power_spectrum(carriers, raman_pumps)

        alphap_fiber = self.fiber.alpha(freq_array)

        freq_diff = abs(freq_array - reshape(freq_array, (len(freq_array), 1)))
        interp_cr = interp1d(raman_efficiency['frequency_offset'], raman_efficiency['cr'])
        cr = interp_cr(freq_diff)

        # z propagation axis
        z = append(arange(0, fiber_length, z_resolution), fiber_length)

        def ode_function(z, p):
            return self._ode_stimulated_raman(z, p, alphap_fiber, freq_array, cr, prop_direct)

        def boundary_residual(ya, yb):
            return self._residuals_stimulated_raman(ya, yb, power_spectrum, prop_direct)

        initial_guess_conditions = self._initial_guess_stimulated_raman(z, power_spectrum, alphap_fiber, prop_direct)

        # ODE SOLVER
        bvp_solution = solve_bvp(ode_function, boundary_residual, z, initial_guess_conditions, tol=tolerance)

        rho = (bvp_solution.y.transpose() / power_spectrum).transpose()
        rho = sqrt(rho)    # From power attenuation to field attenuation
        stimulated_raman_scattering = StimulatedRamanScattering(freq_array, bvp_solution.x, rho, bvp_solution.y)

        self._stimulated_raman_scattering = stimulated_raman_scattering

    def _residuals_stimulated_raman(self, ya, yb, power_spectrum, prop_direct):

        computed_boundary_value = zeros(ya.size)

        for index, direction in enumerate(prop_direct):
            if direction == +1:
                computed_boundary_value[index] = ya[index]
            else:
                computed_boundary_value[index] = yb[index]

        return power_spectrum - computed_boundary_value

    def _initial_guess_stimulated_raman(self, z, power_spectrum, alphap_fiber, prop_direct):
        """ Computes the initial guess knowing the boundary conditions
        :param z: patial axis [m]. numpy array
        :param power_spectrum: power in each frequency slice [W].
        Frequency axis is defined by freq_array. numpy array
        :param alphap_fiber: frequency dependent fiber attenuation of signal power [1/m].
        Frequency defined by freq_array. numpy array
        :param prop_direct: indicates the propagation direction of each power slice in power_spectrum:
        +1 for forward propagation and -1 for backward propagation. Frequency defined by freq_array. numpy array
        :return: power_guess: guess on the initial conditions [W].
        The first ndarray index identifies the frequency slice,
        the second ndarray index identifies the step in z. ndarray
        """

        power_guess = empty((power_spectrum.size, z.size))
        for f_index, power_slice in enumerate(power_spectrum):
            if prop_direct[f_index] == +1:
                power_guess[f_index, :] = exp(-alphap_fiber[f_index] * z) * power_slice
            else:
                power_guess[f_index, :] = exp(-alphap_fiber[f_index] * z[::-1]) * power_slice

        return power_guess

    def _ode_stimulated_raman(self, z, power_spectrum, alphap_fiber, freq_array, cr_raman_matrix, prop_direct):
        """ Aim of ode_raman is to implement the set of ordinary differential equations (ODEs)
        describing the Raman effect.
        :param z: spatial axis (unused).
        :param power_spectrum: power in each frequency slice [W].
        Frequency axis is defined by freq_array. numpy array. Size n
        :param alphap_fiber: frequency dependent fiber attenuation of signal power [1/m].
        Frequency defined by freq_array. numpy array. Size n
        :param freq_array: reference frequency axis [Hz]. numpy array. Size n
        :param cr_raman: Cr(f) Raman gain efficiency variation in frequency [1/W/m].
        Frequency defined by freq_array. numpy ndarray. Size nxn
        :param prop_direct: indicates the propagation direction of each power slice in power_spectrum:
        +1 for forward propagation and -1 for backward propagation.
        Frequency defined by freq_array. numpy array. Size n
        :return: dP/dz: the power variation in dz [W/m]. numpy array. Size n
        """

        dpdz = nan * ones(power_spectrum.shape)
        for f_ind, power in enumerate(power_spectrum):
            cr_raman = cr_raman_matrix[f_ind, :]
            vibrational_loss = freq_array[f_ind] / freq_array[:f_ind]

            for z_ind, power_sample in enumerate(power):
                raman_gain = sum(cr_raman[f_ind + 1:] * power_spectrum[f_ind + 1:, z_ind])
                raman_loss = sum(vibrational_loss * cr_raman[:f_ind] * power_spectrum[:f_ind, z_ind])

                dpdz_element = prop_direct[f_ind] * (-alphap_fiber[f_ind] + raman_gain - raman_loss) * power_sample
                dpdz[f_ind][z_ind] = dpdz_element

        return vstack(dpdz)

class NliSolver:
    """ This class implements the NLI models.
        Model and method can be specified in `sim_params.nli_params.method`.
        List of implemented methods:
        'gn_model_analytic': eq. 120 from arXiv:1209.0394
        'ggn_spectrally_separated': eq. 21 from arXiv: 1710.02225 spectrally separated
    """

    def __init__(self, fiber=None):
        """ Initialize the Nli solver object.
        :param fiber: instance of elements.py/Fiber.
        """
        self._fiber = fiber
        self._stimulated_raman_scattering = None

    @property
    def stimulated_raman_scattering(self):
        return self._stimulated_raman_scattering

    @stimulated_raman_scattering.setter
    def stimulated_raman_scattering(self, stimulated_raman_scattering):
        self._stimulated_raman_scattering = stimulated_raman_scattering

    @staticmethod
    def effective_length(alpha, length):
        """The effective length identify the region in which the NLI has a significant contribution to
        the signal degradation.
        """
        return (1 - exp(- alpha * length)) / alpha

    def compute_nli(self, spectral_info: SpectralInformation):
        """ Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier`
        at the end of the fiber span.
        """
        logger.debug('Start computing fiber NLI noise')
        # Physical fiber parameters
        fiber = self._fiber
        srs = self._stimulated_raman_scattering

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
        """ Computes the nonlinear interference power evaluated at the fiber input.
        The method uses eq. 120 from arXiv:1209.0394
        """
        spm_weight = (16.0 / 27.0) * gamma ** 2
        xpm_weight = 2 * (16.0 / 27.0) * gamma ** 2

        nch = spectral_info.number_of_channels
        identity = diag(ones(nch))
        weight = spm_weight*identity + xpm_weight*(ones([nch, nch])-identity)

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
        right_extreme = df + pump_baud_rate/2
        left_extreme = df - pump_baud_rate/2
        psi = (arcsinh(pi ** 2 * asymptotic_length * abs(beta2) * cut_baud_rate * right_extreme) -
               arcsinh(pi ** 2 * asymptotic_length * abs(beta2) * cut_baud_rate * left_extreme)) / 2
        psi *= effective_length ** 2 / (2 * pi * abs(beta2) * asymptotic_length)
        return psi

    # Methods for computing the GGN-model
    @staticmethod
    def _ggn_spectrally_separated(spectral_info: SpectralInformation, srs: StimulatedRamanScattering,
                                  alpha, beta2, beta3, f_ref_beta, gamma):
        """ Computes the nonlinear interference power evaluated at the fiber input.
        The method uses eq. 21 from arXiv: 1710.02225
        """
        dispersion_tolerance = sim_params.nli_params.dispersion_tolerance
        phase_shift_tolerance = sim_params.nli_params.phase_shift_tolerance
        slot_width = max(spectral_info.slot_width)
        delta_z = sim_params.raman_params.space_resolution
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
            delta_beta = 4 * pi**2 * (f1 - f_eval) * (f2_array - f_eval) * \
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
        rho_norm = srs.rho * exp(outer(alpha/2, z))
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
