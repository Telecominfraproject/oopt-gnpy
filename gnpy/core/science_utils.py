import progressbar
import numpy as np
from operator import attrgetter
from scipy.interpolate import interp1d

from gnpy.core.utils import load_json
import scipy.constants as ph
from scipy.integrate import solve_bvp
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult
from gnpy.core.elements import RamanFiber


def load_sim_params(path_sim_params):
    sim_params = load_json(path_sim_params)
    return SimParams(params=sim_params)

def configure_network(network, sim_params):
    for node in network.nodes:
        if isinstance(node, RamanFiber):
            node.sim_params = sim_params

class RamanParams():
    def __init__(self, params=None):
        if params:
            self.flag_raman = params['flag_raman']
            self.space_resolution = params['space_resolution']
            self.tolerance = params['tolerance']
            self.verbose = params['verbose']

class NLIParams():
    def __init__(self, params=None):
        if params:
            self.nli_method_name = params['nli_method_name']
            self.wdm_grid_size = params['wdm_grid_size']
            self.dispersion_tolerance = params['dispersion_tolerance']
            self.phase_shift_tollerance = params['phase_shift_tollerance']
            self.verbose = params['verbose']

class SimParams():
    def __init__(self, params=None):
        if params:
            self.list_of_channels_under_test = params['list_of_channels_under_test']
            self.raman_params = RamanParams(params=params['raman_parameters'])
            self.nli_params = NLIParams(params=params['nli_parameters'])

class RamanSolver:
    def __init__(self, fiber_information=None):
        """ Initialize the fiber object with its physical parameters
        :param length: fiber length in m.
        :param alphap: fiber power attenuation coefficient vs frequency in 1/m. numpy array
        :param freq_alpha: frequency axis of alphap in Hz. numpy array
        :param cr_raman: Raman efficiency vs frequency offset in 1/W/m. numpy array
        :param freq_cr: reference frequency offset axis for cr_raman. numpy array
        :param solver_params: namedtuple containing the solver parameters (optional).
        """
        self._fiber_information = fiber_information
        self._solver_params = None
        self._spectral_information = None
        self._raman_pump_information = None
        self._stimulated_raman_scattering = None
        self._spontaneous_raman_scattering = None

    @property
    def fiber_information(self):
        return self._fiber_information

    @fiber_information.setter
    def fiber_information(self, fiber_information):
        self._fiber_information = fiber_information
        self._stimulated_raman_scattering = None

    @property
    def spectral_information(self):
        return self._spectral_information

    @spectral_information.setter
    def spectral_information(self, spectral_information):
        """
        :param spectral_information: namedtuple containing all the spectral information about carriers and eventual Raman pumps
        :return:
        """
        self._spectral_information = spectral_information
        self._stimulated_raman_scattering = None

    @property
    def raman_pump_information(self):
        return self._raman_pump_information

    @raman_pump_information.setter
    def raman_pump_information(self, raman_pump_information):
        self._raman_pump_information = raman_pump_information

    @property
    def solver_params(self):
        return self._solver_params

    @solver_params.setter
    def solver_params(self, solver_params):
        """
        :param solver_params: namedtuple containing the solver parameters (optional).
        :return:
        """
        self._solver_params = solver_params
        self._stimulated_raman_scattering = None
        self._spontaneous_raman_scattering = None

    @property
    def spontaneous_raman_scattering(self):
        if self._spontaneous_raman_scattering is None:
            # SET STUFF
            attenuation_coefficient = self.fiber_information.attenuation_coefficient
            raman_coefficient = self.fiber_information.raman_coefficient
            temp_k = self.fiber_information.temperature

            spectral_info = self.spectral_information
            raman_pump_information = self.raman_pump_information

            verbose = self.solver_params.verbose

            if verbose:
                print('Start computing fiber Spontaneous Raman Scattering')

            power_spectrum, freq_array, prop_direct, bn_array = self._compute_power_spectrum(spectral_info, raman_pump_information)
            temperature = temp_k.temperature

            if len(attenuation_coefficient.alpha_power) >= 2:
                interp_alphap = interp1d(attenuation_coefficient.frequency, attenuation_coefficient.alpha_power)
                alphap_fiber = interp_alphap(freq_array)
            else:
                alphap_fiber = attenuation_coefficient.alpha_power * np.ones(freq_array.shape)

            freq_diff = abs(freq_array - np.reshape(freq_array, (len(freq_array), 1)))
            if len(raman_coefficient.cr) >= 2:
                interp_cr = interp1d(raman_coefficient.frequency, raman_coefficient.cr)
                cr = interp_cr(freq_diff)
            else:
                cr = raman_coefficient.cr * np.ones(freq_diff.shape)

            # z propagation axis
            z_array = self.stimulated_raman_scattering.z
            ase_bc = np.zeros(freq_array.shape)

            # calculate ase power
            spontaneous_raman_scattering = self._int_spontaneous_raman(z_array, self.stimulated_raman_scattering.power, alphap_fiber, freq_array, cr, freq_diff, ase_bc, bn_array, temperature)

            setattr(spontaneous_raman_scattering, 'frequency', freq_array)
            setattr(spontaneous_raman_scattering, 'z', z_array)
            setattr(spontaneous_raman_scattering, 'power', spontaneous_raman_scattering.x)
            delattr(spontaneous_raman_scattering, 'x')

            if verbose:
                print(spontaneous_raman_scattering.message)

            self._spontaneous_raman_scattering = spontaneous_raman_scattering

        return self._spontaneous_raman_scattering

    @staticmethod
    def _compute_power_spectrum(spectral_information, raman_pump_information):
        """
        Rearrangement of spectral and Raman pump information to make them compatible with Raman solver
        :param spectral_information: a namedtuple describing the transmitted channels
        :param raman_pump_information: a namedtuple describing the Raman pumps
        :return:
        """

        # Signal power spectrum
        pow_array = np.array([])
        f_array = np.array([])
        noise_bandwidth_array = np.array([])
        for carrier in sorted(spectral_information.carriers, key=attrgetter('frequency')):
            f_array = np.append(f_array, carrier.frequency)
            pow_array = np.append(pow_array, carrier.power.signal)
            noise_bandwidth_array = np.append(noise_bandwidth_array, carrier.baud_rate)

        propagation_direction = np.ones(len(f_array))

        # Raman pump power spectrum
        if raman_pump_information:
            for pump in raman_pump_information.raman_pumps:
                pow_array = np.append(pow_array, pump.power)
                f_array = np.append(f_array, pump.frequency)
                propagation_direction = np.append(propagation_direction, pump.propagation_direction)
                noise_bandwidth_array = np.append(noise_bandwidth_array, pump.pump_bandwidth)

        # Final sorting
        ind = np.argsort(f_array)
        f_array = f_array[ind]
        pow_array = pow_array[ind]
        propagation_direction = propagation_direction[ind]

        return pow_array, f_array, propagation_direction, noise_bandwidth_array

    def _int_spontaneous_raman(self, z_array, raman_matrix, alphap_fiber, freq_array, cr_raman_matrix, freq_diff, ase_bc, bn_array, temperature):
        spontaneous_raman_scattering = OptimizeResult()

        dx = self.solver_params.z_resolution
        h = ph.value('Planck constant')
        kb = ph.value('Boltzmann constant')

        power_ase = np.nan * np.ones(raman_matrix.shape)
        int_pump = cumtrapz(raman_matrix, z_array, dx=dx, axis=1, initial=0)

        for f_ind, f_ase in enumerate(freq_array):
            cr_raman = cr_raman_matrix[f_ind, :]
            vibrational_loss = f_ase / freq_array[:f_ind]
            eta = 1/(np.exp((h*freq_diff[f_ind, f_ind+1:])/(kb*temperature)) - 1)

            int_fiber_loss = -alphap_fiber[f_ind] * z_array
            int_raman_loss = np.sum((cr_raman[:f_ind] * vibrational_loss * int_pump[:f_ind, :].transpose()).transpose(), axis=0)
            int_raman_gain = np.sum((cr_raman[f_ind + 1:] * int_pump[f_ind + 1:, :].transpose()).transpose(), axis=0)

            int_gain_loss = int_fiber_loss + int_raman_gain + int_raman_loss

            new_ase = np.sum((cr_raman[f_ind+1:] * (1 + eta) * raman_matrix[f_ind+1:, :].transpose()).transpose() * h * f_ase * bn_array[f_ind], axis=0)

            bc_evolution = ase_bc[f_ind] * np.exp(int_gain_loss)
            ase_evolution = np.exp(int_gain_loss) * cumtrapz(new_ase*np.exp(-int_gain_loss), z_array, dx=dx, initial=0)

            power_ase[f_ind, :] = bc_evolution + ase_evolution

        spontaneous_raman_scattering.x = 2 * power_ase
        spontaneous_raman_scattering.success = True
        spontaneous_raman_scattering.message = "Spontaneous Raman Scattering evaluated successfully"

        return spontaneous_raman_scattering

    @property
    def stimulated_raman_scattering(self):
        """ Return rho fiber gain/loss profile induced by stimulated Raman scattering.
        :return: self._stimulated_raman_scattering: the SRS problem solution.
        scipy.interpolate.PPoly instance
        """

        if self._stimulated_raman_scattering is None:
            fiber_length = self.fiber_information.length
            attenuation_coefficient = self.fiber_information.attenuation_coefficient
            raman_coefficient = self.fiber_information.raman_coefficient

            spectral_info = self.spectral_information
            raman_pump_information = self.raman_pump_information

            z_resolution = self.solver_params.z_resolution
            tolerance = self.solver_params.tolerance
            verbose = self.solver_params.verbose

            if verbose:
                print('Start computing fiber Stimulated Raman Scattering')

            power_spectrum, freq_array, prop_direct, _ = self._compute_power_spectrum(spectral_info, raman_pump_information)

            if len(attenuation_coefficient.alpha_power) >= 2:
                interp_alphap = interp1d(attenuation_coefficient.frequency, attenuation_coefficient.alpha_power)
                alphap_fiber = interp_alphap(freq_array)
            else:
                alphap_fiber = attenuation_coefficient.alpha_power * np.ones(freq_array.shape)

            freq_diff = abs(freq_array - np.reshape(freq_array, (len(freq_array), 1)))
            if len(raman_coefficient.cr) >= 2:
                interp_cr = interp1d(raman_coefficient.frequency, raman_coefficient.cr)
                cr = interp_cr(freq_diff)
            else:
                cr = raman_coefficient.cr * np.ones(freq_diff.shape)

            # z propagation axis
            z = np.arange(0, fiber_length+1, z_resolution)

            ode_function = lambda z, p: self._ode_stimulated_raman(z, p, alphap_fiber, freq_array, cr, prop_direct)
            boundary_residual = lambda ya, yb: self._residuals_stimulated_raman(ya, yb, power_spectrum, prop_direct)
            initial_guess_conditions = self._initial_guess_stimulated_raman(z, power_spectrum, alphap_fiber, prop_direct)

            # ODE SOLVER
            stimulated_raman_scattering = solve_bvp(ode_function, boundary_residual, z, initial_guess_conditions, tol=tolerance, verbose=verbose)

            rho = (stimulated_raman_scattering.y.transpose() / power_spectrum).transpose()
            rho = np.sqrt(rho)    # From power attenuation to field attenuation

            setattr(stimulated_raman_scattering, 'frequency', freq_array)
            setattr(stimulated_raman_scattering, 'z', stimulated_raman_scattering.x)
            setattr(stimulated_raman_scattering, 'rho', rho)
            setattr(stimulated_raman_scattering, 'power', stimulated_raman_scattering.y)
            delattr(stimulated_raman_scattering, 'x')
            delattr(stimulated_raman_scattering, 'y')

            self._stimulated_raman_scattering = stimulated_raman_scattering

        return self._stimulated_raman_scattering

    def _residuals_stimulated_raman(self, ya, yb, power_spectrum, prop_direct):

        computed_boundary_value = np.zeros(ya.size)

        for index, direction in enumerate(prop_direct):
            if direction == +1:
                computed_boundary_value[index] = ya[index]
            else:
                computed_boundary_value[index] = yb[index]

        return power_spectrum - computed_boundary_value

    def _initial_guess_stimulated_raman(self, z, power_spectrum, alphap_fiber, prop_direct):
        """ Computes the initial guess knowing the boundary conditions
        :param z: patial axis [m]. numpy array
        :param power_spectrum: power in each frequency slice [W].    Frequency axis is defined by freq_array. numpy array
        :param alphap_fiber: frequency dependent fiber attenuation of signal power [1/m]. Frequency defined by freq_array. numpy array
        :param prop_direct: indicates the propagation direction of each power slice in power_spectrum:
        +1 for forward propagation and -1 for backward propagation. Frequency defined by freq_array. numpy array
        :return: power_guess: guess on the initial conditions [W]. The first ndarray index identifies the frequency slice,
        the second ndarray index identifies the step in z. ndarray
        """

        power_guess = np.empty((power_spectrum.size, z.size))
        for f_index, power_slice in enumerate(power_spectrum):
            if prop_direct[f_index] == +1:
                power_guess[f_index, :] = np.exp(-alphap_fiber[f_index] * z) * power_slice
            else:
                power_guess[f_index, :] = np.exp(-alphap_fiber[f_index] * z[::-1]) * power_slice

        return power_guess

    def _ode_stimulated_raman(self, z, power_spectrum, alphap_fiber, freq_array, cr_raman_matrix, prop_direct):
        """ Aim of ode_raman is to implement the set of ordinary differential equations (ODEs) describing the Raman effect.
        :param z: spatial axis (unused).
        :param power_spectrum: power in each frequency slice [W].    Frequency axis is defined by freq_array. numpy array. Size n
        :param alphap_fiber: frequency dependent fiber attenuation of signal power [1/m]. Frequency defined by freq_array. numpy array. Size n
        :param freq_array: reference frequency axis [Hz]. numpy array. Size n
        :param cr_raman: Cr(f) Raman gain efficiency variation in frequency [1/W/m]. Frequency defined by freq_array. numpy ndarray. Size nxn
        :param prop_direct: indicates the propagation direction of each power slice in power_spectrum:
        +1 for forward propagation and -1 for backward propagation. Frequency defined by freq_array. numpy array. Size n
        :return: dP/dz: the power variation in dz [W/m]. numpy array. Size n
        """

        dpdz = np.nan * np.ones(power_spectrum.shape)
        for f_ind, power in enumerate(power_spectrum):
            cr_raman = cr_raman_matrix[f_ind, :]
            vibrational_loss = freq_array[f_ind] / freq_array[:f_ind]

            for z_ind, power_sample in enumerate(power):
                raman_gain = np.sum(cr_raman[f_ind+1:] * power_spectrum[f_ind+1:, z_ind])
                raman_loss = np.sum(vibrational_loss * cr_raman[:f_ind] * power_spectrum[:f_ind, z_ind])

                dpdz_element = prop_direct[f_ind] * (-alphap_fiber[f_ind] + raman_gain - raman_loss) * power_sample
                dpdz[f_ind][z_ind] = dpdz_element

        return np.vstack(dpdz)

class NLI:
    """ This class implements the NLI models.
        Model and method can be specified in `self.model_parameters.method`.
        List of implemented methods:
        'gn_analytic': brute force triple integral solution
        'GGN_spectrally_separated_xpm_spm': XPM plus SPM
    """

    def __init__(self, fiber_information=None):
        """ Initialize the fiber object with its physical parameters
        """
        self.fiber_information = fiber_information
        self.srs_profile = None
        self.model_parameters = None

    @property
    def fiber_information(self):
        return self.__fiber_information

    @fiber_information.setter
    def fiber_information(self, fiber_information):
        self.__fiber_information = fiber_information

    @property
    def srs_profile(self):
        return self.__srs_profile

    @srs_profile.setter
    def srs_profile(self, srs_profile):
        self.__srs_profile = srs_profile

    @property
    def model_parameters(self):
        return self.__model_parameters

    @model_parameters.setter
    def model_parameters(self, model_params):
        """
        :param model_params: namedtuple containing the parameters used to compute the NLI.
        """
        self.__model_parameters = model_params

    def alpha0(self, f_eval=193.5e12):
        if len(self.fiber_information.attenuation_coefficient.alpha_power) == 1:
            alpha0 = self.fiber_information.attenuation_coefficient.alpha_power[0]
        else:
            alpha_interp = interp1d(self.fiber_information.attenuation_coefficient.frequency,
                                    self.fiber_information.attenuation_coefficient.alpha_power)
            alpha0 = alpha_interp(f_eval)
        return alpha0

    def compute_nli(self, carrier, *carriers):
        """ Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier`
        at the end of the fiber span.
        """
        if 'gn_model_analytic' == self.model_parameters.method.lower():
            carrier_nli = self._gn_analytic(carrier, *carriers)
        else:
            raise ValueError(f'Method {self.model_parameters.method_nli} not implemented.')

        return carrier_nli

    # Methods for computing spectrally separated GN
    def _gn_analytic(self, carrier, *carriers):
        """ Computes the nonlinear interference power on a single carrier.
        The method uses eq. 120 from arXiv:1209.0394.
        :param carrier: the signal under analysis
        :param carriers: the full WDM comb
        :return: carrier_nli: the amount of nonlinear interference in W on the under analysis
        """

        alpha = self.alpha0() / 2
        length = self.fiber_information.length
        effective_length = (1 - np.exp(-2 * alpha * length)) / (2 * alpha)
        asymptotic_length = 1 / (2 * alpha)

        beta2 = self.fiber_information.beta2
        gamma = self.fiber_information.gamma

        g_nli = 0
        for interfering_carrier in carriers:
            g_interfearing = interfering_carrier.power.signal / interfering_carrier.baud_rate
            g_signal = carrier.power.signal / carrier.baud_rate
            g_nli += g_interfearing**2 * g_signal * self._psi(carrier, interfering_carrier)
        g_nli *= (16.0 / 27.0) * (gamma * effective_length)**2 /\
                 (2 * np.pi * abs(beta2) * asymptotic_length)

        carrier_nli = carrier.baud_rate * g_nli
        return carrier_nli

    def _psi(self, carrier, interfering_carrier):
        """ Calculates eq. 123 from arXiv:1209.0394.
        """
        alpha = self.alpha0() / 2
        beta2 = self.fiber_information.beta2

        asymptotic_length = 1 / (2 * alpha)
        if carrier.channel_number == interfering_carrier.channel_number:  # SPM
            psi = np.arcsinh(0.5 * np.pi**2 * asymptotic_length
                              * abs(beta2) * carrier.baud_rate**2)
        else:  # XPM
            delta_f = carrier.frequency - interfering_carrier.frequency
            psi = np.arcsinh(np.pi**2 * asymptotic_length * abs(beta2) *
                             carrier.baud_rate * (delta_f + 0.5 * interfering_carrier.baud_rate))
            psi -= np.arcsinh(np.pi**2 * asymptotic_length * abs(beta2) *
                              carrier.baud_rate * (delta_f - 0.5 * interfering_carrier.baud_rate))

        return psi
