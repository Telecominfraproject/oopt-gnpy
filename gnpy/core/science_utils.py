import numpy as np
from operator import attrgetter
from collections import namedtuple
import scipy.constants as ph
from scipy.integrate import solve_bvp
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult

from gnpy.core.utils import db2lin, load_json


def load_sim_params(path_sim_params):
    sim_params = load_json(path_sim_params)
    return SimParams(params=sim_params)

def configure_network(network, sim_params):
    from gnpy.core.elements import RamanFiber
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

fib_params = namedtuple('FiberParams', 'loss_coef length beta2 gamma raman_efficiency temperature')
pump = namedtuple('RamanPump', 'power frequency propagation_direction')

def propagate_raman_fiber(fiber, *carriers):
    sim_params = fiber.sim_params
    # apply input attenuation to carriers
    attenuation_in = db2lin(fiber.con_in + fiber.att_in)
    chan = []
    for carrier in carriers:
        pwr = carrier.power
        pwr = pwr._replace(signal=pwr.signal / attenuation_in,
                           nonlinear_interference=pwr.nli / attenuation_in,
                           amplified_spontaneous_emission=pwr.ase / attenuation_in)
        carrier = carrier._replace(power=pwr)
        chan.append(carrier)
    carriers = tuple(f for f in chan)

    fiber_params = fib_params(loss_coef=2*fiber.dbkm_2_lin()[1], length=fiber.length, beta2=fiber.beta2(),
                              gamma=fiber.gamma, raman_efficiency=fiber.params.raman_efficiency,
                              temperature=fiber.operational['temperature'])
    # evaluate fiber attenuation involving also SRS if required by sim_params
    raman_params = fiber.sim_params.raman_params
    if 'raman_pumps' in fiber.operational:
        raman_pumps = tuple(pump(p['power'], p['frequency'], p['propagation_direction'])
                            for p in fiber.operational['raman_pumps'])
    else:
        raman_pumps = None
    if raman_params.flag_raman:
        raman_solver = RamanSolver(raman_params=raman_params, fiber_params=fiber_params)
        stimulated_raman_scattering = raman_solver.stimulated_raman_scattering(carriers=carriers,
                                                                               raman_pumps=raman_pumps)
        fiber_attenuation = (stimulated_raman_scattering.rho[:, -1])**-2
    else:
        fiber_attenuation = tuple(fiber.lin_attenuation for _ in carriers)

    # evaluate Raman ASE noise if required by sim_params and if raman pumps are present
    if raman_params.flag_raman and raman_pumps:
        raman_ase = raman_solver.spontaneous_raman_scattering.power[:, -1]
    else:
        raman_ase = tuple(0 for _ in carriers)

    # evaluate nli and propagate in fiber
    attenuation_out = db2lin(fiber.con_out)
    nli_params = fiber.sim_params.nli_params
    nli_solver = NliSolver(nli_params=nli_params, fiber_params=fiber_params)
    for carrier, attenuation, rmn_ase in zip(carriers, fiber_attenuation, raman_ase):
        pwr = carrier.power
        if carrier.channel_number in sim_params.list_of_channels_under_test:
            carrier_nli = nli_solver.compute_nli(carrier, *carriers)
        else:
            carrier_nli = np.nan
        pwr = pwr._replace(signal=pwr.signal/attenuation/attenuation_out,
                           nonlinear_interference=(pwr.nli+carrier_nli)/attenuation/attenuation_out,
                           amplified_spontaneous_emission=((pwr.ase/attenuation)+rmn_ase)/attenuation_out)
        yield carrier._replace(power=pwr)

class RamanSolver:
    def __init__(self, raman_params=None, fiber_params=None):
        """ Initialize the fiber object with its physical parameters
        :param length: fiber length in m.
        :param alphap: fiber power attenuation coefficient vs frequency in 1/m. numpy array
        :param freq_alpha: frequency axis of alphap in Hz. numpy array
        :param cr_raman: Raman efficiency vs frequency offset in 1/W/m. numpy array
        :param freq_cr: reference frequency offset axis for cr_raman. numpy array
        :param raman_params: namedtuple containing the solver parameters (optional).
        """
        self.fiber_params = fiber_params
        self.raman_params = raman_params
        self.__carriers = None
        self.__stimulated_raman_scattering = None
        self.__spontaneous_raman_scattering = None

    @property
    def fiber_params(self):
        return self.__fiber_params

    @fiber_params.setter
    def fiber_params(self, fiber_params):
        self.__stimulated_raman_scattering = None
        self.__fiber_params = fiber_params

    @property
    def carriers(self):
        return self.__carriers

    @carriers.setter
    def carriers(self, carriers):
        """
        :param carriers: tuple of namedtuples containing information about carriers
        :return:
        """
        self.__carriers = carriers
        self.__stimulated_raman_scattering = None

    @property
    def raman_pumps(self):
        return self._raman_pumps

    @raman_pumps.setter
    def raman_pumps(self, raman_pumps):
        self._raman_pumps = raman_pumps
        self.__stimulated_raman_scattering = None

    @property
    def raman_params(self):
        return self.__raman_params

    @raman_params.setter
    def raman_params(self, raman_params):
        """
        :param raman_params: namedtuple containing the solver parameters (optional).
        :return:
        """
        self.__raman_params = raman_params
        self.__stimulated_raman_scattering = None
        self.__spontaneous_raman_scattering = None

    @property
    def spontaneous_raman_scattering(self):
        if self.__spontaneous_raman_scattering is None:
            # SET STUFF
            loss_coef = self.fiber_params.loss_coef
            raman_efficiency = self.fiber_params.raman_efficiency
            temperature = self.fiber_params.temperature

            carriers = self.carriers
            raman_pumps = self.raman_pumps

            verbose = self.raman_params.verbose

            if verbose:
                print('Start computing fiber Spontaneous Raman Scattering')

            power_spectrum, freq_array, prop_direct, bn_array = self._compute_power_spectrum(carriers, raman_pumps)

            if not hasattr(loss_coef, 'alpha_power'):
                alphap_fiber = loss_coef * np.ones(freq_array.shape)
            else:
                interp_alphap = interp1d(loss_coef['frequency'], loss_coef['alpha_power'])
                alphap_fiber = interp_alphap(freq_array)

            freq_diff = abs(freq_array - np.reshape(freq_array, (len(freq_array), 1)))
            interp_cr = interp1d(raman_efficiency['frequency_offset'], raman_efficiency['cr'])
            cr = interp_cr(freq_diff)

            # z propagation axis
            z_array = self.__stimulated_raman_scattering.z
            ase_bc = np.zeros(freq_array.shape)

            # calculate ase power
            spontaneous_raman_scattering = self._int_spontaneous_raman(z_array, self.__stimulated_raman_scattering.power,
                                                                       alphap_fiber, freq_array, cr, freq_diff, ase_bc,
                                                                       bn_array, temperature)

            setattr(spontaneous_raman_scattering, 'frequency', freq_array)
            setattr(spontaneous_raman_scattering, 'z', z_array)
            setattr(spontaneous_raman_scattering, 'power', spontaneous_raman_scattering.x)
            delattr(spontaneous_raman_scattering, 'x')

            if verbose:
                print(spontaneous_raman_scattering.message)

            self.__spontaneous_raman_scattering = spontaneous_raman_scattering

        return self.__spontaneous_raman_scattering

    @staticmethod
    def _compute_power_spectrum(carriers, raman_pumps=None):
        """
        Rearrangement of spectral and Raman pump information to make them compatible with Raman solver
        :param carriers: a tuple of namedtuples describing the transmitted channels
        :param raman_pumps: a namedtuple describing the Raman pumps
        :return:
        """

        # Signal power spectrum
        pow_array = np.array([])
        f_array = np.array([])
        noise_bandwidth_array = np.array([])
        for carrier in sorted(carriers, key=attrgetter('frequency')):
            f_array = np.append(f_array, carrier.frequency)
            pow_array = np.append(pow_array, carrier.power.signal)
            ref_bw = carrier.baud_rate
            noise_bandwidth_array = np.append(noise_bandwidth_array, ref_bw)

        propagation_direction = np.ones(len(f_array))

        # Raman pump power spectrum
        if raman_pumps:
            for pump in raman_pumps:
                pow_array = np.append(pow_array, pump.power)
                f_array = np.append(f_array, pump.frequency)
                direction = +1 if pump.propagation_direction.lower() == 'coprop' else -1
                propagation_direction = np.append(propagation_direction, direction)
                noise_bandwidth_array = np.append(noise_bandwidth_array, ref_bw)

        # Final sorting
        ind = np.argsort(f_array)
        f_array = f_array[ind]
        pow_array = pow_array[ind]
        propagation_direction = propagation_direction[ind]

        return pow_array, f_array, propagation_direction, noise_bandwidth_array

    def _int_spontaneous_raman(self, z_array, raman_matrix, alphap_fiber, freq_array, cr_raman_matrix, freq_diff, ase_bc, bn_array, temperature):
        spontaneous_raman_scattering = OptimizeResult()

        dx = self.raman_params.space_resolution
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

    def stimulated_raman_scattering(self, carriers, raman_pumps=None):
        """ Returns stimulated Raman scattering solution including 
        fiber gain/loss profile.
        :return: self.__stimulated_raman_scattering: the SRS problem solution.
        scipy.interpolate.PPoly instance
        """

        if self.__stimulated_raman_scattering is None:
            # fiber parameters
            fiber_length = self.fiber_params.length
            loss_coef = self.fiber_params.loss_coef
            raman_efficiency = self.fiber_params.raman_efficiency
            # raman solver parameters
            z_resolution = self.raman_params.space_resolution
            tolerance = self.raman_params.tolerance
            verbose = self.raman_params.verbose

            if verbose:
                print('Start computing fiber Stimulated Raman Scattering')

            power_spectrum, freq_array, prop_direct, _ = self._compute_power_spectrum(carriers, raman_pumps)

            if not hasattr(loss_coef, 'alpha_power'):
                alphap_fiber = loss_coef * np.ones(freq_array.shape)
            else:
                interp_alphap = interp1d(loss_coef['frequency'], loss_coef['alpha_power'])
                alphap_fiber = interp_alphap(freq_array)

            freq_diff = abs(freq_array - np.reshape(freq_array, (len(freq_array), 1)))
            interp_cr = interp1d(raman_efficiency['frequency_offset'], raman_efficiency['cr'])
            cr = interp_cr(freq_diff)

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

            self.carriers = carriers
            self.raman_pumps = raman_pumps
            self.__stimulated_raman_scattering = stimulated_raman_scattering

        return self.__stimulated_raman_scattering

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

class NliSolver:
    """ This class implements the NLI models.
        Model and method can be specified in `self.nli_params.method`.
        List of implemented methods:
        'gn_model_analytic': brute force triple integral solution
        'GGN_spectrally_separated_xpm_spm': XPM plus SPM
    """

    def __init__(self, nli_params=None, fiber_params=None):
        """ Initialize the fiber object with its physical parameters
        """
        self.fiber_params = fiber_params
        self.nli_params = nli_params
        self.srs_profile = None

    @property
    def fiber_params(self):
        return self.___fiber_params

    @fiber_params.setter
    def fiber_params(self, fiber_params):
        self.___fiber_params = fiber_params

    @property
    def srs_profile(self):
        return self.__srs_profile

    @srs_profile.setter
    def srs_profile(self, srs_profile):
        self.__srs_profile = srs_profile

    @property
    def nli_params(self):
        return self.__nli_params

    @nli_params.setter
    def nli_params(self, nli_params):
        """
        :param model_params: namedtuple containing the parameters used to compute the NLI.
        """
        self.__nli_params = nli_params

    def alpha0(self, f_eval=193.5e12):
        if not hasattr(self.fiber_params.loss_coef, 'alpha_power'):
            alpha0 = self.fiber_params.loss_coef
        else:
            alpha_interp = interp1d(self.fiber_params.loss_coef['frequency'],
                                    self.fiber_params.loss_coef['alpha_power'])
            alpha0 = alpha_interp(f_eval)
        return alpha0

    def compute_nli(self, carrier, *carriers):
        """ Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier`
        at the end of the fiber span.
        """
        if 'gn_model_analytic' == self.nli_params.nli_method_name.lower():
            carrier_nli = self._gn_analytic(carrier, *carriers)
        else:
            raise ValueError(f'Method {self.nli_params.method_nli} not implemented.')

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
        beta2 = self.fiber_params.beta2
        gamma = self.fiber_params.gamma
        length = self.fiber_params.length
        effective_length = (1 - np.exp(-2 * alpha * length)) / (2 * alpha)
        asymptotic_length = 1 / (2 * alpha)

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
        beta2 = self.fiber_params.beta2
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
