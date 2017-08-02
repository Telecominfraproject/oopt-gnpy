# -*- coding: utf-8 -*-

"""Top-level package for gnpy."""

__author__ = """<TBD>"""
__email__ = '<TBD>@<TBD>.com'
__version__ = '0.1.0'

import numpy as np
import multiprocessing as mp
import scipy.interpolate as interp

"""
GNPy: a Python 3 implementation of the Gaussian Noise (GN) Model of nonlinear
propagation, developed by the OptCom group, Department of Electronics and
Telecommunications, Politecnico di Torino, Italy
"""

__credits__ = ["Mattia Cantono", "Vittorio Curri", "Alessio Ferrari"]


def raised_cosine_comb(f, rs, roll_off, center_freq, power):
    """ Returns an array storing the PSD of a WDM comb of raised cosine shaped
    channels at the input frequencies defined in array f

    :param f: Array of frequencies in THz
    :param rs: Array of Symbol Rates in TBaud. One Symbol rate for each channel
    :param roll_off: Array of roll-off factors [0,1). One per channel
    :param center_freq: Array of channels central frequencies in THz. One per channel
    :param power: Array of channel powers in W. One per channel
    :return: PSD of the WDM comb evaluated over f
    """
    ts_arr = 1.0 / rs
    passband_arr = (1.0 - roll_off) / (2.0 * ts_arr)
    stopband_arr = (1.0 + roll_off) / (2.0 * ts_arr)
    g = power / rs
    psd = np.zeros(np.shape(f))
    for ind in range(np.size(center_freq)):
        f_nch = center_freq[ind]
        g_ch = g[ind]
        ts = ts_arr[ind]
        passband = passband_arr[ind]
        stopband = stopband_arr[ind]
        ff = np.abs(f - f_nch)
        tf = ff - passband
        if roll_off[ind] == 0:
            psd = np.where(tf <= 0, g_ch, 0.) + psd
        else:
            psd = g_ch * (np.where(tf <= 0, 1., 0.) + 1.0 / 2.0 * (1 + np.cos(np.pi * ts / roll_off[ind] *
                                                                          tf)) * np.where(tf > 0, 1., 0.) *
                          np.where(np.abs(ff) <= stopband, 1., 0.)) + psd

    return psd


def fwm_eff(a, Lspan, b2, ff):
    """ Computes the four-wave mixing efficiency given the fiber characteristics
    over a given frequency set ff
    :param a: Fiber loss coefficient in 1/km
    :param Lspan: Fiber length in km
    :param b2: Fiber Dispersion coefficient in ps/THz/km
    :param ff: Array of Frequency points in THz
    :return: FWM efficiency rho
    """
    rho = np.power(np.abs((1.0 - np.exp(-2.0 * a * Lspan + 1j * 4.0 * np.pi * np.pi * b2 * Lspan * ff)) / (
        2.0 * a - 1j * 4.0 * np.pi * np.pi * b2 * ff)), 2)
    return rho


def get_freqarray(f, Bopt, fmax, max_step, f_dense_low, f_dense_up, df_dense):
    """ Returns a non-uniformly spaced frequency array useful for fast GN-model.
    integration. The frequency array is made of a denser area, sided by two
    log-spaced arrays
    :param f: Central frequency at which NLI is evaluated in THz
    :param Bopt: Total optical bandwidth of the system in THz
    :param fmax: Upper limit of the integration domain in THz
    :param max_step: Maximum step size for frequency array definition in THz
    :param f_dense_low: Lower limit of denser frequency region in THz
    :param f_dense_up: Upper limit of denser frequency region in THz
    :param df_dense: Step size to be used in the denser frequency region in THz
    :return: Non uniformly defined frequency array
    """
    f_dense = np.arange(f_dense_low, f_dense_up, df_dense)
    k = Bopt / 2.0 / (Bopt / 2.0 - max_step)  # Compute Step ratio for log-spaced array definition
    if f < 0:
        Nlog_short = np.ceil(np.log(fmax / np.abs(f_dense_low)) / np.log(k) + 1.0)
        f1_short = -(np.abs(f_dense_low) * np.power(k, np.arange(Nlog_short, 0.0, -1.0) - 1.0))
        k = (Bopt / 2 + (np.abs(f_dense_up) - f_dense_low)) / (Bopt / 2.0 - max_step + (np.abs(f_dense_up) - f_dense_up))
        Nlog_long = np.ceil(np.log((fmax + (np.abs(f_dense_up) - f_dense_up)) / abs(f_dense_up)) * 1.0 / np.log(k) + 1.0)
        f1_long = np.abs(f_dense_up) * np.power(k, (np.arange(1, Nlog_long + 1) - 1.0)) - (
            np.abs(f_dense_up) - f_dense_up)
        f1_array = np.concatenate([f1_short, f_dense[1:], f1_long])
    else:
        Nlog_short = np.ceil(np.log(fmax / np.abs(f_dense_up)) / np.log(k) + 1.0)
        f1_short = f_dense_up * np.power(k, np.arange(1, Nlog_short + 1.0) - 1.0)
        k = (Bopt / 2.0 + (abs(f_dense_low) + f_dense_low)) / (Bopt / 2.0 - max_step + (abs(f_dense_low) + f_dense_low))
        Nlog_long = np.ceil(np.log((fmax + (np.abs(f_dense_low) + f_dense_low)) / np.abs(f_dense_low)) / np.log(k) + 1)
        f1_long = -(np.abs(f_dense_low) * np.power(k, np.arange(Nlog_long, 0, -1) - 1.0)) + (
            abs(f_dense_low) + f_dense_low)
        f1_array = np.concatenate([f1_long, f_dense[1:], f1_short])
    return f1_array


def GN_integral(b2, Lspan, a_db, gam, f_ch, b_ch, roll_off, power, Nch, model_param):
    """ GN_integral computes the GN reference formula via smart brute force integration. The Gaussian Noise model is
    applied in its incoherent form (phased-array factor =1). The function computes the integral by columns: for each f1,
    a non-uniformly spaced f2 array is generated, and the integrand function is computed there. At the end of the loop
    on f1, the overall GNLI is computed. Accuracy can be tuned by operating on model_param argument.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param Lspan: Fiber Span length in km. Scalar
    :param a_db: Fiber loss coeffiecient in dB/km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param b_ch: Channels' -3 dB bandwidth. Array of size 1xNch
    :param roll_off: Channels' Roll-off factors [0,1). Array of size 1xNch
    :param power: Channels' power values in W. Array of size 1xNch
    :param Nch: Number of channels. Scalar
    :param model_param: Dictionary with model parameters for accuracy tuning
                        model_param['min_FWM_inv']: Minimum FWM efficiency value to be considered for high density
                        integration in dB
                        model_param['n_grid']: Maximum Number of integration points to be used in each frequency slot of
                        the spectrum
                        model_param['n_grid_min']: Minimum Number of integration points to be used in each frequency
                        slot of the spectrum
                        model_param['f_array']: Frequencies at which evaluate GNLI, expressed in THz
    :return: GNLI: power spectral density in W/THz of the nonlinear interference at frequencies model_param['f_array']
    """
    alpha_lin = a_db / 20.0 / np.log10(np.e)  # Conversion in linear units 1/km
    min_FWM_inv = np.power(10, model_param['min_FWM_inv'] / 10)  # Conversion in linear units
    n_grid = model_param['n_grid']
    n_grid_min = model_param['n_grid_min']
    f_array = model_param['f_array']
    fmax = (f_ch[-1] - (b_ch[-1] / 2.0)) - (f_ch[0] - (b_ch[0] / 2.0))  # Get frequency limit
    f2eval = np.max(np.diff(f_ch))
    Bopt = f2eval * Nch  # Overall optical bandwidth [THz]
    min_step = f2eval / n_grid  # Minimum integration step
    max_step = f2eval / n_grid_min  # Maximum integration step
    f_dense_start = np.abs(
        np.sqrt(np.power(alpha_lin, 2) / (4.0 * np.power(np.pi, 4) * b2 * b2) * (min_FWM_inv - 1.0)) / f2eval)
    f_ind_eval = 0
    GNLI = np.full(f_array.size, np.nan)  # Pre-allocate results
    for f in f_array:  # Loop over f
        f_dense_low = f - f_dense_start
        f_dense_up = f + f_dense_start
        if f_dense_low < -fmax:
            f_dense_low = -fmax
        if f_dense_low == 0.0:
            f_dense_low = -min_step
        if f_dense_up == 0.0:
            f_dense_up = min_step
        if f_dense_up > fmax:
            f_dense_up = fmax
        f_dense_width = np.abs(f_dense_up - f_dense_low)
        n_grid_dense = np.ceil(f_dense_width / min_step)
        df = f_dense_width / n_grid_dense
        # Get non-uniformly spaced f1 array
        f1_array = get_freqarray(f, Bopt, fmax, max_step, f_dense_low, f_dense_up, df)
        G1 = raised_cosine_comb(f1_array, b_ch, roll_off, f_ch, power)  # Get corresponding spectrum
        Gpart = np.zeros(f1_array.size)  # Pre-allocate partial result for inner integral
        f_ind = 0
        for f1 in f1_array:  # Loop over f1
            if f1 != f:
                f_lim = np.sqrt(np.power(alpha_lin, 2) / (4.0 * np.power(np.pi, 4) * b2 * b2) * (min_FWM_inv - 1.0)) / (
                    f1 - f) + f
                f2_dense_up = np.maximum(f_lim, -f_lim)
                f2_dense_low = np.minimum(f_lim, -f_lim)
                if f2_dense_low == 0:
                    f2_dense_low = -min_step
                if f2_dense_up == 0:
                    f2_dense_up = min_step
                if f2_dense_low < -fmax:
                    f2_dense_low = -fmax
                if f2_dense_up > fmax:
                    f2_dense_up = fmax
            else:
                f2_dense_up = fmax
                f2_dense_low = -fmax
            f2_dense_width = np.abs(f2_dense_up - f2_dense_low)
            n2_grid_dense = np.ceil(f2_dense_width / min_step)
            df2 = f2_dense_width / n2_grid_dense
            # Get non-uniformly spaced f2 array
            f2_array = get_freqarray(f, Bopt, fmax, max_step, f2_dense_low, f2_dense_up, df2)
            f2_array = f2_array[f2_array >= f1]  # Do not consider points below the bisector of quadrants I and III
            if f2_array.size > 0:
                G2 = raised_cosine_comb(f2_array, b_ch, roll_off, f_ch, power)  # Get spectrum there
                f3_array = f1 + f2_array - f  # Compute f3
                G3 = raised_cosine_comb(f3_array, b_ch, roll_off, f_ch, power)  # Get spectrum over f3
                G = G2 * G3 * G1[f_ind]
                if np.count_nonzero(G):
                    FWM_eff = fwm_eff(alpha_lin, Lspan, b2, (f1 - f) * (f2_array - f))  # Compute FWM efficiency
                    Gpart[f_ind] = 2.0 * np.trapz(FWM_eff * G, f2_array)  # Compute inner integral
            f_ind += 1
            # Compute outer integral. Nominal span loss already compensated
        GNLI[f_ind_eval] = 16.0 / 27.0 * gam * gam * np.trapz(Gpart, f1_array)
        f_ind_eval += 1  # Next frequency
    return GNLI  # Return GNLI array in W/THz and the array of the corresponding frequencies


def compute_psi(b2, l_eff_a, f_ch, channel_index, interfering_index, b_ch):
    """ compute_psi computes the psi coefficient of the analytical formula.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_eff_a: Asymptotic effective length in km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param channel_index: Index of the channel. Scalar
    :param interfering_index: Index of the interfering signal. Scalar
    :param b_ch: Channels' -3 dB bandwidth [THz]. Array of size 1xNch
    :return: psi: the coefficient
    """
    b2 = np.abs(b2)

    if channel_index == interfering_index:  # The signal interferes with itself
        b_ch_sig = b_ch[channel_index]
        psi = np.arcsinh(0.5 * np.pi ** 2.0 * l_eff_a * b2 * b_ch_sig ** 2.0)
    else:
        f_sig = f_ch[channel_index]
        b_ch_sig = b_ch[channel_index]
        f_int = f_ch[interfering_index]
        b_ch_int = b_ch[interfering_index]
        del_f = f_sig - f_int
        psi = np.arcsinh(np.pi ** 2.0 * l_eff_a * b2 * b_ch_sig * (del_f + 0.5 * b_ch_int))
        psi -= np.arcsinh(np.pi ** 2.0 * l_eff_a * b2 * b_ch_sig * (del_f - 0.5 * b_ch_int))

    return psi


def analytic_formula(ind, b2, l_eff, l_eff_a, gam, f_ch, g_ch, b_ch, n_ch):
    """ analytic_formula computes the analytical formula.

    :param ind: index of the channel at which g_nli is computed. Scalar
    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_eff: Effective length in km. Scalar
    :param l_eff_a: Asymptotic effective length in km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param g_ch: Power spectral density W/THz. Array of size 1xNch
    :param b_ch: Channels' -3 dB bandwidth [THz]. Array of size 1xNch
    :param n_ch: Number of channels. Scalar
    :return: g_nli: power spectral density in W/THz of the nonlinear interference
    """
    ch_psd = g_ch[ind]
    b2 = abs(b2)

    g_nli = 0.0
    for n in np.arange(0, n_ch):
        psi = compute_psi(b2, l_eff_a, f_ch, ind, n, b_ch)
        g_nli += g_ch[n] * ch_psd ** 2.0 * psi

    g_nli *= (16.0 / 27.0) * (gam * l_eff) ** 2.0 / (2.0 * np.pi * b2 * l_eff_a)

    return g_nli


def gn_analytic(b2, l_span, a_db, gam, f_ch, b_ch, power, n_ch):
    """ gn_analytic computes the GN reference formula via analytical solution.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_span: Fiber Span length in km. Scalar
    :param a_db: Fiber loss coefficient in dB/km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param b_ch: Channels' -3 dB bandwidth [THz]. Array of size 1xNch
    :param power: Channels' power values in W. Array of size 1xNch
    :param n_ch: Number of channels. Scalar
    :return: g_nli: power spectral density in W/THz of the nonlinear interference at frequencies model_param['f_array']
    """
    g_ch = power / b_ch
    alpha_lin = a_db / 20.0 / np.log10(np.e)  # Conversion in linear units 1/km
    l_eff = (1.0 - np.exp(-2.0 * alpha_lin * l_span)) / (2.0 * alpha_lin)  # Effective length
    l_eff_a = 1.0 / (2.0 * alpha_lin)  # Asymptotic effective length
    g_nli = np.zeros(f_ch.size)
    for ind in np.arange(0, f_ch.size):
        g_nli[ind] = analytic_formula(ind, b2, l_eff, l_eff_a, gam, f_ch, g_ch, b_ch, n_ch)

    return g_nli


def get_f_computed_interp(f_ch, n_not_interp):
    """ get_f_computed_array returns the arrays containing the frequencies at which g_nli is computed and interpolated.

    :param f_ch: the overall frequency array. Array of size 1xnum_ch
    :param n_not_interp: the number of points at which g_nli has to be computed
    :return: f_nli_comp: the array containing the frequencies at which g_nli is computed
    :return: f_nli_interp: the array containing the frequencies at which g_nli is interpolated
    """
    num_ch = len(f_ch)
    if num_ch < n_not_interp:  # It's useless to compute g_nli in a number of points larger than num_ch
        n_not_interp = num_ch

    # Compute f_nli_comp
    n_not_interp_left = np.ceil((n_not_interp - 1.0) / 2.0)
    n_not_interp_right = np.floor((n_not_interp - 1.0) / 2.0)
    central_index = len(f_ch) // 2
    print(central_index)

    f_nli_central = np.array([f_ch[central_index]], copy=True)

    if n_not_interp_left > 0:
        index = np.linspace(0, central_index - 1, n_not_interp_left, dtype='int')
        f_nli_left = np.array(f_ch[index], copy=True)
    else:
        f_nli_left = np.array([])

    if n_not_interp_right > 0:
        index = np.linspace(-1, -central_index, n_not_interp_right, dtype='int')
        f_nli_right = np.array(f_ch[index], copy=True)
        f_nli_right = f_nli_right[::-1]  # Reverse the order of the array
    else:
        f_nli_right = np.array([])

    f_nli_comp = np.concatenate([f_nli_left, f_nli_central, f_nli_right])

    # Compute f_nli_interp
    f_ch_sorted = np.sort(f_ch)
    index = np.searchsorted(f_ch_sorted, f_nli_comp)

    f_nli_interp = np.array(f_ch, copy=True)
    f_nli_interp = np.delete(f_nli_interp, index)
    return f_nli_comp, f_nli_interp


def interpolate_in_range(x, y, x_new, kind_interp):
    """ Given some samples y of the function y(x), interpolate_in_range returns the interpolation of values y(x_new)

    :param x: The points at which y(x) is evaluated. Array
    :param y: The values of y(x). Array
    :param x_new: The values at which y(x) has to be interpolated. Array
    :param kind_interp: The interpolation method of the function scipy.interpolate.interp1d. String
    :return: y_new: the new interpolates samples
    """
    if x.size == 1:
        y_new = y * np.ones(x_new.size)
    elif x.size == 2:
        x = np.append(x, x_new[-1])
        y = np.append(y, y[-1])
        func = interp.interp1d(x, y, kind=kind_interp, bounds_error=False)
        y_new = func(x_new)
    else:
        func = interp.interp1d(x, y, kind=kind_interp, bounds_error=False)
        y_new = func(x_new)

    return y_new


def gn_model(spectrum_param, fiber_param, accuracy_param, n_cores):
    """ gn_model can compute the gn model both analytically or through the smart brute force
    integral.

    :param spectrum_param: Dictionary with spectrum parameters
                           spectrum_param['num_ch']: Number of channels. Scalar
                           spectrum_param['f_ch']: Baseband channels center frequencies in THz. Array of size 1xnum_ch
                           spectrum_param['b_ch']: Channels' -3 dB band [THz]. Array of size 1xnum_ch
                           spectrum_param['roll_off']: Channels' Roll-off factors [0,1). Array of size 1xnum_ch
                           spectrum_param['power']: Channels' power values in W. Array of size 1xnum_ch
    :param fiber_param: Dictionary with the parameters of the fiber
                        fiber_param['alpha']: Fiber loss coefficient in dB/km. Scalar
                        fiber_param['span_length']: Fiber Span length in km. Scalar
                        fiber_param['beta_2']: Fiber dispersion coefficient in ps/THz/km. Scalar
                        fiber_param['gamma']: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param accuracy_param: Dictionary with model parameters for accuracy tuning
                           accuracy_param['is_analytic']: A boolean indicating if you want to compute the NLI through
                           the analytic formula (is_analytic = True) of the smart brute force integration (is_analytic =
                           False). Boolean
                           accuracy_param['points_not_interp']: The number of NLI which will be calculated. Others are
                           interpolated
                           accuracy_param['kind_interp']: The kind of interpolation using the function
                           scipy.interpolate.interp1d
                           accuracy_param['th_fwm']: Minimum FWM efficiency value to be considered for high density
                           integration in dB
                           accuracy_param['n_points']: Maximum Number of integration points to be used in each frequency
                           slot of the spectrum
                           accuracy_param['n_points_min']: Minimum Number of integration points to be used in each
                           frequency
                           slot of the spectrum
    :return: g_nli_comp: the NLI power spectral density in W/THz computed through GN model
    :return: f_nli_comp: the frequencies at which g_nli_comp is evaluated
    :return: g_nli_interp: the NLI power spectral density in W/THz computed through interpolation of g_nli_comp
    :return: f_nli_interp: the frequencies at which g_nli_interp is estimated
    """
    # Take signal parameters
    num_ch = spectrum_param['num_ch']
    f_ch = spectrum_param['f_ch']
    b_ch = spectrum_param['b_ch']
    roll_off = spectrum_param['roll_off']
    power = spectrum_param['power']

    # Take fiber parameters
    a_db = fiber_param['alpha']
    l_span = fiber_param['span_length']
    beta2 = fiber_param['beta_2']
    gam = fiber_param['gamma']

    # Take accuracy parameters
    is_analytic = accuracy_param['is_analytic']
    n_not_interp = accuracy_param['points_not_interp']
    kind_interp = accuracy_param['kind_interp']
    th_fwm = accuracy_param['th_fwm']
    n_points = accuracy_param['n_points']
    n_points_min = accuracy_param['n_points_min']

    # Computing NLI
    if is_analytic:  # Analytic solution
        g_nli_comp = gn_analytic(beta2, l_span, a_db, gam, f_ch, b_ch, power, num_ch)
        f_nli_comp = np.copy(f_ch)
        g_nli_interp = []
        f_nli_interp = []
    else:  # Smart brute force integration
        f_nli_comp, f_nli_interp = get_f_computed_interp(f_ch, n_not_interp)

        model_param = {'min_FWM_inv': th_fwm, 'n_grid': n_points, 'n_grid_min': n_points_min,
                       'f_array': np.array(f_nli_comp, copy=True)}

        g_nli_comp = GN_integral(beta2, l_span, a_db, gam, f_ch, b_ch, roll_off, power, num_ch, model_param)

        # Interpolation
        g_nli_interp = interpolate_in_range(f_nli_comp, g_nli_comp, f_nli_interp, kind_interp)

    a_zero = fiber_param['alpha'] * fiber_param['span_length']
    a_tilting = fiber_param['alpha_1st'] * fiber_param['span_length']

    attenuation_db_comp = compute_attenuation_profile(a_zero, a_tilting, f_nli_comp)
    attenuation_lin_comp = 10 ** (-abs(attenuation_db_comp) / 10)

    g_nli_comp *= attenuation_lin_comp

    attenuation_db_interp = compute_attenuation_profile(a_zero, a_tilting, f_nli_interp)
    attenuation_lin_interp = 10 ** (-np.abs(attenuation_db_interp) / 10)

    g_nli_interp *= attenuation_lin_interp

    return g_nli_comp, f_nli_comp, g_nli_interp, f_nli_interp


def compute_gain_profile(gain_zero, gain_tilting, freq):
    """ compute_gain_profile evaluates the gain at the frequencies freq.

        :param gain_zero: the gain at f=0 in dB. Scalar
        :param gain_tilting: the gain tilt in dB/THz. Scalar
        :param freq: the baseband frequencies at which the gain profile is computed in THz. Array
        :return: gain: the gain profile in dB
        """
    gain = gain_zero + gain_tilting * freq
    return gain


def compute_ase_noise(noise_fig, gain, central_freq, freq):
    """ compute_ase_noise evaluates the ASE spectral density at the frequencies freq.

        :param noise_fig: the amplifier noise figure in dB. Scalar
        :param gain: the gain profile in dB at the frequencies contained in freq array. Array
        :param central_freq: the central frequency of the WDM comb. Scalar
        :param freq: the baseband frequencies at which the ASE noise is computed in THz. Array
        :return: g_ase: the ase noise profile
        """
    # the Planck constant in W/THz^2
    planck = (6.62607004 * 1e-34) * 1e24

    # Conversion from dB to linear
    gain_lin = np.power(10, gain / 10.0)
    noise_fig_lin = np.power(10, noise_fig / 10.0)

    g_ase = (gain_lin - 1) * noise_fig_lin * planck * (central_freq + freq)
    return g_ase


def compute_edfa_profile(gain_zero, gain_tilting, noise_fig, central_freq, freq):
    """ compute_edfa_profile evaluates the gain profile and the ASE spectral density at the frequencies freq.

        :param gain_zero: the gain at f=0 in dB. Scalar
        :param gain_tilting: the gain tilt in dB/THz. Scalar
        :param noise_fig: the amplifier noise figure in dB. Scalar
        :param central_freq: the central frequency of the WDM comb. Scalar
        :param freq: the baseband frequencies at which the ASE noise is computed in THz. Array
        :return: gain: the gain profile in dB
        :return: g_ase: the ase noise profile in W/THz
        """
    gain = compute_gain_profile(gain_zero, gain_tilting, freq)
    g_ase = compute_ase_noise(noise_fig, gain, central_freq, freq)

    return gain, g_ase


def compute_attenuation_profile(a_zero, a_tilting, freq):
    """compute_attenuation_profile returns the attenuation profile at the frequencies freq

    :param a_zero: the attenuation [dB] @ the baseband central frequency. Scalar
    :param a_tilting: the attenuation tilt in dB/THz. Scalar
    :param freq: the baseband frequencies at which attenuation is computed [THz]. Array
    :return: attenuation: the attenuation profile in dB
    """

    if len(freq):
        attenuation = a_zero + a_tilting * freq

        # abs in order to avoid ambiguity due to the sign convention
        attenuation = abs(attenuation)
    else:
        attenuation = []

    return attenuation


def passive_component(spectrum, a_zero, a_tilting, freq):
    """passive_component updates the input spectrum with the attenuation described by a_zero and a_tilting

    :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
    :param a_zero: attenuation at the central frequency [dB]. Scalar
    :param a_tilting: attenuation tilting [dB/THz]. Scalar
    :param freq: the baseband frequency of each WDM channel [THz]. Array
    :return: None
    """
    attenuation_db = compute_attenuation_profile(a_zero, a_tilting, freq)
    attenuation_lin = 10 ** np.divide(-abs(attenuation_db), 10.0)

    for index, s in enumerate(spectrum['signals']):
        spectrum['signals'][index]['p_ch'] *= attenuation_lin[index]
        spectrum['signals'][index]['p_nli'] *= attenuation_lin[index]
        spectrum['signals'][index]['p_ase'] *= attenuation_lin[index]

    return None


def optical_amplifier(spectrum, gain_zero, gain_tilting, noise_fig, central_freq, freq, b_eq):
    """optical_amplifier updates the input spectrum with the gain described by gain_zero and gain_tilting plus ASE noise

        :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
        :param gain_zero: gain at the central frequency [dB]. Scalar
        :param gain_tilting: gain tilting [dB/THz]. Scalar
        :param noise_fig: the noise figure of the amplifier [dB]. Scalar
        :param central_freq: the central frequency of the optical band [THz]. Scalar
        :param freq: the central frequency of each WDM channel [THz]. Array
        :param b_eq: the equivalent -3 dB bandwidth of each WDM channel [THZ]. Array
        :return: None
        """

    gain_db, g_ase = compute_edfa_profile(gain_zero, gain_tilting, noise_fig, central_freq, freq)

    p_ase = np.multiply(g_ase, b_eq)

    gain_lin = 10 ** np.divide(gain_db, 10.0)

    for index, s in enumerate(spectrum['signals']):
        spectrum['signals'][index]['p_ch'] *= gain_lin[index]
        spectrum['signals'][index]['p_nli'] *= gain_lin[index]
        spectrum['signals'][index]['p_ase'] *= gain_lin[index]
        spectrum['signals'][index]['p_ase'] += p_ase[index]

    return None


def fiber(spectrum, fiber_param, fiber_length, f_ch, b_ch, roll_off, control_param):
    """ fiber updates spectrum with the effects of the fiber

    :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
    :param fiber_param: Dictionary with the parameters of the fiber
                fiber_param['alpha']: Fiber loss coeffiecient in dB/km. Scalar
                fiber_param['beta_2']: Fiber dispersion coefficient in ps/THz/km. Scalar
                fiber_param['n_2']: second-order nonlinear refractive index [m^2/W]. Scalar
                fiber_param['a_eff']: the effective area of the fiber [um^2]. Scalar
    :param fiber_length: the span length [km]. Scalar
    :param f_ch: the baseband frequencies of the WDM channels [THz]. Scalar
    :param b_ch: the -3 dB bandwidth of each WDM channel [THz]. Array
    :param roll_off: the roll off of each WDM channel. Array
    :param control_param: Dictionary with the control parameters
                control_param['save_each_comp']: a boolean flag. If true, it saves in output folder one spectrum file at
                    the output of each component, otherwise it saves just the last spectrum. Boolean
                control_param['is_linear']: a bool flag. If true, is doesn't compute NLI, if false, OLE will consider
                 NLI. Boolean
                control_param['is_analytic']:  a boolean flag. If true, the NLI is computed through the analytic
                    formula, otherwise it uses the double integral. Warning: the double integral is very slow. Boolean
                control_param['points_not_interp']: if the double integral is used, it indicates how much points are
                    calculated, others will be interpolated. Scalar
                control_param['kind_interp']:  the interpolation method when double integral is used. String
                control_param['th_fwm']: he threshold of the four wave mixing efficiency for the double integral. Scalar
                control_param['n_points']: number of points in the high FWM efficiency region in which the double
                    integral is computed. Scalar
                control_param['n_points_min']:  number of points in which the double integral is computed in the low FWM
                    efficiency region. Scalar
                control_param['n_cores']: number of cores for parallel computation [not yet implemented]. Scalar
    :return: None
    """

    n_cores = control_param['n_cores']

    # Evaluation of NLI
    if not control_param['is_linear']:
        num_ch = len(spectrum['signals'])
        spectrum_param = {
            'num_ch': num_ch,
            'f_ch': f_ch,
            'b_ch': b_ch,
            'roll_off': roll_off
        }

        p_ch = np.zeros(num_ch)
        for index, signal in enumerate(spectrum['signals']):
            p_ch[index] = signal['p_ch']

        spectrum_param['power'] = p_ch
        fiber_param['span_length'] = fiber_length

        nli_cmp, f_nli_cmp, nli_int, f_nli_int = gn_model(spectrum_param, fiber_param, control_param, n_cores)
        f_nli = np.concatenate((f_nli_cmp, f_nli_int))
        order = np.argsort(f_nli)
        g_nli = np.concatenate((nli_cmp, nli_int))
        g_nli = np.array(g_nli)[order]

        p_nli = np.multiply(g_nli, b_ch)

    a_zero = fiber_param['alpha'] * fiber_length
    a_tilting = fiber_param['alpha_1st'] * fiber_length

    # Apply attenuation
    passive_component(spectrum, a_zero, a_tilting, f_ch)

    # Apply NLI
    if not control_param['is_linear']:
        for index, s in enumerate(spectrum['signals']):
            spectrum['signals'][index]['p_nli'] += p_nli[index]

    return None


def get_frequencies_wdm(spectrum, sys_param):
    """ the function computes the central frequency of the WDM comb and the frequency of each channel.

    :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
    :param sys_param: a dictionary containing the system parameters:
                'f0': the starting frequency, i.e the frequency of the first spectral slot [THz]
                'ns': the number of spectral slots. The space between two slots is 6.25 GHz
    :return: f_cent: the central frequency of the WDM comb [THz]
    :return: f_ch: the baseband frequency of each WDM channel [THz]
    """

    delta_f = 6.25E-3
    # Evaluate the central frequency
    f0 = sys_param['f0']
    ns = sys_param['ns']

    f_cent = f0 + ((ns // 2.0) * delta_f)

    # Evaluate the baseband frequencies
    n_ch = spectrum['laser_position'].count(1)
    f_ch = np.zeros(n_ch)
    count = 0
    for index, bool_laser in enumerate(spectrum['laser_position']):
        if bool_laser:
            f_ch[count] = (f0 - f_cent) + delta_f * index
            count += 1

    return f_cent, f_ch


def get_spectrum_param(spectrum):
    """ the function returns the number of WDM channels and 3 arrays containing the power, the equivalent bandwidth
     and the roll off of each WDM channel.

    :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
    :return: power: the power of each WDM channel [W]
    :return: b_eq: the equivalent bandwidth of each WDM channel [THz]
    :return: roll_off: the roll off of each WDM channel
    :return: p_ase: the power of the ASE noise [W]
    :return: p_nli: the power of NLI [W]
    :return: n_ch: the number of WDM channels
    """

    n_ch = spectrum['laser_position'].count(1)
    roll_off = np.zeros(n_ch)
    b_eq = np.zeros(n_ch)
    power = np.zeros(n_ch)
    p_ase = np.zeros(n_ch)
    p_nli = np.zeros(n_ch)
    for index, signal in enumerate(spectrum['signals']):
        b_eq[index] = signal['b_ch']
        roll_off[index] = signal['roll_off']
        power[index] = signal['p_ch']
        p_ase[index] = signal['p_ase']
        p_nli[index] = signal['p_nli']

    return power, b_eq, roll_off, p_ase, p_nli, n_ch


def change_component_ref(f_ref, link, fibers):
    """ it updates the reference frequency of OA gain, PC attenuation and fiber attenuation coefficient

    :param f_ref: the new reference frequency [THz]. Scalar
    :param link: the link structure. A list in which each element indicates one link component (PC, OA or fiber). List
    :param fibers: a dictionary containing the description of each fiber type. Dictionary
    :return: None
    """

    light_speed = 3e8       # [m/s]

    # Change reference to the central frequency f_cent for OA and PC
    for index, component in enumerate(link):
        if component['comp_cat'] is 'PC':

            old_loss = component['loss']
            delta_loss = component['loss_tlt']
            old_ref = component['ref_freq']
            new_loss = old_loss + delta_loss * (f_ref - old_ref)

            link[index]['ref_freq'] = f_ref
            link[index]['loss'] = new_loss

        elif component['comp_cat'] is 'OA':

            old_gain = component['gain']
            delta_gain = component['gain_tlt']
            old_ref = component['ref_freq']
            new_gain = old_gain + delta_gain * (f_ref - old_ref)

            link[index]['ref_freq'] = f_ref
            link[index]['gain'] = new_gain

        elif not component['comp_cat'] is 'fiber':

            error_string = 'Error in link structure: the ' + str(index+1) + '-th component have unknown category \n'\
                + 'allowed values are (case sensitive): PC, OA and fiber'
            print(error_string)

    # Change reference to the central frequency f_cent for fiber
    for fib_type in fibers:
        old_ref = fibers[fib_type]['reference_frequency']
        old_alpha = fibers[fib_type]['alpha']
        alpha_1st = fibers[fib_type]['alpha_1st']
        new_alpha = old_alpha + alpha_1st * (f_ref - old_ref)

        fibers[fib_type]['reference_frequency'] = f_ref
        fibers[fib_type]['alpha'] = new_alpha

        fibers[fib_type]['gamma'] = (2 * np.pi) * (f_ref / light_speed) * \
                                    (fibers[fib_type]['n_2'] / fibers[fib_type]['a_eff']) * 1e27

    return None


def compute_and_save_osnr(spectrum, flag_save=False, file_name='00', output_path='./output/'):
    """ Given the spectrum structure, the function returns the linear and non linear OSNR. If the boolean variable
    flag_save is true, the function also saves the osnr values for the central channel, the osnr for each channel and
    spectrum in a file with the name file_name, in the folder indicated by output_path

    :param spectrum: the spectrum dictionary containing the laser position (a list of boolean) and the list signals,
        which is a list of dictionaries (one for each channel) containing:
            'b_ch': the -3 dB bandwidth of the signal [THz]
            'roll_off': the roll off of the signal
            'p_ch': the signal power [W]
            'p_nli': the equivalent nli power [W]
            'p_ase': the ASE noise [W]
    :param flag_save: if True it saves all the data, otherwise it doesn't
    :param file_name: the name of the file in which the variables are saved
    :param output_path: the path in which you want to save the file
    :return: osnr_lin_db: the linear OSNR [dB]
    :return: osnr_nli_db: the non-linear equivalent OSNR (in linear units, NOT in [dB]
    """

    # Get the parameters from spectrum
    p_ch, b_eq, roll_off, p_ase, p_nli, n_ch = get_spectrum_param(spectrum)

    # Compute the linear OSNR
    if (p_ase == 0).any():
        osnr_lin = np.zeros(n_ch)
        for index, p_noise in enumerate(p_ase):
            if p_noise == 0:
                osnr_lin[index] = float('inf')
            else:
                osnr_lin[index] = p_ch[index] / p_noise

    else:
        osnr_lin = np.divide(p_ch, p_ase)

    # Compute the non-linear OSNR
    if ((p_ase + p_nli) == 0).any():
        osnr_nli = np.zeros(n_ch)
        for index, p_noise in enumerate(p_ase + p_nli):

            if p_noise == 0:
                osnr_nli[index] = float('inf')
            else:
                osnr_nli[index] = p_ch[index] / p_noise

    osnr_nli = np.divide(p_ch, p_ase + p_nli)

    # Compute linear and non linear OSNR for the central channel
    ind_c = n_ch // 2
    osnr_lin_central_channel_db = 10 * np.log10(osnr_lin[ind_c])
    osnr_nl_central_channel_db = 10 * np.log10(osnr_nli[ind_c])

    # Conversion in dB
    osnr_lin_db = 10 * np.log10(osnr_lin)
    osnr_nli_db = 10 * np.log10(osnr_nli)

    # Save spectrum, the non linear OSNR and the linear OSNR
    out_fle_name = output_path + file_name

    if flag_save:

        f = open(out_fle_name, 'w')
        f.write(''.join(('# Output parameters. The values of OSNR are evaluated in the -3 dB channel band', '\n\n')))
        f.write(''.join(('osnr_lin_central_channel_db = ', str(osnr_lin_central_channel_db), '\n\n')))
        f.write(''.join(('osnr_nl_central_channel_db = ', str(osnr_nl_central_channel_db), '\n\n')))
        f.write(''.join(('osnr_lin_db = ', str(osnr_lin_db), '\n\n')))
        f.write(''.join(('osnr_nl_db = ', str(osnr_nli_db), '\n\n')))
        f.write(''.join(('spectrum = ', str(spectrum), '\n')))

        f.close()

    return osnr_nli_db, osnr_lin_db


def ole(spectrum, link, fibers, sys_param, control_param, output_path='./output/'):
    """ The function takes the input spectrum, the link description, the fiber description, the system parameters,
    the control parameters and a string describing the destination folder of the output files. After the function is
    executed the spectrum is updated with all the impairments of the link. The function also returns the linear and
    non linear OSNR, computed in the equivalent bandwidth.

    :param spectrum: the spectrum dictionary containing the laser position (a list of boolean) and the list signals,
        which is a list of dictionaries (one for each channel) containing:
            'b_ch': the -3 dB bandwidth of the signal [THz]
            'roll_off': the roll off of the signal
            'p_ch': the signal power [W]
            'p_nli': the equivalent nli power [W]
            'p_ase': the ASE noise [W]
    :param link: the link structure. A list in which each element is a dictionary and it indicates one link component
        (PC, OA or fiber). List
    :param fibers: fibers is a dictionary containing a dictionary for each kind of fiber. Each dictionary has to report:
        reference_frequency: the frequency at which the parameters are evaluated [THz]
        alpha: the attenuation coefficient [dB/km]
        alpha_1st: the first derivative of alpha indicating the alpha slope [dB/km/THz]
                if you assume a flat attenuation with respect to the frequency you put it as zero
        beta_2: the dispersion coefficient [ps^2/km]
        n_2: second-order nonlinear refractive index [m^2/W]
            a typical value is 2.5E-20 m^2/W
        a_eff: the effective area of the fiber [um^2]
    :param sys_param: a dictionary containing the general system parameters:
            f0: the starting frequency of the laser grid used to describe the WDM system
            ns: the number of 6.25 GHz slots in the grid
    :param control_param: a dictionary containing the following parameters:
            save_each_comp: a boolean flag. If true, it saves in output folder one spectrum file at the output of each
                            component, otherwise it saves just the last spectrum
            is_linear: a bool flag. If true, is doesn't compute NLI, if false, OLE will consider NLI
            is_analytic: a boolean flag. If true, the NLI is computed through the analytic formula, otherwise it uses
                the double integral. Warning: the double integral is very slow.
            points_not_interp: if the double integral is used, it indicates how much points are calculated, others will
                be interpolated
            kind_interp: a string indicating the interpolation method for the double integral
            th_fwm: the threshold of the four wave mixing efficiency for the double integral
            n_points: number of points in which the double integral is computed in the high FWM efficiency region
            n_points_min: number of points in which the double integral is computed in the low FWM efficiency region
            n_cores: number of cores for parallel computation [not yet implemented]
    :param output_path: the path in which the output files are saved. String
    :return: osnr_nli_db: an array containing the non-linear OSNR [dB], one value for each WDM channel. Array
    :return: osnr_lin_db: an array containing the linear OSNR [dB], one value for each WDM channel. Array
    """

    # Take control parameters
    flag_save_each_comp = control_param['save_each_comp']

    # Evaluate frequency parameters
    f_cent, f_ch = get_frequencies_wdm(spectrum, sys_param)

    # Evaluate spectrum parameters
    power, b_eq, roll_off, p_ase, p_nli, n_ch = get_spectrum_param(spectrum)

    # Change reference to the central frequency f_cent for OA, PC and fibers
    change_component_ref(f_cent, link, fibers)

    # Emulate the link
    for component in link:
        if component['comp_cat'] is 'PC':
            a_zero = component['loss']
            a_tilting = component['loss_tlt']

            passive_component(spectrum, a_zero, a_tilting, f_ch)

        elif component['comp_cat'] is 'OA':
            gain_zero = component['gain']
            gain_tilting = component['gain_tlt']
            noise_fig = component['noise_figure']

            optical_amplifier(spectrum, gain_zero, gain_tilting, noise_fig, f_cent, f_ch, b_eq)

        elif component['comp_cat'] is 'fiber':
            fiber_type = component['fiber_type']
            fiber_param = fibers[fiber_type]
            fiber_length = component['length']

            fiber(spectrum, fiber_param, fiber_length, f_ch, b_eq, roll_off, control_param)

        else:
            error_string = 'Error in link structure: the ' + component['comp_cat'] + ' category is unknown \n' \
                           + 'allowed values are (case sensitive): PC, OA and fiber'
            print(error_string)

        if flag_save_each_comp:
            f_name = 'Output from component ID #' + component['comp_id']
            osnr_nli_db, osnr_lin_db = \
                compute_and_save_osnr(spectrum, flag_save=True, file_name=f_name, output_path=output_path)

    osnr_nli_db, osnr_lin_db = \
        compute_and_save_osnr(spectrum, flag_save=True, file_name='link_output', output_path=output_path)

    return osnr_nli_db, osnr_lin_db
