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


def GN_integral(b2, Lspan, a_db, gam, f_ch, rs, roll_off, power, Nch, model_param):
    """ GN_integral computes the GN reference formula via smart brute force integration. The Gaussian Noise model is
    applied in its incoherent form (phased-array factor =1). The function computes the integral by columns: for each f1,
    a non-uniformly spaced f2 array is generated, and the integrand function is computed there. At the end of the loop
    on f1, the overall GNLI is computed. Accuracy can be tuned by operating on model_param argument.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param Lspan: Fiber Span length in km. Scalar
    :param a_db: Fiber loss coeffiecient in dB/km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param rs: Channels' Symbol Rates in TBaud. Array of size 1xNch
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
    fmax = (f_ch[-1] - (rs[-1] / 2.0)) - (f_ch[0] - (rs[0] / 2.0))  # Get frequency limit
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
        G1 = raised_cosine_comb(f1_array, rs, roll_off, f_ch, power)  # Get corresponding spectrum
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
                G2 = raised_cosine_comb(f2_array, rs, roll_off, f_ch, power)  # Get spectrum there
                f3_array = f1 + f2_array - f  # Compute f3
                G3 = raised_cosine_comb(f3_array, rs, roll_off, f_ch, power)  # Get spectrum over f3
                G = G2 * G3 * G1[f_ind]
                if np.count_nonzero(G):
                    FWM_eff = fwm_eff(alpha_lin, Lspan, b2, (f1 - f) * (f2_array - f))  # Compute FWM efficiency
                    Gpart[f_ind] = 2.0 * np.trapz(FWM_eff * G, f2_array)  # Compute inner integral
            f_ind += 1
            # Compute outer integral. Nominal span loss already compensated
        GNLI[f_ind_eval] = 16.0 / 27.0 * gam * gam * np.trapz(Gpart, f1_array)
        f_ind_eval += 1  # Next frequency
    return GNLI  # Return GNLI array in W/THz and the array of the corresponding frequencies


def compute_psi(b2, l_eff_a, f_ch, channel_index, interfering_index, rs):
    """ compute_psi computes the psi coefficient of the analytical formula.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_eff_a: Asymptotic effective length in km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param channel_index: Index of the channel. Scalar
    :param interfering_index: Index of the interfering signal. Scalar
    :param rs: Channels' Symbol Rates in TBaud. Array of size 1xNch
    :return: psi: the coefficient
    """
    b2 = np.abs(b2)

    if channel_index == interfering_index:  # The signal interfere with itself
        rs_sig = rs[channel_index]
        psi = np.arcsinh(0.5 * np.pi ** 2.0 * l_eff_a * b2 * rs_sig ** 2.0)
    else:
        f_sig = f_ch[channel_index]
        rs_sig = rs[channel_index]
        f_int = f_ch[interfering_index]
        rs_int = rs[interfering_index]
        del_f = f_sig - f_int
        psi = np.arcsinh(np.pi ** 2.0 * l_eff_a * b2 * rs_sig * (del_f + 0.5 * rs_int))
        psi -= np.arcsinh(np.pi ** 2.0 * l_eff_a * b2 * rs_sig * (del_f - 0.5 * rs_int))

    return psi


def analytic_formula(ind, b2, l_eff, l_eff_a, gam, f_ch, g_ch, rs, n_ch):
    """ analytic_formula computes the analytical formula.

    :param ind: index of the channel at which g_nli is computed. Scalar
    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_eff: Effective length in km. Scalar
    :param l_eff_a: Asymptotic effective length in km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param g_ch: Power spectral density W/THz. Array of size 1xNch
    :param rs: Channels' Symbol Rates in TBaud. Array of size 1xNch
    :param n_ch: Number of channels. Scalar
    :return: g_nli: power spectral density in W/THz of the nonlinear interference
    """
    ch_psd = g_ch[ind]
    b2 = abs(b2)

    g_nli = 0.0
    for n in np.arange(0, n_ch):
        psi = compute_psi(b2, l_eff_a, f_ch, ind, n, rs)
        g_nli += g_ch[n] * ch_psd ** 2.0 * psi

    g_nli *= (16.0 / 27.0) * (gam * l_eff) ** 2.0 / (2.0 * np.pi * b2 * l_eff_a)

    return g_nli


def gn_analytic(b2, l_span, a_db, gam, f_ch, rs, power, n_ch):
    """ gn_analytic computes the GN reference formula via analytical solution.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param l_span: Fiber Span length in km. Scalar
    :param a_db: Fiber loss coeffiecient in dB/km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param rs: Channels' Symbol Rates in TBaud. Array of size 1xNch
    :param power: Channels' power values in W. Array of size 1xNch
    :param n_ch: Number of channels. Scalar
    :return: g_nli: power spectral density in W/THz of the nonlinear interference at frequencies model_param['f_array']
    """
    g_ch = power / rs
    alpha_lin = a_db / 20.0 / np.log10(np.e)  # Conversion in linear units 1/km
    l_eff = (1.0 - np.exp(-2.0 * alpha_lin * l_span)) / (2.0 * alpha_lin)  # Effective length
    l_eff_a = 1.0 / (2.0 * alpha_lin)  # Asymptotic effective length
    g_nli = np.zeros(f_ch.size)
    for ind in np.arange(0, f_ch.size):
        g_nli[ind] = analytic_formula(ind, b2, l_eff, l_eff_a, gam, f_ch, g_ch, rs, n_ch)

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
                           spectrum_param['rs']: Channels' Symbol Rates in TBaud. Array of size 1xnum_ch
                           spectrum_param['roll_off']: Channels' Roll-off factors [0,1). Array of size 1xnum_ch
                           spectrum_param['power']: Channels' power values in W. Array of size 1xnum_ch
    :param fiber_param: Dictionary with the parameters of the fiber
                        fiber_param['a_db']: Fiber loss coeffiecient in dB/km. Scalar
                        fiber_param['span_length']: Fiber Span length in km. Scalar
                        fiber_param['beta2']: Fiber dispersion coefficient in ps/THz/km. Scalar
                        fiber_param['gamma']: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param accuracy_param: Dictionary with model parameters for accuracy tuning
                           accuracy_param['is_analytic']: A boolean indicating if you want to compute the NLI through
                           the analytic formula (is_analytic = True) of the smart brute force integration (is_analytic =
                           False). Boolean
                           accuracy_param['n_not_interp']: The number of NLI which will be calculated. Others are
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
    rs = spectrum_param['rs']
    roll_off = spectrum_param['roll_off']
    power = spectrum_param['power']

    # Take fiber parameters
    a_db = fiber_param['a_db']
    l_span = fiber_param['span_length']
    beta2 = fiber_param['beta2']
    gam = fiber_param['gamma']

    # Take accuracy parameters
    is_analytic = accuracy_param['is_analytic']
    n_not_interp = accuracy_param['n_not_interp']
    kind_interp = accuracy_param['kind_interp']
    th_fwm = accuracy_param['th_fwm']
    n_points = accuracy_param['n_points']
    n_points_min = accuracy_param['n_points_min']

    # Computing NLI
    if is_analytic:  # Analytic solution
        g_nli_comp = gn_analytic(beta2, l_span, a_db, gam, f_ch, rs, power, num_ch)
        f_nli_comp = np.copy(f_ch)
        g_nli_interp = []
        f_nli_interp = []
    else:  # Smart brute force integration
        f_nli_comp, f_nli_interp = get_f_computed_interp(f_ch, n_not_interp)

        model_param = {'min_FWM_inv': th_fwm, 'n_grid': n_points, 'n_grid_min': n_points_min,
                       'f_array': np.array(f_nli_comp, copy=True)}

        g_nli_comp = GN_integral(beta2, l_span, a_db, gam, f_ch, rs, roll_off, power, num_ch, model_param)

        # Interpolation
        g_nli_interp = interpolate_in_range(f_nli_comp, g_nli_comp, f_nli_interp, kind_interp)

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
    planck = 6.62607004 * 1e-34 * 1e24

    # Conversion from dB to linear
    gain_lin = np.power(10, gain / 10)
    noise_fig_lin = np.power(10, noise_fig / 10)

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
    gain = 10 * np.log10(gain)

    return gain, g_ase


def compute_attenuation_profile(a_zero, a_tilting, freq):
    """compute_attenuation_profile returns the attenuation profile at the frequencies freq

    :param a_zero: the attenuation [dB] @ the baseband central frequency. Scalar
    :param a_tilting: the attenuation tilt in dB/THz. Scalar
    :param freq: the baseband frequencies at which attenuation is computed [THz]. Array
    :return: attenuation: the attenuation profile in dB
    """

    attenuation = a_zero + a_tilting * freq

    return attenuation


def passive_component(spectrum, a_zero, a_tilting, freq):
    """passive_component updates the input spectrum with the attenuation described by a_zero and a_tilting

    :param spectrum: the WDM spectrum to be attenuated. List of dictionaries
    :param a_zero: attenuation at the central frequency [dB]. Scalar
    :param a_tilting: attenuation tilting [dB/THz]. Scalar
    :param freq: the central frequency of each WDM channel [THz]. Array
    :return: None
    """
    attenuation_db = compute_attenuation_profile(a_zero, a_tilting, freq)
    attenuation_lin = 10**(-abs(attenuation_db) / 10)

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
        :param b_eq: the equivalent bandwidth of each WDM channel [THZ]. Array
        :return: None
        """

    gain_db, g_ase = compute_edfa_profile(gain_zero, gain_tilting, noise_fig, central_freq, freq)

    p_ase = g_ase * b_eq

    gain_lin = 10**(gain_db / 10)

    for index, s in enumerate(spectrum['signals']):
        spectrum['signals'][index]['p_ch'] *= gain_lin[index]
        spectrum['signals'][index]['p_nli'] *= gain_lin[index]
        spectrum['signals'][index]['p_ase'] *= gain_lin[index]
        spectrum['signals'][index]['p_ase'] += p_ase[index]

    return None

