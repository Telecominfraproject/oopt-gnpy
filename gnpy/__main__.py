import gnpy as gn
import numpy as np
import matplotlib.pyplot as plt
import time


def main():

    # Accuracy parameters
    flag_analytic = True
    num_computed_values = 2
    interp_method = 'linear'
    threshold_fwm = 50
    n_points = 500
    n_points_min = 4
    accuracy_param = {'is_analytic': flag_analytic, 'n_not_interp': num_computed_values, 'kind_interp': interp_method,
                      'th_fwm': threshold_fwm, 'n_points': n_points, 'n_points_min': n_points_min}

    # Parallelization Parameters
    n_cores = 1

    # Spectrum parameters
    num_ch = 95
    rs = np.ones(num_ch) * 0.032
    roll_off = np.ones(num_ch) * 0.05
    power = np.ones(num_ch) * 0.001
    central_freq = 193.5
    if num_ch % 2 == 1:  # odd number of channels
        fch = np.arange(-(num_ch // 2), (num_ch // 2) + 1, 1) * 0.05  # noqa: E501
    else:
        fch = (np.arange(0, num_ch) - (num_ch / 2.0) + 0.5) * 0.05
    spectrum_param = {'num_ch': num_ch, 'f_ch': fch, 'rs': rs, 'roll_off': roll_off, 'power': power}

    # Fiber Parameters
    beta2 = 21.27
    l_span = 100.0
    loss = 0.2
    gam = 1.27
    fiber_param = {'a_db': loss, 'span_length': l_span, 'beta2': beta2, 'gamma': gam}

    # EDFA Parameters
    noise_fig = 5.5
    gain_zero = 25.0
    gain_tilting = 0.5

    # Compute the GN model
    t = time.time()
    nli_cmp, f_nli_cmp, nli_int, f_nli_int = gn.gn_model(spectrum_param, fiber_param, accuracy_param, n_cores)  # noqa: E501
    print('Elapsed: %s' % (time.time() - t))

    # Compute the EDFA profile
    gain, g_ase = gn.compute_edfa_profile(gain_zero, gain_tilting, noise_fig, central_freq, fch)

    # Compute the raised cosine comb
    f1_array = np.linspace(np.amin(fch), np.amax(fch), 1e3)
    gtx = gn.raised_cosine_comb(f1_array, rs, roll_off, fch, power)
    gtx = gtx + 10 ** -6  # To avoid log10 issues.

    # Plot the results
    plt.figure(1)
    plt.plot(f1_array, 10 * np.log10(gtx), '-b', label='WDM comb')
    plt.plot(f_nli_cmp, 10 * np.log10(nli_cmp), 'ro', label='GNLI computed')
    plt.plot(f_nli_int, 10 * np.log10(nli_int), 'g+', label='GNLI interpolated')
    plt.plot(fch, 10 * np.log10(g_ase), 'yo', label='GASE')
    plt.ylabel('PSD [dB(W/THz)]')
    plt.xlabel('f [THz]')
    plt.legend(loc='upper left')
    plt.grid()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    main()
