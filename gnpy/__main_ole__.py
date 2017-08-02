import os
import gnpy as gn
import numpy as np
import matplotlib.pyplot as plt
import time


def main_ole():

    # String indicating the folder in which outputs will be saved
    string_date_time = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
    output_path = './output/' + string_date_time + '/'

    # Creates the directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    from configuration.fiber_parameters import fibers
    from configuration.general_parameters import sys_param, control_param
    from configuration.link_description import link
    from input.spectrum_in import spectrum

    # adapt the laser position to the grid
    if len(spectrum['laser_position']) < sys_param['ns']:
        n = sys_param['ns'] - len(spectrum['laser_position'])
        missing_zeros = [0 for _ in range(n)]
        spectrum['laser_position'] += missing_zeros
    elif len(spectrum['laser_position']) > sys_param['ns']:
        print('Error: the spectrum definition requires a larger number of slots ns in the spectrum grid')

    delta_f = 6.25E-3
    f_0 = sys_param['f0']
    f_cent = f_0 + ((sys_param['ns'] // 2.0) * delta_f)

    n_ch = spectrum['laser_position'].count(1)
    # Get comb parameters
    f_ch = np.zeros(n_ch)
    count = 0
    for index, bool_laser in enumerate(spectrum['laser_position']):
        if bool_laser:
            f_ch[count] = delta_f * index + (f_0 - f_cent)
            count += 1

    t = time.time()
    # It runs the OLE
    osnr_nl_db, osnr_lin_db = gn.ole(spectrum, link, fibers, sys_param, control_param, output_path=output_path)
    print('Elapsed: %s' % (time.time() - t))

    # Compute the raised cosine comb
    power, rs, roll_off, p_ase, p_nli, n_ch = gn.get_spectrum_param(spectrum)
    f1_array = np.linspace(np.amin(f_ch), np.amax(f_ch), 1e3)
    gtx = gn.raised_cosine_comb(f1_array, rs, roll_off, f_ch, power)
    gtx = gtx + 10 ** -6  # To avoid log10 issues.


    # OSNR at in the central channel
    ind_c = n_ch // 2
    osnr_lin_central_db = osnr_lin_db[ind_c]
    osnr_nl_central_db = osnr_nl_db[ind_c]
    print('The linear OSNR in the central channel is: ' + str(osnr_lin_central_db) + ' dB')
    print('The non linear OSNR in the central channel is: ' + str(osnr_nl_central_db) + ' dB')

    # Plot the results
    plt.figure(1)
    plt.plot(f1_array, 10 * np.log10(gtx), '-b', label='WDM comb PSD [dB(W/THz)]')
    plt.plot(f_ch, 10 * np.log10(p_nli), 'ro', label='NLI [dBw]')
    plt.plot(f_ch, 10 * np.log10(p_ase), 'g+', label='ASE noise [dBw]')
    plt.ylabel('')
    plt.xlabel('f [THz]')
    plt.legend(loc='upper right')
    plt.grid()
    plt.draw()

    plt.figure(2)
    plt.plot(f_ch, osnr_nl_db, 'ro', label='non-linear OSNR')
    plt.plot(f_ch, osnr_lin_db, 'g+', label='linear OSNR')
    plt.ylabel('OSNR [dB]')
    plt.xlabel('f [THz]')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main_ole()

