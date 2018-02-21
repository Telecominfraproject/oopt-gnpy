#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from gnpy.core.utils import (load_json,
                             itufs,
                             freq2wavelength,
                             lin2db,
                             db2lin)
from gnpy.core import network

topology = load_json('edfa_example_network.json')
nw = network.network_from_json(topology)
pch2d_legend_data = np.loadtxt('Pchan2DLegend.txt')
pch2d = np.loadtxt('Pchan2D.txt')

ch_spacing = 0.05
fc = itufs(ch_spacing)
lc = freq2wavelength(fc) / 1000
nchan = np.arange(len(lc))
df = np.ones(len(lc)) * ch_spacing

edfa1 = [n for n in nw.nodes() if n.uid == 'Edfa1'][0]
edfa1.gain_target = 20.0
edfa1.tilt_target = -0.7
edfa1.calc_nf()

results = []
for Pin in pch2d:
    chgain = edfa1.gain_profile(Pin)
    pase = edfa1.noise_profile(chgain, fc, df)
    pout = lin2db(db2lin(Pin + chgain) + db2lin(pase))
    results.append(pout)

# Generate legend text

pch2d_legend = []
for ea in pch2d_legend_data:
    s = ''.join([chr(xx) for xx in ea.astype(dtype=int)]).strip()
    pch2d_legend.append(s)

# Plot
axis_font = {'fontname': 'Arial', 'size': '16', 'fontweight': 'bold'}
title_font = {'fontname': 'Arial', 'size': '17', 'fontweight': 'bold'}
tic_font = {'fontname': 'Arial', 'size': '12'}
plt.rcParams["font.family"] = "Arial"
plt.figure()
plt.plot(nchan, pch2d.T, '.-', lw=2)
plt.xlabel('Channel Number', **axis_font)
plt.ylabel('Channel Power [dBm]', **axis_font)
plt.title('Input Power Profiles for Different Channel Loading', **title_font)
plt.legend(pch2d_legend, loc=5)
plt.grid()
plt.ylim((-100, -10))
plt.xlim((0, 110))
plt.xticks(np.arange(0, 100, 10), **tic_font)
plt.yticks(np.arange(-110, -10, 10), **tic_font)
plt.figure()
for result in results:
    plt.plot(nchan, result, '.-', lw=2)
plt.title('Output Power w/ ASE for Different Channel Loading', **title_font)
plt.xlabel('Channel Number', **axis_font)
plt.ylabel('Channel Power [dBm]', **axis_font)
plt.grid()
plt.ylim((-50, 10))
plt.xlim((0, 100))
plt.xticks(np.arange(0, 100, 10), **tic_font)
plt.yticks(np.arange(-50, 10, 10), **tic_font)
plt.legend(pch2d_legend, loc=5)
plt.show()
