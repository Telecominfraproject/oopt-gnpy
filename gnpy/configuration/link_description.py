# coding=utf-8
""" link_description.py contains the full description of that OLE has to emulate.
    It contains a list of dictionaries, following the structure of the link and each element of the list describes one
    component.

    'comp_cat': the kind of link component:
        PC: a passive component defined by a loss at a certain frequency and a loss tilt
        OA: an optical amplifier defined by a gain at a certain frequency, a gain tilt and a noise figure
        fiber: a span of fiber described by the type and the length
    'comp_id': is an id identifying the component. It has to be unique for each component!

    extra fields for PC:
        'ref_freq': the frequency at which the 'loss' parameter is evaluated [THz]
        'loss': the loss at the frequency 'ref_freq' [dB]
        'loss_tlt': the frequency dependent loss [dB/THz]
    extra fields for OA:
        'ref_freq': the frequency at which the 'gain' parameter is evaluated [THz]
        'gain': the gain at the frequency 'ref_freq' [dB]
        'gain_tlt': the frequency dependent gain [dB/THz]
        'noise_figure': the noise figure of the optical amplifier [dB]
    extra fields for fiber:
        'fiber_type': a string calling the type of fiber described in the file fiber_parameters.py
        'length': the fiber length [km]

"""
smf = {
    'comp_cat': 'fiber',
    'comp_id': '',
    'fiber_type': 'SMF',
    'length': 100
    }

oa = {
    'comp_cat': 'OA',
    'comp_id': '',
    'ref_freq': 193.5,
    'gain': 20,
    'gain_tlt': 0.0,
    'noise_figure': 5
    }

pc = {
    'comp_cat': 'PC',
    'comp_id': '04',
    'ref_freq': 193.,
    'loss': 2.0,
    'loss_tlt': 0.0
    }

link = []

for index in range(20):
    smf['comp_id'] = '%03d' % (2 * index)
    oa['comp_id'] = '%03d' % (2 * index + 1)
    link += [dict(smf)]
    link += [dict(oa)]

pc['comp_id'] = '%03d' % 40
link += [dict(pc)]
