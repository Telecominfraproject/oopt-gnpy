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

Link = [{
    'comp_cat': 'PC',
    'comp_id': '01',
    'ref_freq': 193.5e3,
    'loss': 2.0,
    'loss_tlt': 0.0
    },
    {
    'comp_cat': 'OA',
    'comp_id': '02',
    'ref_freq': 193.5e3,
    'gain': 20,
    'gain_tlt': 0.5e-3,
    'noise_figure': 5
    },
    {
    'comp_cat': 'fiber',
    'comp_id': '03',
    'fiber_type': 'SMF',
    'length': 100
    },
    {
    'comp_cat': 'PC',
    'comp_id': '04',
    'ref_freq': 193.5e-3,
    'loss': 2.0,
    'loss_tlt': 0.0
    },
    {
    'comp_cat': 'OA',
    'comp_id': '05',
    'ref_freq': 193.5e3,
    'gain': 20,
    'gain_tlt': 0.5e-3,
    'noise_figure': 5
    },
    {
    'comp_cat': 'fiber',
    'comp_id': '06',
    'fiber_type': 'NZDF',
    'length': 80
    },
    {
    'comp_cat': 'OA',
    'comp_id': '07',
    'ref_freq': 193.5e3,
    'gain': 20,
    'gain_tlt': 0.5e-3,
    'noise_figure': 5
    },
    {
    'comp_cat': 'PC',
    'comp_id': '08',
    'ref_freq': 193.5e3,
    'loss': 2.0,
    'loss_tlt': 0.0
    }
]
