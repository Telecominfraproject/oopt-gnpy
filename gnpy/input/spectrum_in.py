# coding=utf-8
""" spectrum_in.py describes the input spectrum of OLE, i.e. spectrum.
    spectrum is a dictionary containing two fields:
        laser_position: a list of bool indicating if a laser is turned on or not
        signals: a list of dictionaries each of them, describing one channel in the WDM comb

        The laser_position is defined respect to a frequency grid of 6.25 GHz space and the first slot is at the
        frequency described by the variable f0 in the dictionary sys_param in the file "general_parameters.py"

        Each dictionary element of the list 'signals' describes the profile of a WDM channel:
            b_ch: the -3 dB channel bandwidth (for a root raised cosine, it is equal to the symbol rate)
            roll_off: the roll off parameter of the root raised cosine shape
            p_ch: the channel power [W]
            p_nli: power of accumulated NLI in b_ch [W]
            p_ase: power of accumulated ASE noise in b_ch [W]
"""

n_ch = 41

spectrum = {
    'laser_position': [1, 0, 0, 0, 0, 0, 0, 0] * n_ch,
    'signals': [{
        'b_ch': 0.032,
        'roll_off': 0.15,
        'p_ch': 1E-3,
        'p_nli': 0,
        'p_ase': 0
        } for _ in range(n_ch)]
}
