import json
from gnpy.constants import c, h
import numpy as np


class Utils:

    @staticmethod
    def read_config(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print("File not found:", filepath)

    @staticmethod
    def find_by_node_id(g, nid):
        # ?ODO: What if nid is not found in graph (g)?
        return next((n for n in g.nodes() if n.id == nid), None)

    @staticmethod
    def dbkm_2_lin(loss_coef):
        """ calculates the linear loss coefficient
        """
        alpha_pcoef = loss_coef
        alpha_acoef = alpha_pcoef/(2*4.3429448190325184)
        s = 'alpha_pcoef is linear loss coefficient in [dB/km^-1] units'
        s = ''.join([s, "alpha_acoef is linear loss field amplitude \
                     coefficient in [km^-1] units"])
        d = {'alpha_pcoef': alpha_pcoef, 'alpha_acoef': alpha_acoef,
             'description:': s}
        return d

    @staticmethod
    def db_to_lin(val):
        return 10 ** (val / 10)

    @staticmethod
    def chan_osnr(chan_params, amp_params):
        in_osnr = Utils.db_to_lin(chan_params['osnr'])
        pin = Utils.db_to_lin(chan_params['power']) / 1e3
        nf = Utils.db_to_lin(amp_params.nf[0])
        ase_cont = nf * h * chan_params['frequency'] * 12.5 * 1e21
        ret = -10 * np.log10(1 / in_osnr + ase_cont / pin)
        return ret
