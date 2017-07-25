import gnpy
import json
import networkx as nx


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
    def load_network(filepath):
        config = Utils.read_config(filepath)
        g = nx.Graph()
        for elem in config['elements']:
            ne = getattr(gnpy, elem['type'].capitalize())
            g.add_node(ne(**elem))
        return config, g

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
