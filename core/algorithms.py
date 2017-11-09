#!/usr/bin/env python3

from networkx import all_simple_paths

from .compute import propagate

def all_paths(network, source, sink, spectral_info):
    for path in all_simple_paths(network, source, sink):
        yield propagate(path, spectral_info)

def osnr(carrier):
    return carrier.power.signal / (carrier.power.nli + carrier.power.ase)

def path_closes(osnr):
    def pred(result):
        _, _, out_si = result[-1]
        return all(osnr(carrier) < osnr for carrier in out_si)
    return pred

def closed_paths(network, source, sink, spectral_info, pred=path_closes(osnr=1e-3)):
    for result in all_paths(network, source, sink, spectral_info,):
        result = list(result)
        if path_closes(result):
            yield result
