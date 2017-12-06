#!/usr/bin/env python3

from core.node import Node
from core.units import UNITS


class Transceiver(Node):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, spectral_info):
        return spectral_info


class Fiber(Node):
    def __init__(self, config):
        super().__init__(config)
        metadata = self.config.metadata
        self.length = metadata.length * UNITS[metadata.units]

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

    def propagate(self, *carriers):
        for carrier in carriers:
            power = carrier.power
            power = power._replace(signal=0.5 * power.signal * .5,
                                   nonlinear_interference=2 * power.nli,
                                   amplified_spontaneous_emission=2 * power.ase)
            yield carrier._replace(power=power)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        return spectral_info.update(carriers=carriers)
