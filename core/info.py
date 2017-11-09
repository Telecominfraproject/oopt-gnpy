#!/usr/bin/env python3

from collections import namedtuple

class Power(namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')):
    # convenient access
    nli = property(lambda self: self.nonlinear_interference)
    ase = property(lambda self: self.amplified_spontaneous_emission)

class Carrier(namedtuple('Carrier', 'channel_number frequency modulation baud_rate alpha power')):
    # convenient access
    ch = channel = property(lambda self: self.channel_number)
    ffs = freq = property(lambda self: self.frequency)

class SpectralInformation(namedtuple('SpectralInformation', 'carriers')):
    def __new__(cls, *carriers):
        return super().__new__(cls, carriers)

if __name__ == '__main__':
    si = SpectralInformation(
        Carrier(1, 193.95e12, '16-qam', 32e9, 0,  # 193.95 THz, 32 Gbaud
            Power(1e-3, 1e-6, 1e-6)),             # 1 mW, 1uW, 1uW
        Carrier(1, 195.95e12, '16-qam', 32e9, 0,  # 195.95 THz, 32 Gbaud
            Power(1.2e-3, 1e-6, 1e-6)),           # 1.2 mW, 1uW, 1uW
    )
    print(f'si = {si}')
