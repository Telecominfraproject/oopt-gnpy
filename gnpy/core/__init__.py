"""
Simulation of signal propagation in the DWDM network

Optical signals, as defined via :class:`.info.SpectralInformation`, enter
:py:mod:`.elements` which compute how these signals are affected as they travel
through the :py:mod:`.network`.
The simulation is controlled via :py:mod:`.parameters` and implemented mainly
via :py:mod:`.science_utils`.
"""
