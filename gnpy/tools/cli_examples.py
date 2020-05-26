#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.cli_examples
=======================

Common code for CLI examples
'''

import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.core.elements import RamanFiber
import gnpy.core.exceptions as exceptions
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.tools.json_io import load_equipment, load_network, load_json


def load_common_data(equipment_filename, topology_filename, simulation_filename=None, fuzzy_name_matching=False):
    '''Load common configuration from JSON files'''
    try:
        equipment = load_equipment(equipment_filename)
        network = load_network(topology_filename, equipment, fuzzy_name_matching)
        sim_params = SimParams(**load_json(simulation_filename)) if simulation_filename is not None else None
        if not sim_params:
            if next((node for node in network if isinstance(node, RamanFiber)), None) is not None:
                print(f'{ansi_escapes.red}Invocation error:{ansi_escapes.reset} '
                      f'RamanFiber requires passing simulation params via --sim-params')
                exit(1)
        else:
            Simulation.set_params(sim_params)
    except exceptions.EquipmentConfigError as e:
        print(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.ParametersError as e:
        print(f'{ansi_escapes.red}Simulation parameters error:{ansi_escapes.reset} {e}')
        exit(1)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
        exit(1)

    return (equipment, network)
