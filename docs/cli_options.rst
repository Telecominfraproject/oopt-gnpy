.. _cli-options:

Options Documentation for `gnpy-path-request` and `gnpy-transmission-example`
=============================================================================

Common options
--------------

**Option**: `--no-insert-edfas`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Disables the automatic insertion of EDFAs after ROADMs and fibers, as well as the splitting
of fibers during the auto-design process.

The `--no-insert-edfas` option is a command-line argument available in GNPy that allows users to control the
automatic insertion of amplifiers during the network design process. This option provides flexibility for
users who may want to manually manage amplifier placements or who have specific design requirements that
do not necessitate automatic amplification.

To use the `--no-insert-edfas` option, simply include it in the command line when running your GNPy program. For example:

.. code-block:: shell-session

  gnpy-transmission-example my_network.json --no-insert-edfas

When the `--no-insert-edfas` option is specified:

1. **No Automatic Amplifiers**: The program will not automatically add EDFAs to the network topology after
   ROADMs or fiber elements. This means that if the network design requires amplification, users must ensure
   that amplifiers are manually defined in the network topology file. Users should be aware that disabling
   automatic amplifier insertion may lead to insufficient amplification in the network if not managed properly.
   It is essential to ensure that the network topology includes the necessary amplifiers to meet performance requirements.

2. **No Fiber Splitting**: The option also prevents the automatic splitting of fibers during the design process.
   This is particularly useful for users who want to maintain specific fiber lengths or configurations without
   the program altering them.


**Option**: `--equipment`, `-e`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Specifies the equipment library file.

**Usage**: 

.. code-block:: shell-session

  gnpy-transmission-example my_network.json --equipment <FILE.json>

**Default**: Uses the default equipment configuration in the example-data folder if not specified.

**Functionality**: This option allows users to load a specific equipment configuration that defines the characteristics of the network elements.

**Option**: `--extra-equipment` and `--extra-config`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `--extra-equipment` and `--extra-config` options allow users to extend the default equipment library and configuration
settings used by the GNPy program. This feature is particularly useful for users who need to incorporate additional
equipment types or specific configurations that are not included in the standard equipment library (such as third party pluggables).

**Usage**:
  
.. code-block:: shell-session

  --extra-equipment <file1.json> [<file2.json> ...]

**Parameters**:

  - `<file1.json>`: Path to the first additional equipment file.
  - `<file2.json>`: Path to any subsequent additional equipment files (optional).

**Functionality**:

  - The program will merge the equipment definitions from the specified files into the main equipment library.
  - If an equipment type defined in the additional files has the same name as one in the main library, the program
    will issue a warning about the duplicate entry and will include ony the last definition.
  - This allows for flexibility in defining equipment that may be specific to certain use cases or vendor-specific models.

**`--extra-config`**:

**Description**: This option allows users to specify additional configuration files that can override
  or extend the default configuration settings used by the program. This is useful for customizing simulation
  parameters or equipment settings. To set an amplifier with a specific such config, it must be defined in the
  library with the keyword "default_config_from_json" filled with the file name containing the config in the case of
  "variable_gain" amplifier or with the "advanced_config_from_json" for the "advanced_model" amplifier.

**Usage**:
  
.. code-block:: shell-session

  --extra-config <file1.json> [<file2.json> ...]

**Parameters**:
  - `<file1.json>`: Path to the first additional configuration file.
  - `<file2.json>`: Path to any subsequent additional configuration files (optional).

**Functionality**:
  The program will load the configurations from the specified files and consider them instead of the
  default configurations for the amplifiers that use the "default_config_from_json" or "advanced_config_from_json" keywords.

Example
-------
To run the program with additional equipment and configuration files, you can use the following command:

.. code-block:: shell-session

gnpy-transmission-example --equipment main_equipment.json \
                          --extra-equipment additional_equipment1.json additional_equipment2.json \
                          --extra-config additional_config1.json


In this example:
- `main_equipment.json` is the primary equipment file.
- `additional_equipment1.json` and `additional_equipment2.json` are additional equipment files that will be merged into the main library.
- `additional_config1.json` is an additional configuration file that will override the default settings for the amplifiers pointing to it.


**Option**: `--save-network`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Saves the final network configuration to a specified JSON file.

**Usage**:

.. code-block:: shell-session

  --save-network <FILE.json>

**Functionality**: This option allows users to save the network state after the simulation, which can be useful for future reference or analysis.


**Option**: `--save-network-before-autodesign`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Dumps the network into a JSON file prior to autodesign.

**Usage**:

.. code-block:: shell-session

  gnpy-path-request my_network.json my_services.json --save-network-before-autodesign <FILE.json>

**Functionality**: This option is useful for users who want to inspect the network configuration before any automatic design adjustments are made.


**Option**: `--sim-params`
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Path to the JSON file containing simulation parameters.

**Usage**:

.. code-block:: shell-session

  gnpy-transmission-example my_network.json --sim-params <FILE.json>

**Functionality**: The `--sim-params` option is a command-line argument available in GNPy that allows users to specify a
JSON file containing simulation parameters. This option is crucial for customizing the behavior of the simulation:
the file ``sim_params.json`` contains the tuning parameters used within both the ``gnpy.science_utils.RamanSolver`` and
the ``gnpy.science_utils.NliSolver`` for the evaluation of the Raman profile and the NLI generation, respectively.

The tuning of the parameters is detailed here: :ref:`json input sim-params<sim-params>`.


`gnpy-transmission-example` options
-----------------------------------

**Option**: `--show-channels`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Displays the final per-channel OSNR and GSNR summary.

**Usage**: 

.. code-block:: shell-session

  gnpy-transmission-example my_network.json --show-channels

**Functionality**: This option provides a summary of the optical signal-to-noise ratio (OSNR)
and generalized signal-to-noise ratio (GSNR) for each channel after the simulation.


**Option**: `-pl`, `--plot`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Generates plots of the results.

**Usage**: 

.. code-block:: shell-session

  gnpy-transmission-example my_network.json -pl
  
**Functionality**: This option allows users to visualize the results of the simulation through graphical plots.


**Option**: `-l`, `--list-nodes`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Lists all transceiver nodes in the network.

**Usage**: 

.. code-block:: shell-session

  gnpy-transmission-example my_network.json -l

**Functionality**: This option provides a quick way to view all transceiver nodes present in the network topology.

**Option**: `-po`, `--power`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Specifies the reference channel power in span in dBm.

**Usage**:

.. code-block:: shell-session

  gnpy-transmission-example my_network.json -po <value>

**Functionality**: This option allows users to set the input power level for the reference channel used in the simulation.
It replaces the value specified in the `SI` section of the equipment library (:ref:`power_dbm<spectral_info>`).


**Option**: `--spectrum`
~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Specifies a user-defined mixed rate spectrum JSON file for propagation.

**Usage**:

.. code-block:: shell-session

  gnpy-transmission-example my_network.json --spectrum <FILE.json>

**Functionality**: This option allows users to define a custom spectrum for the simulation, which can
include varying channel rates and configurations. More details here: :ref:`mixed-rate<mixed-rate>`.


Options for `path_requests_run`
-------------------------------

The `gnpy-path-request` script provides a simple path computation function that supports routing, transceiver mode selection, and spectrum assignment.

It supports include and disjoint constraints for the path computation, but does not provide any optimisation.
It requires two mandatory arguments: network file and service file (see :ref:`XLS files<excel-service-sheet>` or :ref:`JSON files<legacy-json>`).

The `gnpy-path-request` computes:

  - design network once and propagate the service requests on this design
  - computes performance of each request defined in the service file independently from each other, considering full load (based on the request settings),
  - assigns spectrum for each request according to the remaining spectrum, on a first arrived first served basis.
    Lack of spectrum leads to blocking, but performance estimation is still returned for information.


**Option**: `-bi`, `--bidir`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Indicates that all demands are bidirectional.

**Usage**:

.. code-block:: shell-session

  gnpy-path-request my_network.json my_service.json -e my_equipment.json -bi

**Functionality**: This option allows users to specify that the performance of the service requests should be
computed in both directions (source to destination and destination to source). This forces the 'bidirectional'
attribute to true in the service file, possibly affecting feasibility if one direction is not feasible.


**Option**: `-o`, `--output`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Stores computation results requests into a JSON or CSV file.

**Usage**: 

.. code-block:: shell-session

  gnpy-path-request my_network.json my_service.json -o <FILE.json|FILE.csv>

**Functionality**: This option allows users to save the results of the path requests into a specified output file
for further analysis.


**Option**: `--redesign-per-request`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Redesigns the network for each request using the request as the reference channel
(replaces the `SI` section of the equipment library with the request specifications).

**Usage**:
.. code-block:: shell-session

  gnpy-path-request my_network.json my_services.json --redesign-per-request

**Functionality**: This option enables checking different scenarios for design.
