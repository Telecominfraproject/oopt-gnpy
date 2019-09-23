
How to prepare the Excel input file
-----------------------------------

`examples/transmission_main_example.py <examples/transmission_main_example.py>`_ gives the possibility to use an excel input file instead of a json file. The program then will generate the corresponding json file for you.

The file named 'meshTopologyExampleV2.xls' is an example.

In order to work the excel file MUST contain at least 2 sheets:

- Nodes
- Links

(In progress) The File MAY contain an additional sheet:

- Eqt
- Service

Nodes sheet
-----------

Nodes sheet contains nine columns.
Each line represents a 'node' (ROADM site or an in line amplifier site ILA or a Fused)::

  City (Mandatory) ; State ; Country ; Region ; Latitude ; Longitude ; Type

- **City** is used for the name of a node of the graph. It accepts letters, numbers,underscore,dash, blank... (not exhaustive). The user may want to avoid commas for future CSV exports.

   **City name MUST be unique** 

- **Type** is not mandatory. 

  - If not filled, it will be interpreted as an 'ILA' site if node degree is 2 and as a ROADM otherwise.
  - If filled, it can take "ROADM", "FUSED" or "ILA" values. If another string is used, it will be considered as not filled. FUSED means that ingress and egress spans will be fused together.  

- *State*, *Country*, *Region* are not mandatory.
  "Region" is a holdover from the CORONET topology reference file `CORONET_Global_Topology.xls <examples/CORONET_Global_Topology.xls>`_. CORONET separates its network into geographical regions (Europe, Asia, Continental US.) This information is not used by gnpy.

- *Longitude*, *Latitude* are not mandatory. If filled they should contain numbers.

- **Booster_restriction** and **Preamp_restriction** are not mandatory.
  If used, they must contain one or several amplifier type_variety names separated by ' | '. This information is used to restrict types of amplifiers used in a ROADM node during autodesign. If a ROADM booster or preamp is already specified in the Eqpt sheet , the field is ignored. The field is also ignored if the node is not a ROADM node.

**There MUST NOT be empty line(s) between two nodes lines**


Links sheet
-----------

Links sheet must contain sixteen columns::

                   <--           east cable from a to z                                   --> <--                  west from z to                                   -->
   NodeA ; NodeZ ; Distance km ; Fiber type ; Lineic att ; Con_in ; Con_out ; PMD ; Cable Id ; Distance km ; Fiber type ; Lineic att ; Con_in ; Con_out ; PMD ; Cable Id


Links sheets MUST contain all links between nodes defined in Nodes sheet.
Each line represents a 'bidir link' between two nodes. The two directions are represented on a single line with "east cable from a to z" fields and "west from z to a" fields. Values for 'a to z' may be different from values from 'z to a'. 
Since both direction of a bidir 'a-z' link are described on the same line (east and west), 'z to a' direction MUST NOT be repeated in a different line. If repeated, it will generate another parrallel bidir link between the same end nodes.


Parameters for "east cable from a to z" and "west from z to a" are detailed in 2x7 columns. If not filled, "west from z to a" is copied from "east cable from a to z".

For example, a line filled with::

  node6 ; node3 ; 80 ; SSMF ; 0.2 ; 0.5 ; 0.5 ; 0.1 ; cableB ;  ;  ; 0.21 ; 0.2 ;  ;  ;  

will generate a unidir fiber span from node6 to node3 with::
 
  [node6 node3 80 SSMF 0.2 0.5 0.5 0.1 cableB] 

and a fiber span from node3 to node6::

 [node6 node3 80 SSMF 0.21 0.2 0.5 0.1 cableB] attributes. 

- **NodeA** and **NodeZ** are Mandatory. 
  They are the two endpoints of the link. They MUST contain a node name from the **City** names listed in Nodes sheet.

- **Distance km** is not mandatory. 
  It is the link length.

  - If filled it MUST contain numbers. If empty it is replaced by a default "80" km value. 
  - If value is below 150 km, it is considered as a single (bidirectional) fiber span.
  - If value is over 150 km the `transmission_main_example.py <examples/transmission_main_example.py>`_ program will automatically suppose that intermediate span description are required and will generate fiber spans elements with "_1","_2", ... trailing strings which are not visible in the json output. The reason for the splitting is that current edfa usually do not support large span loss. The current assumption is that links larger than 150km will require intermediate amplification. This value will be revisited when Raman amplification is added”

- **Fiber type** is not mandatory. 

  If filled it must contain types listed in `eqpt_config.json <examples/eqpt_config.json>`_ in "Fiber" list "type_variety".
  If not filled it takes "SSMF" as default value.

- **Lineic att** is not mandatory. 

  It is the lineic attenuation expressed in dB/km.
  If filled it must contain positive numbers.
  If not filled it takes "0.2" dB/km value

- *Con_in*, *Con_out* are not mandatory. 

  They are the connector loss in dB at ingress and egress of the fiber spans.
  If filled they must contain positive numbers.
  If not filled they take "0.5" dB default value.

- *PMD* is not mandatory and and is not used yet. 

  It is the PMD value of the link in ps.
  If filled they must contain positive numbers.
  If not filled, it takes "0.1" ps value.

- *Cable Id* is not mandatory. 
  If filled they must contain strings with the same constraint as "City" names. Its value is used to differenate links having the same end points. In this case different Id should be used. Cable Ids are not meant to be unique in general.




(in progress)

Eqpt sheet 
----------

Eqt sheet is optional. It lists the amplifiers types and characteristics on each degree of the *Node A* line.
Eqpt sheet must contain twelve columns::

                   <--           east cable from a to z        --> <--        west from z to a                 -->
  Node A ; Node Z ; amp type ; att_in ; amp gain ; tilt ; att_out ; amp type ; att_in ; amp gain ; tilt ; att_out

If the sheet is present, it MUST have as many lines as egress directions of ROADMs defined in Links Sheet. 

For example, consider the following list of links (A,B and C being a ROADM and amp# ILAs)

::

  A    - amp1
  amp1 - amp2
  Amp2 - B
  A    - amp3
  amp3 - C

then Eqpt sheet should contain:
  - one line for each ILAs: amp1, amp2, amp3 
  - one line for each degree 1 ROADMs B and C  
  - two lines for ROADM A  which is a degree 2 ROADM 

::

  A    - amp1
  amp1 - amp2
  Amp2 - B
  A    - amp3
  amp3 - C
  B    - amp2
  C    - amp3


In case you already have filled Nodes and Links sheets `create_eqpt_sheet.py <examples/create_eqpt_sheet.py>`_  can be used to automatically create a template for the mandatory entries of the list.

.. code-block:: shell

    $ cd examples
    $ python create_eqpt_sheet.py meshTopologyExampleV2.xls

This generates a text file meshTopologyExampleV2_eqt_sheet.txt  whose content can be directly copied into the Eqt sheet of the excel file. The user then can fill the values in the rest of the columns.


- **Node A** is mandatory. It is the name of the node (as listed in Nodes sheet).
  If Node A is a 'ROADM' (Type attribute in sheet Node), its number of occurence must be equal to its degree.
  If Node A is an 'ILA' it should appear only once.

- **Node Z** is mandatory. It is the egress direction from the *Node A* site. Multiple Links between the same Node A and NodeZ is not supported.

- **amp type** is not mandatory. 
  If filled it must contain types listed in `eqpt_config.json <examples/eqpt_config.json>`_ in "Edfa" list "type_variety".
  If not filled it takes "std_medium_gain" as default value.
  If filled with fused, a fused element with 0.0 dB loss will be placed instead of an amplifier. This might be used to avoid booster amplifier on a ROADM direction.

- **amp_gain** is not mandatory. It is the value to be set on the amplifier (in dB).
  If not filled, it will be determined with design rules in the convert.py file.
  If filled, it must contain positive numbers.

- *att_in* and *att_out* are not mandatory and are not used yet. They are the value of the attenautor at input and output of amplifier (in dB).
  If filled they must contain positive numbers.

- *tilt* --TODO--

# to be completed #

(in progress)

Service sheet 
-------------

Service sheet is optional. It lists the services for which path and feasibility must be computed with path_requests_run.py.

Service sheet must contain 11 columns::  

   route id ; Source ; Destination ; TRX type ; Mode ; System: spacing ; System: input power (dBm) ; System: nb of channels ;  routing: disjoint from ; routing: path ; routing: is loose?

- **route id** is mandatory. It must be unique. It is the identifier of the request. It can be an integer or a string (do not  use blank or dash or coma)

- **Source** is mandatory. It is the name of the source node (as listed in Nodes sheet). Source MUST be a ROADM node. (TODO: relax this and accept trx entries)

- **Destination** is mandatory. It is the name of the destination node (as listed in Nodes sheet). Source MUST be a ROADM node. (TODO: relax this and accept trx entries)

- **TRX type** is mandatory. They are the variety type and selected mode of the transceiver to be used for the propagation simulation. These modes MUST be defined in the equipment library. The format of the mode is used as the name of the mode. (TODO: maybe add another  mode id on Transceiver library ?). In particular the mode selection defines the channel baudrate to be used for the propagation simulation.

- **mode** is optional. If not specified, the program will search for the mode of the defined transponder with the highest baudrate fitting within the spacing value. 

- **System: spacing** is mandatory. Spacing is the channel spacing defined in GHz difined for the feasibility propagation simulation, assuming system full load.

- **System: input power (dBm) ; System: nb of channels** are optional input defining the system parameters for the propagation simulation.

  - input power is the channel optical input power in dBm
  - nb of channels is the number of channels to be used for the simulation.

- **routing: disjoint from ; routing: path ; routing: is loose?** are optional.

  - disjoint from: identifies the requests from which this request must be disjoint. If filled it must contain request ids separated by ' | ' 
  - path: is the set of ROADM nodes that must be used by the path. It must contain the list of ROADM names that the path must cross. TODO : only ROADM nodes are accepted in this release. Relax this with any type of nodes. If filled it must contain ROADM ids separated by ' | '. Exact names are required. 
  - is loose?  'no' value means that the list of nodes should be strictly followed, while any other value means that the constraint may be relaxed if the node is not reachable. 

- ** path bandwidth** is optional. It is the amount of capacity required between source and destination in Gbit/s. Default value is 0.0 Gbit/s. 

path_requests_run.py
------------------------

**Usage**: path_requests_run.py [-h] [-v] [-o OUTPUT]
                            [network_filename xls or json] [service_filename xls or json] [eqpt_filename json]

.. code-block:: shell

    $ cd examples
    $ python path_requests_run.py meshTopologyExampleV2.xls service_file.json eqpt_file -o output_file.json

A function that computes performances for a list of services provided in the service file (accepts json or excel format.

if the service <file.xls> is in xls format, path_requests_run.py converts it to a json file <file_services.json> following the Yang model for requesting Path Computation defined in `draft-ietf-teas-yang-path-computation-01.txt <https://www.ietf.org/id/draft-ietf-teas-yang-path-computation-01.pdf>`_. For PSE use, additional fields with trx type and mode have been added to the te-bandwidth field. 

A template for the json file can be found here: `service_template.json <service_template.json>`_


If no output file is given, the computation is shown on standard output for demo.
If a file is specified with the optional -o argument, the result of the computation is converted into a json format following  the Yang model for requesting Path Computation defined in `draft-ietf-teas-yang-path-computation-01.txt <https://www.ietf.org/id/draft-ietf-teas-yang-path-computation-01.pdf>`_. TODO: verify that this implementation is correct + give feedback to ietf on what is missing for our specific application.

A template for the result of computation json file can be found here: `path_result_template.json <path_result_template.json>`_

Important note: path_requests_run.py is not a network dimensionning tool : each service does not reserve spectrum, or occupy ressources such as transponders. It only computes path feasibility assuming the spectrum (between defined frequencies) is loaded with "nb of channels" spaced by "spacing" values as specified in the system parameters input in the service file, each cannel having the same characteristics in terms of baudrate, format, ... as the service transponder. The transceiver element acts as a "logical starting/stopping point" for the spectral information propagation. At that point it is not meant to represent the capacity of add drop ports
As a result transponder type is not part of the network info. it is related to the list of services requests.

In a next step we plan to provide required features to enable dimensionning : alocation of ressources, counting channels, limitation of the number of channels, ...

(in progress)


