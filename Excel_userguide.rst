
How to prepare the Excel input file
-----------------------------------

`examples/transmission_main_example.py`_ gives the possibility to use an excel input file instead of a json file. The program then will generate the corresponding json file for you.

The file named 'meshTopologyExampleV2.xls' is an example.

In order to work the excel file MUST contain at least 2 sheets
- Nodes
- Links

(In progress) The File MAY contain an additional sheet:
- Eqt

Nodes sheet
-----------

Nodes sheet contains seven columns
each line represents a 'node' (ROADM site or an in line amplifier site ILA)
- City (Mandatory)
- State
- Country
- Region
- Latitude
- Longitude
- Type

"City" is used for the name of a node of the digraph. It accepts letters, numbers,commas,underscore,dash, blank... (not exhaustive).

It Must be unique. 

"Type" is not mandatory. 
- If not filled, it will be interpreted as an 'ILA' site if node degree is 2 and as a ROADM otherwise.
- If filled, it can take "ROADM" or "ILA" values. if another string is used, it will be considered as not filled.

"State", "Country", "Region" are not mandatory.  

"Longitude", "Latitude" are not mandatory. They should contain numbers.

there must not be empty line(s) between two nodes lines


Links sheet
-----------

Links sheets MUST contain all links between nodes defined in Nodes sheet.
each line represents a 'bidir link' between two nodes. The wo direction are represented on a single line with "east cable from a to z" fields and "west from z to a" fields

- "NodeA" and "NodeZ" are Mandatory. They are the two endpoints of the link. They MUST contain a node name from the "City" names listed in Nodes sheet.

parameters for "east cable from a to z" and "west from z to a" are detailed in 2x7 columns. If not filled, "west from z to a" is copied from "east cable from a to z".

- "Distance km" : is not Mandatory. 
If filled it MUST contain numbers. If empty it is replaced by a default "80" km value. 
If values are over 150 km the program will automatically suppose that intermediate span description are required and will generate spans in the json output.

- "Fiber type" is not mandatory. 
if filled it must contain types listed in eqpt_config.json in "Fiber" list "type_variety".
if not filled it takes "SSMF" as default value.

- "Lineic att" is not mandatory. It is the lineic attenuation expressed in dB/km.
if filled it must contain positive numbers.
if not filled it takes "0.2" dB/km value

- "Con_in", "Con_out" are not mandatory and are not used yet. They are the connector loss in dB at ingress and egress of the fiber spans.
if filled it must contain positive numbers.
if not filled it takes "0.5" dB default value.

- "PMD" is not mandatory and and is not used yet. It is the PMD value of the link in ps.
If filled they must contain positive numbers.
if not filled it takes "0.1" ps value.

- "Cable Id" is not Mandatory. if filled they must contain strings with the same constraint as "City" names.



(in progress)
Eqpt sheet 
----------
Eqt sheet is optional. It lists the amplifiers types and characteristics on each degree.
If the sheet is present, it MUST have as many lines as egress directions of ROADMs defined in Links Sheet. 

For example, consider the following list of links (A,B and C being a ROADM)
A    - amp1
amp1 - amp2
Amp2 - B
A    - amp3
amp3 - C

then Eqpt sheet should contain:
A    - amp1
amp1 - amp2
Amp2 - B
A    - amp3
amp3 - C
B    - amp2
C    - amp3

`examples/create_eqpt_sheet.py`_ helps you to create a template for the mandatory entries of the list.

.. code-block:: shell
    $ cd examples
    $ python create_eqpt_sheet.py meshTopologyExampleV2.xls


# to be completed #

 





