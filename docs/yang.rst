.. _yang:

YANG-formatted data
===================

The YANG-formatted data are the most flexible way of interacting with GNPy.
Some topologies cannot be described via :ref:`XLS<excel>` input files, and new features are not being added to the :ref:`legacy JSON<legacy-json>` format that GNPy used earlier.

The input data describe the :ref:`optical network<yang-topology>` (which is, internally, using some :ref:`optical hardware<yang-equipment>`) as well as the :ref:`requested simulation<yang-simulation>`.


.. _complete-vs-incomplete:

Fully Specified vs. Partially Designed Networks
-----------------------------------------------

Let's consider a simple triangle topology with three PoPs covering three cities:

.. graphviz::
  :layout: neato

  graph {
    A -- B
    B -- C
    C -- A
  }

In the real world, each city would probably host a ROADM and some transponders:

.. graphviz::
  :layout: neato

  graph {
    "ROADM A" [pos="2,2!"]
    "ROADM B" [pos="4,2!"]
    "ROADM C" [pos="3,1!"]
    "Transponder A" [shape=box, pos="0,2!"]
    "Transponder B" [shape=box, pos="6,2!"]
    "Transponder C" [shape=box, pos="3,0!"]

    "ROADM A" -- "ROADM B"
    "ROADM B" -- "ROADM C"
    "ROADM C" -- "ROADM A"

    "Transponder A" -- "ROADM A"
    "Transponder B" -- "ROADM B"
    "Transponder C" -- "ROADM C"
  }

GNPy simulation works by propagating the optical signal over a sequence of elements, which means that one has to add some preamplifiers and boosters, and perhaps also some inline amplifiers.
The amplifiers are, by definition, unidirectional, so the graph becomes quite complex:

.. graphviz::
  :layout: neato

  digraph {
    "ROADM A"
    "ROADM B"
    "ROADM C"
    "Transponder A" [shape=box]
    "Transponder B" [shape=box]
    "Transponder C" [shape=box]

    "Transponder A" -> "ROADM A"
    "Transponder B" -> "ROADM B"
    "Transponder C" -> "ROADM C"
    "ROADM A" -> "Transponder A"
    "ROADM B" -> "Transponder B"
    "ROADM C" -> "Transponder C"

    "Booster A C" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "Preamp A C" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "ROADM A" -> "Booster A C"
    "Preamp A C" -> "ROADM A"

    "Booster A B" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "Preamp A B" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "ROADM A" -> "Booster A B"
    "Preamp A B" -> "ROADM A"

    "Booster C B" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "Preamp C B" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "ROADM C" -> "Booster C B"
    "Preamp C B" -> "ROADM C"

    "Booster C A" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "Preamp C A" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "ROADM C" -> "Booster C A"
    "Preamp C A" -> "ROADM C"

    "Booster B A" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "Preamp B A" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "ROADM B" -> "Booster B A"
    "Preamp B A" -> "ROADM B"

    "Booster B C" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5]
    "Preamp B C" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5]
    "ROADM B" -> "Booster B C"
    "Preamp B C" -> "ROADM B"

    "Booster A C" -> "Preamp C A"
    "Booster A B" -> "Preamp B A"
    "Booster C A" -> "Preamp A C"
    "Booster C B" -> "Preamp B C"
    "Booster B C" -> "Preamp C B"
    "Booster B A" -> "Preamp A B"
  }

In such networks, GNPy's autodesign features becomes very useful.
It is possible to connect ROADMs via "tentative links" which will be replaced by a sequence of actual fibers and specific amplifiers.
In other cases where the location of amplifier huts is already known, but the specific EDFA models have not yet been decided, one can put in amplifier placeholders and let GNPy assign the best amplifier.

.. _yang-topology:

Network Topology
----------------

The *topology* acts as a "digital self" of the simulated network.
The topology builds upon the ``ietf-network-topology`` from `RFC8345 <https://tools.ietf.org/html/rfc8345#section-4.2>`__, and is implemented in the ``tip-photonic-topology`` YANG model.

In this network, the *nodes* correspond to :ref:`amplifiers<yang-topology-amplifier>`, :ref:`ROADMs<yang-topology-roadm>`, :ref:`transceivers<yang-topology-transceiver>` (and sometimes also :ref:`attenuators<yang-topology-attenuator>`), whilte the *links* model :ref:`optical fiber<yang-topology-fiber>` or :ref:`patchcords<yang-topology-patch>`).
Additional elements are also available for modeling networks which have not been fully specified yet.

Where not every amplifier has been placed already, some links can be represented by a :ref:`tentative-link<yang-topology-tentative-link>`, and some amplifier nodes by :ref:`placeholders<yang-topology-amplifier-placeholder>`.

Below is a reference as to how the individual elements are used.

.. _yang-topology-amplifier:

Amplifiers
~~~~~~~~~~

A physical, unidirectional amplifier.
The amplifier **model** is specified via ``tip-photonic-topology:amplifier/model`` leafref.
The following operational data are supported.
If not set, GNPy determines the optimal operating point of the amplifier for the specified simulation input parameters so that the total GSNR remains at its highest possible value.

+--------------------+---------------------------------------------------------------------+
| Operational data   | Description                                                         |
+====================+=====================================================================+
| ``out-voa-target`` | Attenuation of the output VOA                                       |
+--------------------+---------------------------------------------------------------------+
| ``gain-target``    | Amplifier gain                                                      |
+--------------------+---------------------------------------------------------------------+
| ``tilt-target``    | Amplifier tilt                                                      |
+--------------------+---------------------------------------------------------------------+

Operational data are provided via ``gain-target`` and ``tilt-target``.

.. code-block:: json
  :caption: Amplifier definition in JSON

  {
    "node-id": "edfa-A",
    "tip-photonic-topology:amplifier": {
      "model": "fixed-22",
      "out-voa-target": "0.0",
      "gain-target": "19.0",
      "tilt-target": "10.0"
    }
  }

.. _yang-topology-transceiver:

Transceivers
~~~~~~~~~~~~

FIXME

.. _yang-topology-roadm:

ROADMs
~~~~~~

FIXME

.. _yang-topology-attenuator:

Attenuators
~~~~~~~~~~~

FIXME

.. _yang-topology-fiber:

Fiber
~~~~~

FIXME

.. _yang-topology-patch:

Patch cords
~~~~~~~~~~~

FIXME

.. _yang-topology-tentative-link:

Tentative links
~~~~~~~~~~~~~~~

FIXME

.. _yang-topology-amplifier-placeholder:

Amplifier Placeholders
~~~~~~~~~~~~~~~~~~~~~~

FIXME

.. _yang-topology-common-node-props:

Common Node Properties
~~~~~~~~~~~~~~~~~~~~~~

All *nodes* share a common set of properties for describing their physical location.
These are useful mainly for visualizing the network topology.

.. code-block:: javascript

  {
    "node-id": "123",

    // ...more data go here...

    "tip-photonic-topology:geo-location": {
      "x": "0.5",
      "y": "0.0"
    }
  }

.. _yang-equipment:

Equipment Library
-----------------

Before GNPy can start simulating a network, it needs to be told about types of equipment that is in use.
This database is called the *Equipment library*, and can be thought of as a collection of machine-readable data sheets.
Structure of this database is described in the ``tip-photonic-equipment`` YANG model.
The database describes all :ref:`amplifier models<yang-equipment-amplifier>`, all :ref:`types of fiber<yang-equipment-fiber>`, all possible :ref:`ROADM models<yang-equipment-roadm>`, etc.

.. note::
  GNPy ships with a prepopulated equipment library which contains some "faux models" as well as descriptions of ROADMs and amplifiers from actual vendors who decided to share their datasheets publicly.
  The OOPT-PSE group is always looking forward to working with additional vendors to make sure that their hardware is covered, please get in touch.

.. _yang-equipment-amplifier:

Amplifiers
~~~~~~~~~~

One of the key parameters of an amplifier is the method to use for computing the Noise Figure (NF).
GNPy supports four different noise models:

.. _yang-equipment-amplifier-polynomial-NF:

``polynomial-NF``
*****************

This model computes the NF as a function of the difference between the optimal gain and the current gain.
The NF is expressed as a third-degree polynomial.
The input parameters to be provided by the equipment library are four coefficients ``a``. ``b``, ``c`` and ``d``:

.. math::

       f(x) &= \text{a}x^3 + \text{b}x^2 + \text{c}x + \text{d}

  \text{NF} &= f(\text{gain_max} - \text{gain})

.. code-block:: json
  :caption: JSON example of a whitebox EDFA datasheet

  {
    "type": "Juniper-BoosterHG",
    "gain-min": "10",
    "gain-flatmax": "25",
    "max-power-out": "21",
    "frequency-min": "191350000000000",
    "frequency-max": "196100000000000",
    "polynomial-NF": {
      "a": "0.0008",
      "b": "0.0272",
      "c": "-0.2249",
      "d": "6.4902"
    }
  }

This model can be also used for fixed-gain fixed-NF amplifiers. In that case, use:

.. math::

  a = b = c &= 0

          d &= \text{NF}


.. _yang-equipment-amplifier-polynomial-OSNR-OpenROADM:

``polynomial-OSNR-OpenROADM``
*****************************

FIXME

.. _yang-equipment-amplifier-min-max-NF:

``min-max-NF``
**************

FIXME

.. _yang-equipment-amplifier-dual-stage:

``dual-stage``
**************

Dual-stage amplifier combines two distinct amplifiers.
The first amplifier will be always operated at its maximal gain (and therefore its best NF).

+-------------+------------------------------------------+
| Parameter   | Description                              |
+=============+==========================================+
| ``preamp``  | Reference to the first amplifier model.  |
+-------------+------------------------------------------+
| ``booster`` | Reference to the second amplifier model. |
+-------------+------------------------------------------+

.. _yang-equipment-fiber:

Fiber
~~~~~

FIXME

.. _yang-equipment-roadm:

ROADMs
~~~~~~

FIXME

.. _yang-equipment-transponder:

Transponders
~~~~~~~~~~~~

FIXME

.. _yang-simulation:


Simulation Parameters
---------------------

The ``tip-photonic-simulation`` model holds options which control how a simulation behaves.
These include information such as the spectral allocation to work on, the initial launch power, or the desired precision of the Raman engine.

FIXME
