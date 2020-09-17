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
    "ROADM A" [pos="2,4!"]
    "ROADM B" [pos="6,4!"]
    "ROADM C" [pos="4,0!"]
    "Transponder A" [shape=box, pos="1,5!"]
    "Transponder B" [shape=box, pos="7,5!"]
    "Transponder C" [shape=box, pos="4,-1!"]

    "Transponder A" -> "ROADM A"
    "Transponder B" -> "ROADM B"
    "Transponder C" -> "ROADM C"
    "ROADM A" -> "Transponder A"
    "ROADM B" -> "Transponder B"
    "ROADM C" -> "Transponder C"

    "Booster A C" [shape=triangle, orientation=-150, fixedsize=true, width=0.5, height=0.5, pos="2.2,3.2!"]
    "Preamp A C" [shape=triangle, orientation=0, fixedsize=true, width=0.5, height=0.5, pos="1.5,3.0!"]
    "ROADM A" -> "Booster A C"
    "Preamp A C" -> "ROADM A"

    "Booster A B" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="3,4.3!"]
    "Preamp A B" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="3,3.6!"]
    "ROADM A" -> "Booster A B"
    "Preamp A B" -> "ROADM A"

    "Booster C B" [shape=triangle, orientation=-30, fixedsize=true, width=0.5, height=0.5, pos="4.7,0.9!"]
    "Preamp C B" [shape=triangle, orientation=120, fixedsize=true, width=0.5, height=0.5, pos="5.4,0.7!"]
    "ROADM C" -> "Booster C B"
    "Preamp C B" -> "ROADM C"

    "Booster C A" [shape=triangle, orientation=30, fixedsize=true, width=0.5, height=0.5, pos="2.6,0.7!"]
    "Preamp C A" [shape=triangle, orientation=150, fixedsize=true, width=0.5, height=0.5, pos="3.3,0.9!"]
    "ROADM C" -> "Booster C A"
    "Preamp C A" -> "ROADM C"

    "Booster B A" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="5,3.6!"]
    "Preamp B A" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="5,4.3!"]
    "ROADM B" -> "Booster B A"
    "Preamp B A" -> "ROADM B"

    "Booster B C" [shape=triangle, orientation=-180, fixedsize=true, width=0.5, height=0.5, pos="6.5,3.0!"]
    "Preamp B C" [shape=triangle, orientation=-20, fixedsize=true, width=0.5, height=0.5, pos="5.8,3.2!"]
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

Below is a reference as to how the individual elements are used.

.. _yang-topology-amplifier:

Amplifiers
~~~~~~~~~~

A physical, unidirectional amplifier.
The amplifier **model** is specified via ``tip-photonic-topology:amplifier/model`` leafref.

Operational data
****************

If not set, GNPy determines the optimal operating point of the amplifier for the specified simulation input parameters so that the total GSNR remains at its highest possible value.

``out-voa-target``
  Attenuation of the output VOA
``gain-target``
  Amplifier gain
``tilt-target``
  Amplifier tilt

Example
*******

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

Transceivers can be used as source and destination points of a path when requesting connectivity feasibility checks.

FIXME: topology properties

.. _yang-topology-roadm:

ROADMs
~~~~~~

FIXME: topology

.. _yang-topology-attenuator:

Attenuators
~~~~~~~~~~~

This element (``attenuator``) is suitable for modeling a real-world long-haul fiber with a splice that has a significant attenuation.
It produces more accurate simulations compared to fiber links with attenuation "moved" to either end.
Only one attribute is defined:

``attenuation``
  Attenuation of the splice, in :math:`\text{dB}`.

.. _yang-topology-amplifier-placeholder:

Amplifier Placeholders
~~~~~~~~~~~~~~~~~~~~~~

In cases where the actual amplifier locations are already known, but a specific type of amplifier has not been decided yet, the ``amplifier-placeholder`` will be used.
This is typically put in place either as a preamp or booster at a ROADM site, or in between two ``fiber`` ``nt::link`` elements.
No properties are defined.

.. _yang-topology-fiber:

Fiber
~~~~~

An ``nt:link`` which contains a ``fiber`` represents a specific, tangible fiber which exists in the physical world.
It has a certain length, is made of a particular material, etc.
The following properties are defined:

``type``
  Class of the fiber.
  Refers to the specified fiber material in the equipment library.
``length``
  Total length of the fiber, in :math:`\text{m}`.
``loss-per-km``
  Fiber attenuation per length.
  In :math:`\text{dB}/\text{km}`.
``attenuation-in``
  FIXME: can we remove this and go with a full-blown attenuator instead?
``conn-att-in`` and ``conn-att-out``
  Attenuation of the input and output connectors, respectively.

Raman properties
****************

When using the Raman engine, additional properties are required:

``raman/temperature``
  This is the average temperature of the fiber, given in :math:`\text{K}`.

Raman amplification
*******************

Actual Raman amplification can be activated by adding several pump lasers below the ``raman`` container.
Use one list member per pump:

``raman/pump[]/frequency``
  Operating frequency of this pump.
  In :math:`\text{Hz}`.
``raman/pump[]/power``
  Pumpping power, in :math:`\text{dBm}`.
``raman/pump[]/direction``
  Direction in which the pumping power is being delivered into the fiber.
  One of ``co-propagating`` (pumping in the same direction as the signal), or ``counter-propagating`` (pumping at the fiber end).

.. _yang-topology-patch:

Patch cords
~~~~~~~~~~~

An ``nt:link`` with a ``patch`` element inside corresponds to a short, direct link.
Typically, this is used for direct connections between equipment.
No non-linearities are considered, and the only allowed parameter is:

``attenuation``
  Total attenuation of the patch cord connection, including the connector losses.

.. _yang-topology-tentative-link:

Tentative links
~~~~~~~~~~~~~~~

An ``nt:link`` which contains a ``tentative-link`` is a placeholder for a link that will be constructed by GNPy.
Unlike either ``patch`` or ``fiber``, this type of a link will never be used in a finalized, fully specified topology.

``type``
  Class of the fiber.
  Refers to the specified fiber material in the equipment library.
``length``
  Total length of the fiber, in :math:`\text{m}`.


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

Amplifiers introduce noise to the signal during amplification, and care must be taken to describe their performance correctly.
There are some common input parameters:

``type``
  A free-form name which must be unique within the whole equipment library.
  It will be used in the network topology to specify which amplifier model is deployed at the given place in the network.
``frequency-min`` and ``frequency-max``
  Operating range of the amplifier.
``gain-flatmax``
  The optimal operating point of the amplifier.
  This is the place where the gain tilt and the NF of the amplifier are at its best.
``gain-min``
  Minimal possible gain that can be set for the EDFA.
  Any lower gain requires adding a physical attenuator.
``max-power-out``
  Total power cap at the output of the amplifier, measured across the whole spectrum.
``has-output-voa``
  Specifies if there's a Variable Optical Attenuator (VOA) at the EDFA's output port.

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

  \text{NF} &= f(G_\text{max} - G)

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

This model is useful for amplifiers compliant to the OpenROADM specification for ILA.
In OpenROADM, amplifier performance is evaluated via its incremental OSNR, which is a function of the input power.
The input parameters to this model are once again four coefficients ``a``. ``b``, ``c`` and ``d``:

.. math::

    \text{OSNR}_\text{inc}(P_\text{in}) = \text{a}P_\text{in}^3 + \text{b}P_\text{in}^2 + \text{c}P_\text{in} + \text{d}

.. code-block:: json
  :caption: JSON example of a low-noise OpenROADM in-line amplifier

  {
    "type": "low-noise",
    "gain-min": "12",
    "gain-flatmax": "27",
    "max-power-out": "22",
    "frequency-min": "191350000000000",
    "frequency-max": "196100000000000",
    "polynomial-OSNR-OpenROADM": {
      "a": "-8.104e-4",
      "b": "-6.221e-2",
      "c": "-5.889e-1",
      "d": "37.62",
    }
  }

.. _yang-equipment-amplifier-min-max-NF:

``min-max-NF``
**************

This is an operator-focused model.
Performance is defined by the minimal and maximal NF.
These are especially suited to model a dual-coil EDFA with a VOA in between.

``nf-min``
  Minimal Noise Figure.
  This is achieved when the EDFA operates at its maximal gain (see the ``gain-flatmax`` parameter).
``nf-max``
  Maximal Noise Figure.
  This worst-case scenario applies when the EDFA operates at its minimal gain (see the ``gain-min`` parameter).

.. _yang-equipment-amplifier-dual-stage:

``dual-stage``
**************

Dual-stage amplifier combines two distinct amplifiers.
The first amplifier will be always operated at its maximal gain (and therefore its best NF).

``preamp``
  Reference to the first amplifier model
``booster``
  Reference to the second amplifier model

.. _yang-equipment-amplifier-fine-tuning:

Advanced EDFA parameters
************************

In addition to all parameters specified above, it is also possible to describe the EDFA's performance in higher detail.
All of the following parameters are given as measurement points at arbitrary frequencies.
The more data points provided, the more accurate is the simulation.
The underlying model uses piecewise linear approximation to estimate values which are laying in between the provided values.

``dynamic-gain-tilt``
  FIXME, what is this?
``gain-ripple``
  Difference of the amplifier gain for a specified frequency, as compared to the typical gain over the whole spectrum
``nf-ripple``
  Difference in the resulting Noise Figure (NF) as a function of a carrier frequency

.. code-block:: json
  :caption: DGT is provided at two frequencies

  {
    "type": "vg-15-26",
    "gain-min": "15",
    "gain-flatmax": "26",
    "dynamic-gain-tilt": [
      {
        "frequency": "191350000000000",
        "dynamic-gain-tilt": "0"
      },
      {
        "frequency": "196100000000000",
        "dynamic-gain-tilt": "2.4"
      }
    ],
    "max-power-out": "23",
    "min-max-NF": {
      "nf-min": "6.0",
      "nf-max": "10.0"
    }
  }

These values are optional.
If not provided, gain and NF is assumed to not vary with carrier frequency.

FIXME: what to say about the DGT?

.. _yang-equipment-fiber:

Fiber
~~~~~

An optical fiber attenuates the signal and acts as a medium for non-linear interference (NLI) for all signals in the propagated spectrum.
When using the Raman-aware simulation engine, the Raman effect is also considered.

``type``
  A free-form name which must be unique within the whole equipment library, such as ``G.652``.
``dispersion``
  Chromatic dispersion, in :math:`s\times m^{-1}\times m^{-1}`.
``dispersion-slope``
  Dispersion slope is related to the :math:`\beta 3` coefficient.
  In :math:`s\times m^{-1}\times m^{-1}\times m^{-1}`.
``gamma``
  Fiber's :math:`\gamma` coefficient.
  In :math:`W^{-1}\times m^{-1}`.
``pmd-coefficient``
  Coefficint for the Polarization Mode Dispersion (PMD).
  In :math:`s\times \sqrt{m^{-1}}`.
``raman-efficiency``
  Normalized efficiency of the Raman amplification per operating frequency.
  This is a required parameter if using Rama-aware simulation engine.


.. code-block:: javascript
  :caption: A standard single mode fiber

  {
    "type": "SSMF",
    "dispersion": "1.67e-05",
    "gamma": "0.00127",
    "pmd-coefficient": "1.265e-15",
    "raman-efficiency": [
      {
        "delta-frequency": "0",
        "cr": "0"
      },
      {
        "delta-frequency": "500000000000",
        "cr": "9.4e-06"
      },

      // more frequencies go here

      {
        "delta-frequency": "42000000000000",
        "cr": "1e-07"
      }
    ]
  }


.. _yang-equipment-roadm:

ROADMs
~~~~~~

Compared to EDFAs and fibers, ROADM descriptions are simpler.
In GNPy, ROADM mainly acts as a smart, spectrum-specific attenuator which equalizes carrier power to a specified power level.
The PMD contribution is also taken into account, and the Add and Drop stages affect signal's OSNR as well.

``type``
  Unique model identification, used when cross-referencing from the network topology.
``add-drop-osnr``
  OSNR penalty introduced by the Add stage or the Drop stage of this ROADM type.
``channel-tx-power``
  Per-channel target TX power towards the egress amplifier.
  Within GNPy, a ROADM is expected to attenuate any signal that enters the ROADM node to this level.
  This can be overridden on a per-link in the network topology.
``pmd``
  Polarization mode dispersion (PMD) penalty of the express path within this ROADM model.
  In :math:`\text{s}`.
``compatible-preamp`` and ``compatible-booster``
  List of all allowed booster/preamplifier types.
  Useful for specifying constraints on what amplifier modules fit into ROADM chassis, and when using fully disaggregated ROADM topologies as well.

.. _yang-equipment-transponder:

Transponders
~~~~~~~~~~~~

Transponders are sources and detectors of optical signals.
There are a few parameters which apply to a transponder model:

``type``
  Unique name, for corss-referencing from the topology data.
``frequency-min`` and ``frequency-max``
  Minimal and maximal operating frequencies of the receiver and transmitter.

A lot of transponders can operate in a variety of modes, which are described via the ``transceiver/mode`` list:

``name``
  Identification of the transmission mode.
  Free form, has to be unique within one transponder type.
``bit-rate``
  Data bit rate, in :math:`\text{bits}\times s^{-1}`.
``baud-rate``
  Symbol modulation rate, in :math:`\text{baud}`.
``required-osnr``
  Minimal allowed OSNR for the receiver.
``tx-osnr``
  Initial OSNR at the transmitter's output.
``grid-spacing``
  Minimal grid spacing, i.e., an effective channel spectral bandwidth.
  In :math:`\text{Hz}`.
``tx-roll-off``
  Roll-off parameter (:math:`\beta`) of the TX pulse shaping filter.
  This assumes a raised-cosine filter.

.. _yang-simulation:


Simulation Parameters
---------------------

The ``tip-photonic-simulation`` model holds options which control how a simulation behaves.
These include information such as the spectral allocation to work on, the initial launch power, or the desired precision of the Raman engine.

FIXME
