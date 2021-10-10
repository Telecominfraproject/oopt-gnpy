.. _concepts:

Simulating networks with GNPy
=============================

Running simulations with GNPy requires three pieces of information:

- the :ref:`network topology<concepts-topology>`, which describes how the network looks like, what are the fiber lengths, what amplifiers are used, etc.,
- the :ref:`equipment library<concepts-equipment>`, which holds machine-readable datasheets of the equipment used in the network,
- the :ref:`simulation options<concepts-simulation>` holding instructions about what to simulate, and under which conditions.

.. _concepts-topology:

Network Topology
----------------

The *topology* acts as a "digital self" of the simulated network.
When given a network topology, GNPy can either run a specific simulation as-is, or it can *optimize* the topology before performing the simulation.

A network topology for GNPy is often a generic, mesh network.
This enables GNPy to take into consideration the current spectrum allocation as well as availability and resiliency considerations.
When the time comes to run a particular *propagation* of a signal and its impairments are computed, though, a linear path through the network is used.
For this purpose, the *path* through the network refers to an ordered, acyclic sequence of *nodes* that are processed.
This path is directional, and all "GNPy elements" along the path match the unidirectional part of a real-world network equipment.

.. note::
  In practical terms, an amplifier in GNPy refers to an entity with a single input port and a single output port.
  A real-world inline EDFA enclosed in a single chassis will be therefore represented as two GNPy-level amplifiers.

The network topology contains not just the physical topology of the network, but also references to the :ref:`equipment library<concepts-equipment>` and a set of *operating parameters* for each entity.
These parameters include the **fiber length** of each fiber, the connector **attenutation losses**, or an amplifier's specific **gain setting**.
The topology is specified via :ref:`XLS files<excel>` or via :ref:`JSON<legacy-json>`.

.. _complete-vs-incomplete:

Fully Specified vs. Partially Designed Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider a simple triangle topology with three :abbr:`PoPs (Points of Presence)` covering three cities:

.. graphviz::
  :layout: neato
  :align: center

  graph "High-level topology with three PoPs" {
    A -- B
    B -- C
    C -- A
  }

In the real world, each city would probably host a ROADM and some transponders:

.. graphviz::
  :layout: neato
  :align: center

  graph "Simplified topology with transponders" {
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

GNPy simulation works by propagating the optical signal over a sequence of elements, which means that one has to add some preamplifiers and boosters.
The amplifiers are, by definition, unidirectional, so the graph becomes quite complex:

.. _topo-roadm-preamp-booster:

.. graphviz::
  :layout: neato
  :align: center

  digraph "Preamps and boosters are explicitly modeled in GNPy" {
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

    "Booster A C" [shape=triangle, orientation=-150, fixedsize=true, width=0.5, height=0.5, pos="2.2,3.2!", color=red, label=""]
    "Preamp A C" [shape=triangle, orientation=0, fixedsize=true, width=0.5, height=0.5, pos="1.5,3.0!", color=red, label=""]
    "ROADM A" -> "Booster A C"
    "Preamp A C" -> "ROADM A"

    "Booster A B" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="3,4.3!", color=red, fontcolor=red, labelloc=b, label="\N\n\n"]
    "Preamp A B" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="3,3.6!", color=red, fontcolor=red, labelloc=t, label="\n        \N"]
    "ROADM A" -> "Booster A B"
    "Preamp A B" -> "ROADM A"

    "Booster C B" [shape=triangle, orientation=-30, fixedsize=true, width=0.5, height=0.5, pos="4.7,0.9!", color=red, label=""]
    "Preamp C B" [shape=triangle, orientation=120, fixedsize=true, width=0.5, height=0.5, pos="5.4,0.7!", color=red, label=""]
    "ROADM C" -> "Booster C B"
    "Preamp C B" -> "ROADM C"

    "Booster C A" [shape=triangle, orientation=30, fixedsize=true, width=0.5, height=0.5, pos="2.6,0.7!", color=red, label=""]
    "Preamp C A" [shape=triangle, orientation=-30, fixedsize=true, width=0.5, height=0.5, pos="3.3,0.9!", color=red, label=""]
    "ROADM C" -> "Booster C A"
    "Preamp C A" -> "ROADM C"

    "Booster B A" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="5,3.6!", labelloc=t, color=red, fontcolor=red, label="\n\N        "]
    "Preamp B A" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="5,4.3!", labelloc=b, color=red, fontcolor=red, label="\N\n\n"]
    "ROADM B" -> "Booster B A"
    "Preamp B A" -> "ROADM B"

    "Booster B C" [shape=triangle, orientation=-180, fixedsize=true, width=0.5, height=0.5, pos="6.5,3.0!", color=red, label=""]
    "Preamp B C" [shape=triangle, orientation=-20, fixedsize=true, width=0.5, height=0.5, pos="5.8,3.2!", color=red, label=""]
    "ROADM B" -> "Booster B C"
    "Preamp B C" -> "ROADM B"

    "Booster A C" -> "Preamp C A"
    "Booster A B" -> "Preamp B A"
    "Booster C A" -> "Preamp A C"
    "Booster C B" -> "Preamp B C"
    "Booster B C" -> "Preamp C B"
    "Booster B A" -> "Preamp A B"
  }

In many regions, the ROADMs are not placed physically close to each other, so the long-haul fiber links (:abbr:`OMS (Optical Multiplex Section)`) are split into individual spans (:abbr:`OTS (Optical Transport Section)`) by in-line amplifiers, resulting in an even more complicated topology graphs:

.. graphviz::
  :layout: neato
  :align: center

  digraph "A subset of a real topology with inline amplifiers" {
    "ROADM A" [pos="2,4!"]
    "ROADM B" [pos="6,4!"]
    "ROADM C" [pos="4,-3!"]
    "Transponder A" [shape=box, pos="1,5!"]
    "Transponder B" [shape=box, pos="7,5!"]
    "Transponder C" [shape=box, pos="4,-4!"]

    "Transponder A" -> "ROADM A"
    "Transponder B" -> "ROADM B"
    "Transponder C" -> "ROADM C"
    "ROADM A" -> "Transponder A"
    "ROADM B" -> "Transponder B"
    "ROADM C" -> "Transponder C"

    "Booster A C" [shape=triangle, orientation=-166, fixedsize=true, width=0.5, height=0.5, pos="2.2,3.2!", label=""]
    "Preamp A C" [shape=triangle, orientation=0, fixedsize=true, width=0.5, height=0.5, pos="1.5,3.0!", label=""]
    "ROADM A" -> "Booster A C"
    "Preamp A C" -> "ROADM A"

    "Booster A B" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="3,4.3!", label=""]
    "Preamp A B" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="3,3.6!", label=""]
    "ROADM A" -> "Booster A B"
    "Preamp A B" -> "ROADM A"

    "Booster C B" [shape=triangle, orientation=-30, fixedsize=true, width=0.5, height=0.5, pos="4.7,-2.1!", label=""]
    "Preamp C B" [shape=triangle, orientation=10, fixedsize=true, width=0.5, height=0.5, pos="5.4,-2.3!", label=""]
    "ROADM C" -> "Booster C B"
    "Preamp C B" -> "ROADM C"

    "Booster C A" [shape=triangle, orientation=20, fixedsize=true, width=0.5, height=0.5, pos="2.6,-2.3!", label=""]
    "Preamp C A" [shape=triangle, orientation=-30, fixedsize=true, width=0.5, height=0.5, pos="3.3,-2.1!", label=""]
    "ROADM C" -> "Booster C A"
    "Preamp C A" -> "ROADM C"

    "Booster B A" [shape=triangle, orientation=90, fixedsize=true, width=0.5, height=0.5, pos="5,3.6!", label=""]
    "Preamp B A" [shape=triangle, orientation=-90, fixedsize=true, width=0.5, height=0.5, pos="5,4.3!", label=""]
    "ROADM B" -> "Booster B A"
    "Preamp B A" -> "ROADM B"

    "Booster B C" [shape=triangle, orientation=-180, fixedsize=true, width=0.5, height=0.5, pos="6.5,3.0!", label=""]
    "Preamp B C" [shape=triangle, orientation=-20, fixedsize=true, width=0.5, height=0.5, pos="5.8,3.2!", label=""]
    "ROADM B" -> "Booster B C"
    "Preamp B C" -> "ROADM B"

    "Inline A C 1" [shape=triangle, orientation=-166, fixedsize=true, width=0.5, pos="2.4,2.2!", label="                             \N", color=red, fontcolor=red]
    "Inline A C 2" [shape=triangle, orientation=-166, fixedsize=true, width=0.5, pos="2.6,1.2!", label="                             \N", color=red, fontcolor=red]
    "Inline A C 3" [shape=triangle, orientation=-166, fixedsize=true, width=0.5, pos="2.8,0.2!", label="                             \N", color=red, fontcolor=red]
    "Inline A C n" [shape=triangle, orientation=-166, fixedsize=true, width=0.5, pos="3.0,-1.1!", label="                             \N", color=red, fontcolor=red]

    "Booster A C" -> "Inline A C 1"
    "Inline A C 1" -> "Inline A C 2"
    "Inline A C 2" -> "Inline A C 3"
    "Inline A C 3" -> "Inline A C n" [style=dotted]
    "Inline A C n" -> "Preamp C A"
    "Booster A B" -> "Preamp B A" [style=dotted]
    "Booster C A" -> "Preamp A C" [style=dotted]
    "Booster C B" -> "Preamp B C" [style=dotted]
    "Booster B C" -> "Preamp C B" [style=dotted]
    "Booster B A" -> "Preamp A B" [style=dotted]
  }

In such networks, GNPy's autodesign features becomes very useful.
It is possible to connect ROADMs via "tentative links" which will be replaced by a sequence of actual fibers and specific amplifiers.
In other cases where the location of amplifier huts is already known, but the specific EDFA models have not yet been decided, one can put in amplifier placeholders and let GNPy assign the best amplifier.

.. _concepts-equipment:

The Equipment Library
---------------------

In order to produce an accurate simulation, GNPy needs to know the physical properties of each entity which affects the optical signal.
Entries in the equipment library correspond to actual real-world, tangible entities.
Unlike a typical :abbr:`NMS (Network Management System)`, GNPy considers not just the active :abbr:`NEs (Network Elements)` such as amplifiers and :abbr:`ROADMs (Reconfigurable Optical Add/Drop Multiplexers)`, but also the passive ones, such as the optical fiber.

As the signal propagates through the network, the largest source of optical impairments is the noise introduced from amplifiers.
An accurate description of the :abbr:`EDFA (Erbium-Doped Fiber Amplifier)` and especially its noise characteristics is required.
GNPy describes this property in terms of the **Noise Figure (NF)** of an amplifier model as a function of its operating point.

The amplifiers compensate power losses induced on the signal in the optical fiber.
The linear losses, however, are just one phenomenon of a multitude of effects that affect the signals in a long fiber run.
While a more detailed description is available :ref:`in the literature<physical-model>`, for the purpose of the equipment library, the description of the *optical fiber* comprises its **linear attenutation coefficient**, a set of parameters for the **Raman effect**, optical **dispersion**, etc.

Signals are introduced into the network via *transponders*.
The set of parameters that are required describe the physical properties of each supported *mode* of the transponder, including its **symbol rate**, spectral **width**, etc.

In the junctions of the network, *ROADMs* are used for spectrum routing.
GNPy currently does not take into consideration the spectrum filtering penalties of the :abbr:`WSSes (Wavelength Selective Switches)`, but the equipment library nonetheless contains a list of required parameters, such as the attenuation options, so that the network can be properly simulated.

.. _concepts-nf-model:

Amplifier Noise Figure Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the key parameters of an amplifier is the method to use for computing the Noise Figure (NF).
GNPy supports several different noise models with varying level of accuracy.
When in doubt, contact your vendor's technical support and ask them to :ref:`contribute their equipment descriptions<extending-edfa>` to GNPy.

The most accurate noise models describe the resulting NF of an EDFA as a third-degree polynomial.
GNPy understands polynomials as a NF-yielding function of the :ref:`gain difference from the optimal gain<ext-nf-model-polynomial-NF>`, or as a function of the input power resulting in an incremental OSNR as used in :ref:`OpenROADM inline amplifiers<ext-nf-model-polynomial-OSNR-OpenROADM>` and :ref:`OpenROADM booster/preamps in the ROADMs<ext-nf-model-noise-mask-OpenROADM>`.
For scenarios where the vendor has not yet contributed an accurate EDFA NF description to GNPy, it is possible to approximate the characteristics via an operator-focused, min-max NF model.

.. _nf-model-min-max-NF:

Min-max NF
**********

This is an operator-focused model where performance is defined by the *minimal* and *maximal NF*.
These are especially suited to model a dual-coil EDFA with a VOA in between.
In these amplifiers, the minimal NF is achieved when the EDFA operates at its maximal (and usually optimal, in terms of flatness) gain.
The worst (maximal) NF applies  when the EDFA operates at its minimal gain.

This model is suitable for use when the vendor has not provided a more accurate performance description of the EDFA.

Raman Approximation
*******************

While GNPy is fully Raman-aware, under certain scenarios it is useful to be able to run a simulation without an accurate Raman description.
For these purposes the :ref:`polynomial NF<ext-nf-model-polynomial-NF>` model with :math:`\text{a} = \text{b} = \text{c} = 0`, and :math:`\text{d} = NF` can be used.

.. _concepts-simulation:

Simulation
----------

When the network model has been instantiated and the physical properties and operational settings of the actual physical devices are known, GNPy can start simulating how the signal propagate through the optical fiber.

This set of input parameters include options such as the *spectrum allocation*, i.e., the number of channels and their spacing.
Various strategies for network optimization can be provided as well.
