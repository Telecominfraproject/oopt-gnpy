.. _extending:

Extending GNPy with vendor-specific data
========================================

GNPy ships with an :ref:`equipment library<concepts-equipment>` containing machine-readable datasheets of networking equipment.
Vendors who are willing to contribute descriptions of their supported products are encouraged to `submit a patch <https://review.gerrithub.io/Documentation/intro-gerrit-walkthrough-github.html>`__ -- or just :ref:`get in touch with us directly<contributing>`.

This chapter discusses option for modeling performance of :ref:`EDFA amplifiers<extending-edfa>`, :ref:`Raman amplifiers<extending-raman>`, :ref:`transponders<extending-transponder>` and :ref:`ROADMs<extending-roadm>`.

.. _extending-edfa:

EDFAs
-----

An accurate description of the :abbr:`EDFA (Erbium-Doped Fiber Amplifier)` and especially its noise characteristics is required.
GNPy describes this property in terms of the **Noise Figure (NF)** of an amplifier model as a function of its operating point.
GNPy supports several different :ref:`noise models<concepts-nf-model>`, and vendors are encouraged to pick one which describes performance of their equipment most accurately.

.. _ext-nf-model-polynomial-NF:

Polynomial NF
*************

This model computes the NF as a function of the difference between the optimal gain and the current gain.
The NF is expressed as a third-degree polynomial:

.. math::

       f(x) &= \text{a}x^3 + \text{b}x^2 + \text{c}x + \text{d}

  \text{NF} &= f(G - G_\text{max})

This model can be also used for fixed-gain fixed-NF amplifiers.
In that case, use:

.. math::

  a = b = c &= 0

          d &= \text{NF}

.. _ext-nf-model-polynomial-OSNR-OpenROADM:

Polynomial OSNR (OpenROADM-style for inline amplifier)
******************************************************

This model is useful for amplifiers compliant to the OpenROADM specification for ILA (an in-line amplifier).
The amplifier performance is evaluated via its incremental OSNR, which is a function of the input power.

.. math::

    \text{OSNR}_\text{inc}(P_\text{in}) = \text{a}P_\text{in}^3 + \text{b}P_\text{in}^2 + \text{c}P_\text{in} + \text{d}

.. _ext-nf-model-noise-mask-OpenROADM:

Noise mask (OpenROADM-style for combined preamp and booster)
************************************************************

Unlike GNPy which simluates the preamplifier and the booster separately as two amplifiers for best accuracy, the OpenROADM specification mandates a certain performance level for a combination of these two amplifiers.
For the express path, the effective noise mask comprises the preamplifier and the booster.
When terminating a channel, the same effective noise mask is mandated for a combination of the preamplifier and the drop stage.

GNPy emulates this specification via two special NF models:

- The ``openroadm_preamp`` NF model for preamplifiers.
  This NF model provides all of the linear impairments to the signal, including those which are incured by the booster in a real network.
- The ``openroadm_booster`` NF model is a special "zero noise" faux amplifier in place of the booster.

.. _ext-nf-model-min-max-NF:

Min-max NF
**********

When the vendor prefers not to share the amplifier description in full detail, GNPy also supports describing the NF characteristics via the *minimal* and *maximal NF*.
This approximates a more accurate polynomial description reasonably well for some models of a dual-coil EDFA with a VOA in between.
In these amplifiers, the minimal NF is achieved when the EDFA operates at its maximal (and usually optimal, in terms of flatness) gain.
The worst (maximal) NF applies  when the EDFA operates at the minimal gain.

.. _ext-nf-model-dual-stage-amplifier:

Dual-stage
**********

Dual-stage amplifier combines two distinct amplifiers.
Vendors which provide an accurate description of their preamp and booster stages separately can use the dual-stage model for an aggregate description of the whole amplifier.

.. _ext-nf-model-advanced:

Advanced Specification
**********************

The amplifier performance can be further described in terms of gain ripple, NF ripple, and the dynamic gain tilt.
When provided, the amplifier characteristic is fine-tuned as a function of carrier frequency.

.. _extending-raman:

Raman Amplifiers
----------------

An accurate simulation of Raman amplification requires knowledge of:

* the *power* and *wavelength* of all Raman pumping lasers,
* the *direction*, whether it is co-propagating or counter-propagating,
* the Raman efficiency of the fiber,
* the fiber temperature.

Under certain scenarios it is useful to be able to run a simulation without an accurate Raman description.
For these purposes, it is possible to approximate a Raman amplifier via a fixed-gain EDFA with the :ref:`polynomial NF<ext-nf-model-polynomial-NF>` model using :math:`\text{a} = \text{b} = \text{c} = 0`, and a desired effective :math:`\text{d} = NF`.
This is also useful to quickly approximate a hybrid EDFA+Raman amplifier.

.. _extending-transponder:

Transponders
------------

Since transponders are usually capable of operating in a variety of modes, these are described separately.
A *mode* usually refers to a particular performance point that is defined by a combination of the symbol rate, modulation format, and :abbr:`FEC (Forward Error Correction)`.

The following data are required for each mode:

``bit-rate``
  Data bit rate, in :math:`\text{Gbits}\times s^{-1}`.
``baud-rate``
  Symbol modulation rate, in :math:`\text{Gbaud}`.
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
``rx-power-min`` and ``rx-power-max``
  The allowed range of power at the receiver.
  In :math:`\text{dBm}`.
``cd-max``
  Maximal allowed Chromatic Dispersion (CD).
  In :math:`\text{ps}/\text{nm}`.
``pmd-max``
  Maximal allowed Polarization Mode Dispersion (PMD).
  In :math:`\text{ps}`.
``cd-penalty``
  *Work-in-progress.*
  Describes the increase of the requires GSNR as the :abbr:`CD (Chromatic Dispersion)` deteriorates.
``dgd-penalty``
  *Work-in-progress.*
  Describes the increase of the requires GSNR as the :abbr:`DGD (Differential Group Delay)` deteriorates.
``pmd-penalty``
  *Work-in-progress.*
  Describes the increase of the requires GSNR as the :abbr:`PMD (Polarization Mode Dispersion)` deteriorates.

GNPy does not directly track the FEC performance, so the type of chosen FEC is likely indicated in the *name* of the selected transponder mode alone.

.. _extending-roadm:

ROADMs
------

In a :abbr:`ROADM (Reconfigurable Add/Drop Multiplexer)`, GNPy simulates the impairments of the preamplifiers and boosters of line degrees :ref:`separately<topo-roadm-preamp-booster>`.
The set of parameters for each ROADM model therefore includes:

``add-drop-osnr``
  OSNR penalty introduced by the Add and Drop stages of this ROADM type.
``target-channel-out-power``
  Per-channel target TX power towards the egress amplifier.
  Within GNPy, a ROADM is expected to attenuate any signal that enters the ROADM node to this level.
  This can be overridden on a per-link in the network topology.
``pmd``
  Polarization mode dispersion (PMD) penalty of the express path.
  In :math:`\text{ps}`.

Provisions are in place to define the list of all allowed booster and preamplifier types.
This is useful for specifying constraints on what amplifier modules fit into ROADM chassis, and when using fully disaggregated ROADM topologies as well.
