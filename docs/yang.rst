.. _yang:

YANG-formatted data
===================


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
    "frequency-min": "191.35",
    "frequency-max": "196.1",
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
    "frequency-min": "191.35",
    "frequency-max": "196.1",
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
  FIXME: document this
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
        "frequency": "191.35",
        "dynamic-gain-tilt": "0"
      },
      {
        "frequency": "196.1",
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

.. _yang-equipment-fiber:

Fiber
~~~~~

An optical fiber attenuates the signal and acts as a medium for non-linear interference (NLI) for all signals in the propagated spectrum.
When using the Raman-aware simulation engine, the Raman effect is also considered.

``type``
  A free-form name which must be unique within the whole equipment library, such as ``G.652``.
``dispersion``
  Chromatic dispersion, in :math:`\frac{ps}{nm\times km}`.
``dispersion-slope``
  Dispersion slope is related to the :math:`\beta _3` coefficient.
  In :math:`\frac{ps}{nm^{2}\times km}`.
``gamma``
  Fiber's :math:`\gamma` coefficient.
  In :math:`\frac{1}{W\times km}`.
``pmd-coefficient``
  Coefficint for the Polarization Mode Dispersion (PMD).
  In :math:`\frac{ps}{\sqrt{km}}`.
``raman-efficiency``
  Normalized efficiency of the Raman amplification per operating frequency.
  This is a required parameter if using Rama-aware simulation engine.
  The data type is a YANG list keyed by ``delta-frequency`` (in :math:`\text{THz}`).
  For each ``delta-frequency``, provide the ``cr`` parameter which is a dimensionless number indicating how effective the Raman transfer of energy is at that particular frequency offset from the pumping signal.


.. code-block:: javascript
  :caption: A standard single mode fiber

  {
    "type": "SSMF",
    "dispersion": "16.7",
    "gamma": "1.27",
    "pmd-coefficient": "0.0400028124",
    "raman-efficiency": [
      {
        "delta-frequency": "0",
        "cr": "0"
      },
      {
        "delta-frequency": "0.5",
        "cr": "9.4e-06"
      },

      // more frequencies go here

      {
        "delta-frequency": "42.0",
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
``target-channel-out-power``
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

Transponders (or transceivers) are sources and detectors of optical signals.
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

.. _yang-simulation:


Simulation Parameters
---------------------

The ``tip-photonic-simulation`` model holds options which control how a simulation behaves.
These include information such as the spectral allocation to work on, the initial launch power, or the desired precision of the Raman engine.

Channel allocation is controlled via ``/tip-photonic-simulation:simulation/grid``.
This input structure does not support flexgrid (yet), and it assumes homogeneous channel allocation in a worst-case scenario (all channels allocated):

``frequency-min`` and ``frequency-max``
  Define the range of central channel frequencies.
``spacing``
  How far apart from each other to place channels.
``baud-rate``
  Modulation speed.
``power``
  Launch power, per-channel.
``tx-osnr``
  The initial OSNR of a signal at the transponder's TX port.
``tx-roll-off``
  Roll-off parameter (β) of the TX pulse shaping filter.
  This assumes a raised-cosine filter.

Autodesign is controlled via ``/tip-photonic-simulation:autodesign``.
FIXME: document it.

There are also additional simulation parameters:

``/tip-photonic-simulation:system-margin``
  How many :math:`\text{dB}` of headroom to require.
  This parameter is useful to account for component aging, fiber repairs, etc.
