(yang)=
# YANG-formatted data

(yang-equipment)=
## Equipment Library

The [equipment library](concepts-equipment) is defined via the `tip-photonic-equipment` YANG model.
The database describes all [amplifier models](yang-equipment-amplifier), all [types of fiber](yang-equipment-fiber), all possible [ROADM models](yang-equipment-roadm), etc.

(yang-equipment-amplifier)=
### Amplifiers

Amplifiers introduce noise to the signal during amplification, and care must be taken to describe their performance correctly.
There are some common input parameters:

`type`

: A free-form name which must be unique within the whole equipment library.
It will be used in the network topology to specify which amplifier model is deployed at the given place in the network.

`frequency-min` and `frequency-max`

: Operating range of the amplifier.

`gain-flatmax`

:   The optimal operating point of the amplifier.
This is the place where the gain tilt and the NF of the amplifier are at its best.

`gain-min`

: Minimal possible gain that can be set for the EDFA.
Any lower gain requires adding a physical attenuator.

`max-power-out`

: Total power cap at the output of the amplifier, measured across the whole spectrum.

`has-output-voa`

: Specifies if there's a Variable Optical Attenuator (VOA) at the EDFA's output port.

One of the key parameters of an amplifier is the method to use for [computing the Noise Figure (NF)](concepts-nf-model).
Here's how they are represented in YANG data:

(yang-equipment-amplifier-polynomial-NF)=
#### `polynomial-NF`

The [Polynomial NF model](ext-nf-model-polynomial-NF) requires four coefficients for the polynomial function: `a`, `b`, `c` and `d`.

```json
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
```

(yang-equipment-amplifier-min-max-NF)=
#### `min-max-NF`

This is an operator-focused model.
Performance is defined by the [minimal and maximal NF](nf-model-min-max-NF).

`nf-min`

: Minimal Noise Figure.
This is achieved when the EDFA operates at its maximal flat gain (see the `gain-flatmax` parameter).

`nf-max`

: Maximal Noise Figure.
This worst-case scenario applies when the EDFA operates at its minimal gain (see the `gain-min` parameter).

(yang-equipment-amplifier-openroadm)=
#### OpenROADM

NF models for preamps, boosters and inline amplifiers as defined via the OpenROADM group.

(yang-equipment-amplifier-polynomial-OSNR-OpenROADM)=
##### `OpenROADM-ILA`

This model is useful for [amplifiers compliant to the OpenROADM specification for ILA](ext-nf-model-polynomial-OSNR-OpenROADM).
The input parameters to this model are once again four coefficients `a`. `b`, `c` and `d`:

```json
{
  "type": "low-noise",
  "gain-min": "12",
  "gain-flatmax": "27",
  "max-power-out": "22",
  "frequency-min": "191.35",
  "frequency-max": "196.1",
  "OpenROADM-ILA": {
    "a": "-8.104e-4",
    "b": "-6.221e-2",
    "c": "-5.889e-1",
    "d": "37.62",
  }
}
```

(yang-equipment-amplifier-OpenROADM-preamp-booster)=
##### `OpenROADM-preamp` and `OpenROADM-booster`

No extra parameters are defined for these NF models.
See the [model documentation](ext-nf-model-noise-mask-OpenROADM) for details.

(yang-equipment-amplifier-composite)=
#### `composite`

A [composite](ext-nf-model-dual-stage-amplifier) amplifier combines two distinct amplifiers.
The first amplifier will be always operated at its maximal gain (and therefore its best NF).

`preamp`

: Reference to the first amplifier model

`booster`

: Reference to the second amplifier model

(yang-equipment-amplifier-raman-approximation)=
#### `raman-approximation`

A fixed-NF amplifier, especially suitable for emulating Raman amplifiers
in scenarios where the Raman-aware engine cannot be used.

`nf`

: Noise Figure of the amplifier.

(yang-equipment-amplifier-fine-tuning)=
#### Advanced EDFA parameters

In addition to all parameters specified above, it is also possible to describe the EDFA\'s performance in higher detail.
All of the following parameters are given as measurement points at arbitrary frequencies.
The more data points provided, the more accurate is the simulation.
The underlying model uses piecewise linear approximation to estimate values which are laying in between the provided values.

`dynamic-gain-tilt`

: FIXME: document this

`gain-ripple`

: Difference of the amplifier gain for a specified frequency, as compared to the typical gain over the whole spectrum

`nf-ripple`

: Difference in the resulting Noise Figure (NF) as a function of a carrier frequency

```json
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
```

These values are optional. If not provided, gain and NF is assumed to not vary with carrier frequency.

(yang-equipment-fiber)=
### Fiber

An optical fiber attenuates the signal and acts as a medium for non-linear interference (NLI) for all signals in the propagated spectrum.
When using the Raman-aware simulation engine, the Raman effect is also considered.

`type`

: A free-form name which must be unique within the whole equipment library, such as `G.652`.

`chromatic-dispersion`

: Chromatic dispersion, in $\frac{ps}{nm\times km}$.

`chromatic-dispersion-slope`

: Dispersion slope is related to the $\beta _3$ coefficient.
In $\frac{ps}{nm^{2}\times km}$.

`gamma`

: Fiber\'s $\gamma$ coefficient.
In $\frac{1}{W\times km}$.

`pmd-coefficient`

: Coefficient for the Polarization Mode Dispersion (PMD).
In $\frac{ps}{\sqrt{km}}$.

`raman-efficiency`

: Normalized efficiency of the Raman amplification per operating frequency.
This is a required parameter if using Rama-aware simulation engine.
The data type is a YANG list keyed by `delta-frequency` (in $\text{THz}$). 
For each `delta-frequency`, provide the `cr` parameter which is a dimensionless number indicating how effective the Raman transfer of energy is at that particular frequency offset from the pumping signal.

```javascript
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
```
(yang-equipment-roadm)=
### ROADMs

Compared to EDFAs and fibers, ROADM descriptions are simpler. 
In GNPy, ROADM mainly acts as a smart, spectrum-specific attenuator which equalizes carrier power to a specified power level. 
The PMD contribution is also taken into account, and the Add and Drop stages affect signal\'s OSNR as well.

`type`

: Unique model identification, used when cross-referencing from the network topology.

`add-drop-osnr`

: OSNR penalty introduced by the Add stage or the Drop stage of this ROADM type.

`target-channel-out-power`

: Per-channel target TX power towards the egress amplifier.
Within GNPy, a ROADM is expected to attenuate any signal that enters the ROADM node to this level. 
This can be overridden on a per-link in the network topology.

`pmd`

: Polarization mode dispersion (PMD) penalty of the express path within this ROADM model. 
In $\text{s}$.

`compatible-preamp` and `compatible-booster`

: List of all allowed booster/preamplifier types. 
Useful for specifying constraints on what amplifier modules fit into ROADM chassis, and when using fully disaggregated ROADM topologies as well.

(yang-equipment-transponder)=
### Transponders

Transponders (or transceivers) are sources and detectors of optical signals.
There are a few parameters which apply to a transponder model:

`type`

: Unique name, for corss-referencing from the topology data.

`frequency-min` and `frequency-max`

: Minimal and maximal operating frequencies of the receiver and transmitter.

A lot of transponders can operate in a variety of modes, which are described via the `transceiver/mode` list:

`name`

: Identification of the transmission mode. 
Free form, has to be unique within one transponder type.

`bit-rate`

: Data bit rate, in $\text{Gbits}\times s^{-1}$.

`baud-rate`

: Symbol modulation rate, in $\text{Gbaud}$.

`required-osnr`

: Minimal allowed OSNR for the receiver.

`in-band-tx-osnr`

: Worst-case guaranteed initial OSNR at the Tx port per 0.1nm of bandwidth
Only the in-band OSNR is considered.

`grid-spacing`

: Minimal grid spacing, i.e., an effective channel spectral bandwidth.
In $\text{Hz}$.

`tx-roll-off`

: Roll-off parameter ($\beta$) of the TX pulse shaping filter.
This assumes a raised-cosine filter.

(yang-simulation)=
## Simulation Parameters

The `tip-photonic-simulation` model holds options which control how a simulation behaves.
These include information such as the spectral allocation to work on, the initial launch power, or the desired precision of the Raman engine.

### Propagated spectrum

Channel allocation is controlled via `/tip-photonic-simulation:simulation/grid`.
This input structure does not support flexgrid (yet), and it assumes homogeneous channel allocation in a worst-case scenario (all channels allocated):

`frequency-min` and `frequency-max`

: Define the range of central channel frequencies.

`spacing`

: How far apart from each other to place channels.

`baud-rate`

: Modulation speed.

`power`

: Launch power, per-channel.

`tx-osnr`

: The initial OSNR of a signal at the transponder's TX port.

`tx-roll-off`

: Roll-off parameter (β) of the TX pulse shaping filter.
This assumes a raised-cosine filter.

### Autodesign

Autodesign is controlled via `/tip-photonic-simulation:autodesign`.

`power-adjustment-for-span-loss`

: This adjusts the launch power of each span depending on the span's loss.
When in effect, launch powers to spans are adjusted based on the total span loss.
The span loss is compared to a reference span of 20 dB, and the launch power is adjusted by about 0.3 * `loss_difference`, up to a provided maximal adjustment.
This adjustment is performed for all spans when running in the `power-mode` (see below).
When in `gain-mode`, it affects only EDFAs which do not have an explicitly assigned `delta-p`.

FIXME: there are more.

#### Power mode

FIXME: This is currently mostly undocumented.
Sorry.

In power mode, GNPy can try out several initial launch powers.
This is controlled via the `/tip-photonic-simulation:autodesign/power-mode/power-sweep`:

`start`

: Initial delta from the reference power when determining the best initial launch power.

`stop`

: Final delta from the reference power when determining the best initial launch power

`step-size`

: Step size when determining the best initial launch power

#### Gain mode

FIXME: This is currently mostly undocumented.
Sorry.

In the gain mode, EDFA gain is based on the previous span loss.
For all EDFAs whose gain has not been set manually, set the gain based on the following rules:

1) Set gain to the preceding span loss.
2) Offset the gains around the reference power (FIXME: what does it mean?

This will leave the gain of EDFAs which have their gains set manually in the network topology unchanged.

### Miscellaneous parameters

`/tip-photonic-simulation:system-margin`

: How many $\text{dB}$ of headroom to require.
This parameter is useful to account for component aging, fiber repairs, etc.

(yang-topology)=
## Network Topology

The *topology* acts as a "digital self" of the simulated network.
The topology builds upon the `ietf-network-topology` from [RFC8345](https://tools.ietf.org/html/rfc8345#section-4.2) and is implemented in the `tip-photonic-topology` YANG model.

In this network, the *nodes* correspond to [amplifiers](yang-topology-amplifier), [ROADMs](yang-topology-roadm), [transceivers](yang-topology-transceiver) and [attenuators](yang-topology-attenuator).
The *links* model [optical fiber](yang-topology-fiber) or [patchcords](yang-topology-patch)).
Additional elements are also available for modeling networks which have not been fully specified yet.

Where not every amplifier has been placed already, some links can be represented by a [tentative-link](yang-topology-tentative-link), and some amplifier nodes by [placeholders](yang-topology-amplifier-placeholder).

(yang-topology-common-node-props)=
### Common Node Properties

All *nodes* share a common set of properties for describing their physical location.
These are useful mainly for visualizing the network topology.

```javascript
  {
    "node-id": "123",

    // ...more data go here...

    "tip-photonic-topology:geo-location": {
      "x": "0.5",
      "y": "0.0"
    }
  }
```

Below is a reference as to how the individual elements are used.

(yang-topology-amplifier)=
### Amplifiers

A physical, unidirectional amplifier.
The amplifier *model* is specified via `tip-photonic-topology:amplifier/model` leafref.

#### Operational data

If not set, GNPy determines the optimal operating point of the amplifier for the specified simulation input parameters so that the total GSNR remains at its highest possible value.

`out-voa-target`

: Attenuation of the output VOA

`gain-target`

: Amplifier gain

`tilt-target`

: Amplifier tilt

#### Example

```json
  {
    "node-id": "edfa-A",
    "tip-photonic-topology:amplifier": {
      "model": "fixed-22",
      "out-voa-target": "0.0",
      "gain-target": "19.0",
      "tilt-target": "10.0"
    }
  }
```

(yang-topology-transceiver)=
### Transceivers

Transceivers can be used as source and destination points of a path when requesting connectivity feasibility checks.

`model`

: Cross-reference to the equipment library, specifies the physical model of this transponder.

There are no transceiver-specific parameters.
Mode selection is done via global simulation parameters.

(yang-topology-roadm)=
### ROADMs

FIXME: topology

(yang-topology-attenuator)=
### Attenuators

This element (``attenuator``) is suitable for modeling a concentrated loss -- perhaps a real-world long-haul fiber with a splice that has a significant attenuation.
Only one attribute is defined:

`attenuation`

: Attenuation of the splice, in $\text{dB}$.

In the original data formed used by GNPy, the corresponding element, `Fused`, was often used as a cue which disabled automatic EDFA placement.

(yang-topology-amplifier-placeholder)=
### Amplifier Placeholders

In cases where the actual amplifier locations are already known, but a specific type of amplifier has not been decided yet, the `amplifier-placeholder` will be used.
This is typically put in place either as a preamp or booster at a ROADM site, or in between two `fiber` `nt::link` elements.
No properties are defined.

(yang-topology-fiber)=
### Fiber

An `nt:link` which contains a `fiber` represents a specific, tangible fiber which exists in the physical world.
It has a certain length, is made of a particular material, etc.
The following properties are defined:

`type`

: Class of the fiber.
Refers to the specified fiber material in the equipment library.

`length`

: Total length of the fiber, in :math:`\text{m}`.

`loss-per-km``

: Fiber attenuation per length.
In $\text{dB}/\text{km}$.

`attenuation-in``

: FIXME: can we remove this and go with a full-blown attenuator instead?

`conn-att-in` and `conn-att-out`

: Attenuation of the input and output connectors, respectively.

#### Raman properties

When using the Raman engine, additional properties are required:

`raman/temperature`

: This is the average temperature of the fiber, given in $\text{K}$.

### Raman amplification

Actual Raman amplification can be activated by adding several pump lasers below the `raman` container.
Use one list member per pump:

`raman/pump[]/frequency`
 
: Operating frequency of this pump.
In $\text{Hz}$.

`raman/pump[]/power`
  
: Pumping power, in $\text{dBm}$.

`raman/pump[]/direction`

: Direction in which the pumping power is being delivered into the fiber.
One of `co-propagating` (pumping in the same direction as the signal), or `counter-propagating` (pumping at the fiber end).

(yang-topology-patch)=
### Patch cords

An `nt:link` with a `patch` element inside corresponds to a short, direct link.
Typically, this is used for direct connections between equipment.
No non-linearities are considered.

(yang-topology-tentative-link)=
### Tentative links

An `nt:link` which contains a `tentative-link` is a placeholder for a link that will be constructed by GNPy.
Unlike either `patch` or `fiber`, this type of a link will never be used in a finalized, fully specified topology.

`type`
  
: Class of the fiber.
Refers to the specified fiber material in the equipment library.

`length`

: Total length of the fiber, in $\text{km}$.
