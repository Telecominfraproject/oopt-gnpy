(about-gnpy)=
# About the project

GNPy is a sponsored project of the [OOPT/PSE](https://telecominfraproject.com/open-optical-packet-transport/) working group of the [Telecom Infra Project](http://telecominfraproject.com).

There are weekly calls about our progress.
Newcomers, users and telecom operators are especially welcome there.
We encourage all interested people outside the TIP to [join the project](https://telecominfraproject.com/apply-for-membership/) and especially to [get in touch with us](https://github.com/Telecominfraproject/oopt-gnpy/discussions).

(contributing)=
## Contributing

`gnpy` is looking for additional contributors, especially those with experience planning and maintaining large-scale, real-world mesh optical networks.

To get involved, please contact [Jan Kundrát](mailto:jkt@jankundrat.com) or [Gert Grammel](mailto:ggrammel@juniper.net).

`gnpy` contributions are currently limited to members of [TIP](http://telecominfraproject.com).
Membership is free and open to all.

See the [Onboarding Guide](https://github.com/Telecominfraproject/gnpy/wiki/Onboarding-Guide) for specific details on code contributions, or just [upload patches to our Gerrit](https://review.gerrithub.io/Documentation/intro-gerrit-walkthrough-github.html).
Here is [what we are currently working on](https://review.gerrithub.io/q/project:Telecominfraproject/oopt-gnpy+status:open).

## Project Background

Data Centers are built upon interchangeable, highly standardized node and network architectures rather than a sum of isolated solutions.
This also translates to optical networking.
It leads to a push in enabling multi-vendor optical network by disaggregating HW and SW functions and focusing on interoperability.
In this paradigm, the burden of responsibility for ensuring the performance of such disaggregated open optical systems falls on the operators.
Consequently, operators and vendors are collaborating in defining control models that can be readily used by off-the-shelf controllers.
However, node and network models are only part of the answer.
To take reasonable decisions, controllers need to incorporate logic to simulate and assess optical performance.
Hence, a vendor-independent optical quality estimator is required.
Given its vendor-agnostic nature, such an estimator needs to be driven by a consortium of operators, system and component suppliers.

Founded in February 2016, the Telecom Infra Project (TIP) is an engineering-focused initiative which is operator driven, but features collaboration across operators, suppliers, developers, integrators, and startups with the goal of disaggregating the traditional network deployment approach.
The group’s ultimate goal is to help provide better connectivity for communities all over the world as more people come on-line and demand more bandwidth-intensive experiences like video, virtual reality and augmented reality.

Within TIP, the Open Optical Packet Transport (OOPT) project group is chartered with unbundling monolithic packet-optical network technologies in order to unlock innovation and support new, more flexible connectivity paradigms.

The key to unbundling is the ability to accurately plan and predict the performance of optical line systems based on an accurate simulation of optical parameters.
Under that OOPT umbrella, the Physical Simulation Environment (PSE) working group set out to disrupt the planning landscape by providing an open source simulation model which can be used freely across multiple vendor implementations.

## TIP OOPT/PSE & PSE WG Charter

We believe that openly sharing ideas, specifications, and other intellectual property is the key to maximizing innovation and reducing complexity

TIP OOPT/PSE's goal is to build an end-to-end simulation environment which defines the network models of the optical device transfer functions and their parameters.
This environment will provide validation of the optical performance requirements for the TIP OLS building blocks.

- The model may be approximate or complete depending on the network complexity.
Each model shall be validated against the proposed network scenario.
- The environment must be able to process network models from multiple vendors, and also allow users to pick any implementation in an open source framework.
- The PSE will influence and benefit from the innovation of the DTC, API, and OLS working groups.
- The PSE represents a step along the journey towards multi-layer optimization.

License
-------

GNPy is distributed under a standard BSD 3-Clause License.
