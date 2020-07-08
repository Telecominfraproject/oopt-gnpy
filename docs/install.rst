Installing GNPy
---------------

There are several methods on how to obtain GNPy.
The easiest option for a non-developer is probably going via our :ref:`Docker images<install-docker>`.
Developers are encouraged to install the :ref:`Python package in the same way as any other Python package<install-pip>`.
Note that this needs a :ref:`working installation of Python<install-python>`, for example :ref:`via Anaconda<install-anaconda>`.

.. _install-docker:

Using prebuilt Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our `Docker images <https://hub.docker.com/r/telecominfraproject/oopt-gnpy>`_ contain everything needed to run all examples from this guide.
Docker transparently fetches the image over the network upon first use.
On Linux and Mac, run:


.. code-block:: shell-session

    $ docker run -it --rm --volume $(pwd):/shared telecominfraproject/oopt-gnpy
    root@bea050f186f7:/shared/example-data#

On Windows, launch from Powershell as:

.. code-block:: console

    PS C:\> docker run -it --rm --volume ${PWD}:/shared telecominfraproject/oopt-gnpy
    root@89784e577d44:/shared/example-data#

In both cases, a directory named ``example-data/`` will appear in your current working directory.
GNPy automaticallly populates it with example files from the current release.
Remove that directory if you want to start from scratch.

.. _install-python:

Using Python on your computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   **Note**: `gnpy` supports Python 3 only. Python 2 is not supported.
   `gnpy` requires Python ≥3.6

   **Note**: the `gnpy` maintainers strongly recommend the use of Anaconda for
   managing dependencies.

It is recommended that you use a "virtual environment" when installing `gnpy`.
Do not install `gnpy` on your system Python.

.. _install-anaconda:

We recommend the use of the `Anaconda Python distribution <https://www.anaconda.com/download>`_ which comes with many scientific computing
dependencies pre-installed. Anaconda creates a base "virtual environment" for
you automatically. You can also create and manage your ``conda`` "virtual
environments" yourself (see:
https://conda.io/docs/user-guide/tasks/manage-environments.html)

To activate your Anaconda virtual environment, you may need to do the
following:

.. code-block:: shell-session

    $ source /path/to/anaconda/bin/activate # activate Anaconda base environment
    (base) $                                # note the change to the prompt

You can check which Anaconda environment you are using with:

.. code-block:: shell-session

    (base) $ conda env list                          # list all environments
    # conda environments:
    #
    base                  *  /src/install/anaconda3

    (base) $ echo $CONDA_DEFAULT_ENV                 # show default environment
    base

You can check your version of Python with the following. If you are using
Anaconda's Python 3, you should see similar output as below. Your results may
be slightly different depending on your Anaconda installation path and the
exact version of Python you are using.

.. code-block:: shell-session

    $ which python                   # check which Python executable is used
    /path/to/anaconda/bin/python
    $ python -V                      # check your Python version
    Python 3.6.5 :: Anaconda, Inc.

.. _install-pip:

Installing the Python package
*****************************

From within your Anaconda Python 3 environment, you can clone the master branch
of the `gnpy` repo and install it with:

.. code-block:: shell-session

    $ git clone https://github.com/Telecominfraproject/oopt-gnpy # clone the repo
    $ cd oopt-gnpy
    $ pip install --editable . # note the trailing dot

To test that `gnpy` was successfully installed, you can run this command. If it
executes without a ``ModuleNotFoundError``, you have successfully installed
`gnpy`.

.. code-block:: shell-session

    $ python -c 'import gnpy' # attempt to import gnpy

    $ pytest                  # run tests
