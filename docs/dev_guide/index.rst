===============
Developer guide
===============

Installing the latest development version
-----------------------------------------

BayesianTracker has been tested with Python 3.7+ on OS X, Linux and Win10.
The tracker and hypothesis engine are mostly written in C++ with a Python wrapper.

If you would rather install the latest development version, and/or compile directly from source, you can clone and install from this repo:

.. code:: sh

   $ git clone https://github.com/quantumjot/BayesianTracker.git
   $ conda env create -f ./BayesianTracker/environment.yml
   $ conda activate btrack
   $ cd BayesianTracker
   $ pip install -e .

Additionally, the ``build.sh`` script will download Eigen source, run the makefile and pip install.
