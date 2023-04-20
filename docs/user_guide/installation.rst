.. _installing:

************
Installation
************

Requirements
============

.. |Python| image:: https://img.shields.io/pypi/pyversions/btrack

``btrack`` has been tested with |Python| on ``x86_64`` ``macos>=11``,
``ubuntu>=20.04`` and ``windows>=10.0.17763``.

Installing Scientific Python
============================

If you do have already an Anaconda or miniconda setup you can jump to the next step.

.. note::
   We strongly recommend using a `Python virtual environment <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`__ or a `conda virtual environment. <https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307>`__

If you don't currently have a working scientific Python distribution then follow the `Miniconda Python distribution installation instructions <https://docs.conda.io/en/latest/miniconda.HTML>`__ to install Miniconda.


.. note::
   Miniconda is a lighter version of conda. But all the commands are the same.

Setting up a conda environment
------------------------------

..
   TODO Set the conda-forge channels


Once you have ``conda`` installed, you can create a virtual environment from the terminal/system command prompt or the 'Anaconda Prompt' (under Windows) as::

  conda create -n btrack-env


and access to the environment via::

  conda activate btrack-env


We could have skipped these two steps and install ``btrack`` in the base environment, but virtual environments allow us to keep packages independent of other installations.

Installing btrack
-----------------

After we've created and activated the virtual environment, on the same terminal, we install ``btrack`` with::

  pip install btrack

This will download and install ``btrack`` and all its dependencies.
