===============
Developer guide
===============

Installing the latest development version
-----------------------------------------



``btrack`` has been tested with ![Python](https://img.shields.io/pypi/pyversions/btrack) on ``x86_64`` ``macos>=11``, ``ubuntu>=20.04`` and ``windows>=2019``.
The tracker and hypothesis engine are mostly written in C++ with a Python wrapper.

If you would rather install the latest development version, and/or compile directly from source, you can clone and install from this repo:

.. code:: sh

   git clone https://github.com/quantumjot/btrack.git
   conda env create -f ./btrack/environment.yml
   conda activate btrack
   cd btrack
   ./build.sh
   pip install -e .

If developing the documentation then run the following

.. code:: sh

   pip install -e .[docs]

Releasing
---------

Releases are published to PyPI automatically when a tag is pushed to GitHub.

.. code-block:: sh

   # Set next version number
   export RELEASE=x.x.x

   # Create tags
   git commit --allow-empty -m "Release $RELEASE"
   git tag -a v$RELEASE -m "v$RELEASE"

   # Push
   git push upstream --tags
