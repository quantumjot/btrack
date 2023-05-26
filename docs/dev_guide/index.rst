===============
Developer guide
===============

Installing the latest development version
-----------------------------------------

.. |Python| image:: https://img.shields.io/pypi/pyversions/btrack

``btrack`` has been tested with |Python| on ``x86_64`` ``macos>=11``,
``ubuntu>=20.04`` and ``windows>=10.0.17763``.
The tracker and hypothesis engine are mostly written in C++ with a Python wrapper.

If you would rather install the latest development version, and/or compile directly from source, you can clone and install from this repo:

.. code:: sh

    git clone https://github.com/quantumjot/btrack.git
    cd btrack
    ./build.sh
    pip install -e .

If working on Apple Silicon then also run::

.. code:: sh

    conda install -c conda-forge cvxopt

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
