Bayesian Tracker (btrack) 🔬💻's
================================

|logo|

BayesianTracker (``btrack``) is a Python library for multi object tracking, used to reconstruct trajectories in crowded fields.
Here, we use a probabilistic network of information to perform the trajectory linking.
This method uses spatial information as well as appearance information for track linking.

The tracking algorithm assembles reliable sections of track that do not contain splitting events (tracklets).
Each new tracklet initiates a probabilistic model, and utilises this to predict future states (and error in states) of each of the objects in the field of view.
We assign new observations to the growing tracklets (linking) by evaluating the posterior probability of each potential linkage from a Bayesian belief matrix for all possible linkages.

The tracklets are then assembled into tracks by using multiple hypothesis testing and integer programming to identify a globally optimal solution.
The likelihood of each hypothesis is calculated for some or all of the tracklets based on heuristics.
The global solution identifies a sequence of high-likelihood hypotheses that accounts for all observations.

| |LineageTree|
| *Automated cell tracking and lineage tree reconstruction*.
  Visualization is provided by our plugin to Napari, :ref:`arboretum<using Napari>`.

| |CellTracking|
| *Video of tracking, showing automatic lineage determination*

`Read about the science <http://lowe.cs.ucl.ac.uk/cellx.html>`_.


.. panels::

    Getting started
    ^^^^^^^^^^^^^^^
    .. toctree::
      :maxdepth: 1

      user_guide/installation
      user_guide/index

    ---

    Other info
    ^^^^^^^^^^
    .. toctree::
      :maxdepth: 1

      about
      dev_guide/index


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |logo| image:: /_static/btrack_logo.png
   :alt: Btrack Logo
.. |LineageTree| image:: https://raw.githubusercontent.com/quantumjot/arboretum/master/examples/napari.png
   :target: http://lowe.cs.ucl.ac.uk/cellx.html
.. |CellTracking| image:: http://lowe.cs.ucl.ac.uk/images/youtube.png
   :target: https://youtu.be/EjqluvrJGCg
