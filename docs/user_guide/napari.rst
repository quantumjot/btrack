.. _using Napari:

======
napari
======

``btrack`` comes with a number of optional napari plugins. To install the
dependencies needed to use these plugins, install the ``napari`` extra via.::

    pip install btrack[napari]

If working on Apple Silicon then also run::

    conda install -c conda-forge cvxopt pyqt

The Tracks layer
================

We developed the ``Tracks`` layer that is now part of the multidimensional image viewer `napari <https://napari.org/>`__ -- you can use this to visualize the output of ``btrack``:


.. code:: python

    import napari

    viewer = napari.Viewer()
    viewer.add_labels(segmentation)
    viewer.add_tracks(data, properties=properties, graph=graph)


Read more about `the tracks API at Napari's documentation <https://napari.org/api/napari.layers.Tracks.html>`_.

Visualising trees
=================
`arboretum <https://github.com/quantumjot/arboretum>`__ is a separate plugin that we have developed to visualise lineage trees in napari.
