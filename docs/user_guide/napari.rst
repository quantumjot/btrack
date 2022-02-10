.. _using Napari:

==================================
Visualizing track data with napari
==================================

We developed the ``Tracks`` layer that is now part of the multidimensional image viewer `napari <https://napari.org/>`__ -- you can use this to visualize the output of ``btrack``:


.. code:: python

   import napari

   viewer = napari.Viewer()
   viewer.add_labels(segmentation)
   viewer.add_tracks(data, properties=properties, graph=graph)


Read more about `the tracks API at Napari's documentation <https://napari.org/api/stable/napari.layers.Tracks.html#napari.layers.Tracks>`_.

In addition, we provide a `plugin for napari that enables users to visualize lineage trees (arboretum) <https://github.com/quantumjot/arboretum>`_.
