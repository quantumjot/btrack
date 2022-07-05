Saving and loading track data
*****************************

Tracks can be exported using the HDF5 file format, as follows:


.. code:: python

  # export tracks using the HDF5 format
  with btrack.BayesianTracker() as tracker:

    # run the tracking
    ...

    # export the data in an HDF5 file
    tracker.export('/path/to/tracks.h5', obj_type='obj_type_1')

Or, if you have a list of :py:class:`btrack.btypes.Tracklet`, you can write them directly to a file:

.. code:: python

  with btrack.dataio.HDF5FileHandler(
    '/path/to/tracks.h5', 'w', obj_type='obj_type_1'
  ) as writer:
    writer.write_tracks(tracks)

Later, tracks can then be loaded using the inbuilt track reader, as follows:

.. code:: python

  with btrack.dataio.HDF5FileHandler(
    '/path/to/tracks.h5', 'r', obj_type='obj_type_1'
  ) as reader:
    tracks = reader.tracks

``tracks`` will now be a Python list of :py:class:`btrack.btypes.Tracklet` objects.
