******************************
Using features during tracking
******************************

.. note::
  This applies to versions of btrack>=0.5.

By default, ``btrack`` uses a :py:class:`btrack.models.MotionModel` to make predictions about the future position of an object. These predictions can be used to link objects in time to produce the final tracks.

However, it is also possible to utilise other features, such as those derived from the images or segmentation during the Bayesian update step.


Adding features to objects
==========================

In order to do this, one must specify the image features as ``properties`` of an :py:class:`btrack.btypes.PyTrackObject`, either using built-in functions or your own code. Examples of these are given below.

Using built-in functions
------------------------

The built-in function :py:meth:`btrack.utils.segmentation_to_objects` uses ``regionprops`` from ``scikit-image`` to calculate image features.  These can be specified as follows:

.. code:: python

  # features to be calculated from image data
  FEATURES = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
  ]

  objects = btrack.utils.segmentation_to_objects(
    segmentation,
    properties=tuple(FEATURES),
  )


Adding your own features to an object
-------------------------------------

You can also add your own features to an object, which can be utilised for analysis.

.. code:: python

  features_to_add = {"my_feature": 0.1}

  obj = objects[0]
  obj.properties = features_to_add

Inspecting an object
--------------------

One can inspect an object to show the features associated with it.

.. code:: python

  obj

will return a table (assuming use of a Jupyter Notebook):

.. list-table:: object features
       :header-rows: 1

       * - ID
         - x
         - y
         - z
         - t
         - dummy
         - states
         - label
         - n_features
         - area
         - major_axis_length
         - minor_axis_length
         - orientation
         - solidity
       * - 0
         - 517.573657
         - 9.07279
         - 0
         - 0
         - False
         - 7
         - 5
         - 0
         - 577
         - 33.260603
         - 22.856213
         - 1.455708
         - 0.968121


Specifying features to use during tracking
==========================================
