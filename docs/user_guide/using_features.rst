******************************
Using features during tracking
******************************

.. note::
  The following applies to versions of btrack>=0.5.

By default, ``btrack`` uses a :py:class:`btrack.models.MotionModel` to make predictions about the future position of an object. Further, if specified, object labels defined in :py:class:`btrack.btypes.PyTrackObject` can be used to predict future states.  Together, these predictions can be used to link objects in time to produce the final tracks.

However, it is also possible to utilise other features, such as those derived from image data or segmentations during the Bayesian update step.

.. warning::
  The tracking update makes no assumption about the features being normalised. You should take this into account when designing features to be used for tracking, either by normalising them before tracking, or by using features that fall in a defined range.


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
         - 577
         - 33.260603
         - 22.856213
         - 1.455708
         - 0.968121


Specifying features to use during tracking
==========================================

When running the tracking step, all that is required is to pass the list of features that you want to use to the tracker:

.. code:: python

  # features to be used for tracking updates
  FEATURES = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
  ]

  TRACKING_UPDATES = [
    "motion",
    "visual",
  ]

  # initialise a tracker session using a context manager
  with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure('/path/to/your/models/cell_config.json')

    # set up the features to use as a list
    tracker.features = FEATURES

    # append the objects to be tracked
    tracker.append(objects)

    # tell the tracker to use certain information while
    # performing tracking
    tracker.track(tracking_updates=TRACKING_UPDATES)

    ...

You must specify which information to use when performing the tracking:

* ``motion`` - this uses the motion predictions to link objects in time
* ``visual`` - this uses the features supplied to link objects in time

At least one of these options must be used. The default is only ``motion``. However, you can chose to use ``visual`` only, or a combination of both.

.. warning::
  You must pass the list of features before using the :py:meth:`btrack.BayesianTracker.append` function to add the objects.
