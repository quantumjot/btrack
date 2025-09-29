"""
Show the btrack widget with example data
=========================
This example:
- loads a sample segmentation and cell config
- adds the segmentation to a napari viewer
- loads the btrack plugin
- opens the napari viewer
"""

from btrack import datasets

import napari

viewer = napari.Viewer()
napari.current_viewer()

_, btrack_widget = viewer.window.add_plugin_dock_widget(
    plugin_name="btrack", widget_name="Track"
)


segmentation = datasets.example_segmentation()
viewer.add_labels(segmentation)
# napari takes the first image layer as default anyway here, but better to be explicit
btrack_widget.segmentation.setCurrentText(viewer.layers["segmentation"].name)

if __name__ == "__main__":
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    napari.run()
