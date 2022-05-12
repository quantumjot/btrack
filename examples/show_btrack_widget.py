"""
Load and show sample data
=========================
This example:
- loads some sample data
- adds the data to a napari viewer
- loads the arboretum plugin
- opens the napari viewer
"""
from btrack import datasets
import napari

viewer = napari.Viewer()

_, btrack_widget = viewer.window.add_plugin_dock_widget(
    plugin_name="napari-btrack", widget_name="Track"
)


segmentation = datasets.example_segmentation()
viewer._add_layer_from_data(segmentation)
# napari takes the first image layer as default anyway here, but better to be explicit
btrack_widget.segmentation.value = viewer.layers['segmentation']

if __name__ == '__main__':
    # The napari event loop needs to be run under here to allow the window
    # to be spawned from a Python script
    napari.run()
