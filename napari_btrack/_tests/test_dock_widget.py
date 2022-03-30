from ..track import track


def test_add_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_function_widget(function=track)
    assert len(list(viewer.window._dock_widgets)) == num_dw + 1
