from napari_btrack.track import track

def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [(track, {"name", "Track cells."})]
