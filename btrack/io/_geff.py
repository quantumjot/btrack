"""GEFF (Graph Exchange File Format) import/export functionality for btrack.

This module provides functions to convert between btrack data structures and
the GEFF format for exchanging spatial graph data.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from btrack import btypes

from .utils import check_track_type

if TYPE_CHECKING:
    try:
        import geff

        from btrack import BayesianTracker
    except ImportError:
        geff = None
        BayesianTracker = None

# get the logger instance
logger = logging.getLogger(__name__)

# Constants
MIN_POSITION_DIMENSIONS = 2
COORDINATE_Z_INDEX = 2


def _validate_geff_prerequisites(filename: os.PathLike):
    """Validate GEFF import prerequisites."""
    try:
        import geff  # noqa: F401
    except ImportError as e:
        msg = (
            "GEFF library is required for importing GEFF files. "
            "Install with: pip install geff"
        )
        raise ImportError(msg) from e

    if not os.path.exists(filename):
        raise FileNotFoundError(f"GEFF file not found: {filename}")


def _read_geff_graph(filename: os.PathLike):
    """Read GEFF file and return graph."""
    import geff

    try:
        graph, _metadata = geff.read(filename, backend="networkx")
    except Exception as e:
        raise ValueError(f"Failed to read GEFF file {filename}: {e}") from e

    if not isinstance(graph, nx.Graph):
        raise ValueError(f"Expected NetworkX graph, got {type(graph)}")

    return graph


def _extract_position_coordinates(position, node_id):
    """Extract x, y, z coordinates from position data."""
    if position is None:
        logger.warning(f"Node {node_id} missing position data, skipping")
        return None

    if not isinstance(position, np.ndarray):
        position = np.array(position)

    if len(position) < MIN_POSITION_DIMENSIONS:
        logger.warning(f"Node {node_id} has insufficient position dimensions, skipping")
        return None

    x = float(position[0])
    y = float(position[1])
    z = (
        float(position[COORDINATE_Z_INDEX])
        if len(position) > COORDINATE_Z_INDEX
        else 0.0
    )

    return x, y, z


def _extract_node_properties(node_data, node_id):
    """Extract properties from node data."""
    # Extract time
    t = node_data.get("t", node_data.get("time", 0))
    if isinstance(t, (list, np.ndarray)) and len(t) > 0:
        t = t[0]
    t = int(t)

    # Extract ID
    obj_id = node_data.get("ID", node_data.get("id", node_id))
    if isinstance(obj_id, str):
        try:
            obj_id = int(obj_id)
        except ValueError:
            obj_id = hash(obj_id) % (2**31)

    # Extract label
    label = node_data.get("label", 0)
    if isinstance(label, str):
        try:
            label = int(label)
        except ValueError:
            label = 0

    dummy = node_data.get("dummy", False)

    return int(obj_id), t, int(label), bool(dummy)


def _create_track_object_from_node(node_id, node_data):
    """Create a PyTrackObject from node data."""
    position = node_data.get("position")
    coords = _extract_position_coordinates(position, node_id)
    if coords is None:
        return None

    x, y, z = coords
    obj_id, t, label, dummy = _extract_node_properties(node_data, node_id)

    data_dict = {
        "ID": obj_id,
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "label": label,
        "dummy": dummy,
    }

    # Add additional features as properties
    excluded_keys = ["position", "ID", "id", "t", "time", "label", "dummy"]
    for key, value in node_data.items():
        if key not in excluded_keys and isinstance(value, (int, float)):
            data_dict[key] = float(value)

    return btypes.PyTrackObject.from_dict(data_dict)


def import_GEFF(filename: os.PathLike) -> list[btypes.PyTrackObject]:
    """Import localizations from a GEFF file.

    Parameters
    ----------
    filename : PathLike
        The filename of the GEFF file to import.

    Returns
    -------
    objects : list[btypes.PyTrackObject]
        A list of objects from the GEFF file.

    Notes
    -----
    GEFF (Graph Exchange File Format) stores spatial graph data in zarr format.
    This function reads the graph and converts nodes back to PyTrackObject instances.

    The GEFF file should contain:
    - Nodes with spatial coordinates (position property)
    - Node properties including time, ID, and other features
    - Edges representing temporal or lineage connections

    Raises
    ------
    ImportError
        If the geff library is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the GEFF file structure is invalid or missing required properties.
    """
    _validate_geff_prerequisites(filename)
    logger.info(f"Reading GEFF file: {filename}")

    graph = _read_geff_graph(filename)
    logger.info(
        f"Loaded graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges"
    )

    objects = []
    for node_id, node_data in graph.nodes(data=True):
        try:
            obj = _create_track_object_from_node(node_id, node_data)
            if obj is not None:
                objects.append(obj)
        except Exception as e:
            logger.warning(f"Failed to convert node {node_id} to PyTrackObject: {e}")
            continue

    if not objects:
        logger.warning("No valid objects found in GEFF file")
    else:
        logger.info(f"Successfully imported {len(objects)} objects from GEFF file")

    return objects


def _create_track_object_with_track_info(node_id, node_data):
    """Create a PyTrackObject from node data with track info."""
    obj = _create_track_object_from_node(node_id, node_data)
    if obj is None:
        return None

    track_id = node_data.get("track_id", 1)
    return {"object": obj, "track_id": track_id, "time": obj.t}


def _group_objects_by_track(node_objects):
    """Group objects by track_id and sort by time."""
    track_groups = {}
    for _node_id, node_info in node_objects.items():
        track_id = node_info["track_id"]
        if track_id not in track_groups:
            track_groups[track_id] = []
        track_groups[track_id].append(node_info["object"])

    # Sort objects within each track by time
    for track_objects in track_groups.values():
        track_objects.sort(key=lambda obj: obj.t)

    return track_groups


def _create_tracklets_from_groups(track_groups):
    """Create Tracklet objects from grouped objects."""
    tracks = []
    for track_id, objects in track_groups.items():
        if objects:  # Only create tracklet if it has objects
            track = btypes.Tracklet(track_id, objects)
            tracks.append(track)
    return tracks


def _apply_lineage_relationships(graph, node_objects, tracks):
    """Apply parent-child lineage relationships to tracks."""
    if not isinstance(graph, nx.DiGraph):
        return

    lineage_map = {}  # child_track_id -> parent_track_id

    for source, target, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("edge_type", "temporal")
        if edge_type == "lineage":
            source_track_id = node_objects.get(source, {}).get("track_id")
            target_track_id = node_objects.get(target, {}).get("track_id")

            if source_track_id is not None and target_track_id is not None:
                lineage_map[target_track_id] = source_track_id

    # Apply parent relationships to tracks
    for track in tracks:
        if track.ID in lineage_map:
            track.parent = lineage_map[track.ID]


def import_GEFF_tracks(filename: os.PathLike) -> list[btypes.Tracklet]:
    """Import tracks from a GEFF file, reconstructing track structure from graph edges.

    Parameters
    ----------
    filename : PathLike
        The filename of the GEFF file to import.

    Returns
    -------
    tracks : list[btypes.Tracklet]
        A list of Tracklet objects reconstructed from the GEFF graph structure.

    Notes
    -----
    This function reads the GEFF graph and reconstructs the track structure by:
    1. Reading all nodes as PyTrackObject instances
    2. Following temporal edges to group objects into tracks
    3. Handling lineage edges to set parent-child relationships
    4. Creating Tracklet objects for each connected track component

    Raises
    ------
    ImportError
        If the geff library is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the GEFF file structure is invalid or missing required properties.
    """
    _validate_geff_prerequisites(filename)
    logger.info(f"Reading GEFF tracks from file: {filename}")

    graph = _read_geff_graph(filename)
    logger.info(
        f"Loaded graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges"
    )

    # Convert all nodes to PyTrackObject instances with track info
    node_objects = {}
    for node_id, node_data in graph.nodes(data=True):
        try:
            node_info = _create_track_object_with_track_info(node_id, node_data)
            if node_info is not None:
                node_objects[node_id] = node_info
        except Exception as e:
            logger.warning(f"Failed to convert node {node_id} to PyTrackObject: {e}")
            continue

    if not node_objects:
        logger.warning("No valid objects found in GEFF file")
        return []

    # Group objects by track and create tracklets
    track_groups = _group_objects_by_track(node_objects)
    tracks = _create_tracklets_from_groups(track_groups)

    # Apply lineage relationships
    _apply_lineage_relationships(graph, node_objects, tracks)

    logger.info(f"Successfully reconstructed {len(tracks)} tracks from GEFF file")
    return tracks


def _create_graph_from_tracks(tracks, axis_names, axis_units):
    """Create NetworkX graph from tracks data."""
    graph = nx.DiGraph()

    # Set default axis information
    if axis_names is None:
        axis_names = ["x", "y", "z", "t"]
    if axis_units is None:
        axis_units = [None, None, None, None]

    # Convert tracks to graph nodes and edges
    for track in tracks:
        _add_track_to_graph(graph, track, tracks)

    return graph


def _add_track_to_graph(graph, track, all_tracks):
    """Add a single track to the graph with nodes and edges."""
    track_objects = track._data

    # Add nodes for each object in the track
    for i, obj in enumerate(track_objects):
        node_id = int(obj.ID)
        position = np.array([obj.x, obj.y, obj.z, obj.t], dtype=float)

        node_attrs = {
            "position": position,
            "x": float(obj.x),
            "y": float(obj.y),
            "z": float(obj.z),
            "t": int(obj.t),
            "track_id": track.ID,
            "object_index": i,
            "ID": obj.ID,
            "label": obj.label,
            "dummy": obj.dummy,
        }

        # Add any additional properties from the object
        if hasattr(obj, "properties") and obj.properties:
            for prop_name, prop_value in obj.properties.items():
                if isinstance(prop_value, (int, float, bool, str)):
                    node_attrs[prop_name] = prop_value

        graph.add_node(node_id, **node_attrs)

        # Add temporal edge to next object in track
        if i > 0:
            prev_obj = track_objects[i - 1]
            prev_node_id = int(prev_obj.ID)
            graph.add_edge(
                prev_node_id, node_id, edge_type="temporal", track_id=track.ID
            )

    # Add lineage edges for parent-child relationships
    _add_lineage_edges(graph, track, all_tracks)


def _add_lineage_edges(graph, track, all_tracks):
    """Add lineage edges for parent-child relationships."""
    if not (hasattr(track, "parent") and track.parent is not None):
        return

    # Find parent track's last node
    for parent_track in all_tracks:
        if track.parent == parent_track.ID:
            parent_last_obj = parent_track._data[-1]
            track_first_obj = track._data[0]
            parent_last_node = int(parent_last_obj.ID)
            track_first_node = int(track_first_obj.ID)
            graph.add_edge(
                parent_last_node,
                track_first_node,
                edge_type="lineage",
                parent_id=parent_track.ID,
                child_id=track.ID,
            )
            break


def _write_graph_to_geff_file(graph, filename, axis_names, axis_units, track_count):
    """Write the graph to a GEFF file."""
    import geff

    metadata = None  # Let GEFF create default metadata

    try:
        geff.write(
            graph=graph,
            store=str(filename),
            metadata=metadata,
            axis_names=axis_names,
            axis_units=axis_units,
            axis_types=["space", "space", "space", "time"],
        )
        logger.info(
            f"Successfully exported {track_count} tracks to GEFF file: {filename}"
        )
    except Exception as e:
        logger.error(f"Failed to write GEFF file {filename}: {e}")
        raise


def export_GEFF(
    filename: os.PathLike,
    tracks: list,
    obj_type: str | None = None,
    axis_names: list[str] | None = None,
    axis_units: list[str] | None = None,
) -> None:
    """Export track data to a GEFF file.

    Parameters
    ----------
    filename : PathLike
        The filename of the GEFF file to write.
    tracks : list[Tracklet]
        A list of Tracklet objects to be exported.
    obj_type : str, optional
        A string describing the object type, e.g. `obj_type_1`.
    axis_names : list[str], optional
        Names for the spatial axes. Defaults to ["x", "y", "z", "t"].
    axis_units : list[str], optional
        Units for the spatial axes. Defaults to [None, None, None, None].

    Notes
    -----
    This function converts btrack Tracklet objects to a NetworkX graph and saves
    it in GEFF format. The conversion creates:
    - Nodes: Individual PyTrackObject instances with spatial coordinates
    - Edges: Temporal connections between consecutive objects in tracks
    - Lineage edges: Parent-child relationships for track divisions

    Raises
    ------
    ImportError
        If the geff library is not installed.
    ValueError
        If no tracks are provided or tracks are invalid.
    """
    try:
        import geff  # noqa: F401
    except ImportError as e:
        msg = (
            "GEFF library is required for exporting GEFF files. "
            "Install with: pip install geff"
        )
        raise ImportError(msg) from e

    if not tracks:
        logger.error(f"No tracks found when exporting to: {filename}")
        return

    if not check_track_type(tracks):
        logger.error("Tracks of incorrect type")
        return

    logger.info(f"Writing GEFF file to: {filename}")

    # Remove existing file to avoid zarr group conflicts
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)

    graph = _create_graph_from_tracks(tracks, axis_names, axis_units)
    logger.info(
        f"Created graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges"
    )

    _write_graph_to_geff_file(graph, filename, axis_names, axis_units, len(tracks))


def export_tracker_GEFF(
    filename: os.PathLike,
    tracker: BayesianTracker,
    obj_type: str | None = None,
) -> None:
    """Export data from a BayesianTracker instance to GEFF format.

    Parameters
    ----------
    filename : PathLike
        The filename of the GEFF file to write.
    tracker : BayesianTracker
        An instance of the tracker containing tracks to export.
    obj_type : str, optional
        The object type to export the data. Usually `obj_type_1`.

    Notes
    -----
    This is a convenience function that extracts tracks from a BayesianTracker
    and exports them using export_GEFF.
    """
    export_GEFF(filename, tracker.tracks, obj_type=obj_type)
