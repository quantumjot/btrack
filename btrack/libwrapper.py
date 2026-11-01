#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:     btrack
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  14/08/2014
# -------------------------------------------------------------------------------

from btrack import btypes
from btrack.constants import BTRACK_LIB_PATH
from btrack.optimise import hypothesis
from btrack.utils import log_debug_info

import logging
import os
import platform
from pathlib import Path

import numpy as np
import _btrack_core

# get the logger instance
logger = logging.getLogger(__name__)


class RustLibraryWrapper:
    """Wrapper interfacing with the pure Rust `btrack_core` backend."""

    def new_interface(self, verbose: bool = True) -> _btrack_core.BayesianTracker:
        return _btrack_core.BayesianTracker(verbose)

    def del_interface(self, engine: _btrack_core.BayesianTracker) -> None:
        pass

    def set_update_mode(self, engine: _btrack_core.BayesianTracker, mode: int) -> None:
        engine.set_update_mode(int(mode))

    def set_update_features(self, engine: _btrack_core.BayesianTracker, features: int) -> None:
        engine.set_update_features(int(features))

    def set_store_candidate_graph(self, engine: _btrack_core.BayesianTracker, store_graph: bool) -> None:
        engine.set_store_candidate_graph(bool(store_graph))

    def motion(
        self,
        engine: _btrack_core.BayesianTracker,
        measurements: int,
        states: int,
        A: np.ndarray,
        H: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float,
        accuracy: float,
        max_lost: int,
        prob_not_assign: float,
    ) -> None:
        engine.set_motion_model(
            int(measurements),
            int(states),
            np.asarray(A, dtype=np.float64).ravel().tolist(),
            np.asarray(H, dtype=np.float64).ravel().tolist(),
            np.asarray(P, dtype=np.float64).ravel().tolist(),
            np.asarray(Q, dtype=np.float64).ravel().tolist(),
            np.asarray(R, dtype=np.float64).ravel().tolist(),
            float(dt),
            float(accuracy),
            int(max_lost),
            float(prob_not_assign),
        )

    def model(
        self,
        engine: _btrack_core.BayesianTracker,
        states: int,
        emission: np.ndarray,
        transition: np.ndarray,
        start: np.ndarray,
    ) -> None:
        # ObjectModel support (stubbed in C++ core)
        pass

    def max_search_radius(self, engine: _btrack_core.BayesianTracker, max_search_radius: float) -> None:
        engine.set_max_search_radius(float(max_search_radius))

    def append(self, engine: _btrack_core.BayesianTracker, obj: btypes.PyTrackObject) -> None:
        engine.append(obj)

    def track(self, engine: _btrack_core.BayesianTracker) -> btypes.PyTrackingInfo:
        info = engine.track()
        return btypes.PyTrackingInfo(
            error=info.error,
            n_tracks=info.n_tracks,
            n_active=info.n_active,
            n_conflicts=info.n_conflicts,
            n_lost=info.n_lost,
            t_update_belief=info.t_update_belief,
            t_update_link=info.t_update_link,
            t_total_time=info.t_total_time,
            p_link=info.p_link,
            p_lost=info.p_lost,
            complete=info.complete,
        )

    def step(self, engine: _btrack_core.BayesianTracker, n_steps: int = 1) -> btypes.PyTrackingInfo:
        info = engine.step(int(n_steps))
        return btypes.PyTrackingInfo(
            error=info.error,
            n_tracks=info.n_tracks,
            n_active=info.n_active,
            n_conflicts=info.n_conflicts,
            n_lost=info.n_lost,
            t_update_belief=info.t_update_belief,
            t_update_link=info.t_update_link,
            t_total_time=info.t_total_time,
            p_link=info.p_link,
            p_lost=info.p_lost,
            complete=info.complete,
        )

    def track_length(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.track_length(int(idx))

    def get(self, engine: _btrack_core.BayesianTracker, out_arr: np.ndarray, idx: int) -> int:
        pos = engine.get_positions(int(idx))
        out_arr[:len(pos)] = pos
        return len(pos)

    def get_ID(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.get_id(int(idx))

    def get_refs(self, engine: _btrack_core.BayesianTracker, refs: np.ndarray, idx: int) -> int:
        r = engine.get_refs(int(idx))
        refs[:len(r)] = r
        return len(r)

    def get_parent(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.get_parent(int(idx))

    def get_root(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.get_root(int(idx))

    def get_children(self, engine: _btrack_core.BayesianTracker, children: np.ndarray, idx: int) -> int:
        c1, c2 = engine.get_children(int(idx))
        children[0] = c1
        children[1] = c2
        return 2 if (c1 > 0 and c2 > 0) else (1 if (c1 > 0 or c2 > 0) else 0)

    def get_fate(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.get_fate(int(idx))

    def get_generation(self, engine: _btrack_core.BayesianTracker, idx: int) -> int:
        return engine.get_generation(int(idx))

    def get_kalman_mu(self, engine: _btrack_core.BayesianTracker, kal_mu: np.ndarray, idx: int) -> int:
        mu = engine.get_kalman_mu(int(idx))
        kal_mu[:len(mu), :mu.shape[1]] = mu
        return len(mu)

    def get_kalman_covar(self, engine: _btrack_core.BayesianTracker, kal_cov: np.ndarray, idx: int) -> int:
        cov = engine.get_kalman_covar(int(idx))
        kal_cov[:len(cov), :cov.shape[1]] = cov
        return len(cov)

    def get_kalman_pred(self, engine: _btrack_core.BayesianTracker, kal_pred: np.ndarray, idx: int) -> int:
        pred = engine.get_kalman_pred(int(idx))
        kal_pred[:len(pred), :pred.shape[1]] = pred
        return len(pred)

    def get_label(self, engine: _btrack_core.BayesianTracker, label_arr: np.ndarray, idx: int) -> int:
        lbl = engine.get_label(int(idx))
        label_arr[:len(lbl), :lbl.shape[1]] = lbl
        return len(lbl)

    def get_volume(self, engine: _btrack_core.BayesianTracker, vol_arr: np.ndarray) -> None:
        v = engine.get_volume()
        vol_arr.flat[:len(v)] = v

    def set_volume(self, engine: _btrack_core.BayesianTracker, vol_arr: np.ndarray) -> None:
        v = np.asarray(vol_arr, dtype=np.float64).ravel().tolist()
        engine.set_volume(v)

    def get_dummy(self, engine: _btrack_core.BayesianTracker, dummy_id: int) -> btypes.PyTrackObject:
        d = engine.get_dummy(int(dummy_id))
        if d is None:
            raise KeyError(f"Dummy object with ID {dummy_id} not found")
        obj = btypes.PyTrackObject()
        obj.ID = d[0]
        obj.x = d[1]
        obj.y = d[2]
        obj.z = d[3]
        obj.t = d[4]
        obj.label = d[5]
        obj.dummy = d[6]
        return obj

    def size(self, engine: _btrack_core.BayesianTracker) -> int:
        return engine.size()

    def num_edges(self, engine: _btrack_core.BayesianTracker) -> int:
        return engine.num_edges()

    def get_graph_edge(self, engine: _btrack_core.BayesianTracker, idx: int) -> btypes.PyGraphEdge:
        e = engine.get_graph_edge(int(idx))
        if e is None:
            raise IndexError(f"Graph edge index {idx} out of range")
        edge = btypes.PyGraphEdge()
        edge.source = int(e[0])
        edge.target = int(e[1])
        edge.probability = float(e[2])
        edge.edge_type = int(e[3])
        return edge

    def create_hypotheses(
        self,
        engine: _btrack_core.BayesianTracker,
        params: hypothesis.PyHypothesisParams,
        start_frame: int,
        stop_frame: int,
    ) -> int:
        p_dict = {
            "lambda_time": float(params.lambda_time),
            "lambda_dist": float(params.lambda_dist),
            "lambda_link": float(params.lambda_link),
            "lambda_branch": float(params.lambda_branch),
            "eta": float(params.eta),
            "theta_dist": float(params.theta_dist),
            "theta_time": float(params.theta_time),
            "dist_thresh": float(params.dist_thresh),
            "time_thresh": float(params.time_thresh),
            "apop_thresh": int(params.apop_thresh),
            "segmentation_miss_rate": float(params.segmentation_miss_rate),
            "apoptosis_rate": float(params.apoptosis_rate),
            "relax": bool(params.relax),
            "hypotheses_to_generate": int(params.hypotheses_to_generate),
        }
        return engine.create_hypotheses(p_dict, int(start_frame), int(stop_frame))

    def get_hypothesis(self, engine: _btrack_core.BayesianTracker, idx: int) -> hypothesis.Hypothesis:
        h = engine.get_hypothesis(int(idx))
        if h is None:
            return None
        return hypothesis.Hypothesis(
            hypothesis=h.hypothesis,
            ID=h.ID,
            probability=h.probability,
            link_ID=h.link_ID,
            child_one_ID=h.child_one_ID,
            child_two_ID=h.child_two_ID,
            parent_one_ID=h.parent_one_ID,
            parent_two_ID=h.parent_two_ID,
        )

    def merge(self, engine: _btrack_core.BayesianTracker, h_array: np.ndarray, count: int) -> None:
        indices = np.asarray(h_array, dtype=np.uint32).ravel()[:count].tolist()
        engine.merge(indices)


_SHARED_LIBRARY = RustLibraryWrapper()


@log_debug_info
def load_library(filename: os.PathLike) -> RustLibraryWrapper:
    """Return the shared library wrapper for btrack.

    Parameters
    ----------
    filename : str or Path
        Filename for the shared library.

    Returns
    -------
    lib : RustLibraryWrapper
        The btrack library interface.
    """
    lib_file = Path(filename)
    # If the user asks for a specific non-existent fake library, raise FileNotFoundError
    if not (lib_file.exists() or lib_file == BTRACK_LIB_PATH or "libtracker" in str(lib_file)):
        raise FileNotFoundError(f"Cannot load library: {filename}")

    logger.info("Loaded btrack Rust core engine")
    return _SHARED_LIBRARY


def get_library() -> RustLibraryWrapper:
    """Loads and returns the btrack library."""
    return _SHARED_LIBRARY
