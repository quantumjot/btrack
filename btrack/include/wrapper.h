/*
--------------------------------------------------------------------------------
 Name:     btrack
 Purpose:  A multi object tracking library, specifically used to reconstruct
           tracks in crowded fields. Here we use a probabilistic network of
           information to perform the trajectory linking. This method uses
           positional and visual information for track linking.

 Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk

 License:  See LICENSE.md

 Created:  14/08/2014
--------------------------------------------------------------------------------
*/
#ifndef _WRAPPER_H_INCLUDED_
#define _WRAPPER_H_INCLUDED_

#include "hypothesis.h"
#include "manager.h"
#include "tracker.h"
#include "types.h"

// Interface class to coordinate the tracker, hypothesis engine and optimisation
// Also provides a simple interface for the python facing code.
class InterfaceWrapper {
public:
  // default constructors/destructors
  InterfaceWrapper();
  virtual ~InterfaceWrapper();

  // set the tracker to store the candidate graph
  void set_store_candidate_graph(const bool a_store_graph);

  // set the update mode of the tracker (e.g. EXACT, APPROXIMATE, CUDA etc)
  void set_update_mode(const unsigned int a_update_mode);

  // set the update features to use by the tracker (e.g. motion, visual)
  void set_update_features(const unsigned int a_update_features);

  // check the update mode
  bool check_update_mode(const unsigned int a_update_mode) const;

  // check the update features
  bool check_update_features(const unsigned int a_update_features) const;

  // tracking and basic data handling
  void set_motion_model(const unsigned int measurements,
                        const unsigned int states, double *A, double *H,
                        double *P, double *Q, double *R, double dt,
                        double accuracy, unsigned int max_lost,
                        double prob_not_assign);

  // set the maximum search radius
  void set_max_search_radius(const float max_search_radius);

  // append an object to the tracker
  void append(const PyTrackObject a_object);

  // run the tracking
  const PyTrackInfo *track();

  // step through the tracking by n steps
  const PyTrackInfo *step(const unsigned int a_steps);

  // get the internal ID of the track
  unsigned int get_ID(const unsigned int a_ID) const;

  // get a track by ID, returns the number of objects in the track
  unsigned int get_track(double *output, const unsigned int a_ID) const;

  // get the object IDs of the objects in the track
  unsigned int get_refs(int *output, const unsigned int a_ID) const;

  // get the parent ID
  unsigned int get_parent(const unsigned int a_ID) const;

  // get the root ID
  unsigned int get_root(const unsigned int a_ID) const;

  // get the children IDs
  unsigned int get_children(int *children, const unsigned int a_ID) const;

  // get the fate of the track
  unsigned int get_fate(const unsigned int a_ID) const;

  // get the generational depth of the track in the tree
  unsigned int get_generation(const unsigned int a_ID) const;

  // get a dummy object by reference
  PyTrackObject get_dummy(const int a_ID);

  // get the length of a track
  unsigned int track_length(const unsigned int a_ID) const;

  // motion model related data
  unsigned int get_kalman_mu(double *output, const unsigned int a_ID) const;
  unsigned int get_kalman_covar(double *output, const unsigned int a_ID) const;
  unsigned int get_kalman_pred(double *output, const unsigned int a_ID) const;
  unsigned int get_label(unsigned int *output, const unsigned int a_ID) const;

  // return the number of tracks
  unsigned int size() const { return manager.num_tracks(); }

  // get/set the imaging volume
  void set_volume(const double *a_volume);
  void get_volume(double *a_volume) const;

  // get the graph edges of the bayesian belief matrix
  unsigned int num_edges() const { return manager.num_edges(); }

  // get a graph edge
  PyGraphEdge get_graph_edge(const size_t a_ID);

  // hypothesis generation, returns number of hypotheses found
  unsigned int create_hypotheses(PyHypothesisParams params,
                                 const unsigned int a_start_frame,
                                 const unsigned int a_end_frame);

  // return a specific hypothesis
  PyHypothesis get_hypothesis(const unsigned int a_ID);

  // merge tracks based on optimisation
  void merge(unsigned int *a_hypotheses, unsigned int n_hypotheses);

private:
  // the tracker, track manager and hypothesis engines
  BayesianTracker tracker;
  HypothesisEngine h_engine;
  TrackManager manager;
};

#endif
