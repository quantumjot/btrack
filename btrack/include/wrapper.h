/*
--------------------------------------------------------------------------------
 Name:     BayesianTracker
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

#include "types.h"
#include "tracker.h"
#include "hypothesis.h"
#include "manager.h"

// Interface class to coordinate the tracker, hypothesis engine and optimisation
// Also provides a simple interface for the python facing code.
class InterfaceWrapper
{
  public:

    // default constructors/destructors
    InterfaceWrapper();
    virtual ~InterfaceWrapper();

    // tracking and basic data handling
    void set_motion_model(const unsigned int measurements,
                          const unsigned int states,
                          double* A,
                          double* H,
                          double* P,
                          double* Q,
                          double* R,
                          double dt,
                          double accuracy,
                          unsigned int max_lost,
                          double prob_not_assign);

    // append an object to the tracker
    void append(const PyTrackObject a_object);

    // run the tracking
    const PyTrackInfo* track();

    // step through the tracking by n steps
    const PyTrackInfo* step(const unsigned int a_steps);

    // get a track by ID, returns the number of objects in the track
    unsigned int get_track(double* output, const unsigned int a_ID) const;

    // get the object IDs of the objects in the track
    unsigned int get_refs(int* output, const unsigned int a_ID) const;

    // get the parent ID
    unsigned int get_parent(const unsigned int a_ID) const;

    // get a dummy object by reference
    PyTrackObject get_dummy(const int a_ID);

    // get the length of a track
    unsigned int track_length(const unsigned int a_ID) const;

    // motion model related data
    unsigned int get_kalman_mu(double* output, const unsigned int a_ID) const;
    unsigned int get_kalman_covar(double* output, const unsigned int a_ID) const;
    unsigned int get_kalman_pred(double* output, const unsigned int a_ID) const;
    unsigned int get_label(unsigned int* output, const unsigned int a_ID) const;

    // return the number of tracks
    unsigned int size() const {
      return tracker.size();
    }

    // get the imaging volume
    void get_volume(double* a_volume) const;

    // hypothesis generation, returns number of hypotheses found
    unsigned int create_hypotheses( PyHypothesisParams params,
                                    const unsigned int a_start_frame,
                                    const unsigned int a_end_frame );
    // return a specific hypothesis
    PyHypothesis get_hypothesis(const unsigned int a_ID);

    // merge tracks based on optimisation
    void merge(unsigned int* a_hypotheses, unsigned int n_hypotheses);

  private:
    // the tracker, track manager and hypothesis engines
    BayesianTracker tracker;
    HypothesisEngine h_engine;
    TrackManager* p_manager;
};

#endif
