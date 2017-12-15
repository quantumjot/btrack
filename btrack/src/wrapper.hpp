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

#include "types.hpp"
#include "tracker.hpp"
#include "hypothesis.hpp"

// Interface class to coordinate the tracker, hypothesis engine and optimisation
// Also provides a simple interface for the python facing code.
class InterfaceWrapper
{
  public:

    // default constructors/destructors
    InterfaceWrapper();
    ~InterfaceWrapper();

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
    unsigned int get_refs(unsigned int* output, const unsigned int a_ID) const;

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
    Hypothesis get_hypothesis(const unsigned int a_ID);

  private:
    // the tracker and hypothesis engines
    BayesianTracker tracker;
    HypothesisEngine h_engine;
};
