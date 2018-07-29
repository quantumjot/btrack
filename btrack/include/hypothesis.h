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

#ifndef _HYPOTHESIS_H_INCLUDED_
#define _HYPOTHESIS_H_INCLUDED_

// #include <python2.7/Python.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>

#include "types.h"
#include "tracklet.h"
#include "hyperbin.h"

// #define TYPE_Pfalse 0
// #define TYPE_Pinit 1
// #define TYPE_Pterm 2
// #define TYPE_Plink 3
// #define TYPE_Pdivn 4
// #define TYPE_Papop 5
// #define TYPE_Pdead 6
// #define TYPE_Pmrge 7
// #define TYPE_undef 999
//
//
// #define STATE_interphase 0
// #define STATE_prometaphase 1
// #define STATE_metaphase 2
// #define STATE_anaphase 3
// #define STATE_apoptosis 4
// #define STATE_null 5

#define MAX_TRACK_LEN 150
#define DEFAULT_LOW_PROBABILITY 1e-308

#define WEIGHT_METAPHASE_ANAPHASE_ANAPHASE 0.01
#define WEIGHT_METAPHASE_ANAPHASE 0.1
#define WEIGHT_METAPHASE 2.0
#define WEIGHT_ANAPHASE_ANAPHASE 1.0
#define WEIGHT_ANAPHASE 2.0
#define WEIGHT_OTHER 5.0


// // Hash index for use with the hash cube
// struct HashIndex {
//   int x = 0;
//   int y = 0;
//   int z = 0;
//   int n = 0;
//
//   // comparison operator for hash map, strict weak ordering
//   bool operator<(const HashIndex &o) const {
//     if (x != o.x) return x < o.x;
//     if (y != o.y) return y < o.y;
//     if (z != o.z) return z < o.z;
//     return n < o.n;
//   }
// };


// Store a hypothesis to return to Python
// NOTE(arl): the probability is actually the log probability
extern "C" struct PyHypothesis {
  unsigned int hypothesis;
  unsigned int ID;
  double probability;
  unsigned int link_ID;
  unsigned int child_one_ID;
  unsigned int child_two_ID;
  unsigned int parent_one_ID;
  unsigned int parent_two_ID;

  PyHypothesis(unsigned int h, unsigned int id): hypothesis(h), ID(id) {};
};


// Internal hypothesis class
class Hypothesis
{
  public:
    Hypothesis() {};
    ~Hypothesis() {};
    Hypothesis( const unsigned int h,
                const TrackletPtr a_trk ): hypothesis(h), ID(a_trk->ID),
                trk_ID(a_trk) {};

    unsigned int hypothesis = TYPE_undef;
    unsigned int ID;
    double probability;

    // store pointers to the tracks
    TrackletPtr trk_ID;

    // used only for track joining
    TrackletPtr trk_link_ID;

    // used for track branching
    TrackletPtr trk_child_one_ID;
    TrackletPtr trk_child_two_ID;

    // used for track merging
    TrackletPtr trk_parent_one_ID;
    TrackletPtr trk_parent_two_ID;


    // return a python compatible hypothesis
    PyHypothesis get_hypothesis() const {

      assert(this->hypothesis != TYPE_undef);

      PyHypothesis h = PyHypothesis(this->hypothesis, this->trk_ID->ID);
      h.probability = this->probability;


      // track joining
      if (this->hypothesis == TYPE_Plink &&
          trk_link_ID != NULL) {
        h.link_ID = this->trk_link_ID->ID;
      };

      // track branching
      if (this->hypothesis == TYPE_Pdivn &&
          trk_child_one_ID != NULL &&
          trk_child_two_ID != NULL) {
        h.child_one_ID = this->trk_child_one_ID->ID;
        h.child_two_ID = this->trk_child_two_ID->ID;
      };

      // track merging
      if (this->hypothesis == TYPE_Pmrge &&
          trk_parent_one_ID != NULL &&
          trk_parent_two_ID != NULL) {
        h.parent_one_ID = this->trk_parent_one_ID->ID;
        h.parent_two_ID = this->trk_parent_two_ID->ID;
      };

      return h;
    };

  private:
};





// // A 4D hash (hyper) cube object.
// //
// // Essentially a way of binsorting trajectory data for easy lookup,
// // thus preventing excessive searching over non-local trajectories.
// //
// class HashCube
// {
// public:
//   HashCube();
//   HashCube(const unsigned int bin_xyz, const unsigned int bin_n);
//   ~HashCube();
//
//   // member functions
//   HashIndex hash_index(TrackletPtr a_trk, const bool a_start) const;
//   HashIndex hash_index( const float x,
//                         const float y,
//                         const float z,
//                         const float n ) const;
//   void add(TrackletPtr a_trk);
//   std::vector<TrackletPtr> get(TrackletPtr a_trk, const bool a_start);
//
// private:
//   // bin size x,y,z,t
//   float m_bin_size[4] = {0., 0., 0., 0.};
//   std::map<HashIndex, std::vector<TrackletPtr>> m_cube;
// };





// A structure to store hypothesis generation parameters
extern "C" struct PyHypothesisParams {
  double lambda_time;
  double lambda_dist;
  double lambda_link;
  double lambda_branch;
  double eta;
  double theta_dist;
  double theta_time;
  double dist_thresh;
  double time_thresh;
  unsigned int apop_thresh;
  double segmentation_miss_rate;
  double apoptosis_rate;
  bool relax;
  unsigned int hypotheses_to_generate;
};




// count the number of apoptosis detections
unsigned int count_apoptosis(const TrackletPtr a_trk);

// calculate the linkage distance
double link_distance(const TrackletPtr a_trk,
                     const TrackletPtr a_trk_lnk);


// calculate the time between the start of new track and end of old track
double link_time(const TrackletPtr a_trk,
                 const TrackletPtr a_trk_lnk);

// safe log function
double safe_log(double value);






// HypothesisEngine
//
// Hypothesis generation for global track optimisation. Uses the tracks from
// BayesianTracker to generate hypotheses.
//
// Generates six different hypotheses, based on the track data provided:
//
//   1. P_FP: a false positive trajectory (probably very short)
//   2. P_init: an initialising trajectory near the edge of the screen or
//      beginning of the movie
//   3. P_term: a terminating trajectory near the edge of the screen or
//      end of movie
//   4. P_link: a broken trajectory which needs to be linked. Probably a
//      one-to-one mapping
//   5. P_branch: a division event where two new trajectories initialise
//   6. P_dead: an apoptosis event

class HypothesisEngine
{
  public:
    // constructors and destructors
    HypothesisEngine();
    ~HypothesisEngine();
    HypothesisEngine( const unsigned int a_start_frame,
                      const unsigned int a_stop_frame,
                      const PyHypothesisParams& a_params );

    // add a track to the hypothesis engine
    void add_track(TrackletPtr a_trk);

    // process the trajectories
    void create();
    //void log_error(Hypothesis *h);

    // return the number of hypotheses
    size_t size() const {
      return m_hypotheses.size();
    }

    // test whether we need to generate this hypothesis
    bool hypothesis_allowed(const unsigned int a_hypothesis_type) const;

    // get a hypothesis
    // TODO(arl): return a reference?
    const PyHypothesis get_hypothesis(const unsigned int a_ID) const {
      return m_hypotheses[a_ID].get_hypothesis();
    }

    // space to store the hypotheses
    std::vector<Hypothesis> m_hypotheses;

    // frame size and number of frames
    // NOTE(arl): technically, this info is already in the imaging volume...
    unsigned int m_frame_range[2] = {0,1};

    // space to store the imaging volume when setting up the HashCube
    ImagingVolume volume;

  private:
    // calculation of probabilities
    double P_TP(TrackletPtr a_trk) const;
    double P_FP(TrackletPtr a_trk) const;
    double P_init(TrackletPtr a_trk) const;
    double P_term(TrackletPtr a_trk) const;
    double P_link(TrackletPtr a_trk,
                  TrackletPtr a_trk_link) const;
    double P_link(TrackletPtr a_trk,
                  TrackletPtr a_trk_link,
                  float d,
                  float dt) const;
    double P_branch(TrackletPtr a_trk,
                    TrackletPtr a_trk_c0,
                    TrackletPtr a_trk_c1) const;
    double P_dead(TrackletPtr a_trk,
                  const unsigned int n_dead) const;
    double P_dead(TrackletPtr a_trk) const;
    double P_merge(TrackletPtr a_trk_m0,
                   TrackletPtr a_trk_m1,
                   TrackletPtr a_trk) const;

    // calculate the distance of a track from the border of the imaging volume
    float dist_from_border( TrackletPtr a_trk, bool a_start ) const;

    // storage for the trajectories
    unsigned int m_num_tracks = 0;
    std::vector<TrackletPtr> m_tracks;

    // space to store a hash cube
    HypercubeBin m_cube;

    // store the hypothesis generation parameters
    PyHypothesisParams m_params;
};




#endif
