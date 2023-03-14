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

#include "wrapper.h"

// Interface class to coordinate the tracker, hypothesis engine and optimisation
// Also provides a simple interface for the python facing code.
InterfaceWrapper::InterfaceWrapper() {
  if (DEBUG) {
    std::cout << "Instantiating BTRACK interface wrapper (";
    std::cout << "v" << v_major << "." << v_minor;
    std::cout << "." << v_build << ", compiled " << build_date << " at ";
    std::cout << build_time << ")" << std::endl;
  }

  // set up the track manager
  manager = TrackManager();

  // create a track manager instance, pass it to the tracker
  tracker = BayesianTracker(true, UPDATE_MODE_EXACT, &manager);
};

InterfaceWrapper::~InterfaceWrapper() {
  if (DEBUG) {
    std::cout << "Deleting BTRACK interface wrapper" << std::endl;
  }
};


// sanity check that this is the correct verion of the shared library
bool InterfaceWrapper::check_library_version(const uint8_t a_major,
                                             const uint8_t a_minor,
                                             const uint8_t a_build ) const
{
  if (a_major==v_major && a_minor==v_minor && a_build==v_build){
    return true;
  }

  // throw an error or let python deal with it?
  //throw std::runtime_error("Library version number mismatch.");
  return false;
}

// set the update mode of the tracker
void InterfaceWrapper::set_update_mode(const unsigned int a_update_mode) {
  tracker.set_update_mode(a_update_mode);
}

// set the update features of the tracker
void InterfaceWrapper::set_update_features(const unsigned int a_update_features) {

  // set the update mode for the main tracker
  tracker.set_update_features(a_update_features);
}

// check the update mode of the tracker

// check the update features of the tracker

// tracking and basic data handling
void InterfaceWrapper::set_motion_model(
  const unsigned int measurements,
  const unsigned int states,
  double* A,
  double* H,
  double* P,
  double* Q,
  double* R,
  double dt,
  double accuracy,
  unsigned int max_lost,
  double prob_not_assign
) {

  // set the motion model of the tracker
  tracker.set_motion_model(
    measurements,
    states,
    A,
    H,
    P,
    Q,
    R,
    dt,
    accuracy,
    max_lost,
    prob_not_assign
  );
};

// set the maximum search radius
void InterfaceWrapper::set_max_search_radius(const float max_search_radius)
{
  tracker.set_max_search_radius(max_search_radius);
}

// append a new object to the tracker
void InterfaceWrapper::append(const PyTrackObject a_object)
{
  tracker.append( a_object );
};

// run the complete tracking
const PyTrackInfo* InterfaceWrapper::track()
{
  tracker.track_all();
  return tracker.stats();
};

// track for n steps (interactive mode)
const PyTrackInfo* InterfaceWrapper::step(const unsigned int a_steps)
{
  tracker.step(a_steps);
  return tracker.stats();
};

// return the length of a track by ID
unsigned int InterfaceWrapper::track_length(const unsigned int a_idx) const
{
  // return tracker.tracks[a_ID]->length();
  return manager.get_track(a_idx)->length();
};

// return the internal ID of the track
unsigned int InterfaceWrapper::get_ID(const unsigned int a_idx) const
{
  // TODO(arl): all renamed tracks should have been removed but do we need
  // to do a sanity check?!
  return manager.get_track(a_idx)->ID;
}

// get a track by ID
unsigned int InterfaceWrapper::get_track(
  double* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  unsigned int n_frames = trk->length();
  const unsigned int N = 3+1;

  // NOTE: Grab the actual track
  for (unsigned int i=0; i<n_frames; i++) {
    // frame number
    output[i*N+0] = trk->track[i]->t;
    output[i*N+1] = trk->track[i]->x;
    output[i*N+2] = trk->track[i]->y;
    output[i*N+3] = trk->track[i]->z;
    //output[i*N+4] = tracker.tracks[a_idx]->track[i]->dummy;
  }

  return n_frames;
};

// return the references to the objects
unsigned int InterfaceWrapper::get_refs(
  int* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  unsigned int n_frames = trk->length();

  for (unsigned int i=0; i<n_frames; i++) {
    output[i] = trk->track[i]->ID;
  }
  return n_frames;
};


// return the ID of the parent
unsigned int InterfaceWrapper::get_parent(const unsigned int a_idx) const {

  TrackletPtr trk = manager.get_track(a_idx);

  return trk->parent;
}


// return the ID of the root
unsigned int InterfaceWrapper::get_root(const unsigned int a_idx) const {

  TrackletPtr trk = manager.get_track(a_idx);

  return trk->root;
}


// return the ID of the children
unsigned int InterfaceWrapper::get_children(
  int* children,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  // if the track has no children, return zero
  if (!trk->has_children()) {
    return 0;
  }

  // otherwise return the IDs of the children
  children[0] = trk->child_one;
  children[1] = trk->child_two;
  return 2;
}

// return the fate (as assigned by the optimizer)
unsigned int InterfaceWrapper::get_fate(const unsigned int a_idx) const {

  TrackletPtr trk = manager.get_track(a_idx);

  return trk->fate;
}

// get the generational depth of the track
unsigned int InterfaceWrapper::get_generation(const unsigned int a_idx) const {

  TrackletPtr trk = manager.get_track(a_idx);

  return trk->generation;
}

unsigned int InterfaceWrapper::get_kalman_mu(
  double* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  // NOTE: This is grabbing the Kalman filter rather than the track!
  const unsigned int N = 3+1;
  unsigned int n_frames = trk->length();
  for (unsigned int i=0; i<n_frames; i++) {
    output[i*N+0] = trk->track[i]->t;
    output[i*N+1] = trk->kalman[i].mu(0);
    output[i*N+2] = trk->kalman[i].mu(1);
    output[i*N+3] = trk->kalman[i].mu(2);
  }
  return n_frames;
};

unsigned int InterfaceWrapper::get_kalman_covar(
  double* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  // NOTE: This is grabbing the Kalman filter rather than the track!
  const unsigned int N = 9+1;
  unsigned int n_frames = trk->length();
  for (unsigned int i=0; i<n_frames; i++) {
    output[i*N+0] = trk->track[i]->t;
    output[i*N+1] = trk->kalman[i].covar(0,0);
    output[i*N+2] = trk->kalman[i].covar(0,1);
    output[i*N+3] = trk->kalman[i].covar(0,2);
    output[i*N+4] = trk->kalman[i].covar(1,0);
    output[i*N+5] = trk->kalman[i].covar(1,1);
    output[i*N+6] = trk->kalman[i].covar(1,2);
    output[i*N+7] = trk->kalman[i].covar(2,0);
    output[i*N+8] = trk->kalman[i].covar(2,1);
    output[i*N+9] = trk->kalman[i].covar(2,2);
  }
  return n_frames;
};

unsigned int InterfaceWrapper::get_kalman_pred(
  double* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  // NOTE: This is grabbing the Kalman filter rather than the track!
  const unsigned int N = 3+1;
  unsigned int n_frames = trk->length();
  for (unsigned int i=0; i<n_frames; i++) {
    output[i*N+0] = trk->track[i]->t;
    output[i*N+1] = trk->prediction[i].mu(0);
    output[i*N+2] = trk->prediction[i].mu(1);
    output[i*N+3] = trk->prediction[i].mu(2);
  }
  return n_frames;
};

unsigned int InterfaceWrapper::get_label(
  unsigned int* output,
  const unsigned int a_idx
) const {

  TrackletPtr trk = manager.get_track(a_idx);

  // NOTE: This is grabbing the labels from the track!
  const unsigned int N = 1+1;
  unsigned int n_frames = trk->length();
  for (unsigned int i=0; i<n_frames; i++) {
    output[i*N+0] = trk->track[i]->t;
    output[i*N+1] = trk->track[i]->label;
  }
  return n_frames;
};

// return the imaging volume
void InterfaceWrapper::get_volume(double* a_volume) const
{
  a_volume[0] = tracker.volume.min_xyz(0);
  a_volume[1] = tracker.volume.max_xyz(0);
  a_volume[2] = tracker.volume.min_xyz(1);
  a_volume[3] = tracker.volume.max_xyz(1);
  a_volume[4] = tracker.volume.min_xyz(2);
  a_volume[5] = tracker.volume.max_xyz(2);
};

// set the imaging volume
void InterfaceWrapper::set_volume(const double* a_volume)
{
  tracker.volume.set_volume(a_volume);
}


PyTrackObject InterfaceWrapper::get_dummy(const int a_idx)
{
  // get a pointer to the track manager
  // p_manager = &tracker.tracks;
  return manager.get_dummy(a_idx)->get_pytrack_object();
}


PyGraphEdge InterfaceWrapper::get_graph_edge(const size_t a_idx)
{
  // p_manager = &tracker.tracks;
  return manager.get_edge(a_idx);
}


// hypothesis generation
unsigned int InterfaceWrapper::create_hypotheses(
  const PyHypothesisParams a_params,
  const unsigned int a_start_n,
  const unsigned int a_end_n
) {
  // set up a new hypothesis engine with the parameters supplied
  h_engine = HypothesisEngine(a_start_n, a_end_n, a_params, &manager);
  h_engine.volume = tracker.volume;

  // set the hypothesis engine update features to match the tracker
  h_engine.set_update_features(tracker.get_update_features());

  // add all of the tracks to the engine
  for (size_t i=0; i<manager.num_tracks(); i++) {
    h_engine.add_track(manager.get_track(i));
  }

  // create the hypotheses
  h_engine.create();

  return h_engine.size();
};

// get a hypothesis by ID
PyHypothesis InterfaceWrapper::get_hypothesis(const unsigned int a_idx)
{
  return h_engine.get_hypothesis(a_idx);
};


// merge tracks based on hypothesis IDs
void InterfaceWrapper::merge(
  unsigned int* a_hypotheses,
  unsigned int n_hypotheses
) {

  // make a vector of hypotheses to merge
  std::vector<Hypothesis> merges;
  merges.reserve(n_hypotheses);

  // get a pointer to the track manager
  // p_manager = &tracker.tracks;

  // add all of the hypotheses
  for (size_t i=0; i<n_hypotheses; i++) {
    unsigned int idx = a_hypotheses[i];
    merges.push_back(h_engine.m_hypotheses[idx]);
  }

  // now run the merging
  manager.merge(merges);

}
