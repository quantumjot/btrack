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

#include "tracker.h"

using namespace ProbabilityDensityFunctions;
using namespace BayesianUpdateFunctions;

// void write_belief_matrix_to_CSV(std::string a_filename,
//                                 Eigen::Ref<Eigen::MatrixXd> a_belief) {
//   std::cout << "Writing: " << a_filename << std::endl;
//   std::ofstream belief_file;
//   belief_file.open(a_filename);
//   belief_file << a_belief.format(CSVFormat);
//   belief_file.close();
// }

// set up the tracker using an existing track manager
BayesianTracker::BayesianTracker(const bool verbose,
                                 const unsigned int update_mode,
                                 TrackManager *a_manager) {
  // set up verbosity
  this->verbose = verbose;

  // set up the tracks
  this->manager = a_manager;

  // NOTE(arl): This isn't really necessary
  // set up the frame map
  frames.clear();

  // set the current frame to zero
  current_frame = 0;

  // reserve some space for the tracks
  active.reserve(RESERVE_ACTIVE_TRACKS);
  new_objects.reserve(RESERVE_NEW_OBJECTS);

  // set the appropriate cost function
  set_update_mode(update_mode);

  // // set a outputfile path
  // // define a filepath for debugging output
  // if (WRITE_BELIEF_MATRIX) {
  //   m_debug_filepath = std::experimental::filesystem::temp_directory_path();
  //   std::cout << "Using temp file path: " << m_debug_filepath << std::endl;
  // }
}

BayesianTracker::~BayesianTracker() {
  // std::cout << "Destruction of BayesianTracker" << std::endl;
  //  clean up gracefully
}

void BayesianTracker::set_update_mode(const unsigned int update_mode) {
  cost_function_mode = update_mode;
  if (DEBUG)
    std::cout << "Update mode: " << cost_function_mode << std::endl;

  if (cost_function_mode == UPDATE_MODE_EXACT) {
    m_cost_fn = &BayesianTracker::cost_EXACT;
  } else if (cost_function_mode == UPDATE_MODE_APPROXIMATE) {
    m_cost_fn = &BayesianTracker::cost_APPROXIMATE;
  } else {
    // throw std::runtime_error("CUDA update method not supported");

    std::cout << "CUDA update method not currently supported, reverting to "
                 "EXACT.";
    std::cout << std::endl;

    m_cost_fn = &BayesianTracker::cost_EXACT;
  }
}

unsigned int BayesianTracker::set_motion_model(
    const unsigned int measurements, const unsigned int states, double *A_raw,
    double *H_raw, double *P_raw, double *Q_raw, double *R_raw, const double dt,
    const double accuracy, const unsigned int max_lost,
    const double prob_not_assign) {
  // do some error checking
  if (prob_not_assign <= 0. || prob_not_assign >= 1.)
    return ERROR_prob_not_assign_out_of_range;

  if (max_lost > 10)
    return ERROR_max_lost_out_of_range;

  if (accuracy < 0. || accuracy > 10.)
    return ERROR_accuracy_out_of_range;

  this->max_lost = max_lost;
  this->prob_not_assign = prob_not_assign;
  this->accuracy = accuracy;

  if (verbose && DEBUG) {
    std::cout << "MAX_LOST: " << this->max_lost << std::endl;
    std::cout << "ACCURACY: " << this->accuracy << std::endl;
    std::cout << "PROB_NOT_ASSIGN: " << this->prob_not_assign << std::endl;
  }

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajMat;

  // map the arrays to Eigen matrices
  Eigen::MatrixXd H = Eigen::Map<RowMajMat>(H_raw, measurements, states);
  Eigen::MatrixXd Q = Eigen::Map<RowMajMat>(Q_raw, states, states);
  Eigen::MatrixXd P = Eigen::Map<RowMajMat>(P_raw, states, states);
  Eigen::MatrixXd R = Eigen::Map<RowMajMat>(R_raw, measurements, measurements);
  Eigen::MatrixXd A = Eigen::Map<RowMajMat>(A_raw, states, states);

  // set up a new motion model
  motion_model = MotionModel(A, H, P, R, Q);

  return SUCCESS;
}

unsigned int BayesianTracker::append(const PyTrackObject &new_object) {
  // take a vector of TrackObjects as input and perform the tracking step

  // from the PyTrackObject create a track object with a shared pointer.
  // This will be stored
  TrackObjectPtr p = std::make_shared<TrackObject>(new_object);

  // NOTE: THIS IS PROBABLY UNNECESSARY UNTIL WE RETURN PyTrackObjects...
  // p->original_object = &new_object;
  // std::cout << "features: " << p->features << std::endl;

  // update the imaging volume with this new measurement
  volume.update(p);

  // add a new object and maintain a set of frame numbers...
  // objects.push_back( p );
  manager->push_node(p);
  frames_set.insert(p->t);

  // set this flag to true
  // initialised = true;
  return SUCCESS;
}

// track all objects
void BayesianTracker::track_all() {
  if (statistics.complete) {
    statistics.error = ERROR_no_tracks;
  }

  // iterate over the whole set
  while (!statistics.complete) {
    step();
  }
}

// initialise the first frame
unsigned int BayesianTracker::initialise() {
  // if we have already initialised, return early
  if (initialised)
    return SUCCESS;

  if (DEBUG) {
    std::cout << "Using motion features: " << use_motion_features()
              << std::endl;
    std::cout << "Using visual features: " << use_visual_features()
              << std::endl;
  }

  // check to make sure that we've got some objects to track
  if (manager->num_nodes() < 1) {
    if (verbose) {
      std::cout << "Object queue is empty. " << std::endl;
    }
    return ERROR_empty_queue;
  }

  if (manager->num_tracks() > 0) {
    if (verbose) {
      std::cout << "Tracking has already been performed. " << std::endl;
    }
    return ERROR_no_tracks;
  }

  // // sort the objects vector by time
  // std::sort( objects.begin(), objects.end(), compare_obj_time );
  manager->sort_nodes();

  // NOTE: should check that we have some frames which can be tracked
  // start by converting the set to a vector
  std::vector<unsigned int> f(frames_set.begin(), frames_set.end());
  frames = f;

  bool useable_frames = false;
  for (size_t n = 1; n < frames.size(); n++) {
    if ((frames[n] - frames[n - 1]) <= max_lost) {
      useable_frames = true;
      continue;
    }
  }

  if (!useable_frames) {
    if (verbose) {
      std::cout << "No trackable frames have been found. " << std::endl;
    }
    return ERROR_no_useable_frames;
  }

  if (verbose && DEBUG) {
    std::cout << "FRAME RANGE: " << frames.front() << "-" << frames.back();
    std::cout << std::endl;
  }

  // get the number of objects and a counter to the first object
  n_objects = manager->num_nodes();
  o_counter = 0;

  // set the current frame of the tracker
  current_frame = frames.front();

  // set up the first tracklets based on the first set of objects
  while (manager->get_node(o_counter)->t == current_frame &&
         o_counter != n_objects - 1) {
    // add a new tracklet
    TrackletPtr trk = std::make_shared<Tracklet>(
        get_new_ID(),
        manager->get_node(o_counter), // objects[o_counter],
        max_lost, this->motion_model);
    // tracks.push_back( trk );
    manager->push_track(trk);
    o_counter++;
  }

  // add one to the iteration
  current_frame++;

  // set the initialised flag
  initialised = true;

  return SUCCESS;
}

// update the tracker by some number of steps
void BayesianTracker::step(const unsigned int steps) {
  // make sure that we have steps greater than zero
  // assert(steps>0);
  if (steps < 1)
    return;

  // reset the step counter
  unsigned int step = 0;

  // first check the iteration, if it is zero, initialise
  // TODO(arl): we don't necessarily start on frame zero?
  // if (current_frame == 0) {
  if (!initialised) {
    // initialise!
    unsigned int ret = initialise();
    if (ret != SUCCESS) {
      // return the error in a statistics structure
      statistics.error = ret;
      return;
    }
    // take a step
    // step++;
  }

  while (step < steps && current_frame <= frames.back()) {
    // update the list of active tracks
    update_active();

    // clear the list of objects
    new_objects.clear();

    // std::cout << "Full Frame: " << current_frame << std::endl;

    // loop over all tracks found in this frame
    if (o_counter < n_objects) {
      while (manager->get_node(o_counter)->t == current_frame) {
        // store a reference to this object
        new_objects.push_back(manager->get_node(o_counter));
        o_counter++;

        // make sure our iterator doesn't run off the end of the vector
        if (o_counter >= n_objects) {
          break;
        }
      }
    }

    // set up some counters
    size_t n_active = active.size();
    size_t n_obs = new_objects.size();

    // if we have an empty frame, append dummies to everthing and continue
    if (new_objects.empty()) {
      // std::cout << "Frame " << current_frame << " is empty..." <<
      // std::endl;
      for (size_t i = 0; i < n_active; i++) {
        active[i]->append_dummy(use_motion_features());
      }
      step++;
      current_frame++;
      continue;
    }

    // make some space for the belief matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> belief;

    // now do the Bayesian updates
    belief.setZero(n_obs + 1, n_active);

    // set the update iteration
    // on the first pass, use a uniform prior
    // on the second pass, use the current prior and so forth
    unsigned int update_iteration = 0;

    // now do the Bayesian updates using the correct mode
    // use an implicit function pointer call to the appropriate cost
    // function
    if (use_motion_features()) {
      m_update_fn = &BayesianTracker::prob_update_motion;
      (this->*m_cost_fn)(belief, n_active, n_obs,
                         (update_iteration == USE_UNIFORM_PRIOR));
      update_iteration++;
    }

    if (use_visual_features()) {
      m_update_fn = &BayesianTracker::prob_update_visual;
      (this->*m_cost_fn)(belief, n_active, n_obs,
                         (update_iteration == USE_UNIFORM_PRIOR));
      update_iteration++;
    }

    // // write out belief matrix here
    // if (WRITE_BELIEF_MATRIX) {
    //   std::stringstream belief_filename;
    //   belief_filename << m_debug_filepath << "belief_" << current_frame
    //                   << ".csv";
    //   write_belief_matrix_to_CSV(belief_filename.str(), belief);
    // }

    // if we're storing the graph edges for future optimization, do so here
    // this should be done *BEFORE* linking because it relies on the
    // unlinked tracks to store the original hypothesis
    if (manager->get_store_candidate_graph()) {
      for (size_t trk = 0; trk < n_active; trk++) {
        Eigen::VectorXd prob_assign_per_obj;
        prob_assign_per_obj = belief.col(trk);

        // if we are lost, don't store the edge, it will be generated
        // later by the optimizer
        Eigen::MatrixXf::Index best_candidate;
        double prob = prob_assign_per_obj.maxCoeff(&best_candidate);
        if (int(best_candidate) == n_obs)
          continue;

        // note that this doesn't store an edge to `lost`, but we can
        // infer it as 1 - sum(scores) for each association
        for (size_t obj = 0; obj < n_obs; obj++) {
          manager->push_edge(active[trk]->track.back(), new_objects[obj],
                             prob_assign_per_obj[obj], GRAPH_EDGE_link);
        }
      }
    }

    // now that we have the complete belief matrix, we want to associate
    // do naive linking
    link(belief, n_active, n_obs);

    // update the iteration counter
    step++;
    current_frame++;
  }

  // have we finished?
  if (current_frame >= frames.back()) {
    statistics.complete = true;
    // clean();
    manager->finalise();
  }

  // return statistics;
}

bool BayesianTracker::update_active() {
  // TODO: MAKE INTERMEDIATE LIST OF TRACKS TO MINIMISE LOOPING OVER
  // EVERYTHING

  // clear the active list
  active.clear();

  for (size_t i = 0, trks_size = manager->num_tracks(); i < trks_size; i++) {
    TrackletPtr trk = manager->get_track(i);

    // check to see whether we have exceeded the bounds
    if (!volume.inside(trk->position())) {
      trk->set_lost();
      continue;
    }

    // if the track is still active, add it to the update list
    if (trk->active()) {
      active.push_back(trk);
    } else {
      trk->trim(); // remove dummies if this track is lost
    }
  }

  return true;
}

void BayesianTracker::debug_output(const unsigned int frm) const {
  // std::cout << "Tracking objects in Frames " << frm-100 << "-" << frm;
  // std::cout << "... " << std::endl;
  // std::cout << " > Currently tracking " << active.size();
  // std::cout << " objects..." << std::endl;
  // std::cout << " - Lost " << num_lost << " tracks for greater than ";
  // std::cout << max_lost << " frames. Removing..." << std::endl;
  // std::cout << " + Started " << tracks.size()-active.size();
  // std::cout << " new tracklets..." << std::endl;
  // std::cout << " ~ Found " << num_conflicts << " conflicts..." <<
  // std::endl;
}

double BayesianTracker::prob_update_motion(const TrackletPtr &trk,
                                           const TrackObjectPtr &obj) const {
  double prob_assign = 0.;

  // calculate the probability that this is the correct track
  prob_assign =
      ProbabilityDensityFunctions::multivariate_erf(trk, obj, this->accuracy);

  // set the probability of assignment to zero if the track is currently
  // in a metaphase state and the object to link to is anaphase
  if (DISALLOW_METAPHASE_ANAPHASE_LINKING) {
    if (trk->track.back()->label == STATE_metaphase &&
        obj->label == STATE_anaphase) {
      // set the probability of assignment to zero
      prob_assign = 0.0;
    }
  }

  // disallow incorrect linking
  if (DISALLOW_PROMETAPHASE_ANAPHASE_LINKING) {
    if (trk->track.back()->label == STATE_prometaphase &&
        obj->label == STATE_anaphase) {
      // set the probability of assignment to zero
      prob_assign = 0.0;
    }
  }

  if (PROB_ASSIGN_EXP_DECAY) {
    // apply an exponential decay according to number of lost
    // drops to 50% at max lost
    double a = std::pow(2, -(double)trk->lost / (double)max_lost);
    prob_assign = a * prob_assign;
  }

  return prob_assign;
}

double BayesianTracker::prob_update_visual(const TrackletPtr &trk,
                                           const TrackObjectPtr &obj) const {
  double prob_assign = 0.;
  prob_assign = ProbabilityDensityFunctions::cosine_similarity(trk, obj);
  return prob_assign;
}

// make the cost matrix of all possible linkages
void BayesianTracker::cost_EXACT(Eigen::Ref<Eigen::MatrixXd> belief,
                                 const size_t n_tracks, const size_t n_objects,
                                 const bool use_uniform_prior) {
  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some variables for Bayesian updates
  double uniform_prior = 1. / (n_objects + 1);
  double prior_assign, posterior, safe_update;

  // start by intializing the belief matrix with a uniform prior
  if (use_uniform_prior) {
    belief.fill(uniform_prior);
  }

  // Posterior is a misnoma here because it is initially the prior, but
  // becomes the posterior
  Eigen::VectorXd v_posterior;
  Eigen::VectorXd v_update = Eigen::VectorXd(n_objects + 1);

  for (size_t trk = 0; trk != n_tracks; trk++) {
    // make space for the update
    // v_posterior = belief.col(trk);
    v_posterior = belief.col(trk);

    // loop through each candidate object
    for (size_t obj = 0; obj != n_objects; obj++) {
      // call the assignment function
      double prob_assign = (this->*m_update_fn)(active[trk], new_objects[obj]);

      // now do the bayesian updates
      prior_assign = v_posterior(obj);

      std::tie(safe_update, posterior) =
          BayesianUpdateFunctions::safe_bayesian_update(
              prior_assign, prob_assign, prob_not_assign);

      v_update.fill(safe_update);
      v_update(obj) =
          1.; // this means the posterior at obj will not be updated?

      // do the update
      v_posterior = v_posterior.array() * v_update.array();
      v_posterior(obj) = posterior;
    }

    // now update the entire column (i.e. track)
    belief.col(trk) = v_posterior;
  }

  // set the timings
  double t_elapsed_ms =
      (std::clock() - t_update_start) / (double)(CLOCKS_PER_SEC / 1000);
  statistics.t_update_belief = static_cast<float>(t_elapsed_ms);
}

// make the cost matrix of all possible linkages
void BayesianTracker::cost_APPROXIMATE(Eigen::Ref<Eigen::MatrixXd> belief,
                                       const size_t n_tracks,
                                       const size_t n_objects,
                                       const bool use_uniform_prior) {
  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some variables for Bayesian updates
  double prior_assign, posterior, safe_update;

  // Posterior is a misnoma here because it is initially the prior, but
  // becomes the posterior
  Eigen::VectorXd v_posterior;
  Eigen::VectorXd v_update = Eigen::VectorXd(n_objects + 1);

  // make a bin map of the objects
  ObjectBin m_cube = ObjectBin(max_search_radius, 1);
  for (size_t obj = 0; obj != n_objects; obj++) {
    m_cube.add_object(new_objects[obj]);
  }

  // iterate over the tracks
  for (size_t trk = 0; trk != n_tracks; trk++) {
    // make space for the update
    // v_posterior = belief.col(trk);
    v_posterior = belief.col(trk);

    // get the local objects for updating
    std::vector<TrackObjectPtr_and_Index> local_objects;
    local_objects = m_cube.get(active[trk], false);
    size_t n_local_objects = local_objects.size();

    // if there are no local objects, then this track is lost
    // HOWEVER: only mark as lost if we're assuming no prior information
    if (n_local_objects < 1 && use_uniform_prior) {
      v_posterior.fill(0.0);        // all objects have zero probability
      v_posterior(n_objects) = 1.0; // the lost probability is one
      belief.col(trk) = v_posterior;
      continue;
    }

    // TODO(arl):
    // now that we know which local updates are to be made, approximate all
    // of the updates that we would have made, set the prior probabilities
    //
    // calculate the local uniform prior for only those objects that we have
    // selected the local objects

    if (use_uniform_prior) {
      double local_uniform_prior = 1. / (n_local_objects + 1);

      for (size_t obj = 0; obj != n_local_objects; obj++) {
        v_posterior(local_objects[obj].second) = local_uniform_prior;
      }
      // set the lost prior also
      v_posterior(n_objects) = local_uniform_prior;
    }

    // loop through each candidate object
    for (size_t obj = 0; obj != n_local_objects; obj++) {
      // call the assignment function
      double prob_assign =
          (this->*m_update_fn)(active[trk], local_objects[obj].first);

      // now do the bayesian updates
      prior_assign = v_posterior(local_objects[obj].second);

      std::tie(safe_update, posterior) =
          BayesianUpdateFunctions::safe_bayesian_update(
              prior_assign, prob_assign, prob_not_assign);

      v_update.fill(safe_update);

      // NOTE(arl): Is this necessary?
      v_update(local_objects[obj].second) =
          1.; // this means the posterior at obj will not be updated?

      // do the update
      v_posterior = v_posterior.array() * v_update.array();
      v_posterior(local_objects[obj].second) = posterior;

    } // objects

    // now update the entire column (i.e. track)
    belief.col(trk) = v_posterior;
  }

  // set the timings
  double t_elapsed_ms =
      (std::clock() - t_update_start) / (double)(CLOCKS_PER_SEC / 1000);
  statistics.t_update_belief = static_cast<float>(t_elapsed_ms);
}

// make the cost matrix of all possible linkages
void BayesianTracker::link(Eigen::Ref<Eigen::MatrixXd> belief,
                           const size_t n_tracks, const size_t n_objects) {
  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some space for used objects
  std::set<unsigned int> not_used;
  for (size_t i = 0; i < n_tracks; i++) {
    not_used.insert(not_used.end(), i);
  }

  // make a track map
  HypothesisMap<LinkHypothesis> map = HypothesisMap<LinkHypothesis>(n_objects);

  for (size_t trk = 0; trk < n_tracks; trk++) {
    // get the object with the best match for this track...
    Eigen::MatrixXf::Index best_object;
    double prob = belief.col(trk).maxCoeff(&best_object);

    // prevents cases of NaN
    if (std::isnan(prob)) {
      prob = 0.0;
    }

    // since we're using zero-indexing, n_objects is equivalent to the index
    // of the last object + 1, i.e. the column for the lost hypothesis...
    if (int(best_object) != int(n_objects)) {
      // push this putative linkage to the map
      map.push(best_object, LinkHypothesis(trk, prob));

    } else {
      // this track is probably lost, append a dummy to the trajectory
      active[trk]->append_dummy(use_motion_features());
      not_used.erase(trk);
      n_lost++;

      // update the statistics
      statistics.p_lost = prob;
    }
  }

  // now loop through the map
  for (size_t obj = 0, map_size = map.size(); obj < map_size; obj++) {
    unsigned int n_links = map[obj].size();

    // this is a direct correspondence, make the mapping
    if (n_links == 1) {
      //  std::cout << map[trk].size() << std::endl;
      LinkHypothesis lnk = map[obj][0];

      unsigned int trk = lnk.first;

      if (not_used.count(trk) < 1) {
        // TODO(arl): make this error more useful
        std::cout << "ERROR: Exhausted potential linkages." << std::endl;
        continue;
      }

      // make sure that we only make links that are possible
      if (euclidean_dist(active[trk], new_objects[obj]) > max_search_radius &&
          CLIP_MAXIMUM_LINKAGE_DISTANCE) {
        continue;
      }

      // append the new object onto the track
      active[trk]->append(new_objects[obj], use_motion_features());

      // update the statistics
      statistics.p_link = lnk.second;

      // since we've found a correspondence for this one, remove from set
      not_used.erase(trk);

    } else if (n_links < 1) {
      // this object has no matches, add a new tracklet
      TrackletPtr trk = std::make_shared<Tracklet>(
          get_new_ID(), new_objects[obj], max_lost, this->motion_model);
      // tracks.push_back( trk );
      manager->push_track(trk);

#if !STRICT_TRACKLET_LINKING
    } else if (n_links > 1) {
      // conflict, get the best one
      n_conflicts++;

      unsigned int trk;
      double prob = -1.;
      for (size_t i = 0; i < n_links; i++) {
        if (map[obj][i].second > prob) {
          prob = map[obj][i].second;
          trk = map[obj][i].first;
        }
      }

      if (not_used.count(trk) < 1) {
        // TODO(arl): make this error more useful
        std::cout << "ERROR: Exhausted potential linkages." << std::endl;
        continue;
      }

      // make sure that we only make links that are possible
      if (euclidean_dist(active[trk], new_objects[obj]) > max_search_radius &&
          CLIP_MAXIMUM_LINKAGE_DISTANCE) {
        continue;
      }

      // update only this one
      active[trk]->append(new_objects[obj], use_motion_features());

      // since we've found a correspondence for this one, remove from set
      not_used.erase(trk);
      // */
    }
#else
    }
#endif
  }

  // get a vector of updates
  std::vector<unsigned int> to_update(not_used.begin(), not_used.end());

  for (size_t i = 0, update_size = to_update.size(); i < update_size; i++) {
    // update these tracks
    active[to_update[i]]->append_dummy(use_motion_features());
  }

  // set the timings
  double t_elapsed_ms =
      (std::clock() - t_update_start) / (double)(CLOCKS_PER_SEC / 1000);
  statistics.t_update_link = static_cast<float>(t_elapsed_ms);

  // update the statistics
  statistics.n_active = n_tracks;
  statistics.n_lost = n_lost;
  statistics.n_conflicts = n_conflicts;
  statistics.n_tracks = this->size();
}

int main(int, char **) {
  //
  // BayesianTracker b;
}
