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


#include "tracker.h"


// we can assume that the covar matrix is diagonal from the MotionModel
// since only position obeservations are made, therefore we can decompose
// multivariate gaussian into product of univariate gaussians
// http://cs229.stanford.edu/section/gaussians.pdf

double cheat_trivariate_PDF(const Eigen::Vector3d& x, Prediction p)
{

    double prob_density =

    (1./(kRootTwoPi*sqrt(p.covar(0,0)))) * exp(-(1./(2.*p.covar(0,0))) *
    (x(0)-p.mu(0)) * (x(0)-p.mu(0)) ) *
    (1./(kRootTwoPi*sqrt(p.covar(1,1)))) * exp(-(1./(2.*p.covar(1,1))) *
    (x(1)-p.mu(1)) * (x(1)-p.mu(1)) ) *
    (1./(kRootTwoPi*sqrt(p.covar(2,2)))) * exp(-(1./(2.*p.covar(2,2))) *
    (x(2)-p.mu(2)) * (x(2)-p.mu(2)) );

    return prob_density;

}



// we can assume that the covar matrix is diagonal from the MotionModel
// since only position observations are made, therefore we can decompose
// multivariate gaussian into product of univariate gaussians
// http://cs229.stanford.edu/section/gaussians.pdf

// also we need to calculate the probability (the integral), so we use erf
// http://en.cppreference.com/w/cpp/numeric/math/erf

double probability_erf( const Eigen::Vector3d& x,
                        const Prediction p,
                        const double accuracy=2. )
{

  double phi = 1.;
  double phi_x, std_x, d_x;

  for (unsigned int axis=0; axis<3; axis++) {

    std_x = std::sqrt(p.covar(axis,axis));
    d_x = x(axis)-p.mu(axis);

    // intergral +/- accuracy
    phi_x = std::erf((d_x+accuracy) / (std_x*kRootTwo)) -
            std::erf((d_x-accuracy) / (std_x*kRootTwo));

    // calculate product of integrals for the axes i.e. joint probability?
    phi *= .5*phi_x;

  }

  // we don't want a NaN!
  assert(!std::isnan(phi));

  return phi;
}







// void write_belief_matrix_to_CSV(std::string a_filename,
//                                 Eigen::Ref<Eigen::MatrixXd> a_belief)
// {
//   std::cout << a_filename << std::endl;
//   std::ofstream belief_file;
//   belief_file.open(a_filename);
//   belief_file << a_belief.format(CSVFormat);
//   belief_file.close();
// }



















// set up the tracker using an existing track manager
BayesianTracker::BayesianTracker(const bool verbose) {

  // set up verbosity
  this->verbose = verbose;

  // set up the tracks
  this->tracks = TrackManager();

  // NOTE(arl): This isn't really necessary
  // set up the frame map
  frames.clear();

  // set the current frame to zero
  current_frame = 0;

  // reserve some space for the tracks
  active.reserve(RESERVE_ACTIVE_TRACKS);
  new_objects.reserve(RESERVE_NEW_OBJECTS);
}



BayesianTracker::~BayesianTracker() {
  //std::cout << "Destruction of BayesianTracker" << std::endl;
  // clean up gracefully
}



unsigned int BayesianTracker::set_motion_model(
              const unsigned int measurements,
              const unsigned int states,
              double* A_raw,
              double* H_raw,
              double* P_raw,
              double* Q_raw,
              double* R_raw,
              const double dt,
              const double accuracy,
              const unsigned int max_lost,
              const double prob_not_assign)
{

  // do some error checking
  if (prob_not_assign<=0. || prob_not_assign>=1.)
    return ERROR_prob_not_assign_out_of_range;

  if (max_lost>10)
    return ERROR_max_lost_out_of_range;

  if (accuracy<0. || accuracy>10.)
    return ERROR_accuracy_out_of_range;

  this->max_lost = max_lost;
  this->prob_not_assign = prob_not_assign;
  this->accuracy = accuracy;

  if (verbose && DEBUG) {
     std::cout << "MAX_LOST: " << this->max_lost << std::endl;
     std::cout << "ACCURACY: " << this->accuracy << std::endl;
     std::cout << "PROB_NOT_ASSIGN: " << this->prob_not_assign << std::endl;
  }


  typedef Eigen::Matrix<double,
                        Eigen::Dynamic,
                        Eigen::Dynamic,
                        Eigen::RowMajor> RowMajMat;

  // map the arrays to Eigen matrices
  Eigen::MatrixXd H = Eigen::Map<RowMajMat>(H_raw, measurements, states);
  Eigen::MatrixXd Q = Eigen::Map<RowMajMat>(Q_raw, states, states);
  Eigen::MatrixXd P = Eigen::Map<RowMajMat>(P_raw, states, states);
  Eigen::MatrixXd R = Eigen::Map<RowMajMat>(R_raw, measurements, measurements);
  Eigen::MatrixXd A = Eigen::Map<RowMajMat>(A_raw, states, states);

  //set up a new motion model
  motion_model = MotionModel( A, H, P, R, Q );

  return SUCCESS;

}



unsigned int BayesianTracker::append(const PyTrackObject& new_object){
  // take a vector of TrackObjects as input and perform the tracking step

  // from the PyTrackObject create a track object with a shared pointer.
  // This will be stored
  TrackObjectPtr p = std::make_shared<TrackObject>(new_object);

  // NOTE: THIS IS PROBABLY UNNECESSARY UNTIL WE RETURN PyTrackObjects...
  // p->original_object = &new_object;

  // update the imaging volume with this new measurement
  volume.update(p);

  // add a new object and maintain a set of frame numbers...
  objects.push_back( p );
  frames_set.insert( p->t );

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
  while (!statistics.complete){
    step();
  }

}



// initialise the first frame
unsigned int BayesianTracker::initialise() {

  // if we have already initialised, return early
  if (initialised)
    return SUCCESS;

  // check to make sure that we've got some objects to track
  if (objects.empty()) {
    if (verbose) {
      std::cout << "Object queue is empty. " << std::endl;
    }
    return ERROR_empty_queue;
  }

  if (!tracks.empty()) {
    if (verbose) {
      std::cout << "Tracking has already been performed. " << std::endl;
    }
    return ERROR_no_tracks;
  }

  // sort the objects vector by time
  std::sort( objects.begin(), objects.end(), compare_obj_time );

  // NOTE: should check that we have some frames which can be tracked
  // start by converting the set to a vector
  std::vector<unsigned int> f(frames_set.begin(), frames_set.end());
  frames = f;

  bool useable_frames = false;
  for (size_t n=1; n<frames.size(); n++) {
    if ( (frames[n] - frames[n-1]) <= max_lost) {
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
  n_objects = objects.size();
  o_counter = 0;

  // set the current frame of the tracker
  current_frame = frames.front();

  // set up the first tracklets based on the first set of objects
  while ( objects[o_counter]->t == current_frame && o_counter != n_objects ) {
    // add a new tracklet
    TrackletPtr trk = std::make_shared<Tracklet>( get_new_ID(),
                                                  objects[o_counter],
                                                  max_lost,
                                                  this->motion_model );
    tracks.push_back( trk );
    o_counter++;
  }

  // add one to the iteration
  current_frame++;

  // set the initialised flag
  initialised = true;

  return SUCCESS;
}



// update the tracker by some number of steps
void BayesianTracker::step(const unsigned int steps)
{

  // make sure that we have steps greater than zero
  assert(steps>0);

  // reset the step counter
  unsigned int step = 0;

  // first check the iteration, if it is zero, initialise
  // TODO(arl): we don't necessarily start on frame zero?
  //if (current_frame == 0) {
  if (!initialised) {
    // initialise!
    unsigned int ret = initialise();
    if ( ret != SUCCESS ) {
      // return the error in a statistics structure
      statistics.error = ret;
      return;
    }
    // take a step
    step++;
  }


  while (step < steps && current_frame < frames.back()) {

    // update the list of active tracks
    update_active();

    // clear the list of objects
    new_objects.clear();

    // loop over all tracks found in this frame
    while ( objects[o_counter]->t == current_frame && o_counter != n_objects) {
      // store a reference to this object
      new_objects.push_back( objects[o_counter] );
      o_counter++;
    }

    // set up some counters
    size_t n_active = active.size();
    size_t n_obs = new_objects.size();

    // if we have an empty frame, append dummies to everthing and continue
    if (new_objects.empty()) {
      //std::cout << "Frame " << current_frame << " is empty..." << std::endl;
      for (size_t i=0; i<n_active; i++) {
        active[i]->append_dummy();
      }
      step++;
      current_frame++;
      continue;
    }

    // make some space for the belief matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> belief;

    // now do the Bayesian updates
    belief.setZero(n_obs+1, n_active);

    // do we want to do a fast update?
    if (FAST_COST_UPDATE) {
        cost_FAST(belief, n_active, n_obs);
    } else {
        cost(belief, n_active, n_obs);
    }

    // if (WRITE_BELIEF_MATRIX) {
    //   std::stringstream belief_filename;
    //   belief_filename << "/home/arl/Desktop/belief/belief_";
    //   belief_filename << current_frame << ".csv";
    //   write_belief_matrix_to_CSV(belief_filename.str(), belief);
    // }

    // now that we have the complete belief matrix, we want to associate
    // do naive linking
    link(belief, n_active, n_obs);

    // update the iteration counter
    step++;
    current_frame++;

  }

  // have we finished?
  if (current_frame >= frames.back())
  {
    statistics.complete = true;
    //clean();
    tracks.finalise();
  }

  //return statistics;

}



bool BayesianTracker::update_active()
{

  // TODO: MAKE INTERMEDIATE LIST OF TRACKS TO MINIMISE LOOPING OVER EVERYTHING

  // clear the active list
  active.clear();

  for (size_t i=0, trks_size=tracks.size(); i<trks_size; i++) {

    // check to see whether we have exceeded the bounds
    if (!volume.inside( tracks[i]->position() )) {
      tracks[i]->set_lost();
      continue;
    }

    // if the track is still active, add it to the update list
    if (tracks[i]->active()) {
      active.push_back( tracks[i] );
    } else {
      tracks[i]->trim();   // remove dummies if this track is lost
    }

  }

  return true;

}


// // Run a final clean of the data to trim any lost tracks
// bool BayesianTracker::clean() {
//   // trim any tracks
//   for (size_t i=0; i<tracks.size(); i++){
//     tracks[i]->trim();
//   }
//
//   // finalise by renaming any dummy objects
//   tracks.finalise();
//   return true;
// }


void BayesianTracker::debug_output(const unsigned int frm) const
{

  // std::cout << "Tracking objects in Frames " << frm-100 << "-" << frm;
  // std::cout << "... " << std::endl;
  // std::cout << " > Currently tracking " << active.size();
  // std::cout << " objects..." << std::endl;
  // std::cout << " - Lost " << num_lost << " tracks for greater than ";
  // std::cout << max_lost << " frames. Removing..." << std::endl;
  // std::cout << " + Started " << tracks.size()-active.size();
  // std::cout << " new tracklets..." << std::endl;
  // std::cout << " ~ Found " << num_conflicts << " conflicts..." << std::endl;

}



// make the cost matrix of all possible linkages
void BayesianTracker::cost(Eigen::Ref<Eigen::MatrixXd> belief,
                           const size_t n_tracks,
                           const size_t n_objects)
{
  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some variables for Bayesian updates
  Prediction trk_prediction;
  double prob_assign = 0.;
  double uniform_prior = 1. / (n_objects+1);
  double prior_assign, PrDP, posterior, update;

  // set the uniform prior
  belief.fill(uniform_prior);

  // Posterior is a misnoma here because it is initially the prior, but
  // becomes the posterior
  Eigen::VectorXd v_posterior;
  Eigen::VectorXd v_update = Eigen::VectorXd(n_objects+1);

  for (size_t trk=0; trk != n_tracks; trk++) {

    // get the trk prediction
    trk_prediction = active[trk]->predict();

    // make space for the update
    // v_posterior = belief.col(trk);
    v_posterior = belief.col(trk);

    // loop through each candidate object
    for (size_t obj=0; obj != n_objects; obj++) {

      // calculate the probability that this is the correct track
      prob_assign = probability_erf(new_objects[obj]->position(),
                                    trk_prediction,
                                    this->accuracy);

      // set the probability of assignment to zero if the track is currently
      // in a metaphase state and the object to link to is anaphase
      if (DISALLOW_METAPHASE_ANAPHASE_LINKING) {
        if (active[trk]->track.back()->label == STATE_metaphase &&
            new_objects[obj]->label == STATE_anaphase) {

          // set the probability of assignment to zero
          prob_assign = 0.0;
        }
      }

      // disallow incorrect linking
      if (DISALLOW_PROMETAPHASE_ANAPHASE_LINKING) {
        if (active[trk]->track.back()->label == STATE_prometaphase &&
            new_objects[obj]->label == STATE_anaphase) {

          // set the probability of assignment to zero
          prob_assign = 0.0;
        }
      }

      if (PROB_ASSIGN_EXP_DECAY) {
        // apply an exponential decay according to number of lost
        // drops to 50% at max lost
        double a = std::pow(2, -(double)active[trk]->lost/(double)max_lost);
        prob_assign = a*prob_assign;
      }

      // now do the bayesian updates
      // prior_assign = v_posterior(obj);
      prior_assign = v_posterior(obj);
      PrDP = prob_assign * prior_assign + prob_not_assign * (1.-prob_assign);
      posterior = (prob_assign * (prior_assign / PrDP));
      update = (1. + (prior_assign-posterior)/(1.-prior_assign));

      v_update.fill(update);
      v_update(obj) = 1.; // this means the posterior at obj will not be updated?

      // do the update
      v_posterior = v_posterior.array()*v_update.array();
      v_posterior(obj) = posterior;
    }

    // now update the entire column (i.e. track)
    //belief.col(trk) = belief.col(trk).cwiseProduct( v_posterior );
    belief.col(trk) = v_posterior;
  }

  // set the timings
  double t_elapsed_ms = (std::clock() - t_update_start) /
                        (double) (CLOCKS_PER_SEC / 1000);
  statistics.t_update_belief = static_cast<float>(t_elapsed_ms);

}



// make the cost matrix of all possible linkages
void BayesianTracker::cost_FAST(Eigen::Ref<Eigen::MatrixXd> belief,
                                const size_t n_tracks,
                                const size_t n_objects)
{
  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some variables for Bayesian updates
  Prediction trk_prediction;
  double prob_assign = 0.;
  double uniform_prior = 1. / (n_objects+1);
  double prior_assign, PrDP, posterior, update;

  // set the uniform prior
  belief.fill(uniform_prior);

  // Posterior is a misnoma here because it is initially the prior, but
  // becomes the posterior
  Eigen::VectorXd v_posterior;
  Eigen::VectorXd v_update = Eigen::VectorXd(n_objects+1);

  // make a bin map of the objects
  ObjectBin m_cube = ObjectBin(max_search_radius, 1);
  for (size_t obj=0; obj != n_objects; obj++) {
    m_cube.add_object(new_objects[obj]);
  }


  // iterate over the tracks
  for (size_t trk=0; trk != n_tracks; trk++) {

    // get the trk prediction
    trk_prediction = active[trk]->predict();

    // make space for the update
    // v_posterior = belief.col(trk);
    v_posterior = belief.col(trk);

    // get the local objects for updating
    std::vector<TrackObjectPtr_and_Index> local_objects;
    local_objects = m_cube.get(active[trk], false);
    size_t n_local_objects = local_objects.size();

    // loop through each candidate object
    for (size_t obj=0; obj != n_local_objects; obj++) {

      // calculate the probability that this is the correct track
      prob_assign = probability_erf(local_objects[obj].first->position(),
                                    trk_prediction,
                                    this->accuracy);

      // set the probability of assignment to zero if the track is currently
      // in a metaphase state and the object to link to is anaphase
      if (DISALLOW_METAPHASE_ANAPHASE_LINKING) {
        if (active[trk]->track.back()->label == STATE_metaphase &&
            local_objects[obj].first->label == STATE_anaphase) {

          // set the probability of assignment to zero
          prob_assign = 0.0;
        }
      }

      if (PROB_ASSIGN_EXP_DECAY) {
        // apply an exponential decay according to number of lost
        // drops to 50% at max lost
        double a = std::pow(2, -(double)active[trk]->lost/(double)max_lost);
        prob_assign = a*prob_assign;
      }

      // now do the bayesian updates
      // prior_assign = v_posterior(obj);
      prior_assign = v_posterior(local_objects[obj].second);
      PrDP = prob_assign * prior_assign + prob_not_assign * (1.-prob_assign);
      posterior = (prob_assign * (prior_assign / PrDP));
      update = (1. + (prior_assign-posterior)/(1.-prior_assign));

      v_update.fill(update);

      // NOTE(arl): Is this necessary?
      v_update(local_objects[obj].second) = 1.; // this means the posterior at obj will not be updated?

      // do the update
      v_posterior = v_posterior.array()*v_update.array();
      v_posterior(local_objects[obj].second) = posterior;

    }

    // now update the entire column (i.e. track)
    //belief.col(trk) = belief.col(trk).cwiseProduct( v_posterior );
    belief.col(trk) = v_posterior;
  }

  // set the timings
  double t_elapsed_ms = (std::clock() - t_update_start) /
                        (double) (CLOCKS_PER_SEC / 1000);
  statistics.t_update_belief = static_cast<float>(t_elapsed_ms);

}



// make the cost matrix of all possible linkages
void BayesianTracker::link(Eigen::Ref<Eigen::MatrixXd> belief,
                           const size_t n_tracks,
                           const size_t n_objects )
{

  // start a timer
  std::clock_t t_update_start = std::clock();

  // set up some space for used objects
  std::set<unsigned int> not_used;
  for (size_t i=0; i<n_tracks; i++) {
    not_used.insert(not_used.end(), i);
  }

  // make a track map
  HypothesisMap<LinkHypothesis> map = HypothesisMap<LinkHypothesis>(n_objects);

  for (size_t trk=0; trk<n_tracks; trk++) {

    // get the object with the best match for this track...
    Eigen::MatrixXf::Index best_object;
    double prob = belief.col(trk).maxCoeff(&best_object);

    // since we're using zero-indexing, n_objects is equivalent to the index of
    // the last object + 1, i.e. the column for the lost hypothesis...
    if (int(best_object) != int(n_objects)) {
      // push this putative linkage to the map
      map.push( best_object, LinkHypothesis(trk, prob) );

    } else {
      // this track is probably lost, append a dummy to the trajectory
      active[trk]->append_dummy();
      not_used.erase(trk);
      n_lost++;

      // update the statistics
      statistics.p_lost = prob;
    }
  }

  // now loop through the map
  for (size_t obj=0, map_size=map.size(); obj<map_size; obj++) {

    unsigned int n_links = map[obj].size();

    // this is a direct correspondence, make the mapping
    if (n_links == 1) {
      //  std::cout << map[trk].size() << std::endl;
      LinkHypothesis lnk = map[obj][0];

      unsigned int trk = lnk.first;

      if (not_used.count(trk) < 1) {
        // TODO(arl): make this error more useful
        std::cout << "ERROR: Exhausted potential linkages." << std::endl;
      }

      // append the new object onto the track
      active[trk]->append( new_objects[obj] );

      // update the statistics
      statistics.p_link = lnk.second;

      // since we've found a correspondence for this one, remove from set
      not_used.erase(trk);
    } else if (n_links < 1) {
      // this object has no matches, add a new tracklet
      TrackletPtr trk = std::make_shared<Tracklet>( get_new_ID(),
                                                    new_objects[obj],
                                                    max_lost,
                                                    this->motion_model );
      tracks.push_back( trk );

    } else if (n_links > 1) {
      // conflict, get the best one
      n_conflicts++;

      unsigned int trk;
      double prob = -1.;
      for (size_t i=0; i<n_links; i++) {
        if (map[obj][i].second > prob) {
          prob = map[obj][i].second;
          trk = map[obj][i].first;
        }
      }

      if (not_used.count(trk) < 1) {
        // TODO(arl): make this error more useful
        std::cout << "ERROR: Exhausted potential linkages." << std::endl;
      }

      // update only this one
      active[trk]->append( new_objects[obj] );

      // since we've found a correspondence for this one, remove from set
      not_used.erase(trk);
    }
  }


  // get a vector of updates
  std::vector<unsigned int> to_update(not_used.begin(), not_used.end());

  for (size_t i=0, update_size=to_update.size(); i<update_size; i++) {
    // update these tracks
    active[ to_update[i] ]->append_dummy();
  }

  // set the timings
  double t_elapsed_ms = (std::clock() - t_update_start) /
                        (double) (CLOCKS_PER_SEC / 1000);
  statistics.t_update_link = static_cast<float>(t_elapsed_ms);


  // update the statistics
  statistics.n_active = n_tracks;
  statistics.n_lost = n_lost;
  statistics.n_conflicts = n_conflicts;
  statistics.n_tracks = this->size();

}



int main(int, char**)
{
  //
  // BayesianTracker b;

}
