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

#include "hypothesis.h"



// safe log function
double safe_log(double value)
{
  if (value <= 0.) return std::log(DEFAULT_LOW_PROBABILITY);
  return std::log(value);
};



// calculate the Euclidean distance between the end of one track and the
// beginning of the second track
double link_distance( const TrackletPtr a_trk,
                      const TrackletPtr a_trk_lnk )
{
  Eigen::Vector3d d = a_trk->track.back()->position()
                      - a_trk_lnk->track.front()->position();
  return std::sqrt( d.transpose()*d );
}



// calculate the Euclidean distance between the end of one track and the
// beginning of the second track
double link_time( const TrackletPtr a_trk,
                  const TrackletPtr a_trk_lnk )
{
  return a_trk_lnk->track.front()->t - a_trk->track.back()->t;
}



// count the number of apoptosis events, starting at the terminus of the
// track
unsigned int count_apoptosis(const TrackletPtr a_trk)
{
    return count_state_track(a_trk, STATE_apoptosis, COUNT_STATE_FROM_BACK);
}


// generic state counting from the back/or front of the track
// could also be used to look for mitotic events for example
unsigned int count_state_track( const TrackletPtr a_trk,
                                const unsigned int a_state_label,
                                const bool a_from_back )
{

  // check that we have at least one observation in our track
  assert(a_trk->length()>0);

  // set the counter, direction and state_counter
  unsigned int counter;
  int counter_dir;
  unsigned int n_state = 0;

  // if counting from the back, start at the back and reverse the direction
  if (a_from_back == COUNT_STATE_FROM_BACK) {
    counter = a_trk->length()-1;
    counter_dir = -1;
  } else {
    counter = 0;
    counter_dir = 1;
  }


  while (a_trk->track[counter]->label == a_state_label &&
         counter>=0 && counter<a_trk->length()) {

    // increment the number of apoptoses
    n_state++;
    counter+=counter_dir;

    // immediately exit if we have reached the end
    if (counter<0 || counter>=a_trk->length()) break;
  }

  return n_state;
}










HypothesisEngine::HypothesisEngine( void )
{
  // default constructor
}



HypothesisEngine::HypothesisEngine( const unsigned int a_start_frame,
                                    const unsigned int a_stop_frame,
                                    const PyHypothesisParams& a_params )
{
  //m_num_frames = a_stop_frame - a_start_frame;
  m_frame_range[0] = a_start_frame;
  m_frame_range[1] = a_stop_frame;
  m_params = a_params;

  // tell the user which hypotheses are going to be created
  // ['P_FP','P_init','P_term','P_link','P_branch','P_dead','P_merge']

  if (!m_tracks.empty() || !m_cube.empty()) {
    std::cout << "Resetting hypothesis engine." << std::endl;
    reset();
  }

  if (DEBUG) {
    std::cout << "Hypotheses to generate: " << std::endl;
    std::cout << " - P_FP: " << hypothesis_allowed(TYPE_Pfalse) << std::endl;
    std::cout << " - P_init: " << hypothesis_allowed(TYPE_Pinit) << std::endl;
    std::cout << " - P_term: " << hypothesis_allowed(TYPE_Pterm) << std::endl;
    std::cout << " - P_link: " << hypothesis_allowed(TYPE_Plink) << std::endl;
    std::cout << " - P_branch: " << hypothesis_allowed(TYPE_Pdivn) << std::endl;
    std::cout << " - P_dead: " << hypothesis_allowed(TYPE_Papop) << std::endl;
    std::cout << " - P_merge: " << hypothesis_allowed(TYPE_Pmrge) << std::endl;
  }

  // should do some parameter checking here
  assert(m_params.segmentation_miss_rate >= 0.0 &&
         m_params.segmentation_miss_rate <= 1.0);

  assert(m_params.apoptosis_rate >= 0.0 &&
         m_params.apoptosis_rate <= 1.0);

  // set up a HashCube
  m_cube = HypercubeBin(m_params.dist_thresh, m_params.time_thresh);
}


// reset the engine
void HypothesisEngine::reset( void )
{
  // clear the tracks and the hashcube
  m_tracks.clear();
  m_cube.clear();
}



HypothesisEngine::~HypothesisEngine( void )
{
  reset();
}




void HypothesisEngine::add_track(TrackletPtr a_trk)
{
  // push this onto the list of trajectories
  m_tracks.push_back( a_trk );

  // add this one to the hash cube
  m_cube.add_tracklet( a_trk );
}




// test whether to generate a certain type of hypothesis
bool HypothesisEngine::hypothesis_allowed(const unsigned int a_hypothesis_type) const
{

  //TODO(arl): make sure that the type exists!
  unsigned int bitmask = std::pow(2, a_hypothesis_type);

  //std::cout << a_hypothesis_type << "," << bitmask << std::endl;
  if ((m_params.hypotheses_to_generate & bitmask) == bitmask) {
    return true;
  }

  return false;
}



float HypothesisEngine::dist_from_border( TrackletPtr a_trk,
                                          bool a_start ) const
{
  // Calculate the distance from the border of the field of view
  float min_dist, min_this_dim;
  Eigen::Vector3d xyz;

  // take either the first or last localisation of the object
  if (a_start) {
    xyz = a_trk->track.front()->position();
  } else {
    xyz = a_trk->track.back()->position();
  }

  // set the distance to infinity
  min_dist = kInfinity;

  // find the smallest distance between a point and the edge of the volume
  // NOTE(arl): what if we have zero dimensions?
  for (unsigned short dim=0; dim<3; dim++) {

    // skip a dimension if it does not exist, for example, in a 2D dataset
    // all z values will be zero (or at least, the same)
    if (volume.min_xyz[dim] == volume.max_xyz[dim]) continue;

    min_this_dim = std::min(xyz[dim]-volume.min_xyz[dim],
                            volume.max_xyz[dim]-xyz[dim]);

    if (min_this_dim < min_dist) {
      min_dist = min_this_dim;
    }
  }

  return min_dist;
}



// create the hypotheses
void HypothesisEngine::create( void )
{

  if (m_tracks.size() < 1) return;

  // get the tracks
  m_num_tracks = m_tracks.size();

  // reserve some memory for the hypotheses (at least 5 times the number of
  // trajectories)
  m_hypotheses.clear();
  m_hypotheses.reserve( m_num_tracks*5 );

  TrackletPtr trk;

  // loop through trajectories
  for (size_t i=0; i<m_num_tracks; i++) {

    // get the test track
    trk = m_tracks[i];

    // false positive hypothesis calculated for everything
    Hypothesis h_fp(TYPE_Pfalse, trk);
    h_fp.probability = safe_log( P_FP( trk ) );
    m_hypotheses.push_back( h_fp );

    // distance from the frame border
    float d_start = dist_from_border( trk, true );
    float d_stop = dist_from_border( trk, false );

    // now calculate the initialisation
    if (hypothesis_allowed(TYPE_Pinit)) {
      if (m_params.relax ||
          trk->track.front()->t < m_frame_range[0]+m_params.theta_time ||
          d_start < m_params.theta_dist ) {

        Hypothesis h_init(TYPE_Pinit, trk);
        h_init.probability = safe_log(P_init(trk)) + 0.5*safe_log(P_TP(trk));
        m_hypotheses.push_back( h_init );
      }
    }

    // termination?
    if (hypothesis_allowed(TYPE_Pterm)) {
      if (m_params.relax ||
          trk->track.back()->t > m_frame_range[1]-m_params.theta_time ||
          d_stop < m_params.theta_dist) {

        Hypothesis h_term(TYPE_Pterm, trk);
        h_term.probability = safe_log(P_term(trk)) + 0.5*safe_log(P_TP(trk));
        m_hypotheses.push_back( h_term );
      }
    }

    // NEW apoptosis detection hypothesis
    // modify this for apoptosis
    unsigned int n_apoptosis = count_apoptosis(trk);

    if (hypothesis_allowed(TYPE_Papop) &&
        n_apoptosis >= m_params.apop_thresh) {

      Hypothesis h_apoptosis(TYPE_Papop, trk);
      h_apoptosis.probability = safe_log(P_dead(trk, n_apoptosis))
                                + 0.5*safe_log(P_TP(trk));
      m_hypotheses.push_back( h_apoptosis );
    }

    // manage conflicts
    std::vector<TrackletPtr> conflicts;

    // iterate over all of the tracks in the hash cube
    std::vector<TrackletPtr> trks_to_test = m_cube.get( trk, false );

    for (size_t j=0; j<trks_to_test.size(); j++) {
      // get the track
      TrackletPtr this_trk = trks_to_test[j];

      // calculate the time and distance between this track and the reference
      float d = link_distance(trk, this_trk);
      float dt = link_time(trk, this_trk);

      assert(d>0);

      // if we exceed these continue
      if (d  >= m_params.dist_thresh) continue;
      if (dt >= m_params.time_thresh || dt < 1) continue; // this was one

      // TODO(arl): limit the maximum link distance?
      if (hypothesis_allowed(TYPE_Plink)) {

        // if (trk->track.back()->label == STATE_metaphase &&
        //     this_trk->track.front()->label == STATE_anaphase &&
        //     DISALLOW_METAPHASE_ANAPHASE_LINKING) {
        //       // do nothing
        // } else {

          // if we allow this link, make the hypothesis
          Hypothesis h_link(TYPE_Plink, trk);
          h_link.trk_link_ID = this_trk;
          h_link.probability = safe_log(P_link(trk, this_trk, d, dt))
                              + 0.5*safe_log(P_TP(trk))
                              + 0.5*safe_log(P_TP(this_trk));
          m_hypotheses.push_back( h_link );

        // }
      }

      // append this to conflicts
      conflicts.push_back( this_trk );

    } // j

    // if we have conflicts, this may mean divisions have occurred
    if (conflicts.size() < 2) continue;

    // iterate through the conflicts and put division hypotheses into the
    // list, including links to the children
    for (unsigned int p=0; p<conflicts.size()-1; p++) {
      // get the first putative child
      TrackletPtr child_one = conflicts[p];

      for (unsigned int q=p+1; q<conflicts.size(); q++) {
        // get the second putative child
        TrackletPtr child_two = conflicts[q];

        if (hypothesis_allowed(TYPE_Pdivn)) {

          Hypothesis h_divn(TYPE_Pdivn, trk);
          h_divn.trk_child_one_ID = child_one;
          h_divn.trk_child_two_ID = child_two;
          h_divn.probability = safe_log(P_branch(trk, child_one, child_two))
                              + 0.5*safe_log(P_TP(trk))
                              + 0.5*safe_log(P_TP(child_one))
                              + 0.5*safe_log(P_TP(child_two));
          m_hypotheses.push_back( h_divn );
        }
      } // q
    } // p

  }

}



// FALSE POSITIVE HYPOTHESIS
double HypothesisEngine::P_FP( TrackletPtr a_trk ) const
{
  unsigned int len_track = static_cast<unsigned int>(1.+a_trk->duration());
  return std::pow(m_params.segmentation_miss_rate, len_track);
}



// TRUE POSITIVE HYPOTHESIS
double HypothesisEngine::P_TP( TrackletPtr a_trk ) const
{
  return 1.0 - P_FP(a_trk);
}



// INITIALISATION HYPOTHESIS
double HypothesisEngine::P_init( TrackletPtr a_trk ) const
{
  // Probability of a true initialisation event.  These tend to occur close to
  // the beginning of the sequence or at the periphery of the field of view as
  // objects enter.

  float dist = dist_from_border(a_trk, true);

  // NOTE(arl): if we have 'relax' on, then this hypothesis will be generated
  // regardless of whether the cell is at the periphery of the FOV or not.
  // Therefore, we should clamp the distance to the maximum value, so as not
  // to penalize cells at the centre of the FOV
  dist = std::min(dist, static_cast<float>(m_params.theta_dist));

  double prob[2] = {0.0, 0.0};
  bool init = false;

  if (a_trk->track.front()->t < m_frame_range[0]+m_params.theta_time) {
    prob[0] = std::exp(-(a_trk->track.front()->t-(float)m_frame_range[0]+1.0) /
              m_params.lambda_time);
    init = true;
  }

  if (dist < m_params.theta_dist || m_params.relax) {
    prob[1] = std::exp(-dist/m_params.lambda_dist);
    init = true;
  }

  if (init) {
    return std::max(prob[0], prob[1]);
  }

  // default return
  return m_params.eta;
}



// TERMINATION HYPOTHESIS
double HypothesisEngine::P_term( TrackletPtr a_trk ) const
{
  // Probability of termination event.  Similar to initialisation, except that
  // we use the final location/time of the tracklet.

  float dist = dist_from_border(a_trk, false);

  // NOTE(arl): if we have 'relax' on, then this hypothesis will be generated
  // regardless of whether the cell is at the periphery of the FOV or not.
  // Therefore, we should clamp the distance to the maximum value, so as not
  // to penalize cells at the centre of the FOV
  dist = std::min(dist, static_cast<float>(m_params.theta_dist));

  double prob[2] = {0.0, 0.0};
  bool term = false;

  if (m_frame_range[1]-a_trk->track.back()->t < m_params.theta_time) {
    prob[0] = std::exp(-((float)m_frame_range[1]-a_trk->track.back()->t) /
              m_params.lambda_time );
    term = true;
  }

  if (dist < m_params.theta_dist || m_params.relax) {
    prob[1] = std::exp(-dist/m_params.lambda_dist);
    term = true;
  }

  if (term) {
    return std::max(prob[0], prob[1]);
  }

  // default return
  return m_params.eta;
}


// EXTRUSION HYPOTHESIS
double HypothesisEngine::P_extrude( TrackletPtr a_trk ) const
{
  // Probability of an extrusion event, similar to a termination event.
  // TODO(arl): implement this
  return m_params.eta;
}



// APOPTOSIS HYPOTHESIS
double HypothesisEngine::P_dead(TrackletPtr a_trk,
                                const unsigned int n_apoptosis ) const
{
  // want to discount this by how close it is to the border of the field of
  // view - this is to make sure this is a genuine apoptosis and not just a
  // track leaving the field of view

  // float dist = dist_from_border(a_trk, false);
  // float discount = 1.0 - std::exp(-dist/m_params.lambda_dist);
  float discount = 1.0;

  // TODO(arl): rather than calculate the probability as the number of apoptotic
  // observations, perhaps the fraction of the track length that is apoptotic
  // is more appropriate?

  float p_apoptosis = 0.;

  if (USE_ABSOLUTE_APOPTOSIS_COUNTS) {
    p_apoptosis = 1.0 - std::pow(m_params.apoptosis_rate, n_apoptosis);
  } else {
    p_apoptosis = static_cast<float>(n_apoptosis) / static_cast<float>(a_trk->length());
  }

  // sanity check
  assert(p_apoptosis>=0. && p_apoptosis<=1.);

  return p_apoptosis * discount;
}

double HypothesisEngine::P_dead( TrackletPtr a_trk ) const
{
  unsigned int n_apoptosis = count_apoptosis(a_trk);
  return P_dead(a_trk, n_apoptosis);
}



// LINKING HYPOTHESIS
double HypothesisEngine::P_link(TrackletPtr a_trk,
                                TrackletPtr a_trk_lnk) const
{
  float d = link_distance(a_trk, a_trk_lnk);
  float dt = link_time(a_trk, a_trk_lnk);

  // return the full version output
  return P_link(a_trk, a_trk_lnk, d, dt);
}

double HypothesisEngine::P_link(TrackletPtr a_trk,
                                TrackletPtr a_trk_lnk,
                                float d,
                                float dt) const
{

  // make sure that we're looking forward in time, this should never be needed
  assert(dt>0.0);

  // // try to not link metaphase to anaphase
  // if (DISALLOW_METAPHASE_ANAPHASE_LINKING) {
  //   if (a_trk->track.back()->label == STATE_metaphase &&
  //       a_trk_lnk->track.front()->label == STATE_anaphase) {
  //
  //     std::cout << a_trk->ID << " -> " << a_trk_lnk->ID << " forbidden M->A" << std::endl;
  //     return m_params.eta;
  //   }
  // }

  float link_penalty = 1.0;

  // try to not link metaphase to anaphase
  if (DISALLOW_METAPHASE_ANAPHASE_LINKING) {
    if (a_trk->track.back()->label == STATE_metaphase &&
        a_trk_lnk->track.front()->label == STATE_anaphase) {
      link_penalty = PENALTY_METAPHASE_ANAPHASE_LINKING;
    }
  }

  // DONE(arl): need to penalise longer times between tracks, dt acts as
  // a linear scaling penalty - scale the distance linearly by time
  // return std::exp(-(d*dt)/m_params.lambda_link);
  return std::exp(-(d*link_penalty)/m_params.lambda_link);
}



// DIVISION HYPOTHESIS
// Different possible branches:
//
// State |   Parent  |  Child1  |  Child2  | Weight
// ------|-----------|----------|----------|------------------------------------
//       | Metaphase | Anaphase | Anaphase | WEIGHT_METAPHASE_ANAPHASE_ANAPHASE
//       | Metaphase | Anaphase |    -     | WEIGHT_METAPHASE_ANAPHASE
//       | Metaphase |    -     | Anaphase | WEIGHT_METAPHASE_ANAPHASE
//       | Metaphase |    -     |    -     | WEIGHT_METAPHASE
//       |     -     | Anaphase | Anaphase | WEIGHT_ANAPHASE_ANAPHASE
//       |     -     | Anaphase |    -     | WEIGHT_ANAPHASE
//       |     -     |    -     | Anaphase | WEIGHT_ANAPHASE
//       |     -     |    -     |    -     | Based on prob that tracks are dead
//
// NOTE/TODO(arl): need to penalise making brances to (already) dead tracks,
// since cells fragment in apoptosis

double HypothesisEngine::P_branch(TrackletPtr a_trk,
                                  TrackletPtr a_trk_c0,
                                  TrackletPtr a_trk_c1) const
{

  // calculate the distance between the previous observation and both of the
  // putative children these are the normalising factors for the dot product
  // a dot product < 0 would indicate that the cells are aligned with the
  // metaphase phase i.e. a good division
  Eigen::Vector3d d_c0, d_c1;
  d_c0 = a_trk_c0->track.front()->position() - a_trk->track.back()->position();
  d_c1 = a_trk_c1->track.front()->position() - a_trk->track.back()->position();

  // normalise the vectors to calculate the dot product
  double dot_product = d_c0.normalized().transpose() * d_c1.normalized();

  // initialise variables
  double delta_g;
  double weight;

  // parent is metaphase
  if (a_trk->track.back()->label == STATE_metaphase) {
    if (a_trk_c0->track.front()->label == STATE_anaphase &&
        a_trk_c1->track.front()->label == STATE_anaphase) {

        // BEST
        weight = WEIGHT_METAPHASE_ANAPHASE_ANAPHASE;
    } else if ( a_trk_c0->track.front()->label == STATE_anaphase ||
                a_trk_c1->track.front()->label == STATE_anaphase ) {

        // PRETTY GOOD
        weight = WEIGHT_METAPHASE_ANAPHASE;
    } else {

      // OK
      weight = WEIGHT_METAPHASE;
    }

  // parent is not metaphase
  } else {
    if (a_trk_c0->track.front()->label == STATE_anaphase &&
        a_trk_c1->track.front()->label == STATE_anaphase) {

          // PRETTY GOOD
          weight = WEIGHT_ANAPHASE_ANAPHASE;
    } else if ( a_trk_c0->track.front()->label == STATE_anaphase ||
                a_trk_c1->track.front()->label == STATE_anaphase ) {

          // OK
          weight = WEIGHT_ANAPHASE;
    } else {
      // in this case, none of the criteria are satisfied
      weight = WEIGHT_OTHER + 10.*P_dead(a_trk_c0) + 10.*P_dead(a_trk_c1);

      // return here if none of the criteria are satisfied
      return std::exp(-weight/(2.*m_params.lambda_branch));
    }
  }


  // weighted angle between the daughter cells and the parent
  // use an erf as the weighting function
  // dot product scales between -1 (ideal) where the daughters are on opposite
  // sides of the parent, to 1, where the daughters are close in space on the
  // same side (worst case). Error function will scale these from ~0. to ~1.
  // meaning that the ideal case minimises the delta_g

  delta_g = weight * ((1.-std::erf(dot_product / (3.*kRootTwo)))/2.0);
  return std::exp(-delta_g/(2.*m_params.lambda_branch));
}
