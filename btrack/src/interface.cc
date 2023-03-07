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

// export these as C symbols for Python
#define EXTERN_DECL extern "C"

// DLL export for Windows, not fully supported yet(?)
#ifdef _WIN32
  #ifdef BUILD_SHARED_LIB
    #define SHARED_LIB __declspec(dllexport)
  #else
    #define SHARED_LIB __declspec(dllimport)
  #endif
#else
  // this is the default non-windows case
  #define SHARED_LIB
#endif


#include "wrapper.h"

EXTERN_DECL {

  /* =========================================================================
  CREATE AND DELETE THE INTERFACE
  ========================================================================= */


  SHARED_LIB InterfaceWrapper* new_interface() {
    if (DEBUG) {
      std::cout << "InterfaceWrapper constructor called in C++" << std::endl;
    }
    return new InterfaceWrapper();
  }

  SHARED_LIB void del_interface(InterfaceWrapper* h){
    if (DEBUG) {
      std::cout << "InterfaceWrapper destructor called in C++ for ";
      std::cout << h << std::endl;
    }
    delete h;
  }


  /* =========================================================================
  CHECK SHARED LIB VERSION NUMBER
  ========================================================================= */

  SHARED_LIB bool check_library_version( InterfaceWrapper* h,
                              const uint8_t a_major,
                              const uint8_t a_minor,
                              const uint8_t a_build ) {

    return h->check_library_version(a_major, a_minor, a_build);
  }

  /* =========================================================================
  UPDATE METHOD SETTINGS
  ========================================================================= */

  SHARED_LIB void set_update_mode( InterfaceWrapper* h,
                        const unsigned int a_update_mode ) {

    // set the tracker update mode
    h->set_update_mode(a_update_mode);
  }

  SHARED_LIB void set_update_features( InterfaceWrapper* h,
                        const unsigned int a_update_features ) {

    // set the features to use during the tracker update
    h->set_update_features(a_update_features);
  }

  /* =========================================================================
  MOTION MODEL SETTINGS
  ========================================================================= */
  SHARED_LIB void motion( InterfaceWrapper* h,
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
                          double prob_not_assign ) {

      // set the motion_model settings
      h->set_motion_model( measurements, states, A, H, P, Q, R, dt, accuracy,
                           max_lost, prob_not_assign );

  }

  SHARED_LIB void max_search_radius( InterfaceWrapper* h,
                                     const float maximum_search_radius ) {
    if (DEBUG) {
      std::cout << "Set maximum search radius to: ";
      std::cout << maximum_search_radius << std::endl;
    }
    h->set_max_search_radius(maximum_search_radius);
  }

  /* =========================================================================
  APPEND NEW OBJECT
  ========================================================================= */
  SHARED_LIB void append( InterfaceWrapper* h,
              const PyTrackObject new_object ) {
    /* append
    Take a TrackObject and append it to the tracker.
    */
    h->append( new_object );
  }


  /* =========================================================================
  RUN THE TRACKING CODE
  ========================================================================= */
  SHARED_LIB const PyTrackInfo* track( InterfaceWrapper* h ){
    return h->track();
  }

  SHARED_LIB const PyTrackInfo* step( InterfaceWrapper* h, const unsigned int n_steps ){
    //h->step(n_steps);
    return h->step(n_steps);
  }

  /* =========================================================================
  GET A TRACKLET
  ========================================================================= */
  SHARED_LIB unsigned int track_length( InterfaceWrapper *h, const unsigned int trk) {
    return h->track_length(trk);
  }

  SHARED_LIB unsigned int get( InterfaceWrapper* h,
                    double* output,
                    const unsigned int trk ){
    return h->get_track(output, trk);
  }

  SHARED_LIB unsigned int get_refs( InterfaceWrapper* h,
                         int* output,
                         const unsigned int trk ) {
    return h->get_refs(output, trk);
  }

  SHARED_LIB unsigned int get_parent( InterfaceWrapper* h,
                          const unsigned int trk ) {
    return h->get_parent(trk);
  }

  SHARED_LIB unsigned int get_root( InterfaceWrapper* h,
                          const unsigned int trk ) {
    return h->get_root(trk);
  }

  SHARED_LIB unsigned int get_children( InterfaceWrapper* h,
                                        int* children,
                                        const unsigned int trk  ) {
    return h->get_children(children, trk);
  }

  SHARED_LIB unsigned int get_ID( InterfaceWrapper* h,
                                  const unsigned int trk ) {

    return h->get_ID(trk);
  }

  SHARED_LIB unsigned int get_fate( InterfaceWrapper* h,
                                    const unsigned int trk ) {
    return h->get_fate(trk);
  }

  SHARED_LIB unsigned int get_generation( InterfaceWrapper* h,
                                          const unsigned int trk ) {
    return h->get_generation(trk);
  }

  SHARED_LIB unsigned int get_kalman_mu( InterfaceWrapper* h,
                                         double* output,
                                         const unsigned int trk ){
    return h->get_kalman_mu(output, trk);
  }

  SHARED_LIB unsigned int get_kalman_covar( InterfaceWrapper* h,
                                            double* output,
                                            const unsigned int trk ){

    return h->get_kalman_covar(output, trk);
  }

  SHARED_LIB unsigned int get_kalman_pred( InterfaceWrapper* h,
                                           double* output,
                                           const unsigned int trk ){
    return h->get_kalman_pred(output, trk);
  }

  SHARED_LIB unsigned int get_label( InterfaceWrapper* h,
                                     unsigned int* output,
                                     const unsigned int trk ){
    return h->get_label(output, trk);
  }

  SHARED_LIB PyTrackObject get_dummy(InterfaceWrapper* h,
                                     const int obj) {
    return h->get_dummy(obj);
  }

  /* =========================================================================
  RETURN THE NUMBER OF FOUND TRACKS
  ========================================================================= */
  SHARED_LIB unsigned int size( InterfaceWrapper* h ) {
    return h->size();
  }

  /* =========================================================================
  RETURN OR SET THE IMAGING VOLUME
  ========================================================================= */

  SHARED_LIB void get_volume( InterfaceWrapper* h,
                              double* volume ) {
    h->get_volume(volume);
  }

  SHARED_LIB void set_volume(InterfaceWrapper* h,
                             double* volume ) {
    h->set_volume(volume);
  }

  /* =========================================================================
  OPTIMIZER
  ========================================================================= */

  SHARED_LIB unsigned int create_hypotheses(InterfaceWrapper* h,
                                            const PyHypothesisParams params,
                                            const unsigned int a_start_frame,
                                            const unsigned int a_end_frame )
  {
    return h->create_hypotheses(params, a_start_frame, a_end_frame);
  }

  SHARED_LIB PyHypothesis get_hypothesis(InterfaceWrapper* h,
                                         const unsigned int a_ID)
  {
    return h->get_hypothesis(a_ID);
  };

  SHARED_LIB void merge(InterfaceWrapper*h,
                        unsigned int* a_hypotheses,
                        unsigned int n_hypotheses)
  {
    h->merge(a_hypotheses, n_hypotheses);
  }

}
