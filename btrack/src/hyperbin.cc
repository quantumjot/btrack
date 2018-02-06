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


#include "hyperbin.h"

// A 4D hash cube object.
//
// Essentially a way of binsorting trajectory data for easy lookup,
// thus preventing excessive searching over non-local trajectories.
//
HypercubeBin::HypercubeBin( void ){
  // default constructor
}

HypercubeBin::~HypercubeBin( void ){
  // default destructor
  m_cube.clear();
}

// set up a HashCube with a certain bin size
HypercubeBin::HypercubeBin( const unsigned int bin_xyz,
                            const unsigned int bin_n ) {
  // default constructor
  m_bin_size[0] = float(bin_xyz);
  m_bin_size[1] = float(bin_xyz);
  m_bin_size[2] = float(bin_xyz);
  m_bin_size[3] = float(bin_n);
}

// use this to calculate the bin of a certain track
HashIndex HypercubeBin::hash_index( TrackletPtr a_trk,
                                    const bool a_start ) const {

  // get the appropriate object from the track
  TrackObjectPtr obj;

  if (a_start) {
    obj = a_trk->track.front();
  } else {
    obj = a_trk->track.back();
  }

  // return the hash index for this track
  // return hash_index(obj->x, obj->y, obj->z, obj->t);
  return hash_index(obj);
}

HashIndex HypercubeBin::hash_index( const float x,
                                    const float y,
                                    const float z,
                                    const float n ) const {

  // set up a hash index structure
  HashIndex idx;
  idx.x = static_cast<int> ( floor((1./ m_bin_size[0]) * x) );
  idx.y = static_cast<int> ( floor((1./ m_bin_size[1]) * y) );
  idx.z = static_cast<int> ( floor((1./ m_bin_size[2]) * z) );
  idx.n = static_cast<int> ( floor((1./ m_bin_size[3]) * n) );
  return idx;
}

// add a track to the hashcube object
void HypercubeBin::add(TrackletPtr a_trk){
  // get the index of the start (i.e. first object) of the track
  HashIndex idx = hash_index( a_trk, true );

  // try to insert it
  std::pair<std::map<HashIndex,std::vector<TrackletPtr>>::iterator,bool> ret;
  std::vector<TrackletPtr> trk_list = {a_trk};
  ret = m_cube.emplace( idx, trk_list );

  // if this key already exists, append it to the list
  if (ret.second == false) {
    m_cube[idx].push_back(a_trk);
  }
}

// use this to get the tracks in a certain bin (+/-xyz, but only +n)
std::vector<TrackletPtr> HypercubeBin::get( const TrackletPtr a_trk,
                                            const bool a_start ) {

  // space for the tracks to be returned
  std::vector<TrackletPtr> r_trks;

  // make a map iterator and search for each of the bins
  std::map< HashIndex, std::vector<TrackletPtr> >::iterator ret;

  // get the hash_index (either start or end of trajectory)
  HashIndex idx = hash_index( a_trk, a_start );

  // make a bin index structure to interrogate the matrix
  HashIndex bin_idx;

  // iterate over time
  for (int n=idx.n; n<=idx.n+1; n++) {
    bin_idx.n = n;

    // iterate over z
    for (int z=idx.z-1; z<=idx.z+1; z++) {
      bin_idx.z = z;

      // iterate over y
      for (int y=idx.y-1; y<=idx.y+1; y++) {
        bin_idx.y = y;

        // iterate over x
        for (int x=idx.x-1; x<=idx.x+1; x++) {
          bin_idx.x = x;

            // find this bin index and return the list of tracks...
            ret = m_cube.find(bin_idx);

            // if we didn't find anything...
            if (ret == m_cube.end()) continue;

            // if we found something, move these into the out queue
            for (size_t i=0; i<ret->second.size(); i++) {
              // Note - we need to make sure that the track we return doesn't
              // start **BEFORE** the track we're searching for....
              if (ret->second[i]->track.front()->t >=
                  a_trk->track.back()->t) {
                r_trks.push_back( ret->second[i] );
              }

            }
          } // x
        } // y
      } // z
    } // n


  return r_trks;
}
