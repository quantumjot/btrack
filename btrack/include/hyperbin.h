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

#ifndef _HASH_H_INCLUDED_
#define _HASH_H_INCLUDED_

// #include <python2.7/Python.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>

#include "types.h"
#include "tracklet.h"


// Hash index for use with the hash cube
struct HashIndex {
  int x = 0;
  int y = 0;
  int z = 0;
  int n = 0;

  // comparison operator for hash map, strict weak ordering
  bool operator<(const HashIndex &o) const {
    if (x != o.x) return x < o.x;
    if (y != o.y) return y < o.y;
    if (z != o.z) return z < o.z;
    return n < o.n;
  }
};

// A 4D hash (hyper) cube object.
//
// Essentially a way of binsorting trajectory data for easy lookup,
// thus preventing excessive searching over non-local trajectories.
//
class HypercubeBin
{
public:

  // constructors and destructors
  HypercubeBin();
  HypercubeBin(const unsigned int bin_xyz, const unsigned int bin_n);
  ~HypercubeBin();

  // return a 4D index into the hypercube using the object at the start or
  // end of an exisiting track
  HashIndex hash_index(TrackletPtr a_trk, const bool a_start) const;

  // return a 4D index into the hypercube using an object
  HashIndex hash_index(TrackObjectPtr a_obj) const {
    return hash_index(a_obj->x, a_obj->y, a_obj->z, a_obj->t);
  };

  // return a 4D index into the hypercube using cartesian coordinates (and time)
  HashIndex hash_index( const float x,
                        const float y,
                        const float z,
                        const float n ) const;

  // bin sort a track in the hypercube, default behaviour is to add the tracks
  // so that the bin stores the first object of the track. Use the full version
  // with the a_start flag (set to false) to store the last known position of
  // the track during lookup
  void add_tracklet(TrackletPtr a_trk);
  void add_tracklet(TrackletPtr a_trk, const bool a_start);

  // return tracks found in a bin
  std::vector<TrackletPtr> get(TrackletPtr a_trk, const bool a_start);
  std::vector<TrackletPtr> get(TrackObjectPtr a_obj);

private:
  // bin size x,y,z,t
  float m_bin_size[4] = {0., 0., 0., 0.};

  // a map to the tracks found in a certain bin
  std::map<HashIndex, std::vector<TrackletPtr>> m_cube;
};





#endif
