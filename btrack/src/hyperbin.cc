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

#include "hyperbin.h"

// // set up a HashCube with a certain bin size
// template <typename T>
// HashBin<T>::HashBin( const unsigned int bin_xyz,
//                   const unsigned int bin_n )
// {
//   // default constructor
//   m_bin_size[0] = float(bin_xyz);
//   m_bin_size[1] = float(bin_xyz);
//   m_bin_size[2] = float(bin_xyz);
//   m_bin_size[3] = float(bin_n);
// }
//
//
// // return a hash index given the xyzt
// template <typename T>
// HashIndex HashBin<T>::hash_index(const float x,
//                                  const float y,
//                                  const float z,
//                                  const float n) const
// {
//
//   // set up a hash index structure
//   HashIndex idx;
//   idx.x = static_cast<int> ( floor((1./ m_bin_size[0]) * x) );
//   idx.y = static_cast<int> ( floor((1./ m_bin_size[1]) * y) );
//   idx.z = static_cast<int> ( floor((1./ m_bin_size[2]) * z) );
//   idx.n = static_cast<int> ( floor((1./ m_bin_size[3]) * n) );
//   return idx;
// }
//
//
// // get the contents of the bins surronding the xyz coordinates
// template <typename T>
// std::vector<T> HashBin<T>::get_contents(const HashIndex a_idx)
// {
//
//   // set up a vector to store the contents
//   std::vector<T> r_contents;
//
//   // make a bin index structure to interrogate the matrix
//   HashIndex bin_idx;
//
//   // make a map iterator and search for each of the bins
//   typename std::map< HashIndex, std::vector<T> >::iterator ret;
//
//   // iterate over z
//   for (int z=a_idx.z-1; z<=a_idx.z+1; z++) {
//     bin_idx.z = z;
//
//     // iterate over y
//     for (int y=a_idx.y-1; y<=a_idx.y+1; y++) {
//       bin_idx.y = y;
//
//       // iterate over x
//       for (int x=a_idx.x-1; x<=a_idx.x+1; x++) {
//         bin_idx.x = x;
//
//         // find this bin index and return the list of tracks...
//         ret = m_cube.find(bin_idx);
//
//         // if we didn't find anything...
//         if (ret == m_cube.end()) continue;
//
//         // if we found something, move these into the out queue
//         for (size_t i=0; i<ret->second.size(); i++) {
//           r_contents.push_back( ret->second[i] );
//         } // i
//
//       } // x
//     } // y
//   } // z
//
//   return r_contents;
// }

// A 4D hash cube object.
//
// Essentially a way of binsorting trajectory data for easy lookup,
// thus preventing excessive searching over non-local trajectories.
//
HypercubeBin::HypercubeBin(void) {
    // default constructor
}

HypercubeBin::~HypercubeBin(void) {
    // default destructor
    m_cube.clear();
}

// set up a HashCube with a certain bin size
HypercubeBin::HypercubeBin(const unsigned int bin_xyz,
                           const unsigned int bin_n) {
    // default constructor
    m_bin_size[0] = float(bin_xyz);
    m_bin_size[1] = float(bin_xyz);
    m_bin_size[2] = float(bin_xyz);
    m_bin_size[3] = float(bin_n);
}

// use this to calculate the bin of a certain track
HashIndex HypercubeBin::hash_index(TrackletPtr a_trk,
                                   const bool a_start) const {
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

HashIndex HypercubeBin::hash_index(const float x, const float y, const float z,
                                   const float n) const {
    // set up a hash index structure
    HashIndex idx;
    idx.x = static_cast<int>(floor((1. / m_bin_size[0]) * x));
    idx.y = static_cast<int>(floor((1. / m_bin_size[1]) * y));
    idx.z = static_cast<int>(floor((1. / m_bin_size[2]) * z));
    idx.n = static_cast<int>(floor((1. / m_bin_size[3]) * n));
    return idx;
}

// add a track to the hashcube object
void HypercubeBin::add_tracklet(TrackletPtr a_trk) {
    // get the index of the start (i.e. first object) of the track
    add_tracklet(a_trk, true);
}

// add a track to the hashcube object
void HypercubeBin::add_tracklet(TrackletPtr a_trk, const bool a_start) {
    // get the index of the start (i.e. first object) of the track?
    HashIndex idx = hash_index(a_trk, a_start);

    // try to insert it
    std::pair<std::map<HashIndex, std::vector<TrackletPtr>>::iterator, bool>
        ret;
    std::vector<TrackletPtr> trk_list = {a_trk};
    ret = m_cube.emplace(idx, trk_list);

    // if this key already exists, append it to the list
    if (ret.second == false) {
        m_cube[idx].push_back(a_trk);
    }
}

// std::vector<TrackletPtr> HypercubeBin::get(TrackObjectPtr a_obj)
// {
//
// }

// use this to get the tracks in a certain bin (+/-xyz, but only +n)
std::vector<TrackletPtr> HypercubeBin::get(const TrackletPtr a_trk,
                                           const bool a_start) {
    // space for the tracks to be returned
    std::vector<TrackletPtr> r_trks;

    // make a map iterator and search for each of the bins
    std::map<HashIndex, std::vector<TrackletPtr>>::iterator ret;

    // get the hash_index (either start or end of trajectory)
    HashIndex idx = hash_index(a_trk, a_start);

    // make a bin index structure to interrogate the matrix
    HashIndex bin_idx;

    // iterate over time
    for (int n = idx.n; n <= idx.n + 1; n++) {
        bin_idx.n = n;

        // iterate over z
        for (int z = idx.z - 1; z <= idx.z + 1; z++) {
            bin_idx.z = z;

            // iterate over y
            for (int y = idx.y - 1; y <= idx.y + 1; y++) {
                bin_idx.y = y;

                // iterate over x
                for (int x = idx.x - 1; x <= idx.x + 1; x++) {
                    bin_idx.x = x;

                    // find this bin index and return the list of tracks...
                    ret = m_cube.find(bin_idx);

                    // if we didn't find anything...
                    if (ret == m_cube.end()) continue;

                    // if we found something, move these into the out queue
                    for (size_t i = 0; i < ret->second.size(); i++) {
                        // Note - we need to make sure that the track we return
                        // doesn't start **BEFORE** the track we're searching
                        // for....
                        if (ret->second[i]->track.front()->t >=
                            a_trk->track.back()->t) {
                            r_trks.push_back(ret->second[i]);
                        }

                    }  // i
                }      // x
            }          // y
        }              // z
    }                  // n
    return r_trks;
}

// A 4D hash cube object.
//
// Essentially a way of binsorting trajectory data for easy lookup,
// thus preventing excessive searching over non-local trajectories.
//
ObjectBin::ObjectBin(void) {
    // default constructor
}

ObjectBin::~ObjectBin(void) {
    // default destructor
    m_cube.clear();
}

// set up a HashCube with a certain bin size
ObjectBin::ObjectBin(const unsigned int bin_xyz, const unsigned int bin_n) {
    // default constructor
    m_bin_size[0] = float(bin_xyz);
    m_bin_size[1] = float(bin_xyz);
    m_bin_size[2] = float(bin_xyz);
    m_bin_size[3] = float(bin_n);
}

// use this to calculate the bin of a certain track
HashIndex ObjectBin::hash_index(TrackletPtr a_trk, const bool a_start) const {
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

HashIndex ObjectBin::hash_index(const float x, const float y, const float z,
                                const float n) const {
    // set up a hash index structure
    HashIndex idx;
    idx.x = static_cast<int>(floor((1. / m_bin_size[0]) * x));
    idx.y = static_cast<int>(floor((1. / m_bin_size[1]) * y));
    idx.z = static_cast<int>(floor((1. / m_bin_size[2]) * z));
    idx.n = 1;  // static_cast<int> ( floor((1./ m_bin_size[3]) * n) );
    return idx;
}

// add a track to the hashcube object
void ObjectBin::add_object(TrackObjectPtr a_obj) {
    // get the index of the start (i.e. first object) of the track?
    HashIndex idx = hash_index(a_obj);

    // try to insert it
    std::pair<
        std::map<HashIndex, std::vector<TrackObjectPtr_and_Index>>::iterator,
        bool>
        ret;

    // std::cout << "ObjectBin size: " << n_objects << std::endl;

    std::vector<TrackObjectPtr_and_Index> obj_list = {
        TrackObjectPtr_and_Index(a_obj, n_objects)};

    ret = m_cube.emplace(idx, obj_list);

    n_objects += 1;

    // if this key already exists, append it to the list
    if (ret.second == false) {
        m_cube[idx].push_back(obj_list[0]);
    }
}

// use this to get the tracks in a certain bin (+/-xyz, but only +n)
std::vector<TrackObjectPtr_and_Index> ObjectBin::get(const TrackletPtr a_trk,
                                                     const bool a_start) {
    // space for the tracks to be returned
    std::vector<TrackObjectPtr_and_Index> r_objs;

    // make a map iterator and search for each of the bins
    std::map<HashIndex, std::vector<TrackObjectPtr_and_Index>>::iterator ret;

    // get the hash_index (either start or end of trajectory)
    HashIndex idx = hash_index(a_trk, a_start);

    // make a bin index structure to interrogate the matrix
    HashIndex bin_idx;

    bin_idx.n = 1;

    // iterate over z
    for (int z = idx.z - 1; z <= idx.z + 1; z++) {
        bin_idx.z = z;

        // iterate over y
        for (int y = idx.y - 1; y <= idx.y + 1; y++) {
            bin_idx.y = y;

            // iterate over x
            for (int x = idx.x - 1; x <= idx.x + 1; x++) {
                bin_idx.x = x;

                // find this bin index and return the list of tracks...
                ret = m_cube.find(bin_idx);

                // if we didn't find anything...
                if (ret == m_cube.end()) continue;

                // if we found something, move these into the out queue
                for (size_t i = 0; i < ret->second.size(); i++) {
                    r_objs.push_back(ret->second[i]);
                }  // i
            }      // x
        }          // y
    }              // z

    return r_objs;
}
