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

// TODO(arl): clean up the stdout

#include "manager.h"

bool compare_hypothesis_time(
  const Hypothesis &hypothesis_i,
  const Hypothesis &hypothesis_j
) {
  return hypothesis_i.trk_ID->track[0]->t < hypothesis_j.trk_ID->track[0]->t;
}

bool compare_track_start_time(
  const TrackletPtr &trk_i,
  const TrackletPtr &trk_j
) {
    return trk_i->track[0]->t < trk_j->track[0]->t;
}



// take two tracks and merge the contents, rename the merged track and mark for
// removal at the end
void join_tracks(const TrackletPtr &parent_trk, const TrackletPtr &join_trk)
{
  if (DEBUG) std::cout << join_trk->ID << ",";

  // append the pointers to the objects to the parent track object
  for (size_t i=0; i<join_trk->length(); i++) {
    parent_trk->append(join_trk->track[i], true);
  }

  // set the renamed ID and a flag to remove
  join_trk->renamed_ID = parent_trk->ID;
  join_trk->to_remove(true);

  // set the fate of the parent track to that of the joined track
  parent_trk->fate = join_trk->fate;

  // set the children
  if (join_trk->child_one != 0 &&
      join_trk->child_two != 0 ) {

      if (DEBUG) {
        std::cout << parent_trk->ID << " -> " << "[";
        std::cout << join_trk->child_one << ",";
        std::cout << join_trk->child_two << "]" << std::endl;
      }

      // TODO(arl): raise a warning if these have already been set?
      parent_trk->child_one = join_trk->child_one;
      parent_trk->child_two = join_trk->child_two;
  }
}



// branches
void branch_tracks(const BranchHypothesis &branch)
{
  // makes some local pointers to the tracklets
  TrackletPtr parent_trk = std::get<0>(branch);
  TrackletPtr child_one_trk = std::get<1>(branch);
  TrackletPtr child_two_trk = std::get<2>(branch);

  // local ID for parent track (could be renamed, so need to check)
  unsigned int ID;

  // first check to see whether the parent has been renamed
  if (parent_trk->to_remove()) {
    ID = parent_trk->renamed_ID;
  } else {
    ID = parent_trk->ID;
  }

  // output some details?
  if (DEBUG) {
    std::cout << parent_trk->ID << " (renamed: " << ID << ") {";
    std::cout << child_one_trk->ID << ", ";
    std::cout << child_two_trk->ID << "}";
  }

  // set the parent ID for these children
  child_one_trk->parent = ID;
  child_two_trk->parent = ID;

  // set the parent track children also
  parent_trk->child_one = child_one_trk->ID;
  parent_trk->child_two = child_two_trk->ID;

  // set the fate of the parent as 'divided'
  parent_trk->fate = TYPE_Pdivn;
}



// add a graph edge
void TrackManager::push_edge(
      const TrackObjectPtr &a_node_src,
      const TrackObjectPtr &a_node_dst,
      const float a_score,
      const unsigned int a_edge_type
) {
  PyGraphEdge edge;
  edge.source = a_node_src->ID;
  edge.target = a_node_dst->ID;
  edge.score = a_score;
  edge.type = a_edge_type;
  m_graph_edges.push_back(edge);
}


PyGraphEdge TrackManager::get_edge(const size_t idx) const {

  // TODO: we may want to index beyond the end of the edges stored in the
  // `m_graph_edges` vector, to include those edges built by the hypothesis
  // engine. To do so, we need to index into the hypotheses for branches, links
  // and merges.

  return m_graph_edges[idx];
}

size_t TrackManager::num_edges(void) const {
  return m_graph_edges.size();
}



// split a track to remove forbidden transitions
void TrackManager::split(const TrackletPtr &a_trk,
                         const unsigned int a_label_i,
                         const unsigned int a_label_j) {

  // if this is already flagged to be removed, return
  if (a_trk->to_remove()) return;

  // return if the track is too short
  if (a_trk->track.size() < 2) return;

  // iterate over the track and make a list of indices where splits
  // should occur
  std::vector<unsigned int> split_indices;

  for (size_t i=0; i<a_trk->track.size()-1; i++) {
    if (a_trk->track[i]->label == a_label_i &&
        a_trk->track[i+1]->label == a_label_j) {
          // if we satisfy the condition, push the index
          split_indices.push_back(i+1);
    }
  }

  // return if there is nothing to do
  if (split_indices.empty()) return;

  // TODO(arl): actually split the tracks!
  // std::cout << "Residual splits need to be performed -> " << a_trk->ID << std::endl;

}






TrackletPtr TrackManager::get_track_by_ID(const size_t a_ID) const
{
  for (size_t i=0; i<m_tracks.size(); i++) {
    if (m_tracks[i]->ID == a_ID) {
      // std::cout << "ID: " << a_ID << " --> index: " << i << std::endl;
      return m_tracks[i];
    }
  }

  // what happens if this ID is not found?!
  std::cout << "Track ID: " << a_ID << " not found." << std::endl;
  throw std::runtime_error("Track not found.");
}




// take a list of hypotheses and re-organise the tracks following optimisation
void TrackManager::merge(const std::vector<Hypothesis> &a_hypotheses)
{

  if (a_hypotheses.empty()) {
    if (DEBUG) std::cout << "Hypothesis list is empty!" << std::endl;
    return;
  }

  if (m_tracks.empty()) {
    if (DEBUG) std::cout << "Track manager is empty!" << std::endl;
    return;
  }

  // get the number of hypotheses
  size_t n_hypotheses = a_hypotheses.size();

  // make some space for the different hypotheses
  m_links = HypothesisMap<JoinHypothesis>( m_tracks.size() );
  m_branches = HypothesisMap<BranchHypothesis>( m_tracks.size() );

  // loop through the hypotheses, split into link and branch types
  for (size_t i=0; i<n_hypotheses; i++) {

    Hypothesis h = a_hypotheses[i];

    // set the fate of each track as the 'accepted' hypothesis. these will be
    // overwritten in the link and division events
    h.trk_ID->fate = h.hypothesis;

    switch (h.hypothesis) {

      // linkage
      case TYPE_Plink:
        if (DEBUG) {
          std::cout << "P_link: " << h.trk_ID->ID << "->" << h.trk_link_ID->ID;
          std::cout << " [Score: " << h.probability << "]" << std::endl;
        }

        // push a link hypothesis
        m_links.push(h.trk_ID->ID, JoinHypothesis(h.trk_ID, h.trk_link_ID));
        break;


      // branch
      case TYPE_Pdivn:
        if (DEBUG) {
          std::cout << "P_branch: " << h.trk_ID->ID << "->" << h.trk_child_one_ID->ID;
          std::cout << " [Score: " << h.probability << "]" << std::endl;
          std::cout << "P_branch: " << h.trk_ID->ID << "->" << h.trk_child_two_ID->ID;
          std::cout << " [Score: " << h.probability << "]" << std::endl;
        }

        // push a branch hypothesis
        m_branches.push(h.trk_ID->ID, BranchHypothesis(h.trk_ID,
                                                       h.trk_child_one_ID,
                                                       h.trk_child_two_ID));
        break;


      case TYPE_Pmrge:
        // do nothing if we have a merge
        break;

    }
  }


  /* Merge the tracklets.

    i. Traverse the list of linkages.
    ii. Take the first tracklet, append subsequent objects to that tracklet,
        do not update object model
    iii. Rename subsequent tracklets
    iv. set the parent flags for the tracks
    v. Remove merged tracks

  */

  // NOTE(arl): do the branches first?
  // OK, now that we've merged all of the tracks, we want to set various flags
  // to show that divisions have occurred

  for (size_t parent_i=0; parent_i<m_branches.size(); parent_i++) {

    if (!m_branches[parent_i].empty()) {
      if (DEBUG) std::cout << "Branch: [";
      branch_tracks(m_branches[parent_i][0]);
      if (DEBUG) std::cout << "]" << std::endl;
    }

  }

  std::set<unsigned int> used;
  unsigned int child_j;

  // let's try to follow the links, iterate over the link hypotheses
  for (size_t parent_i=0; parent_i<m_links.size(); parent_i++) {

    // if we have a linkage, and we haven't already used this...
    if (!m_links[parent_i].empty() && used.count(parent_i)==0) {

      // now follow the chain
      used.emplace(parent_i);
      child_j = m_links[parent_i][0].second->ID;

      // merge the tracks
      if (DEBUG) std::cout << "Merge: [" << parent_i << ",";
      join_tracks(m_links[parent_i][0].first, m_links[parent_i][0].second);

      // mark the child as used
      used.emplace(child_j);

      // traverse the chain
      while(!m_links[child_j].empty()) {
        // merge the next track
        join_tracks(m_links[parent_i][0].first, m_links[child_j][0].second);

        // iterate
        child_j = m_links[child_j][0].second->ID;
        used.emplace(child_j);
      }
      if (DEBUG) std::cout << "]" << std::endl;
    }
  }

  // // Sanity check, have we used all the possible joins?
  // for (size_t parent_i=0; parent_i<m_links.size(); parent_i++) {
  //   if (used.count(parent_i) == 0) {
  //     std::cout << "Parent track: " << m_links[parent_i][0].first << "not used!" << std::endl;
  //   }
  // }

  // TODO(arl): do a final splitting round to make sure that we haven't
  // joined any tracks that have a METAPHASE->ANAPHASE transition
  if (SPLIT_INCORRECTLY_JOINED_TRACKS) {
    for (size_t i=0; i<m_tracks.size(); i++) {
      split(m_tracks[i], STATE_metaphase, STATE_anaphase);
    }
  }

  // erase those tracks marked for removal (i.e. those that have been merged)
  if (DEBUG) std::cout << "Tracks before merge: " << m_tracks.size();

  // remove the tracks if labelled to_remove
  m_tracks.erase( std::remove_if( m_tracks.begin(), m_tracks.end(),
                  [](const TrackletPtr &t) { return t->to_remove(); }),
                  m_tracks.end() );

  // give the user some more output
  if (DEBUG) std::cout << ", now " << m_tracks.size() << std::endl;

  // check that forward and reverse traversal of the trees is correct
  for (size_t i=0; i<m_tracks.size(); i++) {
    if (m_tracks[i]->has_children()){

      unsigned int parent_ID = m_tracks[i]->ID;

      // get pointers to the children and do a reverse look-up to check the
      // parent ID matches, if not, rename it
      TrackletPtr child_i = get_track_by_ID(m_tracks[i]->child_one);
      TrackletPtr child_j = get_track_by_ID(m_tracks[i]->child_two);

      if (child_i->parent != parent_ID) {
        // std::cout << "Mislabeled parent ID " << child_i->ID;
        // std::cout << "->" << parent_ID << std::endl;
        child_i->parent = parent_ID;
      }

      if (child_j->parent != parent_ID) {
        // std::cout << "Mislabeled parent ID " << child_j->ID;
        // std::cout << "->" << parent_ID << std::endl;
        child_j->parent = parent_ID;
      }

    }
  }

  // build the lineage trees
  build_trees();

  // now finalise everything
  finalise();

}




// build lineage trees
void TrackManager::build_trees(void)
{
  // do nothing yet.
  // return;

  // make a set of used tracks
  std::set<unsigned int> used;

  // sort the tracks in time order
  std::sort(m_tracks.begin(), m_tracks.end(), compare_track_start_time);

  for (size_t i=0; i<m_tracks.size(); i++) {

    // has this track already been used?
    if (used.count(m_tracks[i]->ID) < 1) {

      // create a new node and associate the track with it
      LineageTreeNode root_node = LineageTreeNode(m_tracks[i]);

      // since we know this is a root node, set the generational depth to zero
      root_node.m_depth = 0;

      // check to see whether the track has children, if so, traverse the tree
      if (root_node.has_children())
      {
        // start a queue
        std::vector<LineageTreeNode> queue;
        queue.push_back(root_node);

        while (!queue.empty())
        {
          // get the first item and erase it (i.e. pop front)
          LineageTreeNode node = queue[0];
          queue.erase(queue.begin());

          if (node.has_children()) {

            // get the tracks by ID
            TrackletPtr track_left = get_track_by_ID(node.m_track->child_one);
            TrackletPtr track_right = get_track_by_ID(node.m_track->child_two);

            LineageTreeNode left_node = LineageTreeNode(track_left);
            LineageTreeNode right_node = LineageTreeNode(track_right);

            // set the generational depth as one greater than the parent node
            unsigned int generation_of_children = node.m_depth + 1;
            left_node.m_depth = generation_of_children;
            right_node.m_depth = generation_of_children;

            // update the tracks with the correct root note ID
            track_left->root = root_node.m_track->ID;
            track_right->root = root_node.m_track->ID;

            // set the tracklets generational depth
            track_left->generation = generation_of_children;
            track_right->generation = generation_of_children;

            // push these nodes onto the queue for discovery
            queue.push_back(left_node);
            queue.push_back(right_node);

            // flag these as used
            used.insert(track_left->ID);
            used.insert(track_right->ID);

          } // has_children

        } // queue empty
      } // root has_children

    // push this onto the vector of trees
    m_trees.push_back(root_node);

    // NOTE(arl): this is probably unnecessary, but we will do it anyway
    // add this to the used list
    used.insert(m_tracks[i]->ID);

    } // is track already used?

  } // i
}




// finalise the tracks by trimming them to length and making a list of the
// dummy objects used
void TrackManager::finalise(void)
{
  if (DEBUG) std::cout << "Finalising all tracks..." << std::endl;

  // set the global dummy ID counter here
  int dummy_ID = -1;
  m_dummies.clear();

  // iterate over the tracks, trim and renumber
  for (size_t i=0; i<m_tracks.size(); i++) {

    // first trim any tracks to remove any trailing dummy objects
    m_tracks[i]->trim();

    // now give the dummies unique IDs
    for (size_t o=0; o<m_tracks[i]->track.size(); o++) {
      if (m_tracks[i]->track[o]->dummy) {
        m_tracks[i]->track[o]->ID = dummy_ID;
        dummy_ID--;
        // add this dummy to the list
        m_dummies.push_back(m_tracks[i]->track[o]);
      }
    }

    // finally, if the root node remains unset, change it to the track ID
    if (m_tracks[i]->root == 0) {
      m_tracks[i]->root = m_tracks[i]->ID;
    }

    // if the parent node remains unset, change it to the track ID
    if (m_tracks[i]->parent == 0) {
      m_tracks[i]->parent = m_tracks[i]->ID;
    }

  }
}




TrackObjectPtr TrackManager::get_dummy(const int a_idx) const
{
  // first check that we're trying to get a dummy object (ID should be neg)
  assert(a_idx<0);
  assert(!m_dummies.empty());

  unsigned int dummy_idx = std::abs(a_idx+1);

  // sanity check that we've actually got a dummy
  assert(m_dummies[dummy_idx]->dummy);

  // return the dummies
  return m_dummies[dummy_idx];
}
