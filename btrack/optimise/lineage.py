#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:     BayesianTracker
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  14/08/2014
#-------------------------------------------------------------------------------

__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"


import os
import json

from collections import OrderedDict

from btrack import btypes
from btrack import dataio


class LineageTreeNode:
    """ LineageTreeNode

    Node object to store tree structure and underlying track data

    Args:
        track: the Track object
        depth: depth of the node in the binary tree (should be > parent)
        root: is this the root node?

    Properties:
        left: pointer to left node
        right: pointer to right node
        leaf: returns whether this is also a leaf node (no children)
        children: returns the left and right nodes together
        ID: returns the track ID
        start: start time
        end: end time

    Notes:

    """
    def __init__(self,
                 track=None,
                 root=False,
                 depth=0):

        assert(isinstance(root, bool))
        assert(depth>=0)

        self.root = root
        self.left = None
        self.right = None
        self.track = track
        self.depth = depth

    @property
    def leaf(self):
        return not all([self.left, self.right])

    @property
    def children(self):
        """ return references to the children (if any) """
        if self.leaf: return []
        return [self.left, self.right]

    @property
    def ID(self): return self.track.ID

    @property
    def start(self): return self.track.t[0]

    @property
    def end(self): return self.track.t[-1]

    def to_dict(self):
        """ convert the whole tree (from this node onward) to a dictionary """
        return tree_to_dict(self)

    @property
    def filename(self): return self.track.filename



def tree_to_dict(root):
    """ tree_to_dict

    Convert a tree to a JSON compatible dictionary.  Traverses the tree and
    returns a dictionary structure which can be output as a JSON file.

    Recursive implementation, hopefully there are no loops!

    The JSON representation should look like this:

    {
      "name": "1",
      "children": [
        {
          "name": "2"
        },
        {
          "name": "3"
        }
      ]
    }

    Args:
        root: a root LineageTreeNode

    Returns:
        a dictionary representation of the tree.

    """

    tree = []

    assert(isinstance(root, LineageTreeNode))
    tree.append(("name", str(int(root.ID))))
    tree.append(("start", root.start))
    tree.append(("end", root.end))
    if root.children:
        tree.append(("children", [tree_to_dict(root.left),tree_to_dict(root.right)]))
    # return tree
    return OrderedDict(tree)



def export_tree_to_json(tree, filename):
    """ export a tree to JSON format for visualisation """

    #TODO(arl): proper type checking here
    assert(isinstance(tree, dict))
    assert(isinstance(filename, str))

    with open(filename, 'w') as json_file:
        json.dump(tree, json_file, indent=2, separators=(',', ': '))


def create_and_export_trees_to_json(export_dir,
                                    cell_type,
                                    ignore_single_tracks=True):
    """ create trees from tracks and export a single tree file

    Args:
        export_dir: the directory with the json track files, also where trees
            will be saved
        cell_type: a cell type, e.g. 'GFP'
        ignore_single_tracks: ignore non-trees when exporting

    """

    # set the correct trees filename
    trees_file = os.path.join(export_dir, "trees_{}.json".format(cell_type))

    lineage_tree = LineageTree.from_json(export_dir, cell_type)
    lineage_tree.create()
    json_trees = [tree_to_dict(t) for t in lineage_tree.trees]

    if ignore_single_tracks:
        json_trees = [t for t in json_trees if 'children' in list(t.keys())]

    # now write out the trees
    with open(trees_file, 'w') as json_file:
        json.dump(json_trees, json_file, indent=2, separators=(',', ': '))



def linearise_tree(root_node):
    """ Linearise a tree, i.e. return a list of track objects in the
    tree, but lose the heirarchy

    Essentially the inverse of tree calculation. Useful for plotting.
    """
    assert(isinstance(root_node, LineageTreeNode))
    queue = [root_node]
    linear = []
    while queue:
        node = queue.pop(0)
        linear.append(node)
        if node.children:
            queue.append(node.left)
            queue.append(node.right)
    return linear



class LineageTree:
    """ LineageTree

    Build a lineage tree from track objects.


    Args:
        tracks: a list of Track objects, typically imported from a json/xml file

    Methods:
        get_track_by_ID: return the track object with the corresponding ID
        create: create the lineage trees by performing a BFS
        plot: plot the tree/trees

    Notes:
        Need to update plotting and return other stats from the trees

    """
    def __init__(self, tracks):

        assert(isinstance(tracks, list))

        if not all([isinstance(trk, btypes.Tracklet) for trk in tracks]):
            raise TypeError('Tracks should be of type Track')

        # sort the tracks by the starting frame
        # self.tracks = sorted(tracks, key=lambda trk:trk.t[0], reverse=False)
        tracks.sort(key=lambda trk:trk.t[0], reverse=False)
        self.tracks = tracks


    def get_track_by_ID(self, ID):
        """ return the track object with the corresponding ID """
        return [t for t in self.tracks if t.ID==ID][0]

    def create(self, update_tracks=True):
        """ build the lineage tree """

        used = []
        self.trees = []

        # iterate over the tracks and add them into the growing binary trees
        for trk in self.tracks:
            if trk not in used:

                if not trk.is_root:
                    continue

                root = LineageTreeNode(track=trk, root=True)
                used.append(trk)

                if trk.children:
                    # follow the tree here
                    queue = [root]

                    while len(queue) > 0:
                        q = queue.pop(0)
                        children = q.track.children
                        if children:
                            # make the left node, then the right
                            left = self.get_track_by_ID(children[0])
                            right = self.get_track_by_ID(children[1])

                            # set the children of the current node
                            d = q.depth + 1 # next level from parent
                            q.left = LineageTreeNode(track=left, depth=d)
                            q.right = LineageTreeNode(track=right, depth=d)

                            # append the left and right children to the queue
                            queue.append(q.left)
                            queue.append(q.right)

                            # flag as used, do not need to revisit
                            used.append(left)
                            used.append(right)

                            if update_tracks:
                                left.root = root.ID
                                right.root = root.ID

                # append the root node
                self.trees.append(root)

        return self.trees

    def plot(self):
        """ plot the trees """
        plotter = LineageTreePlotter()
        for t in self.trees:
            plotter.plot([t])

    @property
    def linear_trees(self):
        """ return each tree as a linear list of tracks """
        linear_trees =[linearise_tree(t) for t in self.trees]
        return linear_trees

    @staticmethod
    def from_xml(filename, cell_type=None):
        """ create a lineage tree from an XML file """
        tracks = dataio.read_XML(filename, cell_type=cell_type)
        return LineageTree(tracks)

    @staticmethod
    def from_json(tracks_dir, cell_type=None):
        """ create a lineage tree from an XML file """
        tracks = dataio.read_JSON(tracks_dir, cell_type)
        return LineageTree(tracks)






if __name__ == "__main__":
    pass
