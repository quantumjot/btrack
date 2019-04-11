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


FATE_APOPTOSIS = 5

import btypes

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


class Node(object):
    """ Node object to store tree structure and underlying track data """
    def __init__(self, track=None, depth=0):
        self.root = True
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
        if self.leaf:
            return []
        return [self.left, self.right]

    @property
    def ID(self):
        return self.track.ID

    @property
    def start(self):
        return self.track.t[0]
    @property
    def end(self):
        return self.track.t[-1]





class LineageTree(object):
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
        self.tracks = sorted(tracks, key=lambda trk:trk.t[0], reverse=False)

    def get_track_by_ID(self, ID):
        """ return the track object with the corresponding ID """
        return [t for t in self.tracks if t.ID==ID][0]

    def create(self):
        """ build the lineage tree """

        used = []
        self.trees = []

        # iterate over the tracks and add them into the growing binary trees
        for trk in self.tracks:
            if trk not in used:

                root = Node(track=trk)
                used.append(trk)

                if trk.children:
                    # follow the tree here
                    queue = [root]

                    while len(queue) > 0:
                        q = queue.pop(0)
                        children = q.track.children
                        if children:
                            # make the left node, then the right
                            left_track = self.get_track_by_ID(children[0])
                            right_track = self.get_track_by_ID(children[1])

                            # set the children of the current node
                            q.left = Node(track=left_track, depth=q.depth+1)
                            q.right = Node(track=right_track, depth=q.depth+1)

                            # append the left and right children to the queue
                            queue.append(q.left)
                            queue.append(q.right)

                            # flag as used, do not need to revisit
                            used.append(left_track)
                            used.append(right_track)


                # append the root node
                self.trees.append(root)

        return self.trees

    def plot(self):
        """ plot the trees """
        plotter = LineageTreePlotter()
        for t in self.trees:
            plotter.plot([t])





class LineageTreePlotter(object):
    """ Plotter for lineage trees.

               o-----------X
               |
      o--------o
      |        |
      |        o--------------------
     -o
      |
      o-----------------------------

    Notes:
        This is ugly, and needs cleaning up!

    """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the position iterator """
        self.y = 0

    def plot(self, tree):

        queue, marked, y_pos = [], [], []

        #put the start vertex into the queue, and the marked list
        queue.append(tree[0])
        marked.append(tree[0])
        y_pos.append(0)

        # store the line coordinates that need to be plotted
        line_list = []
        text_list = []
        marker_list = []

        # now step through
        while len(queue) > 0:
            # pop the root from the tree
            node = queue.pop(0)
            y = y_pos.pop(0)

            # draw the root of the tree
            line_list.append(([y,y], [node.start,node.end]))
            marker_list.append((y, node.start,'k.'))

            # mark if this is an apoptotic tree
            if node.leaf:
                if node.track.fate == FATE_APOPTOSIS:
                    marker_list.append((y, node.end, 'rx'))
                    text_list.append((y, node.end, str(node.ID), 'r'))
                else:
                    marker_list.append((y, node.end, 'ks'))
                    text_list.append((y, node.end, str(node.ID), 'k'))

            if tree[0].ID == node.ID:
                text_list.append((y, node.start, str(node.ID), 'b'))


            for child in node.children:
                if child not in marked:

                    # mark the children
                    marked.append(child)
                    queue.append(child)

                    # calculate the depth modifier
                    depth_mod = 2./(2.**(node.depth-1.))

                    if child == node.children[0]:
                        y_pos.append(y+depth_mod)
                    else:
                        y_pos.append(y-depth_mod)

                    # plot a linking line to the children
                    line_list.append(([y,y_pos[-1]], [node.end,child.start]))
                    marker_list.append((y,node.end,'go'))
                    text_list.append((y_pos[-1],
                                      child.end-(child.end-child.start)/2.,
                                      str(child.ID), 'k'))


        # now that we have traversed the tree, calculate the span
        tree_span = []
        for line in line_list:
            tree_span.append(line[0][0])
            tree_span.append(line[0][1])

        min_x = min(tree_span)
        max_x = max(tree_span)

        # now do the plotting
        y_offset = self.y - min_x + 1
        for line in line_list:
            x = line[0]
            y = line[1]
            plt.plot([xx+y_offset for xx in x],y,'k-')

        # markers
        for marker in marker_list:
            plt.plot(marker[0]+y_offset,marker[1],marker[2])

        # labels
        for txt_label in text_list:
            plt.text(txt_label[0]+y_offset-0.1,
                     txt_label[1]-0.1, txt_label[2], fontsize=8,
                     path_effects=[PathEffects.withStroke(linewidth=1,foreground='w')],
                     color=txt_label[3])

        # update the position for next round
        self.y = y_offset + max_x + 1
