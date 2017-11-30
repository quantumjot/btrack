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

import numpy as np

def Jaccard(A, B):
    """ Jaccard

    Return the Jaccard coefficient for two sets

    Jaccard metric is the intersection of two sets divided by the union.

    J(A,B) = \frac{A \intersection B}{A \union B}

    Notes:
        The distribution of the flora in the alphine zone.
        Jaccard, P (1912) The New Phytologist, 11(2): 37-50
    """

    # start by converting these to sets
    A = set(A)
    B = set(B)

    intersection = A.intersection(B)
    union = A.union(B)

    return float(len(intersection)) / float(len(union))



def Dice(A, B):
    """ Dice

    Return the Dice coefficient for two sets

    Dice metric is the intersection of two sets divided by sum of the
    cardinality.

    D(A,B) = 2. \cdot \frac{A \intersection B}{|A| + |B|}

    Notes:
        Measures of the amount of ecologic association between species.
        Dice, LR (1945) Ecology, 26(3): 297-302
    """

    # start by converting these to sets
    A = set(A)
    B = set(B)

    intersection = A.intersection(B)
    sumsets = len(A) + len(B)

    return 2.* float(len(intersection)) / float(sumsets)






def MOTA(stats):
    """ Return the Multiple Object Tracking Accuracy (MOTA)

        MOTA = 1 - \frac{\sum_{t} (FN_t + FP_t + IS_t)}{\sum_t g_t}

    Notes:
        "The MOTA accounts for all object configuration errors made
        by the tracker, false positives, misses, mismatches, over all
        frames. It is similar to metrics widely used in other domains
        ... and gives a very intuitive measure of
        the tracker's performance at detecting objects and keeping
        their trajectories, independent of the precision with which
        the object locations are estimated."

        Evaluating multiple object tracking performance. Bernardin and
        Stiefelhargin (2008) EURASIP Journal on Image and Video Processing

    """

    raise NotImplementedError




def MOTP(stats):
    """ Return the Multiple Object Tracking Precision (MOTP)

        MOTP = \frac{\sum_{i,t} d_t^i}{\sum_t c_t}

    Notes:
        "It is the total error in estimated position for matched
        object-hypothesis pairs over all frames, averaged
        by the total number of matches made. It shows
        the ability of the tracker to estimate precise object
        positions, independent of its skill at recognizing
        object configurations, keeping consistent trajectories,
        and so forth."

        Evaluating multiple object tracking performance. Bernardin and
        Stiefelhargin (2008) EURASIP Journal on Image and Video Processing

    """

    raise NotImplementedError
