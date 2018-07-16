__version__ = '0.2.5'
DEFAULT_LOW_PROBABILITY = -1e5
MAX_LOST = 5
PROB_NOT_ASSIGN = 0.1
DEBUG = True
ERRORS = {901: 'ERROR_empty_queue',
          902: 'ERROR_no_tracks',
          903: 'ERROR_no_useable_frames',
          904: 'ERROR_track_empty',
          905: 'ERROR_incorrect_motion_model',
          906: 'ERROR_max_lost_out_of_range',
          907: 'ERROR_accuracy_out_of_range',
          908: 'ERROR_prob_not_assign_out_of_range',
          909: 'ERROR_not_defined'}
EXPORT_FORMATS = frozenset(['.json','.mat','.hdf5'])
NEW_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
VOLUME = ((0,1024), (0,1024), (-100,100))
