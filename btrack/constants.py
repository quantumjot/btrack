import enum
import os
from pathlib import Path

BTRACK_PATH = os.path.dirname(os.path.abspath(__file__))
BTRACK_LIB_PATH = Path(BTRACK_PATH) / "libs" / "libtracker"

MAX_SEARCH_RADIUS = 100
DEFAULT_LOW_PROBABILITY = -1e5
MAX_LOST = 5
PROB_NOT_ASSIGN = 0.1
DEBUG = True
EXPORT_FORMATS = frozenset([".json", ".mat", ".hdf5"])
VOLUME = ((0, 1024), (0, 1024), (-1e5, 1e5))
GLPK_OPTIONS = {"tm_lim": 60_000}


DEFAULT_OBJECT_KEYS = ["t", "x", "y", "z", "label"]
DEFAULT_EXPORT_PROPERTIES = [
    "ID",
    "t",
    "x",
    "y",
    "z",
    "parent",
    "root",
    "state",
    "generation",
    "dummy",
]


@enum.unique
class Errors(enum.Enum):
    SUCCESS = 900
    EMPTY_QUEUE = 901
    NO_TRACKS = 902
    NO_USEABLE_FRAMES = 903
    TRACK_EMPTY = 904
    INCORRECT_MOTION_MODEL = 905
    MAX_LOST_OUT_OF_RANGE = 906
    ACCURACY_OUT_OF_RANGE = 907
    PROB_NOT_ASSIGN_OUT_OF_RANGE = 908
    NOT_DEFINED = 909
    NO_ERROR = 910


@enum.unique
class Fates(enum.Enum):
    FALSE_POSITIVE = 0
    INITIALIZE = 1
    TERMINATE = 2
    LINK = 3
    DIVIDE = 4
    APOPTOSIS = 5
    MERGE = 6
    EXTRUDE = 7
    INITIALIZE_BORDER = 10
    INITIALIZE_FRONT = 11
    INITIALIZE_LAZY = 12
    TERMINATE_BORDER = 20
    TERMINATE_BACK = 21
    TERMINATE_LAZY = 22
    DEAD = 666
    UNDEFINED = 999


@enum.unique
class States(enum.Enum):
    INTERPHASE = 0
    PROMETAPHASE = 1
    METAPHASE = 2
    ANAPHASE = 3
    APOPTOSIS = 4
    NULL = 5
    DUMMY = 99


@enum.unique
class BayesianUpdates(enum.Enum):
    EXACT = 0
    APPROXIMATE = 1
    CUDA = 2


@enum.unique
class BayesianUpdateFeatures(enum.Enum):
    MOTION = 0b0000001
    VISUAL = 0b0000010


class Dimensionality(enum.IntEnum):
    TWO: int = 2
    THREE: int = 3
    FOUR: int = 4
