from .learn import Learner
from .model import Model
from .predict import Predictor
from .util import make_stratified_k_folds, get_all_labels

# Constants for defining label maps on network nodes
OBSVS = 'mmln_observations'
TARGETS = 'mmln_targets'
TRUTH = 'mmln_truth'
