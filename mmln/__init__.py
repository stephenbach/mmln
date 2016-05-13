# Constants for defining label maps on network nodes
OBSVS = 'mmln_observations'
TARGETS = 'mmln_targets'
TRUTH = 'mmln_truth'

from .generate import generate_mmln, generate_label_model
from .learn import Learner, HomophilyLearner
from .model import Model, intra, inter
from .predict import MRFPredictor, LabelPropPredictor
from .select import select_model_logistic_regression, is_sign_consistent
from .stats import estimate_p_values_inter_node
from .util import make_stratified_k_folds, get_all_labels, prune_labels, count_labels,\
    count_coocurring_intra_node_labels, count_adjacent_labels
