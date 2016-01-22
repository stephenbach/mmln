import logging

import mmln


class Learner:

    def __init__(self, network):
        self.n = network
        self.all_labels = mmln.get_all_labels(network)
        self.logger = logging.getLogger(__name__)

    def learn(self, model, inf=None):
        raise NotImplementedError('This class is abstract.')


class HomophilyLearner(Learner):

    def learn(self, model, inf=None):
        model.regularization = 0.01
        model.inter_node_pos_same_label_default = 0.1
        p = mmln.estimate_p_values_inter_node(self.n)
        for label1 in p:
            for label2 in p[label1]:
                if p[label1][label2] < 0.1:
                    model.inter_node_pos[(label1, label2)] = 2
                elif p[label1][label2] < 0.2:
                    model.inter_node_pos[(label1, label2)] = 1
                elif p[label1][label2] < 0.5:
                    model.inter_node_pos[(label1, label2)] = 0.5
