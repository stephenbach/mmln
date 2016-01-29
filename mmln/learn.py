import logging

import mmln
import mmln.infer
import mmln.ground


class Learner:

    def __init__(self, network):
        self.n = network
        self.all_labels = mmln.get_all_labels(network)
        self.logger = logging.getLogger(__name__)

    def learn(self, model, inf=None):
        raise NotImplementedError('This class is abstract.')


class GradientDescent(Learner):

    def __init__(self, network, n_steps=25, step_size=1.0, step_schedule=True, scale_gradient=True, average_steps=True):
        super(GradientDescent, self).__init__(network)
        self.n_steps = n_steps
        self.step_size = step_size
        self.step_schedule = step_schedule
        self.scale_gradient = scale_gradient
        self.average_steps = average_steps

    def learn(self, model, inf=None):
        self.logger.info('Starting learning. Setting up inference.')
        if inf is None:
            inf = mmln.infer.HLMRF()
        manager = mmln.ground.GroundingManager(model, self.n, self.all_labels, inf)
        manager.init_all_weights()
        self.logger.info('Inference set up. Starting gradient descent.')

        observed_potentials = self._get_observed_potentials(manager)
        scaling_factor = self._get_scaling_factor(manager)

        for i in range(0, self.n_steps):
            inferred_potentials = self._get_inferred_potentials(manager)

            for weight_map in (model.inter_node_pos, model.inter_node_neg, model.intra_node_pos, model.intra_node_neg):
                for (label1, label2), weight in weight_map:
                    step = ()

    def _get_observed_potentials(self, manager):
        pass

    def _get_inferred_potentials(self, manager):
        pass

    def _get_scaling_factor(self, manager):
        pass


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
