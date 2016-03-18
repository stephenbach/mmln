import logging
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.metrics

import mmln
import mmln.ground
import mmln.infer


class AbstractPredictor:

    def __init__(self, network):
        self.n = network
        self.all_labels = mmln.get_all_labels(network)
        self.predict_done = False
        self.logger = logging.getLogger(__name__)

    def get_per_label_predictions(self):
        if not self.predict_done:
            raise Exception('Must call Predictor.predict() first.')

        predictions = {}
        for label in self.all_labels:
            predictions[label] = {}

        for node in self.n.nodes():
            if mmln.TARGETS in self.n.node[node]:
                for label in self.n.node[node][mmln.TARGETS]:
                    predictions[label][node] = self.n.node[node][mmln.TARGETS][label]

        return predictions

    def get_per_label_score(self, metric=sklearn.metrics.roc_auc_score):
        if not self.predict_done:
            raise Exception('Must call Predictor.predict() first.')

        predictions = self.get_per_label_predictions()

        # Computes AUCs
        scores = {}
        for label in self.all_labels:
            y_true = []
            y = []
            for node, prediction in predictions[label].items():
                y_true.append(self.n.node[node][mmln.TRUTH][label])
                y.append(prediction)
            scores[label] = metric(y_true, y)

        return scores


class MRFPredictor(AbstractPredictor):

    def __init__(self, network):
        super(MRFPredictor, self).__init__(network)

    def predict(self, model, inf=None):
        self.logger.info('Starting prediction. Setting up inference.')
        if inf is None:
            inf = mmln.infer.HLMRF()
        manager = mmln.ground.GroundingManager(model, self.n, self.all_labels, inf)
        manager.init_all_weights()

        self.logger.info('Inference set up. Starting inference.')
        inf.infer()

        self.logger.info('Inference done. Collecting the results.')
        for node in self.n.nodes():
            if mmln.TARGETS in self.n.node[node]:
                for label in self.n.node[node][mmln.TARGETS]:
                    self.n.node[node][mmln.TARGETS][label] = manager.get_value(node, label)

        self.logger.info('Prediction done.')
        self.predict_done = True


class LabelPropPredictor(AbstractPredictor):

    def __init__(self, network):
        super(LabelPropPredictor, self).__init__(network)

    def predict(self, lam=1.0):
        self.logger.info('Starting prediction. Setting up inference.')

        # Makes a map from nodes to matrix indices
        node_indices = {}
        next_index = 0
        for node in self.n.nodes():
            node_indices[node] = next_index
            next_index += 1

        # Constructs the network's Laplacian matrix
        L = scipy.sparse.lil_matrix((self.n.number_of_nodes(), self.n.number_of_nodes()))
        for node in self.n.nodes():
            total_weight = 0
            for neighbor in self.n.neighbors(node):
                L[node_indices[node], node_indices[neighbor]] = -1 * self.n.edge[node][neighbor]['weight']
                total_weight += self.n.edge[node][neighbor]['weight']
            L[node_indices[node], node_indices[node]] = 1 + lam * total_weight
        L = scipy.sparse.csc_matrix(L)
        solve = scipy.sparse.linalg.factorized(L)

        self.logger.info('Inference set up. Starting inference.')
        for label in self.all_labels:
            y = np.zeros(self.n.number_of_nodes())

            for node in self.n.nodes():
                if mmln.OBSVS in self.n.node[node] and label in self.n.node[node][mmln.OBSVS]:
                    y[node_indices[node]] = self.n.node[node][mmln.OBSVS][label]

            f = solve(y)

            for node in self.n.nodes():
                if mmln.TARGETS in self.n.node[node] and label in self.n.node[node][mmln.TARGETS]:
                    self.n.node[node][mmln.TARGETS][label] = f[node_indices[node]]

        self.logger.info('Prediction done.')
        self.predict_done = True
