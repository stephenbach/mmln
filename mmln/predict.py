import logging
import sklearn.metrics

import mmln
import mmln.ground
import mmln.infer


class Predictor:

    def __init__(self, network):
        self.n = network
        self.all_labels = mmln.get_all_labels(network)
        self.predict_done = False

        self.logger = logging.getLogger(__name__)

    def predict(self, model, inf=mmln.infer.HLMRF()):
        self.logger.info('Starting prediction. Setting up inference.')
        manager = mmln.ground.GroundingManager(model, self.n, self.all_labels, inf)
        manager.set_all_potentials()

        self.logger.info('Inference set up. Starting inference.')
        inf.infer()

        self.logger.info('Inference done. Collecting the results.')
        for node in self.n.nodes():
            if mmln.TARGETS in self.n.node[node]:
                for label in self.n.node[node][mmln.TARGETS]:
                    self.n.node[node][mmln.TARGETS][label] = manager.get_value(node, label)

        self.logger.info('Prediction done.')
        self.predict_done = True

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
