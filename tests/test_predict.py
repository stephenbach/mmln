from unittest import TestCase
import networkx as nx

import mmln


class TestPredictor(TestCase):

    label1 = 'Label 1'
    label2 = 'Label 2'

    def test_predict(self):
        model = mmln.Model()
        net = self._get_network()
        predictor = mmln.Predictor(net)
        predictor.predict(model)

        self.assertEqual(len(predictor.get_per_label_score()), 2)
        self.assertEqual(predictor.get_per_label_score()[self.label1], 0.75)
        self.assertEqual(predictor.get_per_label_score()[self.label2], 0.5)

    def _get_network(self):
        net = nx.Graph()
        net.add_node(1)
        net.add_node(2)
        net.add_node(3)
        net.add_node(4)

        net.add_edge(1, 2)
        net.add_edge(2, 3)
        net.add_edge(3, 1)
        net.add_edge(3, 4)

        net.node[1][mmln.OBSVS] = {}
        net.node[2][mmln.OBSVS] = {}
        net.node[3][mmln.OBSVS] = {self.label2: 0}
        net.node[4][mmln.OBSVS] = {self.label1: 1, self.label2: 1}

        net.node[1][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[2][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[3][mmln.TARGETS] = {self.label1: 0}
        net.node[4][mmln.TARGETS] = {}

        net.node[1][mmln.TRUTH] = {self.label1: 0, self.label2: 1}
        net.node[2][mmln.TRUTH] = {self.label1: 1, self.label2: 0}
        net.node[3][mmln.TRUTH] = {self.label1: 1}
        net.node[4][mmln.TRUTH] = {}

        return net
