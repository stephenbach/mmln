from unittest import TestCase
import networkx as nx

import mmln


class TestUtil(TestCase):

    label1 = 'Label 1'
    label2 = 'Label 2'

    def test_make_stratified_k_folds(self):
        net = self._get_network()
        folds = mmln.make_stratified_k_folds(net, seed=271828)
        self.assertEqual(len(folds), 2)

    def test_get_all_labels(self):
        net = self._get_network()
        all_labels = mmln.get_all_labels(net)
        self.assertEqual(len(all_labels), 2)

    def test_prune_labels(self):
        net = self._get_network()
        mmln.prune_labels(net, {self.label1})
        for node in range(1, 5):
            self.assertEqual(len(net.node[node][mmln.OBSVS]), 1)
            self.assertTrue(self.label1 in net.node[node][mmln.OBSVS])

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

        net.node[1][mmln.OBSVS] = {self.label1: 1, self.label2: 0}
        net.node[2][mmln.OBSVS] = {self.label1: 0, self.label2: 1}
        net.node[3][mmln.OBSVS] = {self.label1: 1, self.label2: 1}
        net.node[4][mmln.OBSVS] = {self.label1: 0, self.label2: 0}

        return net
