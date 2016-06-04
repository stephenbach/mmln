from unittest import TestCase
import networkx as nx

import mmln


class TestStats(TestCase):

    label1 = 'Label 1'
    label2 = 'Label 2'

    def test_estimate_p_values_inter_node(self):
        net = self._get_network()
        p = mmln.estimate_p_values_inter_node(net, 100)
        self.assertEqual(len(p), 2)

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

        net.node[1][mmln.OBSVS] = {self.label1: 1}
        net.node[2][mmln.OBSVS] = {self.label1: 0, self.label2: 1}
        net.node[3][mmln.OBSVS] = {self.label1: 1, self.label2: 1}
        net.node[4][mmln.OBSVS] = {self.label1: 0, self.label2: 0}

        return net
