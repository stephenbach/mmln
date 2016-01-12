from unittest import TestCase
import networkx as nx

import mmln
import mmln.ground
import mmln.infer


class TestGroundingManager(TestCase):

    label1 = 'Label 1'
    label2 = 'Label 2'

    def test_set_all_potentials(self):
        model = mmln.Model()
        net = self._get_network()

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.set_all_potentials()
        self.assertEqual(len(inf.pots), 14)

        model.inter_node_pos[(self.label1, self.label1)] = 1
        model.inter_node_pos[(self.label2, self.label2)] = 2

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.set_all_potentials()
        self.assertEqual(len(inf.pots), 14)

        model.intra_node_pos[(self.label1, self.label2)] = 5
        model.intra_node_pos[(self.label2, self.label1)] = 5

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.set_all_potentials()
        self.assertEqual(len(inf.pots), 18)

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

        net.node[1][mmln.TARGETS] = {self.label1: 0}
        net.node[2][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[3][mmln.TARGETS] = {self.label1: 0}
        net.node[4][mmln.TARGETS] = {}

        return net


class _FakeInference(mmln.infer.Inference):

    def __init__(self):
        super(_FakeInference, self).__init__()
        self.pots = set()

    def set_potential(self, weight, coefficients, variables, constant, two_sided=False, squared=False):
        self.pots.add((coefficients, variables, constant, two_sided, squared))

    def infer(self):
        raise NotImplementedError('This class is only for unit testing.')