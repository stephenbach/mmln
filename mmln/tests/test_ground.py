from unittest import TestCase
import networkx as nx

import mmln
import mmln.ground
import mmln.infer


class TestGroundingManager(TestCase):

    label1 = 'Label 1'
    label2 = 'Label 2'

    def test_set_all_potentials1(self):
        model = mmln.Model()

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

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.add_all_weights()
        self.assertEqual(len(inf.pots), 14)
        self.assertEqual(inf.set_weight_count, 14)

        model.inter_node_pos[(self.label1, self.label1)] = 1
        model.inter_node_pos[(self.label2, self.label2)] = 2

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.add_all_weights()
        self.assertEqual(len(inf.pots), 14)
        self.assertEqual(inf.set_weight_count, 14)

        model.intra_node_pos[(self.label1, self.label2)] = 5
        model.intra_node_pos[(self.label2, self.label1)] = 5

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.add_all_weights()
        self.assertEqual(len(inf.pots), 18)
        self.assertEqual(inf.set_weight_count, 18)

    def test_set_all_potentials2(self):
        model = mmln.Model()

        net = nx.Graph()
        net.add_node(1)
        net.add_node(2)
        net.add_node(3)

        net.add_edge(1, 2)
        net.add_edge(2, 3)
        net.add_edge(3, 1)

        net.node[1][mmln.OBSVS] = {self.label1: 0}
        net.node[2][mmln.TARGETS] = {self.label1: 0}
        net.node[3][mmln.TARGETS] = {self.label1: 0}

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1}, inf)
        manager.add_all_weights()
        self.assertEqual(len(inf.pots), 8)
        self.assertEqual(inf.set_weight_count, 8)

    def test_set_all_potentials3(self):
        model = mmln.Model()

        net = nx.Graph()
        net.add_node(1)
        net.add_node(2)
        net.add_node(3)
        net.add_node(4)

        net.add_edge(1, 2)
        net.add_edge(2, 3)
        net.add_edge(3, 4)
        net.add_edge(4, 1)

        net.node[1][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[2][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[3][mmln.TARGETS] = {self.label1: 0, self.label2: 0}
        net.node[4][mmln.TARGETS] = {self.label1: 0, self.label2: 0}

        inf = _FakeInference()
        manager = mmln.ground.GroundingManager(model, net, {self.label1, self.label2}, inf)
        manager.add_all_weights()
        self.assertEqual(len(inf.pots), 24)
        self.assertEqual(inf.set_weight_count, 24)


class _FakeInference(mmln.infer.Inference):

    def __init__(self):
        super(_FakeInference, self).__init__()
        self.pots = {}
        self.set_weight_count = 0

    def get_weight(self, coefficients, variables, constant, two_sided=False, squared=False):
        key = (coefficients, variables, constant, two_sided, squared)
        return self.pots[key] if key in self.pots else 0

    def set_weight(self, weight, coefficients, variables, constant, two_sided=False, squared=False):
        self.pots[(coefficients, variables, constant, two_sided, squared)] = weight
        self.set_weight_count += 1

    def infer(self):
        raise NotImplementedError('This class is only for unit testing.')