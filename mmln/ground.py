import logging

import mmln
import mmln.infer


class GroundingManager:

    def __init__(self, model, network, all_labels, inference):
        self.m = model
        self.n = network
        self.all_labels = all_labels
        self.inf = inference

        self.logger = logging.getLogger(__name__)

        # Initializes variables and maps labels to nodes that have them as targets
        self.variables = {}
        self.label_map = {}
        for label in self.all_labels:
            self.label_map[label] = set()

        for node in self.n.nodes():
            if mmln.TARGETS in self.n.node[node]:
                for label in self.n.node[node][mmln.TARGETS]:
                    self.variables[(node, label)] = mmln.infer.Variable((node, label))
                    self.label_map[label].add(node)

    def get_value(self, node, label):
        if (node, label) in self.variables:
            return self.variables[(node, label)].value
        else:
            raise Exception('(' + node + ', ' + label + ') is not a target.')

    def add_all_weights(self):
        self.logger.info('Adding all weights. Starting with regularization.')
        for var in self.variables.values():
            self.inf.add_weight(self.m.regularization, 1, var, -0.5, two_sided=True, squared=True)

        self.logger.info('Added ' + str(len(self.variables)) + ' regularization weights. ' +
                         'Starting intra-node weights.')
        pots_set = 0
        for label_pair, weight in self.m.intra_node_pos.items():
            l1 = label_pair[0]
            l2 = label_pair[1]

            for node in self.label_map[l1]:
                var = self.variables[(node, l1)]
                if mmln.TARGETS in self.n.node[node] and l2 in self.n.node[node][mmln.TARGETS]:
                    other_var = self.variables[(node, l2)]
                    self.inf.add_weight(weight, (1, -1), (var, other_var), 0, squared=True)
                    pots_set += 1
                elif mmln.OBSVS in self.n.node[node] and l2 in self.n.node[node][mmln.OBSVS]:
                    obsv = self.n.node[node][mmln.OBSVS][l2]
                    self.inf.add_weight(weight, 1, var, -1 * obsv, squared=True)
                    pots_set += 1

            # Gets the dependencies we missed: (node, l2) where (node, l1) is observed (explicitly or implicitly)
            for node in self.label_map[l2]:
                if mmln.OBSVS in self.n.node[node] and l1 in self.n.node[node][mmln.OBSVS]:
                    other_var = self.variables[(node, l2)]
                    obsv = self.n.node[node][mmln.OBSVS][l1]
                    self.inf.add_weight(weight, -1, other_var, obsv, squared=True)
                    pots_set += 1

        self.logger.info('Added ' + str(pots_set) + ' intra-node weights. ' +
                         'Starting non-default inter-node weights.')
        pots_set = 0
        for label_pair, weight in self.m.inter_node_pos.items():
            l1 = label_pair[0]
            l2 = label_pair[1]

            for node in self.label_map[l1]:
                var = self.variables[(node, l1)]
                for other_node in self.n.neighbors(node):
                    if mmln.TARGETS in self.n.node[other_node] and l2 in self.n.node[other_node][mmln.TARGETS]:
                        other_var = self.variables[(other_node, l2)]
                        self.inf.add_weight(weight, (1, -1), (var, other_var), 0, squared=True)
                        pots_set += 1
                    elif mmln.OBSVS in self.n.node[other_node] and l2 in self.n.node[other_node][mmln.OBSVS]:
                        obsv = self.n.node[other_node][mmln.OBSVS][l2]
                        self.inf.add_weight(weight, 1, var, -1 * obsv, squared=True)
                        pots_set += 1

            # Gets the dependencies we missed: (other_node, l2) where (node, l1) is observed
            for other_node in self.label_map[l2]:
                other_var = self.variables[(other_node, l2)]
                for node in self.n.neighbors(other_node):
                    if mmln.OBSVS in self.n.node[node] and l1 in self.n.node[node][mmln.OBSVS]:
                        obsv = self.n.node[node][mmln.OBSVS][l1]
                        self.inf.add_weight(weight, -1, other_var, obsv, squared=True)
                        pots_set += 1

        self.logger.info('Added ' + str(pots_set) + ' non-default inter-node weights. ' +
                         'Starting default inter-node weights.')
        pots_set = 0
        weight = self.m.inter_node_pos_same_label_default
        for label in self.all_labels:
            if (label, label) not in self.m.inter_node_pos:
                for node in self.label_map[label]:
                    var = self.variables[(node, label)]
                    for other_node in self.n.neighbors(node):
                        if mmln.TARGETS in self.n.node[other_node] and label in self.n.node[other_node][mmln.TARGETS]:
                            other_var = self.variables[(other_node, label)]
                            self.inf.add_weight(weight, (1, -1), (var, other_var), 0, squared=True)
                            pots_set += 1
                        elif mmln.OBSVS in self.n.node[other_node] and label in self.n.node[other_node][mmln.OBSVS]:
                            obsv = self.n.node[other_node][mmln.OBSVS][label]
                            if obsv == 0:
                                self.inf.add_weight(weight, 1, var, -1 * obsv, squared=True)
                                pots_set += 1
                            elif obsv == 1:
                                self.inf.add_weight(weight, -1, var, obsv, squared=True)
                                pots_set += 1
                            else:
                                self.inf.add_weight(weight, 1, var, -1 * obsv, two_sided=True, squared=True)
                                pots_set += 1

        self.logger.info('Added ' + str(pots_set) + ' default inter-node weights. ' +
                         'Done adding weights.')