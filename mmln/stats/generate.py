import mmln

import math
import networkx as nx
import random


def generate_mmln(weights, all_labels, n_nodes, n_edges_per_node, gibbs_steps=1000, n_mmlns=1):
    # Creates weight maps from model for fast lookups of nonzero weights
    intra_weight_map = {}
    inter_weight_map = {}
    for label in all_labels:
        intra_weight_map[label] = {}
        inter_weight_map[label] = {}
    for weight_type, weight_map in ((mmln.model.INTRA, intra_weight_map), (mmln.model.INTER, inter_weight_map)):
        for (l1, l2), weight in weights[weight_type].items():
            weight_map[l1][l2] = weight
            weight_map[l2][l1] = weight

    for i in range(n_mmlns):
        # Generates a network structure
        net = nx.barabasi_albert_graph(n_nodes, n_edges_per_node)

        # Randomly initializes the network labels
        for node in net.nodes():
            net.node[node][mmln.OBSVS] = {}
            for label in all_labels:
                net.node[node][mmln.OBSVS][label] = 1.0 if random.random() < 0.5 else 0.0

        # Randomly flips labels using Gibbs to draw a sample from the label model
        for i in range(gibbs_steps):
            for node in net.nodes():
                for label in all_labels:
                    score_1 = 0.0
                    score_0 = 0.0

                    # Adds up scores
                    for other_label, weight in intra_weight_map[label]:
                        if net.node[node][mmln.OBSVS][other_label] == 1.0:
                            score_1 += weight
                            score_0 -= weight
                        else:
                            score_0 += weight
                            score_1 -= weight

                    for neighbor in nx.neighbors(net, node):
                        for other_label, weight in inter_weight_map[label].items():
                            if net.node[neighbor][mmln.OBSVS][other_label] == 1.0:
                                score_1 += weight
                                score_0 -= weight
                            else:
                                score_0 += weight
                                score_1 -= weight

                    # Sets label
                    net.node[node][mmln.OBSVS][label] = 1.0 if random.random() < math.exp(score_1) / (math.exp(score_1) + math.exp(score_0)) else 0.0

        return net


def generate_weights(all_labels, intra_density, inter_density):
    all_labels = sorted(all_labels)

    model = mmln.model.Weights()

    for i in range(len(all_labels)):
        for j in range(i+1, len(all_labels)):
            if random.random() < intra_density:
                model[mmln.model.INTRA][(all_labels[i], all_labels[j])] = 1.0 if random.random() < 0.5 else -1.0

    for i in range(len(all_labels)):
        for j in range(i, len(all_labels)):
            if random.random() < inter_density:
                model[mmln.model.INTER][(all_labels[i], all_labels[j])] = 1.0 if random.random() < 0.5 else -1.0

    return model
