import mmln

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class LabelProp(mmln.AbstractModel):

    def __init__(self, lam=1.0):
        self.lam = lam
        super(LabelProp, self).__init__()

    def train(self, network):
        pass

    def predict(self, network):
        self.logger.info('Starting prediction. Setting up inference.')

        # Makes a map from nodes to matrix indices
        node_indices = {}
        next_index = 0
        for node in network.nodes():
            node_indices[node] = next_index
            next_index += 1

        # Constructs the network's Laplacian matrix
        L = scipy.sparse.lil_matrix((network.number_of_nodes(), network.number_of_nodes()))
        for node in network.nodes():
            total_weight = 0
            for neighbor in network.neighbors(node):
                L[node_indices[node], node_indices[neighbor]] = -1 * network.edge[node][neighbor]['weight']
                total_weight += network.edge[node][neighbor]['weight']
            L[node_indices[node], node_indices[node]] = 1 + self.lam * total_weight
        L = scipy.sparse.csc_matrix(L)
        solve = scipy.sparse.linalg.factorized(L)

        self.logger.info('Inference set up. Starting inference.')
        for label in mmln.get_all_labels(network):
            y = np.zeros(network.number_of_nodes())

            for node in network.nodes():
                if mmln.OBSVS in network.node[node] and label in network.node[node][mmln.OBSVS]:
                    y[node_indices[node]] = network.node[node][mmln.OBSVS][label]

            f = solve(y)

            for node in network.nodes():
                if mmln.TARGETS in network.node[node] and label in network.node[node][mmln.TARGETS]:
                    network.node[node][mmln.TARGETS][label] = f[node_indices[node]]

        self.logger.info('Prediction done.')
