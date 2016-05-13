import math
import networkx as nx
import numpy as np
import scipy.sparse
import sklearn.linear_model

import mmln


def select_model_logistic_regression(net, weight_epsilon=1e-3, regularization=1.0):
    all_labels = mmln.get_all_labels(net)
    all_labels = list(all_labels)
    # Sort for reproducibility
    all_labels.sort()

    # row corresponds to target label, column corresponds to feature label
    neighbor_signs = scipy.sparse.lil_matrix((len(all_labels), len(all_labels)))

    # Fits LR model for predicting each label and records the signs of the coefficients
    for i in range(len(all_labels)):
        X = scipy.sparse.lil_matrix((nx.number_of_nodes(net), len(all_labels)))
        y = np.empty((nx.number_of_nodes(net),))
        row = 0
        for node in net.nodes():
            y[row] = 1.0 if all_labels[i] in net.node[node][mmln.OBSVS] \
                            and net.node[node][mmln.OBSVS][all_labels[i]] == 1 else 0.0
            for j in range(len(all_labels)):
                count = 0
                for neighbor in nx.neighbors(net, node):
                    if all_labels[j] in net.node[neighbor][mmln.OBSVS] \
                            and net.node[neighbor][mmln.OBSVS][all_labels[j]] == 1:
                        count += 1
                X[row, j] = float(count)
            row += 1
        X = scipy.sparse.csr_matrix(X)

        lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=(1.0 / regularization), fit_intercept=False)
        lr.fit(X, y)

        for j in range(len(all_labels)):
            if abs(lr.coef_[0, j]) > weight_epsilon:
                neighbor_signs[i, j] = 1.0 if lr.coef_[0, j] > 0.0 else -1.0

    neighbor_signs = scipy.sparse.csr_matrix(neighbor_signs)

    # Merges collected signs into a network model
    model = mmln.Model(inter_same_label_default=0.0)
    for i in range(len(all_labels)):
        for j in range(i, len(all_labels)):
            if neighbor_signs[i, j] != 0.0 and neighbor_signs[i, j] == neighbor_signs[j, i]:
                model[mmln.inter][(all_labels[i], all_labels[j])] = neighbor_signs[i, j]

    return model


def is_sign_consistent(model1, model2):
    for weight_type in (mmln.intra, mmln.inter):
        for m1, m2 in ((model1, model2), (model2, model1)):
            for (l1, l2), weight in m1[weight_type].items():
                if (l1, l2) not in m2[weight_type]:
                    return False

                if weight == 0.0 and m2[weight_type][(l1, l2)] != 0.0:
                    return False

                if math.copysign(1, weight) != math.copysign(1, m2[weight_type][(l1, l2)]):
                    return False
    return True
