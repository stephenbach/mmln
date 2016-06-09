import mmln

import math
import networkx as nx
import numpy as np
import scipy.sparse
import sklearn.linear_model
import subprocess
import time


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
            obsvs = net.node[node][mmln.OBSVS]
            y[row] = 1.0 if all_labels[i] in obsvs and obsvs[all_labels[i]] == 1 else -1.0
            for j in range(len(all_labels)):
                count = 0
                for neighbor in nx.neighbors(net, node):
                    neighbor_obsvs = net.node[neighbor][mmln.OBSVS]
                    if all_labels[j] in neighbor_obsvs and obsvs[all_labels[j]] == 1:
                        count += 1
                    else:
                        count -= 1
                X[row, j] = float(count)
            row += 1
        X = scipy.sparse.csr_matrix(X)

        lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=(1.0 / regularization), fit_intercept=False)
        lr.fit(X, y)

        for j in range(len(all_labels)):
            if abs(lr.coef_[0, j]) > weight_epsilon:
                neighbor_signs[i, j] = 1.0 if lr.coef_[0, j] > 0.0 else -1.0

        print('Finished ' + str(all_labels[i]))

    neighbor_signs = scipy.sparse.csr_matrix(neighbor_signs)

    # Merges collected signs into a network model
    model = mmln.Weights()
    for i in range(len(all_labels)):
        for j in range(i, len(all_labels)):
            if neighbor_signs[i, j] != 0.0 and neighbor_signs[i, j] == neighbor_signs[j, i]:
                model[mmln.INTER][(all_labels[i], all_labels[j])] = neighbor_signs[i, j]

    return model


def is_sign_consistent(model1, model2, weight_epsilon=1e-3):
    for weight_type in (mmln.INTRA, mmln.INTER):
        for m1, m2 in ((model1, model2), (model2, model1)):
            for (l1, l2), weight in m1[weight_type].items():
                if (l1, l2) not in m2[weight_type]:
                    return False

                if weight == 0.0 and m2[weight_type][(l1, l2)] != 0.0:
                    return False

                if math.copysign(1, weight) != math.copysign(1, m2[weight_type][(l1, l2)]):
                    return False
    return True


def select_model_logistic_regression_parallel(net, n_processes, subprocess_script, out_dir, weight_epsilon=1e-3, regularization=1.0):
    all_labels = mmln.get_all_labels(net)
    all_labels = list(all_labels)
    # Sort for reproducibility
    all_labels.sort()

    # Serializes network to disk
    nx.write_gpickle(net, out_dir + '/net.gpickle')

    pool = [None] * n_processes

    # Launches subprocesses
    increment = math.ceil(len(all_labels) / n_processes)
    j = 0
    for i in range(0, len(all_labels), increment):
        script = """
import mmln

import math
import networkx as nx
import numpy as np
import scipy.sparse
import sklearn.linear_model

start = """ + str(i) + """
end = """ + str(min(i + increment, len(all_labels))) + """
out_dir = '""" + out_dir + """'
weight_epsilon = """ + str(weight_epsilon) + """
regularization = """ + str(regularization) + """

net = nx.read_gpickle(out_dir + '/net.gpickle')
all_labels = mmln.get_all_labels(net)
all_labels = list(all_labels)
# Sort for coordination
all_labels.sort()
neighbor_signs = scipy.sparse.lil_matrix((len(all_labels), len(all_labels)))

# Fits LR model for predicting each label and records the signs of the coefficients
for i in range(start, end):
    X = scipy.sparse.lil_matrix((nx.number_of_nodes(net), len(all_labels)))
    y = np.empty((nx.number_of_nodes(net),))
    row = 0
    for node in net.nodes():
        obsvs = net.node[node][mmln.OBSVS]
        y[row] = 1.0 if all_labels[i] in obsvs and obsvs[all_labels[i]] == 1 else -1.0
        for j in range(len(all_labels)):
            count = 0
            for neighbor in nx.neighbors(net, node):
                neighbor_obsvs = net.node[neighbor][mmln.OBSVS]
                if all_labels[j] in neighbor_obsvs and obsvs[all_labels[j]] == 1:
                    count += 1
                else:
                    count -= 1
            X[row, j] = float(count)
        row += 1
    X = scipy.sparse.csr_matrix(X)

    lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=(1.0 / regularization), fit_intercept=False)
    lr.fit(X, y)

    for j in range(len(all_labels)):
        if abs(lr.coef_[0, j]) > weight_epsilon:
            neighbor_signs[i, j] = 1.0 if lr.coef_[0, j] > 0.0 else -1.0

neighbor_signs = scipy.sparse.csr_matrix(neighbor_signs)

with open(out_dir + '/' + str(start) + '.out', 'w') as f:
    for i in range(start, end):
        for j in range(len(all_labels)):
            f.write(str(i) + "\\t" + str(j) + "\\t" + str(neighbor_signs[i, j]) + "\\n")
            """

        pool[j] = subprocess.Popen(['python'], stdin=subprocess.PIPE)
        pool[j].stdin.write(str.encode(script))
        pool[j].stdin.close()
        j += 1

    # Waits for all subprocesses to finish
    finished = False
    while not finished:
        finished = True
        for i in range(len(pool)):
            if pool[i] is not None:
                if pool[i].poll() is not None:
                    pool[i] = None
                else:
                    finished = False
        time.sleep(15)

    # row corresponds to target label, column corresponds to feature label
    neighbor_signs = scipy.sparse.lil_matrix((len(all_labels), len(all_labels)))

    # Reads in results
    for i in range(0, len(all_labels), increment):
        with open(out_dir + '/' + str(i) + '.out', 'r') as f:
            for line in f:
                row = line.strip().split()
                neighbor_signs[int(row[0]), int(row[1])] = float(row[2])

    # Merges collected signs into model weights
    weights = mmln.Weights()
    for i in range(len(all_labels)):
        for j in range(i, len(all_labels)):
            if neighbor_signs[i, j] != 0.0 and neighbor_signs[i, j] == neighbor_signs[j, i]:
                weights[mmln.INTER][(all_labels[i], all_labels[j])] = neighbor_signs[i, j]

    return weights
