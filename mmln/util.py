import random

import mmln


def make_stratified_k_folds(net, k=None, seed=None):
    if seed is not None:
        random.seed(seed)

    all_labels = get_all_labels(net)
    label_map_pos = {}
    label_map_neg = {}
    for label in all_labels:
        label_map_pos[label] = set()
        label_map_neg[label] = set()

    for node in net.nodes():
        if mmln.OBSVS in net.node[node]:
            for label in net.node[node][mmln.OBSVS]:
                if net.node[node][mmln.OBSVS][label] == 1:
                    label_map_pos[label].add(node)
                elif net.node[node][mmln.OBSVS][label] == 0:
                    label_map_neg[label].add(node)
                else:
                    raise Exception('Only values in {0, 1} are accepted observations. Found ' +
                                    str(net.node[node][mmln.OBSVS][label]) + ' for (' + node + ', ' + label + ')')

    minimum = float('inf')
    min_label = None
    for label_map in (label_map_pos, label_map_neg):
        for label, node_set in label_map.items():
            if len(node_set) < minimum:
                minimum = len(node_set)
                min_label = label

    if k is None:
        k = minimum
    elif k > minimum:
        min_type = 'positive' if len(label_map_pos[min_label]) == minimum else 'negative'
        raise Exception('Asked for ' + str(k) + ' folds, but label ' + min_label + ' has only ' + str(minimum) +
                        ' ' + min_type + ' instances')

    folds = []
    for i in range(0, k):
        folds.append(net.copy())
        for node in folds[i].nodes():
            folds[i].node[node][mmln.TARGETS] = {}
            folds[i].node[node][mmln.TRUTH] = {}

    for label in all_labels:
        for node_set in (label_map_pos[label], label_map_neg[label]):
            node_list = list(node_set)
            # Always sorts before shuffling for reproducibility
            node_list.sort()
            if len(node_list) < 2000:
                random.shuffle(node_list)
                for i in range(0, len(node_list)):
                    _make_target(folds[i % k], node_list[i], label)
            else:
                sublist = node_list[0:2000]
                random.shuffle(sublist)
                for i in range(0, 2000):
                    _make_target(folds[i % k], sublist[i], label)
                for i in range(2000, len(node_list)):
                    fold = random.randint(0, k-1)
                    _make_target(folds[fold], node_list[i], label)

    return folds


def _make_target(network, node, label):
    network.node[node][mmln.TRUTH][label] = network.node[node][mmln.OBSVS][label]
    network.node[node][mmln.TARGETS][label] = 0
    del network.node[node][mmln.OBSVS][label]


def get_all_labels(net):
    all_labels = set()
    for node in net.nodes():
        if mmln.OBSVS in net.node[node]:
            for label in net.node[node][mmln.OBSVS]:
                all_labels.add(label)
        if mmln.TARGETS in net.node[node]:
            for label in net.node[node][mmln.TARGETS]:
                all_labels.add(label)
    return all_labels


def prune_labels(net, labels_to_keep):
    for collection in (mmln.OBSVS, mmln.TARGETS, mmln.TRUTH):
        for node in net.nodes():
            if collection in net.node[node]:
                for label in set(net.node[node][collection].keys()).difference(labels_to_keep):
                    del net.node[node][collection][label]


def count_labels(net, label, collections=(mmln.OBSVS,)):
    count = 0
    for collection in collections:
        for node in net.nodes():
            if label in net.node[node][collection]:
                count += 1
    return count


def count_coocurring_intra_node_labels(net, label1, label2, collections=(mmln.OBSVS,)):
    count = 0
    for node in net.nodes():
        for c1 in collections:
            for c2 in collections:
                if label1 in net.node[node][c1] and label2 in net.node[node][c2]:
                    count += 1
    return count


def count_adjacent_labels(net, label1, label2, collections=(mmln.OBSVS,)):
    count = 0
    for node1, node2 in net.edges():
        for c1 in collections:
            for c2 in collections:
                if label1 in net.node[node1][c1] and label2 in net.node[node2][c2]:
                    count += 1
                if label2 in net.node[node1][c1] and label1 in net.node[node2][c2]:
                    count += 1

    if label1 == label2:
        count /= 2
    return count
