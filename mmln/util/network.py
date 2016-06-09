import mmln

import random


def make_stratified_k_folds(net, k=None, seed=None, mix_classes=True):
    if seed is not None:
        random.seed(seed)

    all_labels = get_all_labels(net)
    all_labels = list(all_labels)
    # Sort for reproducibility
    all_labels.sort()

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
        raise Exception('Asked for ' + str(k) + ' folds, but label ' + str(min_label) + ' has only ' + str(minimum) +
                        ' ' + str(min_type) + ' instances')

    folds = []
    for i in range(0, k):
        folds.append(net.copy())
        for node in folds[i].nodes():
            folds[i].node[node][mmln.TARGETS] = {}
            folds[i].node[node][mmln.TRUTH] = {}

    node_sets = [None, None]

    for label in all_labels:
        node_sets[0] = label_map_neg[label]
        node_sets[1] = label_map_pos[label]
        for i in (0, 1):
            node_list = list(node_sets[i])
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
    for collection in (mmln.OBSVS, mmln.TARGETS, mmln.TRUTH):
        for node in net.nodes():
            if collection in net.node[node]:
                for label in net.node[node][collection]:
                    all_labels.add(label)
    return all_labels


def prune_labels(net, labels_to_keep):
    for collection in (mmln.OBSVS, mmln.TARGETS, mmln.TRUTH):
        for node in net.nodes():
            if collection in net.node[node]:
                for label in set(net.node[node][collection].keys()).difference(labels_to_keep):
                    del net.node[node][collection][label]


def count_labels(net, label):
    count = 0
    for node in net.nodes():
        if mmln.OBSVS in net.node[node]:
            if label in net.node[node][mmln.OBSVS]:
                if net.node[node][mmln.OBSVS][label] == 1:
                    count += 1
                elif net.node[node][mmln.OBSVS][label] != 0:
                    raise Exception('Only values in {0, 1} are accepted observations. Found ' +
                                    str(net.node[node][mmln.OBSVS][label]) +
                                    ' for (' + str(node) + ', ' + str(label) + ')')
    return count


def count_coocurring_intra_node_labels(net, label1, label2):
    count = 0
    for node in net.nodes():
        if _check_cooccurence(net, node, label1, node, label2):
            count += 1
    return count


def count_adjacent_labels(net, label1, label2):
    count = 0
    for node1, node2 in net.edges():
        if _check_cooccurence(net, node1, label1, node2, label2):
            count += 1
        elif _check_cooccurence(net, node1, label1, node2, label2):
            count += 1

    if label1 == label2:
        count /= 2
    return count


def _check_cooccurence(net, node1, label1, node2, label2):
    if mmln.OBSVS in net.node[node1] and mmln.OBSVS in net.node[node2]:
        if label1 in net.node[node1][mmln.OBSVS] and label2 in net.node[node2][mmln.OBSVS]:
            obsv1 = net.node[node1][mmln.OBSVS][label1]
            obsv2 = net.node[node2][mmln.OBSVS][label2]
            if obsv1 == 1 and obsv2 == 1:
                return True
            elif obsv1 != 0 and obsv1 != 1:
                raise Exception('Only values in {0, 1} are accepted observations. Found ' +
                                str(net.node[node1][mmln.OBSVS][label1]) +
                                ' for (' + str(node1) + ', ' + str(label1) + ')')
            elif obsv2 != 0 and obsv2 != 1:
                raise Exception('Only values in {0, 1} are accepted observations. Found ' +
                                str(net.node[node1][mmln.OBSVS][label1]) +
                                ' for (' + str(node2) + ', ' + str(label2) + ')')

    return False
