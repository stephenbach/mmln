import random

import mmln


def estimate_p_values_inter_node(net, n_samples=1000, seed=271828):
    random.seed(seed)
    labels = list(mmln.get_all_labels(net))

    p = {}
    for label in labels:
        p[label] = {}

    for i in range(0, len(labels)):
        label1 = labels[i]
        for j in range(i, len(labels)):
            label2 = labels[j]
            obsv = mmln.count_adjacent_labels(net, label1, label2)
            n_label1 = mmln.count_labels(net, label1)
            if label1 == label2:
                n_label2 = n_label1
            else:
                n_label2 = mmln.count_labels(net, label2)

            # Results at least as high as obsv
            pos = 0
            # Results not as high as obsv
            neg = 0

            for sample in range(0, n_samples):
                labelled1 = random.sample(net.nodes(), n_label1)
                if label1 == label2:
                    labelled2 = labelled1
                else:
                    labelled2 = random.sample(net.nodes(), n_label2)

                adj_labels = 0
                for node1 in labelled1:
                    node1_neighbors = set(net.neighbors(node1))
                    for node2 in labelled2:
                        if node2 in node1_neighbors:
                            adj_labels += 1

                if label1 == label2:
                    adj_labels /= 2

                if adj_labels >= obsv:
                    pos += 1
                else:
                    neg += 1

            p[label1][label2] = pos / (pos + neg)
            p[label2][label1] = p[label1][label2]

    return p
