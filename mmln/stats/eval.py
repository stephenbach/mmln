import mmln

import sklearn.metrics


def get_per_label_score(network, metric=sklearn.metrics.roc_auc_score):
    predictions = get_per_label_predictions(network)

    # Computes scores
    scores = {}
    for label in mmln.get_all_labels(network):
        y_true = []
        y = []
        for node, prediction in predictions[label].items():
            y_true.append(network.node[node][mmln.TRUTH][label])
            y.append(prediction)
        scores[label] = metric(y_true, y)

    return scores


def get_per_label_predictions(network):
    predictions = {}
    for label in mmln.get_all_labels(network):
        predictions[label] = {}

    for node in network.nodes():
        if mmln.TARGETS in network.node[node]:
            for label, value in network.node[node][mmln.TARGETS].items():
                predictions[label][node] = value

    return predictions
