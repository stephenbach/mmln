import mmln

import os
import struct


class MRF(mmln.model.AbstractModel):

    def __init__(self, weights, n_samples=5000, n_learning_epochs = 100, temp_dir='./predict.temp'):
        super(MRF, self).__init__()
        self.weights = weights
        self.n_samples = n_samples
        self.n_learning_epochs = n_learning_epochs
        self.temp_dir = temp_dir
        self.vars_file = temp_dir + '/vars.bin'
        self.weights_file = temp_dir + '/weights.bin'
        self.factors_file = temp_dir + '/factors.bin'
        self.meta_file = temp_dir + '/factor_graph.meta'

    def train(self, network):
        self.logger.info('Starting training. Writing out model.')
        var_map = self.write_vars_for_training(network)
        weight_map = self.write_weights()
        n_factors, n_edges = self.write_factors(network, var_map, weight_map)
        with open(self.meta_file, 'w') as f:
            write_metadata(f, weight_map.size(), len(var_map), n_factors, n_edges)

        self.logger.info('Model written. Calling DimmWitted.')
        value = os.system("dw gibbs -q  -m " + self.meta_file + " -w " + self.weights_file + " -v " + self.vars_file +
                          " -f " + self.factors_file + " --n_learning_epoch " + str(self.n_learning_epochs) +
                          " --n_inference_epoch 0 --n_samples_per_learning_epoch " + str(self.n_samples) +
                          " -o " + self.temp_dir)
        if value != 0:
            raise RuntimeError("DimmWitted finished with nonzero exit code.")

        self.logger.info('DimmWitted finished. Collecting the results.')
        results = []
        with open(self.temp_dir + '/inference_result.out.weights.text', 'r') as f:
            for line in f:
                results.append(line.strip().split()[1])
        if len(results) != self.weights.size():
            raise RuntimeError("DimmWitted output an unexpected number of weights.")
        for weight_type in (mmln.model.OTHER, mmln.model.INTER, mmln.model.INTRA):
            for key, index in weight_map[weight_type].items():
                self.weights[weight_type][key] = results[index]

        self.logger.info('Training done.')

    def predict(self, network):
        self.logger.info('Starting prediction. Writing out model.')
        var_map = self.write_vars_for_prediction(network)
        weight_map = self.write_weights()
        n_factors, n_edges = self.write_factors(network, var_map, weight_map)
        with open(self.meta_file, 'w') as f:
            write_metadata(f, weight_map.size(), len(var_map), n_factors, n_edges)

        self.logger.info('Model written. Calling DimmWitted.')
        value = os.system("dw gibbs -q  -m " + self.meta_file + " -w " + self.weights_file + " -v " + self.vars_file +
                          " -f " + self.factors_file + " --n_learning_epoch 0 --n_inference_epoch " + str(self.n_samples) +
                          " --n_samples_per_learning_epoch 0 -o " + self.temp_dir)
        if value != 0:
            raise RuntimeError("DimmWitted finished with nonzero exit code.")

        self.logger.info('DimmWitted finished. Collecting the results.')
        results = []
        with open(self.temp_dir + '/inference_result.out.text', 'r') as f:
            for line in f:
                results.append(line.strip().split()[2])
        if len(results) != len(var_map):
            raise RuntimeError("DimmWitted output an unexpected number of variables.")
        for (node, label), index in var_map:
            network.nodes.node[mmln.TARGETS][label] = results[index]

        self.logger.info('Prediction done.')

    def write_vars_for_training(self, network):
        var_map = {}

        with open(self.vars_file, 'wb') as f:
            for node in network.nodes():
                # Observations
                if mmln.OBSVS in network.node[node]:
                    for label, value in network.node[node][mmln.OBSVS].items():
                        write_variable(f, len(var_map), False, True, value)
                        var_map[(node, label)] = len(var_map)
                # Targets
                has_truth = mmln.TRUTH in network.node[node]
                if mmln.TARGETS in network.node[node]:
                    for label in network.node[node][mmln.TARGETS]:
                        if has_truth and label in network.node[node][mmln.TRUTH]:
                            write_variable(f, len(var_map), True, False, network.node[node][mmln.TRUTH][label])
                        else:
                            write_variable(f, len(var_map), False, False, 0.0)
                        var_map[(node, label)] = len(var_map)
        return var_map

    def write_vars_for_prediction(self, network):
        var_map = {}

        with open(self.vars_file, 'wb') as f:
            for node in network.nodes():
                for label_type in (mmln.TARGETS, mmln.OBSVS):
                    if label_type in network.nodes[node]:
                        for label, value in network.nodes[node][label_type].items():
                            write_variable(f, len(var_map), False, label_type == mmln.OBSVS, value)
                            var_map[(node, label)] = len(var_map)
        return var_map

    def write_weights(self):
        weight_map = mmln.model.Weights()

        with open(self.weights_file, 'wb') as f:
            for weight_type in (mmln.model.OTHER, mmln.model.INTER, mmln.model.INTRA):
                for key, value in self.weights[weight_type].items():
                    write_weight(f, weight_map.size(), False, value)
                    weight_map[weight_type][key] = weight_map.size()
        return weight_map

    def write_factors(self, network, var_map, weight_map):
        n_factors = 0
        n_edges = 0

        with open(self.factors_file, 'wb') as f:
            # Inter-node dependencies
            for node1, node2 in network.edges():
                for label_type_1 in (mmln.TARGETS, mmln.OBSVS):
                    if label_type_1 in network.nodes[node1]:
                        for label1 in network.nodes[node1][label_type_1]:
                            for label_type_2 in (mmln.TARGETS, mmln.OBSVS):
                                if label_type_2 in network.nodes[node2]:
                                    for label2 in network.nodes[node2][label_type_2]:
                                        if label1 <= label2:
                                            wid = weight_map[mmln.model.INTER][(label1, label2)]
                                        else:
                                            wid = weight_map[mmln.model.INTER][(label2, label1)]
                                        vid1 = var_map[(node1, label1)]
                                        vid2 = var_map[(node2, label2)]
                                        write_pairwise_equal_factor(f, wid, vid1, vid2)
                                        n_factors += 1
                                        n_edges += 2
        return n_factors, n_edges


def write_variable(file, vid, is_evidence, is_observed, initial_value):
    file.write(struct.pack('>q', vid))
    if is_evidence and not is_observed:
        file.write(b'\x01')
    elif not is_evidence and is_observed:
        file.write(b'\x02')
    elif not is_evidence and not is_observed:
        file.write(b'\x00')
    else:
        raise ValueError('Variable cannot be both evidence and observed.')
    file.write(struct.pack('>d', initial_value))
    # dataType is Boolean
    file.write(b'\x00\x00')
    # edgeCount is deprecated so write out -1
    file.write(b'\xff\xff\xff\xff\xff\xff\xff\xff')
    # Cardinality is 2 for Boolean variables
    file.write(b'\x00\x00\x00\x00\x00\x00\x00\x02')


def write_weight(file, wid, fixed, initial_value):
    file.write(struct.pack('>q', wid))
    if fixed:
        file.write(b'\x01')
    else:
        file.write(b'\x00')
    file.write(struct.pack('>d', initial_value))


def write_unary_factor(file, wid, vid, is_negated=False):
    # factorFunction is ISTRUE
    file.write(b'\x00\x04')
    # Equal predicate does not apply to Boolean variables, so we write 1
    file.write(b'\x00\x00\x00\x00\x00\x00\x00\x01')
    # Unary factors have 1 edge
    file.write(b'\x00\x00\x00\x00\x00\x00\x00\x01')
    file.write(struct.pack('>q', vid))
    if is_negated:
        file.write(b'\x00')
    else:
        file.write(b'\x01')
    file.write(struct.pack('>q', wid))


def write_pairwise_equal_factor(file, wid, vid1, vid2, is_negated=False):
    # factorFunction is EQUAL
    file.write(b'\x00\x03')
    # Equal predicate does not apply to Boolean variables, so we write 1
    file.write(b'\x00\x00\x00\x00\x00\x00\x00\x01')
    # Pairwise factors have two edges
    file.write(b'\x00\x00\x00\x00\x00\x00\x00\x02')
    file.write(struct.pack('>q', vid1))
    if is_negated:
        file.write(b'\x00')
    else:
        file.write(b'\x01')
    file.write(struct.pack('>q', vid2))
    # The second variable is always positive
    file.write(b'\x01')
    file.write(struct.pack('>q', wid))


def write_metadata(file, n_weights, n_vars, n_factors, n_edges):
    file.write(str(n_weights))
    file.write(',')
    file.write(str(n_vars))
    file.write(',')
    file.write(str(n_factors))
    file.write(',')
    file.write(str(n_edges))
