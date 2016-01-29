inter_node_pos = 'inter_node_pos'
inter_node_neg = 'inter_node_neg'
intra_node_pos = 'intra_node_pos'
intra_node_neg = 'intra_node_neg'

all_potential_types = (inter_node_pos, inter_node_neg, intra_node_pos, intra_node_neg)


class Model:

    def __init__(self, regularization=1.0, inter_node_pos_same_label_default=1.0):
        self.regularization = regularization
        self.inter_node_pos_same_label_default = inter_node_pos_same_label_default

        self._weights = {inter_node_pos: {}, inter_node_neg: {}, intra_node_pos: {}, intra_node_neg: {}}

    def __getitem__(self, item):
        return self._weights[item]

    def __setitem__(self, key, value):
        self._weights[key] = value