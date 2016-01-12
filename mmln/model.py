class Model:

    def __init__(self, regularization=1.0, inter_node_pos_same_label_default=1.0):
        self.regularization = regularization
        self.inter_node_pos_same_label_default = inter_node_pos_same_label_default
        self.inter_node_pos = {}
        self.inter_node_neg = {}
        self.intra_node_pos = {}
        self.intra_node_neg = {}
