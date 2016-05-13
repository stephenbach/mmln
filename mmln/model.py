inter = 0
intra = 1


class Model:

    def __init__(self, regularization=1.0, inter_same_label_default=1.0):
        self.regularization = regularization
        self.inter_same_label_default = inter_same_label_default

        self._weights = ({}, {})

    def __getitem__(self, item):
        return self._weights[item]