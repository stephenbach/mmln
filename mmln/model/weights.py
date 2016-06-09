import mmln


OTHER = 0
INTER = 1
INTRA = 2

DEFAULT = 0


class Weights:

    def __init__(self):
        self.weights = {OTHER: {}, INTER: {}, INTRA: {}}

    def __getitem__(self, item):
        return self.weights[item]

    def size(self):
        return len(self.weights[OTHER]) + len(self.weights[INTER]) \
               + len(self.weights[INTRA])

    def to_zeros(self):
        zero_weights = Weights()
        for weight_type in (OTHER, INTER, INTRA):
            for weight in self[weight_type]:
                zero_weights[weight_type][weight] = 0.0
        return zero_weights

    def __str__(self):
        return "\n".join(['OTHER', str(self[OTHER]), 'INTER', str(self[INTER]), 'INTRA', str(self[INTRA])])
