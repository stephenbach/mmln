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
