import mmln


class Learner:

    def __init__(self, network):
        self.n = network
        self.all_labels = mmln.get_all_labels(network)
