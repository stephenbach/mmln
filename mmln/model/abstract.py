import mmln

import logging


class AbstractModel:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train(self, network):
        raise NotImplementedError()

    def predict(self, network):
        raise NotImplementedError()
