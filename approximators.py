import numpy as np

class FeatureExtractor:
    """Base class for all value approximators."""

    def features(self, stateaction):
        raise NotImplementedError

class ValueApproximator:
    """Base class for all value approximators."""
    pass

class TableLookupApproximator(ValueApproximator):
    """Approximator that use Python dictionary"""

    class SADictionary(dict):
        def __init__(self, default_value):
            self.default_value = default_value

        def __getkey__(self, stateaction):
            state, action = stateaction
            return hash((state.id(), action))

        def __getitem__(self, stateaction):
            if self.__getkey__(stateaction) not in self:
                self.__setitem__(stateaction, self.default_value)
            return super().__getitem__(self.__getkey__(stateaction))

        def __setitem__(self, stateaction, value):
            super().__setitem__(self.__getkey__(stateaction), value)

    def __init__(self, params):
        ValueApproximator.__init__(self)
        self.params = params
        self.qvalues = TableLookupApproximator.SADictionary(self.params.DEFAULT_QVALUE)
        self.eligibility = TableLookupApproximator.SADictionary(self.params.DEFAULT_ELIGIBILITY)

    def num_qvalues(self):
        return len(self.qvalues.keys())

class LinearApproximator(ValueApproximator):
    """Approximator that use simple linear model"""

    class SADictionary(dict):
        def __init__(self, extractor):
            self.__extractor = extractor
            self.weights = None

        def __init_weights(self, shape):
            self.weights = np.random.random(size=shape)

        def __getitem__(self, stateaction):
            features = self.__extractor.features(stateaction)
            if self.weights is None:
                self.__init_weights(features.shape)
            return np.dot(features, self.weights)

        def __setitem__(self, stateaction, value):
            raise NotImplementedError

    def __init__(self, extractor):
        ValueApproximator.__init__(self)
        self.extractor = extractor
        self.qvalues = LinearApproximator.SADictionary(extractor)
        self.eligibility = LinearApproximator.SADictionary(extractor)