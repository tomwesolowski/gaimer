

from params import Parameters

class FeatureExtractor:
    """Base class for all value approximators."""

    def features(self, state):
        raise NotImplementedError

class ValueApproximator:
    """Base class for all value approximators."""
    pass


class TableLookupApproximator(ValueApproximator):
    """Approximator that use Python dictionary"""

    class StateDictionary(dict):
        def __init__(self, default_value):
            self.default_value = default_value

        def __getkey__(self, stateaction):
            state, action = stateaction
            return hash((state.id(), action))

        def __getitem__(self, stateaction):
            if self.__getkey__(stateaction) not in self:
                self.__setitem__(stateaction, self.default_value)
            return super(TableLookupApproximator.StateDictionary, self).__getitem__(self.__getkey__(stateaction))

        def __setitem__(self, stateaction, value):
            super(TableLookupApproximator.StateDictionary, self).__setitem__(self.__getkey__(stateaction), value)

    def __init__(self):
        ValueApproximator.__init__(self)
        self.qvalues = TableLookupApproximator.StateDictionary(Parameters.DEFAULT_QVALUE)
        self.eligibility = TableLookupApproximator.StateDictionary(Parameters.DEFAULT_ELIGIBILITY)

    # def get_eligibility(self, state, action):
    #     if (sid, action) not in self.__eligibility:
    #         self.__eligibility[(sid, action)] = Parameters.DEFAULT_ELIGIBILITY
    #     return self.__eligibility[(sid, action)]
    #
    # def set_eligibility(self, state, action, value):
    #     oldvalue = self.get_eligibility(state, action)
    #     self.inc_eligibility(state, action, value - oldvalue)
    #
    # def get_qvalue(self, state, action):
    #     sid = state.id()
    #     if (sid, action) not in self.__qvalues:
    #         self.__qvalues[(sid, action)] = Parameters.DEFAULT_QVALUE
    #     return self.__qvalues[(sid, action)]
    #
    # def set_qvalue(self, state, action, value):
    #     oldvalue = self.get_qvalue(state, action)
    #     self.inc_qvalue(state, action, value - oldvalue)

    def num_qvalues(self):
        return len(self.qvalues.keys())