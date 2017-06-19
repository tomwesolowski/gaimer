import enum
import time
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, params):
        self.params = params
        self.prepared = False

    @property
    def agents(self):
        raise NotImplementedError

    @property
    def start_state(self):
        return self._start_state.copy()

    def run(self, keep=False):
        self.gui.run(self.get_colorboard(), self.stats)
        self.prepare()
        keep_going = True
        while keep_going:
            keep_going = self.next()
            self.refresh_gui()
            time.sleep(self.params.FREQ)
        if keep:
            plt.ioff()
            plt.show()

    def next(self):
        raise NotImplementedError

    def reward(self, agent, state, action, nstate):
        raise NotImplementedError

    def is_terminating_state(self, state):
        return any([self.is_winning_state(state, agent) or self.is_losing_state(state, agent) for agent in self.agents])

    def is_winning_state(self, agent):
        raise NotImplementedError

    def is_losing_state(self, agent):
        raise NotImplementedError

    def refresh_gui(self):
        raise NotImplementedError


class State:
    def __init__(self, env):
        self.env = env

    def get_default_action(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError

    def get_agent_with_name(self, agent_name):
        raise NotImplementedError

    def get_agent_pos_with_name(self, agent_name):
        raise NotImplementedError

    def get_player_agent(self, agent_name):
        raise NotImplementedError


class Action(type):
    @staticmethod
    def all():
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
        for attr, value in vars().items():
            if not attr.startswith("__") and attr != 'self':
                print(attr, value)
                yield value


class ObjectType(enum.IntEnum):
    pass


class AgentType(enum.IntEnum):
    pass
