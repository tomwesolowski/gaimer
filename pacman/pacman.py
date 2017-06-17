import copy
import enum
import numpy as np
import utils
from utils import Coord

class MetaAction(type):
    def __iter__(self):
        for attr, value in vars(Action).items():
            if not attr.startswith("__"):
                yield value


class Action(metaclass=MetaAction):
    IDLE = Coord(0, 0)
    UP = Coord(-1, 0)
    RIGHT = Coord(0, 1)
    DOWN = Coord(1, 0)
    LEFT = Coord(0, -1)


class ObjectType(enum.IntEnum):
    EMPTY, WALL, EXIT, AGENT_PLAYER, AGENT_GHOST, FOOD, ALL = np.arange(7)


class AgentType(enum.IntEnum):
    PLAYER = ObjectType.AGENT_PLAYER
    GHOST = ObjectType.AGENT_GHOST


class Pacman:
    COLORS = np.ndarray((ObjectType.ALL, 3))
    COLORS[ObjectType.EMPTY] = np.array([150, 150, 150])
    COLORS[ObjectType.FOOD] = np.array([20, 110, 210])
    COLORS[ObjectType.EMPTY] = np.array([150, 150, 150])
    COLORS[ObjectType.EXIT] = np.array([0, 150, 0])
    COLORS[ObjectType.AGENT_PLAYER] = np.array([25, 25, 25])
    COLORS[ObjectType.AGENT_GHOST] = np.array([150, 0, 0])

    ALL_ACTIONS = [
        Action.IDLE,
        Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT
    ]

    @staticmethod
    def get_colorboard(env):
        colorboard = Pacman.COLORS[env.board]
        colorboard[env.exit.y][env.exit.x] = Pacman.COLORS[ObjectType.EXIT]
        for food in env.current_state.foods:
            colorboard[food.y][food.x] = Pacman.COLORS[ObjectType.FOOD]
        for agent in env.current_state.agents:
            utils.Debug.print(agent)
            colorboard[agent.pos.y][agent.pos.x] = Pacman.COLORS[agent.type]
        return colorboard


class State:
    def __init__(self, env, agents, foods):
        self.env = env
        self.agents = agents
        self.foods = foods
        self.boardsize = env.width * env.height

    def get_agent_pos_with_name(self, agent_name):
        for agent in self.agents:
            if agent.name == agent_name:
                return agent.pos
        assert False

    def get_agents_pos_with_type(self, agent_type):
        return [agent.pos for agent in self.agents if agent.type == agent_type]

    def get_player_pos(self):
        return self.get_agents_pos_with_type(AgentType.PLAYER)[0]

    def get_enemy_agents(self):
        return [agent for agent in self.agents if agent.type != AgentType.PLAYER]

    def copy(self):
        return State(self.env, copy.copy(self.agents), copy.copy(self.foods))

    def __hash__(self):
        assert False

    def id(self):
        id = 0
        for agent in self.agents:
            id *= self.boardsize
            id += self.env.width * agent.pos.y + agent.pos.x
        return id * (len(self.foods) + 1)

    def id_for(self, agent_name):
        id = 0
        for agent in self.agents:
            if agent.type != AgentType.GHOST or agent.name == agent_name:
                id *= self.boardsize
                id += self.env.width * agent.pos.y + agent.pos.x
        return hash((id, frozenset(self.foods)))


class Agent:
    def __init__(self, name, type, strategy, pos):
        self.name = name
        self.type = type
        self.strategy = strategy
        self.pos = pos
        self.strategy.agent_name = self.name
        self.strategy.agent_type = self.type

    def __hash__(self):
        return name.__hash__()

    def __repr__(self):
        return str(self.name + ": " + str(self.pos))

    def learn(self, env, epochs, startstate):
        self.strategy.learn(env, epochs, startstate)

    def move(self, state, action=None):
        if action is None:
            return self.move(state, self.strategy.get_action(state))
        else:
            return Agent(self.name, self.type, self.strategy, self.pos + action), action


class PlayerAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, AgentType.PLAYER, strategy, pos)


class GhostAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, AgentType.GHOST, strategy, pos)