from enum import IntEnum
import numpy as np
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


class ObjectType(IntEnum):
    EMPTY, WALL, EXIT, AGENT_PLAYER, AGENT_GHOST, FOOD, ALL = np.arange(7)


class Agent(IntEnum):
    PLAYER = ObjectType.AGENT_PLAYER
    GHOST = ObjectType.AGENT_GHOST

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