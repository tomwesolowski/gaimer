import copy
from enum import Enum
import numpy as np
import sys

class Utils:
    @staticmethod
    def bfs(env, agent_type, start, goals):
        distance = dict()
        queue = copy.copy(goals)
        for g in goals:
            distance[g] = 0
        while queue:
            pos = queue.pop(0)
            if not env.is_valid_position(pos, agent_type):
                continue
            if pos == start:
                break
            for npos, naction in env.get_nbs_positions(pos, agent_type):
                if npos not in distance:
                    distance[npos] = distance[pos] + 1
                    queue.append(npos)
        return distance

    @staticmethod
    def dist(posa, posb):
        diff = posa - posb
        return abs(diff.x) + abs(diff.y)

    @staticmethod
    def cartesian(arrays, out=None):
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n // arrays[0].size
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out


class Coord(object):
    def __init__(self, y, x):
        self.y = y
        self.x = x

    @classmethod
    def from_tuple(cls, p):
        return cls(p[0], p[1])

    def __add__(self, c):
        return Coord(self.y + c.y, self.x + c.x)

    def __sub__(self, c):
        return Coord(self.y - c.y, self.x - c.x)

    def __eq__(self, c):  # compares two coords
        return self.x == c.x and self.y == c.y

    def __repr__(self):
        return str((self.y, self.x))

    def __hash__(self):
        return (self.y, self.x).__hash__()

    def good(self, height, width):
        return 0 <= self.x < width and \
               0 <= self.y < height

    def t(self):  # return a tuple representation.
        return self.x, self.y

class PlotType(Enum):
    REGULAR = 0
    BAR = 1
    MATRIX = 2

class Stats:
    """docstring for Stats"""

    def __init__(self, plot_type):
        self.values = []
        self.plot_type = plot_type

    def append(self, v):
        self.values.append(v)

    def replace(self, newvalues):
        self.values = newvalues[:]


class MatrixStats(Stats):
    def __init__(self, height, width, default_value):
        Stats.__init__(self, PlotType.MATRIX)
        self.height = height
        self.width = width
        self.matrix = np.full((self.height, self.width), default_value)

    def add_in_point(self, y, x, val):
        self.matrix[y][x] += val

    def reset(self):
        self.matrix = np.full((self.height, self.width), 0)


class Debug:
    @staticmethod
    def print(*args):
        print(*args, file=sys.stderr)

class Console:
    @staticmethod
    def print(*args):
        print(*args)