import copy
import sched
import time
import math
import sys
from enum import IntEnum

import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from gui import Gui, PlotType

FPS = 4
FREQ = 1.0 / FPS
GAMMA = 0.95
ALPHA = 0.95
LAMBDA = 0.5
EPS = 0.09
INF = 1e9
LEARN_EPOCHS = 30
EPISODES_PER_EPOCH = 100
MAX_LEN_EPISODE = 100

EMPTY, WALL, EXIT, AGENT_PLAYER, AGENT_GHOST, FOOD, ALL = np.arange(7)

COLORS = np.ndarray((ALL, 3))
COLORS[EMPTY] = np.array([150, 150, 150])
COLORS[FOOD] = np.array([20, 110, 210])
COLORS[EMPTY] = np.array([150, 150, 150])
COLORS[EXIT] = np.array([0, 150, 0])
COLORS[AGENT_PLAYER] = np.array([25, 25, 25])
COLORS[AGENT_GHOST] = np.array([150, 0, 0])


def dist(posa, posb):
    diff = posa - posb
    return abs(diff.x) + abs(diff.y)


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

ALL_ACTIONS = [
    Action.IDLE, Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT
]

##################################################################

class State:
    """docstring for State"""

    def __init__(self, env, agents, foods):
        self.env = env
        self.agents = agents
        self.foods = foods
        self.boardsize = env.width * env.height
        self.cached_id = -1

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

    def clear_cache(self):
        self.cached_id = -1

    def copy(self):
        return State(self.env, copy.copy(self.agents), copy.copy(self.foods))

    def __hash__(self):
        assert False

    def id(self):
        # if self.cached_id < 0:
        self.cached_id = 0
        for agent in self.agents:
            self.cached_id *= self.boardsize
            self.cached_id += self.env.width * agent.pos.y + agent.pos.x
        return self.cached_id * (len(self.foods) + 1)

    def id_for(self, agent_name):
        # if self.cached_id < 0:
        self.cached_id = 0
        for agent in self.agents:
            if agent.type != AgentType.GHOST or agent.name == agent_name:
                self.cached_id *= self.boardsize
                self.cached_id += self.env.width * agent.pos.y + agent.pos.x
        return self.cached_id * (len(self.foods) + 1)

######


class AgentStrategy:
    """docstring for AgentStrategy"""

    def __init__(self):
        self.learnt = False

    def read_agent_data(self, agent):
        pass

    def learn(self, env, epochs, startstate):
        pass

    def register_stats(self, env):
        pass

    def get_action(self, state):
        pass


######

class PolicyStrategy(AgentStrategy):
    """docstring for PolicyStrategy"""

    def __init__(self):
        AgentStrategy.__init__(self)
        self.qvalues, self.eligibility = dict(), dict()

    def register_stats(self, env):
        env.stats['epilens'] = Stats(PlotType.REGULAR)
        env.stats['wins'] = Stats(PlotType.REGULAR)
        #env.stats['actions'] = Stats(PlotType.BAR)
        env.stats['heatmap'] = MatrixStats(env.height, env.width, -INF)

    def learn(self, env, epochs, startstate):
        if self.learnt:
            return
        for epoch in trange(LEARN_EPOCHS):
            total_lens = 0
            print("QValues size: ", len(self.qvalues), file=sys.stderr)
            for _ in range(EPISODES_PER_EPOCH):
                lenepisode = 0
                state = startstate.copy()
                action = Action.IDLE
                current_episode = [(state, action)]
                wins = 0
                # Play and collect samples.
                while not env.is_terminating_state(state) and lenepisode <= MAX_LEN_EPISODE:
                    if lenepisode < MAX_LEN_EPISODE:
                        nstate, reward = state.env.act(state, action)
                    else:
                        nstate, _ = state.env.act(state, action)
                        reward = -INF/10
                    naction = self.q_eps_greedy_action(nstate, epoch)
                    delta = reward + GAMMA * self.get_qvalue(nstate, naction) - self.get_qvalue(state, action)
                    current_episode += [(nstate, naction)]
                    self.inc_eligibility(state, action)
                    for (state, action) in current_episode:
                        stateid = state.id()
                        addval = ALPHA * delta * self.get_eligibility(stateid, action)
                        self.qvalues[(stateid, action)] += addval
                        self.eligibility[(stateid, action)] *= GAMMA * LAMBDA
                        player_pos = state.get_player_pos()
                        env.stats['heatmap'].add_in_point(player_pos.y, player_pos.x, self.qvalues[(stateid, action)])
                    lenepisode += 1
                    state, action = nstate, naction
                total_lens += lenepisode
                wins += env.is_winning_state(state)
            env.stats['epilens'].append(total_lens / EPISODES_PER_EPOCH)
            env.stats['wins'].append(wins)
            env.refresh_stats()
        self.learnt = True

    def q_eps_greedy_action(self, state, epoch):
        actions = state.env.get_player_actions(state)
        raction = actions[np.random.choice(max(1, len(actions)))]
        if np.random.random() > 1-(EPS/math.sqrt(epoch+1)) or len(actions) <= 0:
            naction = raction
        else:
            naction, _ = self.q_greedy_action(state)
        return naction

    def q_greedy_action(self, state):
        if state.env.is_terminating_state(state):
            return Action.IDLE, None
        actions = state.env.get_player_actions(state)
        naction, naction_value = None, -INF
        probs_and_actions = []
        for action in actions:
            value = self.get_qvalue(state, action)
            probs_and_actions.append((value, action))
            if naction is None or naction_value < value:
                naction = action
                naction_value = value
        assert(naction is not None)
        return naction, probs_and_actions

    def get_eligibility(self, sid, action):
        if (sid, action) not in self.eligibility:
            self.eligibility[(sid, action)] = 0
        return self.eligibility[(sid, action)]

    def inc_eligibility(self, state, action):
        sid = state.id()
        if (sid, action) not in self.eligibility:
            self.eligibility[(sid, action)] = 1
        else:
            self.eligibility[(sid, action)] += 1

    def get_qvalue(self, state, action):
        sid = state.id()
        if (sid, action) not in self.qvalues:
            self.qvalues[(sid, action)] = 0
        return self.qvalues[(sid, action)]

    def get_action(self, state):
        action, probs_and_labels = self.q_greedy_action(state)
        #state.env.stats['actions'].replace(probs_and_labels)
        #state.env.refresh_stats()
        return action


######

class ChaseStrategy(AgentStrategy):
    """docstring for ChaseStrategy"""

    def __init__(self, **keywords):
        AgentStrategy.__init__(self)
        if 'target_agent' in keywords:
            self.target_agent_type = keywords['target_agent']
        elif 'target_pos' in keywords:
            self.target_pos = keywords['target_pos']
        else:
            assert False

    def learn(self, env, epochs, startstate):
        self.init(env)
        return
        print("Learning ChaseStrategy...", file=sys.stderr)
        if self.learnt:
            return
        self.policy = self.compute_policy(env)
        self.learnt = True
        print("Done.", file=sys.stderr)

    def init(self, env):
        # fn = np.vectorize(lambda x: Action.IDLE)
        # self.policy = fn(self.__get_new_policy__(env))
        self.policy = dict()

    def __get_new_policy__(self, env):
        return np.empty(env.get_states_dims(), dtype=Action)

    def read_agent_data(self, agent):
        self.agent_name = agent.name
        self.agent_type = agent.type

    def compute_policy(self, env):
        newpolicy = self.__get_new_policy__(env)
        for state in env.get_all_states():
            assert (env.is_valid_state(state))
            bestaction = self.compute_action(state)
            newpolicy[state.id()] = bestaction
        return newpolicy

    def __bfs__(self, env, start, goals):
        distance = dict()
        queue = copy.copy(goals)
        for g in goals:
            distance[g] = 0
        while queue:
            pos = queue.pop(0)
            if not env.is_valid_position(pos, self.agent_type):
                continue
            if pos == start:
                break
            if not env.is_valid_position(pos, self.agent_type):
                print(start, goals, pos, file=sys.stderr)
                assert False
            for npos, naction in env.get_nbs_positions(pos, self.agent_type):
                if npos not in distance:
                    distance[npos] = distance[pos] + 1
                    queue.append(npos)
        return distance

    def compute_action(self, state):
        env = state.env
        curpos = state.get_agent_pos_with_name(self.agent_name)
        if hasattr(self, 'target_agent_type'):
            targetspos = state.get_agents_pos_with_type(self.target_agent_type)
            if self.target_agent_type == AgentType.GHOST:
                targetspos.remove(curpos)
        else:
            targetspos = [self.target_pos]
        # Run BFS.
        distance = self.__bfs__(env, curpos, targetspos)
        # Select best move.
        bestaction, bestpos = Action.IDLE, curpos
        for npos, naction in env.get_nbs_positions(curpos, self.agent_type):
            if (npos in distance and
                        distance[npos] < distance[bestpos]):
                bestpos, bestaction = npos, naction
        return bestaction

    def get_action(self, state):
        stateid = state.id_for(self.agent_name)
        if stateid in self.policy:
            return self.policy[stateid]
        action = self.compute_action(state)
        self.policy[stateid] = action
        return action


#####

class KeyboardStrategy(AgentStrategy):
    """docstring for ChaseStrategy"""

    def __init__(self):
        AgentStrategy.__init__(self)

    def print_game(self, state):
        env = state.env
        print(env.height, env.width)
        for i in range(env.height):
            print(''.join(map(str, env.board[i])))
        print(env.exit.y, env.exit.x)
        print(len(state.agents))
        for agent in state.agents:
            print(agent.pos.y, agent.pos.x)
        print(len(state.foods))
        for food in state.foods:
            print(food.y, food.x)

    def get_action(self, state):
        self.print_game(state)
        action = ALL_ACTIONS[int(input())]
        return action

#########################################################################

class AgentType(IntEnum):
    PLAYER = AGENT_PLAYER
    GHOST = AGENT_GHOST


######

class Agent:
    def __init__(self, name, type, strategy, pos):
        self.name = name
        self.type = type
        self.strategy = strategy
        self.pos = pos
        self.strategy.read_agent_data(self)

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


######

class PlayerAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, AgentType.PLAYER, strategy, pos)


######

class GhostAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, AgentType.GHOST, strategy, pos)


#########################################################################

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


class Environment:
    def __init__(self, board, agents, foods, exit, learning_epochs):
        self.height = board.shape[0]
        self.width = board.shape[1]
        self.board = board
        self.foods = foods
        self.agents = agents
        self.exit = exit
        self.learning_epochs = learning_epochs
        self.gui = Gui(self.height, self.width, LEARN_EPOCHS)
        self.current_state = State(self, agents, foods)
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.stats = dict()
        for agent in self.agents:
            agent.strategy.register_stats(self)

    def start(self):
        self.gui.run(COLORS[self.board], self.stats)
        for agent in self.current_state.get_enemy_agents():
            agent.learn(self, self.learning_epochs, self.current_state)
        for agent in self.agents:
            if agent.type == AgentType.PLAYER:
                agent.learn(self, self.learning_epochs, self.current_state)
        self.play()

    def play(self):
        keep_going = True
        while keep_going:
            keep_going = self.next()
            self.colorboard = COLORS[self.board]
            self.colorboard[self.exit.y][self.exit.x] = COLORS[EXIT]
            for food in self.current_state.foods:
                self.colorboard[food.y][food.x] = COLORS[FOOD]
            for agent in self.current_state.agents:
                print(agent, file=sys.stderr)
                self.colorboard[agent.pos.y][agent.pos.x] = COLORS[agent.type]
            self.refresh_board()
            time.sleep(FREQ)

    # returns (npos, naction)
    def get_nbs_positions(self, pos, agent_type):
        return [(pos + action, action) for action in Action
                if self.is_valid_position(pos + action, agent_type)]

    # returns (nstate, naction, reward)
    def get_transitions(self, state):
        transitions = []
        for naction in self.get_actions(state.get_player_pos(), AgentType.PLAYER):
            nstate, nreward = self.act(state, naction)
            transitions.append((nstate, naction, nreward))
        return transitions

    def get_player_actions(self, state):
        return self.get_actions(state.get_player_pos(), AgentType.PLAYER)

    # returns (state, reward)
    def act(self, state, action=None):
        nstate = state.copy()
        eaten = False
        for idx, agent in enumerate(state.agents):
            if agent.type is AgentType.PLAYER:
                nstate.agents[idx], action = agent.move(state, action)
                newpos = nstate.agents[idx].pos
                if not self.is_valid_state(nstate):
                    print("Agent [", idx, "] Forbidden move!", file=sys.stderr)
                    nstate.agents[idx] = agent
                if newpos in nstate.foods:
                    nstate.foods.remove(newpos)
                    eaten = True
            else:
                # hack. enemies see what move we're going to make.
                nstate.agents[idx], _ = agent.move(nstate)
        return nstate, self.get_reward(nstate, eaten)

    def __generate_all_states__(self):
        valid_positions = []
        for agent in self.agents:
            indices = np.ndenumerate(np.ndarray((self.height, self.width)))
            valid_positions.append(
                [Coord.from_tuple(idx) for idx, _ in indices
                 if self.is_valid_position(Coord.from_tuple(idx), agent.type)])
        for poses in cartesian(valid_positions):
            newagents = \
                [Agent(ag.name, ag.type, ag.strategy, poses[idx]) for idx, ag
                 in enumerate(self.agents)]
            self.all_states.append(State(self, newagents))

    def get_all_states(self):
        return self.all_states

    def get_states_dims(self):
        size = 1
        for _ in self.agents:
            size *= self.height * self.width
        return size

    def get_actions(self, pos, agent_type):
        return [action for action in Action
                if self.is_valid_position(pos + action, agent_type)]

    def is_valid_state(self, state):
        return all(self.is_valid_position(agent.pos, agent.type)
                   for agent in state.agents)

    def is_valid_position(self, pos, agent_type):
        if agent_type == AgentType.PLAYER:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != WALL)
        elif agent_type == AgentType.GHOST:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != WALL)
        else:
            assert False

    def refresh_board(self):
        self.gui.update_board(self.colorboard)

    def refresh_stats(self):
        self.gui.update_stats(self.stats)

    def next(self):
        # assert(all(ag.strategy.learnt for ag in self.agents))
        self.current_state, _ = self.act(self.current_state)

        if self.is_winning_state(self.current_state):
            print("-1 -1")
            print("Player won.", file=sys.stderr)
            return False
        elif self.is_losing_state(self.current_state):
            print("-1 -1")
            print("Player lost.", file=sys.stderr)
            return False
        return True

    def is_winning_state(self, state):
        return state.get_player_pos() == self.exit and \
               len(state.foods) == 0

    def is_losing_state(self, state):
        for agent in state.get_enemy_agents():
            if dist(state.get_player_pos(), agent.pos) <= 0:
                return True
        return False

    def is_terminating_state(self, state):
        return self.is_winning_state(state) or self.is_losing_state(state)

    def get_reward(self, state, eaten):
        if self.is_winning_state(state):
            return INF
        elif self.is_losing_state(state):
            return -INF
        else:
            return -1 + eaten

    def get(self, pos):
        return self.board[pos.y][pos.x]


##################################################################

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
        self.updates = np.zeros((self.height, self.width))

    def add_in_point(self, y, x, val):
        #print("Before; ", self.matrix[y][x])
        self.updates[y][x] += 1
        #self.matrix[y][x] += (val - self.matrix[y][x] + 0.0) / (self.updates[y][x])
        self.matrix[y][x] = max(self.matrix[y][x], val)
        #print("After; ", self.matrix[y][x])


####################### PRZYKLADOWE MAPY ###########################################

def get_simple_environment_without_foods():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    return Environment(
        board=board,
        agents=[PlayerAgent("Player", PolicyStrategy(), Coord(9, 0)),
                # GhostAgent("Ghost #1", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(0, 8)),
                # GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(8, 10)),
                # GhostAgent("Ghost #3", ChaseStrategy(target_agentAgentType.PLAYER), Coord(0, 9))
                ],
        foods=[Coord(0, 10)],
        exit=Coord(10, 10),
        learning_epochs=LEARN_EPOCHS)


def get_simple_environment():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    exitposition = Coord(4, 8)
    return Environment(
        board=board,
        agents=[PlayerAgent("Player", PolicyStrategy(), Coord(9, 0)),
                # Kazdy duszek moze albo scigac dany typ agenta (w tym przypadku gracza)
                # albo moze biec do okreslonego pola.
                #GhostAgent("Ghost #1", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(0, 8)),
                # Ten duszek sciga inne duszki.
                #GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.GHOST), Coord(8, 10)),
                # A ten biegnie do wyjscia.
                #GhostAgent("Ghost #3", ChaseStrategy(target_pos=exitposition), Coord(4, 4))
                ],
        foods=[Coord(2, 0), Coord(0, 1), Coord(0, 2), Coord(1, 0), Coord(0, 0),
               Coord(3, 0), Coord(4, 0), Coord(10, 0), Coord(10, 1), Coord(10, 2),
               Coord(9, 2), Coord(10, 4), Coord(10, 5), Coord(10, 6), Coord(10, 7),
               Coord(0, 10), Coord(1, 10), Coord(2, 10), Coord(3, 10), Coord(4, 10),
               Coord(4, 4), Coord(4, 5), Coord(4, 6)],
        exit=exitposition,
        learning_epochs=LEARN_EPOCHS)


def get_bartek_environment():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    return Environment(
        board=board,
        agents=[PlayerAgent("Player", KeyboardStrategy(), Coord(0, 0)),
                GhostAgent("Ghost #1", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(10, 0)),
                GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.GHOST), Coord(10, 10)),
                ],
        foods=[Coord(2, 2), Coord(8, 2), Coord(0, 10)],
        exit=Coord(5, 4),
        learning_epochs=LEARN_EPOCHS)

# Żeby utworzyć własną mapę, skopiuj jedną z powyższych funkcji, zmodyfikuj ją i wywołaj w funkcji main() w
# env = get_twoje_srodowisko()

##################################################################

def main():
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    env = get_simple_environment_without_foods()
    env.start()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
