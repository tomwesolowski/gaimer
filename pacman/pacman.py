import copy
import sched
import time
import numpy as np
from approximators import FeatureExtractor
from env import Environment, State, Action, AgentType, ObjectType
from gui import Gui
from params import Parameters
from utils import Console, Coord, Debug, Utils


class PacmanParameters(Parameters):
    FPS = 4
    FREQ = 1.0 / FPS
    GAMMA = 0.95
    ALPHA = 0.1  # changed from 0.2
    LAMBDA = 0.35
    EPS = 0.1  # changed from 0.09
    DEFAULT_ELIGIBILITY = 0
    DEFAULT_QVALUE = 0
    INC_ELIGIBILITY = 1
    INF = 1e9
    LEARN_EPOCHS = 50
    EPISODES_PER_EPOCH = 100
    MAX_LEN_EPISODE = 75


class PacmanAction(Action):
    IDLE = Coord(0, 0)
    UP = Coord(-1, 0)
    RIGHT = Coord(0, 1)
    DOWN = Coord(1, 0)
    LEFT = Coord(0, -1)

    @staticmethod
    def all():
        return [PacmanAction.IDLE,
                PacmanAction.UP,
                PacmanAction.RIGHT,
                PacmanAction.DOWN,
                PacmanAction.LEFT]


class PacmanObjectType(ObjectType):
    EMPTY, WALL, EXIT, AGENT_PLAYER, AGENT_GHOST, FOOD, ALL = np.arange(7)


class PacmanAgentType(AgentType):
    PLAYER = PacmanObjectType.AGENT_PLAYER
    GHOST = PacmanObjectType.AGENT_GHOST


class PacmanState(State):
    def __init__(self, env, agents, foods):
        super().__init__(env)
        self.agents = agents
        self.foods = foods
        self.boardsize = env.width * env.height

    def get_default_action(self):
        return PacmanAction.IDLE

    def get_actions(self, agent):
        return self.env.get_actions(self, agent)

    def get_agent_with_name(self, agent_name):
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        assert False

    def get_agent_pos_with_name(self, agent_name):
        return self.get_agent_with_name(agent_name).pos

    def get_player_agent(self):
        for agent in self.agents:
            if agent.type == PacmanAgentType.PLAYER:
                return agent
        assert False

    def get_agents_pos_with_type(self, agent_type):
        return [agent.pos for agent in self.agents if agent.type == agent_type]

    def get_player_pos(self):
        return self.get_player_agent().pos

    def get_enemy_agents(self):
        return [agent for agent in self.agents if agent.type != PacmanAgentType.PLAYER]

    def copy(self):
        return PacmanState(self.env, copy.copy(self.agents), copy.copy(self.foods))

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
            if agent.type != PacmanAgentType.GHOST or agent.name == agent_name:
                id *= self.boardsize
                id += self.env.width * agent.pos.y + agent.pos.x
        return hash((id, frozenset(self.foods)))


class Agent:
    def __init__(self, name, type, strategy, pos, id=None):
        self.id = id
        self.name = name
        self.type = type
        self.strategy = strategy
        self.pos = pos
        self.strategy.agent_name = self.name
        self.strategy.agent_type = self.type

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return str(self.name + ": " + str(self.pos))

    def learn(self, env):
        self.strategy.learn(env)

    def move(self, state, action=None):
        if action is None:
            return self.move(state, self.strategy.get_action(state))
        else:
            return Agent(self.name, self.type, self.strategy, self.pos + action, self.id), action


class PlayerAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, PacmanAgentType.PLAYER, strategy, pos)


class GhostAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, PacmanAgentType.GHOST, strategy, pos)


class PacmanFeatureExtractor(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)

    def features(self, stateaction):
        state, action = stateaction
        env = state.env
        state, _ = env.act(state, action)
        feats = list(map(lambda x: 1 if x in state.foods else 0, state.env.all_foods))
        feats.append(state.get_player_pos().y)
        feats.append(state.get_player_pos().x)
        for agent in state.get_enemy_agents():
            feats.appends(agent.pos.y)
            feats.appends(agent.pos.x)
        return np.array(feats)


class PacmanEnvironment(Environment):
    def __init__(self, params, board, agents, foods, exit):
        super().__init__(params)
        self.height, self.width = board.shape
        self.board = board
        self.foods = foods
        self.all_foods = copy.copy(foods)
        self._agents = agents
        self.exit = exit
        self.gui = Gui(self.height, self.width)
        self._start_state = PacmanState(self, agents, foods)
        self.current_state = self.start_state
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.stats = dict()
        for agent in self.agents:
            agent.strategy.params = params
            agent.strategy.register_stats(self)
        self.colors = np.ndarray((PacmanObjectType.ALL, 3))
        self.init_colors()

    @property
    def agents(self):
        return self._agents

    def prepare(self):
        if self.prepared:
            return
        self.prepared = True
        for (id, agent) in enumerate(self.agents):
            agent.id = id
        for agent in self.current_state.get_enemy_agents():
            agent.learn(self)
        for agent in self.agents:
            if agent.type == PacmanAgentType.PLAYER:
                agent.learn(self)

    def refresh_gui(self):
        self.gui.update_board(self.get_colorboard())

    # returns (npos, naction)
    def get_nbs_positions(self, pos, agent_type):
        return [(pos + action, action) for action in PacmanAction.all()
                if self.is_valid_position(pos + action, agent_type)]

    # returns (nstate, naction, reward)
    def get_transitions(self, state):
        transitions = []
        for naction in self.get_actions(state, state.get_player_agent()):
            nstate, nreward = self.act(state, naction)
            transitions.append((nstate, naction, nreward))
        return transitions

    def get_player_actions(self, state):
        return self.get_actions(state, state.get_player_agent())

    # returns (state, reward)
    def act(self, state, action=None):
        nstate = state.copy()
        for idx, agent in enumerate(state.agents):
            if agent.type is PacmanAgentType.PLAYER:
                nstate.agents[idx], action = agent.move(state, action)
                newpos = nstate.agents[idx].pos
                if not self.is_valid_state(nstate):
                    Debug.print("Agent [", agent.name, "] Forbidden move!")
                    raise RuntimeError
                if newpos in nstate.foods:
                    nstate.foods.remove(newpos)
            else:
                # hack. enemies see what move we're going to make.
                nstate.agents[idx], _ = agent.move(nstate)
                if not self.is_valid_state(nstate):
                    Debug.print(agent.pos, nstate.agents[idx].pos)
                    Debug.print("Agent [", agent.name, "] Forbidden move!")
                    raise RuntimeError
        return nstate, self.reward(state.get_player_agent(), state, action, nstate)

    def get_actions(self, state, agent):
        return [action for action in PacmanAction.all()
                if self.is_valid_position(agent.pos + action, agent.type)]

    def is_valid_state(self, state):
        return all(self.is_valid_position(agent.pos, agent.type)
                   for agent in state.agents)

    def is_valid_position(self, pos, agent_type):
        if agent_type == PacmanAgentType.PLAYER:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != PacmanObjectType.WALL)
        elif agent_type == PacmanAgentType.GHOST:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != PacmanObjectType.WALL)
        else:
            assert False

    def refresh_stats(self):
        self.gui.update_stats(self.stats)

    def next(self):
        self.current_state, _ = self.act(self.current_state)
        if self.is_winning_state(self.current_state):
            Console.print("-1 -1")
            Debug.print("Player won.")
            return False
        elif self.is_losing_state(self.current_state):
            Console.print("-1 -1")
            Debug.print("Player lost.")
            return False
        return True

    def is_winning_state(self, state, agent=None):
        return state.get_player_pos() == self.exit and \
               len(state.foods) == 0

    def is_losing_state(self, state, agent=None):
        for agent in state.get_enemy_agents():
            if Utils.dist(state.get_player_pos(), agent.pos) <= 0:
                return True
        return False

    def reward(self, agent, state, action, nstate):
        nagent, _ = agent.move(state, action)
        eaten = nagent.pos in state.foods
        if self.is_winning_state(nstate, None):
            return self.params.INF
        elif self.is_losing_state(nstate, None):
            return -self.params.INF
        else:
            return -1 + eaten*10

    def get(self, pos):
        return self.board[pos.y][pos.x]

    def init_colors(self):
        self.colors[PacmanObjectType.EMPTY] = np.array([150, 150, 150])
        self.colors[PacmanObjectType.FOOD] = np.array([20, 110, 210])
        self.colors[PacmanObjectType.EMPTY] = np.array([150, 150, 150])
        self.colors[PacmanObjectType.EXIT] = np.array([0, 150, 0])
        self.colors[PacmanObjectType.AGENT_PLAYER] = np.array([25, 25, 25])
        self.colors[PacmanObjectType.AGENT_GHOST] = np.array([150, 0, 0])

    @property
    def all_actions(self):
        return [PacmanAction.IDLE, PacmanAction.UP, PacmanAction.RIGHT, PacmanAction.DOWN, PacmanAction.LEFT]

    def get_colorboard(self):
        colorboard = self.colors[self.board]
        colorboard[self.exit.y][self.exit.x] = self.colors[PacmanObjectType.EXIT]
        for food in self.current_state.foods:
            colorboard[food.y][food.x] = self.colors[PacmanObjectType.FOOD]
        for agent in self.current_state.agents:
            Debug.print(agent)
            colorboard[agent.pos.y][agent.pos.x] = self.colors[agent.type]
        return colorboard