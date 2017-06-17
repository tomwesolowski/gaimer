import copy
import sched
import time
import sys

import autograd.numpy as np
import matplotlib.pyplot as plt

from gui import Gui
import pacman
from params import Parameters
from strategies import PolicyStrategy, KeyboardStrategy, ChaseStrategy
from utils import Coord, Utils

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
        return self.get_agents_pos_with_type(pacman.Agent.PLAYER)[0]

    def get_enemy_agents(self):
        return [agent for agent in self.agents if agent.type != pacman.Agent.PLAYER]

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
            if agent.type != pacman.Agent.GHOST or agent.name == agent_name:
                self.cached_id *= self.boardsize
                self.cached_id += self.env.width * agent.pos.y + agent.pos.x
        return self.cached_id * hash(frozenset(self.foods))

######s

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
        Agent.__init__(self, name, pacman.Agent.PLAYER, strategy, pos)


######

class GhostAgent(Agent):
    def __init__(self, name, strategy, pos):
        Agent.__init__(self, name, pacman.Agent.GHOST, strategy, pos)


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
        self.gui = Gui(self.height, self.width)
        self.current_state = State(self, agents, foods)
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.stats = dict()
        for agent in self.agents:
            agent.strategy.register_stats(self)

    def start(self):
        self.gui.run(self.get_colorboard(), self.stats)
        for agent in self.current_state.get_enemy_agents():
            agent.learn(self, self.learning_epochs, self.current_state)
        for agent in self.agents:
            if agent.type == pacman.Agent.PLAYER:
                agent.learn(self, self.learning_epochs, self.current_state)
        self.play()

    def get_colorboard(self):
        colorboard = pacman.COLORS[self.board]
        colorboard[self.exit.y][self.exit.x] = pacman.COLORS[pacman.ObjectType.EXIT]
        for food in self.current_state.foods:
            colorboard[food.y][food.x] = pacman.COLORS[pacman.ObjectType.FOOD]
        for agent in self.current_state.agents:
            print(agent, file=sys.stderr)
            colorboard[agent.pos.y][agent.pos.x] = pacman.COLORS[agent.type]
        return colorboard

    def play(self):
        keep_going = True
        while keep_going:
            keep_going = self.next()
            self.gui.update_board(self.get_colorboard())
            time.sleep(Parameters.FREQ)

    # returns (npos, naction)
    def get_nbs_positions(self, pos, agent_type):
        return [(pos + action, action) for action in pacman.Action
                if self.is_valid_position(pos + action, agent_type)]

    # returns (nstate, naction, reward)
    def get_transitions(self, state):
        transitions = []
        for naction in self.get_actions(state.get_player_pos(), pacman.Agent.PLAYER):
            nstate, nreward = self.act(state, naction)
            transitions.append((nstate, naction, nreward))
        return transitions

    def get_player_actions(self, state):
        return self.get_actions(state.get_player_pos(), pacman.Agent.PLAYER)

    # returns (state, reward)
    def act(self, state, action=None):
        nstate = state.copy()
        eaten = False
        for idx, agent in enumerate(state.agents):
            if agent.type is pacman.Agent.PLAYER:
                nstate.agents[idx], action = agent.move(state, action)
                newpos = nstate.agents[idx].pos
                if not self.is_valid_state(nstate):
                    print("Agent [", agent.name, "] Forbidden move!", file=sys.stderr)
                    assert False
                    nstate.agents[idx] = agent
                if newpos in nstate.foods:
                    nstate.foods.remove(newpos)
                    eaten = True
            else:
                # hack. enemies see what move we're going to make.
                nstate.agents[idx], _ = agent.move(nstate)
                if not self.is_valid_state(nstate):
                    print(agent.pos, nstate.agents[idx].pos)
                    print("Agent [", agent.name, "] Forbidden move!", file=sys.stderr)
                    assert False
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
        return [action for action in pacman.Action
                if self.is_valid_position(pos + action, agent_type)]

    def is_valid_state(self, state):
        return all(self.is_valid_position(agent.pos, agent.type)
                   for agent in state.agents)

    def is_valid_position(self, pos, agent_type):
        if agent_type == pacman.Agent.PLAYER:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != pacman.ObjectType.WALL)
        elif agent_type == pacman.Agent.GHOST:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != pacman.ObjectType.WALL)
        else:
            assert False

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
            if Utils.dist(state.get_player_pos(), agent.pos) <= 0:
                return True
        return False

    def is_terminating_state(self, state):
        return self.is_winning_state(state) or self.is_losing_state(state)

    def get_reward(self, state, eaten):
        if self.is_winning_state(state):
            return Parameters.INF
        elif self.is_losing_state(state):
            return -Parameters.INF
        else:
            return -1 + eaten*10

    def get(self, pos):
        return self.board[pos.y][pos.x]


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
                GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.Agent.PLAYER), Coord(0, 8)),
                GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.Agent.PLAYER), Coord(8, 10)),
                # GhostAgent("Ghost #3", ChaseStrategy(target_agentpacman.Agent.PLAYER), Coord(0, 9))
                ],
        foods=[Coord(0, 10), Coord(4, 5), Coord(0, 0)],
        exit=Coord(10, 10),
        learning_epochs=Parameters.LEARN_EPOCHS)


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
                #GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.Agent.PLAYER), Coord(0, 8)),
                # Ten duszek sciga inne duszki.
                #GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.Agent.GHOST), Coord(8, 10)),
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
                GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.Agent.PLAYER), Coord(10, 0)),
                GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.Agent.GHOST), Coord(10, 10)),
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
