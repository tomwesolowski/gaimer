import sched
import time

import autograd.numpy as np
import matplotlib.pyplot as plt

from gui import Gui

import pacman
from params import Parameters
from approximators import TableLookupApproximator
from strategies import PolicyStrategy, KeyboardStrategy, ChaseStrategy
from utils import Console, Coord, Debug, Utils

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
        self.current_state = pacman.State(self, agents, foods)
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.stats = dict()
        for agent in self.agents:
            agent.strategy.register_stats(self)

    def start(self):
        self.gui.run(pacman.Pacman.get_colorboard(self), self.stats)
        for agent in self.current_state.get_enemy_agents():
            agent.learn(self, self.learning_epochs, self.current_state)
        for agent in self.agents:
            if agent.type == pacman.AgentType.PLAYER:
                agent.learn(self, self.learning_epochs, self.current_state)
        self.play()

    def play(self):
        keep_going = True
        while keep_going:
            keep_going = self.next()
            self.gui.update_board(pacman.Pacman.get_colorboard(self))
            time.sleep(Parameters.FREQ)

    # returns (npos, naction)
    def get_nbs_positions(self, pos, agent_type):
        return [(pos + action, action) for action in pacman.Action
                if self.is_valid_position(pos + action, agent_type)]

    # returns (nstate, naction, reward)
    def get_transitions(self, state):
        transitions = []
        for naction in self.get_actions(state.get_player_pos(), pacman.AgentType.PLAYER):
            nstate, nreward = self.act(state, naction)
            transitions.append((nstate, naction, nreward))
        return transitions

    def get_player_actions(self, state):
        return self.get_actions(state.get_player_pos(), pacman.AgentType.PLAYER)

    # returns (state, reward)
    def act(self, state, action=None):
        nstate = state.copy()
        eaten = False
        for idx, agent in enumerate(state.agents):
            if agent.type is pacman.AgentType.PLAYER:
                nstate.agents[idx], action = agent.move(state, action)
                newpos = nstate.agents[idx].pos
                if not self.is_valid_state(nstate):
                    Debug.print("Agent [", agent.name, "] Forbidden move!")
                    raise RuntimeError
                    nstate.agents[idx] = agent
                if newpos in nstate.foods:
                    nstate.foods.remove(newpos)
                    eaten = True
            else:
                # hack. enemies see what move we're going to make.
                nstate.agents[idx], _ = agent.move(nstate)
                if not self.is_valid_state(nstate):
                    Debug.print(agent.pos, nstate.agents[idx].pos)
                    Debug.print("Agent [", agent.name, "] Forbidden move!")
                    raise RuntimeError
        return nstate, self.get_reward(nstate, eaten)

    def __generate_all_states__(self):
        valid_positions = []
        for agent in self.agents:
            indices = np.ndenumerate(np.ndarray((self.height, self.width)))
            valid_positions.append(
                [Coord.from_tuple(idx) for idx, _ in indices
                 if self.is_valid_position(Coord.from_tuple(idx), agent.type)])
        for poses in Utils.cartesian(valid_positions):
            newagents = \
                [Agent(ag.name, ag.type, ag.strategy, poses[idx]) for idx, ag
                 in enumerate(self.agents)]
            self.all_states.append(pacman.State(self, newagents))

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
        if agent_type == pacman.AgentType.PLAYER:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != pacman.ObjectType.WALL)
        elif agent_type == pacman.AgentType.GHOST:
            return pos.good(self.height, self.width) \
                   and (self.get(pos) != pacman.ObjectType.WALL)
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


'''
Environments
'''

def get_simple_environment_with_three_foods():
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
        agents=[pacman.PlayerAgent("Player", PolicyStrategy(approximator=TableLookupApproximator()), Coord(9, 0)),
                pacman.GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.AgentType.PLAYER), Coord(0, 8)),
                #pacman.GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.AgentType.PLAYER), Coord(8, 10)),
                ],
        foods=[Coord(0, 10), Coord(4, 5), Coord(0, 0)],
        exit=Coord(10, 10),
        learning_epochs=Parameters.LEARN_EPOCHS)


def get_simple_environment_without_ghosts():
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
        agents=[pacman.PlayerAgent("Player", PolicyStrategy(), Coord(9, 0)),
                #GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.AgentType.PLAYER), Coord(0, 8)),
                #GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.AgentType.GHOST), Coord(8, 10)),
                #GhostAgent("Ghost #3", ChaseStrategy(target_pos=exitposition), Coord(4, 4))
                ],
        foods=[Coord(2, 0), Coord(0, 1), Coord(0, 2), Coord(1, 0), Coord(0, 0),
               Coord(3, 0), Coord(4, 0), Coord(10, 0), Coord(10, 1), Coord(10, 2),
               Coord(9, 2), Coord(10, 4), Coord(10, 5), Coord(10, 6), Coord(10, 7),
               Coord(0, 10), Coord(1, 10), Coord(2, 10), Coord(3, 10), Coord(4, 10),
               Coord(4, 4), Coord(4, 5), Coord(4, 6)],
        exit=exitposition,
        learning_epochs=LEARN_EPOCHS)


def get_simple_environment():
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
        agents=[pacman.PlayerAgent("Player", KeyboardStrategy(), Coord(0, 0)),
                pacman.GhostAgent("Ghost #1", ChaseStrategy(target_agent=pacman.AgentType.PLAYER), Coord(10, 0)),
                pacman.GhostAgent("Ghost #2", ChaseStrategy(target_agent=pacman.AgentType.GHOST), Coord(10, 10)),
                ],
        foods=[Coord(2, 2), Coord(8, 2), Coord(0, 10)],
        exit=Coord(5, 4),
        learning_epochs=LEARN_EPOCHS)

def main():
    get_simple_environment_with_three_foods().start()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    main()
