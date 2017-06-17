import math
import numpy as np
import pacman
from params import Parameters
from tqdm import trange
from utils import Console, Debug, PlotType, MatrixStats, Stats, Utils


class AgentStrategy:
    """Base class for all Agent strategies"""

    def __init__(self):
        self.learnt = False

    def learn(self, env, epochs, startstate):
        pass

    def register_stats(self, env):
        pass

    def get_action(self, state):
        pass


class PolicyStrategy(AgentStrategy):
    """Makes move according to TD(lambda) algorithm"""

    def __init__(self):
        AgentStrategy.__init__(self)
        self.qvalues, self.eligibility = dict(), dict()

    def register_stats(self, env):
        env.stats['rewards'] = Stats(PlotType.REGULAR)
        env.stats['loses'] = Stats(PlotType.REGULAR)
        env.stats['heatmap'] = MatrixStats(env.height, env.width, 0)

    def learn(self, env, epochs, startstate):
        if self.learnt:
            return
        for epoch in trange(Parameters.LEARN_EPOCHS):
            total_rewards = 0.0
            total_loses = 0.0
            Debug.print("QValues size: ", len(self.qvalues))
            for _ in range(Parameters.EPISODES_PER_EPOCH):
                # Initialize state and values for statistics.
                lenepisode = 0
                state = startstate.copy()
                action = pacman.Action.IDLE
                current_episode = [(state, action)]

                # Play and collect samples.
                while not env.is_terminating_state(state) and lenepisode <= Parameters.MAX_LEN_EPISODE:
                    # Make move and get reward.
                    nstate, reward = state.env.act(state, action)

                    # If it's last turn in episode, rewrite reward value to speed-up learning process.
                    if lenepisode == Parameters.MAX_LEN_EPISODE and not env.is_terminating_state(nstate):
                        # Gets ultimate reward based on number foods collected.
                        reward = -Parameters.INF / (5 - len(nstate.foods))

                    # Take eps-greedy action.
                    naction = self.q_eps_greedy_action(nstate, epoch)

                    # Compute delta based on Bellman equation.
                    delta = reward + Parameters.GAMMA * self.get_qvalue(nstate, naction) - self.get_qvalue(state, action)

                    # Increment eligibility of current state-action pair.
                    self.inc_eligibility(state, action)
                    player_pos = state.get_player_pos()
                    env.stats['heatmap'].add_in_point(player_pos.y, player_pos.x, 1)

                    # Update qvalues based on eligibility for all states in history.
                    for (state, action) in current_episode:
                        stateid = state.id()
                        addval = Parameters.ALPHA * delta * self.get_eligibility(stateid, action)
                        self.qvalues[(stateid, action)] += addval
                        self.eligibility[(stateid, action)] *= Parameters.GAMMA * Parameters.LAMBDA

                    # Update counters.
                    lenepisode += 1
                    total_rewards += reward

                    # Save state-action pair into replay history of current episode.
                    current_episode += [(nstate, naction)]

                    # Move on to next state.
                    state, action = nstate, naction
                total_loses += env.is_losing_state(state)
            # Update statistics.
            env.stats['rewards'].append(total_rewards / Parameters.EPISODES_PER_EPOCH)
            env.stats['loses'].append(total_loses / Parameters.EPISODES_PER_EPOCH)
            env.refresh_stats()
        self.learnt = True

    def q_eps_greedy_action(self, state, epoch):
        """ Eps-greedy move selection. """
        actions = state.env.get_player_actions(state)
        raction = actions[np.random.choice(max(1, len(actions)))]
        if np.random.random() > 1-(Parameters.EPS/math.sqrt(epoch+1)) or len(actions) <= 0:
            naction = raction
        else:
            naction, _ = self.q_greedy_action(state)
        return naction

    def q_greedy_action(self, state):
        """ Greedy move selection. """
        if state.env.is_terminating_state(state):
            return pacman.Action.IDLE, None
        actions = state.env.get_player_actions(state)
        naction, naction_value = None, -Parameters.INF
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
            self.eligibility[(sid, action)] = Parameters.DEFAULT_ELIGIBILITY
        return self.eligibility[(sid, action)]

    def inc_eligibility(self, state, action):
        sid = state.id()
        if (sid, action) not in self.eligibility:
            self.eligibility[(sid, action)] = Parameters.INC_ELIGIBILITY
        else:
            self.eligibility[(sid, action)] += Parameters.INC_ELIGIBILITY

    def get_qvalue(self, state, action):
        sid = state.id()
        if (sid, action) not in self.qvalues:
            self.qvalues[(sid, action)] = Parameters.DEFAULT_QVALUE
        return self.qvalues[(sid, action)]

    def get_action(self, state):
        action, _ = self.q_greedy_action(state)
        return action


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

    def init(self, env):
        self.policy = dict()

    def __get_new_policy__(self, env):
        return np.empty(env.get_states_dims(), dtype=pacman.Action)

    def compute_policy(self, env):
        newpolicy = self.__get_new_policy__(env)
        for state in env.get_all_states():
            assert (env.is_valid_state(state))
            bestaction = self.compute_action(state)
            newpolicy[state.id()] = bestaction
        return newpolicy

    def compute_action(self, state):
        env = state.env
        curpos = state.get_agent_pos_with_name(self.agent_name)
        if hasattr(self, 'target_agent_type'):
            targetspos = state.get_agents_pos_with_type(self.target_agent_type)
            if self.target_agent_type == pacman.AgentType.GHOST:
                targetspos.remove(curpos)
        else:
            targetspos = [self.target_pos]
        # Run BFS.
        distance = Utils.bfs(env, self.agent_type, curpos, targetspos)
        # Select best move.
        bestaction, bestpos = pacman.Action.IDLE, curpos
        assert(bestpos in distance)
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


class KeyboardStrategy(AgentStrategy):
    """Writes game description to console output and reads next move from standard input"""
    def __init__(self):
        AgentStrategy.__init__(self)

    def print_game(self, state):
        env = state.env
        Console.print(env.height, env.width)
        for i in range(env.height):
            Console.print(''.join(map(str, env.board[i])))
        Console.print(env.exit.y, env.exit.x)
        Console.print(len(state.agents))
        for agent in state.agents:
            Console.print(agent.pos.y, agent.pos.x)
        Console.print(len(state.foods))
        for food in state.foods:
            Console.print(food.y, food.x)

    def get_action(self, state):
        self.print_game(state)  # communicate with player before you ask for move.
        return Pacman.ALL_ACTIONS[int(input())]