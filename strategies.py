import math
import numpy as np
import pacman
from tqdm import trange
from utils import Console, Debug, PlotType, MatrixStats, Stats, Utils

class AgentStrategy:
    """Base class for all Agent strategies"""

    def __init__(self):
        self.params = None
        self.learnt = False

    def learn(self, env):
        raise NotImplementedError

    def register_stats(self, env):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

class PolicyStrategy(AgentStrategy):
    """Makes move according to TD(lambda) algorithm"""

    def __init__(self, approximator):
        super().__init__()
        self.approximator = approximator

    def register_stats(self, env):
        """ Creates charts/stats that we want to show. """
        env.stats['rewards'] = Stats(PlotType.REGULAR)
        env.stats['loses'] = Stats(PlotType.REGULAR)
        env.stats['heatmap'] = MatrixStats(env.height, env.width, 0)

    # def learn(self, env):
    #     """ Main function. It learns a strategy by trial/error with TD(lambda) algorithm."""
    #     if self.learnt:
    #         return
    #     for epoch in trange(self.params.LEARN_EPOCHS):
    #         total_rewards = 0.0
    #         total_loses = 0.0
    #         Debug.print("QValues size: ", self.approximator.num_qvalues())
    #         for _ in range(self.params.EPISODES_PER_EPOCH):
    #             # Initialize state and values for statistics.
    #             lenepisode = 0
    #             state = env.start_state
    #             action = state.get_default_action()
    #             current_episode = [(state, action)]
    #
    #             # Play and collect samples.
    #             while not env.is_terminating_state(state) and lenepisode <= self.params.MAX_LEN_EPISODE:
    #                 # Make move and get reward.
    #                 nstate, reward = state.env.act(state, action)
    #
    #                 # If it's last turn in episode, rewrite reward value to speed-up learning process.
    #                 if lenepisode == self.params.MAX_LEN_EPISODE and not env.is_terminating_state(nstate):
    #                     # Gets ultimate reward based on number foods collected.
    #                     reward = -self.params.INF / (5 - len(nstate.foods))
    #
    #                 # Take eps-greedy action.
    #                 naction = self.q_eps_greedy_action(nstate, epoch)
    #
    #                 # Compute delta based on Bellman equation.
    #                 delta = reward + self.params.GAMMA * \
    #                                  self.approximator.qvalues[(nstate, naction)] - \
    #                                  self.approximator.qvalues[(state, action)]
    #
    #                 # Increment eligibility of current state-action pair.
    #                 self.approximator.eligibility[(state, action)] += self.params.INC_ELIGIBILITY
    #                 player_pos = state.get_player_pos()
    #                 env.stats['heatmap'].add_in_point(player_pos.y, player_pos.x, 1)
    #
    #                 # Update qvalues based on eligibility for all states in history.
    #                 for (state, action) in current_episode:
    #                     addval = self.params.ALPHA * delta * self.approximator.eligibility[(state, action)]
    #                     self.approximator.qvalues[(state, action)] += addval
    #                     self.approximator.eligibility[(state, action)] *= self.params.GAMMA * self.params.LAMBDA
    #
    #                 # Update counters.
    #                 lenepisode += 1
    #                 total_rewards += reward
    #
    #                 # Save state-action pair into replay history of current episode.
    #                 current_episode += [(nstate, naction)]
    #
    #                 # Move on to next state.
    #                 state, action = nstate, naction
    #             total_loses += env.is_losing_state(state)
    #         # Update statistics.
    #         env.stats['rewards'].append(total_rewards / self.params.EPISODES_PER_EPOCH)
    #         env.stats['loses'].append(total_loses / self.params.EPISODES_PER_EPOCH)
    #         env.refresh_stats()
    #     self.learnt = True

    def learn(self, env):
        """ Main function. It learns a strategy by trial/error with TD(lambda) algorithm."""
        if self.learnt:
            return
        for epoch in trange(self.params.LEARN_EPOCHS):
            total_rewards = 0.0
            total_loses = 0.0
            for _ in range(self.params.EPISODES_PER_EPOCH):
                # Initialize state and values for statistics.
                lenepisode = 0
                state = env.start_state
                action = state.get_default_action()
                current_episode = [(state, action)]

                # Play and collect samples.
                while not env.is_terminating_state(state) and lenepisode <= self.params.MAX_LEN_EPISODE:
                    # Make move and get reward.
                    nstate, reward = state.env.act(state, action)

                    # If it's last turn in episode, rewrite reward value to speed-up learning process.
                    if lenepisode == self.params.MAX_LEN_EPISODE and not env.is_terminating_state(nstate):
                        # Gets ultimate reward based on number foods collected.
                        reward = -self.params.INF / (5 - len(nstate.foods))

                    # Take eps-greedy action.
                    naction = self.q_eps_greedy_action(nstate, epoch)

                    # Compute delta based on Bellman equation.
                    delta = reward + self.params.GAMMA * \
                                     self.approximator.qvalues[(nstate, naction)] - \
                                     self.approximator.qvalues[(state, action)]

                    #delta *= self.params.ALPHA
                    #Debug.print(naction)
                    #Debug.print(reward, delta, self.approximator.qvalues.weights)
                    #delta *= self.approximator.extractor.features((nstate, naction))

                    player_pos = state.get_player_pos()
                    env.stats['heatmap'].add_in_point(player_pos.y, player_pos.x, 1)
                    #print(player_pos)

                    # Update qvalues approx. weights.
                    # self.approximator.qvalues.weights += delta

                    # Increment eligibility of current state-action pair.
                    self.approximator.eligibility[(state, action)] += self.params.INC_ELIGIBILITY
                    player_pos = state.get_player_pos()
                    env.stats['heatmap'].add_in_point(player_pos.y, player_pos.x, 1)

                    # Update qvalues based on eligibility for all states in history.
                    for (state, action) in current_episode:
                        addval = self.params.ALPHA * delta * self.approximator.eligibility[(state, action)]
                        self.approximator.qvalues[(state, action)] += addval
                        self.approximator.eligibility[(state, action)] *= self.params.GAMMA * self.params.LAMBDA

                    # Update counters.
                    lenepisode += 1
                    total_rewards += reward

                    # Save state-action pair into replay history of current episode.
                    current_episode += [(nstate, naction)]

                    # Move on to next state.
                    state, action = nstate, naction
                total_loses += env.is_losing_state(state, state.get_player_pos())
            # Update statistics.
            env.stats['rewards'].append(total_rewards / self.params.EPISODES_PER_EPOCH)
            env.stats['loses'].append(total_loses / self.params.EPISODES_PER_EPOCH)
            env.refresh_stats()
        self.learnt = True

    def q_eps_greedy_action(self, state, epoch):
        """ Eps-greedy move selection. """
        actions = state.env.get_player_actions(state)
        raction = actions[np.random.choice(max(1, len(actions)))]
        if np.random.random() > 1 - (self.params.EPS / math.sqrt(epoch + 1)) or len(actions) <= 0:
            naction = raction
        else:
            naction, _ = self.q_greedy_action(state)
        return naction

    def q_greedy_action(self, state):
        """ Greedy move selection. """
        if state.env.is_terminating_state(state):
            return state.get_default_action(), None

        actions = state.env.get_player_actions(state)
        naction, naction_value = None, -self.params.INF
        values_and_actions = []

        # Pick the best action.
        for action in actions:
            value = self.approximator.qvalues[(state, action)]
            values_and_actions.append((value, action))
            if naction is None or naction_value < value:
                naction = action
                naction_value = value
        assert (naction is not None)

        return naction, values_and_actions

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

    def learn(self, env):
        self.policy = dict()

    def register_stats(self, env):
        pass

    def compute_action(self, state):
        env = state.env
        agent = state.get_agent_with_name(self.agent_name)
        if hasattr(self, 'target_agent_type'):
            targetspos = state.get_agents_pos_with_type(self.target_agent_type)
            if self.target_agent_type == agent.type and agent.pos in targetspos:
                targetspos.remove(agent.pos)
        else:
            targetspos = [self.target_pos]
        # Run BFS.
        distance = Utils.bfs(env, self.agent_type, agent.pos, targetspos)
        # Select best move.
        bestaction, bestpos = state.get_default_action(), agent.pos
        assert (bestpos in distance)
        for npos, naction in env.get_nbs_positions(agent.pos, self.agent_type):
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
        actions = state.get_actions(self)
        chosen_action = int(input())
        return actions[chosen_action]