import copy
import logging

import numpy as np
import itertools
from ast import literal_eval

import gym
from gym import spaces
from gym.spaces import Discrete, MultiDiscrete
from gym.utils import seeding

from env.utils import MultiAgentActionSpace, MultiAgentObservationSpace
from env.define_exp2_env import define_exp2_env


class Exp2Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self._get_agents = define_exp2_env
        self.agents = self._get_agents()

        self.n_agents = len(self.agents)
        self.max_step = 200
        self.step_count = None
        self.undelivered_penalty = self.args.undelivered_penalty

        self.observation_space = MultiAgentObservationSpace([spaces.Box(np.zeros(len(agent_i.get_obs())),
                                                                        np.full(len(agent_i.get_obs()),
                                                                                fill_value=np.finfo(np.float32).max))
                                                             for agent_i in self.agents])
        self.action_space = MultiAgentActionSpace(
            [spaces.MultiDiscrete([len(fulfillment_unit.operations) for fulfillment_unit in agent_i.fulfillment_units])
             for agent_i in self.agents])
        self._init_action_space()
        self._init_joint_action_space()

    def reset(self):
        self.agents = self._get_agents()

        self.step_count = 0
        self.agent_dones = [False for _ in range(self.n_agents)]
        _obs = []
        for _agent_i in range(self.n_agents):
            _obs.append(self.get_agent_obs(_agent_i))
        return _obs

    def step(self, agents_action):
        self.step_count += 1
        _costs = [0 for agent_i in range(self.n_agents)]
        for agent_i, action in enumerate(agents_action):
            if not (self.agent_dones[agent_i]):
                _costs[agent_i] = self.agents[agent_i].step(action, self.step_count)
        if self.step_count >= self.max_step:
            for agent_i in range(self.n_agents):
                # punish undelivered orders in containers when episode ends
                _costs[agent_i] += self.agents[agent_i].clearance(self.undelivered_penalty)
                self.agent_dones[agent_i] = True
        _obs = []
        for agent_i in range(self.n_agents):
            _obs.append(self.get_agent_obs(agent_i))
        _rewards = [-_cost + self.args.r_baseline for _cost in _costs]
        return _obs, _rewards, self.agent_dones, {}

    def get_agent_obs(self, agent_i):
        _obs = []
        _obs += self.agents[agent_i].get_obs()
        return _obs

    def get_global_obs(self):
        _obs = []
        for _agent_i in range(self.n_agents):
            _obs.append(self.agents[_agent_i].get_obs())
        return _obs

    def get_agent_reward(self, agent_i):
        return self.agents[agent_i].get_reward()

    def get_global_reward(self):
        return [self.agents[_agent_i].get_reward() for _agent_i in range(self.n_agents)]

    def _init_action_space(self):
        self.action_space_discrete = MultiAgentActionSpace(
            [spaces.Discrete(np.prod(multi_discrete_action.nvec)) for multi_discrete_action in self.action_space])
        self.md_to_d = []
        self.d_to_md = []
        for _agent_i, multi_discrete_action in enumerate(self.action_space):
            possible_action_each_dim = [list(range(_dim)) for _dim in multi_discrete_action.nvec]
            possible_action_tuple = list(itertools.product(*possible_action_each_dim))
            possible_action_multi_discrete = [list(_tuple) for _tuple in possible_action_tuple]
            possible_action_discrete = list(range(self.action_space_discrete[_agent_i].n))
            _md_to_d = dict()
            _d_to_md = dict()
            for md, d in zip(possible_action_multi_discrete, possible_action_discrete):
                _md_to_d[str(md)] = str(d)
                _d_to_md[str(d)] = str(md)
            self.md_to_d.append(_md_to_d)
            self.d_to_md.append(_d_to_md)

    def agent_discrete_to_multi_discrete(self, agent_i, discrete):
        _discrete = str(discrete)
        _multi_discrete = self.d_to_md[agent_i][_discrete]
        return literal_eval(_multi_discrete)

    def agent_multi_discrete_to_discrete(self, agent_i, multi_discrete):
        _multi_discrete = str(multi_discrete)
        _discrete = self.md_to_d[agent_i][_multi_discrete]
        return int(_discrete)

    def _init_joint_action_space(self):
        # possible joint action (Discrete)
        _joint_action_space_dim = []
        for _agent_i, multi_discrete_action in enumerate(self.action_space):
            _joint_action_space_dim += [_dim for _dim in multi_discrete_action.nvec]
        _joint_action_space_multi_discrete = spaces.MultiDiscrete(_joint_action_space_dim)
        self.joint_action_space_discrete = spaces.Discrete(np.prod(_joint_action_space_multi_discrete.nvec))
        possible_joint_action_discrete = list(range(self.joint_action_space_discrete.n))
        # possible joint action (MultiDiscrete)
        _possible_joint_action_multi_discrete_sep = []
        for _agent_i, _multi_discrete_action in enumerate(self.action_space):
            _possible_action_each_dim = [list(range(_dim)) for _dim in _multi_discrete_action.nvec]
            _possible_action_tuple = list(itertools.product(*_possible_action_each_dim))
            _possible_action_multi_discrete = [list(_tuple) for _tuple in _possible_action_tuple]
            _possible_joint_action_multi_discrete_sep.append(_possible_action_multi_discrete)
        _possible_joint_action_multi_discrete_tuple = list(
            itertools.product(*_possible_joint_action_multi_discrete_sep))
        possible_joint_action_multi_discrete = [list(_tuple) for _tuple in _possible_joint_action_multi_discrete_tuple]
        # dict for Discrete-MultiDiscrete and MultiDiscrete-Discrete
        self.joint_md_to_d = dict()
        self.joint_d_to_md = dict()
        for md, d in zip(possible_joint_action_multi_discrete, possible_joint_action_discrete):
            self.joint_md_to_d[str(md)] = str(d)
            self.joint_d_to_md[str(d)] = str(md)

    def joint_discrete_to_multi_discrete(self, discrete):
        _discrete = str(discrete)
        _multi_discrete = self.joint_d_to_md[_discrete]
        return literal_eval(_multi_discrete)

    def joint_multi_discrete_to_discrete(self, multi_discrete):
        _multi_discrete = str(multi_discrete)
        _discrete = self.joint_md_to_d[_multi_discrete]
        return int(_discrete)

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
