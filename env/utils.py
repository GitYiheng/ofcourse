from enum import Enum
import argparse
import gym


class ServiceLevel(Enum):
    premium = 1
    standard = 2
    economy = 3


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for _a in agents_action_space: assert isinstance(_a, gym.spaces.space.Space)
        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        return [_agent_action_space.sample() for _agent_action_space in self._agents_action_space]


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for _a in agents_observation_space: assert isinstance(_a, gym.spaces.space.Space)
        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        return [_agent_observation_space.sample() for _agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        for _space, _ob in zip(self._agents_observation_space, obs):
            if not _space.contains(_ob):
                return False
        else:
            return True
