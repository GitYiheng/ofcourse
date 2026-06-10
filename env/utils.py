from enum import Enum
from gymnasium import spaces


class ServiceLevel(Enum):
    premium = 1
    standard = 2
    economy = 3


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for action_space in agents_action_space:
            if not isinstance(action_space, spaces.Space):
                raise TypeError(f"Expected gymnasium Space, got {type(action_space)!r}")
        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        return [_agent_action_space.sample() for _agent_action_space in self._agents_action_space]


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for observation_space in agents_observation_space:
            if not isinstance(observation_space, spaces.Space):
                raise TypeError(f"Expected gymnasium Space, got {type(observation_space)!r}")
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
