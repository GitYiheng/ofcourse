from argparse import Namespace

from env.exp1_env import Exp1Env
from env.exp2_env import Exp2Env


def make_args():
    return Namespace(undelivered_penalty=50.0, r_baseline=200.0)


def test_env_reset_and_step_follow_gymnasium_api():
    for env_cls in (Exp1Env, Exp2Env):
        env = env_cls(make_args())
        observations, info = env.reset(seed=123)

        assert isinstance(info, dict)
        assert len(observations) == env.n_agents

        next_observations, rewards, terminated, truncated, step_info = env.step(env.action_space.sample())

        assert isinstance(step_info, dict)
        assert len(next_observations) == env.n_agents
        assert len(rewards) == env.n_agents
        assert len(terminated) == env.n_agents
        assert len(truncated) == env.n_agents


def test_agent_action_conversion_round_trip():
    env = Exp1Env(make_args())

    for agent_i, action_space in enumerate(env.action_space_discrete):
        for discrete_action in range(action_space.n):
            multi_discrete_action = env.agent_discrete_to_multi_discrete(agent_i, discrete_action)
            assert env.agent_multi_discrete_to_discrete(agent_i, multi_discrete_action) == discrete_action


def test_joint_action_conversion_round_trip():
    env = Exp1Env(make_args())

    for discrete_action in range(env.joint_action_space_discrete.n):
        multi_discrete_action = env.joint_discrete_to_multi_discrete(discrete_action)
        assert env.joint_multi_discrete_to_discrete(multi_discrete_action) == discrete_action


def test_max_step_sets_truncated_not_terminated():
    env = Exp1Env(make_args())
    env.max_step = 1
    env.reset(seed=123)

    _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    assert terminated == [False] * env.n_agents
    assert truncated == [True] * env.n_agents
