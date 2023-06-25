# OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

Code for paper "OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment" under review of NeurIPS 2023 Datasets and Benchmarks Track.

# Installation

This library requires Python >= 3.7.
[Miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements)/[Anaconda](https://docs.anaconda.com/anaconda/install/) is our recommended Python distribution.

Required libraries can be installed via pip:

```console
>>> cd ofcourse
>>> pip install -r requirements.txt
```

# Reproducing Results for Paper

`task 1: fulfillment of physical and virtual orders in one system`
```console
>>> sh ./run_exp/exp1/run_exp1_ppo.sh
>>> sh ./run_exp/exp1/run_exp1_happo.sh
>>> sh ./run_exp/exp1/run_exp1_ippo.sh
>>> sh ./run_exp/exp1/run_exp1_clo.sh
```

`task 2: cross-border order fulfillment`
```console
>>> sh ./run_exp/exp2/run_exp2_ppo.sh
>>> sh ./run_exp/exp2/run_exp2_happo.sh
>>> sh ./run_exp/exp2/run_exp2_ippo.sh
>>> sh ./run_exp/exp2/run_exp2_clo.sh
```

For these two tasks, the fulfillment agents are defined in `env/define_exp1_env.py` and `env/define_exp2_env.py`.

# Customized Usage

## Training

`main.py`
```python
from algo.runner import Runner                          # import runner
from algo.arguments import get_args                     # import argument parser
args = get_args()                                       # parse arguments
runner = Runner(args)                                   # create a runner instance with specified arguments
runner.run()                                            # start learning or evaluation
```

We can train `happo` on `exp1`:

`terminal`
```console
>>> python main.py --env=exp1 --algo=happo --mode=learn --log_dir=runs/exp1_happo --seed=10
```

You can monitor the training progress with [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html):

```console
>>> tensorboard --log_dir=runs
```

## Environment

Our order fulfillment environment is structured according to the format of [OpenAI Gym](https://github.com/openai/gym).
It is the standard API to communicate between reinforcement learning algorithms and environments.

`min_env_usage.py`
```python
from env.exp1_env import Exp1Env                       # import env
env = Exp1()                                           # create an env instance
obs = env.reset()                                      # start a new episode
num_steps = 10                                         # number of steps
for _t in range(num_steps):
    sampled_actions = env.action_space.sample()        # sample actions (not from algo)
    obs, rewards, dones, _ = env.step(sampled_actions) # interact with env
    if all(dones):
        obs = env.reset()                              # start a new episode when current one ends
```

### Action Space

#### Joint Action Space (MultiDiscrete)
```console
>>> env.action_space
[MultiDiscrete([1 3 2 1]), MultiDiscrete([1 2 2 1])]
```

#### Sampled Joint Action (MultiDiscrete)
```console
>>> env.action_space.sample()
[array([0, 2, 1, 0]), array([0, 0, 1, 0])]
```

#### Joint Action Space (Discrete)
```console
>>> env.action_space_discrete
[Discrete(6), Discrete(4)]
```

#### Sampled Joint Action (Discrete)
```console
>>> env.action_space_discrete.sample()
[4, 1]
```

#### Sampled Joint Action of Agent 0 (MultiDiscrete)
```console
>>> env.action_space[0].sample()
array([0, 1, 0, 0])
```

#### Sampled Joint Action of Agent 1 (MultiDiscrete)
```console
>>> env.action_space[1].sample()
array([0, 1, 0, 0])
```

#### Sampled Joint Action of Agent 0 (Discrete)
```console
>>> env.action_space_discrete[0].sample()
5
```

#### Sampled Joint Action of Agent 1 (Discrete)
```console
>>> env.action_space_discrete[1].sample()
1
```

#### Discrete-MultiDiscrete Joint Action Conversion
```console
>>> joint_action_discrete = 1
>>> env.joint_discrete_to_multi_discrete(joint_action_discrete)
[[0, 0, 0, 0], [0, 0, 1, 0]]
```

#### MultiDiscrete-Discrete Joint Action Conversion
```console
>>> joint_action_multi_discrete = [[0, 2, 1, 0], [0, 0, 0, 0]]
>>> env.joint_multi_discrete_to_discrete(joint_action_multi_discrete)
20
```

#### Discrete-MultiDiscrete Agent Action Conversion
```console
>>> agent_index = 0
>>> action_discrete = 1
>>> env.agent_discrete_to_multi_discrete(agent_index, action_discrete)
[0, 0, 1, 0]
```

#### MultiDiscrete-Discrete Agent Action Conversion
```console
>>> agent_index = 1
>>> action_multi_discrete = [0, 1, 1, 0]
>>> env.agent_multi_discrete_to_discrete(agent_index, action_multi_discrete)
3
```

### Observation Space

#### Observation Space Shape of Agent 0
```console
>>> env.observation_space[0].shape[0]
934
```

### Observation Space Shape of Agent 2
```console
>>> env.observation_space[1].shape[0]
932
```

# Order Source Management

The order source is a mechanism that takes in the simulation step as its input and generates a set of order instances as its output.
Currently, the orders are placed according to a repeating pattern.
Support for external order source management will be added soon.

