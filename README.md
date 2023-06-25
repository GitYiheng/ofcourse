# OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

Code repository for paper "OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment" under review of NeurIPS 2023 Datasets and Benchmarks Track.

![OFCOURSE Teaser](./figs/teaser.png)

# Installation

This library requires Python >= 3.7.
[Miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements)/[Anaconda](https://docs.anaconda.com/anaconda/install/) is our recommended Python distribution.

Required libraries can be installed via pip:

```console
>>> git clone https://github.com/GitYiheng/ofcourse.git
>>> cd ofcourse
>>> pip install -r requirements.txt
```

# Reproducing Results for Paper

## Task 1 &mdash; Fulfillment of Physical and Virtual Orders in One System

```console
>>> sh ./run_exp/exp1/run_exp1_ppo.sh
>>> sh ./run_exp/exp1/run_exp1_happo.sh
>>> sh ./run_exp/exp1/run_exp1_ippo.sh
>>> sh ./run_exp/exp1/run_exp1_clo.sh
```

## Task 2 &mdash; Cross-Border Order Fulfillment

```console
>>> sh ./run_exp/exp2/run_exp2_ppo.sh
>>> sh ./run_exp/exp2/run_exp2_happo.sh
>>> sh ./run_exp/exp2/run_exp2_ippo.sh
>>> sh ./run_exp/exp2/run_exp2_clo.sh
```

For these two tasks, the fulfillment agents are defined in [env/define_exp1_env.py](env/define_exp1_env.py) and [env/define_exp2_env.py](env/define_exp2_env.py).

# Training

```python
from algo.runner import Runner                          # import runner
from algo.arguments import get_args                     # import argument parser
args = get_args()                                       # parse arguments
runner = Runner(args)                                   # create a runner instance with specified arguments
runner.run()                                            # start learning or evaluation
```

We can train `happo` on `exp1`:

```console
>>> python main.py --env=exp1 --algo=happo --mode=learn --log_dir=runs/exp1_happo --seed=10
```

We can monitor the training progress with [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html):

```console
>>> tensorboard --log_dir=runs
```

# Environment

Our order fulfillment environment is structured according to the format of [OpenAI Gym](https://github.com/openai/gym).
It is the standard API to communicate between reinforcement learning algorithms and environments.

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

# Action Space and Observation Space

See [docs/act_obs.md](docs/act_obs.md).

# Data Collection and Organization

See [docs/data.md](docs/data.md).

# Order Source Management

The order source is a mechanism that takes in the simulation step as its input and generates a set of order instances as its output.
Currently, orders are placed according to a prescribed repeating pattern.
External order source management will be added soon.

