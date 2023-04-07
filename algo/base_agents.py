import torch


class BaseAgents:
    def __init__(self, env, args):
        self.env = env
        self.n_agents = env.n_agents
        self.args = args

    def generate_action(self):
        raise NotImplementedError

    def generate_value(self):
        raise NotImplementedError

    def generate_episode(self):
        raise NotImplementedError

    def generate_rollouts(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def compute_return_and_advantage(self, rewards_th, values_th, dones_th):
        # generalized advantage estimation
        # https://arxiv.org/abs/1506.02438
        gae_gamma = torch.tensor(self.args.gae_gamma)
        gae_lambda = torch.tensor(self.args.gae_lambda)
        n_step = len(rewards_th)
        advantages_th = [torch.tensor(0.) for step in range(n_step)]
        returns_th = [torch.tensor(0.) for step in range(n_step)]
        next_advantage_th = torch.tensor(0.)
        next_value_th = values_th[-1]
        mask_th = torch.tensor(0.)
        delta_th = torch.tensor(0.)
        for t in reversed(range(n_step)):
            mask_th = torch.tensor(1.) - dones_th[t]
            delta_th = rewards_th[t] + gae_gamma * next_value_th * mask_th - values_th[t]
            next_advantage_th = delta_th + gae_gamma * gae_lambda * next_advantage_th * mask_th
            advantages_th[t] = next_advantage_th
            next_value_th = values_th[t]
            returns_th[t] = advantages_th[t] + values_th[t]
        return returns_th, advantages_th
