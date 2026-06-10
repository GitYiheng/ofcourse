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
        gae_gamma = torch.as_tensor(self.args.gae_gamma, dtype=torch.float32)
        gae_lambda = torch.as_tensor(self.args.gae_lambda, dtype=torch.float32)
        n_step = len(rewards_th)
        advantages_th = [torch.zeros((), dtype=torch.float32) for step in range(n_step)]
        returns_th = [torch.zeros((), dtype=torch.float32) for step in range(n_step)]
        next_advantage_th = torch.zeros((), dtype=torch.float32)
        next_value_th = values_th[-1]
        mask_th = torch.zeros((), dtype=torch.float32)
        delta_th = torch.zeros((), dtype=torch.float32)
        for t in reversed(range(n_step)):
            mask_th = torch.ones((), dtype=torch.float32) - dones_th[t]
            delta_th = rewards_th[t] + gae_gamma * next_value_th * mask_th - values_th[t]
            next_advantage_th = delta_th + gae_gamma * gae_lambda * next_advantage_th * mask_th
            advantages_th[t] = next_advantage_th
            next_value_th = values_th[t]
            returns_th[t] = advantages_th[t] + values_th[t]
        return returns_th, advantages_th

    @staticmethod
    def format_progress_value(value):
        return f"{value:.4f}"

    @staticmethod
    def format_reward(value):
        return f"{float(value):.1f}"
