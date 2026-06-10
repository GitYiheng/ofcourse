import numpy as np
import random
import torch
from tqdm import tqdm


def format_progress_value(value):
    return f"{value:.4f}"


def format_reward(value):
    return f"{float(value):.1f}"


class CLOAgents:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.n_agents = env.n_agents
        self.num_episodes = 0

    def generate_action(self, observation_th):
        action_n_md = [[np.argmin([op.op_price for op in fulfillment_unit.operations]) for fulfillment_unit in
                        agent.fulfillment_units] for agent in self.env.agents]
        return action_n_md

    def generate_episode(self):
        rewards_th = []
        observation_n, _ = self.env.reset()
        step_episode = 0
        terminal = False
        while not terminal:
            # agents.generate_action & env.step
            observation_th = torch.cat([
                torch.as_tensor(observation_n[agent_i], dtype=torch.float32)
                for agent_i in range(self.n_agents)
            ], dim=0)
            action_n_md = self.generate_action(observation_th)
            next_observation_n, reward_n, terminated_n, truncated_n, _ = self.env.step(action_n_md)
            reward = sum(reward_n)
            reward_th = torch.as_tensor(reward, dtype=torch.float32)
            done_n = [terminated or truncated for terminated, truncated in zip(terminated_n, truncated_n)]
            terminal = all(done_n) or step_episode >= self.env.max_step
            done_th = torch.tensor(terminal, dtype=torch.float32)
            # store transitions
            rewards_th.append(reward_th)
            # increment step_episode
            observation_n = next_observation_n
            step_episode += 1
        returns_th = self.compute_return(rewards_th)
        return returns_th, rewards_th

    def generate_rollouts(self, logger=None, summary=None):
        rollout_returns_th = []
        for t_rollout in range(self.args.n_rollout):
            returns_th, rewards_th = self.generate_episode()
            rollout_returns_th += returns_th

            if logger and self.num_episodes % self.args.log_episode_freq == 0:
                cumulative_reward = sum(np.asarray(rewards_th) - self.n_agents * self.args.r_baseline)
                logger.info(
                    "Episode={}, Cumulative Reward={}".format(
                        self.num_episodes,
                        format_reward(cumulative_reward),
                    )
                )

            self.num_episodes += 1

        if logger:
            logger.info("\n")
        batch_size = min(self.args.batch_size, len(rollout_returns_th))
        batch_indices = random.sample([*range(len(rollout_returns_th))], batch_size)
        return [rollout_returns_th[batch_index] for batch_index in batch_indices]

    def train(self, batch_log_probs_n_th, batch_advantages_n_th, batch_values_n_th, batch_returns_n_th):
        raise NotImplementedError

    def evaluate(self, logger=None, summary=None):
        progress = tqdm(range(self.args.n_epoch), desc="Evaluating CLO", unit="epoch")
        for t_epoch in progress:
            batch_returns_th = self.generate_rollouts(logger=logger, summary=summary)

            if logger and summary:
                mean_batch_return_th = torch.mean(torch.tensor(batch_returns_th, dtype=torch.float32))
                summary.add_scalar(tag="Return", scalar_value=mean_batch_return_th.item(), global_step=t_epoch)
                progress.set_postfix(return_=format_progress_value(mean_batch_return_th.item()))
                logger.info("Return={}".format(mean_batch_return_th.item()))

    def compute_return(self, rewards_th):
        n_step = len(rewards_th)
        gamma_th = torch.as_tensor(self.args.gae_gamma, dtype=torch.float32)
        returns_th = [torch.zeros((), dtype=torch.float32) for step in range(n_step)]
        returns_th[-1] = rewards_th[-1]
        for t in reversed(range(len(rewards_th) - 1)):
            returns_th[t] = rewards_th[t] + gamma_th * returns_th[t + 1]
        return returns_th
