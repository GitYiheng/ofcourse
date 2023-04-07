import numpy as np
import random
import torch


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
        observation_n = self.env.reset()
        step_episode = 0
        terminal = False
        while not terminal:
            # agents.generate_action & env.step
            observation_th = torch.cat([torch.tensor(observation_n[agent_i]) for agent_i in range(self.n_agents)],
                                       dim=0)
            action_n_md = self.generate_action(observation_th)
            next_observation_n, reward_n, done_n, _ = self.env.step(action_n_md)
            reward = sum(reward_n)
            reward_th = torch.tensor(reward)
            terminal = all((done_n, step_episode >= self.env.max_step))
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
                logger.info("Episode={}, Cumulative Reward={}".format(self.num_episodes, cumulative_reward))

            self.num_episodes += 1

        logger.info("\n")
        batch_indices = random.sample([*range(len(rollout_returns_th))], self.args.batch_size)
        batch_returns_th = [rollout_returns_th[batch_index] for batch_index in batch_indices]
        return rollout_returns_th

    def train(self, batch_log_probs_n_th, batch_advantages_n_th, batch_values_n_th, batch_returns_n_th):
        raise NotImplementedError

    def evaluate(self, logger=None, summary=None):
        for t_epoch in range(self.args.n_epoch):
            batch_returns_th = self.generate_rollouts(logger=logger, summary=summary)

            if logger and summary:
                mean_batch_return_th = torch.mean(torch.tensor(batch_returns_th, dtype=torch.float32))
                summary.add_scalar(tag="Return", scalar_value=mean_batch_return_th.item(), global_step=t_epoch)

                cols = 50
                ratio = t_epoch / self.args.n_epoch
                percent = int(ratio * 100)
                s1 = int(cols * ratio)
                s2 = cols - s1
                logger.info(
                    f"\r=== Evaluate Progress [{'#' * s1}{' ' * s2}] epoch={t_epoch}/{self.args.n_epoch} [{percent}%] ===")
                logger.info("Return={}\n===\n".format(mean_batch_return_th.item()))

    def compute_return(self, rewards_th):
        n_step = len(rewards_th)
        gamma_th = torch.tensor(self.args.gae_gamma)
        returns_th = [torch.tensor(0.) for step in range(n_step)]
        returns_th[-1] = rewards_th[-1]
        for t in reversed(range(len(rewards_th) - 1)):
            returns_th[t] = rewards_th[t] + gamma_th * returns_th[t + 1]
        return returns_th
