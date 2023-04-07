import random
import torch
import numpy as np
from algo.base_agents import BaseAgents
from algo.network import Critic, Actor


class PPOAgents(BaseAgents):
    def __init__(self, env, args):
        super().__init__(env, args)
        observation_size = sum([env.observation_space[agent_i].shape[0] for agent_i in range(env.n_agents)])
        action_size = env.joint_action_space_discrete.n
        self.actor = Actor(observation_size, action_size)
        self.critic = Critic(observation_size)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        self.num_episodes = 0

    def generate_action(self, observation, explore=True):
        dist = self.actor(observation)
        action = dist.sample() if explore else torch.argmax(dist.probs)
        log_prob = dist.log_prob(action)
        return action, log_prob

    def generate_value(self, observation):
        value = self.critic(observation)
        return value

    def generate_episode(self, explore=True):
        observations_th = []
        actions_th = []
        rewards_th = []
        dones_th = []
        log_probs_th = []
        values_th = []

        observation_n = self.env.reset()
        step_episode = 0
        terminal = False

        while not terminal:
            # agents.generate_action & env.step
            observation_th = torch.cat([torch.tensor(observation_n[agent_i]) for agent_i in range(self.n_agents)],
                                       dim=0)
            action_th, log_prob_th = self.generate_action(observation_th, explore)
            action_d = action_th.item()
            action_n_md = self.env.joint_discrete_to_multi_discrete(action_d)
            value_th = self.generate_value(observation_th)
            next_observation_n, reward_n, done_n, _ = self.env.step(action_n_md)
            reward = sum(reward_n)
            reward_th = torch.tensor(reward)
            terminal = all((done_n, step_episode >= self.env.max_step))
            done_th = torch.tensor(terminal, dtype=torch.float32)

            # store transitions
            observations_th.append(observation_th)
            actions_th.append(action_th)
            rewards_th.append(reward_th)
            dones_th.append(done_th)
            log_probs_th.append(log_prob_th)
            values_th.append(value_th)

            # increment step_episode
            observation_n = next_observation_n
            step_episode += 1

        returns_th, advantages_th = self.compute_return_and_advantage(rewards_th, values_th, dones_th)

        return observations_th, actions_th, rewards_th, dones_th, log_probs_th, values_th, returns_th, advantages_th

    def generate_rollouts(self, explore=True, logger=None, summary=None):
        rollout_observations_th = []
        rollout_actions_th = []
        rollout_rewards_th = []  # [return] [advantage]
        rollout_dones_th = []  # [return] [advantage]
        rollout_values_th = []  # [return] [advantage] [critic loss]
        rollout_log_probs_th = []  # [actor loss]
        rollout_returns_th = []  # [critic loss]
        rollout_advantages_th = []  # [actor loss]

        for t_rollout in range(self.args.n_rollout):
            observations_th, actions_th, rewards_th, dones_th, log_probs_th, values_th, returns_th, advantages_th = self.generate_episode(
                explore)

            rollout_observations_th += observations_th
            rollout_actions_th += actions_th
            rollout_rewards_th += rewards_th
            rollout_dones_th += dones_th
            rollout_values_th += values_th
            rollout_log_probs_th += log_probs_th
            rollout_returns_th += returns_th
            rollout_advantages_th += advantages_th

            if logger and self.num_episodes % self.args.log_episode_freq == 0:
                cumulative_reward = sum(np.asarray(rewards_th) - self.n_agents * self.args.r_baseline)
                logger.info("Episode={}, Cumulative Reward={}".format(self.num_episodes, cumulative_reward))

            self.num_episodes += 1

        logger.info("\n")
        batch_indices = random.sample([*range(len(rollout_observations_th))], self.args.batch_size)

        batch_observations_th = [rollout_observations_th[batch_index] for batch_index in batch_indices]
        batch_actions_th = [rollout_actions_th[batch_index] for batch_index in batch_indices]
        batch_rewards_th = [rollout_rewards_th[batch_index] for batch_index in batch_indices]
        batch_dones_th = [rollout_dones_th[batch_index] for batch_index in batch_indices]
        batch_values_th = [rollout_values_th[batch_index] for batch_index in batch_indices]
        batch_log_probs_th = [rollout_log_probs_th[batch_index] for batch_index in batch_indices]
        batch_returns_th = [rollout_returns_th[batch_index] for batch_index in batch_indices]
        batch_advantages_th = [rollout_advantages_th[batch_index] for batch_index in batch_indices]

        return batch_observations_th, batch_actions_th, batch_rewards_th, batch_dones_th, batch_values_th, batch_log_probs_th, batch_returns_th, batch_advantages_th

    def train(self, batch_log_probs_th, batch_advantages_th, batch_values_th, batch_returns_th):
        batch_actor_loss_th = [torch.tensor(0.) for step in range(self.args.batch_size)]

        for t in reversed(range(self.args.batch_size)):
            batch_actor_loss_th[t] = -(batch_log_probs_th[t] * batch_advantages_th[t])

        actor_loss_th = torch.stack([batch_actor_loss_th[t] for t in range(self.args.batch_size)]).mean()

        self.optim_actor.zero_grad()
        actor_loss_th.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.args.clip_grad_norm_actor)
        self.optim_actor.step()

        batch_critic_loss_th = [torch.tensor(0.) for step in range(self.args.batch_size)]

        for t in range(self.args.batch_size):
            batch_critic_loss_th[t] = torch.nn.functional.smooth_l1_loss(batch_values_th[t], batch_returns_th[t])

        critic_loss_th = torch.stack([batch_critic_loss_th[t] for t in range(self.args.batch_size)]).mean()

        self.optim_critic.zero_grad()
        critic_loss_th.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.args.clip_grad_norm_critic)
        self.optim_critic.step()

        return actor_loss_th, critic_loss_th

    def learn(self, logger=None, summary=None):
        actor_loss_th, critic_loss_th = None, None
        for t_epoch in range(self.args.n_epoch):
            batch_observations_th, batch_actions_th, batch_rewards_th, batch_dones_th, batch_values_th, batch_log_probs_th, batch_returns_th, batch_advantages_th = self.generate_rollouts(
                explore=True, logger=logger, summary=summary)
            actor_loss_th, critic_loss_th = self.train(batch_log_probs_th, batch_advantages_th, batch_values_th,
                                                       batch_returns_th)

            if logger and summary:
                mean_batch_return_th = torch.mean(torch.tensor(batch_returns_th, dtype=torch.float32))
                summary.add_scalar(tag="Return", scalar_value=mean_batch_return_th.item(), global_step=t_epoch)
                summary.add_scalar(tag="Loss/actor", scalar_value=actor_loss_th.item(), global_step=t_epoch)
                summary.add_scalar(tag="Loss/critic", scalar_value=critic_loss_th.item(), global_step=t_epoch)

                cols = 50
                ratio = t_epoch / self.args.n_epoch
                percent = int(ratio * 100)
                s1 = int(cols * ratio)
                s2 = cols - s1
                logger.info(
                    f"\r=== Training Progress [{'#' * s1}{' ' * s2}] epoch={t_epoch}/{self.args.n_epoch} [{percent}%] ===")
                logger.info("Return={}".format(mean_batch_return_th.item()))
                logger.info('Actor Loss={}'.format(actor_loss_th.item()))
                logger.info("Critic Loss={}\n===\n".format(critic_loss_th.item()))

    def evaluate(self, logger=None, summary=None):
        for t_epoch in range(self.args.n_epoch):
            batch_observations_th, batch_actions_th, batch_rewards_th, batch_dones_th, batch_values_th, batch_log_probs_th, batch_returns_th, batch_advantages_th = self.generate_rollouts(
                explore=False, logger=logger, summary=summary)

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
