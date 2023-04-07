import random
import torch
import numpy as np
from algo.base_agents import BaseAgents
from algo.network import Critic, Actor


class IPPOAgents(BaseAgents):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.actors = [Actor(env.observation_space[agent_i].shape[0], env.action_space_discrete[agent_i].n) for agent_i
                       in range(env.n_agents)]
        self.critics = [Critic(env.observation_space[agent_i].shape[0]) for agent_i in range(env.n_agents)]
        self.optim_actors = [torch.optim.Adam(self.actors[agent_i].parameters(), lr=self.args.lr_actor) for agent_i in
                             range(env.n_agents)]
        self.optim_critics = [torch.optim.Adam(self.critics[agent_i].parameters(), lr=self.args.lr_critic) for agent_i
                              in range(env.n_agents)]
        self.num_episodes = 0

    def generate_action(self, observations, explore=True):
        dists = [self.actors[agent_i](observations[agent_i]) for agent_i in range(self.n_agents)]
        actions = [dists[agent_i].sample() if explore else torch.argmax(dists[agent_i].probs) for agent_i in
                   range(self.n_agents)]
        log_probs = [dists[agent_i].log_prob(actions[agent_i]) for agent_i in range(self.n_agents)]
        return actions, log_probs

    def generate_value(self, observations):
        values = [self.critics[agent_i](observations[agent_i]) for agent_i in range(self.n_agents)]
        return values

    def generate_episode(self, explore=True):
        observations_n_th = []
        actions_n_th = []
        rewards_n_th = []
        dones_n_th = []
        log_probs_n_th = []
        values_n_th = []

        observation_n = self.env.reset()
        step_episode = 0
        terminal = False

        while not terminal:
            # agents.generate_action & env.step
            observation_n_th = [torch.tensor(observation_n[agent_i]) for agent_i in range(self.n_agents)]
            action_n_th, log_prob_n_th = self.generate_action(observation_n_th, explore)
            action_n = [action_n_th[agent_i].detach().tolist() for agent_i in range(self.n_agents)]
            value_n_th = self.generate_value(observation_n_th)
            action_n_md = [self.env.agent_discrete_to_multi_discrete(agent_i, action_n[agent_i]) for agent_i in
                           range(self.n_agents)]
            next_observation_n, reward_n, done_n, _ = self.env.step(action_n_md)
            reward_n_th = [torch.tensor(reward_n[agent_i]) for agent_i in range(self.n_agents)]
            terminal = all((done_n, step_episode >= self.env.max_step))
            done_n = [terminal for agent_i in range(self.n_agents)]
            done_n_th = [torch.tensor(done_n[agent_i], dtype=torch.float32) for agent_i in range(self.n_agents)]

            # store transitions
            observations_n_th.append(observation_n_th)
            actions_n_th.append(action_n_th)
            rewards_n_th.append(reward_n_th)
            dones_n_th.append(done_n_th)
            log_probs_n_th.append(log_prob_n_th)
            values_n_th.append(value_n_th)

            # increment step_episode
            observation_n = next_observation_n
            step_episode += 1

        returns_n_th = [[] for _ in range(step_episode)]
        advantages_n_th = [[] for _ in range(step_episode)]
        for agent_i in range(self.n_agents):
            rewards_th = [rewards[agent_i] for rewards in rewards_n_th]
            values_th = [values[agent_i] for values in values_n_th]
            dones_th = [dones[agent_i] for dones in dones_n_th]
            returns_i_th, advantages_i_th = self.compute_return_and_advantage(rewards_th, values_th, dones_th)
            for idx in range(step_episode):
                returns_n_th[idx].append(returns_i_th[idx])
                advantages_n_th[idx].append(advantages_i_th[idx])

        return observations_n_th, actions_n_th, rewards_n_th, dones_n_th, log_probs_n_th, values_n_th, returns_n_th, advantages_n_th

    def generate_rollouts(self, explore=True, logger=None, summary=None):
        rollout_observations_n_th = []
        rollout_actions_n_th = []
        rollout_rewards_n_th = []  # [return] [advantage]
        rollout_dones_n_th = []  # [return] [advantage]
        rollout_values_n_th = []  # [return] [advantage] [critic loss]
        rollout_log_probs_n_th = []  # [actor loss]
        rollout_returns_n_th = []  # [critic loss]
        rollout_advantages_n_th = []  # [actor loss]

        for t_rollout in range(self.args.n_rollout):
            observations_n_th, actions_n_th, rewards_n_th, dones_n_th, log_probs_n_th, values_n_th, returns_n_th, advantages_n_th = self.generate_episode(
                explore)

            rollout_observations_n_th += observations_n_th
            rollout_actions_n_th += actions_n_th
            rollout_rewards_n_th += rewards_n_th
            rollout_dones_n_th += dones_n_th
            rollout_values_n_th += values_n_th
            rollout_log_probs_n_th += log_probs_n_th
            rollout_returns_n_th += returns_n_th
            rollout_advantages_n_th += advantages_n_th

            if logger and self.num_episodes % self.args.log_episode_freq == 0:
                cumulative_reward_n = np.sum(np.asarray(rewards_n_th) - self.args.r_baseline, axis=0)
                for agent_i in range(self.n_agents):
                    logger.info("Episode={}, Cumulative Reward[{}]={}".format(self.num_episodes, agent_i,
                                                                              cumulative_reward_n[agent_i]))

            self.num_episodes += 1

        logger.info("\n")
        batch_indices = random.sample([*range(len(rollout_observations_n_th))], self.args.batch_size)

        batch_observations_n_th = [rollout_observations_n_th[batch_index] for batch_index in batch_indices]
        batch_actions_n_th = [rollout_actions_n_th[batch_index] for batch_index in batch_indices]
        batch_rewards_n_th = [rollout_rewards_n_th[batch_index] for batch_index in batch_indices]
        batch_dones_n_th = [rollout_dones_n_th[batch_index] for batch_index in batch_indices]
        batch_values_n_th = [rollout_values_n_th[batch_index] for batch_index in batch_indices]
        batch_log_probs_n_th = [rollout_log_probs_n_th[batch_index] for batch_index in batch_indices]
        batch_returns_n_th = [rollout_returns_n_th[batch_index] for batch_index in batch_indices]
        batch_advantages_n_th = [rollout_advantages_n_th[batch_index] for batch_index in batch_indices]

        return batch_observations_n_th, batch_actions_n_th, batch_rewards_n_th, batch_dones_n_th, batch_values_n_th, batch_log_probs_n_th, batch_returns_n_th, batch_advantages_n_th

    def train(self, batch_log_probs_n_th, batch_advantages_n_th, batch_values_n_th, batch_returns_n_th):
        batch_actor_loss_n_th = [[torch.tensor(0.) for agent_i in range(self.n_agents)] for step in
                                 range(self.args.batch_size)]
        for agent_i in range(self.n_agents):
            for t in reversed(range(self.args.batch_size)):
                batch_actor_loss_n_th[t][agent_i] = -(
                            batch_log_probs_n_th[t][agent_i] * batch_advantages_n_th[t][agent_i])

        actor_loss_n_th = [torch.tensor(0.) for agent_i in range(self.n_agents)]
        for agent_i in range(self.n_agents):
            actor_loss_n_th[agent_i] = torch.stack(
                [batch_actor_loss_n_th[t][agent_i] for t in range(self.args.batch_size)]).mean()

        for agent_i in range(self.n_agents):
            self.optim_actors[agent_i].zero_grad()
            actor_loss_n_th[agent_i].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actors[agent_i].parameters(), max_norm=self.args.clip_grad_norm_actor)
            self.optim_actors[agent_i].step()

        batch_critic_loss_n_th = [[torch.tensor(0.) for agent_i in range(self.n_agents)] for step in
                                  range(self.args.batch_size)]
        for agent_i in range(self.n_agents):
            for t in range(self.args.batch_size):
                batch_critic_loss_n_th[t][agent_i] = torch.nn.functional.smooth_l1_loss(batch_values_n_th[t][agent_i],
                                                                                        batch_returns_n_th[t][agent_i])

        critic_loss_n_th = [torch.tensor(0.) for agent_i in range(self.n_agents)]
        for agent_i in range(self.n_agents):
            critic_loss_n_th[agent_i] = torch.stack(
                [batch_critic_loss_n_th[t][agent_i] for t in range(self.args.batch_size)]).mean()

        for agent_i in range(self.n_agents):
            self.optim_critics[agent_i].zero_grad()
            critic_loss_n_th[agent_i].backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_i].parameters(), max_norm=self.args.clip_grad_norm_critic)
            self.optim_critics[agent_i].step()

        return actor_loss_n_th, critic_loss_n_th

    def learn(self, logger=None, summary=None):
        for t_epoch in range(self.args.n_epoch):
            batch_observations_n_th, batch_actions_n_th, batch_rewards_n_th, batch_dones_n_th, batch_values_n_th, batch_log_probs_n_th, batch_returns_n_th, batch_advantages_n_th = self.generate_rollouts(
                explore=True, logger=logger, summary=summary)
            actor_loss_n_th, critic_loss_n_th = self.train(batch_log_probs_n_th, batch_advantages_n_th,
                                                           batch_values_n_th, batch_returns_n_th)

            if logger and summary:
                mean_batch_return_th = torch.mean(
                    torch.sum(torch.tensor(batch_returns_n_th, dtype=torch.float32), dim=1))
                summary.add_scalar(tag="Return", scalar_value=mean_batch_return_th.item(), global_step=t_epoch)
                for agent_i in range(self.n_agents):
                    summary.add_scalar(tag="Loss/actor_{}".format(agent_i),
                                       scalar_value=actor_loss_n_th[agent_i].item(), global_step=t_epoch)
                    summary.add_scalar(tag="Loss/critic_{}".format(agent_i),
                                       scalar_value=critic_loss_n_th[agent_i].item(), global_step=t_epoch)

                cols = 50
                ratio = t_epoch / self.args.n_epoch
                percent = int(ratio * 100)
                s1 = int(cols * ratio)
                s2 = cols - s1
                logger.info(
                    f"\r=== Training Progress [{'#' * s1}{' ' * s2}] epoch={t_epoch}/{self.args.n_epoch} [{percent}%] ===")
                logger.info("Return={}".format(mean_batch_return_th.item()))
                for agent_i in range(self.n_agents):
                    logger.info('Actor[{}] Loss={}'.format(agent_i, actor_loss_n_th[agent_i].item()))
                    logger.info("Critic[{}] Loss={}".format(agent_i, critic_loss_n_th[agent_i].item()))
                logger.info("\n")

    def evaluate(self, logger=None, summary=None):
        for t_epoch in range(self.args.n_epoch):
            batch_observations_n_th, batch_actions_n_th, batch_rewards_n_th, batch_dones_n_th, batch_values_n_th, batch_log_probs_n_th, batch_returns_n_th, batch_advantages_n_th = self.generate_rollouts(
                explore=False, logger=logger, summary=summary)

            if logger and summary:
                mean_batch_return_th = torch.mean(
                    torch.sum(torch.tensor(batch_returns_n_th, dtype=torch.float32), dim=1))
                summary.add_scalar(tag="Return", scalar_value=mean_batch_return_th.item(), global_step=t_epoch)

                cols = 50
                ratio = t_epoch / self.args.n_epoch
                percent = int(ratio * 100)
                s1 = int(cols * ratio)
                s2 = cols - s1
                logger.info(
                    f"\r=== Evaluate Progress [{'#' * s1}{' ' * s2}] epoch={t_epoch}/{self.args.n_epoch} [{percent}%] ===")
                logger.info("Return={}\n===\n".format(mean_batch_return_th.item()))
