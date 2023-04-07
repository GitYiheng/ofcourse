import argparse


def get_args():
    parser = argparse.ArgumentParser("Order Fulfillment by Multi-Agent Reinforcement Learning")

    parser.add_argument('--seed', type=int, default=348407, help='seed for random number generator')

    parser.add_argument('--log_dir', type=str, default='runs/ofcourse', help='directory for logging')
    parser.add_argument('--log_episode_freq', type=int, default=1, help='frequency for logging episode')

    parser.add_argument('--env', type=str, default='exp1', help='environment: exp1, exp2')

    parser.add_argument('--algo', type=str, default='happo', help='algorithm: ppo, ippo, happo, or clo')
    parser.add_argument('--mode', type=str, default='learn', help='mode: learn or evaluate')

    parser.add_argument('--n_epoch', type=int, default=128, help='number of epochs')
    parser.add_argument('--n_rollout', type=int, default=16, help='number of rollouts')
    parser.add_argument('--batch_size', type=int, default=1024, help='number of rollouts')

    parser.add_argument('--undelivered_penalty', type=float, default=50., help='penalty per undelivered order')
    parser.add_argument('--r_baseline', type=float, default=200., help='reward baseline')

    parser.add_argument('--lr_actor', type=float, default=3e-3, help='learning rate of actor network')
    parser.add_argument('--lr_critic', type=float, default=3e-3, help='learning rate of critic network')

    # gradient clipping
    # https://arxiv.org/abs/1707.06347
    parser.add_argument('--clip_grad_norm_actor', type=float, default=1.0, help='clips gradient norm of actor')
    parser.add_argument('--clip_grad_norm_critic', type=float, default=1.0, help='clips gradient norm of critic')

    # generalized advantage estimation
    # https://arxiv.org/abs/1506.02438
    parser.add_argument('--gae_gamma', type=float, default=0.99, help='gamma of generalized advantage estimation')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='lambda of generalized advantage estimation')

    parser.add_argument('--plot', type=bool, default=False, help='plot epoch-return')

    args = parser.parse_args()

    return args
