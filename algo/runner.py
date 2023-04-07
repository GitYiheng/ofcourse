# env
from env.exp1_env import Exp1Env
from env.exp2_env import Exp2Env
# algo
from algo.ippo_agents import IPPOAgents
from algo.ppo_agents import PPOAgents
from algo.happo_agents import HAPPOAgents
from algo.clo_agents import CLOAgents

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger
from utils.plot import plot


class Runner:
    def __init__(self, args):
        self.args = args
        self.summary = None
        self.logger = Logger(args)
        self._print_arg()
        self._init_env()
        self._set_seed()
        self._init_agents()

    def _init_env(self):
        if self.args.env == 'exp1':
            self.env = Exp1Env(self.args)
        elif self.args.env == 'exp2':
            self.env = Exp2Env(self.args)
        else:
            raise ValueError("env_name must be one of: ''exp1', 'exp2'!")

    def _init_agents(self):
        if self.args.algo == 'ippo':
            self.agents = IPPOAgents(self.env, self.args)
        elif self.args.algo == 'ppo':
            self.agents = PPOAgents(self.env, self.args)
        elif self.args.algo == 'happo':
            self.agents = HAPPOAgents(self.env, self.args)
        elif self.args.algo == 'clo':
            self.agents = CLOAgents(self.env, self.args)
        else:
            raise ValueError("algo_name must be one of: 'ippo', 'ppo', 'happo', or 'clo'!")

    def _set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def _print_arg(self):
        logger_args = self.logger.get_logger(format='')
        self.logger.print_logo()
        logger_args.info("Hyperparameters:")
        for k in self.args.__dict__.keys():
            logger_args.info("    {} = {}".format(k, self.args.__dict__[k]))
        logger_args.info("\n")

    def learn(self):
        self.summary = SummaryWriter(log_dir=self.args.log_dir)
        self.agents.learn(logger=self.logger.get_logger(), summary=self.summary)
        self.summary.close()

    def evaluate(self):
        self.summary = SummaryWriter(log_dir=self.args.log_dir)
        self.agents.evaluate(logger=self.logger.get_logger(), summary=self.summary)
        self.summary.close()

    def run(self):
        if self.args.mode == 'learn':
            self.learn()
        elif self.args.mode == 'evaluate':
            self.evaluate()
        else:
            raise ValueError("mode must be 'learn' or 'evaluate'!")

        if self.summary and self.args.plot:
            plot(self.args.log_dir, tag="Return", smooth=3)
