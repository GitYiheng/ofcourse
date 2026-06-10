from env.base_env import BaseFulfillmentEnv
from env.define_exp1_env import define_exp1_env


class Exp1Env(BaseFulfillmentEnv):
    def __init__(self, args):
        super().__init__(args, define_exp1_env)
