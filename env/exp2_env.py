from env.base_env import BaseFulfillmentEnv
from env.define_exp2_env import define_exp2_env


class Exp2Env(BaseFulfillmentEnv):
    def __init__(self, args):
        super().__init__(args, define_exp2_env)
