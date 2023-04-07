from gym import Env, spaces
from gym.envs.registration import register

import numpy as np
from collections import deque

from env.resource import Resource
from env.container import Buffer, Inventory
from env.operation import OpDispatch, OpStore, OpRoute, OpConsoRoute
from env.fulfillment_unit import FulfillmentUnit


class Agent:
    def __init__(self):
        self.fulfillment_units = deque()  # seller to customer rightward

    def add_fulfillment_unit(self, fulfillment_unit=None):
        if fulfillment_unit: self.fulfillment_units.append(fulfillment_unit)

    def step(self, action, step_count=None):
        _reward_update = self.update()
        _reward_execute = self.execute(action, step_count)
        return _reward_update + _reward_execute

    def update(self):
        _step_price = 0
        for _fulfillment_unit in self.fulfillment_units:
            _step_price += _fulfillment_unit.update()
        return _step_price

    def execute(self, action, step_count=None):
        _step_price = 0
        for _agent_i, _fulfillment_unit in enumerate(self.fulfillment_units):
            _step_price += _fulfillment_unit.execute(action[_agent_i], step_count)
        return _step_price

    def get_obs(self):
        _obs = []
        for _fulfillment_unit in self.fulfillment_units:
            _obs += _fulfillment_unit.get_obs()
        return _obs

    def get_reward(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def clearance(self, undelivered_penalty=50.):
        _num_failed_order = self.fulfillment_units[-1].operations[0].order_src.num_order - len(
            self.fulfillment_units[0].containers[0].inventory)
        _total_penalty = _num_failed_order * undelivered_penalty
        return _total_penalty
