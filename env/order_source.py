import numpy as np
from collections import deque, Counter
from env.order import Order


class OrderSource:
    def __init__(self, order_queue=None, order_queue_len=150):
        self.order_queue_len = order_queue_len
        self._init_order_queue(order_queue=order_queue)

    def _init_order_queue(self, order_queue=None, order_ratio=0.1):
        if order_queue.any():
            _order_queue_bool = order_queue
        else:
            _prob = np.random.random_sample(self.order_queue_len)
            _order_queue_bool = (_prob < order_ratio).astype(bool)
        self.order_queue = deque()
        for _create_order in _order_queue_bool:
            _order_queue_chunk = deque()
            _temp_order = Order() if _create_order else None
            _order_queue_chunk.append(_temp_order)
            self.order_queue.append(_order_queue_chunk)
        # number of non-empty chunk
        non_empty_bool = Counter([order_chunk[0] != None for order_chunk in self.order_queue])
        self.num_order = non_empty_bool[True]

    def order_at_step_count(self, step_count=None):
        # example usage: env.agents[0].fulfillment_units[-1].operations[0].sod_src.order_at_step_count(step_count=5)
        return self.order_queue[step_count] if step_count < self.order_queue_len else False

    def onboard(self):
        raise NotImplementedError

    def offboard(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError
