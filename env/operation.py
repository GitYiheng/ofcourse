from collections import deque
from env.order_source import OrderSource


class Operation:
    def __init__(self, op_price, op_time):
        self.op_price = op_price  # one-off operation price
        self.op_time = op_time  # one-off operation time
        self._init_workstation()

    def _init_workstation(self):
        self.workstation = deque()  # operable orders

    def load(self, order=None):
        if order: self.workstation.append(order)

    def execute(self, step_count=None):
        raise NotImplementedError

    def get_obs(self):
        return [self.op_price, self.op_time]


class OpRoute(Operation):
    # origin: buffers_orig [list of buffers]
    # destination: buffer_dest [buffer]
    def __init__(self, buffers_orig, buffer_dest, op_price=0, op_time=1):
        Operation.__init__(self, op_price, op_time)
        self.buffers_orig = buffers_orig
        self.buffer_dest = buffer_dest

    def execute(self, step_count=None):
        for _buffer in self.buffers_orig:
            _buffer_chunk = _buffer.proceed()
            for _order in _buffer_chunk: self.load(_order)
        _empty_workstation = False if self.workstation else True
        if self.workstation: _avg_op_price = self.op_price / len(self.workstation)  # average operation price
        while self.workstation:
            _order = self.workstation.popleft()
            _order.update(specified_price=_avg_op_price, specified_time=1)
            self.buffer_dest.onboard(_order, self.op_time)
        return 0 if _empty_workstation else self.op_price


class OpConsoRoute(Operation):
    # origin: buffers_orig [list of buffers], inventory_orig [inventory]
    # destination: buffer_dest [buffer]
    def __init__(self, buffers_orig, inventory_orig, buffer_dest, op_price=0, op_time=1):
        Operation.__init__(self, op_price, op_time)
        self.buffers_orig = buffers_orig
        self.inventory_orig = inventory_orig
        self.buffer_dest = buffer_dest

    def execute(self, step_count=None):
        # from buffers_orig
        for _buffer in self.buffers_orig:
            _buffer_chunk = _buffer.proceed()
            for _order in _buffer_chunk: self.load(_order)
        # from inventory_src
        _inventory_chunk = self.inventory_orig.offboard()
        for _order in _inventory_chunk:
            _order.update(specified_price=0, specified_time=-1)  # time will be reupdated later
            self.load(_order)
        _empty_workstation = False if self.workstation else True
        # average operation price
        if self.workstation: avg_op_price = self.op_price / len(self.workstation)
        while self.workstation:
            _order = self.workstation.popleft()
            _order.update(specified_price=avg_op_price, specified_time=1)
            self.buffer_dest.onboard(_order, self.op_time)
        return 0 if _empty_workstation else self.op_price


class OpDispatch(Operation):
    # origin: order_src [order_source]
    # destination: buffer_dest [buffer]
    def __init__(self, order_src, buffer_dest, op_price=0, op_time=1):
        Operation.__init__(self, op_price, op_time)
        self.order_src = order_src  # orders with same origin-destination
        self.buffer_dest = buffer_dest

    def execute(self, step_count=None):
        _order_chunk = self.order_src.order_at_step_count(step_count) if step_count else None
        # for _order in _order_chunk: self.load(_order)
        if _order_chunk:
            for _order in _order_chunk: self.load(_order)
        _empty_workstation = False if self.workstation else True
        while self.workstation:
            _order = self.workstation.popleft()
            _order.update(specified_price=self.op_price, specified_time=1)
            self.buffer_dest.onboard(_order, self.op_time)
        return 0 if _empty_workstation else self.op_price


class OpStore(Operation):
    # origin: buffer_orig [list of buffers]
    # destination: inventory_dest [inventory]
    def __init__(self, buffers_orig, inventory_dest, op_price=0, op_time=0):
        Operation.__init__(self, op_price, op_time)
        self.buffers_orig = buffers_orig
        self.inventory_dest = inventory_dest

    def execute(self, step_count=None):
        for _buffer_orig in self.buffers_orig:
            _buffer_chunk = _buffer_orig.proceed()
            for _order in _buffer_chunk: self.load(_order)
        _empty_workstation = False if self.workstation else True
        while self.workstation:
            _order = self.workstation.popleft()
            _order.update(specified_price=self.op_price, specified_time=self.op_time)
            self.inventory_dest.onboard(_order)
        return 0 if _empty_workstation else self.op_price
