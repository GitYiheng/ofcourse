from collections import deque


class Container:
    def __init__(self, resource):
        self.resource = resource

    def _init_container(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def onboard(self):
        raise NotImplementedError

    def offboard(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError


class Buffer(Container):
    def __init__(self, resource, buffer_len=337, buffer_chunk_limit=6): # 24*7*2+1=337
        Container.__init__(self, resource)
        self.buffer_len = buffer_len
        self.buffer_chunk_limit = buffer_chunk_limit
        self._init_container()

    def _init_container(self):
        self.buffer = deque()
        for _buffer_chunk_index in range(self.buffer_len):
            _buffer_chunk = deque()
            self.buffer.append(_buffer_chunk)

    def update(self):
        _step_price = 0
        for _buffer_chunk_index in range(1, self.buffer_len):
            for _order in self.buffer[_buffer_chunk_index]:
                _step_price += _order.update()
        return _step_price

    def onboard(self, order, op_time=1):
        _recurring_price = self.resource.get_recurring_price() # get recurring price by resource status
        order.set_recurring_price(_recurring_price) # set recurring price for the onboarding order
        self.resource.occupy(1) # occupy(size), taking up unit resource
        self.buffer[op_time].append(order) # put order on target buffer chunk

    def offboard(self, op_time):
        _offboard_size = len(self.buffer[op_time])
        self.resource.release(_offboard_size)
        return deque([self.buffer[op_time].popleft() for _order_index in range(_offboard_size)])

    def proceed(self):
        _in_buffer_chunk = deque()
        self.buffer.append(_in_buffer_chunk)
        _out_buffer_chunk = self.buffer.popleft()
        _released_size = len(_out_buffer_chunk)
        self.resource.release(_released_size)
        return _out_buffer_chunk

    def get_obs(self):
        _obs = []
        for _buffer_chunk in self.buffer:
            _num_order = len(_buffer_chunk)
            for _order_index in range(self.buffer_chunk_limit):
                _order_obs = _buffer_chunk[_order_index].get_obs() if _order_index < _num_order else [0]*6
                _obs += _order_obs
        return _obs + self.resource.get_obs()


class Inventory(Container):
    def __init__(self, resource, recurring_time=1, inventory_limit=30):
        Container.__init__(self, resource)
        self.recurring_time = recurring_time # 0 represents sign for receipt; 1 represent a single time step
        self.inventory_limit = inventory_limit
        self._init_container()

    def _init_container(self):
        self.inventory = deque()

    def update(self):
        _step_price = 0
        for _order in self.inventory:
            _step_price += _order.update()
        return _step_price

    def onboard(self, order):
        _recurring_price = self.resource.get_recurring_price()
        order.set_recurring_price(_recurring_price)
        order.recurring_time = self.recurring_time
        self.resource.occupy(1) # occupy(size), taking up unit resource
        self.inventory.append(order)

    def offboard(self):
        _offboard_size = len(self.inventory)
        self.resource.release(_offboard_size)
        return deque([self.inventory.popleft() for _order_index in range(_offboard_size)])

    def get_obs(self):
        _obs = []
        _num_order = len(self.inventory)
        for _order_index in range(self.inventory_limit):
            _order_obs = self.inventory[_order_index].get_obs() if _order_index < _num_order else [0]*6
            _obs += _order_obs
        return _obs + self.resource.get_obs()